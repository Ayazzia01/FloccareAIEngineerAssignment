#!/usr/bin/env python3
"""
Place a band-aid on a wound using a segmentation mask from
https://github.com/Nico-Curti/Deepskin.

Workflow:
  1) Run the external Deepskin repo to generate a wound mask.
  2) Find wound contour/center/orientation from that mask.
  3) Overlay a rotated/scaled band-aid with OpenCV alpha blending.
"""

from __future__ import annotations

import argparse
import math
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class WoundDetection:
    contour: np.ndarray
    center: Tuple[int, int]
    angle: float
    bbox: Tuple[int, int, int, int]


def _binary_from_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def _infer_semantic_masks(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infer wound + skin + background masks from a 3-channel semantic mask.

    Expected channel order: [background, skin, wound]. Falls back to area-based
    inference if wound/skin channels are empty.
    """
    background = _binary_from_mask(mask[:, :, 0])
    skin = _binary_from_mask(mask[:, :, 1])
    wound = _binary_from_mask(mask[:, :, 2])

    if np.count_nonzero(wound) > 0 and np.count_nonzero(skin) > 0:
        return wound, skin, background

    binaries = [_binary_from_mask(mask[:, :, idx]) for idx in range(mask.shape[2])]
    areas = [int(np.count_nonzero(b)) for b in binaries]
    nonzero = [i for i, a in enumerate(areas) if a > 0]
    if len(nonzero) < 2:
        return wound, skin, background
    order = sorted(nonzero, key=lambda i: areas[i])
    wound_idx = order[0]
    background_idx = order[-1]
    skin_candidates = [i for i in nonzero if i not in (wound_idx, background_idx)]
    skin_idx = skin_candidates[0] if skin_candidates else (order[1] if len(order) > 1 else wound_idx)
    return binaries[wound_idx], binaries[skin_idx], binaries[background_idx]


def _detections_from_mask(mask: np.ndarray) -> List[WoundDetection]:
    binary = _binary_from_mask(mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    h, w = binary.shape[:2]
    min_area = (h * w) * 0.0001
    valid = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    if not valid:
        return []

    detections: List[WoundDetection] = []
    for contour in sorted(valid, key=cv2.contourArea, reverse=True):
        moments = cv2.moments(contour)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            (cx, cy), _, _ = cv2.minAreaRect(contour)
        angle = _contour_angle(contour)
        x, y, bw, bh = cv2.boundingRect(contour)
        detections.append(
            WoundDetection(contour=contour, center=(int(cx), int(cy)), angle=float(angle), bbox=(x, y, bw, bh))
        )
    return detections


def _largest_contour_from_mask(mask: np.ndarray) -> Optional[WoundDetection]:
    detections = _detections_from_mask(mask)
    return detections[0] if detections else None


def _default_deepskin_mask_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_deepskin_mask.png")


def _split_command(cmd: str) -> list[str]:
    parts = shlex.split(cmd, posix=False)
    cleaned: list[str] = []
    for part in parts:
        if len(part) >= 2 and part[0] == part[-1] and part[0] in ("\"", "'"):
            part = part[1:-1]
        cleaned.append(part)
    return cleaned


def detect_wounds_with_deepskin(
    image_path: Path,
    seg_repo: Path,
    seg_command: str,
    keep_mask: Optional[Path] = None,
) -> Tuple[List[WoundDetection], Optional[np.ndarray], Optional[np.ndarray]]:
    """Run a Deepskin command to produce a mask, then estimate wound geometry.

    seg_command must include placeholder:
      - {input}: absolute input image path
    If {mask} is omitted, the function looks for the default Deepskin CLI output:
      - <input>_deepskin_mask.png next to the input file
    """
    seg_repo = seg_repo.resolve()
    image_path = image_path.resolve()
    if not seg_repo.exists():
        raise FileNotFoundError(f"Segmentation repo path not found: {seg_repo}")

    if "{input}" not in seg_command:
        raise ValueError("--seg-command must include the {input} placeholder")

    with tempfile.TemporaryDirectory(prefix="wound_mask_") as tmpdir:
        tmp_mask = Path(tmpdir) / "wound_mask.png"
        cmd = seg_command.format(input=str(image_path), mask=str(tmp_mask), python=sys.executable)

        proc = subprocess.run(
            _split_command(cmd),
            cwd=str(seg_repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Deepskin command failed.\n"
                f"Command: {cmd}\n"
                f"Exit code: {proc.returncode}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        if "{mask}" in seg_command:
            produced_mask = tmp_mask
        else:
            produced_mask = _default_deepskin_mask_path(image_path)

        if not produced_mask.exists():
            if "{mask}" in seg_command:
                raise RuntimeError("Segmentation mask was not produced; verify --seg-command output path handling.")
            raise RuntimeError(
                "Segmentation mask was not produced; expected Deepskin output at "
                f"{produced_mask}. Consider providing --seg-command with {{mask}}."
            )

        mask = cv2.imread(str(produced_mask), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read generated mask: {produced_mask}")

        skin_mask = None
        if mask.ndim == 3 and mask.shape[2] >= 3:
            wound_mask, skin_mask, background_mask = _infer_semantic_masks(mask)
            skin_mask = cv2.bitwise_not(background_mask)
            detections = _detections_from_mask(wound_mask)
        else:
            detections = _detections_from_mask(mask)

        if keep_mask is not None:
            keep_mask.parent.mkdir(parents=True, exist_ok=True)
            if produced_mask.resolve() != keep_mask.resolve():
                try:
                    shutil.move(str(produced_mask), str(keep_mask))
                except Exception:
                    cv2.imwrite(str(keep_mask), mask)
                    if produced_mask.exists():
                        try:
                            produced_mask.unlink()
                        except OSError:
                            pass
            else:
                if not keep_mask.exists():
                    cv2.imwrite(str(keep_mask), mask)

        return detections, skin_mask, mask


def detect_wound_heuristic(image_bgr: np.ndarray) -> Optional[WoundDetection]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    mask_hsv = cv2.inRange(hsv, np.array([0, 35, 35]), np.array([12, 255, 255])) | cv2.inRange(
        hsv, np.array([160, 35, 35]), np.array([179, 255, 255])
    )
    mask_lab = cv2.threshold(lab[:, :, 1], 145, 255, cv2.THRESH_BINARY)[1]
    return _largest_contour_from_mask(cv2.bitwise_and(mask_hsv, mask_lab))


def create_default_bandaid(width: int = 320, height: int = 120) -> np.ndarray:
    img = np.zeros((height, width, 4), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (196, 190, 173, 255), -1)
    pad_w, pad_h = int(width * 0.34), int(height * 0.62)
    px0, py0 = (width - pad_w) // 2, (height - pad_h) // 2
    cv2.rectangle(img, (px0, py0), (px0 + pad_w, py0 + pad_h), (220, 215, 200, 255), -1)
    for x in (int(width * 0.12), int(width * 0.88)):
        for y in range(int(height * 0.22), int(height * 0.82), 10):
            cv2.circle(img, (x, y), 2, (160, 155, 140, 180), -1)
    img[:, :, 3] = cv2.GaussianBlur(img[:, :, 3], (7, 7), 0)
    return img


def load_bandaid_image(path: Optional[Path]) -> np.ndarray:
    if path is None:
        return create_default_bandaid()
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read band-aid image: {path}")
    if img.ndim != 3:
        raise ValueError("Band-aid image must be RGB/RGBA")
    if img.shape[2] == 3:
        img = np.dstack([img, np.full(img.shape[:2], 255, dtype=np.uint8)])
    return img


def _ensure_odd_ksize(value: int) -> int:
    if value <= 0:
        return 0
    return value if value % 2 == 1 else value + 1


def _normalize_angle(angle: float) -> float:
    while angle < -90:
        angle += 180
    while angle >= 90:
        angle -= 180
    return angle


def _pca_axis_info(
    contour: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]]:
    pts = contour.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] < 2:
        return None
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)
    order = np.argsort(eigenvalues.ravel())[::-1]
    v_major = eigenvectors[int(order[0])]
    if eigenvectors.shape[0] > 1:
        v_minor = eigenvectors[int(order[1])]
    else:
        v_minor = np.array([-v_major[1], v_major[0]], dtype=np.float32)
    centered = pts - mean
    proj_major = centered @ v_major
    proj_minor = centered @ v_minor
    length_major = float(np.max(proj_major) - np.min(proj_major))
    length_minor = float(np.max(proj_minor) - np.min(proj_minor))
    angle = math.degrees(math.atan2(float(v_major[1]), float(v_major[0])))
    return mean[0], v_major, v_minor, length_major, length_minor, angle


def _contour_angle(contour: np.ndarray) -> float:
    info = _pca_axis_info(contour)
    if info is not None:
        angle = info[5]
        return _normalize_angle(angle)
    (_, _), _, angle = cv2.minAreaRect(contour)
    if angle < -45:
        angle += 90
    return _normalize_angle(angle)


def _wrap_weight(xn: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        weight = 1.0 - (xn + 1.0) * 0.5
    elif side == "right":
        weight = (xn + 1.0) * 0.5
    else:
        weight = np.ones_like(xn)
    return np.clip(weight, 0.0, 1.0) ** 1.5


def _warp_wrap(overlay_rgba: np.ndarray, wrap_strength: float, side: str = "both") -> np.ndarray:
    if wrap_strength <= 0 or side == "none":
        return overlay_rgba
    h, w = overlay_rgba.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    xn = (x - cx) / (w * 0.5)
    yn = (y - cy) / (h * 0.5)

    side_weight = _wrap_weight(xn, side)
    dx = wrap_strength * (1.0 - yn ** 2) * np.sin(math.pi * xn) * (w * 0.5) * side_weight
    dy = wrap_strength * (1.0 - xn ** 2) * np.sin(math.pi * yn) * (h * 0.5) * side_weight

    map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)

    return cv2.remap(
        overlay_rgba,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _soften_edges(overlay_rgba: np.ndarray, edge_blur: int) -> np.ndarray:
    k = _ensure_odd_ksize(edge_blur)
    if k == 0:
        return overlay_rgba
    softened = overlay_rgba.copy()
    softened[:, :, :3] = cv2.GaussianBlur(softened[:, :, :3], (k, k), 0)
    softened[:, :, 3] = cv2.GaussianBlur(softened[:, :, 3], (k, k), 0)
    return softened


def _apply_shadow(
    base_bgr: np.ndarray, mask: np.ndarray, roi: Tuple[int, int, int, int], strength: float, blur: int
) -> None:
    if strength <= 0:
        return
    k = _ensure_odd_ksize(blur)
    shadow = mask.astype(np.float32) / 255.0
    if k > 0:
        shadow = cv2.GaussianBlur(shadow, (k, k), 0)
    shadow = np.clip(shadow * strength, 0.0, 1.0)
    bx0, by0, bx1, by1 = roi
    base_crop = base_bgr[by0:by1, bx0:bx1].astype(np.float32)
    base_bgr[by0:by1, bx0:bx1] = np.clip(base_crop * (1.0 - shadow[..., None]), 0, 255).astype(np.uint8)


def _place_overlay(
    overlay_rgba: np.ndarray, center: Tuple[int, int], base_shape: Tuple[int, int, int]
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    oh, ow = overlay_rgba.shape[:2]
    x0, y0 = center[0] - ow // 2, center[1] - oh // 2
    x1, y1 = x0 + ow, y0 + oh
    bx0, by0 = max(0, x0), max(0, y0)
    bx1, by1 = min(base_shape[1], x1), min(base_shape[0], y1)
    if bx0 >= bx1 or by0 >= by1:
        return None, None

    ox0, oy0 = bx0 - x0, by0 - y0
    overlay_crop = overlay_rgba[oy0 : oy0 + (by1 - by0), ox0 : ox0 + (bx1 - bx0)]
    return overlay_crop, (bx0, by0, bx1, by1)


def _alpha_blend(base_bgr: np.ndarray, overlay_crop: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    bx0, by0, bx1, by1 = roi
    out = base_bgr.copy()
    alpha = overlay_crop[:, :, 3:4].astype(np.float32) / 255.0
    base_crop = out[by0:by1, bx0:bx1].astype(np.float32)
    out[by0:by1, bx0:bx1] = (alpha * overlay_crop[:, :, :3] + (1.0 - alpha) * base_crop).astype(np.uint8)
    return out


def _seamless_blend(base_bgr: np.ndarray, overlay_crop: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    bx0, by0, bx1, by1 = roi
    mask = overlay_crop[:, :, 3]
    if np.max(mask) == 0:
        return base_bgr
    mask_bin = np.where(mask > 0, 255, 0).astype(np.uint8)
    center = (bx0 + (bx1 - bx0) // 2, by0 + (by1 - by0) // 2)
    return cv2.seamlessClone(overlay_crop[:, :, :3], base_bgr, mask_bin, center, cv2.NORMAL_CLONE)


def _mask_has_value(mask: np.ndarray, point: Tuple[float, float], radius: int = 3) -> bool:
    h, w = mask.shape[:2]
    x = int(round(point[0]))
    y = int(round(point[1]))
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(w, x + radius + 1)
    y1 = min(h, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return False
    return np.any(mask[y0:y1, x0:x1] > 0)


def _wrap_side_for_mask(
    allowed_mask: np.ndarray, center: Tuple[int, int], angle: float, length: float
) -> str:
    theta = math.radians(angle)
    dx = (length * 0.5) * math.cos(theta)
    dy = (length * 0.5) * math.sin(theta)
    end_left = (center[0] - dx, center[1] - dy)
    end_right = (center[0] + dx, center[1] + dy)
    left_on = _mask_has_value(allowed_mask, end_left)
    right_on = _mask_has_value(allowed_mask, end_right)
    if left_on and not right_on:
        return "right"
    if right_on and not left_on:
        return "left"
    if not left_on and not right_on:
        return "both"
    return "none"


def overlay_rgba(
    base_bgr: np.ndarray,
    overlay_rgba: np.ndarray,
    center: Tuple[int, int],
    angle: float,
    blend: str,
    edge_blur: int,
    shadow_strength: float,
    shadow_blur: int,
    wrap_strength: float,
    wrap_side: str,
    overlay_canvas: Optional[np.ndarray],
    allowed_mask: Optional[np.ndarray],
) -> np.ndarray:
    out = base_bgr.copy()
    oh, ow = overlay_rgba.shape[:2]
    if wrap_strength > 0 and wrap_side != "none":
        overlay_rgba = _warp_wrap(overlay_rgba, wrap_strength, side=wrap_side)
    rotated = cv2.warpAffine(
        overlay_rgba,
        cv2.getRotationMatrix2D((ow // 2, oh // 2), angle, 1.0),
        (ow, oh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    softened = _soften_edges(rotated, edge_blur)
    overlay_crop, roi = _place_overlay(softened, center, out.shape)
    if overlay_crop is None or roi is None:
        return out

    if allowed_mask is not None:
        bx0, by0, bx1, by1 = roi
        allowed_crop = allowed_mask[by0:by1, bx0:bx1]
        inside = allowed_crop > 0
        if not np.any(inside):
            return out
        overlay_crop = overlay_crop.copy()
        inside_f = inside.astype(np.float32)
        overlay_crop[:, :, :3] = (overlay_crop[:, :, :3].astype(np.float32) * inside_f[..., None]).astype(np.uint8)
        overlay_crop[:, :, 3] = (overlay_crop[:, :, 3].astype(np.float32) * inside_f).astype(np.uint8)

    if overlay_canvas is not None:
        bx0, by0, bx1, by1 = roi
        existing = overlay_canvas[by0:by1, bx0:bx1]
        overlay_canvas[by0:by1, bx0:bx1] = np.maximum(existing, overlay_crop)

    _apply_shadow(out, overlay_crop[:, :, 3], roi, shadow_strength, shadow_blur)

    if blend == "seamless":
        return _seamless_blend(out, overlay_crop, roi)
    return _alpha_blend(out, overlay_crop, roi)


def apply_bandaid(
    image_bgr: np.ndarray,
    bandaid_rgba: np.ndarray,
    detection: WoundDetection,
    size_scale: float,
    margin_scale: float,
    blend: str,
    edge_blur: int,
    shadow_strength: float,
    shadow_blur: int,
    wrap_strength: float,
    overlay_canvas: Optional[np.ndarray],
    skin_mask: Optional[np.ndarray],
) -> np.ndarray:
    angle = detection.angle
    info = _pca_axis_info(detection.contour)
    if info is not None:
        _, _, _, length_major, length_minor, angle = info
    else:
        _, _, bw, bh = detection.bbox
        length_major, length_minor = float(bw), float(bh)

    length_major = max(length_major, 1.0)
    length_minor = max(length_minor, 1.0)
    scale = max(0.0, size_scale) * max(0.0, margin_scale)
    target_major = int(max(40, length_major * scale))
    target_minor = int(max(40, length_minor * scale))

    wound_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(wound_mask, [detection.contour], -1, 255, thickness=-1)
    if skin_mask is not None:
        allowed_mask = cv2.bitwise_or(skin_mask, wound_mask)
    else:
        allowed_mask = wound_mask

    def apply_one(base_img: np.ndarray, target_w: int, target_h: int, band_angle: float) -> np.ndarray:
        resized = cv2.resize(bandaid_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if wrap_strength > 0 and allowed_mask is not None:
            wrap_side = _wrap_side_for_mask(allowed_mask, detection.center, band_angle, target_w)
        else:
            wrap_side = "none"
        return overlay_rgba(
            base_img,
            resized,
            detection.center,
            band_angle,
            blend=blend,
            edge_blur=edge_blur,
            shadow_strength=shadow_strength,
            shadow_blur=shadow_blur,
            wrap_strength=wrap_strength,
            wrap_side=wrap_side,
            overlay_canvas=overlay_canvas,
            allowed_mask=allowed_mask,
        )

    out = apply_one(image_bgr, target_major, target_minor, angle)
    return apply_one(out, target_minor, target_major, _normalize_angle(angle + 90))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect wound and place a band-aid.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bandaid", type=Path, default=None)
    parser.add_argument("--seg-repo", type=Path, default=None, help="Path to cloned Deepskin repo")
    parser.add_argument(
        "--seg-command",
        type=str,
        default="\"{python}\" -m deepskin --input \"{input}\" --mask",
        help=(
            "Deepskin command template with {input} placeholder. "
            "Use {python} to run with the current interpreter. "
            "If {mask} is included, it should point to the output mask path."
        ),
    )
    parser.add_argument("--mask-out", type=Path, default=None)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--overlay-out", type=Path, default=None)
    parser.add_argument("--overlay-dir", type=Path, default=None)
    parser.add_argument("--blend", choices=("seamless", "alpha"), default="seamless")
    parser.add_argument("--size-scale", type=float, default=1.5)
    parser.add_argument("--margin-scale", type=float, default=1.15)
    parser.add_argument("--edge-blur", type=int, default=5)
    parser.add_argument("--shadow-strength", type=float, default=0.18)
    parser.add_argument("--shadow-blur", type=int, default=21)
    parser.add_argument("--wrap-strength", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers for batch mode.")
    parser.add_argument("--fallback-heuristic", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input is None and args.input_dir is None:
        raise ValueError("Provide --input or --input-dir.")
    if args.input is not None and args.input_dir is not None:
        raise ValueError("Provide only one of --input or --input-dir.")

    def iter_inputs() -> list[Path]:
        if args.input_dir is None:
            return [args.input] if args.input is not None else []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return sorted([p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

    inputs = iter_inputs()
    if not inputs:
        raise FileNotFoundError("No input images found.")

    if args.input_dir is not None:
        base_dir = args.input_dir
        output_dir = args.output_dir or (base_dir / "output")
        mask_dir = args.mask_dir or (base_dir / "masks")
        overlay_dir = args.overlay_dir or (base_dir / "overlays")
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
        mask_dir = None
        overlay_dir = None

    def process_one(input_path: Path) -> Optional[Path]:
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {input_path}")
            return None

        if args.input_dir is not None:
            output_path = output_dir / f"{input_path.stem}_with_bandaid.jpg"
            compare_path = output_dir / f"{input_path.stem}_with_bandaid_compare.jpg"
            mask_path = mask_dir / f"{input_path.stem}_deepskin_mask.png"
            overlay_path = overlay_dir / f"{input_path.stem}_overlay.png"
        else:
            output_path = args.output or input_path.with_name(f"{input_path.stem}_with_bandaid.jpg")
            compare_path = output_path.with_name(f"{output_path.stem}_compare{output_path.suffix}")
            mask_path = args.mask_out
            overlay_path = args.overlay_out

        detections: List[WoundDetection] = []
        skin_mask: Optional[np.ndarray] = None
        raw_mask: Optional[np.ndarray] = None
        repo_error: Optional[Exception] = None
        if args.seg_repo is not None:
            try:
                detections, skin_mask, raw_mask = detect_wounds_with_deepskin(
                    input_path, args.seg_repo, args.seg_command, mask_path
                )
            except Exception as exc:
                repo_error = exc

        if not detections and args.fallback_heuristic:
            fallback = detect_wound_heuristic(image)
            if fallback is not None:
                detections = [fallback]

        if not detections:
            msg = "Wound detection failed."
            if args.seg_repo is None:
                msg += " Provide --seg-repo for Deepskin segmentation."
            if repo_error is not None:
                msg += f" Repo error: {repo_error}"
            if not args.fallback_heuristic:
                msg += " You can also enable --fallback-heuristic."
            print(f"{input_path}: {msg}")
            return None

        output = image.copy()
        overlay_canvas = np.zeros((output.shape[0], output.shape[1], 4), dtype=np.uint8)

        for detection in detections:
            output = apply_bandaid(
                output,
                load_bandaid_image(args.bandaid),
                detection,
                size_scale=args.size_scale,
                margin_scale=args.margin_scale,
                blend=args.blend,
                edge_blur=args.edge_blur,
                shadow_strength=args.shadow_strength,
                shadow_blur=args.shadow_blur,
                wrap_strength=args.wrap_strength,
                overlay_canvas=overlay_canvas,
                skin_mask=skin_mask,
            )

        if overlay_path is not None:
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlay_path), overlay_canvas)
        mask_img = raw_mask

        def to_bgr(img: Optional[np.ndarray], target_shape: Tuple[int, int, int]) -> np.ndarray:
            if img is None:
                return np.zeros(target_shape, dtype=np.uint8)
            if img.ndim == 2:
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                alpha = img[:, :, 3:4].astype(np.float32) / 255.0
                bgr = img[:, :, :3].astype(np.float32)
                bgr = (bgr * alpha).astype(np.uint8)
            else:
                bgr = img[:, :, :3]
            if bgr.shape[:2] != target_shape[:2]:
                bgr = cv2.resize(bgr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
            return bgr

        mask_vis = to_bgr(mask_img, image.shape)
        overlay_vis = to_bgr(overlay_canvas, image.shape)
        top_row = np.concatenate([image, mask_vis], axis=1)
        bottom_row = np.concatenate([overlay_vis, output], axis=1)
        grid = np.concatenate([top_row, bottom_row], axis=0)
        cv2.imwrite(str(compare_path), grid)

        if args.input_dir is None and not args.no_show:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            axes[1].set_title("With Band-Aid")
            axes[1].axis("off")
            plt.tight_layout()
            plt.show()

        print(f"Saved comparison to: {compare_path}")
        return output_path

    if args.input_dir is None or args.workers <= 1:
        for input_path in inputs:
            process_one(input_path)
        return

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_path = {executor.submit(process_one, p): p for p in inputs}
        for future in as_completed(future_to_path):
            _ = future.result()


if __name__ == "__main__":
    main()
