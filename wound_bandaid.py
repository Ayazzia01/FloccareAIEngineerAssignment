#!/usr/bin/env python3
"""
Place a band-aid on a wound using a segmentation mask from
https://github.com/uwm-bigdata/wound-segmentation.

Primary workflow:
  1) Run the external wound-segmentation repo to generate a mask.
  2) Find wound contour/center/orientation from that mask.
  3) Overlay a rotated/scaled band-aid.

Example:
  python wound_bandaid.py \
    --input arm.jpg \
    --seg-repo /path/to/wound-segmentation \
    --seg-command "python predict.py --input {input} --mask {mask}" \
    --output arm_with_bandaid.jpg
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class WoundDetection:
    contour: np.ndarray
    center: Tuple[int, int]
    angle: float
    bbox: Tuple[int, int, int, int]


def _largest_contour_from_mask(mask: np.ndarray) -> Optional[WoundDetection]:
    """Convert a binary mask into contour geometry for overlay placement."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = binary.shape[:2]
    min_area = (h * w) * 0.0001
    valid = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    if not valid:
        return None

    contour = max(valid, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    (cx, cy), _, angle = rect
    if angle < -45:
        angle += 90

    x, y, bw, bh = cv2.boundingRect(contour)
    return WoundDetection(
        contour=contour,
        center=(int(cx), int(cy)),
        angle=float(angle),
        bbox=(x, y, bw, bh),
    )


def detect_wound_with_repo(
    image_path: Path,
    seg_repo: Path,
    seg_command: str,
    keep_mask: Optional[Path] = None,
) -> Optional[WoundDetection]:
    """Run wound-segmentation repository command to produce a mask and detect wound geometry.

    seg_command supports placeholders:
    - {input}: absolute input image path
    - {mask}: absolute output mask path

    Example command template:
      python predict.py --input {input} --mask {mask}
    """
    seg_repo = seg_repo.resolve()
    image_path = image_path.resolve()

    if not seg_repo.exists():
        raise FileNotFoundError(f"Segmentation repo path not found: {seg_repo}")

    with tempfile.TemporaryDirectory(prefix="wound_mask_") as tmpdir:
        tmp_mask = Path(tmpdir) / "wound_mask.png"
        cmd = seg_command.format(input=str(image_path), mask=str(tmp_mask))

        proc = subprocess.run(
            shlex.split(cmd),
            cwd=str(seg_repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Wound segmentation command failed.\n"
                f"Command: {cmd}\n"
                f"Exit code: {proc.returncode}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        if not tmp_mask.exists():
            raise RuntimeError(
                "Segmentation finished but mask was not produced. "
                "Check --seg-command placeholders and output argument."
            )

        mask = cv2.imread(str(tmp_mask), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read generated mask: {tmp_mask}")

        if keep_mask is not None:
            cv2.imwrite(str(keep_mask), mask)

        return _largest_contour_from_mask(mask)


def detect_wound_heuristic(image_bgr: np.ndarray) -> Optional[WoundDetection]:
    """Fallback redness-based wound detection when repo model is unavailable."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    lower_red1 = np.array([0, 35, 35], dtype=np.uint8)
    upper_red1 = np.array([12, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([160, 35, 35], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

    mask_hsv = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask_lab = cv2.threshold(lab[:, :, 1], 145, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.bitwise_and(mask_hsv, mask_lab)
    return _largest_contour_from_mask(mask)


def create_default_bandaid(width: int = 320, height: int = 120) -> np.ndarray:
    """Create a synthetic RGBA band-aid image."""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    strip_color = (196, 190, 173, 255)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), strip_color, -1)

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


def overlay_rgba(base_bgr: np.ndarray, overlay_rgba: np.ndarray, center: Tuple[int, int], angle: float) -> np.ndarray:
    out = base_bgr.copy()
    oh, ow = overlay_rgba.shape[:2]

    rotation = cv2.getRotationMatrix2D((ow // 2, oh // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        overlay_rgba,
        rotation,
        (ow, oh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    x0, y0 = center[0] - ow // 2, center[1] - oh // 2
    x1, y1 = x0 + ow, y0 + oh

    bx0, by0 = max(0, x0), max(0, y0)
    bx1, by1 = min(out.shape[1], x1), min(out.shape[0], y1)
    if bx0 >= bx1 or by0 >= by1:
        return out

    ox0, oy0 = bx0 - x0, by0 - y0
    ox1, oy1 = ox0 + (bx1 - bx0), oy0 + (by1 - by0)
    overlay_crop = rotated[oy0:oy1, ox0:ox1]

    alpha = overlay_crop[:, :, 3:4].astype(np.float32) / 255.0
    base_crop = out[by0:by1, bx0:bx1].astype(np.float32)
    over_rgb = overlay_crop[:, :, :3].astype(np.float32)

    out[by0:by1, bx0:bx1] = (alpha * over_rgb + (1.0 - alpha) * base_crop).astype(np.uint8)
    return out


def apply_bandaid(image_bgr: np.ndarray, bandaid_rgba: np.ndarray, detection: WoundDetection) -> np.ndarray:
    _, _, bw, bh = detection.bbox
    wound_scale = max(bw, bh)
    target_w = int(max(90, wound_scale * 2.8))
    target_h = int(target_w * (bandaid_rgba.shape[0] / bandaid_rgba.shape[1]))
    resized = cv2.resize(bandaid_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return overlay_rgba(image_bgr, resized, detection.center, detection.angle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect wound and place a band-aid.")
    parser.add_argument("--input", required=True, type=Path, help="Path to arm image")
    parser.add_argument("--output", type=Path, default=Path("output_with_bandaid.jpg"), help="Output image path")
    parser.add_argument("--bandaid", type=Path, default=None, help="Optional RGBA band-aid image")

    parser.add_argument("--seg-repo", type=Path, default=None, help="Path to cloned wound-segmentation repo")
    parser.add_argument(
        "--seg-command",
        type=str,
        default="python predict.py --input {input} --mask {mask}",
        help="Command template run in --seg-repo. Must use {input} and {mask} placeholders.",
    )
    parser.add_argument("--mask-out", type=Path, default=None, help="Optional path to save generated wound mask")
    parser.add_argument(
        "--fallback-heuristic",
        action="store_true",
        help="Use redness heuristic if repo segmentation is missing/fails.",
    )
    parser.add_argument("--no-show", action="store_true", help="Skip side-by-side display window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {args.input}")

    detection = None
    repo_error = None

    if args.seg_repo is not None:
        try:
            detection = detect_wound_with_repo(
                image_path=args.input,
                seg_repo=args.seg_repo,
                seg_command=args.seg_command,
                keep_mask=args.mask_out,
            )
        except Exception as exc:
            repo_error = exc

    if detection is None and args.fallback_heuristic:
        detection = detect_wound_heuristic(image)

    if detection is None:
        msg = "Wound detection failed."
        if args.seg_repo is None:
            msg += " Provide --seg-repo for model-based segmentation."
        if repo_error is not None:
            msg += f" Repo error: {repo_error}"
        if not args.fallback_heuristic:
            msg += " You can also enable --fallback-heuristic."
        raise RuntimeError(msg)

    bandaid = load_bandaid_image(args.bandaid)
    output = apply_bandaid(image, bandaid, detection)
    cv2.imwrite(str(args.output), output)

    if not args.no_show:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        axes[1].set_title("With Band-Aid")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    print(f"Saved result to: {args.output}")


if __name__ == "__main__":
    main()
