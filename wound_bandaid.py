#!/usr/bin/env python3
"""
Place a band-aid on a wound using a segmentation mask from
https://github.com/Nico-Curti/Deepskin.

Primary workflow:
  1) Run the external Deepskin repo to generate a wound mask.
  2) Find wound contour/center/orientation from that mask.
  3) Overlay a rotated/scaled band-aid.
"""

from __future__ import annotations

import argparse
import math
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request
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


@dataclass
class PoseVector:
    dx: float
    dy: float
    mx: float
    my: float
    p0: Tuple[int, int]
    p1: Tuple[int, int]


def _contour_info_from_mask(mask: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int], float, Tuple[int, int, int, int]]]:
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
    valid = [cnt for cnt in contours if cv2.contourArea(cnt) > (h * w) * 0.0001]
    if not valid:
        return None

    contour = max(valid, key=cv2.contourArea)
    (cx, cy), _, angle = cv2.minAreaRect(contour)
    if angle < -45:
        angle += 90

    x, y, bw, bh = cv2.boundingRect(contour)
    return contour, (int(cx), int(cy)), float(angle), (x, y, bw, bh)


def _largest_contour_from_mask(mask: np.ndarray) -> Optional[WoundDetection]:
    info = _contour_info_from_mask(mask)
    if info is None:
        return None
    contour, center, angle, bbox = info
    return WoundDetection(contour=contour, center=center, angle=angle, bbox=bbox)


def _default_deepskin_mask_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_deepskin_mask.png")


def _default_deepblend_repo() -> Path:
    return Path(__file__).resolve().parents[1] / "DeepImageBlending"


def _split_command(cmd: str) -> list[str]:
    parts = shlex.split(cmd, posix=False)
    cleaned: list[str] = []
    for part in parts:
        if len(part) >= 2 and part[0] == part[-1] and part[0] in ("\"", "'"):
            part = part[1:-1]
        cleaned.append(part)
    return cleaned


def detect_wound_with_deepskin(
    image_path: Path,
    seg_repo: Path,
    seg_command: str,
    keep_mask: Optional[Path] = None,
) -> Optional[WoundDetection]:
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
        python_exe = sys.executable
        cmd = seg_command.format(input=str(image_path), mask=str(tmp_mask), python=python_exe)

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

        if mask.ndim == 3:
            # Deepskin semantic mask uses red for wound ROI (OpenCV loads BGR).
            mask = mask[:, :, 2]

        return _largest_contour_from_mask(mask)


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


def _rotate_rgba(overlay_rgba: np.ndarray, angle: float) -> np.ndarray:
    oh, ow = overlay_rgba.shape[:2]
    return cv2.warpAffine(
        overlay_rgba,
        cv2.getRotationMatrix2D((ow // 2, oh // 2), angle, 1.0),
        (ow, oh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _warp_wrap(overlay_rgba: np.ndarray, wrap_x: float, wrap_y: float) -> np.ndarray:
    if wrap_x == 0 and wrap_y == 0:
        return overlay_rgba
    h, w = overlay_rgba.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    xn = (x - cx) / (w * 0.5)
    yn = (y - cy) / (h * 0.5)

    dx = wrap_x * (1.0 - yn ** 2) * np.sin(math.pi * xn) * (w * 0.5)
    dy = wrap_y * (1.0 - xn ** 2) * np.sin(math.pi * yn) * (h * 0.5)

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


def _pad_to_square_rgb(img: np.ndarray, fill: Tuple[int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = img.shape[:2]
    size = max(h, w)
    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left
    padded = cv2.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=fill,
    )
    return padded, (pad_left, pad_top, size, size)


def _pad_to_square_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    size = max(h, w)
    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left
    return cv2.copyMakeBorder(
        mask,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )


def _deep_blend_with_repo(
    base_bgr: np.ndarray,
    overlay_crop: np.ndarray,
    roi: Tuple[int, int, int, int],
    center: Tuple[int, int],
    repo_path: Path,
    target_size: int,
    num_steps: int,
    gpu_id: int,
) -> Optional[np.ndarray]:
    run_py = repo_path / "run.py"
    if not run_py.exists():
        print(f"DeepImageBlending run.py not found at {run_py}")
        return None

    target_h, target_w = base_bgr.shape[:2]
    ts = int(max(64, target_size))
    ss = int(max(64, target_size))
    if ss >= ts:
        ss = max(32, ts - 2)

    overlay_rgb = overlay_crop[:, :, :3]
    overlay_mask = overlay_crop[:, :, 3]
    overlay_mask = np.where(overlay_mask > 0, 255, 0).astype(np.uint8)

    overlay_rgb, _ = _pad_to_square_rgb(overlay_rgb, (0, 0, 0))
    overlay_mask = _pad_to_square_mask(overlay_mask)

    with tempfile.TemporaryDirectory(prefix="deepblend_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        source_path = tmpdir_path / "source.png"
        mask_path = tmpdir_path / "mask.png"
        target_path = tmpdir_path / "target.png"
        output_dir = tmpdir_path / "out"
        output_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(source_path), overlay_rgb)
        cv2.imwrite(str(mask_path), overlay_mask)
        cv2.imwrite(str(target_path), base_bgr)

        cx, cy = center
        x = int(cy * ts / max(1, target_h))
        y = int(cx * ts / max(1, target_w))
        half = ss // 2
        x = max(half, min(ts - half, x))
        y = max(half, min(ts - half, y))

        cmd = [
            sys.executable,
            str(run_py),
            "--source_file",
            str(source_path),
            "--mask_file",
            str(mask_path),
            "--target_file",
            str(target_path),
            "--output_dir",
            str(output_dir),
            "--ss",
            str(ss),
            "--ts",
            str(ts),
            "--x",
            str(x),
            "--y",
            str(y),
            "--gpu_id",
            str(gpu_id),
            "--num_steps",
            str(num_steps),
        ]
        proc = subprocess.run(cmd, cwd=str(repo_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            print("DeepImageBlending failed.")
            print(proc.stdout)
            print(proc.stderr)
            return None

        result_path = output_dir / "second_pass.png"
        if not result_path.exists():
            print("DeepImageBlending output not found.")
            return None

        blended = cv2.imread(str(result_path), cv2.IMREAD_COLOR)
        if blended is None:
            print("Failed to read DeepImageBlending output.")
            return None

        if blended.shape[0] != target_h or blended.shape[1] != target_w:
            blended = cv2.resize(blended, (target_w, target_h), interpolation=cv2.INTER_AREA)

        return blended


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


def _alpha_blend(
    base_bgr: np.ndarray, overlay_crop: np.ndarray, roi: Tuple[int, int, int, int]
) -> np.ndarray:
    bx0, by0, bx1, by1 = roi
    out = base_bgr.copy()
    alpha = overlay_crop[:, :, 3:4].astype(np.float32) / 255.0
    base_crop = out[by0:by1, bx0:bx1].astype(np.float32)
    out[by0:by1, bx0:bx1] = (alpha * overlay_crop[:, :, :3] + (1.0 - alpha) * base_crop).astype(np.uint8)
    return out


def _seamless_blend(
    base_bgr: np.ndarray, overlay_crop: np.ndarray, roi: Tuple[int, int, int, int]
) -> np.ndarray:
    bx0, by0, bx1, by1 = roi
    mask = overlay_crop[:, :, 3]
    if np.max(mask) == 0:
        return base_bgr
    mask_bin = np.where(mask > 0, 255, 0).astype(np.uint8)
    center = (bx0 + (bx1 - bx0) // 2, by0 + (by1 - by0) // 2)
    return cv2.seamlessClone(overlay_crop[:, :, :3], base_bgr, mask_bin, center, cv2.NORMAL_CLONE)


def render_pose_debug(
    image_bgr: np.ndarray,
    landmarks,
    wound_center: Tuple[int, int],
    chosen: PoseVector,
    min_visibility: float,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    debug_img = image_bgr.copy()
    for lm in landmarks:
        vis = getattr(lm, "visibility", 1.0)
        if vis < min_visibility:
            continue
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(debug_img, (x, y), 2, (0, 255, 0), -1)

    cv2.circle(debug_img, wound_center, 4, (0, 0, 255), -1)
    cv2.line(debug_img, chosen.p0, chosen.p1, (255, 0, 0), 2)
    cv2.putText(
        debug_img,
        f"pose angle: {math.degrees(math.atan2(chosen.dy, chosen.dx)):.1f}",
        (max(5, wound_center[0] + 6), max(15, wound_center[1] - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return debug_img


def _angle_to_vector(angle_deg: float) -> Tuple[float, float]:
    radians = math.radians(angle_deg)
    return math.cos(radians), math.sin(radians)


def _blend_angles(angle_a: float, angle_b: float, weight_b: float) -> float:
    weight_b = float(np.clip(weight_b, 0.0, 1.0))
    weight_a = 1.0 - weight_b
    ax, ay = _angle_to_vector(angle_a)
    bx, by = _angle_to_vector(angle_b)
    vx = ax * weight_a + bx * weight_b
    vy = ay * weight_a + by * weight_b
    if vx == 0 and vy == 0:
        return angle_a
    return math.degrees(math.atan2(vy, vx))


def _default_pose_model_url() -> str:
    return (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    )


def _default_pose_model_cache_path() -> Path:
    return Path.home() / ".cache" / "mediapipe" / "pose_landmarker_heavy.task"


def _download_pose_model(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".download")
    try:
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
            out.write(response.read())
        tmp.replace(dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return dest


def _pose_vector(landmarks, idx_a, idx_b, idx_c, image_shape, min_visibility: float) -> Optional[Tuple[float, float, float]]:
    h, w = image_shape[:2]
    a = landmarks[idx_a]
    b = landmarks[idx_b]
    c = landmarks[idx_c]

    b_vis = getattr(b, "visibility", 1.0)
    c_vis = getattr(c, "visibility", 1.0)
    a_vis = getattr(a, "visibility", 1.0)

    if b_vis < min_visibility or c_vis < min_visibility:
        if a_vis < min_visibility or b_vis < min_visibility:
            return None
        bx, by = int(b.x * w), int(b.y * h)
        ax, ay = int(a.x * w), int(a.y * h)
        dx, dy = bx - ax, by - ay
        mx, my = float((bx + ax) / 2), float((by + ay) / 2)
        return PoseVector(float(dx), float(dy), mx, my, (ax, ay), (bx, by))

    bx, by = int(b.x * w), int(b.y * h)
    cx, cy = int(c.x * w), int(c.y * h)
    dx, dy = cx - bx, cy - by
    mx, my = float((bx + cx) / 2), float((by + cy) / 2)
    return PoseVector(float(dx), float(dy), mx, my, (bx, by), (cx, cy))


def estimate_pose_angle(
    image_bgr: np.ndarray,
    wound_center: Tuple[int, int],
    min_visibility: float,
    min_detection_confidence: float,
    model_complexity: int,
    model_path: Optional[Path],
    model_url: Optional[str],
    auto_download: bool,
    debug: bool,
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    try:
        import mediapipe as mp  # type: ignore
    except Exception:
        return None, None

    if hasattr(mp, "solutions"):
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
        ) as pose:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks is None:
                return None, None

            landmarks = results.pose_landmarks.landmark
            left = _pose_vector(
                landmarks,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.LEFT_WRIST,
                image_bgr.shape,
                min_visibility,
            )
            right = _pose_vector(
                landmarks,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_WRIST,
                image_bgr.shape,
                min_visibility,
            )

            candidates: list[Tuple[float, PoseVector]] = []
            cx, cy = wound_center
            for vec in (left, right):
                if vec is None:
                    continue
                dist = (vec.mx - cx) ** 2 + (vec.my - cy) ** 2
                candidates.append((dist, vec))

            if not candidates:
                return None, None

            _, chosen = min(candidates, key=lambda item: item[0])
            if chosen.dx == 0 and chosen.dy == 0:
                return None, None
            angle = math.degrees(math.atan2(chosen.dy, chosen.dx))
            debug_img = None
            if debug:
                debug_img = render_pose_debug(image_bgr, landmarks, wound_center, chosen, min_visibility)
            return angle, debug_img

    if model_path is None:
        model_path = _default_pose_model_cache_path()
        if not model_path.exists():
            if not auto_download:
                print("MediaPipe Tasks API detected but no model found; set --pose-model-path or enable download.")
                return None, None
            url = model_url or _default_pose_model_url()
            try:
                _download_pose_model(url, model_path)
                print(f"Downloaded MediaPipe pose model to {model_path}")
            except Exception as exc:
                print(
                    "Failed to download MediaPipe pose model. "
                    "Provide --pose-model-path or check network access. "
                    f"Error: {exc}"
                )
                return None, None

    try:
        from mediapipe.tasks import python as mp_python  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore
    except Exception:
        return None

    model_path = model_path.expanduser().resolve()
    if not model_path.exists():
        print("MediaPipe pose model not found. Provide --pose-model-path or enable download.")
        return None, None

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_detection_confidence,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        pose_landmarks = getattr(result, "pose_landmarks", None)
        if not pose_landmarks:
            return None, None

        landmarks = pose_landmarks[0]
        left = _pose_vector(
            landmarks,
            vision.PoseLandmark.LEFT_SHOULDER,
            vision.PoseLandmark.LEFT_ELBOW,
            vision.PoseLandmark.LEFT_WRIST,
            image_bgr.shape,
            min_visibility,
        )
        right = _pose_vector(
            landmarks,
            vision.PoseLandmark.RIGHT_SHOULDER,
            vision.PoseLandmark.RIGHT_ELBOW,
            vision.PoseLandmark.RIGHT_WRIST,
            image_bgr.shape,
            min_visibility,
        )

        candidates: list[Tuple[float, PoseVector]] = []
        cx, cy = wound_center
        for vec in (left, right):
            if vec is None:
                continue
            dist = (vec.mx - cx) ** 2 + (vec.my - cy) ** 2
            candidates.append((dist, vec))

        if not candidates:
            return None, None

        _, chosen = min(candidates, key=lambda item: item[0])
        if chosen.dx == 0 and chosen.dy == 0:
            return None, None
        angle = math.degrees(math.atan2(chosen.dy, chosen.dx))
        debug_img = None
        if debug:
            debug_img = render_pose_debug(image_bgr, landmarks, wound_center, chosen, min_visibility)
        return angle, debug_img


def apply_bandaid(
    image_bgr: np.ndarray,
    bandaid_rgba: np.ndarray,
    detection: WoundDetection,
    blend: str,
    edge_blur: int,
    shadow_strength: float,
    shadow_blur: int,
    size_scale: float,
    wrap_strength: float,
    stretch_to_wound: bool,
    deep_blend: bool,
    deep_blend_repo: Optional[Path],
    deep_blend_size: int,
    deep_blend_steps: int,
    deep_blend_gpu_id: int,
    overlay_out: Optional[Path],
) -> np.ndarray:
    rect = cv2.minAreaRect(detection.contour)
    rect_w, rect_h = rect[1]
    if rect_w <= 0 or rect_h <= 0:
        _, _, bw, bh = detection.bbox
        rect_w, rect_h = float(bw), float(bh)
    target_w = int(max(40, rect_w * size_scale))
    target_h = int(max(40, rect_h * size_scale))

    if stretch_to_wound:
        resized = cv2.resize(bandaid_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        oh, ow = bandaid_rgba.shape[:2]
        scale = max(target_w / max(1, ow), target_h / max(1, oh))
        resized = cv2.resize(bandaid_rgba, (int(ow * scale), int(oh * scale)), interpolation=cv2.INTER_AREA)

    wrap_x = wrap_strength * min(0.6, target_h / max(1.0, target_w))
    wrap_y = wrap_strength * min(0.6, target_w / max(1.0, target_h))
    wrapped = _warp_wrap(resized, wrap_x, wrap_y)
    rotated = _rotate_rgba(wrapped, detection.angle)
    softened = _soften_edges(rotated, edge_blur)

    overlay_crop, roi = _place_overlay(softened, detection.center, image_bgr.shape)
    if overlay_crop is None or roi is None:
        return image_bgr

    if overlay_out is not None:
        overlay_canvas = np.zeros((image_bgr.shape[0], image_bgr.shape[1], 4), dtype=np.uint8)
        bx0, by0, bx1, by1 = roi
        overlay_canvas[by0:by1, bx0:bx1] = overlay_crop
        overlay_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_out), overlay_canvas)

    out = image_bgr.copy()
    _apply_shadow(out, overlay_crop[:, :, 3], roi, shadow_strength, shadow_blur)

    if deep_blend:
        repo_path = deep_blend_repo or _default_deepblend_repo()
        blended = _deep_blend_with_repo(
            out,
            overlay_crop,
            roi,
            detection.center,
            repo_path=repo_path,
            target_size=deep_blend_size,
            num_steps=deep_blend_steps,
            gpu_id=deep_blend_gpu_id,
        )
        if blended is not None:
            return blended
        print("DeepImageBlending unavailable; falling back to seamless blend.")
        return _seamless_blend(out, overlay_crop, roi)

    if blend == "seamless":
        return _seamless_blend(out, overlay_crop, roi)
    return _alpha_blend(out, overlay_crop, roi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect wound and place a band-aid.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None, help="Process all images in a folder.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (defaults to <input>_with_bandaid.jpg).",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output folder for batch runs.")
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
    parser.add_argument("--mask-dir", type=Path, default=None, help="Mask output folder for batch runs.")
    parser.add_argument(
        "--blend",
        choices=("seamless", "alpha", "deep"),
        default="seamless",
        help="Blending mode for the band-aid overlay.",
    )
    parser.add_argument(
        "--edge-blur",
        type=int,
        default=5,
        help="Gaussian blur kernel size to soften band-aid edges (0 to disable).",
    )
    parser.add_argument(
        "--shadow-strength",
        type=float,
        default=0.18,
        help="Shadow strength under the band-aid (0 to disable).",
    )
    parser.add_argument(
        "--shadow-blur",
        type=int,
        default=21,
        help="Gaussian blur kernel size for the shadow (0 to disable).",
    )
    parser.add_argument(
        "--size-scale",
        type=float,
        default=2.2,
        help="Scale factor for band-aid size relative to wound size.",
    )
    parser.add_argument(
        "--wrap-strength",
        type=float,
        default=0.18,
        help="Warp strength to wrap the band-aid in both directions (0 to disable).",
    )
    parser.add_argument(
        "--stretch-to-wound",
        action="store_true",
        help="Stretch band-aid to match wound width/height (non-uniform scaling).",
    )
    parser.add_argument(
        "--deep-blend-repo",
        type=Path,
        default=None,
        help="Path to DeepImageBlending repo (optional).",
    )
    parser.add_argument(
        "--deep-blend-size",
        type=int,
        default=512,
        help="Target size for DeepImageBlending (square).",
    )
    parser.add_argument(
        "--deep-blend-steps",
        type=int,
        default=200,
        help="Optimization steps for DeepImageBlending.",
    )
    parser.add_argument(
        "--deep-blend-gpu-id",
        type=int,
        default=0,
        help="GPU id for DeepImageBlending.",
    )
    parser.add_argument(
        "--overlay-out",
        type=Path,
        default=None,
        help="Optional path to save the transformed overlay (defaults to <input>_overlay.png).",
    )
    parser.add_argument("--overlay-dir", type=Path, default=None, help="Overlay output folder for batch runs.")
    parser.add_argument(
        "--use-mediapipe",
        action="store_true",
        help="Use MediaPipe Pose to refine band-aid rotation if available.",
    )
    parser.add_argument(
        "--pose-weight",
        type=float,
        default=0.7,
        help="Blend weight for MediaPipe pose angle (0=mask only, 1=pose only).",
    )
    parser.add_argument(
        "--pose-min-visibility",
        type=float,
        default=0.5,
        help="Minimum landmark visibility for pose-derived orientation.",
    )
    parser.add_argument(
        "--pose-min-confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence for MediaPipe Pose.",
    )
    parser.add_argument(
        "--pose-model-complexity",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="MediaPipe Pose model complexity (0=lite, 2=heavy).",
    )
    parser.add_argument(
        "--pose-model-path",
        type=Path,
        default=None,
        help="Path to MediaPipe Pose task model (required for MediaPipe Tasks API).",
    )
    parser.add_argument(
        "--pose-model-url",
        type=str,
        default=None,
        help="Optional URL to download the MediaPipe Pose task model.",
    )
    parser.add_argument(
        "--no-pose-model-download",
        action="store_false",
        dest="pose_model_auto_download",
        help="Disable auto-download of the MediaPipe Pose model (Tasks API only).",
    )
    parser.set_defaults(pose_model_auto_download=True)
    parser.add_argument(
        "--pose-debug-out",
        type=Path,
        default=None,
        help="Optional path to save a pose debug visualization image (defaults to <input>_pose_debug.jpg).",
    )
    parser.add_argument("--pose-debug-dir", type=Path, default=None, help="Pose debug output folder for batch runs.")
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

    def resolve_dir(arg: Optional[Path], default_dir: Path) -> Path:
        return arg if arg is not None else default_dir

    inputs = iter_inputs()
    if not inputs:
        raise FileNotFoundError("No input images found.")

    for input_path in inputs:
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {input_path}")
            continue

        if args.input_dir is not None:
            base_dir = args.input_dir
            output_dir = resolve_dir(args.output_dir, base_dir / "output")
            overlay_dir = resolve_dir(args.overlay_dir, base_dir / "overlays")
            mask_dir = resolve_dir(args.mask_dir, base_dir / "masks")
            pose_dir = resolve_dir(args.pose_debug_dir, base_dir / "pose_debug")
            output_dir.mkdir(parents=True, exist_ok=True)
            overlay_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            pose_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{input_path.stem}_with_bandaid.jpg"
            overlay_path = overlay_dir / f"{input_path.stem}_overlay.png"
            mask_path = mask_dir / f"{input_path.stem}_deepskin_mask.png"
            pose_debug_path = pose_dir / f"{input_path.stem}_pose_debug.jpg"
        else:
            output_path = args.output or input_path.with_name(f"{input_path.stem}_with_bandaid.jpg")
            overlay_path = args.overlay_out or input_path.with_name(f"{input_path.stem}_overlay.png")
            mask_path = args.mask_out
            pose_debug_path = args.pose_debug_out or input_path.with_name(f"{input_path.stem}_pose_debug.jpg")

        detection = None
        repo_error: Optional[Exception] = None
        if args.seg_repo is not None:
            try:
                detection = detect_wound_with_deepskin(input_path, args.seg_repo, args.seg_command, mask_path)
            except Exception as exc:
                repo_error = exc

        if detection is None and args.fallback_heuristic:
            detection = detect_wound_heuristic(image)

        if detection is None:
            msg = "Wound detection failed."
            if args.seg_repo is None:
                msg += " Provide --seg-repo for Deepskin segmentation."
            if repo_error is not None:
                msg += f" Repo error: {repo_error}"
            if not args.fallback_heuristic:
                msg += " You can also enable --fallback-heuristic."
            print(f"{input_path}: {msg}")
            continue

        if args.use_mediapipe:
            pose_angle, pose_debug = estimate_pose_angle(
                image,
                detection.center,
                min_visibility=args.pose_min_visibility,
                min_detection_confidence=args.pose_min_confidence,
                model_complexity=args.pose_model_complexity,
                model_path=args.pose_model_path,
                model_url=args.pose_model_url,
                auto_download=args.pose_model_auto_download,
                debug=pose_debug_path is not None,
            )
            if pose_debug is not None and pose_debug_path is not None:
                pose_debug_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(pose_debug_path), pose_debug)
            if pose_angle is not None:
                detection.angle = _blend_angles(detection.angle, pose_angle, args.pose_weight)
            else:
                print("MediaPipe pose not available; using mask orientation.")

        output = apply_bandaid(
            image,
            load_bandaid_image(args.bandaid),
            detection,
            blend=args.blend,
            edge_blur=args.edge_blur,
            shadow_strength=args.shadow_strength,
            shadow_blur=args.shadow_blur,
            size_scale=args.size_scale,
            wrap_strength=args.wrap_strength,
            stretch_to_wound=args.stretch_to_wound,
            deep_blend=args.blend == "deep",
            deep_blend_repo=args.deep_blend_repo,
            deep_blend_size=args.deep_blend_size,
            deep_blend_steps=args.deep_blend_steps,
            deep_blend_gpu_id=args.deep_blend_gpu_id,
            overlay_out=overlay_path,
        )
        cv2.imwrite(str(output_path), output)

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

        print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    main()
