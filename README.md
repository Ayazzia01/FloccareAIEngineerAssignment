# Wound Band‑Aid Overlay (Deepskin + OpenCV)

This script detects wound regions using a Deepskin segmentation mask and overlays band‑aids aligned to the wound’s PCA axes. It saves a 2×2 grid image: **original | mask | overlay | overlayed**.

## Requirements

- Python 3.9+
- `opencv-python`, `numpy`, `matplotlib`
- Local clone of `Nico-Curti/Deepskin`

```bash
pip install opencv-python numpy matplotlib
```

## Mask Format

The segmentation mask is expected to be **3‑channel** with this order:

- channel 0: background
- channel 1: skin
- channel 2: wound

If wound/skin channels are empty, the script falls back to area‑based inference.

## Usage

Single image:

```bash
python wound_bandaid.py \
  --input path/to/image.jpg \
  --seg-repo /path/to/Deepskin
```

Batch:

```bash
python wound_bandaid.py \
  --input-dir wound_images \
  --seg-repo /path/to/Deepskin
```

### Example (tuned settings)

```bash
python wound_bandaid.py \
  --input-dir ..\wound_images \
  --bandaid ..\band-aid.png \
  --seg-repo ..\Deepskin \
  --workers 5 \
  --blend seamless \
  --size-scale 2.5 \
  --margin-scale 1.05 \
  --edge-blur 2 \
  --shadow-strength 0.03 \
  --shadow-blur 7 \
  --wrap-strength 0.05
```

## Outputs

For each input image the script saves a **2×2 grid**:

1) Original image  
2) Raw segmentation mask  
3) Band‑aid overlay (RGBA)  
4) Final overlayed image

By default, the combined grid is saved as:

- `*_with_bandaid_compare.jpg`

If `--overlay-out` or `--overlay-dir` is provided, the raw overlay PNG is also written.

## How Placement Works

- Wound orientation is computed with **PCA**.
- Two band‑aids are applied per wound:
  - one along the **major axis**
  - one along the **minor axis**
- Band‑aids are resized independently along both axes.
- Band‑aid pixels are **clipped to skin + wound** so they never spill onto background.
- If the band‑aid spills onto background, it is **wrapped only on the spilling edge**.

## Key Arguments

- `--size-scale`: base size multiplier along PCA axes.
- `--margin-scale`: extra padding beyond the wound size.
- `--edge-blur`: softens band‑aid edges.
- `--shadow-strength`: shadow intensity (0 disables).
- `--shadow-blur`: shadow softness (0 disables).
- `--wrap-strength`: wrapping deformation at edges.
- `--blend`: `seamless` or `alpha`.

## Troubleshooting

- If band‑aids appear on background, confirm your mask channel order is `background, skin, wound`.
- If coverage is short, increase `--margin-scale`.
- If it looks too sharp, increase `--edge-blur` slightly.
