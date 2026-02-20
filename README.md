# Wound Band-Aid Overlay (Model-based)

This project provides a Python script that places a band-aid on a wound, using
**wound segmentation from**:

- https://github.com/uwm-bigdata/wound-segmentation

## What it does

1. Takes an arm photo (`--input`).
2. Runs a segmentation command from the wound-segmentation repo to generate a wound mask.
3. Detects wound center/size/orientation from that mask.
4. Overlays a rotated/scaled band-aid on the wound.
5. Shows original and modified images side-by-side.

## Requirements

- Python 3.9+
- `opencv-python`
- `numpy`
- `matplotlib`
- A local clone of `uwm-bigdata/wound-segmentation`

Install Python deps:

```bash
pip install opencv-python numpy matplotlib
```

Clone segmentation repo (outside this repo):

```bash
git clone https://github.com/uwm-bigdata/wound-segmentation.git
```

## Usage

Run with repo-based segmentation:

```bash
python wound_bandaid.py \
  --input path/to/arm_photo.jpg \
  --seg-repo /path/to/wound-segmentation \
  --seg-command "python predict.py --input {input} --mask {mask}" \
  --output result.jpg
```

### Key arguments

- `--seg-repo`: local path to cloned `wound-segmentation` repo.
- `--seg-command`: command template executed in `--seg-repo`.
  - Must include placeholders:
    - `{input}`: absolute input image path
    - `{mask}`: absolute output mask path
- `--bandaid path/to/bandaid.png`: optional custom RGBA band-aid image.
- `--mask-out wound_mask.png`: save the generated segmentation mask.
- `--fallback-heuristic`: fallback to redness heuristic if repo detection fails.
- `--no-show`: skip matplotlib side-by-side display.

## Notes

- The default `--seg-command` is `python predict.py --input {input} --mask {mask}`.
  Adjust this command to match the exact CLI provided by your checked-out version
  of `uwm-bigdata/wound-segmentation`.
- If you omit `--seg-repo`, detection fails unless `--fallback-heuristic` is enabled.
