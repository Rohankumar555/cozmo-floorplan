# Reproduction bundle

Raw sensor data is **not** in git (photos ~97 MB, walkthrough MOV ~307 MB, Stray RGB-D ~171 MB). The zip is a GitHub Release asset on this same repo:

https://github.com/Rohankumar555/cozmo-floorplan/releases/download/reproduction-v1/cozmo-reproduction-bundle.zip

Release page: https://github.com/Rohankumar555/cozmo-floorplan/releases/tag/reproduction-v1

## Unpack

From a clone of this repo:

```bash
curl -L -o cozmo-reproduction-bundle.zip \
  https://github.com/Rohankumar555/cozmo-floorplan/releases/download/reproduction-v1/cozmo-reproduction-bundle.zip
unzip cozmo-reproduction-bundle.zip
```

That writes `captures/photos/`, `captures/video/IMG_3589.MOV`, `captures/lidar/property/6532fa113b/`, and `captures/incumbent/polycam/`. Do **not** unzip the failed Polycam-web zip (`Cozmox AI Capture.zip`); it is the same MOV.

## Regenerate every reported number

```bash
/opt/homebrew/bin/python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -e .
pip install einops huggingface_hub safetensors
pip install "git+https://github.com/facebookresearch/vggt.git"
export PYTORCH_ENABLE_MPS_FALLBACK=1

python -m cozmo run captures/ --out out/ --backend vggt
python -m cozmo run captures/ --out out/video --tier video --backend vggt
python -m cozmo run captures/ --out out/lidar --tier lidar
python -m cozmo run captures/ --out out/fix-after --only room_01 --backend vggt
```

Compare `out/*/plan.json` to `reports/cache/` (committed snapshots of the same commands). Tape is `reports/ground-truth.json`. Polycam dimensions are `captures/incumbent/polycam/measurements.json` (PNG is in the zip and in git).

Weights: VGGT from Hugging Face, YOLO-World `yolov8s-worldv2.pt` on first opening detect. Offline replay of `reports/cache/*.json` does not need them; the live walk-in does.

## Rebuild this zip

```bash
python scripts/pack_reproduction.py
```

Skips `captures/video/Cozmox AI Capture.zip` (duplicate MOV).

## What is not here

Repeat capture, kitchen tape, damage classes — skipped this submit. See `reports/compliance-matrix.md`.
