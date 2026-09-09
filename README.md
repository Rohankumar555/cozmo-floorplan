# cozmo-floorplan

Local pipeline for the Cozmo case study: handheld iPhone capture in, one stitched dimensioned floor plan plus JSON out.

Capture is Route 2 (stock iOS Camera + a LiDAR logging app). This repo is the reconstruction CLI, not an iPhone app.

## Capture layout

```text
captures/
  photos/<room_name>/*.jpg   # 2–8 stills per room, no depth, no poses
  video/*.mov                # one walkthrough; cut at door holds (`--tier video`)
  lidar/                     # depth + poses + intrinsics export (later)
```

## Run

```bash
/opt/homebrew/bin/python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -e .
python -m cozmo run captures/ --out out/ --only room_01
```

Omit `--only` to reconstruct **every** photo folder and **stitch** them at doors (`stitch: door_graph`). `--only` stays one room, unstitched.

`--backend auto` uses VGGT when the `vggt` package is installed, otherwise a Manhattan line-box fallback. **Manhattan is not removed.**

```bash
source .venv312/bin/activate
pip install einops huggingface_hub safetensors
pip install "git+https://github.com/facebookresearch/vggt.git"
python -m cozmo run captures/ --out out/ --backend vggt
python -m cozmo run captures/ --out out/video --tier video --backend vggt
```

Output:

- `out/plan.json` — internal schema `cozmo.plan.v0` (CIs on every measurement)
- `out/plan.svg` / `out/rooms/<id>.svg` — top-down sketch from the same object

## What this slice does

Photo-tier per room, then a **door-graph stitch** (hub = hallway folder) when you run all folders. `--tier video` cuts the walkthrough at door holds and stitches **in walk order** (`walk_graph`). Drift: door snaps, not poses-as-is.

Not yet: LiDAR, damage rules.

## Disclosures

- VGGT `facebook/VGGT-1B` — pretrained, inference only
- YOLO-World `yolov8s-worldv2.pt` — pretrained, prompts `door` / `window`
- Photo scale: 0.80 m interior door-width prior, not LiDAR-metric
