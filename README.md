# cozmo-floorplan

**Submit / source of truth:** [github.com/Rohankumar555/cozmo-floorplan](https://github.com/Rohankumar555/cozmo-floorplan)

Local pipeline for the Cozmo case study: handheld iPhone capture in, one stitched dimensioned floor plan plus JSON out.

Capture is Route 2 (stock iOS Camera + a LiDAR logging app). This repo is the reconstruction CLI, not an iPhone app.

**Raw captures** (photos, walkthrough MOV, Stray LiDAR, Polycam PNG) live on the GitHub Release, not in git:

https://github.com/Rohankumar555/cozmo-floorplan/releases/download/reproduction-v1/cozmo-reproduction-bundle.zip

## Capture layout

```text
captures/
  photos/<room_name>/*.jpg   # 2–8 stills per room, no depth, no poses
  video/*.mov                # one walkthrough; cut at door holds (`--tier video`)
  lidar/<stray_export>/     # Stray Scanner: depth/, odometry.csv, camera_matrix.csv, rgb.mp4
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
python -m cozmo run captures/ --out out/lidar --tier lidar
```

Output:

- `out/plan.json` — internal schema `cozmo.plan.v0` (CIs on every measurement)
- `out/plan.svg` / `out/rooms/<id>.svg` — top-down sketch from the same object

## Docs (submit packet)

- `docs/protocol.md` — one-page stock capture (Camera, Stray, Polycam)
- `docs/device-matrix.md` — which hardware runs which tier
- `docs/reproduction.md` — unpack the zip, regenerate every reported number
- `reports/compliance-matrix.md`
- `reports/benchmark.md` — tape vs Polycam vs our tiers
- `reports/ground-truth.json` — steel-tape GT
- `reports/cache/` — committed `plan.json` snapshots from the runs above
- `reports/fix-loop.md` — worst gate + shipped hull fix
- `reports/technical-report.md` — architecture (keep under 6 pages)

Raw photos, the walkthrough MOV, and Stray depth are **not** in git. Download the release zip, then unzip at the clone root:

```bash
curl -L -o cozmo-reproduction-bundle.zip \
  https://github.com/Rohankumar555/cozmo-floorplan/releases/download/reproduction-v1/cozmo-reproduction-bundle.zip
unzip cozmo-reproduction-bundle.zip
```

Details: `docs/reproduction.md`. Rebuild locally with `python scripts/pack_reproduction.py`.

## What this slice does

Photo-tier per room, then a **door-graph stitch** (hub = hallway folder) when you run all folders. `--tier video` cuts the walkthrough at door holds and stitches **in walk order** (`walk_graph`). `--tier lidar` unprojects a Stray RGB-D walk (metric depth) and aligns the floor plane (not poses-as-is).

Not yet: damage rules.

## Disclosures

- VGGT `facebook/VGGT-1B` — pretrained, inference only
- YOLO-World `yolov8s-worldv2.pt` — pretrained, prompts `door` / `window`
- Photo scale: 0.80 m interior door-width prior, not LiDAR-metric
