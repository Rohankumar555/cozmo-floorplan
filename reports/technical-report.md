# Technical report (cozmo-floorplan)

Route 2 reconstruction CLI. Three input tiers, one JSON/SVG contract. Max length by design; engineering is in the repo.

## 1. Architecture

`python -m cozmo run captures/ [--tier photos|video|lidar] [--out DIR]`. Ingest is folder-shaped: `photos/<room>/*.jpg`, `video/*.mov`, `lidar/**/odometry.csv`. Each room becomes a `RoomPlan` (polygon, walls, openings, ceiling, area — every metric an `Interval`). Multi-room output is a `PropertyPlan` plus `plan.svg`.

Photo/video share `reconstruct_room`: few-view points (VGGT if installed, else Manhattan lines), RANSAC planes, min-area rectangle, YOLO-World doors/windows, scale from a 3D door width = 0.80 m else floor-tile period. LiDAR does **not** use VGGT: Stray millimetre depth is unprojected with `camera_matrix.csv` and ARKit poses (`cozmo/ingest/lidar.py`, `cozmo/reconstruct/lidar.py`).

Disclosures: VGGT `facebook/VGGT-1B` and YOLO-World `yolov8s-worldv2.pt`, inference only. LiDAR recon has no learned weights.

## 2. Tiers and device matrix

See `docs/device-matrix.md`. Photos and video run on any iPhone 15+. LiDAR and Polycam floorplan need a **Pro** (LiDAR scanner). The Mac only reconstructs.

**Photo.** Per-folder stills. Stitch is a **door graph**: satellites snap to unused hub (hallway) detector doors; same-wall rooms pack along that wall. No house-name priors. Intervals ±8% on walls.

**Video.** Portrait MOV rotated 90° CW. Door holds = still camera ≥1.2 s with walk on both sides. Stills per segment go through the same recon. Stitch is **walk order**, not YOLO pairing. Wall intervals ±3% — honest only if scale is right; on our clip most rooms used `tile_0.80m` and walls were 8–12 m.

**LiDAR.** Fuse subsampled RGB-D, RANSAC floor, rotate to z-up, occupancy polygon, optional cuts on odometry holds. Ablation in JSON: AABB **poses-as-is** vs **plane-aligned**. That is the drift row; poses are not used as-is.

## 3. Calibration and error budget

Order of scale: (1) tightest 3D door = 0.80 m, (2) 0.80×1.20 m tile grout, (3) unscaled scene units with CI 0.50. We never lock a wall at 6.67 m via “door = 12% of wall.”

Tape (Rohan): `room_01` 3.4×4.0 m ceiling 2.45; `room_02` 3.2×4.0; hallway 4.35×8.77. Kitchen none (skipped).

Photo vs tape (before hull fix): hallway 8.47 / 4.22 inside ±8%; `room_01` 3.48 vs 4.00 **−13%**; `room_02` 4.63 vs 4.00 **+16%**. Ceilings 4–6 m.

LiDAR ceiling **2.30 m** vs 2.45 m (15 cm) — still fails 1.5 cm, but is the only metric height. Occupancy areas 26.6 + 10.0 m² vs Polycam rooms; not a plaster floor plan.

Video: treat as fail except `walk_03` (1.86×3.52, ceiling 2.72).

## 4. Drift

Photo/video: no cross-folder poses. Correction is door snaps (and video sequential snaps). Ablation: native-frame AABB vs stitched AABB in `extra.stitch_ablation`.

LiDAR: global floor-plane rotation. Off = raw odometry XY; on = aligned XY. Residual: fused outliers still inflate AABB (~13×11 m vs ~8.8 m dining).

## 5. Incumbent

Polycam Space 2D plan (`captures/incumbent/polycam/IMG_0053.PNG`): dining long wall **8.78 m** vs tape **8.77 m**. We do **not** claim ≥70% beat on shared dimensions: Polycam bedroom labels 4.44 / 2.57 m look like furniture. Head-to-head table: `reports/benchmark.md`.

## 6. Fix loop

Worst gate: photo `room_01` 4.0 m wall 3.48 m (−13%). Hypothesis: 80th-percentile near-floor clip = furniture. Fix: clip **96** (`planes.py`). After: **4.07 × 3.62 m**, both inside ±8%. Ceiling still 4.10 m. Details: `reports/fix-loop.md`.

## 7. Known failures (walk-in)

- Non-Pro phone → LiDAR will not exist; run photos/video.
- Photo/video scale is a door prior; confident ±3% on video is **wrong** if tiles won.
- LiDAR rooms are occupancy blobs, not four walls; openings empty this submit.
- Damage, repeatability, kitchen tape: **not submitted**.
- Mirrors, glass, low light: protocol says avoid; we do not inpaint them.

## 8. Reproduction

Clean Mac, Python 3.12, `pip install -e .`, VGGT optional as in README. Raw inputs stay local (do not git the MOV or Stray depth). Commands:

```text
python -m cozmo run captures/ --out out/ --backend vggt
python -m cozmo run captures/ --out out/video --tier video --backend vggt
python -m cozmo run captures/ --out out/lidar --tier lidar
```

Fix-loop after: `python -m cozmo run captures/ --out out/fix-after --only room_01 --backend vggt`.
