# Benchmark notes

Tape: Rohan. Photos: `out/plan.json` (VGGT, door scale, **before** hull fix unless noted). Video: `out/video/plan.json`. LiDAR: `out/lidar/plan.json`. Incumbent: Polycam 2D `captures/incumbent/polycam/IMG_0053.PNG`.

Repeat capture, kitchen tape, and damage classes were **not** collected this submit.

## Room map

| Cozmo folder | Polycam label | Tape GT | Polycam area |
| --- | --- | --- | --- |
| hallway | Dining 28 m² + Living 11 m² | 4.35 × 8.77 m | 39 m² combined |
| room_01 | Bedroom 1 | 3.4 × 4.0 m, ceiling 2.45 m | 16 m² |
| room_02 | Bedroom 2 | 3.2 × 4.0 m | 12 m² |
| room_03 | Kitchen | none | 11 m² |
| — | Whole plan | — | 86 m² |

## Head-to-head: tape vs Polycam vs cozmo photo (before hull fix)

| Room | Axis | Tape (m) | Polycam (m) | Poly err | Cozmo photo (m) | Photo err | Photo ±8% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hallway | long | 8.77 | 8.78 | +0.01 (0.1%) | 8.47 | −3.4% | pass |
| hallway | short | 4.35 | 4.64 | +0.29 (6.7%) | 4.22 | −3.0% | pass |
| room_01 | 4.0 side | 4.00 | 4.00 | 0.0% | 3.48 | −13.0% | **fail** |
| room_01 | 3.4 side | 3.40 | 4.44 | +30.6% | 3.25 | −4.4% | pass |
| room_02 | 3.2 side | 3.20 | 3.32 | +3.8% | 3.06 | −4.4% | pass |
| room_02 | 4.0 side | 4.00 | 2.57 | −35.8% | 4.63 | +15.8% | **fail** |

Polycam ties tape on the dining long wall. We **do not claim ≥70% beat**: two Polycam bedroom sides look like furniture, and we lack plaster-to-plaster LiDAR walls. App: Polycam Space/Floorplan on iPad Pro; export is the 2D PNG (free path).

## LiDAR (our pipeline)

Ceiling **2.30 m** vs tape 2.45 m. Rooms after hold-cuts: `lidar_01` 26.6 m², `lidar_02` 10.0 m² (occupancy, not 4-wall). Plane-align ablation in `out/lidar/plan.json` `extra.stitch_ablation`.

## Video

Walk-graph ran (6 segments). Scale mostly `tile_0.80m`; walls 8–12 m. Only `walk_03` (1.86 × 3.52 m, ceiling 2.72 m) is near tape. **Do not put video in the accuracy table** except as a fail.

## Photo ceilings

`room_01` 4.10 m vs 2.45 m tape. Fail 1.5 cm by metres.

## Timing (this machine, Apple silicon)

| Tier | Order of runtime |
| --- | --- |
| Photos, 4 folders, VGGT | minutes |
| Video, 6 segments, VGGT | ~1 min after frame dump |
| LiDAR, 202 fused frames | ~1–2 s |

## Fix loop

See `reports/fix-loop.md`. **After (clip 80→96):** `room_01` 4.07 × 3.62 m vs tape 4.00 × 3.40 (**both inside ±8%**). Ceiling still 4.10 m. Files: `reports/fix-loop/before_room_01.json`, `after_room_01.json`.
