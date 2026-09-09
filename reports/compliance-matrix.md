# Compliance matrix

Requirement → path → artifact → status. Items marked skip were out of this submit by instruction (repeat capture, kitchen tape, damage schema).

| Requirement | Path | Artifact | Status |
| --- | --- | --- | --- |
| Route 2 stock protocol | `docs/protocol.md` | one-page protocol | **done** |
| Device matrix | `docs/device-matrix.md` | hardware × tier | **done** |
| Photos 2–8 stills / room, stitched plan | `python -m cozmo run captures/` | `out/plan.json`, `out/plan.svg` | **runs**; adjacency from door graph; some walls fail ±8% |
| Video walkthrough | `--tier video` | `out/video/plan.json` | **runs**; walk_graph; scale often wrong |
| LiDAR depth+poses+intrinsics | `--tier lidar`, Stray under `captures/lidar/` | `out/lidar/plan.json` | **runs**; floor-plane ablation; occupancy rooms |
| CI on every measurement | `cozmo/schema.py` `Interval` | JSON | **done** |
| One command per capture | `README.md` | CLI | **done** |
| JSON + rendered plan | `cozmo/export/svg.py` | `plan.json` / `plan.svg` | **done** |
| Drift not poses-as-is | photo door snaps; video walk snaps; lidar floor align | `extra.stitch_ablation` | **done** (method exists; lidar AABB still loose) |
| Photo whole-property stitch | `cozmo/stitch/door_graph.py` | `stitch: door_graph` | **runs**; overlaps/layout not homeowner-clean |
| Opening width ≤2 cm / 85% | YOLO-World + 3D door | openings in JSON | **partial**; not scored to 2 cm |
| Ceiling ≤1.5 cm | LiDAR percentile | lidar ceiling 2.30 vs tape 2.45 | **fail** (15 cm); photo fail |
| Repeatability 1 cm / 0.5% | — | — | **skipped** (no second capture) |
| Damage + concealed flags + scope | — | — | **skipped** (not in schema this submit) |
| Head-to-head vs consumer app | Polycam 2D PNG | `captures/incumbent/polycam/IMG_0053.PNG`, `reports/benchmark.md` | **table exists**; ≥70% beat **not claimed** |
| Fix loop | `reports/fix-loop.md` + `planes.py` | before/after JSON; clip 80→96 | **pass on that wall** (3.48→4.07 m); ceiling still fail |
| Process evidence | git history | photo → stitch → video → lidar commits | **done** |
| Pretrained disclosure | README | VGGT, YOLO-World | **done** |
| Kitchen tape / damage staging | — | — | **skipped** |
