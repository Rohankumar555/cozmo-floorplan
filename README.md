# cozmo-floorplan

Local pipeline for the Cozmo case study: handheld iPhone capture in, one stitched dimensioned floor plan plus JSON out.

Capture is Route 2 (stock iOS Camera + a LiDAR logging app). This repo is the reconstruction CLI, not an iPhone app.

## Capture layout

```text
captures/
  photos/<room_name>/*.jpg   # 2–8 stills per room, no depth, no poses
  video/                     # one walkthrough clip (later)
  lidar/                     # depth + poses + intrinsics export (later)
```

One command (to be added as the pipeline is built):

```bash
python -m cozmo run captures/ --out out/
```

## Status

Repo layout and capture contract only. Reconstruction, JSON schema export, and the rendered plan are not implemented yet.
