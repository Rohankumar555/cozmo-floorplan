# Capture protocol (Route 2)

One page. At the defense, follow this literally. Stock apps only — no custom iOS build.

## Install (App Store)

| Role | App | Device |
| --- | --- | --- |
| Photos + video | **Camera** (built-in) | Any iPhone 15 or newer, or iPad |
| LiDAR (our pipeline) | **Stray Scanner** (free) | **Pro only** (iPhone 12 Pro+ / iPad Pro 2020+). LiDAR is not on a regular iPhone 15. |
| Incumbent comparison | **Polycam** — Space or Floorplan, floor-plan toggle **on** | Same Pro device |

Do **not** upload a Camera walkthrough to Polycam Web and expect a floor plan. That path fails (blur, overlap). Scan live in Polycam Space.

## Photos (2–8 stills per room)

1. Create one album/folder per room: `room_01`, `room_02`, `room_03`, `hallway` (connector / dining+living).
2. Stand in each corner, then one or two mid-wall shots. Include the **door leaf** full-frame once (we scale from 0.80 m).
3. JPEG, no Portrait-mode depth. Lights on. Do not shoot mirrors head-on.
4. AirDrop into `captures/photos/<folder>/`.

## Video (one handheld walk)

1. Camera app, 1080p, **portrait or landscape** (we rotate portrait 90° CW).
2. Walk room to room. **Pause still ≥2 s in each doorway** (door hold). Then keep walking.
3. ~2–4 min is enough. One file: `captures/video/*.mov`.

## LiDAR (Stray)

1. Open Stray Scanner. One **continuous** walk of the whole flat is enough (or one scan per room).
2. Landscape, chest height, slow. Point at **walls**, tilt to ceiling and floor. Pause ~2 s in doorways.
3. Stop → share the dataset folder (`rgb.mp4`, `depth/`, `odometry.csv`, `camera_matrix.csv`).
4. AirDrop to `captures/lidar/property/` (hash subfolder from Stray is fine).

## Polycam (competitor only)

Space/Floorplan on the Pro. Walk the same rooms. Export the **2D plan with dimensions** (PNG/PDF). Put it in `captures/incumbent/polycam/`. This is **not** our LiDAR output.

## Hand-off to the Mac

```text
captures/photos/<room>/*.jpg
captures/video/*.mov
captures/lidar/property/<stray_id>/
captures/incumbent/polycam/*.png
```

Then one command per tier (see README). Reconstruction is local; no our servers.

## Avoid

Fast walking, dark rooms, wet floors, starting mid-wall, exporting Polycam’s original video instead of a processed plan.
