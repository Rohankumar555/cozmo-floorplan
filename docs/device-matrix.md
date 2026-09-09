# Device matrix

| Device | Photos | Video | LiDAR (Stray depth+poses) | Polycam 2D floorplan |
| --- | --- | --- | --- | --- |
| iPhone 15 / 16 (non-Pro) | yes | yes | **no** | **no** |
| iPhone 12 Pro or newer Pro/Max | yes | yes | yes | yes |
| iPad Pro (2020 or later) | yes | yes | yes | yes |
| iPad Air / mini / non-Pro | photos/video only | same | no | no |
| Mac (Apple silicon) | reconstruct | reconstruct | ingest only | view/export |

**Honest accuracy we will defend**

| Tier | What actually ran | Walls | Ceiling | Notes |
| --- | --- | --- | --- | --- |
| Photos | VGGT + 0.80 m door | ±8% intervals; hallway inside tape ±8%; `room_01` 4.0 m wall **failed** at −13% before the hull fix | metres off vs 2.45 m tape | Not LiDAR-metric |
| Video | VGGT on door-hold stills | ±3% **around a wrong scale** (tiles). Do not treat as ±3% true | 7–22 m except one segment | Walk stitch only |
| LiDAR | Stray unproject + floor align | occupancy blobs, not 4-wall plaster; ceiling **2.30 m** vs tape 2.45 m | best height we have | Metric millimetres |

Walk-in: if they hand a non-Pro, run **photos or video**. If they hand a Pro, run **`--tier lidar`**.
