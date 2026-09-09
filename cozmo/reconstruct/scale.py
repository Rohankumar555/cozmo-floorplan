"""Metric scale for photo tier: real door in 3D, else 0.80×1.20 m floor tiles.

Never scale from a hypothesized 'door = 12% of wall' (that locks a wall at 6.67 m).
"""

from __future__ import annotations

from dataclasses import dataclass

from cozmo.reconstruct.openings import Detection
from cozmo.reconstruct.tiles import spacings_to_metres_per_unit

DOOR_WIDTH_M = 0.80
DOOR_WIDTH_REL = 0.12

PHOTO_WALL_REL = 0.08
PHOTO_AREA_REL = 0.16
PHOTO_CEILING_REL = 0.12

VIDEO_WALL_REL = 0.03
VIDEO_AREA_REL = 0.06
VIDEO_CEILING_REL = 0.08

LIDAR_WALL_REL = 0.02
LIDAR_AREA_REL = 0.04
LIDAR_CEILING_REL = 0.02


@dataclass
class Scale:
    metres_per_unit: float
    method: str
    rel_error: float


def estimate_scale(
    dets: list[Detection],
    real_door_width_u: float | None = None,
    tile_spacings_u: list[float] | None = None,
    rel_error: float | None = None,
) -> Scale:
    """Order: 3D door width, tile grout, else leave scene units (wide CI)."""
    del dets  # detections never imply a 12% wall door
    if real_door_width_u and real_door_width_u > 1e-6:
        scale = Scale(
            metres_per_unit=DOOR_WIDTH_M / real_door_width_u,
            method="door_width_3d",
            rel_error=max(PHOTO_WALL_REL, DOOR_WIDTH_REL),
        )
    else:
        tile = spacings_to_metres_per_unit(tile_spacings_u or [])
        if tile is not None:
            mpu, method = tile
            scale = Scale(metres_per_unit=mpu, method=method, rel_error=0.10)
        else:
            scale = Scale(metres_per_unit=1.0, method="unscaled_scene_units", rel_error=0.50)
    if rel_error is not None and scale.method != "unscaled_scene_units":
        return Scale(scale.metres_per_unit, scale.method, rel_error)
    return scale
