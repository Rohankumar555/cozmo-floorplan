"""Metric scale for photo tier: door priors, wide calibrated intervals.

Photos have no poses/depth. We never claim LiDAR centimetres here.
"""

from __future__ import annotations

from dataclasses import dataclass

from cozmo.reconstruct.openings import Detection, bbox_aspect_is_door

# Typical Indian interior door (clear opening). Wide CI because this is a prior.
DOOR_WIDTH_M = 0.80
DOOR_WIDTH_REL = 0.12  # ±12% on the scale factor from width
DOOR_HEIGHT_M = 2.04
DOOR_HEIGHT_REL = 0.08

# Photo-tier wall lengths: assignment allows ±8% when calibrated.
PHOTO_WALL_REL = 0.08
PHOTO_AREA_REL = 0.16
PHOTO_CEILING_REL = 0.12


@dataclass
class Scale:
    metres_per_unit: float
    method: str
    rel_error: float


def estimate_scale(dets: list[Detection], door_width_units: float | None) -> Scale:
    """Prefer a 3D opening width in scene units; else door-height in pixels (last resort)."""
    if door_width_units and door_width_units > 1e-6:
        return Scale(
            metres_per_unit=DOOR_WIDTH_M / door_width_units,
            method="door_width_prior",
            rel_error=max(PHOTO_WALL_REL, DOOR_WIDTH_REL),
        )
    doors = [d for d in dets if d.kind == "door" and bbox_aspect_is_door(d)]
    if doors:
        # Pixel height as a stand-in for door height in a *frontal* view — very weak.
        heights = [d.xyxy[3] - d.xyxy[1] for d in doors]
        px = float(sorted(heights)[len(heights) // 2])
        # Treat the median door pixel height as DOOR_HEIGHT_M in a canonical 1000px-tall image
        # only to get *some* scale; rel_error is large.
        return Scale(
            metres_per_unit=DOOR_HEIGHT_M / px if px else 1.0,
            method="door_height_pixel_prior",
            rel_error=0.25,
        )
    return Scale(metres_per_unit=1.0, method="unscaled_scene_units", rel_error=0.50)
