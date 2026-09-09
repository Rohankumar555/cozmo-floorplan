"""Per-room reconstruction. Same function is used for every photo folder."""

from __future__ import annotations

import numpy as np

from cozmo.ingest.photos import PhotoRoom
from cozmo.reconstruct.openings import Detection, detect_openings, door_widths_scene_units
from cozmo.reconstruct.planes import (
    floor_polygon_from_planes,
    polygon_area,
    sequential_planes,
    walls_from_polygon,
)
from cozmo.reconstruct.scale import PHOTO_AREA_REL, PHOTO_CEILING_REL, Scale, estimate_scale
from cozmo.reconstruct.sparse3d import reconstruct_sparse
from cozmo.reconstruct.tiles import tile_spacings_scene_units
from cozmo.schema import Interval, Opening, Point2, RoomPlan, Wall


def reconstruct_room(
    room: PhotoRoom,
    backend: str = "auto",
    *,
    wall_rel: float | None = None,
    area_rel: float | None = None,
    ceiling_rel: float | None = None,
) -> RoomPlan:
    area_rel = PHOTO_AREA_REL if area_rel is None else area_rel
    ceiling_rel = PHOTO_CEILING_REL if ceiling_rel is None else ceiling_rel
    scale_rel = wall_rel  # None keeps photo door CI; video passes VIDEO_WALL_REL
    if not (2 <= len(room.images) <= 8):
        notes = [
            f"expected 2–8 stills, found {len(room.images)}",
        ]
        if not room.images:
            return _empty_room(room, notes + ["no images"])
    else:
        notes = []

    scene = reconstruct_sparse(list(room.images), backend=backend)
    notes.extend(scene.notes)

    planes = sequential_planes(scene.points)
    try:
        poly, ceiling_u, _up = floor_polygon_from_planes(planes, scene.points)
    except ValueError as exc:
        notes.append(f"plane_layout_failed: {exc}")
        poly = _axis_aligned_from_points(scene.points)
        ceiling_u = 0.6

    if len(poly) < 3:
        poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        notes.append("polygon_degenerate_used_unit_square")

    dets: list[Detection] = []
    try:
        dets = detect_openings(list(room.images))
    except Exception as exc:  # noqa: BLE001
        notes.append(f"opening_detector_failed: {type(exc).__name__}: {exc}")

    segs = walls_from_polygon(poly)
    openings_u = _place_openings(poly, segs, dets, scene)
    real_doors_u = door_widths_scene_units(dets, list(room.images), scene)
    # YOLO boxes are fat (frame + wall). Tightest jamb-to-jamb is the 0.80 m leaf.
    door_u = float(min(real_doors_u)) if real_doors_u else None
    tile_u = tile_spacings_scene_units(list(room.images), scene)
    scale = estimate_scale(
        dets, real_door_width_u=door_u, tile_spacings_u=tile_u, rel_error=scale_rel
    )
    if scale.method == "door_width_3d" and not _plausible_metric_room(poly, scale.metres_per_unit):
        notes.append("door_scale_implausible_falling_back_to_tiles")
        scale = estimate_scale(dets, real_door_width_u=None, tile_spacings_u=tile_u)
    if scale.method == "tile_0.80m" and tile_u and not _plausible_metric_room(poly, scale.metres_per_unit):
        alt = Scale(1.20 / tile_u[0], "tile_1.20m", 0.10)
        if _plausible_metric_room(poly, alt.metres_per_unit):
            notes.append("tile_period_assigned_1.20m")
            scale = alt
    if scale.method.startswith("tile") and not _plausible_metric_room(poly, scale.metres_per_unit):
        notes.append("tile_scale_implausible_unscaled")
        scale = Scale(1.0, "unscaled_scene_units", 0.50)
    notes.append(f"scale={scale.method} mpu={scale.metres_per_unit:.5f}")
    if real_doors_u:
        notes.append(f"door_widths_u={[round(w, 4) for w in real_doors_u]}")
    if tile_u:
        notes.append(f"tile_spacings_u={[round(s, 4) for s in tile_u]}")
    if scene.backend != "vggt":
        scale = Scale(scale.metres_per_unit, scale.method, max(scale.rel_error, 0.35))

    poly_m = poly * scale.metres_per_unit
    ceiling_m = ceiling_u * scale.metres_per_unit
    area_m = polygon_area(poly_m)

    walls: list[Wall] = []
    for i, (a, b) in enumerate(walls_from_polygon(poly_m)):
        length = float(np.linalg.norm(b - a))
        walls.append(
            Wall(
                id=f"{room.room_id}-W{i}",
                start=Point2(x=float(a[0]), y=float(a[1])),
                end=Point2(x=float(b[0]), y=float(b[1])),
                length=Interval.measured(length, scale.rel_error, scale.method),
            )
        )

    openings: list[Opening] = []
    for k, (kind, wall_i, t0, width_u, evidence) in enumerate(openings_u):
        wall_len = walls[wall_i].length.value if wall_i < len(walls) else 1.0
        width_m = width_u * scale.metres_per_unit
        # After door-width scale, the scaling door is tautological — say so.
        method = scale.method
        rel = scale.rel_error
        if kind == "door" and scale.method == "door_width_3d":
            method = "door_width_3d_applied"
            rel = 0.05
        if "hypothesized" in evidence:
            method = "hypothesized_not_used_for_scale"
            rel = max(rel, 0.25)
        t1 = min(1.0, t0 + (width_m / wall_len if wall_len else 0.1))
        openings.append(
            Opening(
                id=f"{room.room_id}-O{k}",
                kind=kind,  # type: ignore[arg-type]
                wall_id=walls[wall_i].id if wall_i < len(walls) else walls[0].id,
                width=Interval.measured(width_m, rel, method),
                t_start=float(t0),
                t_end=float(t1),
                evidence=evidence,
            )
        )

    notes.append(f"images={len(room.images)}")
    notes.append(f"detections={len(dets)}")
    return RoomPlan(
        id=room.room_id,
        source_folder=room.folder.name,
        polygon=[Point2(x=float(p[0]), y=float(p[1])) for p in poly_m],
        walls=walls,
        openings=openings,
        ceiling_height=Interval.measured(ceiling_m, max(scale.rel_error, ceiling_rel), scale.method),
        floor_area=Interval.measured(area_m, max(scale.rel_error, area_rel), scale.method, unit="m2"),
        notes=notes,
        backend=scene.backend,
    )


def _plausible_metric_room(poly: np.ndarray, mpu: float) -> bool:
    """Photo-tier rooms are metres, not 6.67 m locked walls or centimetre boxes."""
    segs = walls_from_polygon(poly * mpu)
    if len(segs) < 2:
        return False
    lens = [float(np.linalg.norm(b - a)) for a, b in segs]
    return min(lens) >= 1.8 and max(lens) <= 14.0


def _empty_room(room: PhotoRoom, notes: list[str]) -> RoomPlan:
    dummy = Interval.measured(0.0, 0.5, "empty")
    return RoomPlan(
        id=room.room_id,
        source_folder=room.folder.name,
        polygon=[],
        walls=[],
        openings=[],
        ceiling_height=dummy,
        floor_area=Interval.measured(0.0, 0.5, "empty", unit="m2"),
        notes=notes,
        backend="none",
    )


def _axis_aligned_from_points(pts: np.ndarray) -> np.ndarray:
    # Drop the gravity-ish axis (largest variance of normals handled upstream; here min-extent axis)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    spans = maxs - mins
    drop = int(np.argmin(spans))
    keep = [i for i in range(3) if i != drop]
    x0, y0 = mins[keep[0]], mins[keep[1]]
    x1, y1 = maxs[keep[0]], maxs[keep[1]]
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def _place_openings(
    poly: np.ndarray,
    segs: list[tuple[np.ndarray, np.ndarray]],
    dets: list[Detection],
    scene,
) -> list[tuple[str, int, float, float, str]]:
    """Returns (kind, wall_index, t_start, width_scene_units, evidence)."""
    if not segs:
        return []
    wall_lens = [float(np.linalg.norm(b - a)) + 1e-9 for a, b in segs]
    placed: list[tuple[str, int, float, float, str]] = []

    # Map each image to a wall (mod n). Pointmap snap lands in a later commit.
    images = []
    for d in dets:
        if d.image not in images:
            images.append(d.image)

    used_on_wall: dict[int, float] = {i: 0.12 for i in range(len(segs))}
    for d in dets:
        img_i = images.index(d.image) if d.image in images else 0
        wall_i = img_i % len(segs)
        t = used_on_wall.get(wall_i, 0.12)
        x0, y0, x1, y1 = d.xyxy
        box_w, box_h = max(1.0, x1 - x0), max(1.0, y1 - y0)
        # Scene width: fraction of this wall, clamped. Scale stage turns it into metres.
        frac = min(0.35, max(0.08, box_w / (box_h * 2.5)))
        width_u = frac * wall_lens[wall_i]
        if t + frac > 0.9:
            t = 0.08
        used_on_wall[wall_i] = t + frac + 0.05
        evidence = f"{d.kind} conf={d.conf:.2f} {d.image.name}"
        placed.append((d.kind, wall_i, t, width_u, evidence))

    if not placed:
        # Still emit one hypothesized door so scale has a handle; flagged in evidence.
        wall_i = int(np.argmax(wall_lens))
        width_u = 0.12 * wall_lens[wall_i]
        placed.append(("door", wall_i, 0.15, width_u, "hypothesized_door_no_detection"))
    return placed
