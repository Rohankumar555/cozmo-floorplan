"""Per-room reconstruction. Same function is used for every photo folder."""

from __future__ import annotations

import numpy as np

from cozmo.ingest.photos import PhotoRoom
from cozmo.reconstruct.openings import Detection, detect_openings
from cozmo.reconstruct.planes import (
    floor_polygon_from_planes,
    polygon_area,
    sequential_planes,
    walls_from_polygon,
)
from cozmo.reconstruct.scale import PHOTO_AREA_REL, PHOTO_CEILING_REL, Scale, estimate_scale
from cozmo.reconstruct.sparse3d import reconstruct_sparse
from cozmo.schema import Interval, Opening, Point2, RoomPlan, Wall


def reconstruct_room(room: PhotoRoom, backend: str = "auto") -> RoomPlan:
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
    door_widths_u = [o[3] for o in openings_u if o[0] == "door"]
    median_door_u = float(np.median(door_widths_u)) if door_widths_u else None
    scale = estimate_scale(dets, median_door_u)
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
        if kind == "door" and scale.method == "door_width_prior":
            method = "door_width_prior_applied"
            rel = 0.05
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
        ceiling_height=Interval.measured(ceiling_m, max(scale.rel_error, PHOTO_CEILING_REL), scale.method),
        floor_area=Interval.measured(area_m, max(scale.rel_error, PHOTO_AREA_REL), scale.method, unit="m2"),
        notes=notes,
        backend=scene.backend,
    )


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
        # Prefer longer walls for doors
        if d.kind == "door":
            wall_i = int(np.argmax(wall_lens)) if wall_lens[wall_i] < np.median(wall_lens) else wall_i
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
