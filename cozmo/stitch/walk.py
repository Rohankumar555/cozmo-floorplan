"""Walk-order stitch: consecutive segments share the door you walked through."""

from __future__ import annotations

from cozmo.reconstruct.scale import VIDEO_WALL_REL
from cozmo.schema import Adjacency, Interval, Opening, RoomPlan
from cozmo.stitch.door_graph import (
    SAT_OVERLAP_M2,
    StitchResult,
    _Door,
    _ablation,
    _door_geom,
    _footprint,
    _primary_door,
    _snap_satellite,
    polygon_overlap_area,
)


def stitch_walk(rooms: list[RoomPlan]) -> StitchResult:
    """Keep the first segment; snap each later one onto the previous at doors."""
    if len(rooms) < 2:
        return StitchResult(list(rooms), [], ["stitch_skipped_lt2_rooms"], _ablation(rooms, rooms))

    placed = [r.model_copy(deep=True) for r in rooms]
    notes = [
        "stitch=walk_graph",
        "drift: sequential door snaps along the walk, not poses-as-is",
        "adjacency: consecutive segments (door holds), not YOLO width pairing",
        "missing detector door: hypothesized walk_hold on shortest wall, not used for scale",
    ]
    adjacency: list[Adjacency] = []
    for i in range(1, len(placed)):
        placed[i - 1], a_door = _ensure_walk_door(placed[i - 1])
        placed[i], b_door = _ensure_walk_door(placed[i])
        if a_door is None or b_door is None:
            notes.append(f"stitch_unattached={placed[i].id}")
            continue
        if a_door.hypothesized:
            notes.append(f"walk_hold_door={placed[i - 1].id}:{a_door.opening_id}")
        if b_door.hypothesized:
            notes.append(f"walk_hold_door={placed[i].id}:{b_door.opening_id}")
        placed[i] = _snap_satellite(placed[i], b_door, a_door)
        adjacency.append(
            Adjacency(
                room_a=placed[i - 1].id,
                room_b=placed[i].id,
                opening_a=a_door.opening_id,
                opening_b=b_door.opening_id,
            )
        )
        notes.append(
            f"walk_snap {placed[i].id}:{b_door.opening_id} -> {placed[i - 1].id}:{a_door.opening_id}"
        )

    overlap = _chain_overlap(placed)
    notes.append(f"max_sat_overlap_m2={overlap['max_sat_overlap_m2']}")
    ablation = _ablation(rooms, placed)
    ablation.update(overlap)
    ablation["property_footprint"] = _footprint(
        placed, ci_rel=VIDEO_WALL_REL, method="walk_graph_aabb_video_pm3pct"
    )
    ablation["method"] = "walk_door_snaps_no_poses"
    ablation["note"] = "off = each room in its own frame. on = sequential walk-door snaps."
    return StitchResult(placed, adjacency, notes, ablation)


def _ensure_walk_door(room: RoomPlan) -> tuple[RoomPlan, _Door | None]:
    """Detector door if YOLO saw one; else a walk-hold slot so the chain does not break."""
    d = _primary_door(room)
    if d is not None:
        return room, d
    existing = next((o for o in room.openings if o.id.endswith("-Owalk")), None)
    if existing is not None:
        return room, _door_geom(room, existing)
    if not room.walls:
        return room, None
    wall = min(room.walls, key=lambda w: w.length.value)
    op = Opening(
        id=f"{room.id}-Owalk",
        kind="door",
        wall_id=wall.id,
        width=Interval.measured(0.80, 0.25, "hypothesized_walk_hold"),
        t_start=0.40,
        t_end=0.50,
        evidence="walk_hold hypothesized (detector missed door)",
    )
    room = room.model_copy(deep=True)
    room.openings = list(room.openings) + [op]
    return room, _door_geom(room, op)


def _chain_overlap(rooms: list[RoomPlan]) -> dict:
    sat_over = 0.0
    for i, a in enumerate(rooms):
        for b in rooms[i + 1 :]:
            sat_over = max(sat_over, polygon_overlap_area(a, b))
    return {
        "max_sat_overlap_m2": round(sat_over, 4),
        "overlap_clear": sat_over <= SAT_OVERLAP_M2,
    }
