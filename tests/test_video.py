import numpy as np

from cozmo.ingest.segment import holds_from_motion, segments_from_holds
from cozmo.schema import Interval, Opening, Point2, RoomPlan, Wall
from cozmo.stitch.walk import stitch_walk


def test_door_holds_need_walk_on_both_sides():
    t = np.linspace(0, 24, 96)
    motion = np.full_like(t, 12.0)
    motion[t < 1.5] = 0.3  # standing at start — not a door
    motion[(t >= 6) & (t <= 8.2)] = 0.4  # hold
    motion[(t >= 14) & (t <= 16.1)] = 0.4  # hold
    motion[t > 22] = 0.3  # standing at end — not a door
    holds = holds_from_motion(t, motion, min_hold_s=1.2)
    assert len(holds) == 2
    segs = segments_from_holds(24.0, holds, min_seg_s=3.0)
    assert len(segs) == 3
    assert segs[0].id == "walk_00"
    assert segs[-1].t_end == 24.0


def test_end_of_clip_pause_is_not_a_door():
    t = np.linspace(0, 20, 80)
    motion = np.full_like(t, 12.0)
    motion[(t >= 6) & (t <= 8)] = 0.4
    motion[t > 17.5] = 0.3  # stop recording
    holds = holds_from_motion(t, motion, min_hold_s=1.2)
    assert len(holds) == 1
    assert holds[0].t_start < 10


def test_no_holds_is_one_segment():
    segs = segments_from_holds(10.0, [], min_seg_s=3.0)
    assert len(segs) == 1
    assert segs[0].t_start == 0.0
    assert segs[0].t_end == 10.0


def _box(room_id: str, x0: float, y0: float, x1: float, y1: float, door_wall: str) -> RoomPlan:
    poly = [
        Point2(x=x0, y=y0),
        Point2(x=x1, y=y0),
        Point2(x=x1, y=y1),
        Point2(x=x0, y=y1),
    ]
    walls = [
        Wall(id=f"{room_id}-W0", start=poly[0], end=poly[1], length=Interval.measured(x1 - x0, 0.03, "test")),
        Wall(id=f"{room_id}-W1", start=poly[1], end=poly[2], length=Interval.measured(y1 - y0, 0.03, "test")),
        Wall(id=f"{room_id}-W2", start=poly[2], end=poly[3], length=Interval.measured(x1 - x0, 0.03, "test")),
        Wall(id=f"{room_id}-W3", start=poly[3], end=poly[0], length=Interval.measured(y1 - y0, 0.03, "test")),
    ]
    wall_id = {"s": f"{room_id}-W0", "e": f"{room_id}-W1", "n": f"{room_id}-W2", "w": f"{room_id}-W3"}[door_wall]
    opening = Opening(
        id=f"{room_id}-O0",
        kind="door",
        wall_id=wall_id,
        width=Interval.measured(0.8, 0.1, "test"),
        t_start=0.4,
        t_end=0.5,
        evidence="door conf=0.9 walk.jpg",
    )
    return RoomPlan(
        id=room_id,
        source_folder=room_id,
        polygon=poly,
        walls=walls,
        openings=[opening],
        ceiling_height=Interval.measured(2.45, 0.08, "test"),
        floor_area=Interval.measured((x1 - x0) * (y1 - y0), 0.06, "test", unit="m2"),
        backend="test",
    )


def test_walk_stitch_is_a_chain_not_a_star():
    a = _box("walk_00", 0, 0, 4, 3, "e")
    b = _box("walk_01", 0, 0, 4, 3, "w")
    out = stitch_walk([a, b])
    assert len(out.adjacency) == 1
    assert out.adjacency[0].room_a == "walk_00"
    assert out.adjacency[0].room_b == "walk_01"
    placed = {r.id: r for r in out.rooms}
    ac = np.mean([[p.x, p.y] for p in placed["walk_00"].polygon], axis=0)
    bc = np.mean([[p.x, p.y] for p in placed["walk_01"].polygon], axis=0)
    assert bc[0] > ac[0]
    assert "walk_graph" in out.notes[0]


def test_walk_stitch_survives_missing_detector_door():
    a = _box("walk_00", 0, 0, 4, 3, "e")
    b = _box("walk_01", 0, 0, 4, 3, "w")
    b = b.model_copy(update={"openings": []})
    out = stitch_walk([a, b])
    assert len(out.adjacency) == 1
    assert any(o.id.endswith("-Owalk") for o in out.rooms[1].openings)
    assert any("walk_hold" in n for n in out.notes)
