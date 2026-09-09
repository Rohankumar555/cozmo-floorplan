import numpy as np

from cozmo.schema import Interval, Opening, Point2, RoomPlan, Wall
from cozmo.stitch.door_graph import polygon_overlap_area, stitch_rooms


def _box(room_id: str, x0: float, y0: float, x1: float, y1: float, door_wall: str, t0: float, t1: float) -> RoomPlan:
    """Axis-aligned room with one door. door_wall in {n,s,e,w}."""
    poly = [
        Point2(x=x0, y=y0),
        Point2(x=x1, y=y0),
        Point2(x=x1, y=y1),
        Point2(x=x0, y=y1),
    ]
    walls = [
        Wall(id=f"{room_id}-W0", start=poly[0], end=poly[1], length=Interval.measured(x1 - x0, 0.08, "test")),
        Wall(id=f"{room_id}-W1", start=poly[1], end=poly[2], length=Interval.measured(y1 - y0, 0.08, "test")),
        Wall(id=f"{room_id}-W2", start=poly[2], end=poly[3], length=Interval.measured(x1 - x0, 0.08, "test")),
        Wall(id=f"{room_id}-W3", start=poly[3], end=poly[0], length=Interval.measured(y1 - y0, 0.08, "test")),
    ]
    wall_id = {
        "s": f"{room_id}-W0",
        "e": f"{room_id}-W1",
        "n": f"{room_id}-W2",
        "w": f"{room_id}-W3",
    }[door_wall]
    opening = Opening(
        id=f"{room_id}-O0",
        kind="door",
        wall_id=wall_id,
        width=Interval.measured(0.8, 0.1, "test"),
        t_start=t0,
        t_end=t1,
        evidence="door conf=0.9 test.jpg",
    )
    area = (x1 - x0) * (y1 - y0)
    return RoomPlan(
        id=room_id,
        source_folder=room_id,
        polygon=poly,
        walls=walls,
        openings=[opening],
        ceiling_height=Interval.measured(2.45, 0.1, "test"),
        floor_area=Interval.measured(area, 0.1, "test", unit="m2"),
        backend="test",
    )


def _opening_mid(room: RoomPlan, oid: str) -> np.ndarray:
    op = next(o for o in room.openings if o.id == oid)
    wall = next(w for w in room.walls if w.id == op.wall_id)
    a = np.array([wall.start.x, wall.start.y])
    b = np.array([wall.end.x, wall.end.y])
    p0 = a + op.t_start * (b - a)
    p1 = a + op.t_end * (b - a)
    return 0.5 * (p0 + p1)


def test_two_rooms_snap_at_doors_without_overlap():
    hub = _box("hallway", 0.0, 0.0, 8.0, 4.0, "e", 0.4, 0.5)
    sat = _box("room_01", 0.0, 0.0, 4.0, 3.2, "w", 0.3, 0.45)
    out = stitch_rooms([hub, sat])
    assert len(out.adjacency) == 1
    assert out.adjacency[0].room_a == "hallway"
    assert out.adjacency[0].room_b == "room_01"
    placed = {r.id: r for r in out.rooms}
    hmid = _opening_mid(placed["hallway"], "hallway-O0")
    smid = _opening_mid(placed["room_01"], "room_01-O0")
    assert float(np.linalg.norm(hmid - smid)) < 1e-6
    hc = np.mean([[p.x, p.y] for p in placed["hallway"].polygon], axis=0)
    sc = np.mean([[p.x, p.y] for p in placed["room_01"].polygon], axis=0)
    # Bedroom should sit east of the hub, not on top of it.
    assert sc[0] > hc[0]
    assert abs(sc[0] - hc[0]) > 2.0


def test_single_room_skips_stitch():
    hub = _box("hallway", 0.0, 0.0, 8.0, 4.0, "e", 0.4, 0.5)
    out = stitch_rooms([hub])
    assert out.adjacency == []
    assert "stitch_skipped_lt2_rooms" in out.notes


def test_three_rooms_follow_doors_without_overlap():
    """Doors all on the hub's east wall → rooms go there. No name/side/short-face priors."""
    hub = _box("hallway", 0.0, 0.0, 8.0, 4.35, "e", 0.4, 0.5)
    for k, (t0, t1) in enumerate(((0.1, 0.2), (0.55, 0.65), (0.75, 0.85)), start=1):
        hub.openings.append(
            Opening(
                id=f"hallway-O{k}",
                kind="door",
                wall_id="hallway-W1",
                width=Interval.measured(0.8, 0.1, "test"),
                t_start=t0,
                t_end=t1,
                evidence=f"door conf=0.4 phantom{k}.jpg",
            )
        )
    r1 = _box("room_01", 0.0, 0.0, 4.0, 3.4, "w", 0.3, 0.5)
    r2 = _box("room_02", 0.0, 0.0, 4.0, 3.2, "w", 0.3, 0.5)
    r3 = _box("room_03", 0.0, 0.0, 3.7, 1.3, "w", 0.3, 0.5)
    out = stitch_rooms([hub, r1, r2, r3])
    assert len(out.adjacency) == 3
    placed = {r.id: r for r in out.rooms}
    sats = [placed["room_01"], placed["room_02"], placed["room_03"]]
    hc = np.mean([[p.x, p.y] for p in placed["hallway"].polygon], axis=0)
    for i, a in enumerate(sats):
        for b in sats[i + 1 :]:
            assert polygon_overlap_area(a, b) < 0.08
        assert polygon_overlap_area(placed["hallway"], a) < 0.30
        sc = np.mean([[p.x, p.y] for p in a.polygon], axis=0)
        assert sc[0] > hc[0]
    assert out.ablation["overlap_clear"] is True


def test_shared_wall_is_the_detected_door_wall():
    """Door on the long south face → that long face meets the hub, not the short face."""
    hub = _box("hallway", 0.0, 0.0, 8.0, 4.35, "e", 0.4, 0.5)
    sat = _box("room_02", 0.0, 0.0, 4.0, 3.2, "s", 0.3, 0.5)
    out = stitch_rooms([hub, sat])
    placed = {r.id: r for r in out.rooms}
    xs = [p.x for p in placed["room_02"].polygon]
    ys = [p.y for p in placed["room_02"].polygon]
    assert abs((max(ys) - min(ys)) - 4.0) < 0.15
    assert abs((max(xs) - min(xs)) - 3.2) < 0.15
