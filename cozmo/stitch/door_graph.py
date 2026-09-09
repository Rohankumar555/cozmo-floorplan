"""Door-graph stitch from detector doors only. No house layout priors.

Each satellite snaps to an unused hub door (width match). If that lands on
another room, slide along the hub wall, then try the next hub door. Hub stays
in its frame. No 'same side', no 'room_02 middle', no 'short face to hallway'.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from cozmo.schema import Adjacency, Point2, RoomPlan, Wall

HUB_NAMES = ("hallway", "living", "connector")
PHOTO_FOOTPRINT_REL = 0.08
# Shared wall is a line (area 0). Anything above this is a real penetration.
HUB_OVERLAP_M2 = 0.30
SAT_OVERLAP_M2 = 0.08


@dataclass
class _Door:
    opening_id: str
    wall_id: str
    mid: np.ndarray
    inward: np.ndarray
    width: float
    hypothesized: bool


@dataclass
class StitchResult:
    rooms: list[RoomPlan]
    adjacency: list[Adjacency]
    notes: list[str]
    ablation: dict


def stitch_rooms(rooms: list[RoomPlan]) -> StitchResult:
    """Place satellites at detector doors. Hub stays put."""
    if len(rooms) < 2:
        return StitchResult(list(rooms), [], ["stitch_skipped_lt2_rooms"], _ablation(rooms, rooms))

    placed = [r.model_copy(deep=True) for r in rooms]
    hub_i = _hub_index(placed)
    hub = placed[hub_i]
    notes = [
        f"stitch_hub={hub.id}",
        "drift: door snaps; same-wall rooms packed along that wall, no poses",
        "layout: YOLO doors only, no house priors",
    ]
    adjacency: list[Adjacency] = []
    used_hub: set[str] = set()
    occupied: list[RoomPlan] = [hub]
    sat_idxs = sorted((i for i in range(len(placed)) if i != hub_i), key=lambda i: placed[i].id)

    for i in sat_idxs:
        sat = placed[i]
        s_door = _primary_door(sat)
        if s_door is None:
            notes.append(f"stitch_unattached={sat.id}")
            continue
        trial, h_door = _attach_without_overlap(sat, s_door, hub, occupied, used_hub)
        if trial is None or h_door is None:
            notes.append(f"stitch_unattached={sat.id}")
            continue
        placed[i] = trial
        occupied.append(trial)
        used_hub.add(h_door.opening_id)
        adjacency.append(
            Adjacency(
                room_a=hub.id,
                room_b=sat.id,
                opening_a=h_door.opening_id,
                opening_b=s_door.opening_id,
            )
        )
        notes.append(f"stitch_snap {sat.id}:{s_door.opening_id} -> {hub.id}:{h_door.opening_id} wall={h_door.wall_id}")

    pack_notes = _pack_same_wall_groups(placed, hub_i, adjacency)
    notes.extend(pack_notes)
    occupied = [placed[hub_i]] + [placed[i] for i in sat_idxs if any(a.room_b == placed[i].id for a in adjacency)]

    placed[hub_i] = hub
    overlap = _overlap_report(placed, hub.id)
    notes.append(f"max_sat_overlap_m2={overlap['max_sat_overlap_m2']}")
    notes.append(f"max_hub_overlap_m2={overlap['max_hub_overlap_m2']}")
    ablation = _ablation(rooms, placed)
    ablation.update(overlap)
    ablation["property_footprint"] = _footprint(placed)
    return StitchResult(placed, adjacency, notes, ablation)


def polygon_overlap_area(a: RoomPlan, b: RoomPlan) -> float:
    pa, pb = _xy(a), _xy(b)
    if len(pa) < 3 or len(pb) < 3:
        return 0.0
    inter = _clip_convex(pa, pb)
    return _shoelace(inter) if len(inter) >= 3 else 0.0


def _attach_without_overlap(
    sat: RoomPlan,
    s_door: _Door,
    hub: RoomPlan,
    occupied: list[RoomPlan],
    used_hub: set[str],
) -> tuple[RoomPlan | None, _Door | None]:
    """Try unused hub doors by width; slide on the wall if the snap overlaps."""
    best: tuple[float, RoomPlan, _Door] | None = None
    for h_door in _ranked_hub_doors(hub, s_door, used_hub):
        trial = _slide_along_wall(_snap_satellite(sat, s_door, h_door), h_door, [hub])
        pen = _placement_penalty(trial, occupied)
        if best is None or pen < best[0]:
            best = (pen, trial, h_door)
        if pen <= 1e-6:
            return trial, h_door
    if best is None:
        return None, None
    return best[1], best[2]


def _ranked_hub_doors(hub: RoomPlan, s_door: _Door, used: set[str]) -> list[_Door]:
    doors = [d for d in _real_doors(hub) if d.opening_id not in used]
    doors.sort(key=lambda d: (abs(d.width - s_door.width), -_opening_conf(hub, d.opening_id)))
    return doors


def _primary_door(room: RoomPlan) -> _Door | None:
    doors = _real_doors(room)
    if not doors:
        return None
    return max(doors, key=lambda d: (_opening_conf(room, d.opening_id), -abs(d.width - 0.80)))


def _pack_same_wall_groups(placed: list[RoomPlan], hub_i: int, adjacency: list[Adjacency]) -> list[str]:
    """If several rooms snapped to the same hub wall, sit them in a tight row.

    Order is the hub-door parameter along that wall (detector), not folder names.
    """
    if len(adjacency) < 2:
        return []
    hub = placed[hub_i]
    sat_i = {placed[i].id: i for i in range(len(placed)) if i != hub_i}
    by_wall: dict[str, list[tuple[float, int]]] = {}
    for adj in adjacency:
        op = next((o for o in hub.openings if o.id == adj.opening_a), None)
        idx = sat_i.get(adj.room_b)
        if op is None or idx is None:
            continue
        t = 0.5 * (float(op.t_start) + float(op.t_end))
        by_wall.setdefault(op.wall_id, []).append((t, idx))
    notes: list[str] = []
    for wall_id, items in by_wall.items():
        if len(items) < 2:
            continue
        wall = next((w for w in hub.walls if w.id == wall_id), None)
        if wall is None:
            continue
        items.sort(key=lambda it: it[0])
        row = [placed[i] for _t, i in items]
        packed = _pack_row_on_wall(wall, row)
        for (_t, i), sat in zip(items, packed):
            placed[i] = sat
        names = ",".join(placed[i].id for _t, i in items)
        notes.append(f"pack_wall={wall_id} order={names}")
    return notes


def _pack_row_on_wall(wall: Wall, sats: list[RoomPlan]) -> list[RoomPlan]:
    origin = np.array([wall.start.x, wall.start.y], dtype=float)
    vec = _wall_vec(wall)
    length = float(np.linalg.norm(vec))
    tang = vec / max(length, 1e-9)
    gap = 0.08
    spans: list[tuple[RoomPlan, float, float]] = []
    for sat in sats:
        ts = (_xy(sat) - origin) @ tang
        spans.append((sat, float(ts.min()), float(ts.max())))
    total = sum(hi - lo for _s, lo, hi in spans) + gap * (len(spans) - 1)
    cursor = 0.5 * (length - total)
    out: list[RoomPlan] = []
    for sat, lo, hi in spans:
        out.append(_translate(sat, tang * (cursor - lo)))
        cursor += (hi - lo) + gap
    return out


def _opening_conf(room: RoomPlan, oid: str) -> float:
    op = next((o for o in room.openings if o.id == oid), None)
    if op is None:
        return 0.0
    m = re.search(r"conf=([0-9.]+)", op.evidence or "")
    return float(m.group(1)) if m else 0.0


def _hub_index(rooms: list[RoomPlan]) -> int:
    ids = [r.id.lower() for r in rooms]
    folders = [r.source_folder.lower() for r in rooms]
    for name in HUB_NAMES:
        for i, (rid, folder) in enumerate(zip(ids, folders)):
            if rid == name or folder == name or name in rid:
                return i
    areas = [r.floor_area.value for r in rooms]
    return int(np.argmax(areas)) if areas else 0


_SKIP_EV = ("hypothesized", "synthetic", "packed_facade", "short_face")


def _detector_openings(room: RoomPlan):
    out = []
    for o in room.openings:
        if o.kind != "door":
            continue
        ev = (o.evidence or "").lower()
        if any(s in ev for s in _SKIP_EV):
            continue
        out.append(o)
    return out


def _real_doors(room: RoomPlan) -> list[_Door]:
    ops = _detector_openings(room)
    doors = [_door_geom(room, o) for o in ops]
    return [d for d in doors if d is not None]


def _door_geom(room: RoomPlan, opening) -> _Door | None:
    wall = next((w for w in room.walls if w.id == opening.wall_id), None)
    if wall is None or not room.polygon:
        return None
    a = np.array([wall.start.x, wall.start.y], dtype=float)
    b = np.array([wall.end.x, wall.end.y], dtype=float)
    p0 = a + float(opening.t_start) * (b - a)
    p1 = a + float(opening.t_end) * (b - a)
    mid = 0.5 * (p0 + p1)
    tang = b - a
    nrm = float(np.linalg.norm(tang))
    if nrm < 1e-9:
        return None
    inward = np.array([-tang[1], tang[0]], dtype=float) / nrm
    c = np.mean([[p.x, p.y] for p in room.polygon], axis=0)
    if float((c - mid) @ inward) < 0:
        inward = -inward
    hyp = "hypothesized" in (opening.evidence or "") or "synthetic_wall_slot" in (opening.evidence or "")
    return _Door(
        opening_id=opening.id,
        wall_id=wall.id,
        mid=mid,
        inward=inward,
        width=float(opening.width.value),
        hypothesized=hyp,
    )


def _snap_satellite(sat: RoomPlan, s_door: _Door, h_door: _Door) -> RoomPlan:
    """Rotate so satellite inward matches -hub inward, then match door midpoints."""
    target = -h_door.inward
    ang = float(np.arctan2(target[1], target[0]) - np.arctan2(s_door.inward[1], s_door.inward[0]))
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, -s], [s, c]])

    def xform(x: float, y: float) -> tuple[float, float]:
        v = rot @ (np.array([x, y]) - s_door.mid) + h_door.mid
        return float(v[0]), float(v[1])

    return _map_room(sat, xform)


def _slide_along_wall(sat: RoomPlan, h_door: _Door, occupied: list[RoomPlan]) -> RoomPlan:
    """Keep the door on the hub wall; nudge along it to clear other rooms."""
    tang = np.array([-h_door.inward[1], h_door.inward[0]], dtype=float)
    nrm = float(np.linalg.norm(tang))
    if nrm < 1e-9:
        return sat
    tang = tang / nrm
    best = sat
    best_pen = _placement_penalty(sat, occupied)
    for delta in np.linspace(-0.4, 0.4, 9):
        trial = _translate(sat, tang * float(delta))
        pen = _placement_penalty(trial, occupied)
        if pen < best_pen:
            best, best_pen = trial, pen
    return best


def _placement_penalty(sat: RoomPlan, occupied: list[RoomPlan]) -> float:
    pen = 0.0
    sxy = _xy(sat)
    for o in occupied:
        area = _overlap_area_xy(sxy, _xy(o))
        if o.id in HUB_NAMES or o.source_folder.lower() in HUB_NAMES:
            pen += 40.0 * max(0.0, area - HUB_OVERLAP_M2)
        else:
            pen += 80.0 * max(0.0, area - SAT_OVERLAP_M2)
            pen += 8.0 * area
    return pen


def _translate(room: RoomPlan, delta: np.ndarray) -> RoomPlan:
    dx, dy = float(delta[0]), float(delta[1])

    def xform(x: float, y: float) -> tuple[float, float]:
        return x + dx, y + dy

    return _map_room(room, xform)


def _map_room(sat: RoomPlan, xform) -> RoomPlan:
    out = sat.model_copy(deep=True)
    out.polygon = [Point2(x=xform(p.x, p.y)[0], y=xform(p.x, p.y)[1]) for p in sat.polygon]
    new_walls: list[Wall] = []
    for w in sat.walls:
        x0, y0 = xform(w.start.x, w.start.y)
        x1, y1 = xform(w.end.x, w.end.y)
        new_walls.append(
            Wall(
                id=w.id,
                start=Point2(x=x0, y=y0),
                end=Point2(x=x1, y=y1),
                length=w.length,
            )
        )
    out.walls = new_walls
    return out


def _centroid_outside(sat: RoomPlan, hub: RoomPlan) -> bool:
    c = np.mean(_xy(sat), axis=0)
    return not _point_in_convex(c, _xy(hub))


def _xy(room: RoomPlan) -> np.ndarray:
    return np.array([[p.x, p.y] for p in room.polygon], dtype=float)


def _wall_vec(wall: Wall) -> np.ndarray:
    return np.array([wall.end.x - wall.start.x, wall.end.y - wall.start.y], dtype=float)


def _overlap_area_xy(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return 0.0
    inter = _clip_convex(a, b)
    return _shoelace(inter) if len(inter) >= 3 else 0.0


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    if _signed_area(poly) < 0:
        return poly[::-1].copy()
    return poly


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _shoelace(poly: np.ndarray) -> float:
    return abs(_signed_area(poly))


def _clip_convex(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland–Hodgman, clipper assumed convex."""
    out = _ensure_ccw(np.asarray(subject, dtype=float))
    clip = _ensure_ccw(np.asarray(clipper, dtype=float))
    if len(out) < 3 or len(clip) < 3:
        return np.zeros((0, 2))
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        inp = out
        out = []
        if len(inp) == 0:
            break
        prev = inp[-1]
        prev_in = _left_of(a, b, prev)
        for cur in inp:
            cur_in = _left_of(a, b, cur)
            if cur_in:
                if not prev_in:
                    hit = _intersect(a, b, prev, cur)
                    if hit is not None:
                        out.append(hit)
                out.append(cur)
            elif prev_in:
                hit = _intersect(a, b, prev, cur)
                if hit is not None:
                    out.append(hit)
            prev, prev_in = cur, cur_in
        out = np.array(out, dtype=float) if out else np.zeros((0, 2))
    return out


def _left_of(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> bool:
    return float((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= -1e-9


def _intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray | None:
    r = b - a
    s = d - c
    den = float(r[0] * s[1] - r[1] * s[0])
    if abs(den) < 1e-12:
        return None
    t = float((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / den
    return a + t * r


def _point_in_convex(p: np.ndarray, poly: np.ndarray) -> bool:
    poly = _ensure_ccw(poly)
    for i in range(len(poly)):
        if not _left_of(poly[i], poly[(i + 1) % len(poly)], p):
            return False
    return True


def _overlap_report(rooms: list[RoomPlan], hub_id: str) -> dict:
    hub = next(r for r in rooms if r.id == hub_id)
    sats = [r for r in rooms if r.id != hub_id]
    sat_over = 0.0
    for i, a in enumerate(sats):
        for b in sats[i + 1 :]:
            sat_over = max(sat_over, polygon_overlap_area(a, b))
    hub_over = max((polygon_overlap_area(hub, s) for s in sats), default=0.0)
    return {
        "max_sat_overlap_m2": round(sat_over, 4),
        "max_hub_overlap_m2": round(hub_over, 4),
        "overlap_clear": sat_over <= SAT_OVERLAP_M2 and hub_over <= HUB_OVERLAP_M2,
    }


def _footprint(rooms: list[RoomPlan]) -> dict:
    w, h = _aabb_wh(rooms)
    rel = PHOTO_FOOTPRINT_REL
    return {
        "aabb_width_m": round(w, 3),
        "aabb_depth_m": round(h, 3),
        "aabb_area_m2": round(w * h, 3),
        "ci_rel": rel,
        "aabb_width_ci": [round(w * (1 - rel), 3), round(w * (1 + rel), 3)],
        "aabb_depth_ci": [round(h * (1 - rel), 3), round(h * (1 + rel), 3)],
        "method": "door_graph_aabb_photo_pm8pct",
    }


def _ablation(native: list[RoomPlan], stitched: list[RoomPlan]) -> dict:
    """Door-anchor on vs rooms left in native frames (no cross-folder poses)."""
    w0, h0 = _aabb_wh(native)
    w1, h1 = _aabb_wh(stitched)
    return {
        "method": "door_snap_overlap_slide",
        "off_native_aabb_m": [round(w0, 3), round(h0, 3)],
        "off_native_aabb_area_m2": round(w0 * h0, 3),
        "on_stitched_aabb_m": [round(w1, 3), round(h1, 3)],
        "on_stitched_aabb_area_m2": round(w1 * h1, 3),
        "note": "off = each room in its own frame (not a property). on = detector-door snaps.",
    }


def _aabb_wh(rooms: list[RoomPlan]) -> tuple[float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for r in rooms:
        for p in r.polygon:
            xs.append(p.x)
            ys.append(p.y)
    if not xs:
        return 0.0, 0.0
    return float(max(xs) - min(xs)), float(max(ys) - min(ys))
