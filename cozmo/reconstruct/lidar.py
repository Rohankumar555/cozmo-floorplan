"""LiDAR-tier reconstruction from a Stray RGB-D walk.

Scale is metric (depth in millimetres). Drift correction is a global floor-plane
alignment, not poses-as-is. Ablation compares footprint before/after that align.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cozmo.ingest.lidar import LidarDataset, depth_intrinsics, load_dataset
from cozmo.ingest.segment import holds_from_motion, segments_from_holds
from cozmo.reconstruct.planes import polygon_area, ransac_plane, walls_from_polygon
from cozmo.reconstruct.scale import LIDAR_AREA_REL, LIDAR_CEILING_REL, LIDAR_WALL_REL
from cozmo.schema import Adjacency, Interval, Point2, RoomPlan, Wall

CONF_MIN = 1
DEPTH_MAX_M = 8.0
PIXEL_STRIDE = 3
FRAME_STRIDE = 12
WALL_Z0 = 0.35
WALL_Z1 = 1.85


def reconstruct_lidar(ds: LidarDataset) -> tuple[list[RoomPlan], dict]:
    world, raw_xy, meta = _fuse_world(ds)
    aligned, R, floor_h, ceiling = _align_floor(world)
    xy_wall = aligned[(aligned[:, 2] > WALL_Z0) & (aligned[:, 2] < WALL_Z1)][:, :2]
    if len(xy_wall) < 200:
        xy_wall = aligned[:, :2]
    xy_wall = _clip_xy(xy_wall, 93)

    times = ds.timestamps
    t_rel = times - float(times[0]) if len(times) else times
    speed = _speed_series(ds)
    holds = holds_from_motion(t_rel[1:], speed, min_hold_s=1.2) if len(speed) > 8 else []
    duration = float(t_rel[-1]) if len(t_rel) else 0.0
    segs = segments_from_holds(duration, holds, min_seg_s=12.0)
    t0 = float(times[0]) if len(times) else 0.0

    rooms: list[RoomPlan] = []
    if len(segs) <= 1:
        poly = _occupancy_polygon(xy_wall)
        rooms.append(_room_from_poly("property", ds.path.name, poly, ceiling, ["single_walk_no_room_cuts"]))
    else:
        pose_xyz = ds.poses_wc[:, :3, 3]
        for seg in segs:
            a, b = t0 + seg.t_start, t0 + seg.t_end
            mask_t = (times >= a) & (times <= b)
            if int(mask_t.sum()) < 3:
                continue
            # Approximate: keep fused points whose nearest pose time is in-window.
            # Cheaper: crop by XY of those poses after the same R.
            mid = pose_xyz[mask_t]
            mid_h = np.c_[mid, np.ones(len(mid))]
            mid_a = (R @ mid_h.T).T[:, :3]
            mid_a[:, 2] -= floor_h
            c = mid_a[:, :2].mean(axis=0)
            rad = np.linalg.norm(mid_a[:, :2] - c, axis=1)
            reach = float(np.percentile(rad, 90)) + 0.8
            local = xy_wall[np.linalg.norm(xy_wall - c, axis=1) <= reach]
            if len(local) < 80:
                continue
            poly = _occupancy_polygon(local)
            rooms.append(
                _room_from_poly(
                    seg.id.replace("walk_", "lidar_"),
                    ds.path.name,
                    poly,
                    ceiling,
                    [f"t={seg.t_start:.1f}-{seg.t_end:.1f}s"],
                )
            )
        rooms = [r for r in rooms if r.floor_area.value >= 4.0]
        if not rooms:
            poly = _occupancy_polygon(xy_wall)
            rooms.append(_room_from_poly("property", ds.path.name, poly, ceiling, ["segment_fallback_whole"]))

    adjacency: list[Adjacency] = []
    for i in range(1, len(rooms)):
        adjacency.append(
            Adjacency(
                room_a=rooms[i - 1].id,
                room_b=rooms[i].id,
                opening_a=f"{rooms[i - 1].id}-walk",
                opening_b=f"{rooms[i].id}-walk",
            )
        )

    on_wh = _aabb(xy_wall)
    off_wh = _aabb(raw_xy)
    extra = {
        "lidar_dataset": str(ds.path),
        "frames_fused": meta["frames"],
        "points": int(len(world)),
        "holds": [{"t_start": round(h.t_start, 3), "t_end": round(h.t_end, 3)} for h in holds],
        "ceiling_m": round(ceiling, 3),
        "stitch_ablation": {
            "method": "plane_anchored_floor_align_not_poses_as_is",
            "off_native_aabb_m": [round(off_wh[0], 3), round(off_wh[1], 3)],
            "off_native_aabb_area_m2": round(off_wh[0] * off_wh[1], 3),
            "on_stitched_aabb_m": [round(on_wh[0], 3), round(on_wh[1], 3)],
            "on_stitched_aabb_area_m2": round(on_wh[0] * on_wh[1], 3),
            "note": "off = raw odometry XY (ARKit, poses as-is). on = rotate so RANSAC floor is z=0.",
            "property_footprint": {
                "aabb_width_m": round(on_wh[0], 3),
                "aabb_depth_m": round(on_wh[1], 3),
                "aabb_area_m2": round(on_wh[0] * on_wh[1], 3),
                "ci_rel": LIDAR_WALL_REL,
                "method": "lidar_plane_aligned_aabb",
            },
        },
        "stitch_notes": [
            "stitch=lidar_poses",
            "drift: floor-plane alignment, not poses-as-is",
            f"holds={len(holds)} rooms={len(rooms)}",
        ],
    }
    extra["property_footprint"] = extra["stitch_ablation"]["property_footprint"]
    extra["adjacency"] = adjacency
    extra["_meta"] = meta
    return rooms, extra


def reconstruct_lidar_dir(captures: Path) -> tuple[list[RoomPlan], dict]:
    from cozmo.ingest.lidar import find_lidar_dataset

    path = find_lidar_dataset(captures)
    ds = load_dataset(path)
    return reconstruct_lidar(ds)


def _fuse_world(ds: LidarDataset) -> tuple[np.ndarray, np.ndarray, dict]:
    Kd = depth_intrinsics(ds.K_rgb)
    fx, fy, cx, cy = float(Kd[0, 0]), float(Kd[1, 1]), float(Kd[0, 2]), float(Kd[1, 2])
    conf_by_stem = {p.stem: p for p in ds.conf_paths}
    chunks: list[np.ndarray] = []
    used = 0
    for i in range(0, len(ds.depth_paths), FRAME_STRIDE):
        dpath = ds.depth_paths[i]
        depth = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        z = depth.astype(np.float32) / 1000.0
        if dpath.stem in conf_by_stem:
            conf = cv2.imread(str(conf_by_stem[dpath.stem]), cv2.IMREAD_UNCHANGED)
            if conf is not None:
                z = np.where(conf >= CONF_MIN, z, 0.0)
        z[z > DEPTH_MAX_M] = 0.0
        z[z < 0.2] = 0.0
        h, w = z.shape
        us, vs = np.meshgrid(np.arange(0, w, PIXEL_STRIDE), np.arange(0, h, PIXEL_STRIDE))
        zz = z[vs, us]
        m = zz > 0
        if int(m.sum()) < 30:
            continue
        X = (us[m] - cx) * zz[m] / fx
        Y = (vs[m] - cy) * zz[m] / fy
        P = np.stack([X, Y, zz[m]], axis=1)
        T = ds.poses_wc[min(i, len(ds.poses_wc) - 1)]
        R, t = T[:3, :3], T[:3, 3]
        W = (R @ P.T).T + t
        chunks.append(W)
        used += 1
    if not chunks:
        raise RuntimeError("No LiDAR points fused")
    world = np.concatenate(chunks, axis=0)
    if len(world) > 450_000:
        rng = np.random.default_rng(0)
        world = world[rng.choice(len(world), 450_000, replace=False)]
    # Raw plan view: drop the smallest-span axis (usually ARKit Y).
    spans = world.max(axis=0) - world.min(axis=0)
    drop = int(np.argmin(spans))
    keep = [j for j in range(3) if j != drop]
    raw_xy = world[:, keep]
    return world, raw_xy, {"frames": used, "dropped_axis": drop}


def _align_floor(world: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    rng = np.random.default_rng(1)
    sample = world if len(world) < 80_000 else world[rng.choice(len(world), 80_000, replace=False)]
    plane = ransac_plane(sample, thresh=0.04, iters=250, min_inliers=400, rng=rng)
    if plane is None:
        up = np.array([0.0, 1.0, 0.0])
        offset = -float(np.median(world[:, 1]))
    else:
        up = plane.normal.copy()
        offset = plane.offset
    # Rotate `up` to +Z.
    z = np.array([0.0, 0.0, 1.0])
    up = up / (np.linalg.norm(up) + 1e-12)
    if float(up @ z) < 0:
        up = -up
        offset = -offset
    v = np.cross(up, z)
    c = float(up @ z)
    if np.linalg.norm(v) < 1e-8:
        R3 = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=float)
        R3 = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
    aligned = (R3 @ world.T).T
    floor_h = float(np.percentile(aligned[:, 2], 12))
    ceil_h = float(np.percentile(aligned[:, 2], 88))
    aligned = aligned.copy()
    aligned[:, 2] -= floor_h
    ceiling = float(max(1.8, min(4.0, ceil_h - floor_h)))
    R4 = np.eye(4)
    R4[:3, :3] = R3
    return aligned, R4, floor_h, ceiling


def _occupancy_polygon(xy: np.ndarray, cell: float = 0.07) -> np.ndarray:
    from cozmo.reconstruct.planes import _min_area_rect

    if len(xy) < 30:
        return _min_area_rect(xy)
    lo = xy.min(axis=0) - cell
    span = xy.max(axis=0) - lo + cell
    nx = max(8, int(np.ceil(span[0] / cell)))
    ny = max(8, int(np.ceil(span[1] / cell)))
    nx = min(nx, 400)
    ny = min(ny, 400)
    grid = np.zeros((ny, nx), np.uint8)
    ix = np.clip(((xy[:, 0] - lo[0]) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((xy[:, 1] - lo[1]) / cell).astype(int), 0, ny - 1)
    grid[iy, ix] = 255
    k = np.ones((5, 5), np.uint8)
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, k, iterations=2)
    grid = cv2.dilate(grid, k, iterations=1)
    cnts, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return _min_area_rect(xy)
    c = max(cnts, key=cv2.contourArea)
    eps = 0.015 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, max(eps, 2.0), True).reshape(-1, 2).astype(float)
    if len(approx) < 4:
        return _min_area_rect(xy)
    return approx * cell + lo


def _room_from_poly(room_id: str, source: str, poly: np.ndarray, ceiling: float, notes: list[str]) -> RoomPlan:
    segs = walls_from_polygon(poly)
    walls: list[Wall] = []
    for i, (a, b) in enumerate(segs):
        length = float(np.linalg.norm(b - a))
        walls.append(
            Wall(
                id=f"{room_id}-W{i}",
                start=Point2(x=float(a[0]), y=float(a[1])),
                end=Point2(x=float(b[0]), y=float(b[1])),
                length=Interval.measured(length, LIDAR_WALL_REL, "lidar_depth_mm"),
            )
        )
    area = polygon_area(poly)
    notes = [
        "pretrained: none (Stray RGB-D unproject)",
        "scale=lidar_depth_mm mpu=1.0",
        *notes,
    ]
    return RoomPlan(
        id=room_id,
        source_folder=source,
        polygon=[Point2(x=float(p[0]), y=float(p[1])) for p in poly],
        walls=walls,
        openings=[],
        ceiling_height=Interval.measured(ceiling, LIDAR_CEILING_REL, "lidar_depth_percentile"),
        floor_area=Interval.measured(area, LIDAR_AREA_REL, "lidar_occupancy", unit="m2"),
        notes=notes,
        backend="lidar_stray",
    )


def _clip_xy(xy: np.ndarray, pct: float) -> np.ndarray:
    if len(xy) < 20:
        return xy
    c = np.median(xy, axis=0)
    r = np.linalg.norm(xy - c, axis=1)
    return xy[r <= np.percentile(r, pct)]


def _speed_series(ds: LidarDataset) -> np.ndarray:
    xyz = ds.poses_wc[:, :3, 3]
    t = ds.timestamps
    if len(xyz) < 2:
        return np.array([])
    dt = np.diff(t)
    dt[dt < 1e-3] = 1e-3
    dist = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return dist / dt


def _aabb(xy: np.ndarray) -> tuple[float, float]:
    if len(xy) < 2:
        return 0.0, 0.0
    span = xy.max(axis=0) - xy.min(axis=0)
    return float(span[0]), float(span[1])
