"""Classical geometry: plane RANSAC, floor polygon, wall segments.

Used after VGGT (or any point cloud). No learned weights here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Plane:
    normal: np.ndarray  # (3,) unit
    offset: float  # n·x + offset = 0  with offset = -n·p0
    points: np.ndarray

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.normal + self.offset


def _fit_plane_svd(pts: np.ndarray) -> tuple[np.ndarray, float]:
    c = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - c, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    offset = -float(normal @ c)
    return normal, offset


def ransac_plane(
    pts: np.ndarray,
    thresh: float = 0.02,
    iters: int = 400,
    min_inliers: int = 100,
    rng: np.random.Generator | None = None,
) -> Plane | None:
    if len(pts) < 3:
        return None
    rng = rng or np.random.default_rng(0)
    best_idx: np.ndarray | None = None
    best_n = 0
    n = len(pts)
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        sample = pts[idx]
        if np.linalg.matrix_rank(sample - sample[0]) < 2:
            continue
        normal, offset = _fit_plane_svd(sample)
        dist = np.abs(pts @ normal + offset)
        inliers = dist < thresh
        count = int(inliers.sum())
        if count > best_n:
            best_n = count
            best_idx = np.flatnonzero(inliers)
    if best_idx is None or best_n < min_inliers:
        return None
    normal, offset = _fit_plane_svd(pts[best_idx])
    dist = np.abs(pts @ normal + offset)
    keep = dist < thresh * 1.5
    return Plane(normal=normal, offset=offset, points=pts[keep])


def scene_scale(pts: np.ndarray) -> float:
    """Typical point-to-centroid distance; used to set RANSAC thresholds."""
    c = np.median(pts, axis=0)
    d = np.linalg.norm(pts - c, axis=1)
    return float(np.median(d)) + 1e-6


def sequential_planes(
    pts: np.ndarray,
    max_planes: int = 8,
    thresh: float | None = None,
    rng: np.random.Generator | None = None,
) -> list[Plane]:
    remaining = pts.copy()
    planes: list[Plane] = []
    if thresh is None:
        thresh = 0.04 * scene_scale(pts)
    min_inliers = max(80, len(pts) // 40)
    for _ in range(max_planes):
        if len(remaining) < min_inliers:
            break
        plane = ransac_plane(remaining, thresh=thresh, min_inliers=min_inliers, rng=rng)
        if plane is None:
            break
        planes.append(plane)
        dist = np.abs(remaining @ plane.normal + plane.offset)
        remaining = remaining[dist >= thresh * 1.5]
    return planes


def _up_axis(planes: list[Plane]) -> np.ndarray:
    """Floor/ceiling normals are the most horizontal-supporting (largest |n_y| after we pick gravity)."""
    if not planes:
        return np.array([0.0, 1.0, 0.0])
    # Prefer the plane whose inliers have the smallest coordinate along its normal (floor).
    scored = []
    for p in planes:
        n = p.normal
        # Gravity is the normal with most support among near-parallel planes
        scored.append((len(p.points), p))
    scored.sort(key=lambda t: t[0], reverse=True)
    # Among large planes, pick the one whose centroid is lowest in some axis — try all 3
    best_up = np.array([0.0, 1.0, 0.0])
    best_score = -1.0
    for axis in range(3):
        cand = np.zeros(3)
        cand[axis] = 1.0
        align = sum(abs(float(p.normal @ cand)) * len(p.points) for p in planes)
        if align > best_score:
            best_score = align
            best_up = cand
    # Orient up so floor centroid is below ceiling
    floors = [p for p in planes if abs(float(p.normal @ best_up)) > 0.75]
    if len(floors) >= 1:
        heights = [(float(p.points.mean(axis=0) @ best_up), p) for p in floors]
        heights.sort()
        # If the lowest plane's normal points "down", flip up
        if float(heights[0][1].normal @ best_up) > 0:
            best_up = -best_up
    return best_up


def floor_polygon_from_planes(
    planes: list[Plane],
    points: np.ndarray | None = None,
    up: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return (quad in floor coords, ceiling height in scene units, up vector).

    VGGT clouds include furniture. An 80th-percentile radius clip around the
    dense near-floor core throws away sparse plaster. Keep the outer 96%.
    """
    if points is None or len(points) < 10:
        if not planes:
            raise ValueError("No planes")
        points = np.concatenate([p.points for p in planes], axis=0)
    up = up if up is not None else (_up_axis(planes) if planes else _up_from_points(points))
    up = up / (np.linalg.norm(up) + 1e-12)

    heights = points @ up
    floor_h = float(np.percentile(heights, 18))
    ceil_h = float(np.percentile(heights, 82))
    ceiling = float(abs(ceil_h - floor_h))
    if ceiling < 1e-6:
        raise ValueError("Degenerate height range")

    band = 0.15 * ceiling
    near = points[np.abs(heights - floor_h) <= band]
    if len(near) < 30:
        near = points[heights <= floor_h + 0.3 * ceiling]

    tmp = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(up, tmp)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    y_axis = np.cross(up, x_axis)
    origin = up * floor_h

    xy = np.stack([(near - origin) @ x_axis, (near - origin) @ y_axis], axis=1)
    c = np.median(xy, axis=0)
    r = np.linalg.norm(xy - c, axis=1)
    xy = xy[r <= np.percentile(r, 96)]
    poly = _min_area_rect(xy)
    return poly, ceiling, up


def _up_from_points(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts, full_matrices=False)
    up = vh[-1]
    return up / (np.linalg.norm(up) + 1e-12)


def _min_area_rect(xy: np.ndarray) -> np.ndarray:
    """PCA-aligned bounding rectangle (no OpenCV)."""
    if len(xy) < 3:
        return xy
    c = xy.mean(axis=0)
    _, _, vh = np.linalg.svd(xy - c, full_matrices=False)
    a, b = vh[0], vh[1]
    proj = np.stack([(xy - c) @ a, (xy - c) @ b], axis=1)
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    local = np.array(
        [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]],
        dtype=float,
    )
    world = c + np.outer(local[:, 0], a) + np.outer(local[:, 1], b)
    return _order_poly(world)


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(pts, 5), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    return hull


def _order_poly(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def walls_from_polygon(poly: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    segs = []
    n = len(poly)
    for i in range(n):
        segs.append((poly[i], poly[(i + 1) % n]))
    return segs
