"""Floor-tile grout → spacing in VGGT scene units.

Hallway tiles are 0.80 m × 1.20 m. Scale never uses a fake 'door = 12% of wall'.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cozmo.reconstruct.sparse3d import SparseScene

TILE_SHORT_M = 0.80
TILE_LONG_M = 1.20
RATIO_LO = 1.25
RATIO_HI = 1.85


def tile_spacings_scene_units(images: list[Path], scene: SparseScene) -> list[float]:
    """Return 1–2 grout periods in the same units as VGGT world points."""
    if not scene.pointmaps:
        return []
    raw: list[float] = []
    n = min(len(images), len(scene.pointmaps))
    for i in range(n):
        conf = scene.point_conf[i] if scene.point_conf else None
        raw.extend(_floor_raster_spacings(images[i], scene.pointmaps[i], conf))
        raw.extend(_spacings_for_view(images[i], scene.pointmaps[i], conf))
    return _cluster_periods(raw)


def spacings_to_metres_per_unit(spacings: list[float]) -> tuple[float, str] | None:
    """Map 3D grout period(s) to metres_per_unit using 0.80×1.20 m tiles."""
    vals = sorted(s for s in spacings if s > 1e-6)
    if not vals:
        return None
    if len(vals) == 1:
        s = vals[0]
        return TILE_SHORT_M / s, "tile_0.80m"
    a, b = vals[0], vals[-1]
    ratio = b / a
    if RATIO_LO <= ratio <= RATIO_HI:
        mpu = 0.5 * (TILE_SHORT_M / a + TILE_LONG_M / b)
        return mpu, "tile_0.80x1.20m"
    if 1.85 < ratio < 2.35:
        # Every-other grout on the same family, not 80 vs 120.
        return TILE_SHORT_M / a, "tile_0.80m"
    mid = 0.5 * (a + b)
    return TILE_SHORT_M / mid, "tile_0.80m"


def _floor_raster_spacings(path: Path, pointmap: np.ndarray, conf: np.ndarray | None) -> list[float]:
    """Project floor pixels to a metric-ish raster, then Hough the grout grid."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return []
    ph, pw = pointmap.shape[:2]
    small = cv2.resize(img, (pw, ph))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    y0 = int(ph * 0.38)
    sl_pm = pointmap[y0:, :, :]
    sl_g = gray[y0:, :]
    sl_c = conf[y0:, :] if conf is not None else np.ones(sl_g.shape, dtype=np.float32)
    pts = sl_pm.reshape(-1, 3)
    inten = sl_g.reshape(-1)
    c = sl_c.reshape(-1)
    m = (c > 0.2) & np.isfinite(pts).all(axis=1)
    pts, inten = pts[m], inten[m]
    if len(pts) < 400:
        return []
    centre = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centre, full_matrices=False)
    uv = (pts - centre) @ vh[:2].T
    span = uv.max(axis=0) - uv.min(axis=0)
    if float(span.min()) < 1e-6:
        return []
    res = float(span.max() / 220.0)
    origin = uv.min(axis=0)
    iu = np.clip(((uv[:, 0] - origin[0]) / res).astype(int), 0, 511)
    iv = np.clip(((uv[:, 1] - origin[1]) / res).astype(int), 0, 511)
    gh, gw = int(iv.max()) + 1, int(iu.max()) + 1
    acc = np.zeros((gh, gw), dtype=np.float64)
    cnt = np.zeros((gh, gw), dtype=np.float64)
    np.add.at(acc, (iv, iu), inten)
    np.add.at(cnt, (iv, iu), 1.0)
    raster = acc / np.maximum(cnt, 1.0)
    raster_u8 = np.clip(raster, 0, 255).astype(np.uint8)
    raster_u8 = cv2.GaussianBlur(raster_u8, (3, 3), 0)
    # Grout is darker than tile body.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    tophat = cv2.morphologyEx(raster_u8, cv2.MORPH_BLACKHAT, kernel)
    edges = cv2.Canny(tophat, 20, 80)
    if int(edges.sum()) < 200:
        edges = cv2.Canny(raster_u8, 40, 120)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=max(18, min(gh, gw) // 14))
    if lines is None or len(lines) < 4:
        return []
    groups = _cluster_theta(lines[:, 0])
    out: list[float] = []
    for pack in groups:
        rhos = sorted(float(r) for r, _t in pack)
        uniq: list[float] = []
        min_sep = max(6.0, 0.05 * float(span.max()) / res)
        for r in rhos:
            if not uniq or abs(r - uniq[-1]) > min_sep:
                uniq.append(r)
        gaps = [abs(r1 - r0) * res for r0, r1 in zip(uniq, uniq[1:])]
        lo, hi = 0.06 * float(span.max()), 0.40 * float(span.max())
        gaps = [g for g in gaps if lo < g < hi]
        if gaps:
            out.extend(gaps)
    return out


def _spacings_for_view(path: Path, pointmap: np.ndarray, conf: np.ndarray | None) -> list[float]:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return []
    ph, pw = pointmap.shape[:2]
    small = cv2.resize(img, (pw, ph))
    h, w = small.shape[:2]
    floor = small[int(h * 0.42) :, :]
    gray = cv2.cvtColor(floor, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=max(40, min(h, w) // 12))
    if lines is None or len(lines) < 4:
        return []
    y_off = int(h * 0.42)
    groups = _cluster_theta(lines[:, 0])
    out: list[float] = []
    for pack in groups:
        rhos = sorted(float(r) for r, _t in pack)
        uniq: list[float] = []
        for r in rhos:
            if not uniq or abs(r - uniq[-1]) > 4:
                uniq.append(r)
        theta = float(np.median([t for _r, t in pack]))
        for r0, r1 in zip(uniq, uniq[1:]):
            dist = _line_pair_world_gap(r0, r1, theta, pointmap, conf, y_off, ph, pw)
            if dist is not None:
                out.append(dist)
    return out


def _cluster_theta(lines: np.ndarray, n_bins: int = 2) -> list[list[tuple[float, float]]]:
    thetas = np.array([float(t) for _r, t in lines])
    xs = np.stack([np.cos(2 * thetas), np.sin(2 * thetas)], axis=1)
    rng = np.random.default_rng(0)
    n_bins = min(n_bins, len(xs))
    centres = xs[rng.choice(len(xs), size=n_bins, replace=False)]
    lab = np.zeros(len(xs), dtype=int)
    for _ in range(8):
        d = ((xs[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        lab = d.argmin(axis=1)
        for k in range(len(centres)):
            if np.any(lab == k):
                centres[k] = xs[lab == k].mean(axis=0)
    groups: list[list[tuple[float, float]]] = [[] for _ in range(len(centres))]
    for (rho, theta), k in zip(lines, lab):
        groups[k].append((float(rho), float(theta)))
    return [g for g in groups if len(g) >= 3]


def _line_pair_world_gap(
    rho0: float,
    rho1: float,
    theta: float,
    pointmap: np.ndarray,
    conf: np.ndarray | None,
    y_off: int,
    ph: int,
    pw: int,
) -> float | None:
    pts0 = _sample_line_world(rho0, theta, pointmap, conf, y_off, ph, pw)
    pts1 = _sample_line_world(rho1, theta, pointmap, conf, y_off, ph, pw)
    if pts0 is None or pts1 is None:
        return None
    d = [float(np.linalg.norm(pts1 - p, axis=1).min()) for p in pts0[:: max(1, len(pts0) // 24)]]
    if not d:
        return None
    gap = float(np.median(d))
    char = float(np.linalg.norm(pointmap[ph // 2, pw // 2])) + 1.0
    if not (0.02 < gap < 0.5 * max(char, 1.0)):
        return None
    return gap


def _sample_line_world(
    rho: float,
    theta: float,
    pointmap: np.ndarray,
    conf: np.ndarray | None,
    y_off: int,
    ph: int,
    pw: int,
) -> np.ndarray | None:
    c, s = np.cos(theta), np.sin(theta)
    pts = []
    for t in np.linspace(0, max(ph, pw), 40):
        if abs(s) > abs(c):
            x = t
            y = (rho - x * c) / (s + 1e-9)
        else:
            y = t
            x = (rho - y * s) / (c + 1e-9)
        yi = int(round(y + y_off))
        xi = int(round(x))
        if 0 <= yi < ph and 0 <= xi < pw:
            if conf is not None and conf[yi, xi] < 0.2:
                continue
            p = pointmap[yi, xi]
            if np.all(np.isfinite(p)):
                pts.append(p)
    if len(pts) < 6:
        return None
    return np.asarray(pts, dtype=np.float64)


def _cluster_periods(raw: list[float]) -> list[float]:
    if not raw:
        return []
    v = np.array(raw, dtype=float)
    v = v[np.isfinite(v)]
    v = v[(v > 1e-4) & (v < 8.0)]
    if len(v) >= 4:
        # Drop tiny raster-bin harmonics (≪ the median grout).
        med = float(np.median(v))
        v = v[(v > 0.45 * med) & (v < 2.4 * med)]
    if len(v) < 2:
        return [float(np.median(v))] if len(v) else []
    log = np.log(np.clip(v, 1e-4, None)).reshape(-1, 1)
    c0, c1 = float(np.percentile(log, 25)), float(np.percentile(log, 75))
    for _ in range(6):
        d0 = np.abs(log[:, 0] - c0)
        d1 = np.abs(log[:, 0] - c1)
        m0 = d0 <= d1
        if m0.any():
            c0 = float(log[m0].mean())
        if (~m0).any():
            c1 = float(log[~m0].mean())
    a, b = float(np.exp(c0)), float(np.exp(c1))
    if a > b:
        a, b = b, a
    if b / a < 1.15:
        return [float(np.median(v))]
    return [a, b]
