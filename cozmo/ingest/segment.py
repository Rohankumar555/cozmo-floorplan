"""Cut a walkthrough at door holds (still camera, motion on both sides)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Hold:
    t_start: float
    t_end: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.t_start + self.t_end)


@dataclass(frozen=True)
class WalkSegment:
    id: str
    t_start: float
    t_end: float


def holds_from_motion(
    times: np.ndarray,
    motion: np.ndarray,
    min_hold_s: float = 1.2,
) -> list[Hold]:
    """Still runs ≥ min_hold_s with walking before and after = a door pause."""
    times = np.asarray(times, dtype=float)
    motion = np.asarray(motion, dtype=float)
    if len(motion) < 8:
        return []
    k = 5
    ker = np.ones(k) / k
    sm = np.convolve(motion, ker, mode="same")
    hi = float(np.percentile(sm, 70))
    thresh = 0.32 * hi if hi > 1e-6 else 1e-3
    still = sm < thresh
    holds: list[Hold] = []
    i = 0
    n = len(still)
    while i < n:
        if not still[i]:
            i += 1
            continue
        j = i
        while j < n and still[j]:
            j += 1
        t0, t1 = float(times[i]), float(times[min(j, n - 1)])
        if t1 - t0 >= min_hold_s:
            before = sm[times < t0 - 0.25]
            after = sm[times > t1 + 0.25]
            moved_b = len(before) > 3 and float(np.percentile(before, 80)) > thresh
            # Recording-stop pause has almost no walk after; door holds do.
            moved_a = (
                (float(times[-1]) - t1) >= 3.0
                and len(after) > 3
                and float(np.percentile(after, 80)) > thresh
            )
            if moved_b and moved_a:
                holds.append(Hold(t_start=t0, t_end=t1))
        i = j
    return _merge_holds(holds, gap_s=0.8)


def _merge_holds(holds: list[Hold], gap_s: float) -> list[Hold]:
    if not holds:
        return []
    out = [holds[0]]
    for h in holds[1:]:
        prev = out[-1]
        if h.t_start - prev.t_end <= gap_s:
            out[-1] = Hold(t_start=prev.t_start, t_end=h.t_end)
        else:
            out.append(h)
    return out


def segments_from_holds(
    duration_s: float,
    holds: list[Hold],
    min_seg_s: float = 3.0,
) -> list[WalkSegment]:
    """One segment per room: cuts at the midpoint of each door hold."""
    cuts = [0.0]
    for h in holds:
        cuts.append(h.mid)
    cuts.append(float(duration_s))
    raw: list[tuple[float, float]] = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a >= min_seg_s:
            raw.append((a, b))
    if not raw:
        raw = [(0.0, float(duration_s))]
    return [WalkSegment(id=f"walk_{i:02d}", t_start=a, t_end=b) for i, (a, b) in enumerate(raw)]
