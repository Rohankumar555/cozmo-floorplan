"""Walkthrough clip: find the MOV, rotate portrait to landscape, sample frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v"}


@dataclass(frozen=True)
class VideoClip:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    rotate_90_cw: bool


def find_walkthrough(captures: Path) -> Path:
    root = captures / "video"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing video root: {root}")
    clips = sorted(
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES and not p.name.startswith(".")
    )
    if not clips:
        raise FileNotFoundError(f"No .mov/.mp4 under {root}")
    return clips[0]


def open_clip(path: Path) -> VideoClip:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 1e-3:
        fps = 30.0
    duration = n / fps if n else 0.0
    # Stored portrait (h > w) or ORIENTATION_META 90 → landscape for VGGT.
    rotate = h > w
    return VideoClip(
        path=path,
        width=w,
        height=h,
        fps=fps,
        frame_count=n,
        duration_s=duration,
        rotate_90_cw=rotate,
    )


def _orient(frame: np.ndarray, rotate_90_cw: bool) -> np.ndarray:
    if rotate_90_cw:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def motion_series(clip: VideoClip, sample_hz: float = 4.0, thumb_w: int = 320) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_s, mean-abs-diff) at sample_hz. Cheap still vs walk detector."""
    cap = cv2.VideoCapture(str(clip.path))
    step = max(1, int(round(clip.fps / sample_hz)))
    prev = None
    times: list[float] = []
    motion: list[float] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step != 0:
            i += 1
            continue
        frame = _orient(frame, clip.rotate_90_cw)
        h, w = frame.shape[:2]
        scale = thumb_w / max(w, 1)
        small = cv2.resize(frame, (thumb_w, max(1, int(h * scale))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        t = i / clip.fps
        if prev is not None:
            times.append(t)
            motion.append(float(np.mean(np.abs(gray.astype(np.float32) - prev))))
        prev = gray.astype(np.float32)
        i += 1
    cap.release()
    return np.asarray(times, dtype=float), np.asarray(motion, dtype=float)


def extract_frames(
    clip: VideoClip,
    t_start: float,
    t_end: float,
    dest: Path,
    count: int = 6,
    max_side: int = 1280,
) -> list[Path]:
    """Write `count` JPEGs from (t_start, t_end), inset from the edges."""
    dest.mkdir(parents=True, exist_ok=True)
    if count < 2:
        count = 2
    span = max(0.4, t_end - t_start)
    inset = min(0.35, 0.12 * span)
    lo, hi = t_start + inset, t_end - inset
    if hi <= lo:
        lo, hi = t_start, t_end
    stamps = np.linspace(lo, hi, count)
    cap = cv2.VideoCapture(str(clip.path))
    written: list[Path] = []
    for k, t in enumerate(stamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = _orient(frame, clip.rotate_90_cw)
        h, w = frame.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        path = dest / f"{k:02d}.jpg"
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        written.append(path)
    cap.release()
    return written
