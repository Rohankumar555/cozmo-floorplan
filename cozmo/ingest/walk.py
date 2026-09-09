"""Turn a walkthrough clip into PhotoRoom segments (2–8 stills each)."""

from __future__ import annotations

from pathlib import Path

from cozmo.ingest.photos import PhotoRoom
from cozmo.ingest.segment import WalkSegment, holds_from_motion, segments_from_holds
from cozmo.ingest.video import VideoClip, extract_frames, find_walkthrough, motion_series, open_clip


def load_video_rooms(captures: Path, work_dir: Path, stills_per_seg: int = 6) -> tuple[list[PhotoRoom], dict]:
    """Find the MOV, cut at door holds, dump stills, return rooms in walk order."""
    path = find_walkthrough(captures)
    clip = open_clip(path)
    times, motion = motion_series(clip)
    holds = holds_from_motion(times, motion)
    segs = segments_from_holds(clip.duration_s, holds)
    stills_per_seg = min(8, max(2, stills_per_seg))
    rooms: list[PhotoRoom] = []
    extra_segs: list[dict] = []
    for seg in segs:
        dest = work_dir / "frames" / seg.id
        images = extract_frames(clip, seg.t_start, seg.t_end, dest, count=stills_per_seg)
        rooms.append(PhotoRoom(room_id=seg.id, folder=dest, images=tuple(images)))
        extra_segs.append(
            {
                "id": seg.id,
                "t_start": round(seg.t_start, 3),
                "t_end": round(seg.t_end, 3),
                "stills": len(images),
            }
        )
    extra = {
        "video": str(path),
        "duration_s": round(clip.duration_s, 3),
        "rotate_90_cw": clip.rotate_90_cw,
        "holds": [{"t_start": round(h.t_start, 3), "t_end": round(h.t_end, 3)} for h in holds],
        "segments": extra_segs,
    }
    return rooms, extra
