"""Discover per-room photo folders. Same layout later for video/ and lidar/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


@dataclass(frozen=True)
class PhotoRoom:
    room_id: str
    folder: Path
    images: tuple[Path, ...]


def _normalize_room_id(folder_name: str) -> str:
    name = folder_name.strip().lower().replace(" ", "_")
    if name.startswith("room_") and len(name) == 6 and name[-1].isdigit():
        return f"room_0{name[-1]}"
    return name


def load_photo_rooms(captures: Path) -> list[PhotoRoom]:
    root = captures / "photos"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing photo root: {root}")

    rooms: list[PhotoRoom] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        images = tuple(
            sorted(
                p
                for p in folder.iterdir()
                if p.is_file()
                and p.suffix.lower() in PHOTO_SUFFIXES
                and not p.name.startswith(".")
            )
        )
        rooms.append(
            PhotoRoom(room_id=_normalize_room_id(folder.name), folder=folder, images=images)
        )
    return rooms
