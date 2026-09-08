#!/usr/bin/env python3
"""Validate photo-tier folders: captures/photos/<room>/*.{jpg,jpeg,heic,png}"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = ROOT / "captures" / "photos"
ALLOWED = {".jpg", ".jpeg", ".heic", ".heif", ".png"}


def file_kind(path: Path) -> str:
    try:
        out = subprocess.check_output(["file", "-b", str(path)], text=True).strip()
        return out
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    if not PHOTOS.exists():
        print(f"Missing {PHOTOS}")
        print("Create captures/photos/<room_name>/ and put 2–8 stills in each folder.")
        return 1

    rooms = sorted(p for p in PHOTOS.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not rooms:
        print(f"No room folders in {PHOTOS}")
        return 1

    ok = True
    print(f"Photo root: {PHOTOS}\n")
    for room in rooms:
        shots = sorted(
            p
            for p in room.iterdir()
            if p.is_file() and p.suffix.lower() in ALLOWED and not p.name.startswith(".")
        )
        n = len(shots)
        status = "OK" if 2 <= n <= 8 else "WARN"
        if n == 0:
            status = "EMPTY"
            ok = False
        print(f"[{status}] {room.name}: {n} image(s)")
        for shot in shots:
            size_mb = shot.stat().st_size / (1024 * 1024)
            print(f"       {shot.name:40} {size_mb:6.2f} MB  {file_kind(shot)}")
        print()

    if not ok:
        print("Add 2–8 original stills per room, then re-run.")
        return 1

    print("Layout looks usable. Next: drop a walkthrough .mov under captures/video/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
