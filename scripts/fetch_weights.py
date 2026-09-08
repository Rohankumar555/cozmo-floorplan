#!/usr/bin/env python3
"""Download disclosed pretrained weights (YOLO-World; VGGT via Hugging Face on first run)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"


def main() -> int:
    WEIGHTS.mkdir(exist_ok=True)
    dest = WEIGHTS / "yolov8s-worldv2.pt"
    print("Fetching YOLO-World (yolov8s-worldv2.pt) via Ultralytics...")
    from ultralytics import YOLO

    try:
        from ultralytics import YOLOWorld

        m = YOLOWorld("yolov8s-worldv2.pt")
    except Exception:
        m = YOLO("yolov8s-worldv2.pt")
    cwd_pt = Path.cwd() / "yolov8s-worldv2.pt"
    if cwd_pt.exists() and not dest.exists():
        dest.write_bytes(cwd_pt.read_bytes())
        print(f"Copied weights to {dest}")
    print("YOLO-World ready:", type(m))
    print(
        "VGGT: pip install -e '.[vggt]' then the first `cozmo run` downloads "
        "facebook/VGGT-1B from Hugging Face (~1B params)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
