#!/usr/bin/env python3
"""Build reports/cache from local out/ and zip the reproduction bundle.

Raw photos / MOV / Stray depth stay out of git. This zip is what you attach
to the submission. Unzip it at the repo root so captures/ is populated.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
CACHE = ROOT / "reports" / "cache"
BUNDLE_NAME = "cozmo-reproduction-bundle.zip"

SKIP_VIDEO_NAMES = {"Cozmox AI Capture.zip"}


def _sanitize(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in {"captures", "lidar_dataset"} and isinstance(v, str):
                p = Path(v)
                try:
                    out[k] = str(p.relative_to(ROOT))
                except ValueError:
                    out[k] = p.name
            else:
                out[k] = _sanitize(v)
        return out
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    return obj


def _room_summary(room: dict) -> dict:
    walls = [
        round(float(w["length"]["value"]), 3)
        for w in room.get("walls", [])
        if "length" in w
    ]
    ceil = room.get("ceiling_height") or {}
    area = room.get("floor_area") or {}
    return {
        "id": room.get("id"),
        "walls_m": walls,
        "ceiling_m": round(float(ceil["value"]), 3) if "value" in ceil else None,
        "area_m2": round(float(area["value"]), 3) if "value" in area else None,
        "scale": (ceil.get("method") or area.get("method")),
        "n_openings": len(room.get("openings") or []),
    }


def write_cache() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    mapping = {
        "photos_plan.json": ROOT / "out" / "plan.json",
        "photos_fix_after_plan.json": ROOT / "out" / "fix-after" / "plan.json",
        "video_plan.json": ROOT / "out" / "video" / "plan.json",
        "lidar_plan.json": ROOT / "out" / "lidar" / "plan.json",
    }
    summary: dict = {"note": "Cached CLI outputs. Live path: python -m cozmo run ..."}
    for name, src in mapping.items():
        if not src.is_file():
            print(f"skip cache (missing): {src}")
            continue
        data = _sanitize(json.loads(src.read_text(encoding="utf-8")))
        dest = CACHE / name
        dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
        summary[name] = {
            "tier": data.get("tier"),
            "stitch": data.get("stitch"),
            "rooms": [_room_summary(r) for r in data.get("rooms", [])],
        }
        print(f"wrote {dest.relative_to(ROOT)}")
    (CACHE / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {CACHE.relative_to(ROOT)}/summary.json")


def _iter_capture_files() -> list[Path]:
    files: list[Path] = []
    photos = ROOT / "captures" / "photos"
    if photos.is_dir():
        for p in photos.rglob("*"):
            if not p.is_file() or p.name.startswith("."):
                continue
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".heif"}:
                files.append(p)
    video = ROOT / "captures" / "video"
    if video.is_dir():
        for p in video.iterdir():
            if p.is_file() and p.name not in SKIP_VIDEO_NAMES and not p.name.startswith("."):
                if p.suffix.lower() in {".mov", ".mp4", ".m4v"}:
                    files.append(p)
    lidar = ROOT / "captures" / "lidar" / "property"
    if lidar.is_dir():
        for p in lidar.rglob("*"):
            if p.is_file() and p.name != ".DS_Store":
                files.append(p)
    incumbent = ROOT / "captures" / "incumbent"
    if incumbent.is_dir():
        for p in incumbent.rglob("*"):
            if p.is_file() and p.name != ".DS_Store":
                files.append(p)
    return files


def write_manifest(files: list[Path]) -> str:
    lines = [
        "# cozmo-floorplan reproduction bundle",
        "",
        "Unzip at the clone root. Then:",
        "",
        "```",
        "python -m cozmo run captures/ --out out/ --backend vggt",
        "python -m cozmo run captures/ --out out/video --tier video --backend vggt",
        "python -m cozmo run captures/ --out out/lidar --tier lidar",
        "python -m cozmo run captures/ --out out/fix-after --only room_01 --backend vggt",
        "```",
        "",
        "Cached numbers (also in git): `reports/cache/`. Tape: `reports/ground-truth.json`.",
        "Not in this zip: VGGT/YOLO weights (fetched on first run), `Applied AI.md`.",
        "",
        "## Contents",
        "",
    ]
    counts: dict[str, int] = {}
    bytes_by: dict[str, int] = {}
    for p in files:
        rel = p.relative_to(ROOT)
        top = str(rel).split("/")[1] if rel.parts[0] == "captures" and len(rel.parts) > 1 else str(rel)
        counts[top] = counts.get(top, 0) + 1
        bytes_by[top] = bytes_by.get(top, 0) + p.stat().st_size
    for k in sorted(counts):
        mb = bytes_by[k] / (1024 * 1024)
        lines.append(f"- `{k}`: {counts[k]} files, {mb:.1f} MB")
    lines.append("")
    lines.append(f"Total files: {len(files)}")
    lines.append("")
    return "\n".join(lines) + "\n"


def pack_zip() -> Path:
    files = _iter_capture_files()
    if not files:
        raise SystemExit("No capture files found. Nothing to zip.")
    DIST.mkdir(parents=True, exist_ok=True)
    dest = DIST / BUNDLE_NAME
    if dest.exists():
        dest.unlink()
    extra = [
        ROOT / "reports" / "ground-truth.json",
        ROOT / "reports" / "benchmark.md",
        ROOT / "reports" / "fix-loop.md",
        ROOT / "docs" / "protocol.md",
        ROOT / "docs" / "reproduction.md",
    ]
    extra += sorted(CACHE.glob("*.json"))
    extra += list((ROOT / "reports" / "fix-loop").glob("*.json"))
    extra = [p for p in extra if p.is_file()]
    manifest = write_manifest(files)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("MANIFEST.md", manifest)
        for p in files + extra:
            zf.write(p, p.relative_to(ROOT).as_posix())
    print(f"packed {dest} ({dest.stat().st_size / (1024 * 1024):.1f} MB) from {len(files)} capture files")
    return dest


def main() -> int:
    write_cache()
    pack_zip()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
