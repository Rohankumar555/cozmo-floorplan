"""CLI: one command per capture. Room reconstruction is per-folder and reusable."""

from __future__ import annotations

import argparse
from pathlib import Path

from cozmo.export.svg import write_property_svgs
from cozmo.ingest.photos import load_photo_rooms
from cozmo.reconstruct.room import reconstruct_room
from cozmo.schema import PropertyPlan, SCHEMA_VERSION
from cozmo.stitch.door_graph import stitch_rooms

DISCLOSURES = [
    "VGGT facebook/VGGT-1B (if installed): pretrained few-view reconstruction, inference only.",
    "YOLO-World yolov8s-worldv2.pt: pretrained open-vocab detector, prompts door/window, inference only.",
    "Photo-tier scale: 0.80 m door measured in 3D, else 0.80×1.20 m floor tiles. Never a fake 12% door. Not LiDAR-metric.",
    "Stitch: detector-door snaps; rooms that share a hub wall pack along it (order = door position, not folder names). No house layout priors.",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cozmo", description="iPhone capture → dimensioned plan")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Reconstruct a capture directory")
    run.add_argument("captures", type=Path, help="Directory with photos/, video/, lidar/")
    run.add_argument("--out", type=Path, default=Path("out"))
    run.add_argument(
        "--only",
        type=str,
        default=None,
        help="Room folder id to reconstruct (e.g. room_01). Default: all photo folders.",
    )
    run.add_argument(
        "--backend",
        choices=("auto", "vggt", "manhattan"),
        default="auto",
        help="auto uses VGGT when installed, else Manhattan fallback",
    )
    args = parser.parse_args(argv)
    if args.cmd == "run":
        return _cmd_run(args.captures, args.out, args.only, args.backend)
    return 2


def _cmd_run(captures: Path, out: Path, only: str | None, backend: str) -> int:
    captures = captures.resolve()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rooms = load_photo_rooms(captures)
    if only:
        key = only.lower()
        rooms = [r for r in rooms if r.room_id == key or r.folder.name.lower() == key]
        if not rooms:
            raise SystemExit(f"No photo folder matching --only {only}")

    built = [reconstruct_room(r, backend=backend) for r in rooms]
    extra: dict = {"captures": str(captures), "backend": backend}
    stitch_flag = "unstitched"
    adjacency = []
    if only is None and len(built) >= 2:
        result = stitch_rooms(built)
        built = result.rooms
        adjacency = result.adjacency
        extra["stitch_ablation"] = result.ablation
        extra["stitch_notes"] = result.notes
        extra["property_footprint"] = result.ablation.get("property_footprint", {})
        if adjacency:
            stitch_flag = "door_graph"
    plan = PropertyPlan(
        schema_version=SCHEMA_VERSION,
        tier="photos",
        stitch=stitch_flag,  # type: ignore[arg-type]
        rooms=built,
        adjacency=adjacency,
        disclosures=DISCLOSURES,
        extra=extra,
    )
    json_path = out / "plan.json"
    json_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    write_property_svgs(plan, out)
    print(f"Wrote {json_path}")
    print(f"Wrote {out / 'plan.svg'}")
    for r in built:
        print(f"  {r.id}: backend={r.backend} walls={len(r.walls)} openings={len(r.openings)}")
    print(f"  stitch={stitch_flag} adjacencies={len(adjacency)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
