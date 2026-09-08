"""Homeowner-readable SVG from the same RoomPlan that JSON uses."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from cozmo.schema import PropertyPlan, RoomPlan


def write_property_svgs(plan: PropertyPlan, out_dir: Path) -> list[Path]:
    rooms_dir = out_dir / "rooms"
    rooms_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for room in plan.rooms:
        path = rooms_dir / f"{room.id}.svg"
        path.write_text(_room_svg(room), encoding="utf-8")
        written.append(path)
    if len(plan.rooms) == 1:
        (out_dir / "plan.svg").write_text(_room_svg(plan.rooms[0]), encoding="utf-8")
    else:
        # Unstitched: tile rooms in a row until door-graph exists.
        (out_dir / "plan.svg").write_text(_unstitched_svg(plan), encoding="utf-8")
    return written


def _room_svg(room: RoomPlan, width: int = 900, height: int = 700) -> str:
    if len(room.polygon) < 3:
        return _svg_wrap(width, height, f'<text x="40" y="40">No polygon for {escape(room.id)}</text>')
    xs = [p.x for p in room.polygon]
    ys = [p.y for p in room.polygon]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    pad = 80
    spanx = max(maxx - minx, 1e-6)
    spany = max(maxy - miny, 1e-6)
    scale = min((width - 2 * pad) / spanx, (height - 2 * pad) / spany)

    def xy(x: float, y: float) -> tuple[float, float]:
        return pad + (x - minx) * scale, height - pad - (y - miny) * scale

    pts = " ".join(f"{xy(p.x, p.y)[0]:.1f},{xy(p.x, p.y)[1]:.1f}" for p in room.polygon)
    parts = [
        f'<polygon points="{pts}" fill="#f4efe6" stroke="#1f1f1f" stroke-width="3"/>',
        f'<text x="24" y="32" font-size="18" font-family="Helvetica">{escape(room.id)}  ·  {escape(room.backend)}</text>',
        (
            f'<text x="24" y="54" font-size="12" font-family="Helvetica" fill="#444">'
            f'area {room.floor_area.value:.2f} {room.floor_area.unit}  '
            f'[{room.floor_area.ci_low:.2f}, {room.floor_area.ci_high:.2f}]  ·  '
            f'ceiling {room.ceiling_height.value:.2f} m  ·  scale {escape(room.ceiling_height.method)}'
            f"</text>"
        ),
    ]
    for wall in room.walls:
        mx, my = xy((wall.start.x + wall.end.x) / 2, (wall.start.y + wall.end.y) / 2)
        parts.append(
            f'<text x="{mx:.1f}" y="{my:.1f}" font-size="11" font-family="Helvetica" fill="#333">'
            f"{wall.length.value:.2f} m</text>"
        )
    for op in room.openings:
        wall = next((w for w in room.walls if w.id == op.wall_id), None)
        if wall is None:
            continue
        t = 0.5 * (op.t_start + op.t_end)
        x = wall.start.x + t * (wall.end.x - wall.start.x)
        y = wall.start.y + t * (wall.end.y - wall.start.y)
        px, py = xy(x, y)
        color = "#c45c26" if op.kind == "door" else "#2b6cb0"
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{color}"/>'
            f'<text x="{px + 8:.1f}" y="{py - 8:.1f}" font-size="10" font-family="Helvetica" fill="{color}">'
            f"{op.kind} {op.width.value:.2f} m</text>"
        )
    return _svg_wrap(width, height, "\n".join(parts))


def _unstitched_svg(plan: PropertyPlan) -> str:
    tiles = []
    x = 0
    for i, room in enumerate(plan.rooms):
        inner = _room_svg(room, 640, 500)
        body = inner.split("<svg", 1)[-1]
        body = body[body.find(">") + 1 :].rsplit("</svg>", 1)[0]
        tiles.append(f'<g transform="translate({x},0)">{body}</g>')
        x += 660
    note = (
        '<text x="24" y="24" font-size="14" font-family="Helvetica">'
        "Unstitched photo rooms (door-graph not applied yet)</text>"
    )
    return _svg_wrap(max(x, 900), 540, note + "\n" + "\n".join(tiles))


def _svg_wrap(w: int, h: int, inner: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
        f'<rect width="100%" height="100%" fill="white"/>\n'
        f"{inner}\n</svg>\n"
    )
