from pathlib import Path

from cozmo.ingest.photos import load_photo_rooms


def test_load_photo_rooms_finds_room_01():
    root = Path(__file__).resolve().parents[1]
    rooms = {r.room_id: r for r in load_photo_rooms(root / "captures")}
    assert "room_01" in rooms
    assert 2 <= len(rooms["room_01"].images) <= 8
    if "room_3" in {r.folder.name for r in rooms.values()} or "room_03" in rooms:
        assert "room_03" in rooms
