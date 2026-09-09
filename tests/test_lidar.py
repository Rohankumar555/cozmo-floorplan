import numpy as np

from cozmo.ingest.lidar import _quat_to_R, depth_intrinsics, find_lidar_dataset
from cozmo.reconstruct.lidar import _occupancy_polygon, _room_from_poly
from cozmo.reconstruct.planes import polygon_area


def test_depth_k_scales_1920_to_256():
    K = np.array([[1600.0, 0.0, 960.0], [0.0, 1600.0, 720.0], [0.0, 0.0, 1.0]])
    Kd = depth_intrinsics(K)
    assert abs(Kd[0, 0] - 1600 * 256 / 1920) < 1e-6
    assert abs(Kd[1, 2] - 720 * 192 / 1440) < 1e-6


def test_identity_quat_is_I():
    R = _quat_to_R(0.0, 0.0, 0.0, 1.0)
    assert np.allclose(R, np.eye(3), atol=1e-6)


def test_occupancy_polygon_on_a_4x3_box():
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 4, 4000)
    ys = rng.uniform(0, 3, 4000)
    # Keep a 0.15 m band near the walls so the grid is a rectangle, not a filled blob's hull only.
    wall = (xs < 0.12) | (xs > 3.88) | (ys < 0.12) | (ys > 2.88)
    xy = np.stack([xs[wall], ys[wall]], axis=1)
    poly = _occupancy_polygon(xy, cell=0.08)
    area = polygon_area(poly)
    assert 8.0 < area < 16.0
    room = _room_from_poly("property", "test", poly, 2.45, [])
    assert room.backend == "lidar_stray"
    assert room.ceiling_height.method == "lidar_depth_percentile"
    assert len(room.walls) >= 4


def test_find_stray_dataset(tmp_path):
    root = tmp_path / "captures" / "lidar" / "property" / "abc"
    (root / "depth").mkdir(parents=True)
    (root / "depth" / "000000.png").write_bytes(b"x")
    (root / "odometry.csv").write_text("timestamp, frame, x, y, z, qx, qy, qz, qw\n")
    (root / "camera_matrix.csv").write_text("1,0,0\n0,1,0\n0,0,1\n")
    found = find_lidar_dataset(tmp_path / "captures")
    assert found == root
