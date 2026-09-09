"""Stray Scanner RGB-D datasets: depth, poses, intrinsics (LiDAR tier)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEPTH_WH = (256, 192)  # Stray depth maps (width, height)
RGB_WH = (1920, 1440)


@dataclass(frozen=True)
class LidarDataset:
    path: Path
    K_rgb: np.ndarray  # 3x3
    timestamps: np.ndarray
    poses_wc: np.ndarray  # (N, 4, 4) camera-to-world
    depth_paths: tuple[Path, ...]
    conf_paths: tuple[Path, ...]
    rgb_path: Path | None


def find_lidar_dataset(captures: Path) -> Path:
    """First folder under captures/lidar that looks like a Stray export."""
    root = captures / "lidar"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing lidar root: {root}")
    hits: list[Path] = []
    for p in root.rglob("odometry.csv"):
        if p.name.startswith("."):
            continue
        folder = p.parent
        if (folder / "camera_matrix.csv").is_file() and (folder / "depth").is_dir():
            hits.append(folder)
    if not hits:
        raise FileNotFoundError(f"No Stray dataset (odometry.csv + depth/) under {root}")
    return sorted(hits, key=lambda x: str(x))[0]


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) + 1e-12)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def load_dataset(path: Path) -> LidarDataset:
    K = np.loadtxt(path / "camera_matrix.csv", delimiter=",")
    if K.shape != (3, 3):
        K = np.asarray(K, dtype=float).reshape(3, 3)
    odo = np.loadtxt(path / "odometry.csv", delimiter=",", skiprows=1, usecols=range(13))
    if odo.ndim == 1:
        odo = odo.reshape(1, -1)
    n = odo.shape[0]
    poses = np.repeat(np.eye(4)[None, ...], n, axis=0)
    for i in range(n):
        poses[i, :3, :3] = _quat_to_R(*odo[i, 5:9])
        poses[i, :3, 3] = odo[i, 2:5]
    depth_dir = path / "depth"
    conf_dir = path / "confidence"
    depths = tuple(sorted(depth_dir.glob("*.png")))
    confs = tuple(sorted(conf_dir.glob("*.png"))) if conf_dir.is_dir() else ()
    if len(depths) != n:
        # Pair by stem so a truncated export still loads.
        by_d = {p.stem: p for p in depths}
        keep = []
        poses_keep = []
        ts_keep = []
        for i in range(n):
            stem = f"{int(odo[i, 1]):06d}"
            if stem in by_d:
                keep.append(by_d[stem])
                poses_keep.append(poses[i])
                ts_keep.append(odo[i, 0])
        depths = tuple(keep)
        poses = np.stack(poses_keep, axis=0) if poses_keep else poses[:0]
        timestamps = np.asarray(ts_keep, dtype=float)
    else:
        timestamps = odo[:, 0].astype(float)
    rgb = path / "rgb.mp4"
    return LidarDataset(
        path=path,
        K_rgb=np.asarray(K, dtype=float),
        timestamps=timestamps,
        poses_wc=poses,
        depth_paths=depths,
        conf_paths=confs,
        rgb_path=rgb if rgb.is_file() else None,
    )


def depth_intrinsics(K_rgb: np.ndarray) -> np.ndarray:
    """Scale RGB 1920×1440 K down to Stray 256×192 depth."""
    sx = DEPTH_WH[0] / RGB_WH[0]
    sy = DEPTH_WH[1] / RGB_WH[1]
    K = K_rgb.copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K
