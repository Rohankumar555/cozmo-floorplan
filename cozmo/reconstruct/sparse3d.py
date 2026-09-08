"""Few-view 3D: VGGT if installed, else a Manhattan line-layout fallback.

VGGT (facebook/VGGT-1B) is disclosed pretrained reconstruction.
The fallback exists so the CLI still emits a plan when weights are not fetched yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SparseScene:
    points: np.ndarray
    backend: str
    pointmaps: list[np.ndarray] = field(default_factory=list)
    pointmap_hw: list[tuple[int, int]] = field(default_factory=list)
    point_conf: list[np.ndarray] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def reconstruct_sparse(images: list[Path], backend: str = "auto") -> SparseScene:
    want = backend
    if backend == "auto":
        want = "vggt" if _vggt_importable() else "manhattan"
    if want == "vggt":
        try:
            return _reconstruct_vggt(images)
        except Exception as exc:  # noqa: BLE001
            if backend == "vggt":
                raise
            scene = _reconstruct_manhattan(images)
            scene.notes.append(f"vggt_failed: {type(exc).__name__}: {exc}")
            return scene
    if want == "manhattan":
        return _reconstruct_manhattan(images)
    raise ValueError(f"Unknown backend {backend}")


def _vggt_importable() -> bool:
    try:
        import vggt  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


_VGGT_MODEL = None


def _torch_device():
    import os
    import torch

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _vggt_model(device):
    global _VGGT_MODEL
    if _VGGT_MODEL is None:
        from vggt.models.vggt import VGGT

        _VGGT_MODEL = VGGT.from_pretrained("facebook/VGGT-1B").eval()
    return _VGGT_MODEL.to(device)


def _downsized_paths(images: list[Path], max_side: int = 1280) -> list[str]:
    """VGGT resizes internally; feeding 8K JPEGs still blows RAM on load."""
    import tempfile

    from PIL import Image

    tmp = Path(tempfile.mkdtemp(prefix="cozmo_vggt_"))
    names: list[str] = []
    for i, src in enumerate(images):
        im = Image.open(src)
        im = im.convert("RGB")
        w, h = im.size
        scale = max_side / max(w, h)
        if scale < 1:
            im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
        dest = tmp / f"{i:02d}.jpg"
        im.save(dest, quality=90)
        names.append(str(dest))
    return names


def _reconstruct_vggt(images: list[Path]) -> SparseScene:
    import torch
    from vggt.utils.load_fn import load_and_preprocess_images

    device = _torch_device()
    model = _vggt_model(device)
    names = _downsized_paths(images)
    tensor = load_and_preprocess_images(names).to(device)
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.cuda.amp.autocast(dtype=dtype):
                pred = model(tensor)
        else:
            pred = model(tensor)

    if not isinstance(pred, dict):
        pred = dict(pred)

    world = pred.get("world_points")
    conf = pred.get("world_points_conf")
    if world is None:
        raise RuntimeError("VGGT output missing world_points")

    world_np = world.detach().float().cpu().numpy()
    while world_np.ndim > 4:
        world_np = world_np[0]
    # world_np: S,H,W,3
    if conf is not None:
        conf_np = conf.detach().float().cpu().numpy()
        while conf_np.ndim > 3:
            conf_np = conf_np[0]
    else:
        conf_np = np.ones(world_np.shape[:3], dtype=np.float32)

    pointmaps = []
    confmaps = []
    hws = []
    chunks = []
    for i in range(world_np.shape[0]):
        pm = world_np[i]
        cf = conf_np[i]
        pointmaps.append(pm)
        confmaps.append(cf)
        hws.append((pm.shape[0], pm.shape[1]))
        mask = cf > 0.25
        pts = pm[mask]
        if len(pts):
            # Subsample for RANSAC
            if len(pts) > 80_000:
                rng = np.random.default_rng(0)
                pts = pts[rng.choice(len(pts), 80_000, replace=False)]
            chunks.append(pts)

    if not chunks:
        raise RuntimeError("VGGT produced no confident points")
    points = np.concatenate(chunks, axis=0)
    return SparseScene(
        points=points.astype(np.float64),
        backend="vggt",
        pointmaps=pointmaps,
        pointmap_hw=hws,
        point_conf=confmaps,
        notes=["pretrained: facebook/VGGT-1B", f"device={device.type}"],
    )


def _reconstruct_manhattan(images: list[Path]) -> SparseScene:
    """Axis-aligned room box in arbitrary units from line extents across views.

    Not metric. Enough to exercise planes → polygon → openings → scale when VGGT
    is not installed. Marked in notes so CIs stay wide.
    """
    import cv2

    xs: list[float] = []
    ys: list[float] = []
    z_top = 0.6
    z_bot = 0.0
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = 800 / max(h, w)
        small = cv2.resize(img, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 60, 160)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=12)
        if lines is None:
            continue
        segs = lines.reshape(-1, 4)
        for x1, y1, x2, y2 in segs:
            dx, dy = x2 - x1, y2 - y1
            ang = abs(np.degrees(np.arctan2(dy, dx)))
            if ang > 90:
                ang = 180 - ang
            # Verticals → x walls; near-horizontal → depth cue
            if ang > 70:
                xs.append(x1 / small.shape[1])
                xs.append(x2 / small.shape[1])
            elif ang < 20:
                ys.append(y1 / small.shape[0])
                ys.append(y2 / small.shape[0])

    if len(xs) < 4:
        xs = [0.1, 0.9]
    if len(ys) < 4:
        ys = [0.2, 0.85]
    x0, x1 = float(np.percentile(xs, 8)), float(np.percentile(xs, 92))
    y0, y1 = float(np.percentile(ys, 15)), float(np.percentile(ys, 90))
    if x1 - x0 < 0.2:
        x0, x1 = 0.1, 0.9
    if y1 - y0 < 0.15:
        y0, y1 = 0.2, 0.8

    # Build a dense box point cloud (scene units ~ image fractions)
    gx = np.linspace(x0, x1, 40)
    gy = np.linspace(y0, y1, 40)
    gz = np.linspace(z_bot, z_top, 16)
    pts = []
    # floor / ceiling
    for z in (z_bot, z_top):
        xx, yy = np.meshgrid(gx, gy)
        pts.append(np.stack([xx.ravel(), np.full(xx.size, z), yy.ravel()], axis=1))
    # four walls
    xx, zz = np.meshgrid(gx, gz)
    pts.append(np.stack([xx.ravel(), zz.ravel(), np.full(xx.size, y0)], axis=1))
    pts.append(np.stack([xx.ravel(), zz.ravel(), np.full(xx.size, y1)], axis=1))
    yy, zz = np.meshgrid(gy, gz)
    pts.append(np.stack([np.full(yy.size, x0), zz.ravel(), yy.ravel()], axis=1))
    pts.append(np.stack([np.full(yy.size, x1), zz.ravel(), yy.ravel()], axis=1))
    cloud = np.concatenate(pts, axis=0)
    return SparseScene(
        points=cloud,
        backend="manhattan",
        notes=[
            "fallback: vanishing/line box, not VGGT",
            "scale must come from door prior; CIs stay photo-wide",
        ],
    )
