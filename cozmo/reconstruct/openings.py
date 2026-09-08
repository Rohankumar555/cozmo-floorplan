"""Door/window detection: pretrained YOLO-World (open-vocab).

COCO YOLO has no door class. Text prompts keep this extensible to damage later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROMPT_CLASSES = ["door", "doorway", "door frame", "open door", "window"]
CLASSES = ["door", "window"]
KIND_MAP = {
    "door": "door",
    "doorway": "door",
    "door frame": "door",
    "open door": "door",
    "window": "window",
}
WEIGHT_NAME = "yolov8s-worldv2.pt"


def _weight_file() -> str:
    root = Path(__file__).resolve().parents[2]
    local = root / "weights" / WEIGHT_NAME
    if local.exists():
        return str(local)
    return WEIGHT_NAME


@dataclass
class Detection:
    kind: str
    conf: float
    xyxy: tuple[float, float, float, float]  # original image pixels
    image: Path


_MODEL = None


def _patch_cv2_headless() -> None:
    import cv2

    if not hasattr(cv2, "imshow"):
        cv2.imshow = lambda *a, **k: None  # type: ignore[method-assign]
        cv2.waitKey = lambda *a, **k: 0  # type: ignore[method-assign]
        cv2.destroyAllWindows = lambda *a, **k: None  # type: ignore[method-assign]


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    _patch_cv2_headless()
    from ultralytics import YOLO

    # YOLO-World if present; else try YOLOWorld class; else generic YOLO with world weights.
    try:
        from ultralytics import YOLOWorld

        model = YOLOWorld(_weight_file())
        model.set_classes(PROMPT_CLASSES)
    except Exception:
        model = YOLO(_weight_file())
        if hasattr(model, "set_classes"):
            model.set_classes(PROMPT_CLASSES)
    _MODEL = model
    return model


def detect_openings(images: list[Path], conf_min: float = 0.08) -> list[Detection]:
    model = _load_model()
    out: list[Detection] = []
    for path in images:
        result = model.predict(str(path), verbose=False, conf=conf_min)[0]
        names = result.names or {}
        boxes = result.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            raw = str(names.get(cls_id, "")).lower()
            if raw not in KIND_MAP:
                if cls_id < len(PROMPT_CLASSES):
                    raw = PROMPT_CLASSES[cls_id].lower()
                else:
                    continue
            label = KIND_MAP.get(raw)
            if label is None:
                continue
            xyxy = boxes.xyxy[i].detach().cpu().numpy().tolist()
            score = float(boxes.conf[i].item())
            out.append(
                Detection(
                    kind=label,
                    conf=score,
                    xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    image=path,
                )
            )
    return _nms_per_kind(out)


def _nms_per_kind(dets: list[Detection], iou_thresh: float = 0.5) -> list[Detection]:
    kept: list[Detection] = []
    for kind in CLASSES:
        group = [d for d in dets if d.kind == kind]
        group.sort(key=lambda d: d.conf, reverse=True)
        used = [False] * len(group)
        for i, a in enumerate(group):
            if used[i]:
                continue
            kept.append(a)
            for j in range(i + 1, len(group)):
                if used[j]:
                    continue
                if a.image != group[j].image:
                    continue
                if _iou(a.xyxy, group[j].xyxy) > iou_thresh:
                    used[j] = True
    return kept


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return inter / denom if denom else 0.0


def bbox_aspect_is_door(det: Detection) -> bool:
    x0, y0, x1, y1 = det.xyxy
    w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    return h / w >= 1.25


def door_widths_scene_units(dets: list[Detection], images: list[Path], scene) -> list[float]:
    """3D width of YOLO doors on VGGT pointmaps. Empty if no real door was measured."""
    widths: list[float] = []
    if not getattr(scene, "pointmaps", None):
        return widths
    for d in dets:
        if d.kind != "door" or d.conf < 0.15:
            continue
        if not bbox_aspect_is_door(d) and d.conf < 0.25:
            continue
        w = _bbox_width_3d(d, images, scene)
        if w is not None:
            widths.append(w)
    return widths


def _bbox_width_3d(det: Detection, images: list[Path], scene) -> float | None:
    try:
        idx = images.index(det.image)
    except ValueError:
        return None
    if idx >= len(scene.pointmaps):
        return None
    pm = scene.pointmaps[idx]
    conf = scene.point_conf[idx] if scene.point_conf else None
    ph, pw = pm.shape[:2]
    from PIL import Image

    ow, oh = Image.open(det.image).size
    x0, y0, x1, y1 = det.xyxy
    ix0 = int(np.clip(x0 / ow * pw, 0, pw - 1))
    ix1 = int(np.clip(x1 / ow * pw, 0, pw - 1))
    iy0 = int(np.clip(y0 / oh * ph, 0, ph - 1))
    iy1 = int(np.clip(y1 / oh * ph, 0, ph - 1))
    if ix1 <= ix0 + 2 or iy1 <= iy0 + 2:
        return None
    bw = ix1 - ix0
    left_x = ix0 + max(1, int(0.08 * bw))
    right_x = ix1 - max(1, int(0.08 * bw))
    left = _column_xyz(pm, conf, iy0, iy1, left_x)
    right = _column_xyz(pm, conf, iy0, iy1, right_x)
    if left is None or right is None:
        return None
    width = float(np.linalg.norm(right - left))
    if not (1e-4 < width < 8.0):
        return None
    return width


def _column_xyz(pm, conf, y0: int, y1: int, x: int):
    sl = pm[y0:y1, max(0, x - 1) : x + 2]
    pts = sl.reshape(-1, 3)
    if conf is not None:
        cf = conf[y0:y1, max(0, x - 1) : x + 2].reshape(-1)
        pts = pts[(cf > 0.2) & np.isfinite(pts).all(axis=1)]
    else:
        pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 4:
        return None
    return np.median(pts, axis=0)
