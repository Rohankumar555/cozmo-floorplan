"""Door/window detection: pretrained YOLO-World (open-vocab).

COCO YOLO has no door class. Text prompts keep this extensible to damage later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CLASSES = ["door", "window"]
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
        model.set_classes(CLASSES)
    except Exception:
        model = YOLO(_weight_file())
        if hasattr(model, "set_classes"):
            model.set_classes(CLASSES)
    _MODEL = model
    return model


def detect_openings(images: list[Path], conf_min: float = 0.15) -> list[Detection]:
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
            label = str(names.get(cls_id, CLASSES[cls_id] if cls_id < len(CLASSES) else "door"))
            label = label.lower()
            if label not in CLASSES:
                # world model may return prompted order
                if cls_id < len(CLASSES):
                    label = CLASSES[cls_id]
                else:
                    continue
            xyxy = boxes.xyxy[i].detach().cpu().numpy().tolist()
            score = float(boxes.conf[i].item())
            out.append(
                Detection(
                    kind=label,  # type: ignore[arg-type]
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
    return h / w >= 1.4
