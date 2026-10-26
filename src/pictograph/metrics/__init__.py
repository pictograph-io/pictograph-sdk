"""Client-side model-evaluation metrics - offline, dependency-free.

Measure how good a model is against a labeled set without a server round-trip or
a third-party library: :func:`evaluate_detections` matches predicted annotations
to ground truth by IoU and returns per-class + overall precision / recall / F1.

    from pictograph import Client
    from pictograph.metrics import evaluate_detections

    client = Client()
    gt = {img: client.annotations.get(img) for img in image_ids}
    preds = {img: run_my_model(img) for img in image_ids}   # your predictions
    result = evaluate_detections(preds, gt, iou_threshold=0.5)
    print(result.precision, result.recall, result.f1)
    for name, m in result.per_class.items():
        print(name, m.precision, m.recall, m.support)
"""

from __future__ import annotations

from pictograph.metrics._detection import (
    BACKGROUND,
    ClassMetrics,
    ConfusionMatrix,
    DetectionMetrics,
    bbox_iou,
    confusion_matrix,
    evaluate_detections,
)

__all__ = [
    "BACKGROUND",
    "ClassMetrics",
    "ConfusionMatrix",
    "DetectionMetrics",
    "bbox_iou",
    "confusion_matrix",
    "evaluate_detections",
]
