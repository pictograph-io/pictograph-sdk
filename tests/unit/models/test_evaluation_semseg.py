"""Semantic-segmentation eval fields parse onto ModelEvaluation (B461).

A semseg eval reuses the same `model_evaluations` row as detection but carries
pixel metrics: `overall_metrics.miou` (the discriminator), per-class `iou`/`dice`,
and the `seg_metrics` payload (per-class IoU/Dice + operating sweep). The response
model is `extra="ignore"`, so these must be DECLARED to survive parsing.
"""

from __future__ import annotations

from pictograph.models.evaluation import ModelEvaluation


def _seg_row() -> dict:
    return {
        "id": "ev-seg-1",
        "organization_id": "org-1",
        "model_id": "mdl-1",
        "project_id": "proj-1",
        "export_id": "exp-1",
        "status": "completed",
        "iou_threshold": 0.5,
        "confidence_threshold": 0.5,
        "total_images": 3,
        "evaluated_images": 3,
        "failed_images": 0,
        "overall_metrics": {
            "tp": 4100,
            "fp": 100,
            "fn": 900,
            "precision": 0.976,
            "recall": 0.82,
            "f1": 0.891,
            "macro_f1": 0.85,
            "map": None,
            "accuracy": None,
            "miou": 0.803,
            "mean_dice": 0.85,
            "pixel_accuracy": 0.9,
            "mean_pixel_accuracy": 0.82,
            "frequency_weighted_iou": 0.803,
        },
        "per_class_metrics": [
            {
                "class_name": "road",
                "tp": 4100,
                "fp": 100,
                "fn": 900,
                "support": 5000,
                "precision": 0.976,
                "recall": 0.82,
                "f1": 0.891,
                "iou": 0.803,
                "dice": 0.891,
            },
        ],
        "confusion_matrix": {
            "iou_threshold": None,
            "labels": ["road", "__background__"],
            "grid": [[4100, 900], [100, 4900]],
        },
        "seg_metrics": {
            "confidence_threshold": 0.5,
            "n_images": 3,
            "n_pixels": 30000,
            "miou": 0.803,
            "mean_dice": 0.85,
            "pixel_accuracy": 0.9,
            "mean_pixel_accuracy": 0.82,
            "frequency_weighted_iou": 0.803,
            "per_class": [{"class_name": "road", "iou": 0.803, "dice": 0.891, "support": 5000}],
            "operating_sweep": {
                "thresholds": [0.0, 0.5, 1.0],
                "miou": [0.803, 0.803, 0.0],
                "mean_dice": [0.891, 0.891, 0.0],
                "pixel_accuracy": [0.9, 0.9, 0.5],
                "best_miou": 0.803,
                "best_miou_threshold": 0.0,
            },
        },
        "worst_images": [
            {
                "image_id": "img-1",
                "filename": "a.jpg",
                "tp": 100,
                "fp": 20,
                "fn": 30,
                "gt_count": 1,
                "pred_count": 1,
                "miou": 0.4,
                "predictions": [],
                "ground_truth": [],
            },
        ],
    }


def test_semseg_overall_and_discriminator() -> None:
    ev = ModelEvaluation.model_validate(_seg_row())
    o = ev.overall_metrics
    assert o is not None
    # miou present, map/accuracy None → the semseg discriminator (vs detection/cls).
    assert o.miou == 0.803
    assert o.map is None and o.accuracy is None
    assert o.pixel_accuracy == 0.9
    assert o.frequency_weighted_iou == 0.803


def test_semseg_per_class_iou_dice() -> None:
    ev = ModelEvaluation.model_validate(_seg_row())
    assert ev.per_class_metrics is not None
    c = ev.per_class_metrics[0]
    assert c.iou == 0.803 and c.dice == 0.891


def test_semseg_metrics_payload_and_sweep() -> None:
    ev = ModelEvaluation.model_validate(_seg_row())
    assert ev.seg_metrics is not None
    sweep = ev.seg_metrics["operating_sweep"]
    assert sweep["best_miou"] == 0.803
    assert len(sweep["thresholds"]) == len(sweep["miou"]) == 3


def test_semseg_per_image_miou_ranking() -> None:
    ev = ModelEvaluation.model_validate(_seg_row())
    assert ev.worst_images is not None
    assert ev.worst_images[0].miou == 0.4


def test_detection_row_leaves_semseg_fields_none() -> None:
    row = _seg_row()
    row["overall_metrics"] = {
        "tp": 10,
        "fp": 2,
        "fn": 1,
        "precision": 0.83,
        "recall": 0.9,
        "f1": 0.86,
        "macro_f1": 0.86,
        "map": 0.75,
    }
    row["seg_metrics"] = None
    ev = ModelEvaluation.model_validate(row)
    assert ev.overall_metrics is not None
    assert ev.overall_metrics.miou is None
    assert ev.overall_metrics.map == 0.75
    assert ev.seg_metrics is None
