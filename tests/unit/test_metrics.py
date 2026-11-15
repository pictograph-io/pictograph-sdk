"""Tests for ``pictograph.metrics`` - client-side detection evaluation.

Hand-verified cases pin the matching semantics: perfect match, false positive,
false negative, wrong-class (FP + FN), below-threshold IoU, confidence-ordered
greedy claiming, and multi-image / multi-class aggregation.
"""

from __future__ import annotations

import pytest

from pictograph.metrics import ClassMetrics, DetectionMetrics, bbox_iou, evaluate_detections
from pictograph.models.annotation import BBoxAnnotation, PolygonAnnotation, PolygonGeometry
from pictograph.models.common import BoundingBox, Point


def _box(
    name: str, x: float, y: float, w: float = 10, h: float = 10, conf: float = 1.0
) -> BBoxAnnotation:
    return BBoxAnnotation(name=name, bounding_box=BoundingBox(x=x, y=y, w=w, h=h), confidence=conf)


# ───────────── bbox_iou ─────────────


def test_bbox_iou_identical() -> None:
    b = BoundingBox(x=0, y=0, w=10, h=10)
    assert bbox_iou(b, b) == 1.0


def test_bbox_iou_disjoint() -> None:
    assert bbox_iou(BoundingBox(x=0, y=0, w=5, h=5), BoundingBox(x=100, y=100, w=5, h=5)) == 0.0


def test_bbox_iou_half_overlap() -> None:
    # inter = 5x10 = 50, union = 100 + 100 - 50 = 150 -> 1/3
    iou = bbox_iou(BoundingBox(x=0, y=0, w=10, h=10), BoundingBox(x=5, y=0, w=10, h=10))
    assert iou == pytest.approx(1 / 3)


# ───────────── evaluate_detections ─────────────


def test_perfect_match() -> None:
    gt = {"a": [_box("car", 0, 0)]}
    m = evaluate_detections(gt, gt)
    assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0
    assert m.per_class["car"].true_positives == 1


def test_false_positive() -> None:
    gt = {"a": [_box("car", 0, 0)]}
    preds = {"a": [_box("car", 0, 0), _box("car", 50, 50)]}  # one extra, no GT
    m = evaluate_detections(preds, gt)
    assert m.true_positives == 1 and m.false_positives == 1 and m.false_negatives == 0
    assert m.precision == pytest.approx(0.5) and m.recall == 1.0


def test_false_negative() -> None:
    gt = {"a": [_box("car", 0, 0), _box("car", 50, 50)]}
    preds = {"a": [_box("car", 0, 0)]}  # missed the second
    m = evaluate_detections(preds, gt)
    assert m.true_positives == 1 and m.false_negatives == 1
    assert m.recall == pytest.approx(0.5) and m.precision == 1.0


def test_wrong_class_is_fp_and_fn() -> None:
    gt = {"a": [_box("car", 0, 0)]}
    preds = {"a": [_box("truck", 0, 0)]}  # right place, wrong class
    m = evaluate_detections(preds, gt)
    assert m.per_class["truck"].false_positives == 1
    assert m.per_class["car"].false_negatives == 1
    assert m.true_positives == 0


def test_below_threshold_iou_no_match() -> None:
    gt = {"a": [_box("car", 0, 0, w=10, h=10)]}
    preds = {"a": [_box("car", 8, 0, w=10, h=10)]}  # small overlap, iou < 0.5
    m = evaluate_detections(preds, gt, iou_threshold=0.5)
    assert m.true_positives == 0 and m.false_positives == 1 and m.false_negatives == 1


def test_greedy_claims_by_confidence() -> None:
    # Two overlapping preds, one GT: the higher-confidence pred should claim it (TP),
    # the other becomes an FP - not two TPs.
    gt = {"a": [_box("car", 0, 0)]}
    preds = {"a": [_box("car", 0, 0, conf=0.6), _box("car", 1, 1, conf=0.9)]}
    m = evaluate_detections(preds, gt)
    assert m.true_positives == 1 and m.false_positives == 1


def test_multi_image_multi_class_aggregation() -> None:
    gt = {
        "a": [_box("car", 0, 0), _box("sign", 50, 50)],
        "b": [_box("car", 0, 0)],
    }
    preds = {
        "a": [_box("car", 0, 0), _box("sign", 50, 50)],  # both correct
        "b": [_box("car", 100, 100)],  # missed (FP + FN)
    }
    m = evaluate_detections(preds, gt)
    assert m.per_class["car"].true_positives == 1
    assert m.per_class["car"].false_positives == 1
    assert m.per_class["car"].false_negatives == 1
    assert m.per_class["sign"].true_positives == 1
    # overall micro: TP=2, FP=1, FN=1
    assert m.true_positives == 2 and m.false_positives == 1 and m.false_negatives == 1


def test_polygon_uses_enclosing_box() -> None:
    poly_gt = PolygonAnnotation(
        name="road",
        polygon=PolygonGeometry(paths=[[Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=10)]]),
    )
    m = evaluate_detections({"a": [_box("road", 0, 0)]}, {"a": [poly_gt]})
    assert m.per_class["road"].true_positives == 1  # box matches the polygon's bbox


def test_empty_inputs() -> None:
    m = evaluate_detections({}, {})
    assert m.precision == 0.0 and m.recall == 0.0 and m.per_class == {}


def test_class_metrics_properties() -> None:
    cm = ClassMetrics(class_name="c", true_positives=3, false_positives=1, false_negatives=2)
    assert cm.precision == pytest.approx(0.75)
    assert cm.recall == pytest.approx(0.6)
    assert cm.f1 == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))
    assert cm.support == 5


def test_macro_f1() -> None:
    m = DetectionMetrics(
        iou_threshold=0.5,
        per_class={
            "a": ClassMetrics("a", 1, 0, 0),  # f1 = 1
            "b": ClassMetrics("b", 0, 1, 1),  # f1 = 0
        },
    )
    assert m.macro_f1 == pytest.approx(0.5)


# ───────────── average precision / mAP ─────────────


def test_ap_perfect_is_one() -> None:
    gt = {"a": [_box("car", 0, 0)]}
    preds = {"a": [_box("car", 0, 0, conf=0.9)]}
    m = evaluate_detections(preds, gt)
    assert m.per_class["car"].average_precision == pytest.approx(1.0)
    assert m.mean_average_precision == pytest.approx(1.0)


def test_ap_tp_fp_tp_hand_verified() -> None:
    # 2 GT; confidence-ranked detections TP, FP, TP → AP = 0.8333…
    gt = {"a": [_box("car", 0, 0), _box("car", 100, 100)]}
    preds = {
        "a": [
            _box("car", 0, 0, conf=0.9),  # TP
            _box("car", 0, 0, conf=0.8),  # FP (box 1 claimed)
            _box("car", 100, 100, conf=0.7),  # TP
        ]
    }
    m = evaluate_detections(preds, gt)
    assert m.per_class["car"].average_precision == pytest.approx(0.5 + 0.5 * (2 / 3))


def test_ap_zero_all_fp_or_all_fn() -> None:
    assert (
        evaluate_detections({"a": [_box("car", 0, 0, conf=0.9)]}, {})
        .per_class["car"]
        .average_precision
        == 0.0
    )
    assert (
        evaluate_detections({}, {"a": [_box("car", 0, 0)]}).per_class["car"].average_precision
        == 0.0
    )


def test_map_averages_per_class_ap() -> None:
    gt = {"a": [_box("cat", 0, 0), _box("dog", 50, 50)]}
    preds = {"a": [_box("cat", 0, 0, conf=0.9)]}
    m = evaluate_detections(preds, gt)
    assert m.per_class["cat"].average_precision == pytest.approx(1.0)
    assert m.per_class["dog"].average_precision == pytest.approx(0.0)
    assert m.mean_average_precision == pytest.approx(0.5)


def test_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        evaluate_detections({}, {}, iou_threshold=0.0)
    with pytest.raises(ValueError, match="iou_threshold"):
        evaluate_detections({}, {}, iou_threshold=1.5)


# ───────────── confusion_matrix ─────────────

from pictograph.metrics import BACKGROUND, ConfusionMatrix, confusion_matrix  # noqa: E402


def test_confusion_matrix_records_class_confusion_fp_fn() -> None:
    gt = {"a": [_box("car", 0, 0), _box("sign", 50, 50)]}
    preds = {
        "a": [_box("truck", 0, 0), _box("car", 100, 100)]
    }  # car->truck, extra car (FP), sign missed (FN)
    cm = confusion_matrix(preds, gt)
    assert isinstance(cm, ConfusionMatrix)
    assert cm.classes == ["car", "sign", "truck"]
    assert cm.count("car", "truck") == 1  # confusion
    assert cm.count(BACKGROUND, "car") == 1  # false positive
    assert cm.count("sign", BACKGROUND) == 1  # false negative


def test_confusion_matrix_perfect_is_diagonal() -> None:
    gt = {"a": [_box("car", 0, 0)], "b": [_box("sign", 0, 0)]}
    cm = confusion_matrix(gt, gt)
    assert cm.count("car", "car") == 1 and cm.count("sign", "sign") == 1
    assert cm.count("car", "sign") == 0


def test_confusion_matrix_labels_and_grid() -> None:
    gt = {"a": [_box("car", 0, 0)]}
    cm = confusion_matrix(gt, gt)
    assert cm.labels == ["car", BACKGROUND]
    grid = cm.grid()
    # rows/cols aligned with labels: [[car->car, car->bg], [bg->car, bg->bg]]
    assert grid == [[1, 0], [0, 0]]


def test_confusion_matrix_class_agnostic_matching() -> None:
    # A truck prediction overlapping a car GT is a confusion, NOT a background FP -
    # matching ignores class (that's the whole point of the confusion matrix).
    cm = confusion_matrix({"a": [_box("truck", 0, 0)]}, {"a": [_box("car", 0, 0)]})
    assert cm.count("car", "truck") == 1
    assert cm.count(BACKGROUND, "truck") == 0


def test_confusion_matrix_empty_and_invalid() -> None:
    assert confusion_matrix({}, {}).classes == []
    with pytest.raises(ValueError, match="iou_threshold"):
        confusion_matrix({}, {}, iou_threshold=2.0)
