"""Client-side detection evaluation - precision / recall / F1 by IoU matching.

Compares a set of *predicted* annotations against *ground-truth* annotations,
entirely on your machine, so you can measure a model's quality against a
held-out set without a server round-trip or a third-party library - the
Pictograph-native answer to "how good is my model?".

Matching is the standard class-aware, greedy-by-confidence scheme: within each
class, predictions (highest confidence first) claim the unmatched ground-truth
box with the greatest IoU at or above ``iou_threshold``. A claimed pair is a true
positive; an unclaimed prediction is a false positive; an unclaimed ground-truth
box is a false negative. Boxes are compared via axis-aligned IoU (polygons /
keypoints use their enclosing box), so this evaluates *detection* quality.

Pure Python - no numpy, no third-party dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pictograph.formats._shared import annotation_bbox

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pictograph.models.annotation import Annotation
    from pictograph.models.common import BoundingBox

    _Bump = Callable[[dict[str, int], str], None]


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union of two axis-aligned boxes (0.0 if disjoint)."""
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class ClassMetrics:
    """Detection metrics for one class."""

    class_name: str
    true_positives: int
    false_positives: int
    false_negatives: int
    #: Average precision at the eval's IoU threshold (all-points interpolation of
    #: the confidence-ranked precision/recall curve). The per-class building block
    #: of :attr:`DetectionMetrics.mean_average_precision` (mAP).
    average_precision: float = 0.0

    @property
    def support(self) -> int:
        """Ground-truth instances of this class (TP + FN)."""
        return self.true_positives + self.false_negatives

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class DetectionMetrics:
    """Aggregate detection metrics over an evaluation set.

    Inspect :attr:`per_class` for per-class :class:`ClassMetrics`, and the
    ``precision`` / ``recall`` / ``f1`` properties for micro-averaged overall
    scores (pooled TP/FP/FN across all classes).
    """

    iou_threshold: float
    per_class: dict[str, ClassMetrics] = field(default_factory=dict)

    @property
    def true_positives(self) -> int:
        return sum(m.true_positives for m in self.per_class.values())

    @property
    def false_positives(self) -> int:
        return sum(m.false_positives for m in self.per_class.values())

    @property
    def false_negatives(self) -> int:
        return sum(m.false_negatives for m in self.per_class.values())

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def macro_f1(self) -> float:
        """Unweighted mean of per-class F1 (each class counts equally)."""
        if not self.per_class:
            return 0.0
        return sum(m.f1 for m in self.per_class.values()) / len(self.per_class)

    @property
    def mean_average_precision(self) -> float:
        """mAP at :attr:`iou_threshold` - the mean of per-class average precision."""
        if not self.per_class:
            return 0.0
        return sum(m.average_precision for m in self.per_class.values()) / len(self.per_class)


def _confidence(ann: Annotation) -> float:
    return float(getattr(ann, "confidence", 1.0))


def _average_precision(records: list[tuple[float, bool]], n_gt: int) -> float:
    """Average precision for one class via all-points (VOC-2010+/COCO area) interpolation.

    ``records`` is ``(confidence, is_true_positive)`` for every prediction of the class;
    ``n_gt`` is the ground-truth count (recall denominator). Predictions are ranked by
    confidence, the precision/recall curve is walked, precision is made monotonically
    non-increasing (the envelope), and AP is the area under it.
    """
    if n_gt <= 0 or not records:
        return 0.0
    ordered = sorted(records, key=lambda r: r[0], reverse=True)
    tp = fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for _conf, is_tp in ordered:
        if is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_gt)
    mrec = [0.0, *recalls, recalls[-1]]
    mpre = [0.0, *precisions, 0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return sum(
        (mrec[i + 1] - mrec[i]) * mpre[i + 1]
        for i in range(len(mrec) - 1)
        if mrec[i + 1] != mrec[i]
    )


def evaluate_detections(
    predictions_by_image: Mapping[str, Sequence[Annotation]],
    ground_truth_by_image: Mapping[str, Sequence[Annotation]],
    *,
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Evaluate predicted annotations against ground truth, per image and class.

    Args:
        predictions_by_image: ``image_key`` → the model's predicted annotations.
        ground_truth_by_image: ``image_key`` → the ground-truth annotations. Keys
            present in only one mapping are still scored (all-FP or all-FN).
        iou_threshold: Minimum IoU for a prediction to match a ground-truth box.

    Returns:
        A :class:`DetectionMetrics` - per-class TP/FP/FN + precision/recall/F1 +
        ``average_precision``, plus micro-averaged overall scores, ``macro_f1``,
        and ``mean_average_precision`` (mAP at ``iou_threshold``).

    Raises:
        ValueError: ``iou_threshold`` is not in ``(0, 1]``.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")

    tp: dict[str, int] = {}
    fp: dict[str, int] = {}
    fn: dict[str, int] = {}
    pred_records: dict[str, list[tuple[float, bool]]] = {}

    def bump(counter: dict[str, int], name: str) -> None:
        counter[name] = counter.get(name, 0) + 1

    image_keys = set(predictions_by_image) | set(ground_truth_by_image)
    for key in image_keys:
        preds = list(predictions_by_image.get(key, []))
        gts = list(ground_truth_by_image.get(key, []))
        _match_image(preds, gts, iou_threshold, tp, fp, fn, bump, pred_records)

    classes = set(tp) | set(fp) | set(fn)
    per_class = {
        name: ClassMetrics(
            class_name=name,
            true_positives=tp.get(name, 0),
            false_positives=fp.get(name, 0),
            false_negatives=fn.get(name, 0),
            average_precision=_average_precision(
                pred_records.get(name, []), tp.get(name, 0) + fn.get(name, 0)
            ),
        )
        for name in classes
    }
    return DetectionMetrics(iou_threshold=iou_threshold, per_class=per_class)


def _match_image(
    preds: list[Annotation],
    gts: list[Annotation],
    iou_threshold: float,
    tp: dict[str, int],
    fp: dict[str, int],
    fn: dict[str, int],
    bump: _Bump,
    pred_records: dict[str, list[tuple[float, bool]]] | None = None,
) -> None:
    """Class-aware greedy IoU match for one image, updating the tp/fp/fn counters.

    When ``pred_records`` is given, records ``(confidence, is_true_positive)`` per
    prediction (keyed by class) so the caller can compute average precision.
    """
    # Precompute each annotation's box once; skip anything with no derivable box.
    pred_boxes: list[tuple[Annotation, BoundingBox]] = [
        (p, box) for p in preds if (box := annotation_bbox(p)) is not None
    ]
    gt_boxes: list[tuple[Annotation, BoundingBox]] = [
        (g, box) for g in gts if (box := annotation_bbox(g)) is not None
    ]

    matched_gt: set[int] = set()
    # Highest-confidence predictions claim their best box first.
    for pred, pbox in sorted(pred_boxes, key=lambda pb: _confidence(pb[0]), reverse=True):
        best_iou = iou_threshold
        best_idx = -1
        for idx, (gt, gbox) in enumerate(gt_boxes):
            if idx in matched_gt or gt.name != pred.name:
                continue
            iou = bbox_iou(pbox, gbox)
            if iou >= best_iou:
                best_iou = iou
                best_idx = idx
        matched = best_idx >= 0
        if matched:
            matched_gt.add(best_idx)
            bump(tp, pred.name)
        else:
            bump(fp, pred.name)
        if pred_records is not None:
            pred_records.setdefault(pred.name, []).append((_confidence(pred), matched))

    # Every unmatched ground-truth box is a false negative for its class.
    for idx, (gt, _gbox) in enumerate(gt_boxes):
        if idx not in matched_gt:
            bump(fn, gt.name)


BACKGROUND = "__background__"
"""Pseudo-label in a :class:`ConfusionMatrix` for an unmatched prediction (a false
positive, row ``__background__``) or an unmatched ground-truth box (a false
negative, column ``__background__``)."""


@dataclass
class ConfusionMatrix:
    """Detection confusion matrix - rows are ground-truth labels, columns predicted.

    ``count(gt, pred)`` is how many ground-truth ``gt`` boxes were matched to a
    prediction labeled ``pred``; the diagonal is correct detections and the
    off-diagonal is class confusion (e.g. a ``car`` predicted as ``truck``).
    :data:`BACKGROUND` appears as a row (unmatched predictions → false positives)
    and a column (unmatched ground truth → false negatives).

    Unlike :func:`evaluate_detections`, matching here is **class-agnostic** (a
    prediction can match a ground-truth box of any class, which is what surfaces
    the confusion), so the diagonal counts need not equal that function's TP.
    """

    iou_threshold: float
    classes: list[str] = field(default_factory=list)
    _counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def count(self, gt_label: str, pred_label: str) -> int:
        """Number of ground-truth ``gt_label`` boxes matched to a ``pred_label`` prediction."""
        return self._counts.get(gt_label, {}).get(pred_label, 0)

    @property
    def labels(self) -> list[str]:
        """Row/column order for display: the real classes (sorted) then ``__background__``."""
        return [*self.classes, BACKGROUND]

    def grid(self) -> list[list[int]]:
        """The matrix as a dense 2-D list aligned with :attr:`labels` (rows=gt, cols=pred)."""
        labels = self.labels
        return [[self.count(gt, pred) for pred in labels] for gt in labels]


def confusion_matrix(
    predictions_by_image: Mapping[str, Sequence[Annotation]],
    ground_truth_by_image: Mapping[str, Sequence[Annotation]],
    *,
    iou_threshold: float = 0.5,
) -> ConfusionMatrix:
    """Build a class-agnostic detection confusion matrix (predictions vs ground truth).

    Highest-confidence predictions greedily claim the unmatched ground-truth box
    with the greatest IoU at or above ``iou_threshold`` **regardless of class**;
    the matched pair's ``(gt.name, pred.name)`` is recorded (revealing cross-class
    confusion). Unmatched predictions land in the :data:`BACKGROUND` row; unmatched
    ground-truth boxes in the :data:`BACKGROUND` column.

    Args:
        predictions_by_image: ``image_key`` → predicted annotations.
        ground_truth_by_image: ``image_key`` → ground-truth annotations.
        iou_threshold: Minimum IoU for a prediction to match a ground-truth box.

    Returns:
        A :class:`ConfusionMatrix`.

    Raises:
        ValueError: ``iou_threshold`` is not in ``(0, 1]``.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")

    counts: dict[str, dict[str, int]] = {}
    seen_classes: set[str] = set()

    def record(gt_label: str, pred_label: str) -> None:
        counts.setdefault(gt_label, {})[pred_label] = (
            counts.get(gt_label, {}).get(pred_label, 0) + 1
        )
        if gt_label != BACKGROUND:
            seen_classes.add(gt_label)
        if pred_label != BACKGROUND:
            seen_classes.add(pred_label)

    image_keys = set(predictions_by_image) | set(ground_truth_by_image)
    for key in image_keys:
        preds = list(predictions_by_image.get(key, []))
        gts = list(ground_truth_by_image.get(key, []))
        _confuse_image(preds, gts, iou_threshold, record)

    return ConfusionMatrix(
        iou_threshold=iou_threshold,
        classes=sorted(seen_classes),
        _counts=counts,
    )


def _confuse_image(
    preds: list[Annotation],
    gts: list[Annotation],
    iou_threshold: float,
    record: Callable[[str, str], None],
) -> None:
    """Class-agnostic greedy IoU match for one image, recording (gt, pred) pairs."""
    pred_boxes: list[tuple[Annotation, BoundingBox]] = [
        (p, box) for p in preds if (box := annotation_bbox(p)) is not None
    ]
    gt_boxes: list[tuple[Annotation, BoundingBox]] = [
        (g, box) for g in gts if (box := annotation_bbox(g)) is not None
    ]

    matched_gt: set[int] = set()
    for pred, pbox in sorted(pred_boxes, key=lambda pb: _confidence(pb[0]), reverse=True):
        best_iou = iou_threshold
        best_idx = -1
        for idx, (_gt, gbox) in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = bbox_iou(pbox, gbox)
            if iou >= best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0:
            matched_gt.add(best_idx)
            record(gt_boxes[best_idx][0].name, pred.name)
        else:
            record(BACKGROUND, pred.name)  # false positive

    for idx, (gt, _gbox) in enumerate(gt_boxes):
        if idx not in matched_gt:
            record(gt.name, BACKGROUND)  # false negative
