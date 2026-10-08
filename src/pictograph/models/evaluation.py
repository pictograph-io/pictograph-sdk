"""Model-evaluation Pydantic models - server-side model diagnostics.

A :class:`ModelEvaluation` scores a trained model against a labeled dataset's
ground truth and returns per-class + overall metrics, a confusion matrix, and the
best/worst-performing images. Ground truth is never mutated. Created via
:meth:`pictograph.resources.model_evaluations.ModelEvaluations.create`.

The row shape is shared across model families and read by the metric present:
detection / instance-seg carry IoU-matched P/R/F1 + ``ap`` / ``map`` (mirroring the
offline :mod:`pictograph.metrics` exactly); classification carries top-1
``accuracy``; semantic-seg (B461) carries pixel ``miou`` / ``dice`` /
``pixel_accuracy`` + the ``seg_metrics`` payload. ``miou`` / ``accuracy`` / ``map``
being non-None is how a reader tells the families apart.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

EvaluationStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

#: Pseudo-label in the confusion matrix for an unmatched prediction (row = false
#: positive) or an unmatched ground-truth box (column = false negative).
BACKGROUND = "__background__"


class EvalClassMetrics(BaseModel):
    """Per-class metrics. Detection/instance-seg carry ``ap``; semantic-seg
    carries pixel ``iou`` / ``dice`` (B461)."""

    model_config = ConfigDict(extra="ignore")

    class_name: str
    tp: int
    fp: int
    fn: int
    support: int
    precision: float
    recall: float
    f1: float
    #: Average precision at the eval's IoU threshold (per-class mAP building block).
    ap: float = 0.0
    #: Semantic-seg only: pixel IoU (Jaccard) for this class (``None`` otherwise).
    iou: float | None = None
    #: Semantic-seg only: pixel Dice (== pixel F1) for this class (``None`` otherwise).
    dice: float | None = None


class EvalOverallMetrics(BaseModel):
    """Micro-averaged detection metrics pooled across all classes."""

    model_config = ConfigDict(extra="ignore")

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    macro_f1: float
    #: mAP at the eval's IoU threshold (mean of per-class :attr:`EvalClassMetrics.ap`).
    #: ``None`` for classification/semseg evals (mAP is not applicable). For
    #: classification, precision/recall/f1 are MACRO-averaged and :attr:`accuracy`
    #: is the headline; for semseg they are MICRO pixel P/R/F1 and :attr:`miou` is.
    map: float | None = 0.0
    #: Top-1 accuracy - classification evals only (``None`` for detection/semseg).
    accuracy: float | None = None
    #: Semantic-seg only (B461): mean IoU over foreground classes (the headline).
    #: ``None`` for detection/classification - and the semseg discriminator, just as
    #: :attr:`accuracy` is the classification one.
    miou: float | None = None
    #: Semantic-seg only: mean Dice over foreground classes (``None`` otherwise).
    mean_dice: float | None = None
    #: Semantic-seg only: global pixel accuracy (``None`` otherwise).
    pixel_accuracy: float | None = None
    #: Semantic-seg only: mean per-class pixel accuracy (``None`` otherwise).
    mean_pixel_accuracy: float | None = None
    #: Semantic-seg only: frequency-weighted IoU (``None`` otherwise).
    frequency_weighted_iou: float | None = None


class EvalConfusionMatrix(BaseModel):
    """Confusion matrix (rows = ground truth, cols = predicted). Class-agnostic +
    IoU-matched for detection; per-pixel for semantic-seg; true-vs-predicted for
    classification."""

    model_config = ConfigDict(extra="ignore")

    #: The IoU threshold the matrix was matched at - ``None`` for classification
    #: (true-vs-predicted, no IoU) and semantic-seg (per-pixel, no IoU).
    iou_threshold: float | None = None
    #: Row/column order: real classes (sorted) then :data:`BACKGROUND`.
    labels: list[str] = Field(default_factory=list)
    #: Dense matrix aligned with :attr:`labels`; ``grid[gt][pred]`` is the count.
    grid: list[list[int]] = Field(default_factory=list)


class EvalImagePerformance(BaseModel):
    """Per-image detection tallies used to rank the best- and worst-performing images."""

    model_config = ConfigDict(extra="ignore")

    image_id: str
    filename: str | None = None
    virtual_directory_path: str | None = None
    tp: int
    fp: int
    fn: int
    gt_count: int
    pred_count: int
    #: The model's predicted annotations for this image (the overlay/inference
    #: set), slimmed to render-relevant keys. None on pre-B453-S4 rows.
    predictions: list[dict[str, Any]] | None = None
    #: The ground-truth annotations for this image (for pred-vs-GT comparison).
    ground_truth: list[dict[str, Any]] | None = None
    #: Semantic-seg only (B461): this image's mIoU - the ranking reason (``None``
    #: for detection/instance-seg/classification, which rank by fp+fn).
    miou: float | None = None


#: Back-compat alias - the worst-image shape is now the shared per-image shape.
EvalWorstImage = EvalImagePerformance


class ModelEvaluation(BaseModel):
    """A model-evaluation run + its metric summary."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=(), populate_by_name=True)

    id: str
    organization_id: str
    model_id: str
    dataset_id: str = Field(validation_alias=AliasChoices("dataset_id", "project_id"))
    export_id: str | None = None  # the export defining the eval set (None on legacy rows)
    status: EvaluationStatus
    progress: int = 0
    iou_threshold: float = 0.5
    confidence_threshold: float = 0.5
    total_images: int = 0
    evaluated_images: int = 0
    failed_images: int = 0
    overall_metrics: EvalOverallMetrics | None = None
    per_class_metrics: list[EvalClassMetrics] | None = None
    confusion_matrix: EvalConfusionMatrix | None = None
    worst_images: list[EvalImagePerformance] | None = None
    #: Top best-performing images (gt_count>0, fewest fp+fn); None on pre-B453-S4 rows.
    best_images: list[EvalImagePerformance] | None = None
    #: Semantic-seg only (B461): per-class pixel IoU/Dice + the operating-threshold
    #: sweep ({thresholds, miou[], mean_dice[], pixel_accuracy[], best_miou,
    #: best_miou_threshold}). ``None`` for detection/instance-seg/classification.
    seg_metrics: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
