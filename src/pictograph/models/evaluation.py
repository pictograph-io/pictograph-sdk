"""Model-evaluation Pydantic models - server-side model diagnostics.

A :class:`ModelEvaluation` scores a trained model against a labeled dataset's
ground truth and returns per-class + overall metrics, a confusion matrix, and the
best/worst-performing images. Ground truth is never mutated. Created via
:meth:`pictograph.resources.model_evaluations.ModelEvaluations.create`.

The row shape is shared across model families and read by the metric present:
detection / instance-seg carry IoU-matched P/R/F1 + ``ap`` / ``map`` (mirroring the
offline :mod:`pictograph.metrics` exactly); classification carries top-1
``accuracy``; semantic-seg carries pixel ``miou`` / ``dice`` /
``pixel_accuracy`` + the ``seg_metrics`` payload. ``miou`` / ``accuracy`` / ``map``
being non-None is how a reader tells the families apart.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from pictograph.models.image import image_path

EvaluationStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

#: Pseudo-label in the confusion matrix for an unmatched prediction (row = false
#: positive) or an unmatched ground-truth box (column = false negative).
BACKGROUND = "__background__"


class EvalClassMetrics(BaseModel):
    """Per-class metrics. Detection/instance-seg carry ``ap``; semantic-seg
    carries pixel ``iou`` / ``dice``."""

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
    #: Anomaly detection: the true negatives (normal images scored normal), the AUROC and the average precision.
    tn: int | None = None
    auroc: float | None = None
    average_precision: float | None = None
    #: Semantic-seg only: mean IoU over foreground classes (the headline).
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


class EvalAnomalyAssignment(BaseModel):
    """What an anomaly model's evaluation scored against (assigned at create): the export's tags that mark a normal
    and an anomalous image, and the classes whose polygons and boxes outline its anomalies for the pixel metrics
    (empty: none)."""

    model_config = ConfigDict(extra="ignore")

    normal_tags: list[str] = Field(default_factory=list)
    anomalous_tags: list[str] = Field(default_factory=list)
    mask_classes: list[str] = Field(default_factory=list)


class EvalAnomalyImageMetrics(BaseModel):
    """An anomaly model's image-level metrics (anomalib's: the normalized score against the assigned truths)."""

    model_config = ConfigDict(extra="ignore")

    n_images: int = 0
    n_normal: int = 0
    n_anomalous: int = 0
    #: The chance an anomalous image outscores a normal one; ``None`` when either kind is missing.
    auroc: float | None = None
    average_precision: float | None = None
    #: The operating point: the image is anomalous when its score is over ``confidence``.
    confidence: float = 0.5
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    #: The best F1 over every threshold, and the confidence that reaches it - a better operating point than the
    #: calibrated 0.5 (the highest normal calibration score) when that one is too conservative.
    best_f1: float = 0.0
    best_f1_confidence: float = 0.5
    #: ``[{"fpr", "tpr"}, ...]`` from (0, 0) to (1, 1); empty without both kinds of image.
    roc_curve: list[dict[str, float]] = Field(default_factory=list)
    #: ``[{"recall", "precision"}, ...]`` from (0, 1); empty without an anomalous image.
    pr_curve: list[dict[str, float]] = Field(default_factory=list)
    score_histogram: dict[str, Any] | None = None


class EvalAnomalyPixelMetrics(BaseModel):
    """An anomaly model's pixel-level metrics: its maps against the mask classes' outlines, at the map's resolution
    (anomalib's: pixel AUROC, AUPR and AUPRO; precision, recall, F1 and IoU at the operating confidence)."""

    model_config = ConfigDict(extra="ignore")

    n_images: int = 0
    n_pixels: int = 0
    anomalous_pixels: int = 0
    #: The outlines' 8-connected regions (AUPRO averages its overlap over them).
    regions: int = 0
    auroc: float | None = None
    average_precision: float | None = None
    #: The per-region overlap's area up to ``aupro_fpr_limit`` false-positive rate, over that rate (anomalib's).
    aupro: float | None = None
    aupro_fpr_limit: float = 0.3
    confidence: float = 0.5
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    iou: float = 0.0


class EvalAnomalyTagMetrics(BaseModel):
    """One tag's row of an anomaly evaluation's breakdown (MVTec AD's per category and per defect type): the tag's
    anomalous images against its own normal images (``negatives="tag"``: a category's) or, when it is on no normal
    image, against every normal image (``"all"``: a defect type's)."""

    model_config = ConfigDict(extra="ignore")

    tag: str
    n_anomalous: int = 0
    n_normal: int = 0
    negatives: Literal["tag", "all"] = "all"
    auroc: float | None = None
    #: The share of the tag's anomalous images flagged at the operating confidence.
    recall: float = 0.0
    #: The share of its negatives flagged.
    false_alarm_rate: float | None = None


class EvalAnomalyMetrics(BaseModel):
    """An anomaly model's evaluation (``ModelEvaluation.anomaly_metrics``): its image metrics, their breakdown by tag,
    its pixel metrics when mask classes were assigned, the assignment, and the images left out (``unassigned``:
    neither group's tag; ``ambiguous``: both)."""

    model_config = ConfigDict(extra="ignore")

    assignment: EvalAnomalyAssignment = Field(default_factory=EvalAnomalyAssignment)
    image: EvalAnomalyImageMetrics = Field(default_factory=EvalAnomalyImageMetrics)
    by_tag: list[EvalAnomalyTagMetrics] = Field(default_factory=list)
    pixel: EvalAnomalyPixelMetrics | None = None
    excluded: dict[str, int] = Field(default_factory=dict)


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

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    image_id: str
    filename: str | None = None
    directory_path: str | None = Field(
        default=None,
        description="Its directory in the dataset.",
        validation_alias=AliasChoices("directory_path", "virtual_directory_path"),
    )
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
    #: Semantic-seg only: this image's mIoU - the ranking reason (``None``
    #: for detection/instance-seg/classification, which rank by fp+fn).
    miou: float | None = None
    #: Anomaly detection: the image's normalized score.
    anomaly_score: float | None = None

    @property
    def path(self) -> str | None:
        """The image's address in its dataset - pass it as ``image`` to any per-image call."""
        return image_path(self.directory_path, self.filename) if self.filename else None


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
    #: Semantic-seg only: per-class pixel IoU/Dice + the operating-threshold
    #: sweep ({thresholds, miou[], mean_dice[], pixel_accuracy[], best_miou,
    #: best_miou_threshold}). ``None`` for detection/instance-seg/classification.
    seg_metrics: dict[str, Any] | None = None
    #: Anomaly detection only: the image and pixel metrics and the assignment they were scored by.
    #: ``None`` for every other model type.
    anomaly_metrics: EvalAnomalyMetrics | None = None
    config: dict[str, Any] | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    #: The precision of the model's TensorRT engine it ran on (``"fp32"`` / ``"fp16"``). Every evaluation before
    #: 1.69.189's server ran on ONNX Runtime at fp32.
    precision: str = "fp32"
    #: The GPU it ran on (``"t4"`` / ``"l4"``).
    gpu_type: str = "t4"
    #: The model version it evaluated (``None`` on an evaluation from before evaluations ran on PRT).
    version_id: str | None = None
    #: The build of the model's TensorRT engine for this GPU and precision, while the evaluation waits on it
    #: (``{"state": "building" | "built", "models": [...]}``); ``None`` when the engine was already built.
    engine_build: dict[str, Any] | None = None


class EvaluationCompute(BaseModel):
    """What an evaluation runs on: a GPU and the precision of the model's TensorRT engine."""

    model_config = ConfigDict(extra="ignore")

    gpu: str = "t4"
    precision: str = "fp32"


class EvaluationComputeGpu(BaseModel):
    """A GPU an evaluation may run on, as
    :meth:`~pictograph.resources.model_evaluations.ModelEvaluations.compute_options` lists it."""

    model_config = ConfigDict(extra="ignore")

    gpu: str
    label: str
    #: Its memory in GB (16 on a T4); ``description`` says the same in words.
    vram_gb: int | None = None
    description: str = ""
    #: An hour of it at the Compute rate an evaluation is billed at (µUSD), as the server prices it.
    cost_per_hour_micro_usd: int = 0


class EvaluationComputePrecision(BaseModel):
    """A precision the model's TensorRT engine can be built at, with its label."""

    model_config = ConfigDict(extra="ignore")

    #: ``"fp32"`` or ``"fp16"`` - a string, so a precision added later never breaks a read.
    precision: str
    label: str
    description: str = ""


class EvaluationComputeOptions(BaseModel):
    """What an evaluation may run on: each GPU with its memory and hourly price, each precision, the default."""

    model_config = ConfigDict(extra="ignore")

    gpus: list[EvaluationComputeGpu] = Field(default_factory=list)
    precisions: list[EvaluationComputePrecision] = Field(default_factory=list)
    default: EvaluationCompute = Field(default_factory=EvaluationCompute)
