"""Model-evaluation Pydantic models - server-side detection diagnostics.

A :class:`ModelEvaluation` scores a trained detection / instance-segmentation
model against a labeled dataset's ground truth: per-class + overall precision /
recall / F1 and a class-agnostic confusion matrix, plus the worst-performing
images. Ground truth is never mutated. Created via
:meth:`pictograph.resources.model_evaluations.ModelEvaluations.create`.

The metric math mirrors the offline :mod:`pictograph.metrics` (``evaluate_detections`` /
``confusion_matrix``) exactly - a server run and a client run on the same data agree.
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
    """Detection metrics for one class."""

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
    map: float = 0.0


class EvalConfusionMatrix(BaseModel):
    """Class-agnostic detection confusion matrix (rows = ground truth, cols = predicted)."""

    model_config = ConfigDict(extra="ignore")

    iou_threshold: float
    #: Row/column order: real classes (sorted) then :data:`BACKGROUND`.
    labels: list[str] = Field(default_factory=list)
    #: Dense matrix aligned with :attr:`labels`; ``grid[gt][pred]`` is the count.
    grid: list[list[int]] = Field(default_factory=list)


class EvalWorstImage(BaseModel):
    """An image with the most detection errors (false positives + false negatives)."""

    model_config = ConfigDict(extra="ignore")

    image_id: str
    filename: str | None = None
    virtual_directory_path: str | None = None
    tp: int
    fp: int
    fn: int
    gt_count: int
    pred_count: int


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
    worst_images: list[EvalWorstImage] | None = None
    config: dict[str, Any] | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
