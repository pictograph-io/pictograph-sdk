"""Dataset Health / Insights models.

Describe what ``GET /developer/datasets/by-name/{name}/insights`` returns.
Response models use ``extra="ignore"`` so a new backend metric doesn't break
older SDK versions.

Every metric is aggregated server-side over the denormalized columns
(annotation-class/type counts, status, dimensions, file size) - the SDK never
scans annotations, so ``client.datasets.insights(name)`` is a single fast call
even for 100k+ image datasets. Non-archived images only.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class InsightsStatusCounts(BaseModel):
    """Image counts by labeling stage (new → complete)."""

    model_config = ConfigDict(extra="ignore")

    new: int = 0
    annotate: int = 0
    review: int = 0
    complete: int = 0


class InsightsSize(BaseModel):
    """One distinct image size and how many images share it."""

    model_config = ConfigDict(extra="ignore")

    w: int
    h: int
    count: int


class InsightsOrientation(BaseModel):
    """Image counts by orientation."""

    model_config = ConfigDict(extra="ignore")

    landscape: int = 0
    portrait: int = 0
    square: int = 0


class InsightsDimensions(BaseModel):
    """Image-dimension insights (ranges, orientation split, size cloud)."""

    model_config = ConfigDict(extra="ignore")

    min_width: int | None = None
    max_width: int | None = None
    avg_width: int | None = None
    min_height: int | None = None
    max_height: int | None = None
    avg_height: int | None = None
    orientation: InsightsOrientation = Field(default_factory=InsightsOrientation)
    #: Top distinct (w, h) sizes by frequency (bounded to 200 server-side).
    sizes: list[InsightsSize] = Field(default_factory=list)
    distinct_size_count: int = 0
    images_with_dimensions: int = 0
    images_missing_dimensions: int = 0


class ModelConfidenceBuckets(BaseModel):
    """Image counts by model-confidence band (active-learning review buckets)."""

    model_config = ConfigDict(extra="ignore")

    lt50: int = 0  #: confidence < 50%
    b50_70: int = 0  #: 50% <= confidence < 70%
    b70_90: int = 0  #: 70% <= confidence < 90%
    b90_100: int = 0  #: 90% <= confidence < 100%


class ModelConfidence(BaseModel):
    """Active-learning rollup: how many images carry a low-confidence model
    prediction (min annotation confidence < 1.0), and how uncertain they are. Use
    it to size a human-review queue, then page it with
    ``client.images.list(dataset_id, min_confidence_lt=...)``."""

    model_config = ConfigDict(extra="ignore")

    flagged: int = 0  #: images with a prediction below 100% confidence
    lowest: float | None = None  #: the single lowest min-confidence among flagged
    avg_flagged: float | None = None  #: mean min-confidence among flagged
    buckets: ModelConfidenceBuckets = Field(default_factory=ModelConfidenceBuckets)


class DatasetInsights(BaseModel):
    """Dataset Health / Insights - headline totals, class balance, and more.

    Attributes:
        total_images: Non-archived image count.
        total_annotations: Sum of per-image annotation counts.
        annotated_images: Images with at least one annotation.
        unannotated_images: Images with zero annotations.
        avg_annotations_per_image: Mean annotations per image.
        total_bytes: Sum of original-image file sizes.
        status_counts: Image counts by labeling stage.
        class_annotation_counts: Per-class annotation-instance totals (balance).
        class_image_counts: Per-class image counts (images containing the class).
        type_counts: Per annotation-type instance totals (bbox/polygon/...).
        annotation_density: Histogram of annotations-per-image buckets.
        dimensions: Image-dimension insights.
        model_confidence: Active-learning model-uncertainty rollup (``None``
            on older backends / datasets with no model predictions).
    """

    model_config = ConfigDict(extra="ignore")

    total_images: int = 0
    total_annotations: int = 0
    annotated_images: int = 0
    unannotated_images: int = 0
    avg_annotations_per_image: float = 0.0
    total_bytes: int = 0
    status_counts: InsightsStatusCounts = Field(default_factory=InsightsStatusCounts)
    class_annotation_counts: dict[str, int] = Field(default_factory=dict)
    class_image_counts: dict[str, int] = Field(default_factory=dict)
    type_counts: dict[str, int] = Field(default_factory=dict)
    annotation_density: dict[str, int] = Field(default_factory=dict)
    dimensions: InsightsDimensions = Field(default_factory=InsightsDimensions)
    #: Active-learning: model-uncertainty rollup (None if the dataset has no
    #: model predictions, or on a backend predating the field).
    model_confidence: ModelConfidence | None = None
