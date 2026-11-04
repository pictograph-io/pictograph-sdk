"""Auto-annotate Pydantic models - single-prompt + batch job state.

Single-prompt responses (point / box / text) carry one or more
:class:`~pictograph.models.annotation.Annotation` instances inline. Batch
responses are job-status snapshots; agents poll until terminal status.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from pictograph.models.annotation import Annotation

PromptStatus = Literal["success", "no_detection", "below_threshold"]
"""Outcome of a single SAM3 prompt.

- ``"success"``: at least one annotation returned.
- ``"no_detection"``: SAM3 found nothing matching the prompt.
- ``"below_threshold"``: detections existed but all below
  ``confidence_threshold``."""

BatchJobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
"""Lifecycle of an auto-annotate batch job."""

SimilarScope = Literal["directory", "dataset"]
"""Which images "Annotate similar" labels: the other images in the example image's
folder (``directory``), or every other image in the dataset (``dataset``)."""

SimilarOutputType = Literal["polygon", "bbox", "obb"]
"""How "Annotate similar" writes each match: a polygon (needs a polygon class), a box,
or a rotated box (``obb``; both need a box class)."""


class PromptResult(BaseModel):
    """Outcome of a single SAM3 prompt (point / box / text).

    Box and text prompts can return zero or multiple annotations; point
    prompts always return one (or ``status != "success"``).
    """

    model_config = ConfigDict(extra="ignore")

    status: PromptStatus
    annotations: list[Annotation] = Field(default_factory=list)
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for the primary detection (point/box only).",
    )
    inference_time: float | None = Field(
        default=None,
        ge=0.0,
        description="GPU inference time in seconds (informational only).",
    )


class BatchJob(BaseModel):
    """Snapshot of an auto-annotate batch job's progress.

    Returned by :meth:`pictograph.resources.auto_annotate.AutoAnnotate.batch`
    (kicker) and by :meth:`AutoAnnotate.get_batch` (polling).
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # The API sends ``job_id`` and also ``job`` (the path's vocabulary). Builds
    # from 2026-08-01 to 2026-10-02 sent ONLY ``job``, which this model required
    # as ``job_id`` - so batch() / get_batch() / cancel_batch() raised on every call.
    job_id: str = Field(validation_alias=AliasChoices("job_id", "job"))
    status: BatchJobStatus
    progress: int = Field(
        default=0,
        ge=0,
        le=100,
        description="0-100. 100 only at terminal state.",
    )
    total_images: int = Field(default=0, ge=0)
    processed_images: int = Field(default=0, ge=0)
    total_annotations_added: int = Field(default=0, ge=0)
    failed_images: int = Field(default=0, ge=0)
    error_message: str | None = None
    estimated_credits: int | None = Field(
        default=None,
        description="Set on the kicker response only - fixed pre-charge amount.",
    )
    completed_at: datetime | None = None


class BatchClass(BaseModel):
    """Class config for a batch job - one entry per class to detect."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    output_type: Literal["polygon", "bbox", "tag"] = "polygon"
    """``polygon``/``bbox`` for SAM3, ``tag`` for classification models."""


class ProjectedImages(BaseModel):
    """Images that don't exist yet - an upload in flight, or a video's future frames.

    Lets :meth:`AutoAnnotate.quote` price work BEFORE it is created. For a video,
    ``count`` is ``floor(duration_seconds * sample_fps)`` - exactly what
    :meth:`Video.extract_frames` will produce.
    """

    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    width: int | None = Field(
        default=None,
        ge=1,
        description=(
            "The group's pixel width, when known - a video's frames are all its own "
            "resolution, which :meth:`Video.probe` reports. Supply it and SAHI tiles are "
            "priced from the real size; omit it and the server falls back to the same "
            "assumed size it uses for a stored image with no recorded dimensions."
        ),
    )
    height: int | None = Field(default=None, ge=1)


class BatchQuote(BaseModel):
    """What a batch job WOULD cost - the same deposit ``batch()`` would take.

    The estimate and the charge come from one function on the server, so this is not an
    approximation of the price: it is the price.
    """

    model_config = ConfigDict(extra="ignore")

    total_images: int
    estimated_credits: int = Field(
        description="Cost in micro-USD (1 USD = 1_000_000). The final customer price."
    )
    sahi_tiles: int = Field(default=0, description="Total SAHI tile passes priced.")
    containers: int = Field(default=0, description="T4 containers the job will fan out over.")
    remaining_credits: int = Field(default=0, description="Your remaining balance (µUSD).")
    sufficient: bool = Field(default=True, description="Whether the balance covers the estimate.")
    max_images: int = Field(default=5000, description="Image ceiling for a single job.")
    exceeds_max_images: bool = Field(
        default=False,
        description="True when this configuration is past the ceiling and must be split.",
    )


class SimilarQuote(BaseModel):
    """What an "Annotate similar" job WOULD run and cost - every check
    :meth:`AutoAnnotate.similar` makes (the examples, the target class, whether SAM3 reads
    the example image, the images in scope) and the estimate its start is gated on.

    The job itself is charged its measured GPU time once, when it completes; a failed or
    cancelled job is free.
    """

    model_config = ConfigDict(extra="ignore")

    total_images: int = Field(description="Images the job would label (at most ``max_images``).")
    total_in_scope: int = Field(default=0, description="Other images in the scope, before the cap.")
    capped: bool = Field(
        default=False, description="True when the scope holds more than ``max_images``."
    )
    max_images: int = Field(default=2000, description="Image ceiling for a single job.")
    estimated_credits: int = Field(
        description="Estimate in micro-USD (1 USD = 1_000_000). The start is gated on it."
    )
    containers: int = Field(default=0, description="T4 containers the job would fan out over.")
    remaining_credits: int = Field(default=0, description="Your remaining balance (µUSD).")
    sufficient: bool = Field(default=True, description="Whether the balance covers the estimate.")
    examples: int = Field(default=0, description="Example polygons the job would learn from.")
    target_class: str = Field(default="", description="The class each match is labelled as.")
    output_type: str = Field(default="polygon", description="polygon, bbox or obb.")
    scope: str = Field(default="directory", description="directory or dataset.")
    directory: str = Field(default="/", description="The example image's folder.")
