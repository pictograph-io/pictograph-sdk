"""Auto-annotate Pydantic models - single-prompt + batch job state.

Single-prompt responses (point / box / text) carry one or more
:class:`~pictograph.models.annotation.Annotation` instances inline. Batch
responses are job-status snapshots; agents poll until terminal status.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pictograph.models.annotation import Annotation

PromptStatus = Literal["success", "no_detection", "below_threshold"]
"""Outcome of a single SAM3 prompt.

- ``"success"``: at least one annotation returned.
- ``"no_detection"``: SAM3 found nothing matching the prompt.
- ``"below_threshold"``: detections existed but all below
  ``confidence_threshold``."""

BatchJobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
"""Lifecycle of an auto-annotate batch job."""


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

    model_config = ConfigDict(extra="ignore")

    job_id: str
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
