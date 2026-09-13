"""TrainingRun Pydantic model - represents a single GPU training job.

Returned by every method on :class:`pictograph.resources.training.Training`.
Response-side schema (``extra="ignore"``) so backend column additions don't
break callers.

Lifecycle states (matches the database CHECK constraint):

============   ====================================================
``status``     Meaning
============   ====================================================
``pending``    Just created, not yet submitted to the training service.
``queued``     Submitted, waiting on a GPU.
``running``    A worker is actively training.
``completed``  Finished successfully - model record is created.
``failed``     Stopped with an error; ``error_message`` is populated.
``cancelled``  User aborted via :meth:`Training.cancel`.
============   ====================================================
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PipelineType = Literal[
    "yolox",
    "sm_pytorch",
    "classification",
    "rfdetr_detection",
    "rfdetr_segmentation",
    "rfdetr_keypoint",
]
"""Training pipeline. Each selects a different server-side trainer."""

# "auto" resolves server-side at create time to the cheapest tier
# whose VRAM fits the config's predicted peak; runs always REPORT a concrete
# tier (the resolution decision is instrumented in config["gpu_autoselect"]).
GpuType = Literal["a10g", "a100", "h100", "auto"]
"""GPU tier. Higher tiers cost more credits per minute (see ``client.credits.estimate``)."""

TrainingStatus = Literal["pending", "queued", "running", "completed", "failed", "cancelled"]


class TrainingRun(BaseModel):
    """A single training job."""

    model_config = ConfigDict(extra="ignore")

    id: str
    organization_id: str
    name: str
    dataset_id: str | None = Field(default=None, description="Dataset this run was trained on.")
    export_id: str | None = Field(default=None, description="Export ZIP this run consumed.")
    model_id: str | None = Field(
        default=None,
        description=("Resulting model. Populated when ``status == 'completed'``."),
    )
    pipeline_type: PipelineType
    gpu_type: GpuType | None = None
    status: TrainingStatus
    progress: int = Field(default=0, ge=0, le=100, description="0-100 percent.")
    current_epoch: int = 0
    total_epochs: int | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    eta_seconds: int | None = None
    training_time_seconds: int | None = Field(
        default=None,
        description="Actual GPU minutes used. Populated on completion.",
    )
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime
    created_by: str | None = None
