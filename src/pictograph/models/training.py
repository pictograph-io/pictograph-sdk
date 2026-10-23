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


class TrainingRunWarning(BaseModel):
    """Something a training run's prep left out of the model, and why.

    Today one ``code``: ``"classes_not_trained"`` - a requested class the run's TRAIN
    split did not supervise is not a class of the model (it would have had an output the
    model was never taught, which RF-DETR fills with the pretrained COCO class at that
    index). ``classes`` names each one with its ``reason``: ``"no_training_data"`` (no
    annotations of the kind this pipeline trains on anywhere in the export) or
    ``"not_in_train_split"`` (annotated only in the validation or test split you assigned).
    """

    model_config = ConfigDict(extra="ignore")

    code: str
    message: str = Field(description="The sentence the app shows for this warning.")
    classes: list[dict[str, Any]] = Field(default_factory=list)
    unit: str | None = Field(
        default=None,
        description='What the per-split counts are in: "annotations", "pixels" or "images".',
    )


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
    warnings: list[TrainingRunWarning] | None = Field(
        default=None,
        description=(
            "What the prep left out of the model and why (e.g. a class the train split "
            "did not supervise). ``None`` when nothing was left out; the model's "
            "``class_mapping`` lists only the classes it trained on."
        ),
    )
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime
    created_by: str | None = None


# ─────────────────────────── the quote (B533 #10) ───────────────────────────


class TrainingClassSupportEntry(BaseModel):
    """One class of a run, counted the way the trainer will count it in this export."""

    model_config = ConfigDict(extra="ignore")

    name: str
    support: int = Field(
        default=0, description="Units (``unit``) of this class across the whole export."
    )
    train: int | None = Field(
        default=None,
        description="Units in the train split - only when the images' assigned split is used.",
    )
    val: int | None = None
    test: int | None = None
    trainable: bool = Field(
        description="Whether the run would train it (a class the train split does not "
        "supervise is left out of the model, or refused for keypoints)."
    )
    reason: str | None = Field(
        default=None,
        description='When not trainable: "no_training_data" or "not_in_train_split".',
    )


class TrainingClassSupport(BaseModel):
    """How much of each class a run would train on - what the training modal shows."""

    model_config = ConfigDict(extra="ignore")

    pipeline_type: str
    unit: str = Field(description='"annotations", or "images" for classification.')
    trains_on: str = Field(description="What this pipeline trains on, in words.")
    split: str = Field(
        description='"assigned" (the images\' own train/val/test) or "random" (the trainer '
        "makes one)."
    )
    images: int = Field(default=0, description="The export's images that still exist.")
    export_images: int | None = Field(default=None, description="The images the export names.")
    classes: list[TrainingClassSupportEntry] = Field(default_factory=list)
    untrainable: list[str] = Field(default_factory=list)
    multi_tag_images: int = Field(
        default=0,
        description="Classification: images with more than one selected tag (dropped).",
    )


class TrainingQuoteTier(BaseModel):
    """One GPU tier a run can use, priced, and whether the config's predicted peak fits."""

    model_config = ConfigDict(extra="ignore")

    gpu_type: str
    total_mib: int
    fits: bool | None = Field(
        default=None,
        description="``None`` when the pipeline has no memory estimate (YOLOX).",
    )
    micro_usd_per_minute: int


class TrainingQuoteEstimate(BaseModel):
    """The estimate the training modal shows - an estimate, not a price."""

    model_config = ConfigDict(extra="ignore")

    minutes: int
    micro_usd_per_minute: int
    total_micro_usd: int
    note: str = ""


class TrainingQuoteBalance(BaseModel):
    """The org's spendable balance against what starting a run needs."""

    model_config = ConfigDict(extra="ignore")

    budget_remaining_micro_usd: int
    required_to_start_micro_usd: int
    covers_estimate: bool


class TrainingQuoteBlock(BaseModel):
    """Something that keeps a valid run from starting."""

    model_config = ConfigDict(extra="ignore")

    code: str = Field(description='"insufficient_credits", "forbidden" or "key_without_creator".')
    message: str


class TrainingQuote(BaseModel):
    """A training run checked and priced without starting it (:meth:`Training.quote`).

    The quote runs every check :meth:`Training.create` runs on the same arguments, so a
    quote that comes back describes a run ``create`` would accept - apart from what
    ``blocking`` names. Nothing is written and nothing is charged.
    """

    model_config = ConfigDict(extra="ignore")

    dataset_id: str
    dataset_name: str
    export_id: str
    export_name: str
    image_count: int | None = None
    annotation_count: int | None = None
    pipeline_type: PipelineType
    requested_gpu_type: str = Field(description='What was asked for, ``"auto"`` included.')
    gpu_type: str = Field(description="The tier the run would use.")
    fits: bool | None = Field(
        default=None,
        description="Whether the config's predicted peak memory fits ``gpu_type``; "
        "``None`` when the pipeline has no estimate.",
    )
    recommended_gpu_type: str | None = Field(
        default=None, description='What ``gpu_type="auto"`` resolves to for this config.'
    )
    gpu_autoselect: dict[str, Any] | None = Field(
        default=None, description="The auto-select decision behind ``recommended_gpu_type``."
    )
    tiers: list[TrainingQuoteTier] = Field(default_factory=list)
    epochs: int
    classes: list[str] = Field(default_factory=list, description="The classes it would train.")
    class_support: TrainingClassSupport | None = None
    deleted_images: int | None = Field(
        default=None,
        description="Images the export names that were deleted since it was taken.",
    )
    estimate: TrainingQuoteEstimate
    balance: TrainingQuoteBalance
    can_start: bool
    blocking: list[TrainingQuoteBlock] = Field(default_factory=list)
    config: dict[str, Any] = Field(
        default_factory=dict, description="The config the run would be stored with."
    )
