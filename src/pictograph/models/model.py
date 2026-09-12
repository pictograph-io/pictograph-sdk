"""Model Pydantic model - represents a trained CV model.

Returned by every method on :class:`pictograph.resources.models.Models`.
Created server-side by the training pipeline's completion callback -
SDK callers don't insert ``Model`` rows directly. Instead they
:meth:`Training.create` a training run and the resulting model is exposed
via this resource.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ModelType = Literal[
    "object_detection",
    "semantic_segmentation",
    "instance_segmentation",
    "classification",
    "keypoint_detection",
]
"""Output category. Pinned by the database CHECK constraint."""

ModelStatus = Literal["training", "ready", "failed", "archived"]
"""Lifecycle. Only ``ready`` models are downloadable."""

ModelVisibility = Literal["private", "public"]
"""``private`` models are org-scoped; ``public`` are visible across orgs."""


class Model(BaseModel):
    """A trained computer vision model."""

    model_config = ConfigDict(extra="ignore")

    id: str
    organization_id: str
    name: str
    description: str | None = None
    model_type: ModelType
    architecture: str | None = Field(
        default=None,
        description="Backbone family (e.g., 'yolox-s', 'rfdetr-base').",
    )
    visibility: ModelVisibility
    status: ModelStatus
    metrics: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Training metrics (mAP, precision, recall, etc.). Populated when ``status == 'ready'``."
        ),
    )
    class_mapping: dict[str, Any] | None = Field(
        default=None,
        description='Class list for inference, as ``{"classes": [name, ...]}`` in label order.',
    )
    training_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Training hyperparameters, including the model's input ``image_height`` / "
            "``image_width``. Used to configure local inference (``pictograph.inference``)."
        ),
    )
    version: str = Field(default="1.0.0")
    parent_model_id: str | None = Field(
        default=None,
        description="When set, this model was fine-tuned from another.",
    )
    forked_from_model_id: str | None = Field(
        default=None,
        description=(
            "When set, this model was imported (forked) from a public model. "
            "Its weights reference the source model's storage path."
        ),
    )
    precision: Literal["fp32", "fp16"] = Field(
        default="fp32",
        description=(
            "Numeric precision of the exported ONNX weights. 'fp16' models carry "
            "half-precision weights with fp32 inputs/outputs - a drop-in for the "
            "fp32 serving path (~2x smaller, faster on GPU)."
        ),
    )
    created_at: datetime
    updated_at: datetime


class ModelPredictResult(BaseModel):
    """Result of a remote single-image test inference (``models.predict``).

    ``annotations`` are Pictograph annotation dicts (``name``/``type`` +
    geometry + ``confidence``) exactly as the inference service emits them;
    ``tags`` is populated for classification models instead.
    """

    model_config = ConfigDict(extra="ignore")

    success: bool = True
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    #: Softmax scores INDEX-PARALLEL to :attr:`tags` (classification only).
    #: Empty for other model types, and empty against a backend that predates
    #: the field - so zip defensively rather than assuming equal length::
    #:
    #:     for name, score in zip(r.tags, r.tag_scores):
    #:         print(name, score)
    #:
    #: :attr:`tags` deliberately stays a list of NAMES for wire-compat.
    tag_scores: list[float] = Field(default_factory=list)
    model_type: ModelType | None = None
    inference_seconds: float = 0.0


ModelFileKind = Literal["weights", "config", "license", "readme"]
"""Artifact kinds in a model's file manifest. ``weights`` covers the
ONNX / PyTorch (and, later, safetensors) exports; ``config`` is the immutable
``config.json`` reproducibility artifact written at training completion;
``license`` / ``readme`` are generated from the model's current metadata at
request time."""


class ModelVersionEntry(BaseModel):
    """One version in a model's file manifest (``models.files``)."""

    model_config = ConfigDict(extra="ignore")

    version_id: str
    version_number: int | None = None
    version_label: str | None = None
    created_at: datetime | None = None
    status: str | None = None
    #: The model's CURRENT version - the RESOLVED effective pointer (the
    #: owner-promoted pin first, else the newest ready), never merely
    #: the highest number.
    is_latest: bool = False
    #: Alias of ``is_latest`` (the ``/versions`` contract's name).
    is_current: bool = False
    precision: str = "fp32"
    #: The run that produced this version (lineage; null on the synthetic
    #: pre-versioning fallback entry).
    training_run_id: str | None = None
    #: The per-version subject: architecture, final metrics, and the
    #: source export this version was trained from.
    architecture: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    export_name: str | None = None
    #: WHICH PIPELINE BUILD produced this version's weights, so "what
    #: built this?" is answerable without reading training logs. Two halves:
    #: DECLARED (``build_id``, a digest over the training service's whole source
    #: closure, plus ``prep_layout_version``) and MEASURED (``libraries``, the
    #: versions actually resolved inside the training container - the half that
    #: catches a dependency drifting under an unchanged pin).
    #:
    #: ``None`` is a REAL answer meaning "trained before provenance was
    #: recorded; the build is not knowable" - it is never a default standing in
    #: for a value, and was deliberately not backfilled. Treat it as unknown,
    #: not as "same as everything else".
    pipeline_provenance: dict[str, Any] | None = None


class ModelFileEntry(BaseModel):
    """One downloadable artifact in a model's file manifest.

    The five fields after ``updated_at`` are an additive block, present on EVERY
    row. They exist because a derived artifact (``.pte`` / ``.engine``) is not
    interchangeable the way ``.onnx`` is: an engine is valid for exactly one GPU
    architecture, one TensorRT version and one precision, so a manifest that only
    said "there is an engine" would be describing a file the caller may well be

    Older backends do not send them; the model tolerates that (``extra="ignore"``
    plus defaults), so a new SDK against an older API degrades to "unknown binding"
    rather than failing to parse.
    """

    model_config = ConfigDict(extra="ignore")

    #: The version this file belongs to (joins :class:`ModelVersionEntry`).
    version_id: str
    name: str
    kind: str
    format: str
    size_bytes: int | None = None
    content_type: str | None = None
    updated_at: datetime | None = None

    #: Which runtime executes this file - ``onnxruntime`` / ``pytorch`` /
    #: ``executorch`` / ``tensorrt``. ``None`` for ``config.json``, which is not
    #: executable by anything.
    runtime: str | None = None
    #: ``fp32`` / ``fp16``. fp16 means fp16 weights with fp32 I/O, so an fp16
    #: artifact is a drop-in for the fp32 serving path.
    precision: str | None = None
    #: What the artifact is BOUND to: the SM capability (``sm75``…) for an engine,
    #: the lowering backend (``xnnpack``…) for a ``.pte``. ``None`` for the
    #: device-independent formats.
    target_key: str | None = None
    #: The builder that produced it - ``trt-10.13.3.9`` / ``executorch-1.0.1``.
    toolchain_version: str | None = None
    #: Whether the platform's pinned toolchain has moved on. **Blocking for an
    #: engine** (a plan from a different TensorRT minor will not deserialize at all)
    #: and **advisory for a ``.pte``** (a newer runtime loads an older program).
    stale: bool = False
    #: ``model_artifacts.id``. ``None`` for the artifacts stored in the 1:1 columns
    #: on the version row rather than in the child table.
    artifact_id: str | None = None


class ModelFileManifest(BaseModel):
    """A model's complete version + file manifest (``models.files``).

    ``versions`` is never empty - a model trained before versioning reports
    one synthetic version whose id equals the model id. Every file carries a
    ``version_id``; new artifact kinds appear as MORE ROWS, never new fields.
    """

    model_config = ConfigDict(extra="ignore")

    versions: list[ModelVersionEntry] = Field(default_factory=list)
    files: list[ModelFileEntry] = Field(default_factory=list)
    #: Non-null iff the owner promoted (pinned) a version; new
    #: trainings then land as versions without going live until promoted.
    pinned_version_id: str | None = None


class ModelVersionsPayload(BaseModel):
    """``models.versions`` - the version list plus promote state.

    ``current_version_id`` is the RESOLVED effective version (the owner-
    promoted pin first, else the newest ready). ``pinned_version_id`` is the
    raw pin (``None`` = the model follows ``latest_version_id``).
    """

    model_config = ConfigDict(extra="ignore")

    versions: list[ModelVersionEntry] = Field(default_factory=list)
    current_version_id: str | None = None
    pinned_version_id: str | None = None
    latest_version_id: str | None = None
