"""Pictograph - agent-native computer vision annotation SDK.

Public surface:

- :class:`Client` - entry point for all API operations.
- Annotation Pydantic models (:class:`BBoxAnnotation`, :class:`PolygonAnnotation`,
  :class:`PolylineAnnotation`, :class:`KeypointAnnotation`) and the
  discriminated :data:`Annotation` union.
- Geometry primitives (:class:`Point`, :class:`BoundingBox`,
  :class:`PolygonGeometry`, :class:`PolylineGeometry`).
- Resource models - :class:`Dataset`, :class:`Image`, :class:`Export`,
  :class:`TrainingRun`, :class:`Model`, :class:`CreditBalance`, etc.
- Exception hierarchy rooted at :class:`PictographError`.

Anything not re-exported here is private and may change without notice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pictograph._version import __version__
from pictograph.augment import Augmenter
from pictograph.client import Client
from pictograph.exceptions import (
    ApiError,
    AuthError,
    ConfigurationError,
    ConflictError,
    ForbiddenError,
    NetworkError,
    NotFoundError,
    PaymentRequiredError,
    PictographError,
    PollTimeoutError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
    ValidationError,
)
from pictograph.inference import (
    DEVICES,
    RUNTIMES,
    AnyModel,
    AnyResult,
    ClassificationModel,
    ClassificationResult,
    ClassScore,
    DetectionModel,
    DetectionResult,
    Device,
    InferenceModel,
    InferenceResult,
    InstanceSegmentationModel,
    InstanceSegmentationResult,
    KeypointModel,
    KeypointResult,
    Runtime,
    SemanticSegmentationModel,
    SemanticSegmentationResult,
    TaskName,
    WeightFormat,
    get_model,
    load_model,
)
from pictograph.models.annotation import (
    Annotation,
    AnnotationType,
    BBoxAnnotation,
    KeypointAnnotation,
    OrientedBoxGeometry,
    PolygonAnnotation,
    PolygonGeometry,
    PolylineAnnotation,
    PolylineGeometry,
)
from pictograph.models.annotation_comment import (
    AnnotationComment,
    AnnotationCommentReaction,
)
from pictograph.models.api_key import ApiKey, ApiKeyRole, CreatedApiKey
from pictograph.models.auto_annotate import (
    BatchClass,
    BatchJob,
    BatchJobStatus,
    BatchQuote,
    ProjectedImages,
    PromptResult,
    PromptStatus,
)
from pictograph.models.batch import BatchFailure, BatchResult, DuplicateHandling
from pictograph.models.common import (
    BoundingBox,
    BulkActionResult,
    BulkDeleteResult,
    NonBlankStr,
    Point,
)
from pictograph.models.connector import (
    ConnectorProvider,
    DatasetImportProgress,
    DatasetImportStatus,
    ImportJob,
    ImportStatus,
    LimitCheckResult,
    LimitType,
    RemoteDataset,
    ValidationResult,
)
from pictograph.models.credit import CreditBalance, CreditEstimate, CreditLedgerEntry
from pictograph.models.dataset import (
    AttributeInputType,
    ClassAttribute,
    Dataset,
    DatasetAnnotationType,
    DatasetClass,
    DatasetImage,
    DatasetRestoreEstimate,
    DatasetStorageStatus,
    DatasetStorageTransition,
)
from pictograph.models.deployment import (
    ComputeType,
    CreatedDeployment,
    Deployment,
    DeploymentComputeOption,
    DeploymentGpuType,
    DeploymentQuote,
    DeploymentStatus,
)
from pictograph.models.directory import Directory, DirectoryStats, DirectoryTreeNode
from pictograph.models.evaluation import (
    BACKGROUND,
    EvalClassMetrics,
    EvalConfusionMatrix,
    EvalImagePerformance,
    EvalOverallMetrics,
    EvaluationStatus,
    EvalWorstImage,
    ModelEvaluation,
)
from pictograph.models.export import Export, ExportFormat, ExportStatus
from pictograph.models.image import Image, ImageStatus
from pictograph.models.insights import (
    DatasetInsights,
    InsightsDimensions,
    InsightsOrientation,
    InsightsSize,
    InsightsStatusCounts,
    ModelConfidence,
    ModelConfidenceBuckets,
)
from pictograph.models.model import (
    Model,
    ModelFileEntry,
    ModelFileManifest,
    ModelPredictResult,
    ModelStatus,
    ModelType,
    ModelVersionEntry,
    ModelVersionsPayload,
    ModelVisibility,
)
from pictograph.models.near_duplicates import NearDuplicatesResult
from pictograph.models.notification import Notification
from pictograph.models.organization import (
    InviteRole,
    InviteStatus,
    Organization,
    OrganizationInvite,
    OrganizationMember,
    OrganizationRole,
    SubscriptionTier,
)
from pictograph.models.search import SimilarImage, TaggedImage
from pictograph.models.task import Task, TaskContribution, TaskContributions
from pictograph.models.training import (
    GpuType,
    PipelineType,
    TrainingRun,
    TrainingStatus,
)
from pictograph.models.video import (
    VideoExtractionJob,
    VideoJobStatus,
    VideoMetadata,
    VideoUploadInfo,
)
from pictograph.models.webhook import (
    CreatedWebhookEndpoint,
    WebhookDelivery,
    WebhookDeliveryStatus,
    WebhookEndpoint,
)
from pictograph.models.workflow import (
    Workflow,
    WorkflowRun,
    WorkflowRunCreated,
    WorkflowRunStatus,
    WorkflowStatus,
)
from pictograph.resources._deployment_client import DeploymentClient
from pictograph.resources.annotations import (
    AnnotationImportFailure,
    AnnotationImportReport,
)
from pictograph.resources.auto_annotate import (
    AnnotateMode,
    AnnotateReport,
    AnnotationFailure,
)
from pictograph.resources.images import (
    AugmentFailure,
    AugmentReport,
    TileFailure,
    TileReport,
    UploadFailure,
    UploadReport,
)
from pictograph.viz import DEFAULT_PALETTE, draw_annotations

__all__ = [
    "BACKGROUND",
    "DEFAULT_PALETTE",
    "DEVICES",
    "REGISTRY",
    "RUNTIMES",
    "AnnotateMode",
    "AnnotateReport",
    "Annotation",
    "AnnotationComment",
    "AnnotationCommentReaction",
    "AnnotationFailure",
    "AnnotationImportFailure",
    "AnnotationImportReport",
    "AnnotationType",
    "AnyModel",
    "AnyResult",
    "ApiError",
    "ApiKey",
    "ApiKeyRole",
    "AsyncClient",
    "AttributeInputType",
    "AugmentFailure",
    "AugmentReport",
    "Augmenter",
    "AuthError",
    "BBoxAnnotation",
    "BatchClass",
    "BatchFailure",
    "BatchJob",
    "BatchJobStatus",
    "BatchQuote",
    "BatchResult",
    "BoundingBox",
    "BulkActionResult",
    "BulkDeleteResult",
    "ClassAttribute",
    "ClassScore",
    "ClassificationModel",
    "ClassificationResult",
    "Client",
    "ComputeType",
    "ConfigurationError",
    "ConflictError",
    "ConnectorProvider",
    "CreatedApiKey",
    "CreatedDeployment",
    "CreatedWebhookEndpoint",
    "CreditBalance",
    "CreditEstimate",
    "CreditLedgerEntry",
    "Dataset",
    "DatasetAnnotationType",
    "DatasetClass",
    "DatasetImage",
    "DatasetImportProgress",
    "DatasetImportStatus",
    "DatasetInsights",
    "DatasetRestoreEstimate",
    "DatasetStorageStatus",
    "DatasetStorageTransition",
    "Deployment",
    "DeploymentClient",
    "DeploymentComputeOption",
    "DeploymentGpuType",
    "DeploymentQuote",
    "DeploymentStatus",
    "DetectionModel",
    "DetectionResult",
    "Device",
    "Directory",
    "DirectoryStats",
    "DirectoryTreeNode",
    "DuplicateHandling",
    "EvalClassMetrics",
    "EvalConfusionMatrix",
    "EvalImagePerformance",
    "EvalOverallMetrics",
    "EvalWorstImage",
    "EvaluationStatus",
    "Export",
    "ExportFormat",
    "ExportStatus",
    "ForbiddenError",
    "GpuType",
    "Image",
    "ImageStatus",
    "ImportJob",
    "ImportStatus",
    "InferenceModel",
    "InferenceResult",
    "InsightsDimensions",
    "InsightsOrientation",
    "InsightsSize",
    "InsightsStatusCounts",
    "InstanceSegmentationModel",
    "InstanceSegmentationResult",
    "InviteRole",
    "InviteStatus",
    "KeypointAnnotation",
    "KeypointModel",
    "KeypointResult",
    "LimitCheckResult",
    "LimitType",
    "Model",
    "ModelConfidence",
    "ModelConfidenceBuckets",
    "ModelEvaluation",
    "ModelFileEntry",
    "ModelFileManifest",
    "ModelPredictResult",
    "ModelStatus",
    "ModelType",
    "ModelVersionEntry",
    "ModelVersionsPayload",
    "ModelVisibility",
    "NearDuplicatesResult",
    "NetworkError",
    "NonBlankStr",
    "NotFoundError",
    "Notification",
    "Organization",
    "OrganizationInvite",
    "OrganizationMember",
    "OrganizationRole",
    "OrientedBoxGeometry",
    "PaymentRequiredError",
    "PictographError",
    "PipelineType",
    "Point",
    "PollTimeoutError",
    "PolygonAnnotation",
    "PolygonGeometry",
    "PolylineAnnotation",
    "PolylineGeometry",
    "ProjectedImages",
    "PromptResult",
    "PromptStatus",
    "RateLimitError",
    "RemoteDataset",
    "RequestTimeoutError",
    "Runtime",
    "SemanticSegmentationModel",
    "SemanticSegmentationResult",
    "ServerError",
    "SimilarImage",
    "SubscriptionTier",
    "TaggedImage",
    "Task",
    "TaskContribution",
    "TaskContributions",
    "TaskName",
    "TileFailure",
    "TileReport",
    "Toolkit",
    "TrainingRun",
    "TrainingStatus",
    "UploadFailure",
    "UploadReport",
    "ValidationError",
    "ValidationResult",
    "VideoExtractionJob",
    "VideoJobStatus",
    "VideoMetadata",
    "VideoUploadInfo",
    "WebhookDelivery",
    "WebhookDeliveryStatus",
    "WebhookEndpoint",
    "WeightFormat",
    "Workflow",
    "WorkflowRun",
    "WorkflowRunCreated",
    "WorkflowRunStatus",
    "WorkflowStatus",
    "__version__",
    "create_toolkit",
    "draw_annotations",
    "for_anthropic_messages",
    "for_claude_agent_sdk",
    "for_openai_agents",
    "for_openai_responses",
    "get_model",
    "load_model",
]


# ────────────── lazy: the async client and the agent toolkit ──────────────
#
# `import pictograph` used to construct the async client and the whole agent
# registry eagerly. The CLI is synchronous and touches neither unless you run
# `pictograph agents`.
#
# Be careful reading -X importtime here, because it misled me first: it reports
# CUMULATIVE time, so pictograph.aio "costs 115ms" and pictograph.agents "costs
# 91ms" while the actual saving from skipping BOTH is ~8ms (201ms vs 209ms for
# `import pictograph`). Nearly all of their cost is the models, resources and
# httpx that the sync client pulls in anyway. So this is a small win plus a
# structural one - two subsystems genuinely not built - and NOT the 200ms the
# cumulative numbers suggest. The dominant import cost is unavoidable: httpx
# ~70ms, typer ~23ms, pydantic ~19ms, and ~2-5ms per model module across ~50 of
# them.
#
# PEP 562 module __getattr__: `pictograph.AsyncClient` / `pictograph.Toolkit`
# and friends still resolve exactly as before, on first USE rather than on
# import. `from pictograph.aio import AsyncClient` was never affected - a direct
# submodule import does not go through here.
# ...but a type checker cannot follow __getattr__, so without the block below it
# resolves every lazy name to plain `object`. That silently un-typed the
# DOCUMENTED async entry point: `from pictograph import AsyncClient` (what the
# README shows) type-checked as `object`, so `AsyncClient()` failed under mypy
# with "object not callable" and every attribute on it was untyped - in a package
# that ships py.typed and advertises strict typing. Re-export under TYPE_CHECKING
# so checkers see the real symbols while runtime keeps the lazy import.
if TYPE_CHECKING:
    from pictograph.agents import (
        REGISTRY,
        Toolkit,
        create_toolkit,
        for_anthropic_messages,
        for_claude_agent_sdk,
        for_openai_agents,
        for_openai_responses,
    )
    from pictograph.aio import AsyncClient

_LAZY: dict[str, str] = {
    "AsyncClient": "pictograph.aio",
    "REGISTRY": "pictograph.agents",
    "Toolkit": "pictograph.agents",
    "create_toolkit": "pictograph.agents",
    "for_anthropic_messages": "pictograph.agents",
    "for_claude_agent_sdk": "pictograph.agents",
    "for_openai_agents": "pictograph.agents",
    "for_openai_responses": "pictograph.agents",
}


def __getattr__(name: str) -> object:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # resolve once; subsequent lookups skip this hook
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
