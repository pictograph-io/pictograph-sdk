"""Tests for ``pictograph.client.Client`` - construction, lifecycle, surface."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import pictograph
from pictograph import Client
from pictograph.exceptions import ConfigurationError

# ───────────── construction precedence ─────────────


def test_client_constructs_with_explicit_api_key() -> None:
    c = Client(api_key="pk_live_test_explicit")
    try:
        assert c._config.base_url == "https://api.pictograph.io"
    finally:
        c.close()


def test_client_uses_env_var_when_no_explicit_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_from_env")
    with Client() as c:
        assert c._config.api_key is not None
        assert c._config.api_key.get_secret_value() == "pk_live_from_env"


def test_client_explicit_kwarg_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_env")
    # Explicit key flows through resolve_api_key → backed into Transport headers.
    with Client(api_key="pk_live_explicit") as c:
        # Transport receives the explicit key, not the env one.
        assert c._transport._api_key == "pk_live_explicit"


def test_client_raises_configuration_error_when_no_key_anywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PICTOGRAPH_API_KEY", raising=False)
    with pytest.raises(ConfigurationError) as exc:
        Client()
    assert exc.value.fix is not None
    assert "PICTOGRAPH_API_KEY" in exc.value.fix


# ───────────── kwarg → config plumbing ─────────────


def test_client_base_url_kwarg_propagates_to_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_x")
    with Client(base_url="https://staging.pictograph.io") as c:
        assert c._config.base_url == "https://staging.pictograph.io"


def test_client_timeout_kwarg_propagates_to_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_x")
    with Client(timeout=60.0) as c:
        assert c._config.timeout == 60.0


def test_client_max_retries_kwarg_propagates_to_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_x")
    with Client(max_retries=10) as c:
        assert c._config.max_retries == 10


def test_client_kwargs_dont_clobber_env_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_x")
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", "https://from-env.pictograph.io")
    monkeypatch.setenv("PICTOGRAPH_TIMEOUT", "45")
    with Client() as c:
        assert c._config.base_url == "https://from-env.pictograph.io"
        assert c._config.timeout == 45.0


def test_client_invalid_timeout_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_x")
    with pytest.raises(ValidationError):
        Client(timeout=-1.0)


def test_client_invalid_max_retries_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_x")
    with pytest.raises(ValidationError):
        Client(max_retries=-1)


# ───────────── repr safety ─────────────


def test_repr_does_not_leak_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_supersecret_must_not_appear_anywhere")
    with Client() as c:
        rep = repr(c)
    assert "supersecret" not in rep
    assert "pk_live_" not in rep


def test_repr_includes_base_url_for_debugging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_x")
    with Client(base_url="https://staging.pictograph.io") as c:
        assert "staging.pictograph.io" in repr(c)


# ───────────── lifecycle ─────────────


def test_client_close_releases_underlying_httpx_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_x")
    c = Client()
    assert c._transport._client.is_closed is False
    c.close()
    assert c._transport._client.is_closed is True


def test_client_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_x")
    c = Client()
    c.close()
    c.close()  # must not raise


def test_client_context_manager_closes_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_x")
    c = Client()
    with c as ctx:
        assert ctx is c
        assert c._transport._client.is_closed is False
    assert c._transport._client.is_closed is True


def test_client_context_manager_closes_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception inside the with-block must still trigger close()."""
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_x")
    c = Client()
    with pytest.raises(RuntimeError, match="planned"), c:
        raise RuntimeError("planned")
    assert c._transport._client.is_closed is True


# ───────────── public surface ─────────────


def test_top_level_module_exports_documented_symbols() -> None:
    """Pin the public surface so accidental removals show up in CI.

    The local-inference rewrite deleted ``LocalModel``/``PyTorchModel`` in
    favour of five task-typed model/result pairs (``DetectionModel`` /
    ``DetectionResult``, etc.), ``AnyModel``, ``TaskName``, and the
    ``format=`` loader vocabulary - see ``pictograph.inference``.
    """
    expected = {
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
        "BACKGROUND",
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
        "DEFAULT_PALETTE",
        "DEVICES",
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
        "Directory",
        "DirectoryStats",
        "DirectoryTreeNode",
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
        "REGISTRY",
        "RUNTIMES",
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
        "Toolkit",
        "TileFailure",
        "TileReport",
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
    }
    assert set(pictograph.__all__) == expected
    # Every name in __all__ must actually be importable.
    for name in pictograph.__all__:
        assert hasattr(pictograph, name), f"{name} declared in __all__ but missing"


def test_version_is_a_non_empty_string() -> None:
    assert isinstance(pictograph.__version__, str)
    assert pictograph.__version__  # non-empty


def test_internal_modules_not_re_exported() -> None:
    """The ``_http`` and ``_internal`` packages must stay hidden from import *."""
    assert "_http" not in pictograph.__all__
    assert "_internal" not in pictograph.__all__
    assert "_version" not in pictograph.__all__


# ───────────── resource accessors ─────────────


def test_client_exposes_all_resource_accessors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every shipped resource is reachable as a public attribute on Client."""
    from pictograph.resources.annotations import Annotations
    from pictograph.resources.api_keys import ApiKeys
    from pictograph.resources.auto_annotate import AutoAnnotate
    from pictograph.resources.batch import Batch
    from pictograph.resources.connectors import Connectors
    from pictograph.resources.credits import Credits
    from pictograph.resources.datasets import Datasets
    from pictograph.resources.exports import Exports
    from pictograph.resources.images import Images
    from pictograph.resources.models import Models
    from pictograph.resources.organizations import Organizations
    from pictograph.resources.search import Search
    from pictograph.resources.training import Training
    from pictograph.resources.video import Video

    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_x")
    with Client() as c:
        assert isinstance(c.datasets, Datasets)
        assert isinstance(c.images, Images)
        assert isinstance(c.annotations, Annotations)
        assert isinstance(c.exports, Exports)
        assert isinstance(c.training, Training)
        assert isinstance(c.models, Models)
        assert isinstance(c.credits, Credits)
        assert isinstance(c.organizations, Organizations)
        assert isinstance(c.batch, Batch)
        assert isinstance(c.search, Search)
        assert isinstance(c.auto_annotate, AutoAnnotate)
        assert isinstance(c.video, Video)
        assert isinstance(c.connectors, Connectors)
        assert isinstance(c.api_keys, ApiKeys)


def test_resources_share_the_same_underlying_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All resources hold a reference to the single Client-owned Transport.

    Sharing the transport means the connection pool, retry policy, auth
    headers, and idempotency keys are configured exactly once and apply to
    every call - calling ``client.datasets.list()`` and ``client.training.create()``
    must talk through the same httpx.Client.
    """
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_x")
    with Client() as c:
        transport = c._transport
        assert c.datasets._transport is transport
        assert c.images._transport is transport
        assert c.annotations._transport is transport
        assert c.exports._transport is transport
        assert c.training._transport is transport
        assert c.models._transport is transport
        assert c.credits._transport is transport
        assert c.api_keys._transport is transport


def test_resource_accessors_are_eagerly_constructed_not_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resources are real instances on the Client, not descriptors / properties.

    This pins the eager-allocation choice - if someone later refactors to
    lazy properties, they'd need a deliberate decision and this test would
    surface the change.
    """
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_x")
    c = Client()
    try:
        # Same object identity across two attribute reads.
        assert c.datasets is c.datasets
        assert c.images is c.images
    finally:
        c.close()


def test_client_close_does_not_invalidate_resource_accessors_for_repeat_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After close(), resource attributes still exist (no AttributeError).

    They will fail on use because the underlying httpx.Client is closed,
    but the Python attribute lookup itself remains safe.
    """
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_x")
    c = Client()
    c.close()
    # Attribute access works.
    assert c.datasets is not None
    assert c.images is not None
    assert c.annotations is not None
    assert c.exports is not None
    assert c.training is not None
    assert c.models is not None
    assert c.credits is not None
    assert c.api_keys is not None
