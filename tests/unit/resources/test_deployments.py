"""Tests for ``pictograph.resources.deployments.Deployments`` + ``DeploymentClient``.

Coverage:
- ``list`` / ``iter`` (filters + pagination), ``get`` (happy + 404).
- ``create`` - returns a typed ``CreatedDeployment`` carrying the one-time token;
  request body carries the compute/scaling fields.
- ``pause`` / ``resume`` / ``delete``.
- ``connect`` - builds a ``DeploymentClient`` (and refuses when no endpoint yet).
- ``DeploymentClient.infer`` - URL, bytes, and local-file image inputs hit the
  endpoint directly with the bearer token (NOT the backend transport).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest

from pictograph._http.retry import RetryPolicy
from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import (
    AuthError,
    ForbiddenError,
    NetworkError,
    NotFoundError,
    PaymentRequiredError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from pictograph.inference.results import (
    ClassificationResult,
    DetectionResult,
    InstanceSegmentationResult,
    KeypointResult,
    SemanticSegmentationResult,
)
from pictograph.models.deployment import (
    CreatedDeployment,
    Deployment,
    DeploymentComputeOption,
    DeploymentQuote,
)
from pictograph.resources._deployment_client import (
    UNREPORTED_DEVICE,
    DeploymentClient,
)
from pictograph.resources.deployments import Deployments

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
_PATH = f"{BASE}/api/v1/developer/deployments/"


@pytest.fixture
def transport() -> Transport:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def deployments(transport: Transport) -> Deployments:
    return Deployments(transport)


def _payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "dddddddd-aaaa-bbbb-cccc-eeeeeeeeeeee",
        "organization_id": "org-uuid",
        "model_id": "abcdef01-2345-6789-abcd-ef0123456789",
        "name": "Stop Sign Detector deployment",
        "status": "active",
        "compute_type": "gpu",
        "gpu_type": "t4",
        "min_containers": 0,
        "max_containers": 1,
        "scaledown_window": 60,
        "endpoint_url": "https://gateway.test/stop-sign-detector/predict",
        "auth_token_prefix": "pk_deploy_ab12…",
        "inference_config": {"confidence": 0.5},
        "cost_rate_per_min": 1,
        "accrued_cost_credits": 12,
        "uptime_seconds": 720,
        "created_at": "2026-05-30T00:00:00Z",
    }
    base.update(overrides)
    return base


# ───────────── list / iter / get ─────────────


def test_list_returns_typed(httpx_mock: HTTPXMock, deployments: Deployments) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}?limit=50&offset=0",
        json={"deployments": [_payload(), _payload(id="d2", name="Other")]},
    )
    result = deployments.list()
    assert len(result) == 2
    assert all(isinstance(d, Deployment) for d in result)


def test_list_passes_filters(httpx_mock: HTTPXMock, deployments: Deployments) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}?limit=20&offset=0&model_id=abcdef01-2345-6789-abcd-ef0123456789&status=active",
        json={"deployments": [_payload()]},
    )
    assert (
        len(
            deployments.list(
                model="abcdef01-2345-6789-abcd-ef0123456789", status="active", limit=20
            )
        )
        == 1
    )


def test_iter_paginates(httpx_mock: HTTPXMock, deployments: Deployments) -> None:
    httpx_mock.add_response(
        url=f"{_PATH}?offset=0&limit=2",
        json={"deployments": [_payload(id="d1"), _payload(id="d2")]},
    )
    httpx_mock.add_response(
        url=f"{_PATH}?offset=2&limit=2", json={"deployments": [_payload(id="d3")]}
    )
    assert [d.id for d in deployments.iter(page_size=2)] == ["d1", "d2", "d3"]


def test_get_returns_typed(httpx_mock: HTTPXMock, deployments: Deployments) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}dddddddd-aaaa-bbbb-cccc-eeeeeeeeeeee",
        json={"deployment": _payload()},
    )
    d = deployments.get("dddddddd-aaaa-bbbb-cccc-eeeeeeeeeeee")
    assert (
        isinstance(d, Deployment)
        and d.id == "dddddddd-aaaa-bbbb-cccc-eeeeeeeeeeee"
        and d.gpu_type == "t4"
    )


def test_get_404(httpx_mock: HTTPXMock, deployments: Deployments) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}00000000-0000-4000-8000-000000000000",
        status_code=404,
        json={"detail": "Deployment not found"},
    )
    with pytest.raises(NotFoundError):
        deployments.get("00000000-0000-4000-8000-000000000000")


# ───────────── create ─────────────


def test_create_returns_token_and_deployment(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=_PATH,
        json={
            "deployment": _payload(status="provisioning"),
            "auth_token": "pk_deploy_secret123",
            "message": "shown once",
        },
    )
    created = deployments.create(
        "abcdef01-2345-6789-abcd-ef0123456789", compute_type="gpu", gpu_type="t4", min_containers=1
    )
    assert isinstance(created, CreatedDeployment)
    assert created.auth_token == "pk_deploy_secret123"  # noqa: S105 - test fixture, not a real secret
    assert created.deployment.model_id == "abcdef01-2345-6789-abcd-ef0123456789"

    # request body carried the compute/scaling selection
    body = httpx_mock.get_requests()[-1].read().decode()
    assert '"compute_type": "gpu"' in body or '"compute_type":"gpu"' in body
    assert "min_containers" in body


def test_create_cpu_nulls_gpu_type(httpx_mock: HTTPXMock, deployments: Deployments) -> None:
    httpx_mock.add_response(
        method="POST",
        url=_PATH,
        json={
            "deployment": _payload(compute_type="cpu", gpu_type=None),
            "auth_token": "pk_deploy_x",
        },
    )
    deployments.create(
        "abcdef01-2345-6789-abcd-ef0123456789", compute_type="cpu", gpu_type="t4"
    )  # gpu_type ignored for cpu
    body = httpx_mock.get_requests()[-1].read().decode()
    assert '"gpu_type": null' in body or '"gpu_type":null' in body


# ───────────── pause / resume / delete ─────────────


def test_pause(httpx_mock: HTTPXMock, deployments: Deployments) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{_PATH}dddddddd-aaaa-bbbb-cccc-eeeeeeeeeeee/pause",
        json={"deployment": _payload(status="paused")},
    )
    assert deployments.pause("dddddddd-aaaa-bbbb-cccc-eeeeeeeeeeee").status == "paused"


def test_resume(httpx_mock: HTTPXMock, deployments: Deployments) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{_PATH}dddddddd-aaaa-bbbb-cccc-eeeeeeeeeeee/resume",
        json={"deployment": _payload(status="active")},
    )
    assert deployments.resume("dddddddd-aaaa-bbbb-cccc-eeeeeeeeeeee").status == "active"


def test_delete(httpx_mock: HTTPXMock, deployments: Deployments) -> None:
    httpx_mock.add_response(
        method="DELETE", url=f"{_PATH}dddddddd-aaaa-bbbb-cccc-eeeeeeeeeeee", json={"success": True}
    )
    assert deployments.delete("dddddddd-aaaa-bbbb-cccc-eeeeeeeeeeee") is None


# ───────────── bulk pause / resume / delete ─────────────


def test_bulk_pause_returns_typed(httpx_mock: HTTPXMock, deployments: Deployments) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{_PATH}bulk-pause",
        json={"success": True, "succeeded": ["d1"], "not_found": ["d2"], "count": 1},
    )
    result = deployments.bulk_pause(
        ["d1111111-1111-1111-1111-111111111111", "d2222222-2222-2222-2222-222222222222"]
    )
    assert result.succeeded == ["d1"]
    assert result.not_found == ["d2"]
    assert result.count == 1
    body = httpx_mock.get_requests()[-1].read().decode()
    assert '"deployment_ids"' in body and "d1" in body and "d2" in body


def test_bulk_resume_returns_typed(httpx_mock: HTTPXMock, deployments: Deployments) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{_PATH}bulk-resume",
        json={"success": True, "succeeded": ["d1", "d2"], "not_found": [], "count": 2},
    )
    result = deployments.bulk_resume(
        ["d1111111-1111-1111-1111-111111111111", "d2222222-2222-2222-2222-222222222222"]
    )
    assert result.succeeded == ["d1", "d2"]
    assert result.count == 2


def test_bulk_delete_returns_typed(httpx_mock: HTTPXMock, deployments: Deployments) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{_PATH}bulk-delete",
        json={"success": True, "deleted": ["d1"], "not_found": ["gone"], "count": 1},
    )
    result = deployments.bulk_delete(
        ["d1111111-1111-1111-1111-111111111111", "d2222222-2222-2222-2222-222222222222"]
    )
    assert result.deleted == ["d1"]
    assert result.not_found == ["gone"]
    assert result.count == 1


# ───────────── connect / DeploymentClient ─────────────

# ── The REAL wire shapes ──
#
# A deployment's serving container returns `dispatch.infer_image`
# verbatim and the gateway passes that dict straight through as its JSON body, so
# these are transcribed from that emitter - including the details that a
# hand-written guess gets wrong: a classifier's entries are keyed `class` (not
# `name`), a keypoint entry carries `instance_id` and NO `bounding_box`, and
# `attributes` arrives as the legacy list form.

_BBOX = {
    "id": "a1",
    "name": "dog",
    "type": "bbox",
    "bounding_box": {"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0},
    "confidence": 0.9,
    "attributes": ["auto-annotate"],
}
_POLYGON = {
    "id": "a2",
    "name": "widget",
    "type": "polygon",
    "bounding_box": {"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0},
    "polygon": {"paths": [[{"x": 1.0, "y": 2.0}, {"x": 4.0, "y": 2.0}, {"x": 4.0, "y": 6.0}]]},
    "confidence": 0.8,
    "attributes": ["auto-annotate"],
}
_KEYPOINT = {
    "id": "a3",
    "name": "nose",
    "type": "keypoint",
    "keypoint": {"x": 5.0, "y": 6.0},
    "confidence": 0.7,
    "attributes": ["auto-annotate"],
    "instance_id": 1,
}
_CLASS = {"class": "dog", "confidence": 0.94}

_WIRE_BY_TASK = {
    "object_detection": {"model_type": "object_detection", "predictions": [_BBOX]},
    "instance_segmentation": {
        "model_type": "instance_segmentation",
        "predictions": [_POLYGON, _BBOX],
    },
    "semantic_segmentation": {"model_type": "semantic_segmentation", "predictions": [_POLYGON]},
    "keypoint_detection": {"model_type": "keypoint_detection", "predictions": [_KEYPOINT]},
    "classification": {
        "model_type": "classification",
        "predictions": [_CLASS],
        "tags": ["dog"],
    },
}

_RESULT_BY_TASK = {
    "object_detection": DetectionResult,
    "instance_segmentation": InstanceSegmentationResult,
    "semantic_segmentation": SemanticSegmentationResult,
    "keypoint_detection": KeypointResult,
    "classification": ClassificationResult,
}


def test_connect_requires_endpoint() -> None:
    dep = Deployment.model_validate(_payload(endpoint_url=None, status="provisioning"))
    with pytest.raises(ValueError):
        Deployments.connect(dep, "pk_deploy_x")


def test_connect_builds_client() -> None:
    dep = Deployment.model_validate(_payload())
    client = Deployments.connect(dep, "pk_deploy_x")
    assert isinstance(client, DeploymentClient)


def test_deployment_client_infer_url(httpx_mock: HTTPXMock) -> None:
    endpoint = "https://gateway.test/deployment/predict"
    httpx_mock.add_response(
        method="POST",
        url=endpoint,
        json={"model_type": "object_detection", "predictions": [_BBOX]},
    )
    client = DeploymentClient(endpoint, "pk_deploy_tok")
    result = client.infer(image="https://example.com/dog.jpg", confidence=0.6)
    # The SAME typed class the local path returns for this task.
    assert isinstance(result, DetectionResult)
    assert result.predictions[0].name == "dog"
    assert result.predictions[0].bounding_box.w == 30.0
    req = httpx_mock.get_requests()[-1]
    assert req.headers["Authorization"] == "Bearer pk_deploy_tok"
    body = req.read().decode()
    assert '"type": "url"' in body or '"type":"url"' in body
    assert "0.6" in body


def test_deployment_client_infer_bytes(httpx_mock: HTTPXMock) -> None:
    endpoint = "https://gateway.test/deployment/predict"
    httpx_mock.add_response(
        method="POST", url=endpoint, json={"model_type": "object_detection", "predictions": []}
    )
    client = DeploymentClient(endpoint, "pk_deploy_tok", task="object_detection")
    result = client.infer(image=b"\x89PNG fake bytes")
    assert result.predictions == []
    req = httpx_mock.get_requests()[-1]
    assert req.headers["Authorization"] == "Bearer pk_deploy_tok"
    # bytes are sent as base64 JSON (not multipart) so options can ride along
    body = req.read().decode()
    assert '"type": "base64"' in body or '"type":"base64"' in body


def test_deployment_client_infer_raw_returns_wire_body(httpx_mock: HTTPXMock) -> None:
    """`infer_raw` is the escape hatch: the endpoint's JSON, untouched.

    Choosing types must never cost access to a field the result classes do not
    model - `tags` on a classifier being the live example.
    """
    endpoint = "https://gateway.test/deployment/predict"
    wire = {"model_type": "classification", "predictions": [_CLASS], "tags": ["dog"]}
    httpx_mock.add_response(method="POST", url=endpoint, json=wire)
    client = DeploymentClient(endpoint, "pk_deploy_tok")
    assert client.infer_raw(image=b"x") == wire


def test_deployment_client_infer_bytes_carries_options(httpx_mock: HTTPXMock) -> None:
    # Regression: the old multipart bytes path dropped confidence/class_filter/
    # top_k (the deployment endpoint set body={} for multipart). They must now
    # reach the wire, same as the URL and local-file paths.
    endpoint = "https://gateway.test/deployment/predict"
    httpx_mock.add_response(
        method="POST", url=endpoint, json={"model_type": "object_detection", "predictions": []}
    )
    client = DeploymentClient(endpoint, "pk_deploy_tok")
    client.infer(image=b"\x89PNG fake", confidence=0.91, class_filter=["dog"], top_k=3)
    body = httpx_mock.get_requests()[-1].read().decode()
    assert "0.91" in body  # confidence reached the wire
    assert "dog" in body  # class_filter reached the wire
    assert '"top_k": 3' in body or '"top_k":3' in body


def test_deployment_client_requires_endpoint_and_key() -> None:
    with pytest.raises(ValueError):
        DeploymentClient("", "k")
    with pytest.raises(ValueError):
        DeploymentClient("https://gateway.test/x/predict", "")


def test_deployment_client_repr_omits_token() -> None:
    """The per-deployment bearer token must never surface in repr() (logs/tracebacks)."""
    client = DeploymentClient("https://gateway.test/x/predict", "pk_deploy_supersecret")
    rendered = repr(client)
    assert "pk_deploy_supersecret" not in rendered
    assert "https://gateway.test/x/predict" in rendered


# ── error mapping + retry: the endpoint's HTTP failures become SDK-typed errors,
#    and a transient gateway 429 (the per-deployment rate limit) / cold-start 5xx
#    / network blip is retried (inference is side-effect-free). ──

_EP = "https://gateway.test/x/predict"


def _fast_retry(max_retries: int = 2) -> RetryPolicy:
    """A RetryPolicy with a no-op sleep so retry-path tests don't actually wait."""
    return RetryPolicy(max_retries=max_retries, sleep=lambda _s: None)


def test_deployment_client_429_raises_typed_ratelimit(httpx_mock: HTTPXMock) -> None:
    """The gateway's per-deployment rate limit surfaces as the SDK's RateLimitError
    (with the parsed Retry-After) - not a raw httpx.HTTPStatusError."""
    httpx_mock.add_response(
        method="POST",
        url=_EP,
        status_code=429,
        headers={"Retry-After": "7"},
        json={"detail": "rate limited"},
    )
    client = DeploymentClient(_EP, "pk_deploy_tok", retry_policy=_fast_retry(0))
    with pytest.raises(RateLimitError) as exc:
        client.infer(image="https://example.com/dog.jpg")
    assert exc.value.retry_after == 7


def test_deployment_client_retries_429_then_succeeds(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=_EP,
        status_code=429,
        headers={"Retry-After": "0"},
        json={"detail": "slow down"},
    )
    httpx_mock.add_response(
        method="POST",
        url=_EP,
        json={"model_type": "object_detection", "predictions": [{**_BBOX, "name": "cat"}]},
    )
    client = DeploymentClient(_EP, "pk_deploy_tok", retry_policy=_fast_retry(2))
    result = client.infer(image=b"\x89PNG")
    assert result.predictions[0].name == "cat"
    assert len(httpx_mock.get_requests()) == 2


def test_deployment_client_retries_cold_start_5xx(httpx_mock: HTTPXMock) -> None:
    """A scale-to-zero cold-start 503 is retried (predict has no side effect)."""
    httpx_mock.add_response(method="POST", url=_EP, status_code=503, text="cold")
    httpx_mock.add_response(
        method="POST", url=_EP, json={"model_type": "object_detection", "predictions": []}
    )
    client = DeploymentClient(_EP, "pk_deploy_tok", retry_policy=_fast_retry(2))
    assert client.infer(image=b"x").predictions == []


@pytest.mark.parametrize(
    "status,exc_cls",
    [
        (401, AuthError),
        (403, ForbiddenError),
        (404, NotFoundError),
        (402, PaymentRequiredError),
        (400, ValidationError),
        (500, ServerError),
    ],
)
def test_deployment_client_maps_http_errors_to_typed(
    httpx_mock: HTTPXMock, status: int, exc_cls: type
) -> None:
    httpx_mock.add_response(method="POST", url=_EP, status_code=status, json={"detail": "nope"})
    client = DeploymentClient(_EP, "pk_deploy_tok", retry_policy=_fast_retry(0))
    with pytest.raises(exc_cls):
        client.infer(image="https://example.com/x.jpg")


def test_deployment_client_network_error_is_typed(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ConnectError("boom"))
    client = DeploymentClient(_EP, "pk_deploy_tok", retry_policy=_fast_retry(0))
    with pytest.raises(NetworkError):
        client.infer(image=b"x")


def test_deployment_client_max_retries_zero_is_single_shot(httpx_mock: HTTPXMock) -> None:
    """A latency-strict caller (max_retries=0) gets exactly one attempt."""
    httpx_mock.add_response(method="POST", url=_EP, status_code=503, text="down")
    client = DeploymentClient(_EP, "pk_deploy_tok", max_retries=0)
    with pytest.raises(ServerError):
        client.infer(image=b"x")
    assert len(httpx_mock.get_requests()) == 1


# ───────────── compute_options / quote (quote-before-commit) ─────────────


def test_compute_options_returns_typed(httpx_mock: HTTPXMock, deployments: Deployments) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}compute-options",
        json={
            "success": True,
            "options": [
                {
                    "key": "cpu",
                    "label": "CPU",
                    "compute_type": "cpu",
                    "gpu_type": None,
                    "is_gpu": False,
                    "description": "Cheapest",
                    "rate_per_min_micro_usd": 100,
                },
                {
                    "key": "t4",
                    "label": "NVIDIA T4",
                    "compute_type": "gpu",
                    "gpu_type": "t4",
                    "is_gpu": True,
                    "description": "Entry GPU",
                    "rate_per_min_micro_usd": 900,
                },
            ],
        },
    )
    opts = deployments.compute_options()
    assert len(opts) == 2
    assert all(isinstance(o, DeploymentComputeOption) for o in opts)
    t4 = next(o for o in opts if o.key == "t4")
    assert t4.gpu_type == "t4" and t4.is_gpu is True and t4.rate_per_min_micro_usd == 900
    cpu = next(o for o in opts if o.key == "cpu")
    assert cpu.gpu_type is None and cpu.compute_type == "cpu"


def test_quote_default_scale_to_zero(httpx_mock: HTTPXMock, deployments: Deployments) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}quote?compute_type=gpu&min_containers=0",
        json={
            "success": True,
            "quote": {
                "rate_per_min_micro_usd": 900,
                "cost_per_hour_micro_usd": 54000,
                "cost_per_day_micro_usd": 1296000,
                "scale_to_zero": True,
                "billing_note": "Charged only while serving requests; scales to zero when idle.",
            },
        },
    )
    q = deployments.quote()
    assert isinstance(q, DeploymentQuote)
    assert q.scale_to_zero is True
    assert q.cost_per_day_micro_usd == q.cost_per_hour_micro_usd * 24


def test_quote_passes_gpu_and_min_containers(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}quote?compute_type=gpu&min_containers=2&gpu_type=a10g",
        json={
            "success": True,
            "quote": {
                "rate_per_min_micro_usd": 5000,
                "cost_per_hour_micro_usd": 300000,
                "cost_per_day_micro_usd": 7200000,
                "scale_to_zero": False,
                "billing_note": "Charged continuously while the deployment is warm.",
            },
        },
    )
    q = deployments.quote(compute_type="gpu", gpu_type="a10g", min_containers=2)
    assert q.scale_to_zero is False
    assert q.rate_per_min_micro_usd == 5000


# ───────────── Remote == Edge: the typed result contract ─────────────


@pytest.mark.parametrize("task", sorted(_WIRE_BY_TASK))
def test_infer_returns_the_task_result_class(httpx_mock: HTTPXMock, task: str) -> None:
    """EVERY task the platform trains parses into its own result class.

    This is the Edge/Remote parity claim, asserted rather than assumed: the
    payloads above are the deployment emitter's real output, and each must
    produce the same class `model.predict()` produces locally for that task.
    """
    httpx_mock.add_response(method="POST", url=_EP, json=_WIRE_BY_TASK[task])
    with DeploymentClient(_EP, "pk_deploy_tok") as client:
        result = client.infer(image=b"x")
    assert isinstance(result, _RESULT_BY_TASK[task])
    assert result.model_type == task


def test_classification_maps_the_class_key_not_name(httpx_mock: HTTPXMock) -> None:
    """A classifier's entries are keyed `class`; `ClassScore` names the field `name`.

    A naive `model_validate` of the payload would drop every score, so this is
    the one task whose conversion is not a passthrough.
    """
    httpx_mock.add_response(method="POST", url=_EP, json=_WIRE_BY_TASK["classification"])
    client = DeploymentClient(_EP, "pk_deploy_tok", task="classification")
    result = client.infer(image=b"x")
    assert result.top.name == "dog"
    assert result.top.confidence == pytest.approx(0.94)
    assert result.tags == ["dog"]


def test_keypoint_groups_by_instance_id(httpx_mock: HTTPXMock) -> None:
    """Joints sharing an `instance_id` reconstruct one object - the same view
    the Edge snippet iterates (`result.instances`)."""
    second = {**_KEYPOINT, "id": "a4", "name": "left_eye"}
    httpx_mock.add_response(
        method="POST",
        url=_EP,
        json={"model_type": "keypoint_detection", "predictions": [_KEYPOINT, second]},
    )
    client = DeploymentClient(_EP, "pk_deploy_tok", task="keypoint_detection")
    result = client.infer(image=b"x")
    assert [len(group) for group in result.instances] == [2]
    assert [j.name for j in result.instances[0]] == ["nose", "left_eye"]


def test_infer_reports_honest_provenance(httpx_mock: HTTPXMock) -> None:
    """A remote result never invents local-run provenance.

    `backend` is real (a deployment always serves ONNX). `device` degrades to
    `remote`, NOT `cpu` - reporting a T4 deployment as CPU would be a silent
    falsehood. `inference_ms` stays None because the wire carries no forward-pass
    timing and the round-trip is a different measurement.
    """
    httpx_mock.add_response(method="POST", url=_EP, json=_WIRE_BY_TASK["object_detection"])
    client = DeploymentClient(_EP, "pk_deploy_tok")
    result = client.infer(image=b"x")
    assert result.backend == "onnxruntime"
    assert result.device == UNREPORTED_DEVICE
    assert result.providers == []
    assert result.inference_ms is None


def test_infer_uses_the_gateway_reported_device(httpx_mock: HTTPXMock) -> None:
    """When the gateway reports the hardware, that is what `device` says."""
    httpx_mock.add_response(
        method="POST",
        url=_EP,
        json=_WIRE_BY_TASK["object_detection"],
        headers={"X-Pictograph-Device": "cuda"},
    )
    client = DeploymentClient(_EP, "pk_deploy_tok")
    assert client.infer(image=b"x").device == "cuda"


def test_declared_task_must_match_what_the_endpoint_serves(httpx_mock: HTTPXMock) -> None:
    """A wrong `task=` raises instead of quietly returning an empty result.

    Parsing a classifier's `{"class": …}` entries as a detector's annotations
    does not fail loudly on its own - it yields zero predictions, which reads as
    'the model found nothing'.
    """
    httpx_mock.add_response(method="POST", url=_EP, json=_WIRE_BY_TASK["classification"])
    client = DeploymentClient(_EP, "pk_deploy_tok", task="object_detection")
    with pytest.raises(ServerError, match="serves a 'classification' model"):
        client.infer(image=b"x")


def test_unknown_model_type_raises(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=_EP, json={"model_type": "wat", "predictions": []})
    client = DeploymentClient(_EP, "pk_deploy_tok")
    with pytest.raises(ServerError, match="unknown model_type"):
        client.infer(image=b"x")


def test_missing_model_type_falls_back_to_the_declared_task(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=_EP, json={"predictions": [_BBOX]})
    client = DeploymentClient(_EP, "pk_deploy_tok", task="object_detection")
    assert client.infer(image=b"x").predictions[0].name == "dog"


def test_missing_model_type_without_a_declared_task_raises(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=_EP, json={"predictions": [_BBOX]})
    client = DeploymentClient(_EP, "pk_deploy_tok")
    with pytest.raises(ServerError, match="no model_type"):
        client.infer(image=b"x")


def test_bad_task_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="Unknown task"):
        DeploymentClient(_EP, "pk_deploy_tok", task="segmentation")  # type: ignore[call-overload]


def test_connect_passes_the_task_through() -> None:
    dep = Deployment.model_validate(_payload())
    client = Deployments.connect(dep, "pk_deploy_x", task="keypoint_detection")
    assert client.task == "keypoint_detection"
    assert repr(client).endswith("task='keypoint_detection')")
