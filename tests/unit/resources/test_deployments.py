"""Tests for ``pictograph.resources.deployments.Deployments`` + ``DeploymentClient``.

Coverage:
- ``list`` / ``iter`` (filters + pagination), ``get`` (happy + 404).
- The ``Deployment`` model parses the row the API really answers on PRT (precision,
  version, engine build, error, the priced costs, request metrics), and a row from
  before the GPU-only engine (``compute_type: "cpu"``) still reads.
- ``create`` - returns a typed ``CreatedDeployment`` carrying the one-time token;
  the request body carries the GPU, the precision and the scaling, never a
  ``compute_type``.
- ``wait_until_active`` - polls by id until ``active``; a failed, paused or
  terminated deployment raises at once; a timeout raises ``PollTimeoutError``.
- ``pause`` / ``resume`` / ``delete`` and the bulk verbs.
- ``update`` - a PATCH carrying only the scaling given (a ``0`` included), by id after
  resolving a name once; nothing given raises before any request; a ``502`` (the live
  deployment refused the change) is retried.
- ``connect`` - builds a ``DeploymentClient`` (and refuses one with no endpoint).
- ``DeploymentClient.infer`` - URL, bytes, and local-file image inputs hit the
  endpoint directly with the bearer token (NOT the backend transport); the
  result's provenance is what the gateway's headers report; the gateway's error
  slugs become typed errors with a ``code`` and a ``fix``.
"""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from pictograph._http.retry import RetryPolicy
from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import (
    ApiError,
    AuthError,
    ConflictError,
    ForbiddenError,
    NetworkError,
    NotFoundError,
    PaymentRequiredError,
    PollTimeoutError,
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
    DeploymentComputeOptions,
    DeploymentComputePrecision,
    DeploymentQuote,
    DeploymentRuntime,
)
from pictograph.resources._deployment_client import (
    DEFAULT_TIMEOUT_S,
    UNREPORTED_DEVICE,
    DeploymentClient,
    backend_of,
    server_timing_ms,
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


DEP_ID = "dddddddd-aaaa-bbbb-cccc-eeeeeeeeeeee"


def _payload(**overrides: Any) -> dict[str, Any]:
    """A deployment row exactly as the developer API answers it on PRT: the public
    columns (``routes/deployments.py::_PUBLIC_COLS``), the ``models`` embed, and what
    ``_enrich`` adds (the priced hourly costs and the gateway's request metrics)."""
    base: dict[str, Any] = {
        "id": DEP_ID,
        "organization_id": "org-uuid",
        "model_id": "abcdef01-2345-6789-abcd-ef0123456789",
        "version_id": "0b1c2d3e-0000-4000-8000-000000000001",
        "name": "Stop Sign Detector deployment",
        "status": "active",
        "compute_type": "gpu",
        "gpu_type": "t4",
        "precision": "fp32",
        "min_containers": 0,
        "max_containers": 1,
        "scaledown_window": 60,
        "modal_app_name": f"pictograph-deploy-{DEP_ID}",
        "endpoint_url": "https://gateway.test/dddddddd/predict",
        "auth_token_prefix": "pk_deploy_ab12…",
        "inference_config": {"confidence": 0.5},
        "rate_limit_rpm": None,
        "cost_rate_per_min": 79320,
        "accrued_cost_micro_usd": 1200,
        "uptime_seconds": 720,
        "transfer_bytes": 0,
        "transfer_cost_micro_usd": 0,
        "last_metered_at": "2026-10-23T12:03:00+00:00",
        "error_message": None,
        "engine_build": None,
        "created_by": "user-uuid",
        "created_at": "2026-10-23T12:00:00+00:00",
        "updated_at": "2026-10-23T12:01:00+00:00",
        "started_at": "2026-10-23T12:01:00+00:00",
        "paused_at": None,
        "models": {
            "id": "abcdef01-2345-6789-abcd-ef0123456789",
            "name": "Stop Sign Detector",
            "model_type": "object_detection",
            "architecture": "rfdetr-small",
        },
        "cost_per_hour_micro_usd": 4759200,
        "max_cost_per_hour_micro_usd": 4759200,
        "metrics": {
            "requests": 12,
            "errors": 1,
            "avg_latency_ms": 41,
            "error_rate": 0.0833,
            "last_request_at": "1761220800",
        },
    }
    base.update(overrides)
    return base


_BUILDING = {
    "state": "building",
    "models": [
        {
            "model_id": "abcdef01-2345-6789-abcd-ef0123456789",
            "name": "Stop Sign Detector",
            "precision": "fp16",
            "state": "building",
            "started_at": "2026-10-23T12:00:05+00:00",
            "estimate_s": 31.1,
        }
    ],
}
_BUILT = {
    "state": "built",
    "models": [
        {
            "model_id": "abcdef01-2345-6789-abcd-ef0123456789",
            "precision": "fp16",
            "state": "built",
            "build_s": 30.5,
        }
    ],
}


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
        "abcdef01-2345-6789-abcd-ef0123456789", gpu_type="t4", min_containers=1
    )
    assert isinstance(created, CreatedDeployment)
    assert created.auth_token == "pk_deploy_secret123"  # noqa: S105 - test fixture, not a real secret
    assert created.deployment.model_id == "abcdef01-2345-6789-abcd-ef0123456789"
    assert created.deployment.status == "provisioning"

    # The body is exactly what the route reads: the GPU, the precision (fp32 by
    # default) and the scaling. Never a compute_type - there is only the GPU now.
    body = json.loads(httpx_mock.get_requests()[-1].content)
    assert body == {
        "model_id": "abcdef01-2345-6789-abcd-ef0123456789",
        "region": "auto",
        "gpu_type": "t4",
        "precision": "fp32",
        "min_containers": 1,
        "max_containers": 1,
        "scaledown_window": 60,
        "inference_config": {},
    }


def test_create_sends_the_gpu_and_precision_chosen(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=_PATH,
        json={
            "deployment": _payload(
                status="provisioning", gpu_type="l4", precision="fp16", engine_build=_BUILDING
            ),
            "auth_token": "pk_deploy_x",
        },
    )
    created = deployments.create(
        "abcdef01-2345-6789-abcd-ef0123456789",
        name="prod",
        gpu_type="l4",
        precision="fp16",
        max_containers=3,
        inference_config={"confidence": 0.4},
    )
    body = json.loads(httpx_mock.get_requests()[-1].content)
    assert (body["gpu_type"], body["precision"], body["name"]) == ("l4", "fp16", "prod")
    assert body["max_containers"] == 3 and body["inference_config"] == {"confidence": 0.4}
    assert "compute_type" not in body
    # A model's first deployment on the GPU at that precision builds its engine first.
    build = created.deployment.engine_build
    assert build is not None and build.state == "building"
    assert build.models[0].estimate_s == pytest.approx(31.1)


def test_create_sends_a_request_ceiling_only_when_given(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    """B584: a deployment has no request ceiling unless one is set."""
    for _ in range(2):
        httpx_mock.add_response(
            method="POST",
            url=_PATH,
            json={"deployment": _payload(status="provisioning"), "auth_token": "pk_deploy_x"},
        )
    deployments.create("abcdef01-2345-6789-abcd-ef0123456789", rate_limit_rpm=6000)
    deployments.create("abcdef01-2345-6789-abcd-ef0123456789")
    first, second = (json.loads(r.content) for r in httpx_mock.get_requests())
    assert first["rate_limit_rpm"] == 6000
    assert "rate_limit_rpm" not in second, "none unless given"


def test_create_takes_no_compute_type(deployments: Deployments) -> None:
    """A CPU tier does not exist, so neither does the argument that chose it."""
    assert "compute_type" not in inspect.signature(Deployments.create).parameters
    with pytest.raises(TypeError):
        deployments.create("abcdef01-2345-6789-abcd-ef0123456789", compute_type="cpu")  # type: ignore[call-arg]


# ───────────── the Deployment model: the row the API answers ─────────────


def test_the_full_row_parses_with_its_prt_fields() -> None:
    dep = Deployment.model_validate(_payload(engine_build=_BUILT))
    assert dep.precision == "fp32" and dep.version_id == "0b1c2d3e-0000-4000-8000-000000000001"
    assert dep.cost_per_hour_micro_usd == 4759200 and dep.max_cost_per_hour_micro_usd == 4759200
    assert dep.accrued_cost_micro_usd == 1200 and dep.cost_rate_per_min == 79320
    assert dep.engine_build is not None and dep.engine_build.state == "built"
    assert dep.engine_build.models[0].build_s == pytest.approx(30.5)
    assert dep.error_message is None
    assert dep.updated_at is not None and dep.paused_at is None
    assert dep.metrics is not None and dep.metrics.requests == 12 and dep.metrics.errors == 1
    # The gateway writes the last request's time as epoch seconds in a string.
    assert dep.metrics.last_request_at is not None
    assert dep.metrics.last_request_at.year == 2025


def test_the_old_cost_fields_are_gone() -> None:
    """``cost_per_hour`` / ``accrued_cost_credits`` never matched the API: they always
    read their defaults. The real fields replace them."""
    fields = Deployment.model_fields
    assert "cost_per_hour" not in fields and "accrued_cost_credits" not in fields
    assert {
        "cost_per_hour_micro_usd",
        "max_cost_per_hour_micro_usd",
        "accrued_cost_micro_usd",
    } <= set(fields)


def test_a_failed_deployment_says_why() -> None:
    dep = Deployment.model_validate(
        _payload(status="failed", error_message="The model's engine could not be built: OOM")
    )
    assert dep.status == "failed" and dep.error_message is not None and "OOM" in dep.error_message


def test_a_deployment_from_before_the_gpu_only_engine_still_reads() -> None:
    legacy = {
        k: v
        for k, v in _payload(compute_type="cpu", gpu_type=None, status="paused").items()
        if k
        not in ("precision", "version_id", "engine_build", "metrics", "max_cost_per_hour_micro_usd")
    }
    dep = Deployment.model_validate(legacy)
    assert dep.compute_type == "cpu" and dep.gpu_type is None
    assert dep.precision == "fp32" and dep.engine_build is None and dep.metrics is None


def test_a_row_with_no_compute_type_reads_as_a_gpu() -> None:
    row = _payload()
    del row["compute_type"]
    assert Deployment.model_validate(row).compute_type == "gpu"


# ───────────── wait_until_active ─────────────

_NAME_LOOKUP = f"{_PATH}?limit=100&offset=0&name=prod-detector"


class _Sleeps:
    """A recording, non-sleeping ``sleep`` hook. Bounded: a wait that never stops fails the
    test instead of hanging the suite."""

    def __init__(self, limit: int = 50) -> None:
        self.calls: list[float] = []
        self._limit = limit

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        assert len(self.calls) <= self._limit, f"polled {len(self.calls)} times: it never stopped"


class _Clock:
    """Stands in for the ``time`` module the resource reads: ``monotonic`` steps through
    the given ticks (holding the last one), so a timeout is reached without waiting."""

    def __init__(self, *ticks: float) -> None:
        self._ticks = list(ticks)

    def monotonic(self) -> float:
        return self._ticks.pop(0) if len(self._ticks) > 1 else self._ticks[0]

    @staticmethod
    def sleep(_seconds: float) -> None:
        raise AssertionError("the sleep hook, not time.sleep, must be used")


def test_wait_until_active_polls_by_id_until_active(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    httpx_mock.add_response(
        url=_NAME_LOOKUP, json={"deployments": [_payload(name="prod-detector")]}
    )
    for row in (
        _payload(status="provisioning", engine_build=_BUILDING),
        _payload(status="provisioning", engine_build=_BUILT),
        _payload(status="active", engine_build=_BUILT),
    ):
        httpx_mock.add_response(method="GET", url=f"{_PATH}{DEP_ID}", json={"deployment": row})
    sleeps = _Sleeps()
    dep = deployments.wait_until_active("prod-detector", poll_interval=0.5, sleep=sleeps)
    assert dep.status == "active" and dep.engine_build is not None
    assert sleeps.calls == [0.5, 0.5]
    # The name is resolved ONCE; every poll is a GET by id.
    urls = [str(r.url) for r in httpx_mock.get_requests()]
    assert urls[0] == _NAME_LOOKUP and urls[1:] == [f"{_PATH}{DEP_ID}"] * 3


def test_wait_until_active_takes_the_deployment_create_returned(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    created = Deployment.model_validate(_payload(status="provisioning"))
    httpx_mock.add_response(method="GET", url=f"{_PATH}{DEP_ID}", json={"deployment": _payload()})
    assert deployments.wait_until_active(created, sleep=_Sleeps()).status == "active"
    assert len(httpx_mock.get_requests()) == 1, "no name lookup for a Deployment in hand"


def test_wait_until_active_raises_on_a_failed_deployment(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}{DEP_ID}",
        json={"deployment": _payload(status="failed", error_message="Provisioning failed: quota")},
    )
    sleeps = _Sleeps()
    with pytest.raises(ApiError) as exc:
        deployments.wait_until_active(DEP_ID, sleep=sleeps)
    assert exc.value.code == "job_failed"
    assert "Provisioning failed: quota" in str(exc.value)
    assert exc.value.response["id"] == DEP_ID and exc.value.response["status"] == "failed"
    assert exc.value.fix is not None and "create it again" in exc.value.fix
    assert sleeps.calls == [], "a failed deployment is not polled again"


@pytest.mark.parametrize(
    ("status", "says"),
    [("paused", "Resume it"), ("terminated", "Create a new deployment")],
)
def test_wait_until_active_raises_when_it_will_never_turn_active(
    httpx_mock: HTTPXMock, deployments: Deployments, status: str, says: str
) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{_PATH}{DEP_ID}", json={"deployment": _payload(status=status)}
    )
    with pytest.raises(ApiError) as exc:
        deployments.wait_until_active(DEP_ID, sleep=_Sleeps())
    assert exc.value.code == "job_cancelled"
    assert status in str(exc.value) and exc.value.fix is not None and says in exc.value.fix


def test_wait_until_active_times_out_with_what_it_waits_on(
    httpx_mock: HTTPXMock, deployments: Deployments, monkeypatch: pytest.MonkeyPatch
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}{DEP_ID}",
        json={"deployment": _payload(status="provisioning", engine_build=_BUILDING)},
        is_reusable=True,
    )
    monkeypatch.setattr("pictograph.resources.deployments.time", _Clock(0.0, 0.0, 5.0, 11.0))
    sleeps = _Sleeps()
    with pytest.raises(PollTimeoutError) as exc:
        deployments.wait_until_active(DEP_ID, timeout=10.0, poll_interval=5.0, sleep=sleeps)
    message = str(exc.value)
    assert "within 10s" in message and "provisioning" in message
    assert "building its TensorRT engine" in message
    assert sleeps.calls == [5.0, 5.0]


@pytest.mark.parametrize(("kwargs"), [{"timeout": 0}, {"poll_interval": 0}, {"timeout": -1}])
def test_wait_until_active_refuses_a_bad_window_before_any_request(
    httpx_mock: HTTPXMock, deployments: Deployments, kwargs: dict[str, float]
) -> None:
    with pytest.raises(ValueError):
        deployments.wait_until_active(DEP_ID, **kwargs)  # type: ignore[arg-type]
    assert httpx_mock.get_requests() == []


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


# ───────────── update: the autoscaling, after create ─────────────


@pytest.mark.parametrize(
    "changes",
    [
        {"min_containers": 2},
        {"max_containers": 4},
        {"scaledown_window": 300},
        {"min_containers": 0, "max_containers": 3, "scaledown_window": 120},
    ],
    ids=["min", "max", "scaledown", "all-three-with-a-zero"],
)
def test_update_sends_only_the_scaling_given(
    httpx_mock: HTTPXMock, deployments: Deployments, changes: dict[str, int]
) -> None:
    scaling = {"min_containers": 0, "max_containers": 1, "scaledown_window": 60, **changes}
    httpx_mock.add_response(
        method="PATCH", url=f"{_PATH}{DEP_ID}", json={"deployment": _payload(**scaling)}
    )
    dep = deployments.update(DEP_ID, **changes)
    # Exactly what was given, never a null for the rest: the route changes only what it is
    # sent. And a 0 is a value - min_containers=0 scales it to zero, it is not "not given".
    assert json.loads(httpx_mock.get_requests()[-1].content) == changes
    assert isinstance(dep, Deployment)
    assert (dep.min_containers, dep.max_containers, dep.scaledown_window) == (
        scaling["min_containers"],
        scaling["max_containers"],
        scaling["scaledown_window"],
    )


def test_update_renames_and_writes_the_readme_only_what_is_given(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    # B572: its name and README change like a workflow's; an empty README is a value (it clears).
    httpx_mock.add_response(
        method="PATCH",
        url=f"{_PATH}{DEP_ID}",
        json={"deployment": _payload(name="dock-gate", readme=None)},
    )
    dep = deployments.update(DEP_ID, name="dock-gate", readme="")
    assert json.loads(httpx_mock.get_requests()[-1].content) == {"name": "dock-gate", "readme": ""}
    assert (dep.name, dep.readme) == ("dock-gate", None)


@pytest.mark.parametrize("ceiling", [1200, 0], ids=["set", "removed-with-0"])
def test_update_sets_or_removes_the_request_ceiling(
    httpx_mock: HTTPXMock, deployments: Deployments, ceiling: int
) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{_PATH}{DEP_ID}",
        json={"deployment": _payload(rate_limit_rpm=ceiling or None)},
    )
    dep = deployments.update(DEP_ID, rate_limit_rpm=ceiling)
    assert json.loads(httpx_mock.get_requests()[-1].content) == {"rate_limit_rpm": ceiling}
    assert dep.rate_limit_rpm == (ceiling or None)


def test_a_deployment_reads_its_ceiling_and_its_data_transfer(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    httpx_mock.add_response(
        url=f"{_PATH}{DEP_ID}",
        json={
            "deployment": _payload(
                rate_limit_rpm=600, transfer_bytes=12 * 1024**3, transfer_cost_micro_usd=2880000
            )
        },
    )
    dep = deployments.get(DEP_ID)
    assert (dep.rate_limit_rpm, dep.transfer_bytes, dep.transfer_cost_micro_usd) == (
        600,
        12 * 1024**3,
        2880000,
    )


def test_a_deployment_from_before_its_transfer_was_metered_still_reads() -> None:
    row = {
        k: v
        for k, v in _payload().items()
        if k not in ("rate_limit_rpm", "transfer_bytes", "transfer_cost_micro_usd")
    }
    dep = Deployment.model_validate(row)
    assert (dep.rate_limit_rpm, dep.transfer_bytes, dep.transfer_cost_micro_usd) == (None, 0, 0)


def test_a_deployment_reads_its_readme(httpx_mock: HTTPXMock, deployments: Deployments) -> None:
    httpx_mock.add_response(
        url=f"{_PATH}{DEP_ID}", json={"deployment": _payload(readme="# Gate cameras")}
    )
    assert deployments.get(DEP_ID).readme == "# Gate cameras"


def test_update_resolves_a_name_once_then_patches_by_id(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    httpx_mock.add_response(
        url=_NAME_LOOKUP, json={"deployments": [_payload(name="prod-detector")]}
    )
    httpx_mock.add_response(
        method="PATCH",
        url=f"{_PATH}{DEP_ID}",
        json={"deployment": _payload(name="prod-detector", status="paused", max_containers=3)},
    )
    dep = deployments.update("prod-detector", max_containers=3)
    # A paused deployment answers with the scaling it will be resumed with.
    assert (dep.status, dep.max_containers) == ("paused", 3)
    assert [(r.method, str(r.url)) for r in httpx_mock.get_requests()] == [
        ("GET", _NAME_LOOKUP),
        ("PATCH", f"{_PATH}{DEP_ID}"),
    ]


@pytest.mark.parametrize(
    "nothing",
    [{}, {"min_containers": None, "max_containers": None, "scaledown_window": None}],
    ids=["no-arguments", "all-none"],
)
def test_update_with_nothing_to_change_raises_before_any_request(
    httpx_mock: HTTPXMock, deployments: Deployments, nothing: dict[str, None]
) -> None:
    with pytest.raises(ValueError, match="Nothing to change") as raised:
        deployments.update("prod-detector", **nothing)
    assert "rate_limit_rpm" in str(raised.value), "every field it takes is named"
    assert httpx_mock.get_requests() == [], "not even the name lookup"


def test_update_retries_the_live_servers_refusal(httpx_mock: HTTPXMock) -> None:
    """A 502 means the running deployment refused the change and NOTHING was changed (the
    API puts its row back), so a retry is safe and is what the API asks for. The SDK makes
    it, with the same Idempotency-Key - a 5xx is never replayed from the key's cache."""
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=1)  # type: ignore[arg-type]
    retrying = Transport(
        config, api_key=KEY, retry_policy=RetryPolicy(max_retries=1, sleep=lambda _s: None)
    )
    httpx_mock.add_response(
        method="PATCH",
        url=f"{_PATH}{DEP_ID}",
        status_code=502,
        json={
            "detail": "The running deployment's scaling could not be changed, so nothing was "
            "changed. Try again."
        },
    )
    httpx_mock.add_response(
        method="PATCH",
        url=f"{_PATH}{DEP_ID}",
        json={"deployment": _payload(min_containers=1)},
    )
    try:
        assert Deployments(retrying).update(DEP_ID, min_containers=1).min_containers == 1
    finally:
        retrying.close()
    first, second = httpx_mock.get_requests()
    assert first.headers["Idempotency-Key"] == second.headers["Idempotency-Key"]
    assert json.loads(first.content) == json.loads(second.content) == {"min_containers": 1}


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


_OPTION_T4 = {
    "key": "t4",
    "label": "T4 GPU",
    "compute_type": "gpu",
    "gpu_type": "t4",
    "is_gpu": True,
    "description": "Entry GPU. Great for detection and classification at low cost.",
    "vram_gb": 16,
    "cpu_cores": 4.0,
    "memory_gib": 16.0,
    "rate_per_min_micro_usd": 79320,
    "cost_per_hour_micro_usd": 4759200,
}


_PRECISIONS = [
    {"precision": "fp32", "label": "FP32", "description": "32-bit, the model as trained"},
    {"precision": "fp16", "label": "FP16", "description": "16-bit, faster, accuracy-equivalent"},
]


_REGIONS = [
    {
        "id": "us-east",
        "label": "US East",
        "location": "Virginia",
        "ping_url": "https://workflows.pictograph.io/us-east/ping",
    },
    {
        "id": "us-west",
        "label": "US West",
        "location": "Oregon",
        "ping_url": "https://workflows.pictograph.io/us-west/ping",
    },
]


def test_create_runs_where_it_is_told_else_auto(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    """B578: ``region`` is sent as given; left out it is ``auto`` - the server picks the region
    nearest the call - and the deployment answered says where it runs."""
    for asked, sent in (("us-west", "us-west"), (None, "auto")):
        httpx_mock.add_response(
            method="POST",
            url=_PATH,
            json={
                "deployment": _payload(status="provisioning", region="us-west"),
                "auth_token": "pk_deploy_x",
            },
        )
        kwargs: dict[str, Any] = {} if asked is None else {"region": asked}
        created = deployments.create("abcdef01-2345-6789-abcd-ef0123456789", **kwargs)
        body = json.loads(httpx_mock.get_requests()[-1].content)
        assert body["region"] == sent
        assert created.deployment.region == "us-west"


def test_a_quote_and_the_options_are_for_a_region(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}quote?gpu_type=t4&min_containers=0&max_containers=1&region=us-west",
        json={"success": True, "quote": _quote()},
    )
    assert (
        deployments.quote(region="us-west").rate_per_min_micro_usd
        == _quote()["rate_per_min_micro_usd"]
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}compute-options?region=us-west",
        json={"success": True, "region": "us-west", "regions": _REGIONS, "options": [_OPTION_T4]},
    )
    offer = deployments.compute_options(region="us-west")
    assert offer.region == "us-west"
    assert [r.id for r in offer.regions] == ["us-east", "us-west"]
    assert offer.regions[1].label == "US West" and offer.regions[1].location == "Oregon"
    assert offer.regions[1].ping_url == "https://workflows.pictograph.io/us-west/ping"


def test_a_server_without_regions_still_reads() -> None:
    """A ``compute-options`` answer from before regions (no ``regions``) parses empty."""
    offer = DeploymentComputeOptions.model_validate({"options": [_OPTION_T4]})
    assert offer.regions == [] and offer.region is None


def test_compute_options_are_the_gpu_tiers_and_the_precisions(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    options = [
        _OPTION_T4,
        {**_OPTION_T4, "key": "l4", "label": "L4 GPU", "gpu_type": "l4", "vram_gb": 24},
        {
            **_OPTION_T4,
            "key": "a100",
            "label": "A100 GPU",
            "gpu_type": "a100",
            "vram_gb": 40,
            "cpu_cores": 8.0,
            "memory_gib": 32.0,
        },
    ]
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}compute-options?region=auto",
        json={"success": True, "options": options, "precisions": _PRECISIONS},
    )
    offer = deployments.compute_options()
    assert isinstance(offer, DeploymentComputeOptions)
    opts = offer.options
    assert [o.key for o in opts] == ["t4", "l4", "a100"]
    assert all(isinstance(o, DeploymentComputeOption) for o in opts)
    assert all(o.is_gpu and o.compute_type == "gpu" for o in opts)
    t4 = opts[0]
    assert (t4.vram_gb, t4.cpu_cores, t4.memory_gib) == (16, 4.0, 16.0)
    assert (t4.rate_per_min_micro_usd, t4.cost_per_hour_micro_usd) == (79320, 4759200)
    assert opts[2].cpu_cores == 8.0 and opts[2].memory_gib == 32.0
    # The precisions, in the server's words, the default (fp32) first.
    assert all(isinstance(p, DeploymentComputePrecision) for p in offer.precisions)
    assert [(p.precision, p.label) for p in offer.precisions] == [
        ("fp32", "FP32"),
        ("fp16", "FP16"),
    ]
    assert offer.precisions[1].description == "16-bit, faster, accuracy-equivalent"


def test_compute_options_from_a_server_that_sends_no_precisions(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    """An older API answers only ``options``: they parse, and ``precisions`` is empty."""
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}compute-options?region=auto",
        json={"success": True, "options": [_OPTION_T4]},
    )
    offer = deployments.compute_options()
    assert [o.key for o in offer.options] == ["t4"] and offer.precisions == []


def test_a_compute_option_without_its_vestigial_keys_reads_as_a_gpu() -> None:
    """``compute_type`` / ``is_gpu`` are the same on every tier; a server that stops
    sending them must not break a read."""
    opt = DeploymentComputeOption.model_validate(
        {k: v for k, v in _OPTION_T4.items() if k not in ("compute_type", "is_gpu")}
    )
    assert opt.compute_type == "gpu" and opt.is_gpu is True


def _quote(**overrides: Any) -> dict[str, Any]:
    """``pricing.realtime_quote``'s answer, every key."""
    base: dict[str, Any] = {
        "rate_per_min_micro_usd": 79320,
        "container_cost_per_hour_micro_usd": 4759200,
        "cost_per_hour_micro_usd": 4759200,
        "cost_per_day_micro_usd": 114220800,
        "warm_cost_per_hour_micro_usd": 0,
        "warm_cost_per_day_micro_usd": 0,
        "max_cost_per_hour_micro_usd": 4759200,
        "transfer_per_gib_micro_usd": 240000,
        "scale_to_zero": True,
        "billing_note": "Charged only while a container is up; it scales to zero when idle.",
    }
    base.update(overrides)
    return base


def test_quote_default_scale_to_zero(httpx_mock: HTTPXMock, deployments: Deployments) -> None:
    # A bare quote prices create's own defaults: a T4, scale to zero, one container.
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}quote?gpu_type=t4&min_containers=0&max_containers=1&region=auto",
        json={"success": True, "quote": _quote()},
    )
    q = deployments.quote()
    assert isinstance(q, DeploymentQuote)
    assert q.scale_to_zero is True
    assert q.cost_per_day_micro_usd == q.cost_per_hour_micro_usd * 24
    assert q.warm_cost_per_hour_micro_usd == 0 and q.warm_cost_per_day_micro_usd == 0
    assert q.container_cost_per_hour_micro_usd == q.max_cost_per_hour_micro_usd == 4759200
    assert q.transfer_per_gib_micro_usd == 240000, "its requests' data, per GiB (B584)"


def test_quote_passes_the_gpu_and_both_container_bounds(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}quote?gpu_type=l4&min_containers=2&max_containers=3&region=auto",
        json={
            "success": True,
            "quote": _quote(
                rate_per_min_micro_usd=97695,
                container_cost_per_hour_micro_usd=5861700,
                cost_per_hour_micro_usd=11723400,
                cost_per_day_micro_usd=281361600,
                warm_cost_per_hour_micro_usd=11723400,
                warm_cost_per_day_micro_usd=281361600,
                max_cost_per_hour_micro_usd=17585100,
                scale_to_zero=False,
                billing_note="Charged continuously for the containers kept warm.",
            ),
        },
    )
    q = deployments.quote(gpu_type="l4", min_containers=2, max_containers=3)
    assert q.scale_to_zero is False and q.rate_per_min_micro_usd == 97695
    assert q.max_cost_per_hour_micro_usd == 3 * q.container_cost_per_hour_micro_usd  # type: ignore[operator]
    assert "compute_type" not in str(httpx_mock.get_requests()[-1].url)


def test_a_quote_without_the_newer_keys_still_reads() -> None:
    """An older API answers only the original five keys: they parse, the rest are None."""
    q = DeploymentQuote.model_validate(
        {
            "rate_per_min_micro_usd": 1,
            "cost_per_hour_micro_usd": 60,
            "cost_per_day_micro_usd": 1440,
            "scale_to_zero": True,
            "billing_note": "x",
        }
    )
    assert q.max_cost_per_hour_micro_usd is None and q.warm_cost_per_hour_micro_usd is None
    assert q.transfer_per_gib_micro_usd is None


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

    With no headers at all (a gateway from before PRT): `backend` is ONNX Runtime,
    what every such gateway served. `device` degrades to `remote`, NOT `cpu` -
    reporting a T4 deployment as CPU would be a silent falsehood. `inference_ms`
    stays None because nothing on the wire measured it, and the round-trip is a
    different measurement.
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
    with pytest.raises(ValidationError, match="serves a 'classification' model") as exc:
        client.infer(image=b"x")
    # The caller's argument is wrong, so the fix says what to pass - never "retry" or
    # "contact support" (a ServerError's), and the CLI exits "invalid input" (B569).
    assert exc.value.fix is not None and "task='classification'" in exc.value.fix
    assert not isinstance(exc.value, ServerError)


def test_unknown_model_type_raises(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=_EP, json={"model_type": "wat", "predictions": []})
    client = DeploymentClient(_EP, "pk_deploy_tok")
    with pytest.raises(ServerError, match="unknown model_type") as exc:
        client.infer(image=b"x")
    assert exc.value.fix is not None and "pip install -U pictograph" in exc.value.fix


def test_missing_model_type_falls_back_to_the_declared_task(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=_EP, json={"predictions": [_BBOX]})
    client = DeploymentClient(_EP, "pk_deploy_tok", task="object_detection")
    assert client.infer(image=b"x").predictions[0].name == "dog"


def test_missing_model_type_without_a_declared_task_raises(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=_EP, json={"predictions": [_BBOX]})
    client = DeploymentClient(_EP, "pk_deploy_tok")
    with pytest.raises(ServerError, match="no model_type") as exc:
        client.infer(image=b"x")
    assert exc.value.fix is not None and "infer_raw()" in exc.value.fix


def test_bad_task_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="Unknown task"):
        DeploymentClient(_EP, "pk_deploy_tok", task="segmentation")  # type: ignore[call-overload]


def test_connect_passes_the_task_through() -> None:
    dep = Deployment.model_validate(_payload())
    client = Deployments.connect(dep, "pk_deploy_x", task="keypoint_detection")
    assert client.task == "keypoint_detection"
    assert repr(client).endswith("task='keypoint_detection')")


# ───────────── PRT provenance: what the gateway's headers report ─────────────

_PRT_HEADERS = {
    "X-Pictograph-Device": "cuda",
    "X-Pictograph-Engine": "prt",
    "Server-Timing": "decode;dur=1.204, infer;dur=13.380, total;dur=15.020",
    "X-Request-Id": "8f9a6b1e-0000-4000-8000-000000000001",
}


@pytest.mark.parametrize("task", sorted(_WIRE_BY_TASK))
def test_a_prt_answer_reports_tensorrt_on_cuda_with_its_engine_time(
    httpx_mock: HTTPXMock, task: str
) -> None:
    httpx_mock.add_response(method="POST", url=_EP, json=_WIRE_BY_TASK[task], headers=_PRT_HEADERS)
    with DeploymentClient(_EP, "pk_deploy_tok") as client:
        result = client.infer(image=b"x")
    assert isinstance(result, _RESULT_BY_TASK[task]), "the result shape is unchanged"
    assert (result.backend, result.device) == ("tensorrt", "cuda")
    assert result.inference_ms == pytest.approx(13.38), "the engine's own infer time"
    assert result.providers == []


def test_infer_raw_is_the_body_alone(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST", url=_EP, json=_WIRE_BY_TASK["object_detection"], headers=_PRT_HEADERS
    )
    assert (
        DeploymentClient(_EP, "pk_deploy_tok").infer_raw(image=b"x")
        == _WIRE_BY_TASK["object_detection"]
    )


@pytest.mark.parametrize(
    ("engine", "backend"),
    [
        (None, "onnxruntime"),
        ("", "onnxruntime"),
        ("prt", "tensorrt"),
        (" PRT ", "tensorrt"),
        ("tensorrt", "tensorrt"),
        ("onnxruntime", "onnxruntime"),
        ("prt-2", "tensorrt"),
    ],
)
def test_the_engine_header_names_the_runtime(engine: str | None, backend: str) -> None:
    assert backend_of(engine) == backend


@pytest.mark.parametrize(
    ("header", "metric", "ms"),
    [
        ("decode;dur=1.2, infer;dur=13.38, total;dur=15", "infer", 13.38),
        ("decode;dur=1.2, infer;dur=13.38, total;dur=15", "decode", 1.2),
        ('infer;desc="TensorRT";dur="7.5"', "infer", 7.5),
        ("INFER;DUR=2", "infer", 2.0),
        (None, "infer", None),
        ("", "infer", None),
        ("decode;dur=1.2", "infer", None),
        ("infer", "infer", None),
        ("infer;desc=x", "infer", None),
        ("infer;dur=abc", "infer", None),
        ("infer;dur=-1", "infer", None),
        ("infer;dur=nan", "infer", None),
        ("infer;dur=inf", "infer", None),
    ],
)
def test_server_timing_reads_a_metrics_duration(
    header: str | None, metric: str, ms: float | None
) -> None:
    got = server_timing_ms(header, metric)
    assert got == (pytest.approx(ms) if ms is not None else None)


def test_the_default_timeout_outlasts_the_gateways_cold_start_hold() -> None:
    """The gateway holds a scaled-to-zero deployment's first request for up to 120 s
    while its container starts; a shorter default abandoned requests it was about to
    answer, and made its ``deployment_starting`` answer unreachable."""
    assert DEFAULT_TIMEOUT_S > 120
    client = DeploymentClient(_EP, "pk_deploy_tok")
    assert client._timeout == DEFAULT_TIMEOUT_S
    for method in (DeploymentClient.__init__, Deployments.connect):
        assert inspect.signature(method).parameters["timeout"].default == DEFAULT_TIMEOUT_S


def test_the_default_retry_honours_a_503s_retry_after() -> None:
    client = DeploymentClient(_EP, "pk_deploy_tok")
    assert client._retry.retry_after_statuses == frozenset({429, 503})
    assert client._retry.max_retries == 2


# ───────────── the gateway's errors: typed, with its slug as the code ─────────────


class _RecordingSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _client_with(sleeps: _RecordingSleep, max_retries: int = 0) -> DeploymentClient[Any]:
    return DeploymentClient(
        _EP,
        "pk_deploy_tok",
        retry_policy=RetryPolicy(
            max_retries=max_retries,
            retry_after_statuses={429, 503},
            sleep=sleeps,
            rng=lambda _lo, _hi: 1.0,
        ),
    )


def test_a_deployment_starting_503_is_retried_after_its_retry_after(httpx_mock: HTTPXMock) -> None:
    starting = {
        "error": "deployment_starting",
        "detail": "The deployment is starting; retry shortly.",
    }
    httpx_mock.add_response(
        method="POST", url=_EP, status_code=503, json=starting, headers={"Retry-After": "5"}
    )
    httpx_mock.add_response(
        method="POST", url=_EP, json=_WIRE_BY_TASK["object_detection"], headers=_PRT_HEADERS
    )
    sleeps = _RecordingSleep()
    result = _client_with(sleeps, max_retries=2).infer(image=b"x")
    assert result.backend == "tensorrt"
    assert sleeps.calls == [5.0], "the gateway's Retry-After, not the 1s backoff"


def test_a_deployment_that_never_starts_says_so(httpx_mock: HTTPXMock) -> None:
    starting = {
        "error": "deployment_starting",
        "detail": "The deployment is starting; retry shortly.",
    }
    httpx_mock.add_response(
        method="POST",
        url=_EP,
        status_code=503,
        json=starting,
        headers={"Retry-After": "5", "X-Request-Id": "req-1"},
        is_reusable=True,
    )
    sleeps = _RecordingSleep()
    with pytest.raises(ServerError) as exc:
        _client_with(sleeps, max_retries=2).infer(image=b"x")
    err = exc.value
    assert err.status_code == 503 and err.code == "deployment_starting"
    assert err.request_id == "req-1"
    assert err.message == "The deployment is starting; retry shortly.", "the body's own words"
    assert err.fix is not None and "min_containers=1" in err.fix
    assert len(httpx_mock.get_requests()) == 3 and sleeps.calls == [5.0, 5.0]


def test_a_deployment_that_is_not_active_names_its_status(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=_EP,
        status_code=409,
        json={"error": "deployment_not_active", "status": "provisioning"},
    )
    with pytest.raises(ConflictError) as exc:
        _client_with(_RecordingSleep()).infer(image=b"x")
    err = exc.value
    assert err.code == "deployment_not_active"
    assert err.message == "The deployment is not active (status: provisioning)."
    assert err.fix is not None and "wait_until_active" in err.fix and "resume" in err.fix


def test_the_rate_limit_says_how_long_and_how_to_raise_it(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=_EP,
        status_code=429,
        json={
            "error": "rate_limited",
            "detail": "Per-deployment request ceiling reached; retry shortly.",
        },
        headers={"Retry-After": "12"},
    )
    with pytest.raises(RateLimitError) as exc:
        _client_with(_RecordingSleep()).infer(image=b"x")
    err = exc.value
    assert err.code == "rate_limited" and err.retry_after == 12
    assert err.fix is not None and "Wait 12s" in err.fix and "rate_limit_rpm" in err.fix
    assert "300" not in err.fix and "inference_config" not in err.fix, (
        "the ceiling is the deployment's own (B584)"
    )


def test_a_rate_limit_without_retry_after_still_says_what_to_do(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=_EP, status_code=429, json={"error": "rate_limited"})
    with pytest.raises(RateLimitError) as exc:
        _client_with(_RecordingSleep()).infer(image=b"x")
    assert exc.value.fix is not None and exc.value.fix.startswith("Retry shortly")
    assert exc.value.message == "The deployment's request ceiling was reached."


@pytest.mark.parametrize(
    ("status", "slug", "exc_cls", "fix_says"),
    [
        (401, "unauthorized", AuthError, "pk_deploy_"),
        (404, "deployment_not_found", NotFoundError, "endpoint_url"),
        (404, "not_found", NotFoundError, "/predict"),
        (404, "wrong_region", NotFoundError, "endpoint_url"),
        (503, "overloaded", ServerError, "max_containers"),
        (503, "stopping", ServerError, "another container"),
        (503, "warming_up", ServerError, "Retry shortly"),
        (504, "inference_timeout", ServerError, "gpu_type"),
        (502, "inference_failed", ServerError, "request_id"),
        (502, "deployment_unreachable", ServerError, "request_id"),
        (502, "deployment_unavailable", ServerError, "request_id"),
        (503, "gateway_unavailable", ServerError, "request_id"),
        (400, "url_images_not_supported", ValidationError, "file or bytes"),
        (400, "could_not_decode_image", ValidationError, "valid image"),
        (400, "no_image_provided", ValidationError, "64 MiB"),
        (400, "invalid_request_body", ValidationError, "multipart"),
        (400, "invalid_option", ValidationError, "class_filter"),
        (413, "request_too_large", ApiError, "smaller image"),
    ],
)
def test_every_gateway_slug_is_the_code_with_a_fix(
    httpx_mock: HTTPXMock, status: int, slug: str, exc_cls: type[ApiError], fix_says: str
) -> None:
    httpx_mock.add_response(method="POST", url=_EP, status_code=status, json={"error": slug})
    with pytest.raises(exc_cls) as exc:
        _client_with(_RecordingSleep()).infer(image=b"x")
    err = exc.value
    assert err.status_code == status and err.code == slug
    assert err.message != slug, "a body with no words of its own gets a sentence, not a slug"
    assert err.fix is not None and fix_says in err.fix


def test_an_unknown_slug_is_still_the_code(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST", url=_EP, status_code=502, json={"error": "something_new", "detail": "Hm."}
    )
    with pytest.raises(ServerError) as exc:
        _client_with(_RecordingSleep()).infer(image=b"x")
    assert exc.value.code == "something_new" and exc.value.message == "Hm."
    assert exc.value.fix == ServerError.fix, "an unknown slug keeps the class's own fix"


def test_an_api_envelope_code_is_never_overwritten(httpx_mock: HTTPXMock) -> None:
    body = {"error": {"code": "insufficient_credits", "message": "Out of credit."}}
    httpx_mock.add_response(method="POST", url=_EP, status_code=402, json=body)
    with pytest.raises(PaymentRequiredError) as exc:
        _client_with(_RecordingSleep()).infer(image=b"x")
    assert exc.value.code == "insufficient_credits" and exc.value.message == "Out of credit."


def test_a_plain_text_error_stays_a_plain_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=_EP, status_code=500, text="upstream exploded")
    with pytest.raises(ServerError) as exc:
        _client_with(_RecordingSleep()).infer(image=b"x")
    assert exc.value.code is None


# ───────────── connect, against a provisioning deployment ─────────────


def test_connect_builds_a_client_for_a_provisioning_deployment() -> None:
    """On PRT the endpoint exists from create: connecting is allowed, and calling
    answers ``deployment_not_active`` until the deployment is active."""
    dep = Deployment.model_validate(_payload(status="provisioning"))
    client = Deployments.connect(dep, "pk_deploy_x")
    assert repr(client).startswith(
        "DeploymentClient(endpoint='https://gateway.test/dddddddd/predict'"
    )


def test_connect_refuses_a_deployment_with_no_endpoint() -> None:
    dep = Deployment.model_validate(_payload(endpoint_url=None))
    with pytest.raises(ValueError, match="no endpoint_url"):
        Deployments.connect(dep, "pk_deploy_x")


# ───────────── runtime: what an active deployment is doing right now ─────────────

_CHECKED = "2026-10-23T12:05:00+00:00"

_RUNTIMES: dict[str, dict[str, Any]] = {
    "live": {
        "state": "live",
        "containers": 2,
        "creating": 0,
        "waiting_since": None,
        "checked_at": _CHECKED,
    },
    "starting": {
        "state": "starting",
        "containers": 0,
        "creating": 1,
        "waiting_since": "2026-10-23T12:04:51+00:00",
        "checked_at": _CHECKED,
    },
    "waiting": {
        "state": "waiting",
        "containers": 0,
        "creating": 0,
        "waiting_since": "2026-10-23T12:03:18+00:00",
        "checked_at": _CHECKED,
    },
    "asleep": {
        "state": "asleep",
        "containers": 0,
        "creating": 0,
        "waiting_since": None,
        "checked_at": _CHECKED,
    },
    "unknown": {
        "state": "unknown",
        "containers": 0,
        "creating": 0,
        "waiting_since": None,
        "checked_at": _CHECKED,
    },
}


@pytest.mark.parametrize("state", sorted(_RUNTIMES))
def test_get_reads_each_live_state(
    httpx_mock: HTTPXMock, deployments: Deployments, state: str
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}{DEP_ID}",
        json={"deployment": _payload(runtime=_RUNTIMES[state])},
    )
    runtime = deployments.get(DEP_ID).runtime
    assert isinstance(runtime, DeploymentRuntime)
    wire = _RUNTIMES[state]
    assert (runtime.state, runtime.containers, runtime.creating) == (
        state,
        wire["containers"],
        wire["creating"],
    )
    assert runtime.checked_at is not None and runtime.checked_at.isoformat() == _CHECKED
    if wire["waiting_since"] is None:
        assert runtime.waiting_since is None
    else:
        assert runtime.waiting_since is not None
        assert runtime.waiting_since.isoformat() == wire["waiting_since"]


def test_a_deployments_streams_are_read_apart_from_its_requests() -> None:
    """Its stream sessions: the containers serving them (``runtime.streams``, never ``containers``) and
    the streams it runs at once (``stream_limit``); a deployment made before streams has a limit of 0."""
    dep = Deployment.model_validate(
        _payload(runtime={**_RUNTIMES["asleep"], "streams": 2}, stream_limit=12)
    )
    assert dep.runtime is not None
    assert (dep.runtime.state, dep.runtime.containers, dep.runtime.streams) == ("asleep", 0, 2)
    assert dep.stream_limit == 12
    old = Deployment.model_validate(_payload(runtime=_RUNTIMES["live"]))
    assert old.stream_limit == 0 and old.runtime is not None and old.runtime.streams == 0


def test_a_state_added_later_still_parses() -> None:
    """``state`` is a ``str``, not a Literal: a word this SDK does not know yet must not break a
    read of the whole deployment."""
    dep = Deployment.model_validate(
        _payload(runtime={**_RUNTIMES["asleep"], "state": "hibernating", "a_new_key": True})
    )
    assert dep.runtime is not None and dep.runtime.state == "hibernating"


@pytest.mark.parametrize("status", ["provisioning", "paused", "failed", "terminated"])
def test_runtime_is_none_unless_the_deployment_is_active(status: str) -> None:
    assert Deployment.model_validate(_payload(status=status, runtime=None)).runtime is None


def test_an_active_deployment_from_a_server_without_runtime_reads_none() -> None:
    row = _payload()
    row.pop("runtime", None)
    assert Deployment.model_validate(row).runtime is None


def test_a_runtime_missing_its_counts_reads_zero() -> None:
    dep = Deployment.model_validate(_payload(runtime={"state": "unknown"}))
    assert dep.runtime is not None
    assert (dep.runtime.containers, dep.runtime.creating) == (0, 0)
    assert dep.runtime.waiting_since is None and dep.runtime.checked_at is None


def test_list_and_the_lifecycle_answers_carry_runtime(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    httpx_mock.add_response(
        url=f"{_PATH}?limit=50&offset=0",
        json={
            "deployments": [
                _payload(runtime=_RUNTIMES["live"]),
                _payload(id="d2", name="other", status="paused", runtime=None),
            ]
        },
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{_PATH}{DEP_ID}/pause",
        json={"deployment": _payload(status="paused", runtime=None)},
    )
    httpx_mock.add_response(
        method="POST",
        url=_PATH,
        json={
            "deployment": _payload(status="provisioning", runtime=None),
            "auth_token": "pk_deploy_x",
        },
    )
    listed = deployments.list()
    assert [d.runtime.state if d.runtime else None for d in listed] == ["live", None]
    assert deployments.pause(DEP_ID).runtime is None
    assert deployments.create("abcdef01-2345-6789-abcd-ef0123456789").deployment.runtime is None
