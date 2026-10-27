"""A workflow served live (B579): ``client.workflows``' endpoint methods and a stream frame's measures.

The workflow's endpoint - made by its deploy (never its create) in a region, paused, resumed, its scaling changed,
waited on until it is live, its live log - and the direct client its streams open on (a deployment's, with the
endpoint's token).
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.async_transport import AsyncTransport
from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.aio.resources.workflows import AsyncWorkflows
from pictograph.exceptions import ApiError, PollTimeoutError
from pictograph.models.workflow import Workflow
from pictograph.resources._deployment_client import DeploymentClient
from pictograph.resources._deployment_stream import StreamFrame
from pictograph.resources.workflows import Workflows

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
API = f"{BASE}/api/v1/developer/workflows"
WID = "fedcba98-1111-2222-3333-444455556666"
ENDPOINT_URL = "https://infer.test/us-west/cdcdcdcd/predict"
TOKEN_X = "pk_deploy_x"  # noqa: S105 - test fixture, not a real secret
TOKEN_Y = "pk_deploy_y"  # noqa: S105 - test fixture, not a real secret
TOKEN_Z = "pk_deploy_z"  # noqa: S105 - test fixture, not a real secret


@pytest.fixture
def transport() -> Transport:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def workflows(transport: Transport) -> Workflows:
    return Workflows(transport)


def _endpoint(**o: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "ep-1",
        "status": "active",
        "region": "us-west",
        "gpu_type": "t4",
        "min_containers": 0,
        "max_containers": 2,
        "scaledown_window": 120,
        "endpoint_url": ENDPOINT_URL,
        "auth_token_prefix": "pk_deploy_abc…",
        "error_message": None,
        "engine_build": None,
        "stream_limit": 8,
        "deployed": True,
        "runtime": {"state": "asleep", "containers": 0, "creating": 0, "streams": 0},
        "live": {
            "models": [
                {
                    "model_id": "m1",
                    "name": "Birds",
                    "precision": "fp16",
                    "model_type": "object_detection",
                }
            ],
            "measures": [
                {"id": "m0", "name": "Gate", "kind": "line", "measure": "line_cross", "label": "L1"}
            ],
            "classes": ["car"],
            "class_colors": {"car": "#ff0000"},
            "class_types": {"car": "bbox"},
            "events": [
                {
                    "trigger": "crossing-0a1b2c3d",
                    "key": "crossing:ms:l1",
                    "kind": "crossing",
                    "name": "Crossing · Gate",
                    "captured": True,
                },
                {
                    "trigger": "passive-12345678",
                    "key": "passive:a",
                    "kind": "passive",
                    "name": "Passive capture",
                    "captured": True,
                },
            ],
        },
        "live_outputs": {
            "events": 12,
            "deliveries": 4,
            "saved": 9,
            "not_saved": 1,
            "skipped": 2,
            "since": "2026-10-26T10:00:00Z",
            "last_not_saved_reason": "The organization is at its image limit.",
            "last_not_saved_at": "2026-10-26T10:05:00Z",
        },
        "cost_per_hour_micro_usd": 4_260_000,
        "max_cost_per_hour_micro_usd": 8_520_000,
        "transfer_per_gib_micro_usd": 240_000,
        "accrued_cost_micro_usd": 0,
        "uptime_seconds": 0,
        "transfer_bytes": 0,
        "transfer_cost_micro_usd": 0,
        "created_at": "2026-10-26T00:00:00Z",
    }
    base.update(o)
    return base


def _wf(**o: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": WID,
        "organization_id": "org-1",
        "name": "Gate counter",
        "graph": {},
        "status": "draft",
        "endpoint": _endpoint(),
    }
    base.update(o)
    return base


# ── create never deploys; deploy does: where it is served, its token once ────


def test_create_never_deploys_it(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    """A create makes the workflow, never its endpoint (its deploy does): it sends no region or scaling, and its
    answer is a workflow not deployed, with no token."""
    httpx_mock.add_response(method="POST", url=f"{API}/", json={"workflow": _wf(endpoint=None)})
    wf = workflows.create("Gate counter", template_key="line_counter")
    (request,) = httpx_mock.get_requests()
    assert json.loads(request.content) == {"name": "Gate counter", "template_key": "line_counter"}
    assert wf.endpoint is None and wf.endpoint_token is None


@pytest.mark.parametrize("method", [Workflows.create, AsyncWorkflows.create])
def test_create_takes_no_endpoint_settings(method: Any) -> None:
    """Where it is served and how it scales are its deploy's arguments, never its create's."""
    params = inspect.signature(method).parameters
    assert not {"region", "min_containers", "max_containers", "scaledown_window"} & set(params)


def test_a_read_answers_its_endpoint_and_carries_no_token(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(method="GET", url=f"{API}/{WID}", json={"workflow": _wf()})
    wf = workflows.get(WID)
    assert wf.endpoint_token is None
    assert (
        wf.endpoint is not None
        and wf.endpoint.status == "active"
        and wf.endpoint.live.measures[0].label == "L1"
    )
    assert wf.endpoint.live.models[0].precision == "fp16"
    assert wf.endpoint.live.class_types == {"car": "bbox"}
    # B579 slice 1: the events its live outputs act on, and what they did.
    assert [(e.kind, e.name, e.captured) for e in wf.endpoint.live.events] == [
        ("crossing", "Crossing · Gate", True),
        ("passive", "Passive capture", True),
    ]
    lo = wf.endpoint.live_outputs
    assert lo is not None and (lo.events, lo.deliveries, lo.saved, lo.not_saved, lo.skipped) == (
        12,
        4,
        9,
        1,
        2,
    )
    assert lo.last_not_saved_reason == "The organization is at its image limit."
    assert lo.since is not None and lo.last_not_saved_at is not None


def test_an_endpoint_whose_live_outputs_never_acted_has_none(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/{WID}",
        json={"workflow": _wf(endpoint=_endpoint(live_outputs=None))},
    )
    ep = workflows.get(WID).endpoint
    assert ep is not None and ep.live_outputs is None


def test_a_workflow_never_deployed_has_no_endpoint(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(method="GET", url=f"{API}/{WID}", json={"workflow": _wf(endpoint=None)})
    assert workflows.get(WID).endpoint is None


def test_deploy_sends_its_region_and_scaling(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/{WID}/deploy",
        json={
            "workflow": _wf(endpoint=_endpoint(status="provisioning")),
            "endpoint_token": TOKEN_Y,
        },
    )
    wf = workflows.deploy(WID, region="us-east", min_containers=1)
    (request,) = httpx_mock.get_requests()
    assert json.loads(request.content) == {"region": "us-east", "min_containers": 1}
    assert (
        wf.endpoint_token == TOKEN_Y
        and wf.endpoint is not None
        and wf.endpoint.status == "provisioning"
    )


@pytest.mark.parametrize(
    "verb, method, path", [("pause", "POST", "/pause"), ("resume", "POST", "/resume")]
)
def test_pause_and_resume(
    httpx_mock: HTTPXMock, workflows: Workflows, verb: str, method: str, path: str
) -> None:
    httpx_mock.add_response(
        method=method,
        url=f"{API}/{WID}{path}",
        json={"workflow": _wf(endpoint=_endpoint(status="paused"))},
    )
    wf = getattr(workflows, verb)(WID)
    assert wf.endpoint.status == "paused"


def test_update_endpoint_sends_only_what_is_given(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(method="PATCH", url=f"{API}/{WID}/endpoint", json={"workflow": _wf()})
    workflows.update_endpoint(WID, scaledown_window=300)
    (request,) = httpx_mock.get_requests()
    assert json.loads(request.content) == {"scaledown_window": 300}
    with pytest.raises(ValueError, match="Give min_containers"):
        workflows.update_endpoint(WID)


# ── waiting until it is live ──────────────────────────────────────────────────


def test_wait_until_live_polls_until_its_endpoint_is_active(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    for status in ("draft", "provisioning", "active"):
        httpx_mock.add_response(
            method="GET",
            url=f"{API}/{WID}",
            json={"workflow": _wf(endpoint=_endpoint(status=status, deployed=status == "active"))},
        )
    slept: list[float] = []
    wf = workflows.wait_until_live(WID, poll_interval=0.5, sleep=slept.append)
    assert wf.endpoint is not None and wf.endpoint.status == "active" and slept == [0.5, 0.5]


@pytest.mark.parametrize(
    "endpoint, code",
    [
        (None, "not_deployed"),
        (
            _endpoint(status="draft", error_message="A SAM3 block cannot run on a live stream"),
            "workflow_not_streamable",
        ),
        (_endpoint(status="failed", error_message="its engine could not be built"), "job_failed"),
        (_endpoint(status="paused"), "job_cancelled"),
    ],
)
def test_wait_until_live_says_why_it_never_will(
    httpx_mock: HTTPXMock, workflows: Workflows, endpoint: Any, code: str
) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{API}/{WID}", json={"workflow": _wf(endpoint=endpoint)}
    )
    with pytest.raises(ApiError) as e:
        workflows.wait_until_live(WID, sleep=lambda _s: None)
    assert e.value.code == code


def test_wait_until_live_times_out(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/{WID}",
        json={"workflow": _wf(endpoint=_endpoint(status="provisioning"))},
        is_reusable=True,
    )
    with pytest.raises(PollTimeoutError, match="not live"):
        workflows.wait_until_live(WID, timeout=0.01, poll_interval=0.01, sleep=lambda _s: None)


# ── its streams: a deployment's client, its frames' measures ──────────────────


def test_connect_is_a_deployment_client_on_its_endpoint() -> None:
    wf = Workflow.model_validate(_wf())
    client = Workflows.connect(wf, TOKEN_X)
    assert isinstance(client, DeploymentClient)
    with pytest.raises(ValueError, match="no endpoint"):
        Workflows.connect(Workflow.model_validate(_wf(endpoint=None)), TOKEN_X)


def test_a_frame_carries_each_line_and_zones_counts() -> None:
    message = {
        "type": "frame",
        "frame": 7,
        "measures": {"m0": {"in": 3, "out": 1, "total": 4}},
        "pictograph": {"annotations": [], "image_width": 1280, "image_height": 720},
    }
    frame = StreamFrame(message, result=object())  # type: ignore[type-var]
    assert frame.measures == {"m0": {"in": 3, "out": 1, "total": 4}}
    assert StreamFrame({"type": "frame"}, result=object()).measures == {}  # type: ignore[type-var]


# ── its live log ──────────────────────────────────────────────────────────────


def test_live_log_reads_a_page_newest_first(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/{WID}/endpoint/log?stream=frames&limit=50&order=desc",
        json={
            "stream": "frames",
            "order": "desc",
            "records": [],
            "next_cursor": None,
            "log_frames": True,
        },
    )
    page = workflows.live_log(WID)
    assert page.records == [] and page.next_cursor is None
    assert page.event_names == {}


def test_a_live_log_events_page_names_its_events(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/{WID}/endpoint/log?stream=events&limit=50&order=desc",
        json={
            "stream": "events",
            "records": [],
            "event_names": {"crossing-0a1b2c3d": {"name": "Crossing · Gate", "kind": "crossing"}},
        },
    )
    page = workflows.live_log(WID, stream="events")
    named = page.event_names["crossing-0a1b2c3d"]
    assert (named.name, named.kind) == ("Crossing · Gate", "crossing")


def test_live_log_summary(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/{WID}/endpoint/log/summary",
        json={"log_frames": True, "streams": {"frames": {"count": 2}}},
    )
    assert workflows.live_log_summary(WID).log_frames is True


# ── async ─────────────────────────────────────────────────────────────────────


def test_the_async_twins(httpx_mock: HTTPXMock) -> None:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]

    async def go() -> None:
        t = AsyncTransport(config, api_key=KEY)
        try:
            wf = AsyncWorkflows(t)
            httpx_mock.add_response(
                method="POST",
                url=f"{API}/{WID}/deploy",
                json={"workflow": _wf(), "endpoint_token": TOKEN_Z},
            )
            deployed = await wf.deploy(WID)
            assert deployed.endpoint_token == TOKEN_Z
            httpx_mock.add_response(method="GET", url=f"{API}/{WID}", json={"workflow": _wf()})
            live = await wf.wait_until_live(WID)
            assert live.endpoint is not None and live.endpoint.status == "active"
        finally:
            await t.aclose()

    asyncio.run(go())
