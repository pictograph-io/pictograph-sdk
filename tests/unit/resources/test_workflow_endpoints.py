"""A workflow served live (B579): ``client.workflows``' endpoint methods and a stream frame's measures.

The workflow's endpoint - made by its deploy (never its create) in a region, paused, resumed, its scaling changed,
waited on until it is live, its live log - and the direct client its streams open on (a deployment's, with the
endpoint's token).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import types
from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.async_transport import AsyncTransport
from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.aio.resources.workflows import AsyncWorkflows
from pictograph.exceptions import ApiError, ConflictError, PollTimeoutError
from pictograph.models.workflow import Workflow
from pictograph.resources import _deployment_stream
from pictograph.resources._deployment_client import DeploymentClient
from pictograph.resources._deployment_stream import StreamFrame
from pictograph.resources.workflows import Workflows
from tests.unit.stream_server import (
    SESSION,
    Connection,
    ResultsServer,
    frame,
    joined_state,
    serve_results,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

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
                    "measure": "m0",
                },
                {
                    "trigger": "passive-12345678",
                    "key": "passive",
                    "kind": "passive",
                    "name": "Passive capture",
                    "measure": None,
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


def test_create_makes_its_endpoint_with_it_its_token_once(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    """B579 slice 5: a workflow's endpoint is made with it - where its runs and streams run. With no endpoint
    argument the server's defaults stand (the region nearest the caller, from zero to one container, a minute idle),
    so none is sent; its token is on the answer once."""
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/",
        json={
            "workflow": _wf(endpoint=_endpoint(status="provisioning")),
            "endpoint_token": TOKEN_X,
        },
    )
    wf = workflows.create("Gate counter", template_key="line_counter")
    (request,) = httpx_mock.get_requests()
    assert json.loads(request.content) == {"name": "Gate counter", "template_key": "line_counter"}
    assert wf.endpoint is not None and wf.endpoint.status == "provisioning"
    assert wf.endpoint_token == TOKEN_X


def test_create_sends_where_its_endpoint_runs_and_how_it_scales(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(method="POST", url=f"{API}/", json={"workflow": _wf()})
    workflows.create(
        "Gate counter",
        compute={"gpu": "l4"},
        region="us-west",
        min_containers=1,
        max_containers=3,
        scaledown_window=120,
    )
    assert json.loads(httpx_mock.get_requests()[-1].content) == {
        "name": "Gate counter",
        "compute": {"gpu": "l4"},
        "region": "us-west",
        "min_containers": 1,
        "max_containers": 3,
        "scaledown_window": 120,
    }


@pytest.mark.parametrize("method", [Workflows.create, AsyncWorkflows.create])
def test_create_takes_its_endpoints_settings_with_the_servers_defaults(method: Any) -> None:
    """Sync and async alike: where its endpoint runs and how it scales, the server's defaults when not given."""
    params = inspect.signature(method).parameters
    assert {
        k: params[k].default
        for k in ("region", "min_containers", "max_containers", "scaledown_window")
    } == {
        "region": "auto",
        "min_containers": 0,
        "max_containers": 1,
        "scaledown_window": 60,
    }


@pytest.mark.parametrize(
    "method", [Workflows.run, AsyncWorkflows.run, Workflows.invoke, AsyncWorkflows.invoke]
)
def test_a_run_names_no_gpu_of_its_own(method: Any) -> None:
    """A run runs on its workflow's endpoint, on the workflow's GPU (B579 slice 5): ``gpu`` is the workflow's
    (``create`` / ``update``), never a run's."""
    assert "gpu" not in inspect.signature(method).parameters


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
    # B607: every event it offers - what a run's Event capture and webhook pick from (by key) - and what its live
    # outputs did.
    assert [(e.key, e.kind, e.name) for e in wf.endpoint.live.events] == [
        ("crossing:ms:l1", "crossing", "Crossing · Gate"),
        ("passive", "passive", "Passive capture"),
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


def test_an_endpoint_says_whether_its_region_falls_back() -> None:
    """B597: a stream or run that finds no GPU free in its region for 30 s is also offered the rest of the US."""
    from pictograph.models.workflow import WorkflowEndpoint

    assert WorkflowEndpoint.model_validate(_endpoint(region_fallback=True)).region_fallback is True
    assert WorkflowEndpoint.model_validate(_endpoint()).region_fallback is False, (
        "an answer from before B597"
    )


def test_an_endpoint_says_whether_its_live_streams_have_a_log() -> None:
    """``logged``: a draft never deployed, or a paused endpoint (a pause ends its log), has none - ``live_log``
    would raise ``no_log``. None from a server that does not say."""
    from pictograph.models.workflow import WorkflowEndpoint

    assert WorkflowEndpoint.model_validate(_endpoint(logged=True)).logged is True
    assert WorkflowEndpoint.model_validate(_endpoint(status="paused", logged=False)).logged is False
    assert WorkflowEndpoint.model_validate(_endpoint()).logged is None, (
        "an answer from before 1.69.164"
    )


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
            _endpoint(status="draft", error_message="Add a model to a Model block to run it."),
            "workflow_not_runnable",
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


def test_an_endpoint_that_runs_files_only_is_live_for_its_runs(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    """B579 slice 5: a workflow that cannot stream as saved (a SAM3 block) runs its files on its endpoint - waiting
    returns once it is up, saying it takes no stream (``streams`` False) and why; a stream is refused by the server."""
    why = "A SAM3 block cannot run on a live stream"
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/{WID}",
        json={"workflow": _wf(endpoint=_endpoint(streams=False, run_limit=2, error_message=why))},
    )
    wf = workflows.wait_until_live(WID, sleep=lambda _s: None)
    assert wf.endpoint is not None and wf.endpoint.streams is False
    assert wf.endpoint.run_limit == 2 and wf.endpoint.error_message == why


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
    client = Workflows.connect(wf, token=TOKEN_X)
    assert isinstance(client, DeploymentClient)
    with pytest.raises(ValueError, match="no endpoint"):
        Workflows.connect(Workflow.model_validate(_wf(endpoint=None)), token=TOKEN_X)


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


def test_live_log_and_its_iterator_filter_by_a_condition_branch(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    """B549: the live log takes the run log's filters, the branch among them - and so does its iterator."""
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/{WID}/endpoint/log?stream=frames&limit=50&order=desc&branch=true",
        json={"stream": "frames", "records": []},
    )
    assert workflows.live_log(WID, branch=True).records == []
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/{WID}/endpoint/log?stream=events&limit=200&order=desc&class=car&branch=false",
        json={"stream": "events", "records": []},
    )
    assert list(workflows.iter_live_log(WID, stream="events", class_name="car", branch=False)) == []


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


# ── a file run's live view, watched (B607 slice C) ───────────────────────────


@pytest.fixture
def results() -> Iterator[ResultsServer]:
    with serve_results() as server:
        yield server


def _run_row(**o: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "run-9",
        "organization_id": "org-1",
        "workflow_id": WID,
        "status": "processing",
        "frames_done": 0,
        "live_view": {"stream": False, "overlay": True, "session": None},
    }
    base.update(o)
    return base


def _view(session: str | None = SESSION) -> dict[str, Any]:
    return {"stream": False, "overlay": True, "session": session}


def test_watch_run_waits_for_its_first_frames_then_joins_its_live_view(
    httpx_mock: HTTPXMock, workflows: Workflows, results: ResultsServer
) -> None:
    """B607 slice C: watch_run waits while its run queues, is dispatched and starts (its engine serves its live view
    once its first frames are in), then joins its live view on the workflow's endpoint with the endpoint's token - its
    frames read as a stream's, ending with its run."""
    run_url = f"{API}/runs/run-9"
    httpx_mock.add_response(
        method="GET", url=run_url, json={"run": _run_row(status="queued", live_view=_view(None))}
    )
    httpx_mock.add_response(method="GET", url=run_url, json={"run": _run_row(live_view=_view())})
    httpx_mock.add_response(
        method="GET", url=run_url, json={"run": _run_row(frames_done=12, live_view=_view())}
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/{WID}",
        json={"workflow": _wf(endpoint=_endpoint(endpoint_url=results.endpoint))},
    )
    httpx_mock.add_response(
        method="GET", url=f"{results.sessions}/{SESSION}?ice=1", json=joined_state()
    )
    results.script = [Connection(messages=[frame(0)], reason="the run completed")]
    slept: list[float] = []
    live = workflows.watch_run("run-9", token=TOKEN_X, sleep=slept.append)
    assert slept == [2.0, 2.0]
    assert (live.joined, live.id, live.stream) == (True, SESSION, False)
    assert httpx_mock.get_requests()[-1].headers["authorization"] == f"Bearer {TOKEN_X}"
    with live:
        assert [f.index for f in live.frames()] == [0]
    assert live.ended == "run_completed"
    assert (results.seen[0]["path"], results.seen[0]["authorization"]) == (
        f"/us-west/abc123/sessions/{SESSION}/results",
        f"Bearer {TOKEN_X}",
    )


@pytest.mark.parametrize(
    ("row", "code"),
    [
        (_run_row(live_view=None), "no_live_view"),
        (_run_row(status="completed", frames_done=600, live_view=_view()), "run_ended"),
        (_run_row(status="error", live_view=_view(None)), "run_ended"),
    ],
)
def test_watch_run_says_why_a_run_cannot_be_watched(
    httpx_mock: HTTPXMock, workflows: Workflows, row: dict[str, Any], code: str
) -> None:
    httpx_mock.add_response(method="GET", url=f"{API}/runs/run-9", json={"run": row})
    with pytest.raises(ConflictError) as exc:
        workflows.watch_run("run-9", token=TOKEN_X, sleep=lambda _s: None)
    assert exc.value.code == code


def test_watch_run_times_out_while_its_run_waits(
    httpx_mock: HTTPXMock, workflows: Workflows, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = iter([0.0, 0.0, 5.0])
    # This module's clock alone (patching time.monotonic itself would move every clock in the process).
    monkeypatch.setattr(
        "pictograph.resources.workflows.time", types.SimpleNamespace(monotonic=lambda: next(clock))
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/runs/run-9",
        json={
            "run": _run_row(
                status="queued",
                live_view=_view(None),
                dispatch_note="Waiting: its endpoint is starting.",
            )
        },
        is_reusable=True,
    )
    with pytest.raises(PollTimeoutError, match="its endpoint is starting"):
        workflows.watch_run("run-9", token=TOKEN_X, timeout=1.0, sleep=lambda _s: None)


@pytest.mark.parametrize(("arg", "value"), [("timeout", 0.0), ("poll_interval", -1.0)])
def test_watch_run_takes_positive_waits(workflows: Workflows, arg: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"{arg} must be > 0"):
        workflows.watch_run("run-9", token=TOKEN_X, **{arg: value})  # type: ignore[arg-type]


def test_a_failed_join_closes_the_connection_watch_run_opened(
    httpx_mock: HTTPXMock,
    workflows: Workflows,
    results: ResultsServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []
    monkeypatch.setattr(DeploymentClient, "close", lambda _self: closed.append(True))
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/runs/run-9",
        json={"run": _run_row(frames_done=3, live_view=_view())},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/{WID}",
        json={"workflow": _wf(endpoint=_endpoint(endpoint_url=results.endpoint))},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{results.sessions}/{SESSION}?ice=1",
        status_code=410,
        json={"error": "session_ended"},
    )
    with pytest.raises(ApiError):
        workflows.watch_run("run-9", token=TOKEN_X, sleep=lambda _s: None)
    assert closed == [True]


def test_stopping_the_watch_closes_its_connection(
    httpx_mock: HTTPXMock,
    workflows: Workflows,
    results: ResultsServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []
    monkeypatch.setattr(DeploymentClient, "close", lambda _self: closed.append(True))
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/runs/run-9",
        json={"run": _run_row(frames_done=3, live_view=_view())},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/{WID}",
        json={"workflow": _wf(endpoint=_endpoint(endpoint_url=results.endpoint))},
    )
    httpx_mock.add_response(
        method="GET", url=f"{results.sessions}/{SESSION}?ice=1", json=joined_state()
    )
    live = workflows.watch_run("run-9", token=TOKEN_X, sleep=lambda _s: None)
    assert closed == []
    live.stop()
    assert closed == [True]


def test_watch_runs_async_twin(
    httpx_mock: HTTPXMock, results: ResultsServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    closed: list[bool] = []
    monkeypatch.setattr(DeploymentClient, "close", lambda _self: closed.append(True))
    run_url = f"{API}/runs/run-9"
    httpx_mock.add_response(
        method="GET", url=run_url, json={"run": _run_row(status="queued", live_view=_view(None))}
    )
    httpx_mock.add_response(
        method="GET", url=run_url, json={"run": _run_row(frames_done=4, live_view=_view())}
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/{WID}",
        json={"workflow": _wf(endpoint=_endpoint(endpoint_url=results.endpoint))},
    )
    httpx_mock.add_response(
        method="GET", url=f"{results.sessions}/{SESSION}?ice=1", json=joined_state()
    )
    slept: list[float] = []

    async def sleep(s: float) -> None:
        slept.append(s)

    async def go() -> Any:
        t = AsyncTransport(config, api_key=KEY)
        try:
            return await AsyncWorkflows(t).watch_run("run-9", token=TOKEN_X, sleep=sleep)
        finally:
            await t.aclose()

    live = asyncio.run(go())
    assert slept == [2.0]
    assert (live.joined, live.id) == (True, SESSION)
    assert closed == []
    live.stop()
    assert closed == [True], "its stop closes the connection it opened"

    async def refused() -> None:
        t = AsyncTransport(config, api_key=KEY)
        try:
            with pytest.raises(ValueError, match="timeout must be > 0"):
                await AsyncWorkflows(t).watch_run("run-9", token=TOKEN_X, timeout=0)
            with pytest.raises(ValueError, match="poll_interval must be > 0"):
                await AsyncWorkflows(t).watch_run("run-9", token=TOKEN_X, poll_interval=0)
        finally:
            await t.aclose()

    asyncio.run(refused())


def test_watch_runs_async_twin_times_out_and_closes_a_failed_join(
    httpx_mock: HTTPXMock, results: ResultsServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    clock = iter([0.0, 0.0, 5.0])
    monkeypatch.setattr(
        "pictograph.aio.resources.workflows.time",
        types.SimpleNamespace(monotonic=lambda: next(clock)),
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/runs/run-9",
        json={"run": _run_row(status="queued", live_view=_view(None))},
        is_reusable=True,
    )
    closed: list[bool] = []
    monkeypatch.setattr(DeploymentClient, "close", lambda _self: closed.append(True))

    async def sleep(_s: float) -> None:
        return None

    async def go() -> None:
        t = AsyncTransport(config, api_key=KEY)
        try:
            with pytest.raises(PollTimeoutError):
                await AsyncWorkflows(t).watch_run("run-9", token=TOKEN_X, timeout=1.0, sleep=sleep)
        finally:
            await t.aclose()

    asyncio.run(go())
    httpx_mock.reset()  # the queued answer above is spent
    monkeypatch.setattr(
        "pictograph.aio.resources.workflows.time", types.SimpleNamespace(monotonic=lambda: 0.0)
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/runs/run-9",
        json={"run": _run_row(frames_done=4, live_view=_view())},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/{WID}",
        json={"workflow": _wf(endpoint=_endpoint(endpoint_url=results.endpoint))},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{results.sessions}/{SESSION}?ice=1",
        status_code=410,
        json={"error": "session_ended"},
    )

    async def failed() -> None:
        t = AsyncTransport(config, api_key=KEY)
        try:
            with pytest.raises(ApiError):
                await AsyncWorkflows(t).watch_run("run-9", token=TOKEN_X, sleep=sleep)
        finally:
            await t.aclose()

    asyncio.run(failed())
    assert closed == [True]
    assert _deployment_stream.JOIN_WAIT_S == 60.0
