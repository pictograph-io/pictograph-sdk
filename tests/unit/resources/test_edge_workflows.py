"""An EDGE workflow through the SDK (ROADMAP Program L, L4): ``create(environment="edge")`` sends no compute, region or
autoscaling - one given is refused before any request; ``connect`` refuses it in words (it has no endpoint to connect
to); ``edge_session`` opens a stream of it to run yourself - its spec, each model's graph and config, its outputs as
resolved, its own token and where its events go - an ``annotated_video`` output refused before any request (an edge
stream is never recorded); a ``WorkflowEndpoint`` names its environment (``cloud`` when a server says nothing); the
async twins do the same; the CLI (``create --environment edge``, ``edge-session``) takes it too, and ``stream`` refuses
an edge workflow in words."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pytest_httpx import HTTPXMock

from pictograph._http.async_transport import AsyncTransport
from pictograph._internal.config import ClientConfig
from pictograph.aio.resources.workflows import AsyncWorkflows
from pictograph.exceptions import ConflictError
from pictograph.models.workflow import Workflow, WorkflowEdgeSession, WorkflowEndpoint
from pictograph.resources.workflows import (
    EDGE_STREAM_NOT_RECORDED,
    EDGE_WORKFLOW_NO_ENDPOINT,
    Workflows,
    create_body,
)
from tests.unit.cli import test_workflows_endpoint_cli as cli_rows
from tests.unit.cli.test_workflows_endpoint_cli import cli
from tests.unit.resources.test_workflow_endpoints import (  # the workflow endpoints suite's fixtures and rows
    API,
    BASE,
    KEY,
    WID,
    _endpoint,
    _wf,
    transport,
    workflows,
)

__all__ = ["cli", "transport", "workflows"]  # pytest fixtures, used by name

MODEL = "abcdef01-2345-6789-abcd-ef0123456789"
SID = "a1b2c3d4e5f60718"
EP = "e0e0e0e0-1111-4222-8333-444455556666"

# An edge workflow's endpoint as core presents it (utils/workflow_endpoints.present): nothing of the cloud's - no
# region, GPU, URL or token; no file runs (run_limit 0) and no session Server (stream_limit 0); nothing billed.
_EDGE = {
    "id": EP,
    "environment": "edge",
    "status": "active",
    "region": None,
    "gpu_type": None,
    "min_containers": 0,
    "endpoint_url": None,
    "auth_token_prefix": None,
    "stream_limit": 0,
    "run_limit": 0,
    "cost_per_hour_micro_usd": 0,
    "max_cost_per_hour_micro_usd": 0,
    "runtime": None,
}

# What core's open hands a stream (utils/edge_workflows.open_session), as it answered live on 2026-11-15.
_SESSION = {
    "session_id": SID,
    "token": "1763240000.p0Rk4Lq9v2xYbN7aC1dE5fG8hJ3kM6nQ0rS2tU4wZ9y",
    "endpoint_id": EP,
    "spec": {
        "name": f"session-{SID}",
        "sources": [{"id": SID, "uri": "whip:", "stabilize": True}],
        "triggers": [
            {
                "id": "crossing-cf12f242",
                "priority": 1,
                "mode": "repeat",
                "actions": [{"type": "snapshot", "quality": 92}],
            }
        ],
        "outputs": {"captures": {"dir": f"/tmp/prt-session-captures/{SID}", "min_interval_s": 1.0}},
    },
    "models": [
        {
            "id": "m0",
            "model_id": MODEL,
            "name": "Street detector",
            "model_type": "object_detection",
            "architecture": "yolox",
            "precision": "fp32",
            "version_id": "v1",
            "graph": {
                "url": "https://api.pictograph.io/api/v1/download?token=t",
                "filename": "street.onnx",
                "bytes": 9_123_456,
                "precision": "fp32",
                "cache_key": "v1:fp32:42",
            },
            "config": {"family": "yolox_detection"},
        }
    ],
    "outputs": {
        "event_capture": {
            "dataset": "d0d0d0d0-1111-4222-8333-444455556666",
            "dataset_name": "e2e-edge-captures",
            "directory": "/",
            "folder": "/",
            "new_folder_each_run": False,
            "events": ["crossing:ms:l1"],
            "min_interval_s": 1.0,
        }
    },
    "captures": {"min_interval_s": 1.0},
    "events_url": f"/api/v1/workflow-live/{EP}/edge-sessions/{SID}/events",
    "log": {
        "url": "https://ingest.pictograph.io",
        "session": "f0f0f0f0-1111-4222-8333-444455556666",
        "token": "pk_ingest_x",
        "streams": ["frames", "events", "stats"],
        "log_frames": True,
    },
    "mark": {"data": "iVBORw0KGgo=", "content_type": "image/png"},
    "opened_at": "2026-11-15T12:54:58+00:00",
    "max_seconds": 14_400,
    "expires_at": "2026-11-15T16:56:58+00:00",
    "graphs_expire_at": "2026-11-15T13:54:58+00:00",
}

_CAPTURE = {
    "event_capture": {
        "dataset": "e2e-edge-captures",
        "events": ["crossing:ms:l1"],
        "min_interval_s": 1,
    }
}


def test_an_edge_create_sends_no_cloud_setting_and_reads_back_no_token(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{API}/", json={"workflow": _wf(endpoint=_endpoint(**_EDGE))}
    )
    wf = workflows.create("Gate counter", template_key="line_counter", environment="edge")
    assert json.loads(httpx_mock.get_requests()[-1].content) == {
        "name": "Gate counter",
        "template_key": "line_counter",
        "environment": "edge",
    }
    assert wf.endpoint is not None and wf.endpoint.environment == "edge"
    assert wf.endpoint_token is None and wf.endpoint.endpoint_url is None


@pytest.mark.parametrize(
    "given",
    [
        {"compute": {"gpu": "l4"}},
        {"region": "us-west"},
        {"min_containers": 1},
        {"max_containers": 2},
        {"scaledown_window": 120},
    ],
)
def test_a_cloud_setting_given_for_an_edge_workflow_is_refused_before_any_request(
    httpx_mock: HTTPXMock, workflows: Workflows, given: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="An edge workflow runs in the browser") as e:
        workflows.create("Gate counter", environment="edge", **given)
    assert next(iter(given)) in str(e.value)
    assert httpx_mock.get_requests() == []


def test_a_cloud_create_is_unchanged() -> None:
    body = create_body(
        "Gate counter",
        graph=None,
        readme=None,
        description=None,
        template_key="line_counter",
        compute=None,
        region="auto",
        min_containers=0,
        max_containers=1,
        scaledown_window=60,
        environment="cloud",
    )
    assert body == {"name": "Gate counter", "template_key": "line_counter"}


def test_an_edge_workflow_has_no_endpoint_to_connect_to() -> None:
    wf = Workflow.model_validate(_wf(endpoint=_endpoint(**_EDGE)))
    with pytest.raises(ValueError) as e:
        Workflows.connect(wf, token="pk_deploy_x")  # noqa: S106 - a test value
    assert str(e.value) == EDGE_WORKFLOW_NO_ENDPOINT
    assert "edge_session" in EDGE_WORKFLOW_NO_ENDPOINT


def test_an_endpoint_names_its_environment_cloud_when_the_server_says_nothing() -> None:
    row = _endpoint()
    row.pop("environment", None)
    assert WorkflowEndpoint.model_validate(row).environment == "cloud"
    assert WorkflowEndpoint.model_validate(_endpoint(**_EDGE)).environment == "edge"


def test_an_edge_session_is_what_a_browser_runs_a_stream_with(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{API}/{WID}/edge-sessions", json={"session": _SESSION}
    )
    session = workflows.edge_session(WID)
    assert json.loads(httpx_mock.get_requests()[-1].content) == {}, "no outputs: its log alone"
    assert isinstance(session, WorkflowEdgeSession)
    assert session.session_id == SID and session.endpoint_id == EP
    (model,) = session.models
    assert model.graph.cache_key == "v1:fp32:42" and model.precision == "fp32"
    assert model.config == {"family": "yolox_detection"}
    assert session.log is not None and session.log.streams == ["frames", "events", "stats"]
    assert session.events_url.endswith(f"/edge-sessions/{SID}/events")
    assert session.spec["triggers"][0]["id"] == "crossing-cf12f242"
    assert session.max_seconds == 14_400 and session.expires_at.tzinfo is not None


def test_an_edge_session_sends_its_outputs_checked_first(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{API}/{WID}/edge-sessions", json={"session": _SESSION}
    )
    workflows.edge_session(WID, outputs=_CAPTURE)
    assert json.loads(httpx_mock.get_requests()[-1].content) == {
        "outputs": {
            "annotated_video": False,  # as every run's outputs say it (the server refuses only True)
            "event_capture": {
                "dataset": "e2e-edge-captures",
                "events": ["crossing:ms:l1"],
                "min_interval_s": 1.0,
                "new_folder_each_run": False,
            },
        }
    }


@pytest.mark.parametrize("workflow", [WID, "Gate counter"])
def test_an_edge_stream_asked_to_record_is_refused_before_any_request(
    httpx_mock: HTTPXMock, workflows: Workflows, workflow: str
) -> None:
    """Even by name: the refusal comes before the name is looked up."""
    with pytest.raises(ValueError) as e:
        workflows.edge_session(workflow, outputs={"annotated_video": True})
    assert str(e.value) == EDGE_STREAM_NOT_RECORDED
    assert httpx_mock.get_requests() == []


def test_an_edge_session_of_a_cloud_workflow_is_a_conflict(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/{WID}/edge-sessions",
        status_code=409,
        json={"detail": "This workflow runs in the cloud: open its Run view to stream to it."},
    )
    with pytest.raises(ConflictError, match="runs in the cloud"):
        workflows.edge_session(WID)


def test_the_async_twins_create_and_open_an_edge_stream(httpx_mock: HTTPXMock) -> None:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]

    async def go() -> None:
        t = AsyncTransport(config, api_key=KEY)
        try:
            wf = AsyncWorkflows(t)
            httpx_mock.add_response(
                method="POST", url=f"{API}/", json={"workflow": _wf(endpoint=_endpoint(**_EDGE))}
            )
            made = await wf.create("Gate counter", template_key="line_counter", environment="edge")
            assert made.endpoint is not None and made.endpoint.environment == "edge"
            assert json.loads(httpx_mock.get_requests()[-1].content)["environment"] == "edge"
            with pytest.raises(ValueError, match="region"):
                await wf.create("Gate counter", environment="edge", region="us-west")
            httpx_mock.add_response(
                method="POST", url=f"{API}/{WID}/edge-sessions", json={"session": _SESSION}
            )
            session = await wf.edge_session(WID, outputs=_CAPTURE)
            assert session.session_id == SID
            with pytest.raises(ValueError):
                await wf.edge_session(WID, outputs={"annotated_video": True})
        finally:
            await t.aclose()

    asyncio.run(go())
    assert len(httpx_mock.get_requests()) == 2, "every refusal before any request"


# ───────────── the CLI ─────────────

_CLI_EDGE = {
    "id": EP,
    "environment": "edge",
    "status": "active",
    "region": None,
    "gpu_type": None,
    "endpoint_url": None,
    "stream_limit": 0,
    "run_limit": 0,
    "deployed": False,
    "runtime": None,
}


def test_the_cli_creates_an_edge_workflow_and_prints_no_token_or_compute(
    httpx_mock: HTTPXMock, cli: Any
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{cli_rows.API}/",
        json={"workflow": cli_rows._wf(endpoint=cli_rows._endpoint(**_CLI_EDGE))},
    )
    out = cli("create", "Gate counter", "--template", "traffic", "--environment", "edge")
    assert out.exit_code == 0, out.output
    assert json.loads(httpx_mock.get_requests()[-1].content) == {
        "name": "Gate counter",
        "template_key": "traffic",
        "environment": "edge",
    }
    printed = json.loads(out.stdout)
    assert printed["endpoint"]["environment"] == "edge" and "endpoint_token" not in printed
    assert "compute" not in printed, "an edge workflow holds no compute"
    assert "Save this token" not in out.stderr
    assert "pictograph workflows edge-session 'Gate counter'" in out.stderr


@pytest.mark.parametrize(
    ("given", "named"),
    [
        (["--gpu", "l4"], "--gpu"),
        (["--region", "us-west"], "region"),
        (["--min-containers", "1"], "min_containers"),
        (["--max-containers", "2"], "max_containers"),
        (["--scaledown-window", "120"], "scaledown_window"),
    ],
)
def test_the_cli_refuses_a_cloud_setting_for_an_edge_workflow_before_any_request(
    httpx_mock: HTTPXMock, cli: Any, given: list[str], named: str
) -> None:
    out = cli("create", "Gate counter", "--template", "traffic", "--environment", "edge", *given)
    assert out.exit_code != 0
    assert named in out.output
    assert "Traceback" not in out.output
    assert httpx_mock.get_requests() == []


def test_the_cli_refuses_an_unknown_environment(httpx_mock: HTTPXMock, cli: Any) -> None:
    out = cli("create", "Gate counter", "--template", "traffic", "--environment", "moon")
    assert out.exit_code != 0
    assert "use cloud or edge" in out.output
    assert httpx_mock.get_requests() == []


def test_the_cli_edge_session_prints_what_a_browser_runs_a_stream_with(
    httpx_mock: HTTPXMock, cli: Any
) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{cli_rows.API}/{WID}/edge-sessions", json={"session": _SESSION}
    )
    out = cli(
        "edge-session",
        WID,
        "--outputs",
        json.dumps(_CAPTURE),
    )
    assert out.exit_code == 0, out.output
    assert json.loads(httpx_mock.get_requests()[-1].content)["outputs"]["event_capture"][
        "events"
    ] == ["crossing:ms:l1"]
    printed = json.loads(out.stdout)
    assert printed["session_id"] == SID
    assert printed["models"][0]["graph"]["cache_key"] == "v1:fp32:42"
    assert printed["log"]["streams"] == ["frames", "events", "stats"]


def test_the_cli_edge_session_refuses_a_recording_before_any_request(
    httpx_mock: HTTPXMock, cli: Any
) -> None:
    out = cli("edge-session", WID, "--outputs", '{"annotated_video": true}')
    assert out.exit_code != 0
    assert "not recorded" in out.output and "Traceback" not in out.output
    assert httpx_mock.get_requests() == []


def test_the_cli_edge_session_of_a_cloud_workflow_exits_with_a_conflict(
    httpx_mock: HTTPXMock, cli: Any
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{cli_rows.API}/{WID}/edge-sessions",
        status_code=409,
        json={"detail": "This workflow runs in the cloud: open its Run view to stream to it."},
    )
    out = cli("edge-session", WID)
    assert out.exit_code != 0
    assert "runs in the cloud" in out.output


def test_the_cli_streams_no_camera_through_an_edge_workflow(
    httpx_mock: HTTPXMock, cli: Any
) -> None:
    """It has no endpoint: said in words, never a traceback - and its edge session named."""
    httpx_mock.add_response(
        method="GET",
        url=f"{cli_rows.API}/{WID}",
        json={"workflow": cli_rows._wf(endpoint=cli_rows._endpoint(**_CLI_EDGE))},
    )
    out = cli("stream", WID, "rtsp://cam.example.com/live", "--token", "pk_deploy_x")
    assert out.exit_code != 0
    assert "edge_session" in out.output and "Traceback" not in out.output
