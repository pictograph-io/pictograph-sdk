"""An EDGE deployment through the SDK (ROADMAP Program L): ``create(environment="edge")`` sends its precision alone and
reads back no token; a cloud setting given for it is refused before any request; ``connect`` refuses it in words (it
has no endpoint); ``edge_session`` reads what a browser runs it with; a ``Deployment`` names its environment (``cloud``
when a server says nothing); the CLI (``create --environment edge``, ``edge-session``) and the agent tool take it
too."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pytest_httpx import HTTPXMock

from pictograph.agents import Toolkit
from pictograph.models.deployment import CreatedDeployment, Deployment, EdgeSession
from pictograph.resources.deployments import EDGE_NO_ENDPOINT, Deployments, create_body
from tests.unit.cli.test_deployments_compute_cli import API, DEP, _dep, run
from tests.unit.resources.test_deployments import (  # the deployments suite's fixtures and its real row
    _PATH,
    DEP_ID,
    _payload,
    deployments,
    transport,
)

__all__ = ["deployments", "run", "transport"]  # pytest fixtures, used by name

MODEL = "abcdef01-2345-6789-abcd-ef0123456789"

_EDGE_ROW = {
    "environment": "edge",
    "gpu_type": None,
    "endpoint_url": None,
    "auth_token_prefix": None,
    "cost_rate_per_min": 0,
    "precision": "fp16",
}

_SESSION = {
    "deployment_id": DEP_ID,
    "name": "Lobby kiosk",
    "model": {
        "id": MODEL,
        "name": "Street detector",
        "model_type": "object_detection",
        "version_id": "v1",
    },
    "graph": {
        "url": "https://api.pictograph.io/api/v1/download?token=t",
        "filename": "street.onnx",
        "bytes": 9_123_456,
        "precision": "fp16",
        "cache_key": "v1:fp16:42",
    },
    "config": {"_pictograph": {"model_type": "object_detection"}},
    "inference_config": {"confidence": 0.35},
    "visualization": {"draw": {}, "class_seed": ["car"], "class_colors": {"car": "#ff0000"}},
    "log": {
        "url": "https://ingest.test",
        "session": "sess-9",
        "token": "pk_ingest_x",
        "streams": ["events", "stats"],
        "log_frames": False,
    },
    "expires_at": "2026-11-15T09:55:00+00:00",
}


def test_an_edge_create_sends_its_precision_alone_and_reads_back_no_token(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=_PATH,
        json={
            "deployment": _payload(**_EDGE_ROW),
            "auth_token": None,
            "message": "Edge deployment created.",
        },
    )
    created = deployments.create(MODEL, name="Lobby kiosk", environment="edge", precision="fp16")
    assert isinstance(created, CreatedDeployment)
    assert created.auth_token is None
    assert created.deployment.environment == "edge"
    assert created.deployment.endpoint_url is None
    body = json.loads(httpx_mock.get_requests()[-1].content)
    assert body == {
        "model": MODEL,
        "environment": "edge",
        "precision": "fp16",
        "inference_config": {},
        "name": "Lobby kiosk",
    }


@pytest.mark.parametrize(
    "given",
    [
        {"gpu_type": "l4"},
        {"region": "us-west"},
        {"min_containers": 1},
        {"max_containers": 2},
        {"scaledown_window": 120},
        {"rate_limit_rpm": 600},
    ],
)
def test_a_cloud_setting_given_for_an_edge_deployment_is_refused_before_any_request(
    httpx_mock: HTTPXMock, deployments: Deployments, given: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="An edge deployment runs in the browser") as e:
        deployments.create(MODEL, environment="edge", **given)
    assert next(iter(given)) in str(e.value)
    assert httpx_mock.get_requests() == []


def test_a_cloud_create_is_unchanged() -> None:
    body = create_body(
        MODEL,
        name=None,
        region="auto",
        gpu_type="t4",
        precision="fp32",
        min_containers=0,
        max_containers=1,
        scaledown_window=60,
        rate_limit_rpm=None,
        inference_config=None,
    )
    assert "environment" not in body
    assert body["gpu_type"] == "t4"


def test_an_edge_deployment_has_no_endpoint_to_connect_to() -> None:
    edge = Deployment.model_validate(_payload(**_EDGE_ROW))
    with pytest.raises(ValueError) as e:
        Deployments.connect(edge, api_key="pk_deploy_x")
    assert str(e.value) == EDGE_NO_ENDPOINT
    assert "edge_session" in EDGE_NO_ENDPOINT


def test_a_deployment_names_its_environment_cloud_when_the_server_says_nothing() -> None:
    row = _payload()
    row.pop("environment", None)
    assert Deployment.model_validate(row).environment == "cloud"
    assert Deployment.model_validate(_payload(**_EDGE_ROW)).environment == "edge"


def test_the_edge_session_reads_what_a_browser_runs_it_with(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{_PATH}{DEP_ID}/edge-session",
        json={"success": True, "session": _SESSION},
    )
    session = deployments.edge_session(DEP_ID)
    assert isinstance(session, EdgeSession)
    assert session.graph.cache_key == "v1:fp16:42" and session.graph.bytes == 9_123_456
    assert (
        session.log is not None
        and session.log.streams == ["events", "stats"]
        and not session.log.log_frames
    )
    assert session.visualization["class_colors"] == {"car": "#ff0000"}
    assert session.inference_config == {"confidence": 0.35}


def test_an_edge_create_naming_no_precision_leaves_it_to_the_server_fp16(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=_PATH,
        json={"deployment": _payload(**{**_EDGE_ROW, "status": "provisioning"}), "auth_token": None},
    )
    created = deployments.create(MODEL, environment="edge")
    body = json.loads(httpx_mock.get_requests()[-1].content)
    assert body == {"model": MODEL, "environment": "edge", "inference_config": {}}, (
        "no precision: the edge's default on the server (FP16)"
    )
    assert created.deployment.status == "provisioning", "its FP16 graph is being built"


def test_an_fp16_edge_session_carries_its_fp32_fallback(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    fallback = {**_SESSION["graph"], "precision": "fp32", "cache_key": "v1:fp32:9"}
    httpx_mock.add_response(
        method="POST",
        url=f"{_PATH}{DEP_ID}/edge-session",
        json={"success": True, "session": {**_SESSION, "fallback": fallback}},
    )
    session = deployments.edge_session(DEP_ID)
    assert session.fallback is not None and session.fallback.cache_key == "v1:fp32:9"
    httpx_mock.add_response(
        method="POST",
        url=f"{_PATH}{DEP_ID}/edge-session",
        json={"success": True, "session": _SESSION},
    )
    assert deployments.edge_session(DEP_ID).fallback is None, "a server of before it sends none"


def test_an_edge_session_without_a_log_reads(
    httpx_mock: HTTPXMock, deployments: Deployments
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{_PATH}{DEP_ID}/edge-session",
        json={"success": True, "session": {**_SESSION, "log": None}},
    )
    assert deployments.edge_session(DEP_ID).log is None


# ───────────── the CLI and the agent tool ─────────────

_EDGE_CLI_ROW = {
    "environment": "edge",
    "status": "active",
    "gpu_type": None,
    "endpoint_url": None,
    "precision": "fp16",
}


def test_the_cli_creates_an_edge_deployment_and_prints_no_token(
    httpx_mock: HTTPXMock, run: Any
) -> None:
    httpx_mock.add_response(
        method="POST", url=API, json={"deployment": _dep(**_EDGE_CLI_ROW), "auth_token": None}
    )
    res = run("create", MODEL, "--environment", "edge", "--precision", "fp16", "--wait")
    assert res.exit_code == 0, res.output
    body = json.loads(httpx_mock.get_requests()[-1].content)
    assert body == {
        "model": MODEL,
        "environment": "edge",
        "precision": "fp16",
        "inference_config": {},
    }
    out = json.loads(res.stdout)
    assert out["environment"] == "edge" and out["status"] == "active"
    assert out["auth_token"] is None
    assert len(httpx_mock.get_requests()) == 1, (
        "an edge deployment active at once: --wait polls nothing"
    )


def test_the_cli_waits_for_an_edge_deployments_fp16_graph(httpx_mock: HTTPXMock, run: Any) -> None:
    building = {**_EDGE_CLI_ROW, "status": "provisioning", "engine_build": {"state": "building", "models": [
        {"model_id": MODEL, "precision": "fp16", "state": "building"}]}}
    httpx_mock.add_response(
        method="POST", url=API, json={"deployment": _dep(**building), "auth_token": None}
    )
    httpx_mock.add_response(method="GET", json={"deployment": _dep(**_EDGE_CLI_ROW)})
    res = run("create", MODEL, "--environment", "edge", "--wait", "--timeout", "30")
    assert res.exit_code == 0, res.output
    body = json.loads(httpx_mock.get_requests()[0].content)
    assert "precision" not in body, "the edge's default, FP16, on the server"
    assert json.loads(res.stdout)["status"] == "active"
    assert "FP16 graph" in res.stderr


@pytest.mark.parametrize(
    ("given", "named"),
    [
        (["--gpu", "l4"], "--gpu"),
        (["--region", "us-west"], "region"),
        (["--min-containers", "1"], "min_containers"),
        (["--rate-limit", "600"], "rate_limit_rpm"),
    ],
)
def test_the_cli_refuses_a_cloud_setting_for_an_edge_deployment_before_any_request(
    httpx_mock: HTTPXMock, run: Any, given: list[str], named: str
) -> None:
    res = run("create", MODEL, "--environment", "edge", *given)
    assert res.exit_code != 0
    assert named in res.output
    assert "Traceback" not in res.output
    assert httpx_mock.get_requests() == []


def test_the_cli_refuses_an_unknown_environment(httpx_mock: HTTPXMock, run: Any) -> None:
    res = run("create", MODEL, "--environment", "moon")
    assert res.exit_code != 0
    assert "use cloud or edge" in res.output
    assert httpx_mock.get_requests() == []


def test_the_cli_edge_session_prints_what_the_browser_runs_it_with(
    httpx_mock: HTTPXMock, run: Any
) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{API}{DEP}/edge-session", json={"success": True, "session": _SESSION}
    )
    res = run("edge-session", DEP)
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    assert out["graph"]["cache_key"] == "v1:fp16:42"
    assert out["log"]["streams"] == ["events", "stats"]
    assert out["visualization"]["class_colors"] == {"car": "#ff0000"}


def test_the_cli_edge_session_of_a_cloud_deployment_exits_with_a_conflict(
    httpx_mock: HTTPXMock, run: Any
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{API}{DEP}/edge-session",
        status_code=409,
        json={"detail": "This is a cloud deployment: it has no edge session."},
    )
    res = run("edge-session", DEP)
    assert res.exit_code != 0
    assert "cloud deployment" in res.output


def test_the_agent_tool_creates_an_edge_deployment() -> None:
    from unittest.mock import MagicMock

    client = MagicMock()
    client.deployments.create.return_value = CreatedDeployment(
        deployment=Deployment.model_validate(_payload(**_EDGE_ROW)), auth_token=None
    )
    result = Toolkit(client).dispatch(
        "create_deployment",
        {"model": "Street detector", "environment": "edge", "precision": "fp16"},
    )
    assert result.get("auth_token") is None, "no token: nothing calls an edge deployment"
    assert result["deployment"]["environment"] == "edge"
    kwargs = client.deployments.create.call_args.kwargs
    assert (kwargs["environment"], kwargs["precision"]) == ("edge", "fp16")
