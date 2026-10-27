"""A workflow served live from the terminal (B579): ``pictograph workflows deploy``, ``pause``, ``resume``,
``scale``, ``stream`` and ``live-log`` - and ``create``, which never deploys it.

These drive the REAL client over a mocked HTTP transport, and ``stream``'s results over a real local WebSocket
(``tests/unit/stream_server.py``) - the same stream loop as ``deployments stream`` (``cli/_endpoint.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from pictograph.cli._app import app
from pictograph.resources import _deployment_stream
from tests.unit.stream_server import (
    SESSION_PATH,
    Connection,
    ResultsServer,
    box,
    frame,
    opened,
    serve_results,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
API = f"{BASE}/api/v1/developer/workflows"
WID = "fedcba98-1111-2222-3333-444455556666"
TOKEN = "pk_deploy_tok"  # noqa: S105 - test fixture, not a real secret


def _endpoint(**o: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "ep-1",
        "status": "active",
        "region": "us-west",
        "gpu_type": "t4",
        "min_containers": 0,
        "max_containers": 2,
        "scaledown_window": 120,
        "endpoint_url": "https://infer.test/us-west/cdcdcdcd/predict",
        "error_message": None,
        "stream_limit": 8,
        "deployed": True,
        "runtime": {"state": "asleep", "containers": 0, "creating": 0, "streams": 0},
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


@pytest.fixture
def results() -> Iterator[ResultsServer]:
    with serve_results() as server:
        yield server


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_test")
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_deployment_stream, "_backoff", lambda _drops: 0.0)
    runner = CliRunner()
    return lambda *args: runner.invoke(app, ["workflows", *args])


def _lines(stdout: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stdout.splitlines() if line.strip()]


# ── create never deploys; deploy does: where it is served, its token once ────


def test_create_makes_its_endpoint_with_it_and_shows_its_token_once(
    httpx_mock: HTTPXMock, cli: Any
) -> None:
    """B579 slice 5: a workflow's endpoint is made with it - where its runs and streams run - its token on the answer
    once. With no endpoint option the server's defaults stand (the region nearest the caller), so none is sent."""
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/",
        json={"workflow": _wf(endpoint=_endpoint(status="provisioning")), "endpoint_token": TOKEN},
    )
    out = cli("create", "Gate counter", "--template", "traffic")
    assert out.exit_code == 0, out.output
    assert json.loads(httpx_mock.get_requests()[-1].content) == {
        "name": "Gate counter",
        "template_key": "traffic",
    }
    printed = json.loads(out.stdout)
    assert printed["endpoint"]["status"] == "provisioning" and printed["endpoint_token"] == TOKEN
    assert "Save this token" in out.stderr
    assert "Run it: pictograph workflows run 'Gate counter'" in out.stderr


def test_create_sends_where_it_runs_and_how_it_scales(httpx_mock: HTTPXMock, cli: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/",
        json={"workflow": _wf(endpoint=_endpoint(status="provisioning")), "endpoint_token": TOKEN},
    )
    out = cli(
        "create",
        "Gate counter",
        "--template",
        "traffic",
        "--gpu",
        "l4",
        "--region",
        "us-west",
        "--min-containers",
        "1",
        "--max-containers",
        "3",
        "--scaledown-window",
        "120",
    )
    assert out.exit_code == 0, out.output
    assert json.loads(httpx_mock.get_requests()[-1].content) == {
        "name": "Gate counter",
        "template_key": "traffic",
        "compute": {"gpu": "l4"},
        "region": "us-west",
        "min_containers": 1,
        "max_containers": 3,
        "scaledown_window": 120,
    }


def test_a_draft_endpoint_says_why_it_cannot_stream(httpx_mock: HTTPXMock, cli: Any) -> None:
    why = "A SAM3 block cannot run on a live stream."
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/{WID}/deploy",
        json={
            "workflow": _wf(endpoint=_endpoint(status="draft", error_message=why, runtime=None)),
            "endpoint_token": TOKEN,
        },
    )
    out = cli("deploy", WID)
    assert out.exit_code == 0, out.output
    assert json.loads(out.stdout)["endpoint"]["error"] == why


def test_deploy_sends_its_region_and_prints_the_token(httpx_mock: HTTPXMock, cli: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/{WID}/deploy",
        json={"workflow": _wf(endpoint=_endpoint(status="provisioning")), "endpoint_token": TOKEN},
    )
    out = cli("deploy", WID, "--region", "us-east", "--min-containers", "1")
    assert out.exit_code == 0, out.output
    assert json.loads(httpx_mock.get_requests()[-1].content) == {
        "region": "us-east",
        "min_containers": 1,
    }
    printed = json.loads(out.stdout)
    assert printed["endpoint_token"] == TOKEN and printed["endpoint"]["status"] == "provisioning"


def test_deploy_wait_polls_until_live_and_keeps_the_token(httpx_mock: HTTPXMock, cli: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/{WID}/deploy",
        json={"workflow": _wf(endpoint=_endpoint(status="provisioning")), "endpoint_token": TOKEN},
    )
    httpx_mock.add_response(method="GET", url=f"{API}/{WID}", json={"workflow": _wf()})
    out = cli("deploy", WID, "--wait")
    assert out.exit_code == 0, out.output
    printed = json.loads(out.stdout)
    assert printed["endpoint"]["status"] == "active" and printed["endpoint_token"] == TOKEN
    assert "Waiting for Gate counter to go live..." in out.stderr


def test_deploy_wait_says_its_engines_are_built_first(httpx_mock: HTTPXMock, cli: Any) -> None:
    building = {
        "state": "building",
        "started_at": "2026-10-26T00:00:00Z",
        "models": [
            {
                "model_id": "m1",
                "name": "Birds",
                "precision": "fp16",
                "state": "building",
                "estimate_s": 95.0,
            }
        ],
    }
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/{WID}/deploy",
        json={
            "workflow": _wf(endpoint=_endpoint(status="provisioning", engine_build=building)),
            "endpoint_token": TOKEN,
        },
    )
    httpx_mock.add_response(method="GET", url=f"{API}/{WID}", json={"workflow": _wf()})
    out = cli("deploy", WID, "--wait")
    assert out.exit_code == 0, out.output
    assert (
        "Building its models' TensorRT engines for t4 first (about 95s), then starting Gate counter"
        in out.stderr
    )


def test_a_wait_that_fails_still_prints_the_token(httpx_mock: HTTPXMock, cli: Any) -> None:
    why = "A SAM3 block cannot run on a live stream."
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/{WID}/deploy",
        json={"workflow": _wf(endpoint=_endpoint(status="provisioning")), "endpoint_token": TOKEN},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/{WID}",
        json={"workflow": _wf(endpoint=_endpoint(status="draft", error_message=why))},
    )
    out = cli("deploy", WID, "--wait")
    assert out.exit_code != 0
    assert json.loads(out.stdout)["endpoint_token"] == TOKEN
    assert why in out.stderr


def test_resume_wait_waits_until_live(httpx_mock: HTTPXMock, cli: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/{WID}/resume",
        json={"workflow": _wf(endpoint=_endpoint(status="provisioning"))},
    )
    httpx_mock.add_response(method="GET", url=f"{API}/{WID}", json={"workflow": _wf()})
    out = cli("resume", WID, "--wait")
    assert out.exit_code == 0, out.output
    assert json.loads(out.stdout)["endpoint"]["status"] == "active"


def test_list_says_what_each_endpoint_is_doing(httpx_mock: HTTPXMock, cli: Any) -> None:
    waiting = {
        "state": "waiting",
        "containers": 0,
        "creating": 0,
        "waiting_since": "2026-10-26T00:00:00Z",
        "checked_at": "2026-10-26T00:01:42Z",
    }
    rows = [
        _wf(name="Gate counter", endpoint=_endpoint(runtime=waiting)),
        _wf(id="w2", name="Never deployed", endpoint=None),
        _wf(
            id="w3",
            name="Held",
            endpoint=_endpoint(status="paused", region="us-east", runtime=None),
        ),
    ]
    httpx_mock.add_response(method="GET", url=f"{API}/", json={"workflows": rows})
    out = cli("list")
    assert out.exit_code == 0, out.output
    assert "active · waiting for a GPU 1m42s" in out.stdout
    assert "paused" in out.stdout and "us-east" in out.stdout and "Never deployed" in out.stdout


@pytest.mark.parametrize("verb", ["pause", "resume"])
def test_pause_and_resume(httpx_mock: HTTPXMock, cli: Any, verb: str) -> None:
    status = "paused" if verb == "pause" else "provisioning"
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/{WID}/{verb}",
        json={"workflow": _wf(endpoint=_endpoint(status=status))},
    )
    out = cli(verb, WID)
    assert out.exit_code == 0, out.output
    assert json.loads(out.stdout)["endpoint"]["status"] == status


def test_an_endpoint_prints_its_live_events_and_what_its_live_outputs_did(
    httpx_mock: HTTPXMock, cli: Any
) -> None:
    """B579 slice 1: the events its outputs act on while it streams, named as the app names them, and every
    outcome counted - a frame not saved says why."""
    live = {
        "events": [
            {
                "trigger": "crossing-0a1b2c3d",
                "key": "crossing:ms:l1",
                "kind": "crossing",
                "name": "Crossing · Gate",
                "captured": True,
            }
        ]
    }
    outputs = {
        "events": 5,
        "deliveries": 2,
        "saved": 3,
        "not_saved": 1,
        "skipped": 0,
        "last_not_saved_reason": "The organization is at its image limit.",
    }
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/{WID}/pause",
        json={
            "workflow": _wf(endpoint=_endpoint(status="paused", live=live, live_outputs=outputs))
        },
    )
    out = cli("pause", WID)
    assert out.exit_code == 0, out.output
    ep = json.loads(out.stdout)["endpoint"]
    assert ep["live_events"] == ["Crossing · Gate"]
    assert ep["live_outputs"] == outputs


def test_scale_sends_only_what_is_given(httpx_mock: HTTPXMock, cli: Any) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{API}/{WID}/endpoint",
        json={"workflow": _wf(endpoint=_endpoint(scaledown_window=300))},
    )
    out = cli("scale", WID, "--scaledown-window", "300")
    assert out.exit_code == 0, out.output
    assert json.loads(httpx_mock.get_requests()[-1].content) == {"scaledown_window": 300}
    assert json.loads(out.stdout)["scaledown_window"] == 300


def test_scale_with_nothing_to_change_is_refused_before_any_request(
    httpx_mock: HTTPXMock, cli: Any
) -> None:
    out = cli("scale", WID)
    assert out.exit_code != 0 and "min_containers" in out.output
    assert httpx_mock.get_requests() == []


# ── stream: every block a frame passes, each frame one JSON line ──────────────


def test_stream_prints_each_frame_with_its_measures_and_stops_the_stream(
    httpx_mock: HTTPXMock, results: ResultsServer, cli: Any
) -> None:
    httpx_mock.add_response(
        method="POST", url=results.sessions, status_code=201, json=opened("rtsp")
    )
    httpx_mock.add_response(
        method="DELETE", url=f"{results.origin}{SESSION_PATH}", json={"stopped": True}
    )
    counted = {**frame(0, box("car", 0.9)), "measures": {"m0": {"in": 1, "out": 0, "total": 1}}}
    results.script = [Connection(messages=[counted, frame(1)], reason="its source ended")]
    out = cli(
        "stream",
        WID,
        "rtsp://cam.example.com/live",
        "--token",
        TOKEN,
        "--endpoint",
        results.endpoint,
        "--frames",
        "2",
    )
    assert out.exit_code == 0, out.output
    lines = _lines(out.stdout)
    assert [m["frame"] for m in lines] == [0, 1]
    assert lines[0]["measures"] == {"m0": {"in": 1, "out": 0, "total": 1}}
    requests = httpx_mock.get_requests()
    assert [r.method for r in requests] == ["POST", "DELETE"]
    assert json.loads(requests[0].content) == {
        "source": {"kind": "rtsp", "uri": "rtsp://cam.example.com/live"},
        "overlay": True,
    }
    assert all(r.headers["Authorization"] == f"Bearer {TOKEN}" for r in requests)


def test_stream_finds_its_endpoint_by_the_workflow(
    httpx_mock: HTTPXMock, results: ResultsServer, cli: Any
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/{WID}",
        json={"workflow": _wf(endpoint=_endpoint(endpoint_url=results.endpoint))},
    )
    httpx_mock.add_response(
        method="POST", url=results.sessions, status_code=201, json=opened("whip")
    )
    httpx_mock.add_response(
        method="DELETE", url=f"{results.origin}{SESSION_PATH}", json={"stopped": True}
    )
    results.script = [Connection(messages=[frame(0)], reason="its source ended")]
    out = cli("stream", WID, "whip", "--token", TOKEN, "--no-overlay")
    assert out.exit_code == 0, out.output
    assert f"Send its video to {results.origin}{SESSION_PATH}/whip" in out.stderr
    assert json.loads(httpx_mock.get_requests()[1].content) == {
        "source": {"kind": "whip"},
        "overlay": False,
    }


def test_stream_of_a_workflow_never_deployed_says_so(httpx_mock: HTTPXMock, cli: Any) -> None:
    httpx_mock.add_response(method="GET", url=f"{API}/{WID}", json={"workflow": _wf(endpoint=None)})
    out = cli("stream", WID, "whip", "--token", TOKEN)
    assert out.exit_code != 0 and "deploy it" in out.output


# ── its live log ──────────────────────────────────────────────────────────────


def test_live_log_reads_a_page(httpx_mock: HTTPXMock, cli: Any) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/{WID}/endpoint/log?stream=events&limit=5&order=desc",
        json={
            "stream": "events",
            "order": "desc",
            "records": [],
            "next_cursor": None,
            "log_frames": True,
        },
    )
    out = cli("live-log", WID, "--stream", "events", "--limit", "5")
    assert out.exit_code == 0, out.output
    assert json.loads(out.stdout)["stream"] == "events"


def test_live_log_filters_by_a_condition_branch(httpx_mock: HTTPXMock, cli: Any) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/{WID}/endpoint/log?stream=frames&limit=50&order=desc&branch=true",
        json={"stream": "frames", "order": "desc", "records": [], "log_frames": True},
    )
    out = cli("live-log", WID, "--branch", "true")
    assert out.exit_code == 0, out.output
    assert json.loads(out.stdout)["stream"] == "frames"


def test_live_log_refuses_an_unknown_stream(httpx_mock: HTTPXMock, cli: Any) -> None:
    out = cli("live-log", WID, "--stream", "nope")
    assert out.exit_code != 0 and httpx_mock.get_requests() == []
