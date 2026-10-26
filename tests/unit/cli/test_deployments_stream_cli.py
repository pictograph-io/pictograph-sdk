"""``pictograph deployments stream``: live video through a deployment, each frame's result one JSON
line, and the stream stopped when the command ends.

These drive the REAL client: the open, the stop and the account lookup over a mocked HTTP
transport, and the results over a real local WebSocket (``tests/unit/stream_server.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from pictograph.cli._app import app
from pictograph.cli._errors import ExitCode
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
DEP = "d1111111-1111-1111-1111-111111111111"
TOKEN = "pk_deploy_tok"  # noqa: S105 - test fixture, not a real secret


@pytest.fixture
def results() -> Iterator[ResultsServer]:
    with serve_results() as server:
        yield server


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_test")
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_deployment_stream, "_backoff", lambda _drops: 0.0)
    runner = CliRunner()
    return lambda *args: runner.invoke(app, ["deployments", "stream", *args])


def _open_and_stop(httpx_mock: HTTPXMock, results: ResultsServer, kind: str = "rtsp") -> None:
    httpx_mock.add_response(method="POST", url=results.sessions, status_code=201, json=opened(kind))
    httpx_mock.add_response(
        method="DELETE", url=f"{results.origin}{SESSION_PATH}", json={"stopped": True}
    )


def _lines(stdout: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stdout.splitlines() if line.strip()]


def test_it_prints_each_frame_as_a_json_line_and_stops_the_stream(
    httpx_mock: HTTPXMock, results: ResultsServer, run: Any
) -> None:
    _open_and_stop(httpx_mock, results)
    stats = {"type": "stats", "sources": [{"frames": 1}]}
    results.script = [
        Connection(
            messages=[stats, frame(0, box("car", 0.9)), frame(1), frame(2)],
            reason="no request for 30 s",
        )
    ]
    res = run(
        DEP,
        "rtsp://cam.example.com/live",
        "--token",
        TOKEN,
        "--endpoint",
        results.endpoint,
        "--frames",
        "2",
    )
    assert res.exit_code == 0, res.output
    lines = _lines(res.stdout)
    assert [m["frame"] for m in lines] == [0, 1]
    assert lines[0]["pictograph"]["annotations"][0]["name"] == "car"
    assert "Watch it at" in res.stderr and f"{SESSION_PATH}/whep" in res.stderr
    requests = httpx_mock.get_requests()
    assert [r.method for r in requests] == ["POST", "DELETE"]
    assert json.loads(requests[0].content) == {
        "source": {"kind": "rtsp", "uri": "rtsp://cam.example.com/live"},
        "overlay": True,
    }
    assert all(r.headers["Authorization"] == f"Bearer {TOKEN}" for r in requests)


def test_all_prints_every_message(httpx_mock: HTTPXMock, results: ResultsServer, run: Any) -> None:
    _open_and_stop(httpx_mock, results)
    stats = {"type": "stats", "sources": [{"frames": 1}]}
    results.script = [Connection(messages=[stats, frame(0)], reason="its source ended")]
    res = run(
        DEP,
        "rtsp://cam.example.com/live",
        "--token",
        TOKEN,
        "--endpoint",
        results.endpoint,
        "--all",
    )
    assert res.exit_code == 0, res.output
    assert [m["type"] for m in _lines(res.stdout)] == ["stats", "frame"]
    assert "its camera stopped sending" in res.stderr


def test_a_whip_stream_says_where_to_send_the_video(
    httpx_mock: HTTPXMock, results: ResultsServer, run: Any
) -> None:
    _open_and_stop(httpx_mock, results, kind="whip")
    results.script = [Connection(messages=[frame(0)], reason="its source ended")]
    res = run(DEP, "whip", "--token", TOKEN, "--endpoint", results.endpoint, "--no-overlay")
    assert res.exit_code == 0, res.output
    assert f"Send its video to {results.origin}{SESSION_PATH}/whip" in res.stderr
    assert json.loads(httpx_mock.get_requests()[0].content) == {
        "source": {"kind": "whip"},
        "overlay": False,
    }


def test_a_name_is_resolved_with_the_account_key(
    httpx_mock: HTTPXMock, results: ResultsServer, run: Any
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/deployments/{DEP}",
        json={
            "deployment": {
                "id": DEP,
                "organization_id": "org",
                "model_id": "m",
                "name": "shelf",
                "status": "active",
                "region": "us-west",
                "compute_type": "gpu",
                "gpu_type": "t4",
                "precision": "fp32",
                "min_containers": 0,
                "max_containers": 1,
                "scaledown_window": 60,
                "endpoint_url": results.endpoint,
            }
        },
    )
    _open_and_stop(httpx_mock, results)
    results.script = [Connection(messages=[frame(0)], reason="its source ended")]
    res = run(DEP, "rtsp://cam.example.com/live", "--token", TOKEN)
    assert res.exit_code == 0, res.output
    assert [m["frame"] for m in _lines(res.stdout)] == [0]


def test_a_full_deployment_is_a_rate_limit_with_what_to_do(
    httpx_mock: HTTPXMock, results: ResultsServer, run: Any
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=results.sessions,
        status_code=429,
        json={"error": "stream_limit", "stream_limit": 4},
    )
    res = run(DEP, "whip", "--token", TOKEN, "--endpoint", results.endpoint)
    assert res.exit_code == ExitCode.RATE_LIMITED
    assert "max_containers" in res.output


def test_a_source_that_is_not_a_camera_is_refused_before_any_request(
    httpx_mock: HTTPXMock, results: ResultsServer, run: Any
) -> None:
    res = run(
        DEP,
        "http://user:secret@cam.example.com/x",
        "--token",
        TOKEN,
        "--endpoint",
        results.endpoint,
    )
    assert res.exit_code != 0
    assert "secret" not in res.output
    assert httpx_mock.get_requests() == []
