"""``pictograph workflows log`` - a run's log from the terminal (L1).

Drives the REAL client over a mocked HTTP transport: the command, the SDK's pages and the
requests on the wire.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from pictograph.cli._app import app

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
API = f"{BASE}/api/v1/developer/workflows"
EVENT = {
    "ts": "2026-10-18T07:00:01Z",
    "engine_run": "e",
    "seq": 1,
    "record_index": 0,
    "type": "event",
    "source": "cam",
    "frame": 12,
    "trigger": "Gate",
    "rule_values": {"count": 1},
    "classes": ["car"],
    "track_ids": ["3"],
}


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_test")
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    return lambda *args: runner.invoke(app, ["workflows", "log", "run-1", *args])


def test_log_reads_pages_up_to_the_limit_as_json(httpx_mock: HTTPXMock, cli: Any) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/runs/run-1/log?stream=events&limit=2&order=desc&trigger=Gate",
        json={
            "stream": "events",
            "order": "desc",
            "records": [EVENT, dict(EVENT, record_index=1)],
            "next_cursor": "c",
        },
    )
    out = cli("--limit", "2", "--newest-first", "--trigger", "Gate", "--json")
    assert out.exit_code == 0, out.output
    got = json.loads(out.output)
    assert [r["record_index"] for r in got] == [0, 1]
    assert len(httpx_mock.get_requests()) == 1, (
        "the limit is reached on the first page: no second request"
    )


def test_log_frames_take_their_filters_and_print_a_table(httpx_mock: HTTPXMock, cli: Any) -> None:
    frame = dict(
        EVENT, type=None, trigger=None, annotation_count=2, max_confidence=0.9, latency_total_ms=4.2
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/runs/run-1/log?stream=frames&limit=50&order=asc&class=car&track_id=3&min_confidence=0.5",
        json={"stream": "frames", "records": [frame]},
    )
    out = cli("--stream", "frames", "--class", "car", "--track", "3", "--min-confidence", "0.5")
    assert out.exit_code == 0, out.output
    assert "Frames of run run-1 (1)" in out.output and "car" in out.output


def test_log_summary_says_when_the_plan_logs_no_frames(httpx_mock: HTTPXMock, cli: Any) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/runs/run-1/log/summary",
        json={
            "log_frames": False,
            "streams": {"frames": {"count": 0}, "events": {"count": 3}, "stats": {"count": 1}},
        },
    )
    out = cli("--summary")
    assert out.exit_code == 0, out.output
    assert "frames are logged on paid plans" in " ".join(out.output.split())


def test_an_unknown_stream_is_refused_before_any_request(httpx_mock: HTTPXMock, cli: Any) -> None:
    out = cli("--stream", "logs")
    assert out.exit_code != 0
    assert httpx_mock.get_requests() == []
