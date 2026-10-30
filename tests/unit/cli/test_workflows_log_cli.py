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


def test_log_events_are_named_as_the_app_names_them(
    httpx_mock: HTTPXMock, cli: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B579 slice 1: one event vocabulary - the table names each event by the log's names (a trigger the page does
    not name reads as itself)."""
    monkeypatch.setenv("COLUMNS", "240")  # the table on one line a row
    crossing = dict(EVENT, trigger="crossing-0a1b2c3d")
    names = {"crossing-0a1b2c3d": {"name": "Crossing · Gate", "kind": "crossing"}}
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/runs/run-1/log?stream=events&limit=50&order=asc",
        json={
            "stream": "events",
            "records": [crossing, dict(EVENT, record_index=1)],
            "event_names": names,
        },
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/runs/run-1/log?stream=events&limit=1&order=asc",
        json={"stream": "events", "records": [crossing], "event_names": names},
    )
    out = cli()
    assert out.exit_code == 0, out.output
    text = " ".join(out.output.split())
    assert "Crossing · Gate" in text and "crossing-0a1b2c3d" not in text
    assert "Gate" in text


def test_log_events_say_their_values_in_words(
    httpx_mock: HTTPXMock, cli: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B579 slice 1: an event's values as the server words them, never PRT's keys (a server that words none: its
    keys)."""
    monkeypatch.setenv("COLUMNS", "240")
    worded = dict(EVENT, rule_values={"zone_events:m1:in": 1}, values_text="Lot: 1 entered")
    raw = dict(EVENT, record_index=1, rule_values={"count": 7})
    for limit in ("50", "1"):
        httpx_mock.add_response(
            method="GET",
            url=f"{API}/runs/run-1/log?stream=events&limit={limit}&order=asc",
            json={"stream": "events", "records": [worded, raw]},
        )
    out = cli()
    assert out.exit_code == 0, out.output
    text = " ".join(out.output.split())
    assert "Lot: 1 entered" in text and "zone_events:m1:in" not in text
    assert "count" in text and "7" in text


def test_log_frames_take_their_filters_and_print_a_table(httpx_mock: HTTPXMock, cli: Any) -> None:
    frame = dict(
        EVENT, type=None, trigger=None, annotation_count=2, max_confidence=0.9, latency_total_ms=4.2
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/runs/run-1/log?stream=frames&limit=50&order=asc&class=car&track=3&min_confidence=0.5",
        json={"stream": "frames", "records": [frame]},
    )
    out = cli("--stream", "frames", "--class", "car", "--track", "3", "--min-confidence", "0.5")
    assert out.exit_code == 0, out.output
    assert "Frames of run run-1 (1)" in out.output and "car" in out.output


def test_log_filters_by_a_condition_branch_and_shows_it(
    httpx_mock: HTTPXMock, cli: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B549: ``--branch true|false`` filters on the branch a Condition routed each frame down, and the table says
    each frame's branch (only when the run has a Condition: a column of blanks otherwise)."""
    monkeypatch.setenv("COLUMNS", "240")
    frame = dict(EVENT, type=None, trigger=None, annotation_count=2, condition=False)
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/runs/run-1/log?stream=frames&limit=50&order=asc&branch=false",
        json={"stream": "frames", "records": [frame]},
    )
    out = cli("--stream", "frames", "--branch", "FALSE")
    assert out.exit_code == 0, out.output
    header = next(line for line in out.output.splitlines() if "annotations" in line)
    assert "branch" in header and "false" in out.output


def test_log_refuses_a_branch_that_is_not_true_or_false(httpx_mock: HTTPXMock, cli: Any) -> None:
    out = cli("--stream", "frames", "--branch", "maybe")
    assert out.exit_code != 0 and httpx_mock.get_requests() == []


def test_log_frames_without_a_condition_have_no_branch_column(
    httpx_mock: HTTPXMock, cli: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "240")
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/runs/run-1/log?stream=frames&limit=50&order=asc",
        json={"stream": "frames", "records": [dict(EVENT, type=None, trigger=None)]},
    )
    out = cli("--stream", "frames")
    assert out.exit_code == 0, out.output
    header = next(line for line in out.output.splitlines() if "annotations" in line)
    assert "branch" not in header


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
