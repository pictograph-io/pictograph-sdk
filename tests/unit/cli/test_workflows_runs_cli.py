"""``pictograph workflows runs`` - a workflow's runs from the terminal, each stream session among them (a stream is a
run of its workflow). Drives the REAL client over a mocked HTTP transport."""

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
WF = "fedcba98-1111-2222-3333-444455556666"


def _run(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "r1",
        "status": "completed",
        "cost_micro_usd": 1,
        "cost_settled": True,
        "frames_done": 672,
        "frames_total": None,
        "progress": 1.0,
        "artifact_count": 0,
        "created_at": "2026-10-28T12:00:00Z",
    }
    base.update(over)
    return base


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_test")
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "200")
    runner = CliRunner()
    return lambda *args: runner.invoke(app, ["workflows", "runs", WF, *args])


def test_runs_say_what_each_ran_over_and_keep_streams_when_asked(
    httpx_mock: HTTPXMock, cli: Any
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/{WF}/runs?limit=50&offset=0&kind=stream",
        json={
            "data": [
                _run(
                    id="s1",
                    source={"kind": "stream", "via": "whip"},
                    stream_session="0123456789abcdef",
                ),
                _run(
                    id="s2",
                    source={"kind": "stream", "via": "rtsp"},
                    stream_session="00000000000000aa",
                ),
            ],
            "pagination": {"limit": 50, "offset": 0, "total": 2, "has_more": False},
        },
    )
    out = cli("--kind", "stream")
    assert out.exit_code == 0, out.output
    assert "source" in out.output and "stream" in out.output and "rtsp" in out.output


def test_runs_as_json_carry_their_source_and_session(httpx_mock: HTTPXMock, cli: Any) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/{WF}/runs?limit=50&offset=0",
        json={
            "data": [
                _run(id="v1", source={"kind": "video", "via": None}),
                _run(
                    id="s1",
                    source={"kind": "stream", "via": "whep"},
                    stream_session="0123456789abcdef",
                ),
            ],
            "pagination": {"limit": 50, "offset": 0, "total": 2, "has_more": False},
        },
    )
    out = cli("--json")
    assert out.exit_code == 0, out.output
    video, stream = json.loads(out.output)
    assert video["source"] == {"kind": "video"} and "stream_session" not in video
    assert (
        stream["source"] == {"kind": "stream", "via": "whep"}
        and stream["stream_session"] == "0123456789abcdef"
    )


def test_a_kind_it_does_not_know_is_refused(cli: Any) -> None:
    out = cli("--kind", "everything")
    assert out.exit_code != 0
    assert "--kind" in out.output
