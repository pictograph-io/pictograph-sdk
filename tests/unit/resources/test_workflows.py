"""Tests for ``pictograph.resources.workflows.Workflows``.

Coverage: create (body carries name+graph), list, get, update, delete, run
(typed WorkflowRunCreated), get_run, cancel_run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import ApiError, ForbiddenError, PollTimeoutError
from pictograph.models.workflow import Workflow, WorkflowRun, WorkflowRunCreated
from pictograph.resources.workflows import Workflows

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
_API = f"{BASE}/api/v1/developer/workflows"


@pytest.fixture
def transport() -> Transport:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def workflows(transport: Transport) -> Workflows:
    return Workflows(transport)


def _wf(**o: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "fedcba98-1111-2222-3333-444455556666",
        "organization_id": "org-1",
        "name": "Line counter",
        "description": None,
        "graph": {"version": 1, "nodes": [], "edges": []},
        "template_key": "line_counter",
        "status": "draft",
        "last_run_id": None,
        "created_at": "2026-06-03T00:00:00Z",
        "updated_at": "2026-06-03T00:00:00Z",
    }
    base.update(o)
    return base


def _run(**o: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "run-1",
        "organization_id": "org-1",
        "workflow_id": "fedcba98-1111-2222-3333-444455556666",
        "status": "processing",
        "progress": 42.0,
        "frames_total": 100,
        "frames_done": 42,
        "sample_fps": 5.0,
        "step_results": {},
        "artifacts": [],
        "final_micro_usd": None,
        "error": None,
        "created_at": "2026-06-03T00:00:00Z",
        "completed_at": None,
    }
    base.update(o)
    return base


def test_create_sends_name_and_graph(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(method="POST", url=f"{_API}/", json={"workflow": _wf()})
    wf = workflows.create(
        "Line counter", {"version": 1, "nodes": [], "edges": []}, template_key="line_counter"
    )
    assert isinstance(wf, Workflow) and wf.template_key == "line_counter"
    body = httpx_mock.get_requests()[-1].read().decode()
    assert "Line counter" in body and "nodes" in body


def test_list_returns_typed(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/",
        json={"workflows": [_wf(), _wf(id="fedcba98-2222-3333-4444-555566667777")]},
    )
    result = workflows.list()
    assert len(result) == 2 and all(isinstance(w, Workflow) for w in result)


def test_get_returns_typed(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{_API}/fedcba98-1111-2222-3333-444455556666", json={"workflow": _wf()}
    )
    assert (
        workflows.get("fedcba98-1111-2222-3333-444455556666").id
        == "fedcba98-1111-2222-3333-444455556666"
    )


def test_update_sends_patch(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{_API}/fedcba98-1111-2222-3333-444455556666",
        json={"workflow": _wf(status="ready")},
    )
    wf = workflows.update("fedcba98-1111-2222-3333-444455556666", status="ready")
    assert wf.status == "ready"


def test_delete(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(
        method="DELETE", url=f"{_API}/fedcba98-1111-2222-3333-444455556666", json={"success": True}
    )
    workflows.delete("fedcba98-1111-2222-3333-444455556666")
    req = httpx_mock.get_requests()[0]
    assert req.method == "DELETE"
    assert req.url.path.endswith("/fedcba98-1111-2222-3333-444455556666")


def test_bulk_delete_round_trip(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{_API}/bulk-delete",
        json={
            "success": True,
            "deleted": [
                "fedcba98-1111-2222-3333-444455556666",
                "fedcba98-2222-3333-4444-555566667777",
            ],
            "not_found": ["fedcba98-3333-4444-5555-666677778888"],
            "count": 2,
        },
    )
    result = workflows.bulk_delete(
        [
            "fedcba98-1111-2222-3333-444455556666",
            "fedcba98-2222-3333-4444-555566667777",
            "fedcba98-3333-4444-5555-666677778888",
        ]
    )
    assert result.deleted == [
        "fedcba98-1111-2222-3333-444455556666",
        "fedcba98-2222-3333-4444-555566667777",
    ]
    assert result.not_found == ["fedcba98-3333-4444-5555-666677778888"]
    assert result.count == 2
    import json as _json

    assert _json.loads(httpx_mock.get_requests()[0].content) == {
        "workflow_ids": [
            "fedcba98-1111-2222-3333-444455556666",
            "fedcba98-2222-3333-4444-555566667777",
            "fedcba98-3333-4444-5555-666677778888",
        ]
    }


def test_bulk_cancel_runs_round_trip(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{_API}/runs/bulk-cancel",
        json={"success": True, "succeeded": ["r1"], "not_found": ["r2"], "count": 1},
    )
    result = workflows.bulk_cancel_runs(["r1", "r2"])
    assert result.succeeded == ["r1"]
    assert result.not_found == ["r2"]
    assert result.count == 1
    import json as _json

    assert _json.loads(httpx_mock.get_requests()[0].content) == {"run_ids": ["r1", "r2"]}


def test_bulk_delete_403_propagates(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{_API}/bulk-delete",
        status_code=403,
        json={"detail": "Insufficient permissions to manage workflows"},
    )
    with pytest.raises(ForbiddenError):
        workflows.bulk_delete(["fedcba98-1111-2222-3333-444455556666"])


def test_run_returns_created(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{_API}/fedcba98-1111-2222-3333-444455556666/run",
        json={"success": True, "run_id": "run-1"},
    )
    created = workflows.run("fedcba98-1111-2222-3333-444455556666")
    assert isinstance(created, WorkflowRunCreated) and created.run_id == "run-1"


def test_get_run_returns_typed(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(method="GET", url=f"{_API}/runs/run-1", json={"run": _run()})
    run = workflows.get_run("run-1")
    assert isinstance(run, WorkflowRun) and run.frames_done == 42 and run.status == "processing"


def test_a_run_names_no_engine(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    """Every run is PRT's (W1.7), so a run says no engine; a server that still sends one (from before
    W1.7) parses, its ``engine`` dropped."""
    httpx_mock.add_response(
        method="GET", url=f"{_API}/runs/run-1", json={"run": _run(engine="prt")}
    )
    run = workflows.get_run("run-1")
    assert not hasattr(run, "engine") and "engine" not in run.model_dump()


def test_cancel_run(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(method="POST", url=f"{_API}/runs/run-1/cancel", json={"success": True})
    workflows.cancel_run("run-1")
    req = httpx_mock.get_requests()[0]
    assert req.method == "POST"
    assert req.url.path.endswith("/runs/run-1/cancel")


def test_wait_for_run_polls_until_completed(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{_API}/runs/run-1", json={"run": _run(status="processing")}
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/runs/run-1",
        json={"run": _run(status="completed", progress=100.0, frames_done=100)},
    )
    sleeps: list[float] = []
    run = workflows.wait_for_run("run-1", poll_interval=2.0, timeout=120.0, sleep=sleeps.append)
    assert run.status == "completed" and run.frames_done == 100
    assert sleeps == [2.0]


def test_wait_for_run_raises_on_error_status(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{_API}/runs/run-1", json={"run": _run(status="error", error="GPU OOM")}
    )
    with pytest.raises(ApiError, match="GPU OOM"):
        workflows.wait_for_run("run-1", poll_interval=0.1, sleep=lambda _: None)


def test_wait_for_run_polltimeout(
    httpx_mock: HTTPXMock, workflows: Workflows, monkeypatch: pytest.MonkeyPatch
) -> None:
    for _ in range(3):
        httpx_mock.add_response(
            method="GET", url=f"{_API}/runs/run-1", json={"run": _run(status="processing")}
        )
    times = iter([100.0, 100.0, 105.0, 110.0])  # deadline = 100 + 10 = 110
    monkeypatch.setattr("pictograph.resources.workflows.time.monotonic", lambda: next(times))
    with pytest.raises(PollTimeoutError, match="did not complete"):
        workflows.wait_for_run("run-1", poll_interval=1.0, timeout=10.0, sleep=lambda _: None)


def test_wait_for_run_argument_validation(workflows: Workflows) -> None:
    with pytest.raises(ValueError, match="poll_interval"):
        workflows.wait_for_run("run-1", poll_interval=0.0)
    with pytest.raises(ValueError, match="timeout"):
        workflows.wait_for_run("run-1", timeout=0.0)


_WF_ID = "fedcba98-1111-2222-3333-444455556666"


def _summary(**o: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "run-1",
        "status": "completed",
        "created_at": "2026-06-03T00:00:00Z",
        "completed_at": "2026-06-03T00:01:00Z",
        "runtime_seconds": 60.0,
        "cost_micro_usd": 850,
        "cost_settled": True,
        "frames_done": 100,
        "frames_total": 100,
        "progress": 100.0,
        "artifact_count": 2,
        "error": None,
    }
    base.update(o)
    return base


def test_list_runs_returns_typed_summaries(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    # A uuid ref resolves locally (looks_like_id), so no extra list round-trip.
    from pictograph.models.workflow import WorkflowRunSummary

    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/{_WF_ID}/runs?limit=50&offset=0",
        json={
            # r2 as a server from before W1.7 lists it, its engine named: dropped.
            "data": [_summary(id="r1"), _summary(id="r2", status="processing", engine="prt")],
            "pagination": {"limit": 50, "offset": 0, "total": 2, "has_more": False},
        },
    )
    runs = workflows.list_runs(_WF_ID)
    assert [r.id for r in runs] == ["r1", "r2"]
    assert all(isinstance(r, WorkflowRunSummary) for r in runs)
    assert runs[0].cost_micro_usd == 850 and runs[0].cost_settled is True
    assert runs[0].artifact_count == 2  # count, not the artifacts payload
    assert all("engine" not in r.model_dump() for r in runs)


def test_a_stream_session_is_a_run_and_list_runs_keeps_streams_or_files(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    """Each stream session on a workflow's endpoint is a run of it: its ``source`` says so (and how its video came),
    its ``stream_session`` beside it; ``kind`` keeps streams or file runs (``"all"`` sends no filter)."""
    stream = _summary(
        id="s1",
        source={"kind": "stream", "via": "rtsp"},
        stream_session="0123456789abcdef",
        frames_total=None,
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/{_WF_ID}/runs?limit=50&offset=0&kind=stream",
        json={
            "data": [stream],
            "pagination": {"limit": 50, "offset": 0, "total": 1, "has_more": False},
        },
    )
    (run,) = workflows.list_runs(_WF_ID, kind="stream")
    assert run.source is not None and (run.source.kind, run.source.via) == ("stream", "rtsp")
    assert run.stream_session == "0123456789abcdef"
    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/{_WF_ID}/runs?limit=50&offset=0",
        json={
            "data": [_summary(id="v1", source={"kind": "video", "via": None}), _summary(id="old")],
            "pagination": {"limit": 50, "offset": 0, "total": 2, "has_more": False},
        },
    )
    video, old = workflows.list_runs(_WF_ID)
    assert video.source is not None and (video.source.kind, video.source.via) == ("video", None)
    assert old.source is None and old.stream_session is None, (
        "a server before streams were runs: none"
    )
    # A kind the platform adds later reads as itself.
    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/{_WF_ID}/runs?limit=50&offset=0&kind=file",
        json={
            "data": [_summary(id="x1", source={"kind": "export", "via": None})],
            "pagination": {"limit": 50, "offset": 0, "total": 1, "has_more": False},
        },
    )
    (later,) = workflows.list_runs(_WF_ID, kind="file")
    assert later.source is not None and later.source.kind == "export"


def test_iter_runs_keeps_its_kind_on_every_page(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/{_WF_ID}/runs?limit=1&offset=0&kind=stream",
        json={
            "data": [_summary(id="a", source={"kind": "stream", "via": "whip"})],
            "pagination": {"limit": 1, "offset": 0, "total": 2, "has_more": True},
        },
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/{_WF_ID}/runs?limit=1&offset=1&kind=stream",
        json={
            "data": [_summary(id="b", source={"kind": "stream", "via": "whep"})],
            "pagination": {"limit": 1, "offset": 1, "total": 2, "has_more": False},
        },
    )
    assert [r.id for r in workflows.iter_runs(_WF_ID, page_size=1, kind="stream")] == ["a", "b"]


def test_iter_runs_pages_through_all(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/{_WF_ID}/runs?offset=0&limit=2",
        json={
            "data": [_summary(id="a"), _summary(id="b")],
            "pagination": {"limit": 2, "offset": 0, "total": 3, "has_more": True},
        },
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/{_WF_ID}/runs?offset=2&limit=2",
        json={
            "data": [_summary(id="c")],
            "pagination": {"limit": 2, "offset": 2, "total": 3, "has_more": False},
        },
    )
    ids = [r.id for r in workflows.iter_runs(_WF_ID, page_size=2)]
    assert ids == ["a", "b", "c"]


# ── a run's log (L1) ──────────────────────────────────────────────────────────

_EVENT = {
    "ts": "2026-10-18T07:00:01.000000Z",
    "engine_run": "run-a",
    "seq": 2,
    "record_index": 0,
    "type": "event",
    "source": "cam",
    "frame": 12,
    "pts": 0.4,
    "image": "cam-000012.jpg",
    "image_id": None,
    "event_id": "e-1",
    "trigger": "Gate",
    "sequence": 1,
    "annotation_count": 1,
    "classes": ["car"],
    "track_ids": ["3"],
    "capture_kind": None,
    "capture_ok": None,
    "capture_detail": None,
    "capture_name": None,
    "capture_uri": None,
    "capture_bytes": None,
    "clock_suspect": None,
    "rule_values": {"count": 1},
}


def test_run_log_reads_a_page_and_sends_only_the_filters_given(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    from pictograph.models.workflow import RunLogPage

    httpx_mock.add_response(
        method="GET",
        url=(
            f"{_API}/runs/run-1/log?stream=frames&limit=20&order=desc&full=true&class=car&track_id=3"
            "&min_confidence=0.5"
        ),
        json={
            "stream": "frames",
            "order": "desc",
            "records": [_EVENT],
            "next_cursor": "c-2",
            "log_frames": True,
        },
    )
    page = workflows.run_log(
        "run-1",
        stream="frames",
        limit=20,
        order="desc",
        full=True,
        class_name="car",
        track=3,
        min_confidence=0.5,
    )
    assert isinstance(page, RunLogPage) and page.next_cursor == "c-2" and page.log_frames is True
    rec = page.records[0]
    assert (rec.engine_run, rec.seq, rec.record_index, rec.trigger, rec.classes) == (
        "run-a",
        2,
        0,
        "Gate",
        ["car"],
    )
    assert rec.ts.year == 2026 and rec.rule_values == {"count": 1}


def test_run_log_filters_by_a_condition_branch_both_ways(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    """B549: the branch a Condition block routed a frame down is a filter - ``branch=False`` sends ``false`` (a
    filter, never "no filter"), ``True`` sends ``true`` - and each record says its branch."""
    for word, branch in (("false", False), ("true", True)):
        httpx_mock.add_response(
            method="GET",
            url=f"{_API}/runs/run-1/log?stream=frames&limit=50&order=asc&branch={word}",
            json={"stream": "frames", "records": [dict(_EVENT, condition=branch)]},
        )
        page = workflows.run_log("run-1", stream="frames", branch=branch)
        assert page.records[0].condition is branch
    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/runs/run-1/log?stream=frames&limit=200&order=asc&branch=false",
        json={"stream": "frames", "records": [dict(_EVENT, condition=False)]},
    )
    assert [
        r.condition for r in workflows.iter_run_log("run-1", stream="frames", branch=False)
    ] == [False]
    # A workflow without a Condition: no branch on its records.
    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/runs/run-1/log?stream=events&limit=50&order=asc",
        json={"stream": "events", "records": [_EVENT]},
    )
    assert workflows.run_log("run-1").records[0].condition is None


def test_run_log_defaults_to_the_first_page_of_events(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/runs/run-1/log?stream=events&limit=50&order=asc",
        json={"stream": "events", "records": []},
    )
    page = workflows.run_log("run-1")
    assert page.records == [] and page.next_cursor is None and page.order == "asc"


def test_iter_run_log_follows_the_cursor_to_the_last_page(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    second = dict(_EVENT, record_index=1)
    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/runs/run-1/log?stream=events&limit=1&order=asc",
        json={"stream": "events", "records": [_EVENT], "next_cursor": "c-1"},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/runs/run-1/log?stream=events&limit=1&order=asc&cursor=c-1",
        json={"stream": "events", "records": [second], "next_cursor": None},
    )
    got = list(workflows.iter_run_log("run-1", page_size=1))
    assert [r.record_index for r in got] == [0, 1]
    assert len(httpx_mock.get_requests()) == 2


def test_iter_run_log_stops_at_max_total(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/runs/run-1/log?stream=events&limit=2&order=asc",
        json={
            "stream": "events",
            "records": [_EVENT, dict(_EVENT, record_index=1)],
            "next_cursor": "c-1",
        },
    )
    assert len(list(workflows.iter_run_log("run-1", page_size=2, max_total=1))) == 1
    assert len(httpx_mock.get_requests()) == 1, "no page past the total"


def test_run_log_summary_counts_each_stream(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/runs/run-1/log/summary",
        json={
            "log_frames": False,
            "streams": {
                "frames": {"count": 0, "first_ts": None, "last_ts": None},
                "events": {"count": 12, "first_ts": _EVENT["ts"], "last_ts": _EVENT["ts"]},
                "stats": {"count": 1, "first_ts": _EVENT["ts"], "last_ts": _EVENT["ts"]},
            },
        },
    )
    s = workflows.run_log_summary("run-1")
    assert (
        s.log_frames is False
        and s.streams["events"].count == 12
        and s.streams["frames"].first_ts is None
    )


def test_run_log_summary_counts_each_condition_branch(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    from pictograph import RunLogBranchCounts

    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/runs/run-1/log/summary",
        json={
            "log_frames": True,
            "streams": {
                "frames": {"count": 80, "condition": {"passed": 37, "failed": 43}},
                "events": {"count": 0, "condition": None},
                "stats": {"count": 1},
            },
        },
    )
    s = workflows.run_log_summary("run-1")
    assert s.streams["frames"].condition == RunLogBranchCounts(passed=37, failed=43)
    assert s.streams["events"].condition is None and s.streams["stats"].condition is None


def test_a_run_says_whether_it_wrote_a_log(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{_API}/runs/run-1", json={"run": _run(has_log=True)}
    )
    assert workflows.get_run("run-1").has_log is True
    httpx_mock.add_response(method="GET", url=f"{_API}/runs/run-2", json={"run": _run(id="run-2")})
    assert workflows.get_run("run-2").has_log is False, "a server that does not say logged nothing"


def test_a_run_reports_the_tensorrt_engines_it_builds(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    """The first run of a model on a GPU builds its TensorRT engine before any frame; the run says so
    while it builds, and keeps the record. A run whose engines were cached has none."""
    building = {
        "state": "building",
        "models": [
            {
                "model_id": "m-1",
                "name": "Swift Falcon",
                "precision": "fp16",
                "state": "built",
                "started_at": "2026-10-18T20:00:00Z",
                "estimate_s": 92.5,
                "build_s": 88.1,
            },
            {
                "model_id": "m-2",
                "precision": "fp32",
                "state": "building",
                "started_at": "2026-10-18T20:01:30Z",
            },
        ],
    }
    httpx_mock.add_response(
        method="GET", url=f"{_API}/runs/run-1", json={"run": _run(engine_build=building)}
    )
    eb = workflows.get_run("run-1").engine_build
    assert eb is not None and eb.state == "building"
    assert [(m.model_id, m.state, m.build_s) for m in eb.models] == [
        ("m-1", "built", 88.1),
        ("m-2", "building", None),
    ]
    assert (
        eb.models[0].name == "Swift Falcon"
        and eb.models[1].name is None
        and eb.models[1].estimate_s is None
    )
    assert eb.models[1].started_at is not None and eb.models[1].started_at.minute == 1
    httpx_mock.add_response(method="GET", url=f"{_API}/runs/run-2", json={"run": _run(id="run-2")})
    assert workflows.get_run("run-2").engine_build is None, "cached: no build to report"


# ── compute options: a workflow's GPU; a Model block's precision ─────────────


def _options() -> dict[str, Any]:
    return {
        "success": True,
        "gpus": [
            {
                "gpu": "t4",
                "label": "T4 GPU",
                "vram_gb": 16,
                "description": "16 GB VRAM",
                "rate_per_min_micro_usd": 19667,
            },
            {
                "gpu": "l4",
                "label": "L4 GPU",
                "vram_gb": 24,
                "description": "24 GB VRAM",
                "rate_per_min_micro_usd": 26667,
            },
        ],
        "precisions": [
            {"precision": "fp32", "label": "Full precision (fp32)", "description": "As trained."},
            {"precision": "fp16", "label": "Fast (fp16)", "description": "PRT's fast mode."},
        ],
        "default": {"gpu": "t4", "precision": "fp32"},
    }


def test_compute_options_lists_each_gpu_with_its_price(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(method="GET", url=f"{_API}/compute-options", json=_options())
    opts = workflows.compute_options()
    assert [(g.gpu, g.vram_gb) for g in opts.gpus] == [("t4", 16), ("l4", 24)]
    assert opts.gpus[1].rate_per_min_micro_usd == 26667
    assert [p.precision for p in opts.precisions] == ["fp32", "fp16"]
    assert opts.default.model_dump() == {"gpu": "t4"}


def test_a_workflow_reads_its_compute(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    wid = "fedcba98-1111-2222-3333-444455556666"
    httpx_mock.add_response(
        method="GET", url=f"{_API}/{wid}", json={"workflow": _wf(compute={"gpu": "l4"})}
    )
    httpx_mock.add_response(method="GET", url=f"{_API}/{wid}", json={"workflow": _wf()})
    wf = workflows.get(wid)
    assert wf.compute.model_dump() == {"gpu": "l4"}
    bare = workflows.get(wid)
    assert bare.compute.model_dump() == {"gpu": "t4"}, "a server that does not say: the default"


def test_create_and_update_send_the_compute_given(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    import json

    from pictograph.models.workflow import WorkflowCompute

    wid = "fedcba98-1111-2222-3333-444455556666"
    httpx_mock.add_response(method="POST", url=f"{_API}/", json={"workflow": _wf()})
    httpx_mock.add_response(method="PATCH", url=f"{_API}/{wid}", json={"workflow": _wf()})
    workflows.create(
        "Line counter", {"version": 1, "nodes": [], "edges": []}, compute={"gpu": "l4"}
    )
    workflows.update(wid, compute=WorkflowCompute(gpu="t4"))
    created, updated = (json.loads(r.content) for r in httpx_mock.get_requests())
    assert created["compute"] == {"gpu": "l4"}
    assert updated == {"compute": {"gpu": "t4"}}


def test_a_run_sends_no_body_it_runs_on_its_workflows_endpoint(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    """B579 slice 5: a run runs on its workflow's endpoint, on the workflow's GPU - it names none."""
    wid = "fedcba98-1111-2222-3333-444455556666"
    httpx_mock.add_response(
        method="POST", url=f"{_API}/{wid}/run", json={"success": True, "run_id": "run-1"}
    )
    workflows.run(wid)
    (request,) = httpx_mock.get_requests()
    assert request.content == b""


def test_a_run_takes_no_precision_it_is_each_model_blocks_own(workflows: Workflows) -> None:
    import inspect

    assert "precision" not in inspect.signature(workflows.run).parameters
    assert "precision" not in inspect.signature(workflows.invoke).parameters


def test_a_run_reads_what_it_ran_with(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/runs/run-1",
        json={"run": _run(compute={"gpu": "l4", "precision": "fp16"})},
    )
    ran = workflows.get_run("run-1").compute
    assert (ran.gpu, ran.precision) == ("l4", "fp16")
