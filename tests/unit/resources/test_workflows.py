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
        "deposit_micro_usd": 1000,
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
        json={"success": True, "run_id": "run-1", "deposit_micro_usd": 1000},
    )
    created = workflows.run("fedcba98-1111-2222-3333-444455556666")
    assert (
        isinstance(created, WorkflowRunCreated)
        and created.run_id == "run-1"
        and created.deposit_micro_usd == 1000
    )


def test_get_run_returns_typed(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(method="GET", url=f"{_API}/runs/run-1", json={"run": _run()})
    run = workflows.get_run("run-1")
    assert isinstance(run, WorkflowRun) and run.frames_done == 42 and run.status == "processing"


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
