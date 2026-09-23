"""Tests for ``client.workflows.invoke`` - the WF-D on-demand invoke (slice 1).

Covers: the per-mode request body (dataset + gcs-uri), None-field exclusion, the
client-side ``extra="forbid"`` guard on a dict source, the async return shape (run_id
+ status_url), and ``wait=True`` reusing ``wait_for_run`` (terminal + error).
"""

from __future__ import annotations

import json as _json
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import ApiError
from pictograph.models.workflow import WorkflowInvokeSource, WorkflowRun, WorkflowRunCreated
from pictograph.resources.workflows import Workflows

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
_API = f"{BASE}/api/v1/developer/workflows"
WF = "fedcba98-1111-2222-3333-444455556666"  # a uuid -> no name-resolution GET


@pytest.fixture
def transport() -> Transport:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def workflows(transport: Transport) -> Workflows:
    return Workflows(transport)


def _run(**o: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "run-9",
        "organization_id": "org-1",
        "workflow_id": WF,
        "status": "processing",
        "progress": 0.0,
        "frames_total": None,
        "frames_done": 0,
        "sample_fps": None,
        "step_results": {},
        "artifacts": [],
        "warnings": [],
        "deposit_micro_usd": 1000,
        "final_micro_usd": None,
        "error": None,
        "created_at": "2026-09-23T00:00:00Z",
        "completed_at": None,
    }
    base.update(o)
    return base


def _created_json() -> dict[str, Any]:
    return {
        "success": True,
        "run_id": "run-9",
        "deposit_micro_usd": 1000,
        "status_url": "/api/v1/developer/workflows/runs/run-9",
    }


def test_invoke_dataset_async_returns_created(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(method="POST", url=f"{_API}/{WF}/invoke", json=_created_json())
    created = workflows.invoke(
        WF, source={"kind": "dataset", "dataset": "My Data", "directory_path": "/sub"}
    )
    assert isinstance(created, WorkflowRunCreated)
    assert created.run_id == "run-9"
    assert created.status_url == "/api/v1/developer/workflows/runs/run-9"
    body = _json.loads(httpx_mock.get_requests()[-1].content)
    assert body == {"source": {"kind": "dataset", "dataset": "My Data", "directory_path": "/sub"}}


def test_invoke_video_model_excludes_none_fields(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(method="POST", url=f"{_API}/{WF}/invoke", json=_created_json())
    src = WorkflowInvokeSource(
        kind="video", gcs_uri="gs://my-bucket/org/clip.mp4", sample_fps=2.0
    )
    workflows.invoke(WF, source=src)
    body = _json.loads(httpx_mock.get_requests()[-1].content)
    # dataset/directory_path/duration_seconds are None -> omitted from the wire body.
    assert body == {
        "source": {
            "kind": "video",
            "gcs_uri": "gs://my-bucket/org/clip.mp4",
            "sample_fps": 2.0,
        }
    }


def test_invoke_image_model_is_just_uri(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(method="POST", url=f"{_API}/{WF}/invoke", json=_created_json())
    workflows.invoke(
        WF, source=WorkflowInvokeSource(kind="image", gcs_uri="gs://my-bucket/org/a.png")
    )
    body = _json.loads(httpx_mock.get_requests()[-1].content)
    assert body == {"source": {"kind": "image", "gcs_uri": "gs://my-bucket/org/a.png"}}


def test_invoke_dict_source_forbids_unknown_field(workflows: Workflows) -> None:
    # A mistyped field is caught CLIENT-SIDE (extra="forbid"), before any request.
    with pytest.raises(ValidationError, match="datasets"):
        workflows.invoke(WF, source={"kind": "dataset", "datasets": "x"})


def test_invoke_wait_returns_terminal_run(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(method="POST", url=f"{_API}/{WF}/invoke", json=_created_json())
    # wait_for_run's FIRST get_run sees completed -> returns immediately (no sleep).
    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/runs/run-9",
        json={"run": _run(status="completed", progress=100.0, frames_done=16, frames_total=16)},
    )
    run = workflows.invoke(
        WF, source={"kind": "dataset", "dataset": "d"}, wait=True, poll_interval=0.01, timeout=30.0
    )
    assert isinstance(run, WorkflowRun)
    assert run.status == "completed" and run.frames_done == 16


def test_invoke_wait_propagates_run_error(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(method="POST", url=f"{_API}/{WF}/invoke", json=_created_json())
    httpx_mock.add_response(
        method="GET", url=f"{_API}/runs/run-9", json={"run": _run(status="error", error="GPU OOM")}
    )
    with pytest.raises(ApiError, match="GPU OOM"):
        workflows.invoke(
            WF, source={"kind": "dataset", "dataset": "d"}, wait=True, poll_interval=0.01
        )
