"""``client.jobs`` - one view of every job (B533 #14).

Driven through the real transport against canned responses shaped as the developer API
answers (``routes/developer/jobs.py``; the backend's contract replay holds the real routes
to this model).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import ApiError, NotFoundError, PollTimeoutError, ValidationError
from pictograph.models.job import Job
from pictograph.resources.jobs import Jobs

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
JOBS = f"{BASE}/api/v1/developer/jobs"
TRAIN_ID = "c1c1c1c1-0000-4000-8000-000000000002"
CANCEL = f"/api/v1/developer/training/{TRAIN_ID}/cancel"


@pytest.fixture
def transport() -> Iterator[Transport]:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def jobs(transport: Transport) -> Jobs:
    return Jobs(transport)


def job_row(**over: Any) -> dict[str, Any]:
    """A job as ``GET /developer/jobs`` serializes it (every key it sends)."""
    row: dict[str, Any] = {
        "id": TRAIN_ID,
        "type": "training",
        "status": "running",
        "progress": 35.0,
        "name": "Swift Falcon",
        "dataset_id": "99999999-0000-4000-8000-000000000009",
        "error": None,
        "created_at": "2026-10-23T10:00:04+00:00",
        "updated_at": "2026-10-23T10:00:30+00:00",
        "completed_at": None,
        "cancel_path": CANCEL,
        "detail": {"pipeline_type": "rfdetr_detection", "current_epoch": 3, "total_epochs": 8},
    }
    row.update(over)
    return row


def page(
    rows: list[dict[str, Any]], *, total: int, limit: int = 50, offset: int = 0
) -> dict[str, Any]:
    return {
        "data": rows,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": total,
            "has_more": offset + len(rows) < total,
        },
    }


def _query(request: Any) -> dict[str, list[str]]:
    return parse_qs(urlsplit(str(request.url)).query)


def test_list_reads_active_jobs_by_default(httpx_mock: HTTPXMock, jobs: Jobs) -> None:
    httpx_mock.add_response(json=page([job_row()], total=1))
    [job] = list(jobs.list())
    assert isinstance(job, Job)
    assert (job.type, job.status, job.progress, job.name) == (
        "training",
        "running",
        35.0,
        "Swift Falcon",
    )
    assert job.active and job.cancellable
    assert job.detail["total_epochs"] == 8
    [request] = httpx_mock.get_requests()
    assert str(request.url).split("?")[0] == JOBS
    assert _query(request) == {"status": ["active"], "limit": ["50"], "offset": ["0"]}


def test_list_passes_status_and_type_and_pages(httpx_mock: HTTPXMock, jobs: Jobs) -> None:
    first = [
        job_row(id=f"c1c1c1c1-0000-4000-8000-00000000000{i}", status="completed", cancel_path=None)
        for i in range(2)
    ]
    httpx_mock.add_response(json=page(first, total=3, limit=2))
    httpx_mock.add_response(
        json=page(
            [job_row(id="c1c1c1c1-0000-4000-8000-00000000000f", status="failed", cancel_path=None)],
            total=3,
            limit=2,
            offset=2,
        )
    )
    found = list(jobs.list(status="finished", type="training", page_size=2))
    assert [j.status for j in found] == ["completed", "completed", "failed"]
    assert not any(j.cancellable or j.active for j in found)
    first_q, second_q = (_query(r) for r in httpx_mock.get_requests())
    assert first_q == {
        "status": ["finished"],
        "type": ["training"],
        "limit": ["2"],
        "offset": ["0"],
    }
    assert second_q["offset"] == ["2"]


def test_list_caps_the_page_at_the_apis_limit_and_stops_at_max_total(
    httpx_mock: HTTPXMock, jobs: Jobs
) -> None:
    httpx_mock.add_response(json=page([job_row(), job_row()], total=500, limit=100))
    found = list(jobs.list(page_size=500, max_total=2))
    assert len(found) == 2
    [request] = httpx_mock.get_requests()
    assert _query(request)["limit"] == ["100"]


def test_a_status_word_this_sdk_does_not_know_still_reads(
    httpx_mock: HTTPXMock, jobs: Jobs
) -> None:
    httpx_mock.add_response(json={"data": job_row(status="paused", cancel_path=None, detail={})})
    job = jobs.get(TRAIN_ID)
    assert job.status == "paused" and not job.active


def test_get_reads_one_job_by_its_quoted_id(httpx_mock: HTTPXMock, jobs: Jobs) -> None:
    httpx_mock.add_response(
        json={"data": job_row(type="connector_import", detail={"processed_items": 20})}
    )
    job = jobs.get("b0b0b0b0000040008000000000000001")
    assert job.type == "connector_import" and job.detail == {"processed_items": 20}
    assert str(httpx_mock.get_requests()[0].url) == f"{JOBS}/b0b0b0b0000040008000000000000001"
    httpx_mock.add_response(status_code=404, json={"detail": "Job not found"})
    with pytest.raises(NotFoundError):
        jobs.get("a/b")
    assert str(httpx_mock.get_requests()[1].url) == f"{JOBS}/a%2Fb"


def test_wait_returns_the_completed_job(httpx_mock: HTTPXMock, jobs: Jobs) -> None:
    httpx_mock.add_response(json={"data": job_row(status="queued", progress=0.0)})
    httpx_mock.add_response(json={"data": job_row(status="running", progress=50.0)})
    httpx_mock.add_response(
        json={"data": job_row(status="completed", progress=100.0, cancel_path=None)}
    )
    slept: list[float] = []
    job = jobs.wait(TRAIN_ID, poll_interval=0.5, sleep=slept.append)
    assert job.status == "completed" and slept == [0.5, 0.5]


@pytest.mark.parametrize(
    ("status", "code"), [("failed", "job_failed"), ("cancelled", "job_cancelled")]
)
def test_wait_raises_with_the_job_when_it_ended_badly(
    httpx_mock: HTTPXMock, jobs: Jobs, status: str, code: str
) -> None:
    httpx_mock.add_response(
        json={"data": job_row(status=status, error="out of memory", cancel_path=None)}
    )
    with pytest.raises(ApiError) as caught:
        jobs.wait(TRAIN_ID, sleep=lambda _s: None)
    assert caught.value.code == code
    assert caught.value.response["id"] == TRAIN_ID
    assert "out of memory" in str(caught.value)


def test_wait_times_out_while_the_job_keeps_running(httpx_mock: HTTPXMock, jobs: Jobs) -> None:
    httpx_mock.add_response(json={"data": job_row()})
    with pytest.raises(PollTimeoutError, match=r"client\.jobs\.wait"):
        jobs.wait(TRAIN_ID, timeout=1e-9, sleep=lambda _s: None)


@pytest.mark.parametrize(("poll_interval", "timeout"), [(0, 10), (1, 0)])
def test_wait_refuses_a_non_positive_interval_or_timeout(
    jobs: Jobs, poll_interval: float, timeout: float
) -> None:
    with pytest.raises(ValueError):
        jobs.wait(TRAIN_ID, poll_interval=poll_interval, timeout=timeout)


def test_cancel_posts_to_the_route_the_job_names(httpx_mock: HTTPXMock, jobs: Jobs) -> None:
    httpx_mock.add_response(json={"data": job_row()})
    httpx_mock.add_response(json={"data": {"id": TRAIN_ID, "status": "cancelled"}})
    httpx_mock.add_response(json={"data": job_row(status="cancelled", cancel_path=None)})
    job = jobs.cancel(TRAIN_ID)
    assert job.status == "cancelled"
    get, post, reread = httpx_mock.get_requests()
    assert (get.method, post.method, reread.method) == ("GET", "POST", "GET")
    assert str(post.url) == f"{BASE}{CANCEL}"


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (job_row(status="completed", cancel_path=None), "already completed"),
        (job_row(type="video_extract", cancel_path=None), "video_extract job cannot be cancelled"),
        (job_row(cancel_path="https://elsewhere.example/steal"), "outside the developer API"),
    ],
)
def test_cancel_refuses_without_posting(
    httpx_mock: HTTPXMock, jobs: Jobs, row: dict[str, Any], message: str
) -> None:
    httpx_mock.add_response(json={"data": row})
    with pytest.raises(ValidationError, match=message):
        jobs.cancel(TRAIN_ID)
    assert [r.method for r in httpx_mock.get_requests()] == ["GET"]
