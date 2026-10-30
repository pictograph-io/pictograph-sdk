"""``AsyncJobs`` - the async twin of ``client.jobs`` (B533 #14)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pictograph.aio.resources.jobs import AsyncJobs
from pictograph.exceptions import ApiError, ValidationError
from tests.unit.resources.test_jobs import BASE, CANCEL, JOBS, TRAIN_ID, job_row, page

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

    from pictograph._http.async_transport import AsyncTransport

pytestmark = pytest.mark.anyio


async def test_list_and_get(httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
    jobs = AsyncJobs(transport)
    httpx_mock.add_response(json=page([job_row()], total=1))
    httpx_mock.add_response(json={"data": job_row(type="workflow_run")})
    found = [j async for j in jobs.list(type="training")]
    job = await jobs.get(TRAIN_ID)
    assert [j.id for j in found] == [TRAIN_ID] and job.type == "workflow_run"
    listing, one = httpx_mock.get_requests()
    assert str(listing.url).split("?")[0] == JOBS and "type=training" in str(listing.url)
    assert str(one.url) == f"{JOBS}/{TRAIN_ID}"


async def test_wait_and_cancel_like_the_sync_twin(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    jobs = AsyncJobs(transport)
    httpx_mock.add_response(json={"data": job_row()})
    httpx_mock.add_response(json={"data": job_row(status="completed", cancel_path=None)})
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    assert (await jobs.wait(TRAIN_ID, poll_interval=2, sleep=sleep)).status == "completed"
    assert slept == [2]

    httpx_mock.add_response(json={"data": job_row(status="failed", error="boom", cancel_path=None)})
    with pytest.raises(ApiError) as caught:
        await jobs.wait(TRAIN_ID, sleep=sleep)
    assert caught.value.code == "job_failed"

    httpx_mock.add_response(json={"data": job_row()})
    httpx_mock.add_response(json={"data": {"id": TRAIN_ID}})
    httpx_mock.add_response(json={"data": job_row(status="cancelled", cancel_path=None)})
    assert (await jobs.cancel(TRAIN_ID)).status == "cancelled"
    assert str(httpx_mock.get_requests()[-2].url) == f"{BASE}{CANCEL}"

    httpx_mock.add_response(json={"data": job_row(type="batch_move", cancel_path=None)})
    with pytest.raises(ValidationError, match="cannot be cancelled"):
        await jobs.cancel(TRAIN_ID)
