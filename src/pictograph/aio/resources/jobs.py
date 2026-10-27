"""Async jobs resource (B533 #14).

Async twin of :class:`pictograph.resources.jobs.Jobs`; see that module for the details.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast

from pictograph._http.pagination import AsyncOffsetPager
from pictograph.models.job import Job, JobStatusFilter
from pictograph.resources._base import AsyncResource
from pictograph.resources.jobs import (
    _API_PATH,
    _DEFAULT_POLL_INTERVAL,
    _DEFAULT_TIMEOUT,
    cancel_path_of,
    check_wait_args,
    ended_badly,
    job_path,
    list_params,
    still_running,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping


class AsyncJobs(AsyncResource):
    """List, read, wait for and cancel any job in your organization (async)."""

    def list(
        self,
        *,
        status: JobStatusFilter = "active",
        type: str | None = None,
        page_size: int = 50,
        max_total: int | None = None,
    ) -> AsyncOffsetPager[Job]:
        """Your organization's jobs, newest first (see the sync twin)."""

        async def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            return cast(
                "Mapping[str, Any]",
                await self._transport.request(
                    "GET", _API_PATH, params=list_params(status, type, offset, limit)
                ),
            )

        return AsyncOffsetPager(
            fetch,
            items_key="data",
            page_size=min(page_size, 100),
            max_total=max_total,
            parse_item=lambda raw: self._parse(Job, raw),
        )

    async def get(self, job_id: str) -> Job:
        """One job, by its id (see the sync twin)."""
        response = await self._transport.request("GET", job_path(job_id))
        return self._parse(Job, response["data"])

    async def wait(
        self,
        job_id: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> Job:
        """Poll a job until it finishes (see the sync twin)."""
        check_wait_args(poll_interval, timeout)
        sleep_fn = sleep if sleep is not None else asyncio.sleep
        deadline = time.monotonic() + timeout
        while True:
            job = await self.get(job_id)
            if job.status == "completed":
                return job
            if not job.active:
                raise ended_badly(job)
            if time.monotonic() >= deadline:
                raise still_running(job, timeout)
            await sleep_fn(poll_interval)

    async def cancel(self, job_id: str) -> Job:
        """Cancel a queued or running job (see the sync twin)."""
        job = await self.get(job_id)
        await self._transport.request("POST", cancel_path_of(job))
        return await self.get(job.id)


__all__ = ["AsyncJobs"]
