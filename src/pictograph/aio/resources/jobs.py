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

    async def get(self, job: str) -> Job:
        """One job, by its id (see the sync twin)."""
        response = await self._transport.request("GET", job_path(job))
        return self._parse(Job, response["data"])

    async def wait(
        self,
        job: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> Job:
        """Poll a current until it finishes (see the sync twin)."""
        check_wait_args(poll_interval, timeout)
        sleep_fn = sleep if sleep is not None else asyncio.sleep
        deadline = time.monotonic() + timeout
        while True:
            current = await self.get(job)
            if current.status == "completed":
                return current
            if not current.active:
                raise ended_badly(current)
            if time.monotonic() >= deadline:
                raise still_running(current, timeout)
            await sleep_fn(poll_interval)

    async def cancel(self, job: str) -> Job:
        """Cancel a queued or running current (see the sync twin)."""
        current = await self.get(job)
        await self._transport.request("POST", cancel_path_of(current))
        return await self.get(current.id)


__all__ = ["AsyncJobs"]
