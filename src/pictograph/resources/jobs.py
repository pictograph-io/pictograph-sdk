"""Jobs resource - one view of every job your organization is running (B533 #14).

Training, batch auto-annotation, workflow runs, model evaluations, exports and the
background jobs (imports, video extraction, bulk image operations, scans, ...) each have
their own status call. ``client.jobs`` reads them all::

    for job in client.jobs.list():  # queued or running, newest first
        print(job.type, job.status, job.progress, job.name)

    job = client.jobs.wait(job_id)  # until it completes; raises if it failed
    client.jobs.cancel(job_id)  # through the route that cancels that kind

``status`` narrows the list to ``active`` (the default), ``finished``, ``all`` or one
status; ``type`` to one kind of job. Finished background jobs are kept 30 days; training
runs, workflow runs, batch auto-annotation and evaluations keep their history.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote

from pictograph._http.pagination import OffsetPager
from pictograph.exceptions import ApiError, PollTimeoutError, ValidationError, job_end_code
from pictograph.models.job import Job, JobStatusFilter
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

_API_PATH = "/api/v1/developer/jobs"
_DEVELOPER_PREFIX = "/api/v1/developer/"
_DEFAULT_POLL_INTERVAL = 3.0
_DEFAULT_TIMEOUT = 3600.0


def list_params(
    status: JobStatusFilter, job_type: str | None, offset: int, limit: int
) -> dict[str, Any]:
    """The query of one page of :meth:`Jobs.list` (shared with the async twin)."""
    params: dict[str, Any] = {"status": status, "limit": limit, "offset": offset}
    if job_type is not None:
        params["type"] = job_type
    return params


def job_path(job_id: str) -> str:
    return f"{_API_PATH}/{quote(job_id, safe='')}"


def cancel_path_of(job: Job) -> str:
    """The route that cancels ``job``, or a :class:`ValidationError` saying why there is none.

    The path comes from the API (each kind of job is cancelled by its own route, with its
    own rules); it is used only when it is a developer-API path.
    """
    path = job.cancel_path
    if path is None:
        if not job.active:
            raise ValidationError(
                f"Job {job.id} is already {job.status}; there is nothing to cancel."
            )
        raise ValidationError(
            f"A {job.type} job cannot be cancelled through the API; it stops on its own."
        )
    if not path.startswith(_DEVELOPER_PREFIX):
        raise ValidationError(
            f"Job {job.id} names a cancel route outside the developer API: {path}"
        )
    return path


def ended_badly(job: Job) -> ApiError:
    """The error :meth:`Jobs.wait` raises when the job failed or was cancelled."""
    return ApiError(
        f"{job.type} job '{job.id}' ended with status '{job.status}': "
        f"{job.error or 'no error message provided'}",
        response=job.model_dump(mode="json"),
        code=job_end_code(job.status),
    )


def still_running(job: Job, timeout: float) -> PollTimeoutError:
    progress = "" if job.progress is None else f", {job.progress:.0f}%"
    return PollTimeoutError(
        f"{job.type} job '{job.id}' did not finish within {timeout:.0f}s (last status: "
        f"{job.status}{progress}). It is still running - wait again with client.jobs.wait()."
    )


def check_wait_args(poll_interval: float, timeout: float) -> None:
    if poll_interval <= 0:
        raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
    if timeout <= 0:
        raise ValueError(f"timeout must be > 0, got {timeout}")


class Jobs(Resource):
    """List, read, wait for and cancel any job in your organization."""

    def list(
        self,
        *,
        status: JobStatusFilter = "active",
        type: str | None = None,
        page_size: int = 50,
        max_total: int | None = None,
    ) -> OffsetPager[Job]:
        """Your organization's jobs, newest first, across every kind of work.

        Args:
            status: ``active`` (queued or running, the default), ``finished``
                (completed, failed or cancelled), ``all``, or one status:
                ``queued``, ``running``, ``completed``, ``failed``, ``cancelled``.
            type: Only jobs of this kind: ``training``, ``auto_annotate``,
                ``workflow_run``, ``evaluation``, ``export``, or a background job's
                type (``connector_import``, ``public_dataset_import``,
                ``video_extract``, ``batch_move``, ...).
            page_size: Jobs per request (at most 100).
            max_total: Stop after this many jobs. The API reads at most the newest
                1,000 of a filter; narrow with ``status`` or ``type`` past that.

        Returns:
            An auto-paging iterator of :class:`~pictograph.models.job.Job`.
        """

        def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            return cast(
                "Mapping[str, Any]",
                self._transport.request(
                    "GET", _API_PATH, params=list_params(status, type, offset, limit)
                ),
            )

        return OffsetPager(
            fetch,
            items_key="data",
            page_size=min(page_size, 100),
            max_total=max_total,
            parse_item=lambda raw: self._parse(Job, raw),
        )

    def get(self, job_id: str) -> Job:
        """One job, by the id the work answered when it started (a training run's,
        an import's, a workflow run's, ...). Another organization's job, like an id that
        does not exist, raises :class:`~pictograph.exceptions.NotFoundError`."""
        response = self._transport.request("GET", job_path(job_id))
        return self._parse(Job, response["data"])

    def wait(
        self,
        job_id: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        sleep: Callable[[float], None] | None = None,
    ) -> Job:
        """Poll a job until it finishes.

        Returns:
            The :class:`~pictograph.models.job.Job`, ``completed``.

        Raises:
            ApiError: the job failed (``code="job_failed"``) or was cancelled
                (``code="job_cancelled"``); the job is in ``.response``.
            PollTimeoutError: ``timeout`` elapsed first. The job keeps running.
        """
        check_wait_args(poll_interval, timeout)
        sleep_fn = sleep if sleep is not None else time.sleep
        deadline = time.monotonic() + timeout
        while True:
            job = self.get(job_id)
            if job.status == "completed":
                return job
            if not job.active:
                raise ended_badly(job)
            if time.monotonic() >= deadline:
                raise still_running(job, timeout)
            sleep_fn(poll_interval)

    def cancel(self, job_id: str) -> Job:
        """Cancel a queued or running job through the route that cancels its kind.

        A training run is refunded what it did not use and a workflow run is free, as
        when it is cancelled from its own resource. Returns the job as it reads after
        the cancel.

        Raises:
            ValidationError: the job is finished, or its kind cannot be cancelled
                through the API (``Job.cancel_path`` is None).
        """
        job = self.get(job_id)
        self._transport.request("POST", cancel_path_of(job))
        return self.get(job.id)


__all__ = ["Jobs"]
