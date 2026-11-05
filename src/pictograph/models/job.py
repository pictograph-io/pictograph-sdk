"""The job model - one view of any long-running work in an organization (B533 #14).

Training runs, batch auto-annotation, workflow runs, model evaluations, exports and the
background jobs (imports, video extraction, bulk image operations, ...) each have their
own status call. :class:`Job` is how every one of them reads in
:meth:`pictograph.resources.jobs.Jobs.list` / ``get``: one status vocabulary, progress
0-100, and ``cancel_path`` - the developer route that cancels it, when one does.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

#: The status every job reads in, whatever its kind: ``queued`` and ``running`` are
#: active, the rest are finished.
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]

#: What :meth:`Jobs.list` filters by: ``active`` (queued or running, the default),
#: ``finished`` (completed, failed or cancelled), ``all``, or one :data:`JobStatus`.
JobStatusFilter = Literal[
    "active", "finished", "all", "queued", "running", "completed", "failed", "cancelled"
]

ACTIVE_STATUSES: frozenset[str] = frozenset({"queued", "running"})


class Job(BaseModel):
    """One job, whatever kind of work it is.

    ``type`` is ``training``, ``auto_annotate``, ``workflow_run``, ``evaluation``,
    ``export``, or a background job's own type (``connector_import``,
    ``public_dataset_import``, ``video_extract``, ``batch_move``, ...). ``detail`` holds
    what only that kind has (a training run's epochs, a workflow run's frames, a
    background job's item counts). ``cancel_path`` is the developer route that cancels
    the job, or None when it is finished or its kind cannot be cancelled through the API.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    type: str
    # A str, not JobStatus: a kind of job added later may report a word this SDK does
    # not know yet, and reading it must not fail.
    status: str
    progress: float | None = None
    name: str | None = None
    dataset_id: str | None = None
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    cancel_path: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)

    @property
    def active(self) -> bool:
        """True while the job is queued or running."""
        return self.status in ACTIVE_STATUSES

    @property
    def cancellable(self) -> bool:
        """True when :meth:`Jobs.cancel` can stop this job."""
        return self.cancel_path is not None


__all__ = ["ACTIVE_STATUSES", "Job", "JobStatus", "JobStatusFilter"]
