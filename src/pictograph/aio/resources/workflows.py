"""Async Workflows resource - node-graph pipelines over an image / video / dataset.

Async twin of :class:`pictograph.resources.workflows.Workflows`.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast

from pictograph._http.pagination import AsyncOffsetPager
from pictograph.aio.resources import _resolve
from pictograph.exceptions import ApiError, PollTimeoutError
from pictograph.models.common import BulkActionResult, BulkDeleteResult
from pictograph.models.workflow import (
    Workflow,
    WorkflowRun,
    WorkflowRunCreated,
    WorkflowRunSummary,
    WorkflowStatus,
)
from pictograph.resources._base import AsyncResource

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

_API = "/api/v1/developer/workflows"
_DEFAULT_POLL_INTERVAL = 5.0
_DEFAULT_RUN_TIMEOUT = 3600.0  # 1 hour - matches the workflow runner's own timeout


class AsyncWorkflows(AsyncResource):
    """Manage node-graph workflows and their runs (async)."""

    async def list(self) -> Sequence[Workflow]:
        """List every workflow in your organization."""
        response = await self._transport.request("GET", f"{_API}/")
        return self._parse_list(Workflow, response.get("workflows", []))

    async def get(self, workflow: str) -> Workflow:
        """Fetch a single workflow by NAME (includes a ``validation`` list).

        A UUID is accepted too.
        """
        workflow_id = await _resolve.workflow_id(self._transport, workflow)
        response = await self._transport.request("GET", f"{_API}/{workflow_id}")
        return self._parse(Workflow, response["workflow"])

    async def create(
        self,
        name: str,
        graph: dict[str, Any],
        *,
        readme: str | None = None,
        description: str | None = None,
        template_key: str | None = None,
    ) -> Workflow:
        """Create a workflow from a graph (``{version, nodes, edges}``)."""
        body: dict[str, Any] = {"name": name, "graph": graph}
        if readme is not None:
            body["readme"] = readme
        if description is not None:
            body["description"] = description
        if template_key is not None:
            body["template_key"] = template_key
        response = await self._transport.request("POST", f"{_API}/", json=body)
        return self._parse(Workflow, response["workflow"])

    async def update(
        self,
        workflow: str,
        *,
        name: str | None = None,
        readme: str | None = None,
        description: str | None = None,
        graph: dict[str, Any] | None = None,
        status: WorkflowStatus | None = None,
    ) -> Workflow:
        """Update a workflow's name / description / graph / status."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if readme is not None:
            body["readme"] = readme
        if description is not None:
            body["description"] = description
        if graph is not None:
            body["graph"] = graph
        if status is not None:
            body["status"] = status
        workflow_id = await _resolve.workflow_id(self._transport, workflow)
        response = await self._transport.request("PATCH", f"{_API}/{workflow_id}", json=body)
        return self._parse(Workflow, response["workflow"])

    async def delete(self, workflow: str) -> None:
        """Delete a workflow, by NAME, and its run history."""
        workflow_id = await _resolve.workflow_id(self._transport, workflow)
        await self._transport.request("DELETE", f"{_API}/{workflow_id}")

    async def bulk_delete(self, workflows: Sequence[str]) -> BulkDeleteResult:
        """Delete many workflows in one atomic, org-scoped server-side call.

        Requires ``member``/``admin``/``owner``. Ids that don't resolve in your
        org land in :attr:`~pictograph.models.common.BulkDeleteResult.not_found`.

        Raises:
            ForbiddenError: Your API key role cannot manage workflows.
            ValidationError: ``workflows`` is empty.
        """
        response = await self._transport.request(
            "POST",
            f"{_API}/bulk-delete",
            json={"workflow_ids": await _resolve.workflow_ids(self._transport, workflows)},
        )
        return self._parse(BulkDeleteResult, response)

    async def run(self, workflow: str) -> WorkflowRunCreated:
        """Validate + start a run.

        Raises ``ValidationError`` (400) on an invalid graph, ``PaymentRequiredError``
        (402) on insufficient compute credit.
        """
        workflow_id = await _resolve.workflow_id(self._transport, workflow)
        response = await self._transport.request("POST", f"{_API}/{workflow_id}/run")
        return self._parse(WorkflowRunCreated, response)

    async def get_run(self, run_id: str) -> WorkflowRun:
        """Poll a run. When ``status == "completed"`` read ``artifacts`` (signed URLs)."""
        response = await self._transport.request("GET", f"{_API}/runs/{run_id}")
        return self._parse(WorkflowRun, response["run"])

    async def list_runs(
        self, workflow: str, *, limit: int = 50, offset: int = 0
    ) -> Sequence[WorkflowRunSummary]:
        """A single page of a workflow's runs, newest first (B494).

        Each item is a slim :class:`WorkflowRunSummary`; call :meth:`get_run` for a
        single run's full :class:`WorkflowRun` (including signed artifact downloads).

        Args:
            workflow: Workflow NAME (a UUID also works).
            limit: Page size, 1-100 (default 50).
            offset: Rows to skip (default 0). Past the last run returns an empty page.
        """
        workflow_id = await _resolve.workflow_id(self._transport, workflow)
        response = await self._transport.request(
            "GET", f"{_API}/{workflow_id}/runs", params={"limit": limit, "offset": offset}
        )
        return self._parse_list(WorkflowRunSummary, response.get("data", []))

    async def iter_runs(
        self, workflow: str, *, page_size: int = 50, max_total: int | None = None
    ) -> AsyncOffsetPager[WorkflowRunSummary]:
        """Auto-paging async iterator over every run of a workflow, newest first::

        async for run in client.workflows.iter_runs("Line counter"):
            print(run.id, run.status)
        """
        workflow_id = await _resolve.workflow_id(self._transport, workflow)

        async def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            return cast(
                "Mapping[str, Any]",
                await self._transport.request(
                    "GET",
                    f"{_API}/{workflow_id}/runs",
                    params={"offset": offset, "limit": limit},
                ),
            )

        return AsyncOffsetPager(
            fetch,
            items_key="data",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(WorkflowRunSummary, raw),
        )

    async def cancel_run(self, run_id: str) -> None:
        """Cancel an in-flight run (stops the GPU job). The run is FREE - workflows
        bill on success only, so a cancelled run is never charged (no refund needed)."""
        await self._transport.request("POST", f"{_API}/runs/{run_id}/cancel")

    async def bulk_cancel_runs(self, run_ids: Sequence[str]) -> BulkActionResult:
        """Cancel many in-flight workflow runs in one org-scoped server-side call.

        Idempotent: duplicate ids are collapsed; ids that don't resolve in your
        org OR are already terminal land in
        :attr:`~pictograph.models.common.BulkActionResult.not_found`. Requires member+.
        """
        response = await self._transport.request(
            "POST", f"{_API}/runs/bulk-cancel", json={"run_ids": list(run_ids)}
        )
        return self._parse(BulkActionResult, response)

    async def wait_for_run(
        self,
        run_id: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_RUN_TIMEOUT,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> WorkflowRun:
        """Poll a run until it reaches a terminal status, then return it.

        Returns the run on ``completed``; raises :class:`ApiError` if it ends
        ``error``/``cancelled`` and :class:`PollTimeoutError` on ``timeout``.

        Args:
            run_id: The run UUID from :meth:`run`.
            poll_interval: Seconds between checks (default 5s).
            timeout: Max seconds to wait (default 3600 = 1h).
            sleep: Injectable async sleep (tests pass a no-op).
        """
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else asyncio.sleep
        deadline = time.monotonic() + timeout
        while True:
            run = await self.get_run(run_id)
            if run.status == "completed":
                return run
            if run.status in ("error", "cancelled"):
                raise ApiError(
                    f"Workflow run {run_id} ended with status '{run.status}': "
                    f"{run.error or 'no error message provided'}",
                    response=run.model_dump(mode="json"),
                )
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Workflow run {run_id} did not complete within {timeout:.0f}s "
                    f"(last status: {run.status}). Fetch later via "
                    f"client.workflows.get_run(...)."
                )
            await sleep_fn(poll_interval)
