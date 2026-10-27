"""Async Workflows resource - node-graph pipelines over an image / video / dataset.

Async twin of :class:`pictograph.resources.workflows.Workflows`.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pictograph._http.pagination import AsyncOffsetPager
from pictograph.aio.resources import _resolve
from pictograph.exceptions import ApiError, PollTimeoutError, job_end_code
from pictograph.models.common import BulkActionResult, BulkDeleteResult
from pictograph.models.workflow import (
    RunLogOrder,
    RunLogPage,
    RunLogRecord,
    RunLogStream,
    RunLogSummary,
    Workflow,
    WorkflowCompute,
    WorkflowComputeOptions,
    WorkflowInvokeSource,
    WorkflowNodeCatalog,
    WorkflowRun,
    WorkflowRunCreated,
    WorkflowRunSummary,
    WorkflowStatus,
    WorkflowTemplate,
    WorkflowUploadInfo,
    WorkflowValidation,
)
from pictograph.resources._base import AsyncResource
from pictograph.resources._deployment_client import DEFAULT_TIMEOUT_S, DeploymentClient
from pictograph.resources.jobs import check_wait_args
from pictograph.resources.workflows import (
    Workflows,
    _compute_body,
    _content_type_for,
    _invoke_timeout,
    _local_source_kind,
    _log_params,
    _prepare_invoke_source,
    _run_compute,
    _sniff_media,
    endpoint_body,
    not_coming_live,
    scaling_body,
    with_token,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence

_API = "/api/v1/developer/workflows"
_DEFAULT_POLL_INTERVAL = 5.0
_DEFAULT_RUN_TIMEOUT = 3600.0  # 1 hour - matches the workflow runner's own timeout
_DEFAULT_LIVE_TIMEOUT = 900.0


class AsyncWorkflows(AsyncResource):
    """Manage node-graph workflows, their runs and their live endpoint (async)."""

    async def compute_options(self) -> WorkflowComputeOptions:
        """The compute a workflow may run with: each GPU with its memory and the price of a minute
        on it, each precision a Model block may take, the default. Async twin of
        :meth:`pictograph.resources.workflows.Workflows.compute_options`."""
        response = await self._transport.request("GET", f"{_API}/compute-options")
        return self._parse(WorkflowComputeOptions, response)

    async def nodes(self) -> WorkflowNodeCatalog:
        """Every block a workflow graph can hold: its ports, settings and rules. Async twin of
        :meth:`pictograph.resources.workflows.Workflows.nodes`."""
        response = await self._transport.request("GET", f"{_API}/nodes")
        return self._parse(WorkflowNodeCatalog, response)

    async def templates(self) -> Sequence[WorkflowTemplate]:
        """The use cases a workflow starts from, each with its graph. Async twin of
        :meth:`pictograph.resources.workflows.Workflows.templates`."""
        response = await self._transport.request("GET", f"{_API}/templates")
        return self._parse_list(WorkflowTemplate, response.get("templates", []))

    async def validate(self, graph: Mapping[str, Any]) -> WorkflowValidation:
        """Whether a run of ``graph`` would start, saving and starting nothing. Async twin of
        :meth:`pictograph.resources.workflows.Workflows.validate`."""
        response = await self._transport.request(
            "POST", f"{_API}/validate", json={"graph": dict(graph)}
        )
        return self._parse(WorkflowValidation, response)

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
        graph: dict[str, Any] | None = None,
        *,
        readme: str | None = None,
        description: str | None = None,
        template_key: str | None = None,
        compute: WorkflowCompute | Mapping[str, Any] | None = None,
    ) -> Workflow:
        """Create a workflow from a graph (``{version, nodes, edges}``), or from a template
        (``template_key`` without a graph), with what its runs run on (``compute``: ``{"gpu"}``;
        without one, a T4). Not served live until :meth:`deploy`. Async twin of
        :meth:`pictograph.resources.workflows.Workflows.create`.
        """
        body: dict[str, Any] = {"name": name}
        if graph is not None:
            body["graph"] = graph
        if compute is not None:
            body["compute"] = _compute_body(compute)
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
        compute: WorkflowCompute | Mapping[str, Any] | None = None,
    ) -> Workflow:
        """Update a workflow's name / description / graph / status / compute."""
        body: dict[str, Any] = {}
        if compute is not None:
            body["compute"] = _compute_body(compute)
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

    async def deploy(
        self,
        workflow: str,
        *,
        region: str = "auto",
        min_containers: int = 0,
        max_containers: int = 1,
        scaledown_window: int = 60,
    ) -> Workflow:
        """Serve a workflow live (async twin of :meth:`pictograph.resources.workflows.Workflows.deploy`).

        Args:
            workflow: Workflow name or id.
            region: Where it is served: ``"us-east"`` (Virginia), ``"us-west"`` (Oregon), or ``"auto"`` - the one
                nearest where this call is made. Fixed once it is first deployed.
            min_containers: Containers kept warm, 0-5. 0 scales to zero when no stream runs; more is billed
                continuously.
            max_containers: The most it runs under load, 1-10, four streams each.
            scaledown_window: Seconds a container stays after its last stream, 2-3600.
        """
        workflow_id = await _resolve.workflow_id(self._transport, workflow)
        body = endpoint_body(
            region=region,
            min_containers=min_containers,
            max_containers=max_containers,
            scaledown_window=scaledown_window,
        )
        response = await self._transport.request("POST", f"{_API}/{workflow_id}/deploy", json=body)
        return self._parse(Workflow, with_token(response))

    async def pause(self, workflow: str) -> Workflow:
        """Pause a workflow by NAME (async twin of :meth:`pictograph.resources.workflows.Workflows.pause`)."""
        workflow_id = await _resolve.workflow_id(self._transport, workflow)
        response = await self._transport.request("POST", f"{_API}/{workflow_id}/pause")
        return self._parse(Workflow, response["workflow"])

    async def resume(self, workflow: str) -> Workflow:
        """Resume a paused workflow by NAME (async twin of :meth:`pictograph.resources.workflows.Workflows.resume`)."""
        workflow_id = await _resolve.workflow_id(self._transport, workflow)
        response = await self._transport.request("POST", f"{_API}/{workflow_id}/resume")
        return self._parse(Workflow, response["workflow"])

    async def update_endpoint(
        self,
        workflow: str,
        *,
        min_containers: int | None = None,
        max_containers: int | None = None,
        scaledown_window: int | None = None,
    ) -> Workflow:
        """Change a workflow's endpoint's autoscaling (async twin of
        :meth:`pictograph.resources.workflows.Workflows.update_endpoint`).

        Args:
            workflow: Workflow name or id.
            min_containers: Containers kept warm, 0-5 (0 scales to zero).
            max_containers: The most it runs under load, 1-10, four streams each.
            scaledown_window: Seconds a container stays after its last stream, 2-3600.
        """
        workflow_id = await _resolve.workflow_id(self._transport, workflow)
        response = await self._transport.request(
            "PATCH",
            f"{_API}/{workflow_id}/endpoint",
            json=scaling_body(min_containers, max_containers, scaledown_window),
        )
        return self._parse(Workflow, response["workflow"])

    async def wait_until_live(
        self,
        workflow: str | Workflow,
        *,
        timeout: float = _DEFAULT_LIVE_TIMEOUT,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> Workflow:
        """Poll a workflow until its endpoint is ``active`` (async twin of
        :meth:`pictograph.resources.workflows.Workflows.wait_until_live`)."""
        check_wait_args(poll_interval, timeout)
        workflow_id = (
            workflow.id
            if isinstance(workflow, Workflow)
            else await _resolve.workflow_id(self._transport, workflow)
        )
        sleep_fn = sleep if sleep is not None else asyncio.sleep
        deadline = time.monotonic() + timeout
        while True:
            response = await self._transport.request("GET", f"{_API}/{workflow_id}")
            current = self._parse(Workflow, response["workflow"])
            if current.endpoint is not None and current.endpoint.status == "active":
                return current
            ended = not_coming_live(current)
            if ended is not None:
                raise ended
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Workflow '{current.name}' was not live within {timeout:.0f}s (its endpoint: "
                    f"{current.endpoint.status if current.endpoint else 'none'}). Wait again with "
                    "client.workflows.wait_until_live(...)."
                )
            await sleep_fn(poll_interval)

    @staticmethod
    def connect(
        workflow: Workflow, token: str, *, timeout: float = DEFAULT_TIMEOUT_S
    ) -> DeploymentClient[Any]:
        """A direct (sync) client of a workflow's endpoint - its streams - with its endpoint token (see
        :meth:`pictograph.resources.workflows.Workflows.connect`).

        Args:
            workflow: The workflow, as ``get``, ``create`` or ``wait_until_live`` answer it: its ``endpoint`` names where its streams open.
            token: Its endpoint's ``pk_deploy_`` token (``endpoint_token`` from ``deploy``, or revealed in the app).
            timeout: Seconds a request to its endpoint may take.
        """
        return Workflows.connect(workflow, token, timeout=timeout)

    async def live_log(
        self,
        workflow: str,
        *,
        stream: RunLogStream = "frames",
        limit: int = 50,
        cursor: str | None = None,
        order: RunLogOrder = "desc",
        full: bool = False,
        class_name: str | None = None,
        track_id: str | int | None = None,
        min_confidence: float | None = None,
        source: str | None = None,
        event_type: str | None = None,
        trigger: str | None = None,
    ) -> RunLogPage:
        """One page of a workflow's live log (async twin of
        :meth:`pictograph.resources.workflows.Workflows.live_log`).

        Args:
            workflow: Workflow name or id.
            stream: ``"frames"`` (default), ``"events"`` or ``"stats"``.
            limit: Records per page, 1-200 (default 50).
            cursor: The previous page's ``next_cursor`` (None: the first page).
            order: ``"desc"`` (newest first, default) or ``"asc"`` (oldest first). A cursor pages only the order
                it was made for.
            full: Also each frame's whole Pictograph record (``pictograph``) / each event's and stats' whole
                message (``message``).
            class_name: Frames and events with an annotation of this class.
            track_id: Frames and events with this track.
            min_confidence: Frames whose best annotation is at least this confident.
            source: One stream's records (each stream logs as its own source).
            event_type: Events of this type (``"event"``, ``"capture"``, ``"capture_uploaded"``).
            trigger: Events of this trigger.
        """
        workflow_id = await _resolve.workflow_id(self._transport, workflow)
        params = _log_params(
            stream=stream,
            limit=limit,
            cursor=cursor,
            order=order,
            full=full,
            class_name=class_name,
            track_id=track_id,
            min_confidence=min_confidence,
            source=source,
            event_type=event_type,
            trigger=trigger,
        )
        response = await self._transport.request(
            "GET", f"{_API}/{workflow_id}/endpoint/log", params=params
        )
        return self._parse(RunLogPage, response)

    async def iter_live_log(
        self,
        workflow: str,
        *,
        stream: RunLogStream = "frames",
        page_size: int = 200,
        order: RunLogOrder = "desc",
        full: bool = False,
        max_total: int | None = None,
    ) -> AsyncIterator[RunLogRecord]:
        """Every record of a workflow's live log stream, page by page (async).

        Args:
            workflow: Workflow name or id.
            stream: ``"frames"`` (default), ``"events"`` or ``"stats"``.
            page_size: Records fetched per request, 1-200.
            order: ``"desc"`` (newest first, default) or ``"asc"``.
            full: Also each record's whole Pictograph record or message.
            max_total: Stop after this many records. ``None`` walks the whole stream.
        """
        cursor: str | None = None
        seen = 0
        while True:
            page = await self.live_log(
                workflow, stream=stream, limit=page_size, cursor=cursor, order=order, full=full
            )
            for record in page.records:
                yield record
                seen += 1
                if max_total is not None and seen >= max_total:
                    return
            if not page.next_cursor:
                return
            cursor = page.next_cursor

    async def live_log_summary(self, workflow: str) -> RunLogSummary:
        """What a workflow's live streams logged (async)."""
        workflow_id = await _resolve.workflow_id(self._transport, workflow)
        response = await self._transport.request(
            "GET", f"{_API}/{workflow_id}/endpoint/log/summary"
        )
        return self._parse(RunLogSummary, response)

    async def delete(self, workflow: str) -> None:
        """Delete a workflow, by NAME, and its run history (its live endpoint is stopped first)."""
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

    async def run(self, workflow: str, *, gpu: str | None = None) -> WorkflowRunCreated:
        """Validate + start a run, optionally on this run's own ``gpu``.

        Raises ``ValidationError`` (400) on an invalid graph or compute, ``PaymentRequiredError``
        (402) on insufficient compute credit.
        """
        workflow_id = await _resolve.workflow_id(self._transport, workflow)
        compute = _run_compute(gpu)
        if compute is None:
            response = await self._transport.request("POST", f"{_API}/{workflow_id}/run")
        else:
            response = await self._transport.request(
                "POST", f"{_API}/{workflow_id}/run", json={"compute": compute}
            )
        return self._parse(WorkflowRunCreated, response)

    async def upload_source(
        self, local_path: str | Path, *, content_type: str | None = None
    ) -> str:
        """Upload a local video/image as a workflow source; returns its ``gcs_uri``.

        Async twin of :meth:`pictograph.resources.workflows.Workflows.upload_source` -
        streams the file straight to object storage (signed PUT, no API hop) under your
        org's prefix. :meth:`invoke` calls this for you when given a local path.
        """
        path = Path(local_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Source file not found: {path}")
        ct = content_type or _content_type_for(path)
        info = self._parse(
            WorkflowUploadInfo,
            await self._transport.request(
                "POST", f"{_API}/upload-url", json={"filename": path.name, "content_type": ct}
            ),
        )
        await self._transport.upload_external(info.upload_url, path, content_type=ct)
        return info.gcs_uri

    async def _upload_data(self, data: bytes, *, filename: str, content_type: str) -> str:
        """:meth:`upload_source` for bytes already in memory (stdin, a buffer)."""
        info = self._parse(
            WorkflowUploadInfo,
            await self._transport.request(
                "POST",
                f"{_API}/upload-url",
                json={"filename": filename, "content_type": content_type},
            ),
        )
        await self._transport.upload_external(info.upload_url, data, content_type=content_type)
        return info.gcs_uri

    async def invoke(
        self,
        workflow: str,
        *,
        source: WorkflowInvokeSource | Mapping[str, Any] | str | Path | bytes,
        wait: bool = False,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_RUN_TIMEOUT,
        gpu: str | None = None,
    ) -> WorkflowRunCreated | WorkflowRun:
        """Run a SAVED workflow on-demand over a source given AT CALL TIME (WF-D).

        The async twin of :meth:`pictograph.resources.workflows.Workflows.invoke` -
        the workflow-DAG equivalent of a model deployment's ``/predict`` endpoint. See
        that method for the ``source`` modes (a local file path, an ``http(s)`` link
        Pictograph downloads, bytes, a dataset or a ``gcs_uri``); charge-on-success (a
        failed/cancelled run is free). ``wait=True`` polls client-side to a terminal
        :class:`WorkflowRun`.
        """
        # Everything that can be refused locally is, before the first request.
        prepared = _prepare_invoke_source(source)
        sniffed = _sniff_media(prepared) if isinstance(prepared, bytes) else None
        local_kind = _local_source_kind(prepared) if isinstance(prepared, Path) else None
        workflow_id = await _resolve.workflow_id(self._transport, workflow)
        if isinstance(prepared, Path) and local_kind is not None:
            src = WorkflowInvokeSource(kind=local_kind, gcs_uri=await self.upload_source(prepared))
        elif isinstance(prepared, bytes) and sniffed is not None:
            kind, filename, content_type = sniffed
            src = WorkflowInvokeSource(
                kind=kind,
                gcs_uri=await self._upload_data(
                    prepared, filename=filename, content_type=content_type
                ),
            )
        else:
            src = cast("WorkflowInvokeSource", prepared)
        body: dict[str, Any] = {"source": src.model_dump(exclude_none=True)}
        compute = _run_compute(gpu)
        if compute is not None:
            body["compute"] = compute
        response = await self._transport.request(
            "POST", f"{_API}/{workflow_id}/invoke", json=body, timeout=_invoke_timeout(src)
        )
        created = self._parse(WorkflowRunCreated, response)
        if not wait:
            return created
        return await self.wait_for_run(created.run_id, poll_interval=poll_interval, timeout=timeout)

    async def get_run(self, run_id: str) -> WorkflowRun:
        """Poll a run. When ``status == "completed"`` read ``artifacts`` (signed URLs)."""
        response = await self._transport.request("GET", f"{_API}/runs/{run_id}")
        return self._parse(WorkflowRun, response["run"])

    async def run_log(
        self,
        run_id: str,
        *,
        stream: RunLogStream = "events",
        limit: int = 50,
        cursor: str | None = None,
        order: RunLogOrder = "asc",
        full: bool = False,
        class_name: str | None = None,
        track_id: str | int | None = None,
        min_confidence: float | None = None,
        source: str | None = None,
        event_type: str | None = None,
        trigger: str | None = None,
    ) -> RunLogPage:
        """One page of a run's log (sync twin: :meth:`pictograph.resources.workflows.Workflows.run_log`)."""
        params = _log_params(
            stream=stream,
            limit=limit,
            cursor=cursor,
            order=order,
            full=full,
            class_name=class_name,
            track_id=track_id,
            min_confidence=min_confidence,
            source=source,
            event_type=event_type,
            trigger=trigger,
        )
        response = await self._transport.request("GET", f"{_API}/runs/{run_id}/log", params=params)
        return self._parse(RunLogPage, response)

    async def iter_run_log(
        self,
        run_id: str,
        *,
        stream: RunLogStream = "events",
        page_size: int = 200,
        order: RunLogOrder = "asc",
        full: bool = False,
        max_total: int | None = None,
        class_name: str | None = None,
        track_id: str | int | None = None,
        min_confidence: float | None = None,
        source: str | None = None,
        event_type: str | None = None,
        trigger: str | None = None,
    ) -> AsyncIterator[RunLogRecord]:
        """Every record of a run's log stream, page by page::

        async for event in client.workflows.iter_run_log(run_id):
            print(event.ts, event.trigger)
        """
        cursor: str | None = None
        seen = 0
        while True:
            page = await self.run_log(
                run_id,
                stream=stream,
                limit=page_size,
                cursor=cursor,
                order=order,
                full=full,
                class_name=class_name,
                track_id=track_id,
                min_confidence=min_confidence,
                source=source,
                event_type=event_type,
                trigger=trigger,
            )
            for record in page.records:
                yield record
                seen += 1
                if max_total is not None and seen >= max_total:
                    return
            if not page.next_cursor:
                return
            cursor = page.next_cursor

    async def run_log_summary(self, run_id: str) -> RunLogSummary:
        """What a run logged, per stream (sync twin: :meth:`pictograph.resources.workflows.Workflows.run_log_summary`)."""
        response = await self._transport.request("GET", f"{_API}/runs/{run_id}/log/summary")
        return self._parse(RunLogSummary, response)

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
                    code=job_end_code(run.status),
                )
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Workflow run {run_id} did not complete within {timeout:.0f}s "
                    f"(last status: {run.status}). Fetch later via "
                    f"client.workflows.get_run(...)."
                )
            await sleep_fn(poll_interval)
