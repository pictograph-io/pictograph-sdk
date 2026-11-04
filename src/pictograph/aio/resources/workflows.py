"""Async Workflows resource - node-graph pipelines run on any input (an image, a video, a dataset, an export) or live.

Async twin of :class:`pictograph.resources.workflows.Workflows`.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, overload

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
    WorkflowNodeCatalog,
    WorkflowRun,
    WorkflowRunCreated,
    WorkflowRunInput,
    WorkflowRunsKind,
    WorkflowRunSummary,
    WorkflowStatus,
    WorkflowTemplate,
    WorkflowUploadInfo,
    WorkflowValidation,
)
from pictograph.resources._base import AsyncResource
from pictograph.resources._deployment_client import DEFAULT_TIMEOUT_S, DeploymentClient
from pictograph.resources._outputs import RunOutputs, outputs_body
from pictograph.resources.jobs import check_wait_args
from pictograph.resources.workflows import (
    RunSource,
    Workflows,
    _compute_body,
    _content_type_for,
    _live_view_fields,
    _local_input,
    _local_source_kind,
    _log_params,
    _prepare_run_input,
    _prepared_kind,
    _run_timeout,
    _runs_params,
    _sniff_media,
    _with_rate,
    endpoint_body,
    not_coming_live,
    scaling_body,
    watchable_session,
    with_token,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence

    from pictograph.resources._deployment_stream import DeploymentStream

_API = "/api/v1/developer/workflows"
_DEFAULT_POLL_INTERVAL = 5.0
_DEFAULT_RUN_TIMEOUT = 3600.0  # 1 hour - matches the workflow runner's own timeout
_DEFAULT_LIVE_TIMEOUT = 900.0
_DEFAULT_WATCH_TIMEOUT = 900.0
_DEFAULT_WATCH_POLL = 2.0


class AsyncWorkflows(AsyncResource):
    """Manage node-graph workflows, their runs and their live endpoint (async)."""

    async def compute_options(self, region: str | None = None) -> WorkflowComputeOptions:
        """The compute a workflow may run with, priced in ``region`` (None: the one nearest you): each
        GPU with its memory and its endpoint's container a minute and an hour there, each precision a
        Model block may take, the default and every region. Async twin of
        :meth:`pictograph.resources.workflows.Workflows.compute_options`."""
        params = {"region": region} if region else None
        response = await self._transport.request("GET", f"{_API}/compute-options", params=params)
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

    async def validate(
        self,
        graph: Mapping[str, Any],
        *,
        source: WorkflowRunInput | Mapping[str, Any] | None = None,
    ) -> WorkflowValidation:
        """Whether a run of ``graph`` would start (on ``source``, the input a run would name), saving and starting
        nothing. Async twin of :meth:`pictograph.resources.workflows.Workflows.validate`."""
        body: dict[str, Any] = {"graph": dict(graph)}
        if source is not None:
            src = (
                source
                if isinstance(source, WorkflowRunInput)
                else WorkflowRunInput.model_validate(dict(source))
            )
            body["source"] = src.model_dump(exclude_none=True)
        response = await self._transport.request("POST", f"{_API}/validate", json=body)
        return self._parse(WorkflowValidation, response)

    async def list(self) -> Sequence[Workflow]:
        """List every workflow in your organization."""
        response = await self._transport.request("GET", f"{_API}/")
        return self._parse_list(Workflow, response.get("workflows", []))

    async def get(self, workflow: str) -> Workflow:
        """Fetch one workflow, by name or id (includes a ``validation`` list)."""
        workflow_id = await _resolve.workflow_id(self._transport, workflow)
        response = await self._transport.request("GET", f"{_API}/{workflow_id}")
        return self._parse(Workflow, response["workflow"])

    async def create(
        self,
        name: str,
        *,
        graph: dict[str, Any] | None = None,
        readme: str | None = None,
        description: str | None = None,
        template_key: str | None = None,
        compute: WorkflowCompute | Mapping[str, Any] | None = None,
        region: str = "auto",
        min_containers: int = 0,
        max_containers: int = 1,
        scaledown_window: int = 60,
    ) -> Workflow:
        """Create a workflow - and its endpoint with it, where its runs and streams run - from a graph
        (``{version, nodes, edges}``), or from a template (``template_key`` without a graph), its
        endpoint on ``compute``'s GPU (``{"gpu"}``; without one, a T4) in ``region`` with its
        autoscaling. Its token is on the answer once (``endpoint_token``). Async twin of
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
        body.update(
            endpoint_body(
                region=region,
                min_containers=min_containers,
                max_containers=max_containers,
                scaledown_window=scaledown_window,
            )
        )
        response = await self._transport.request("POST", f"{_API}/", json=body)
        return self._parse(Workflow, with_token(response))

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
        """Deploy a workflow's endpoint: one for a workflow made before endpoints were made with workflows, or a failed
        or draft one again (async twin of :meth:`pictograph.resources.workflows.Workflows.deploy`).

        Args:
            workflow: Workflow name or id.
            region: Where it is served: ``"us-east"`` (Virginia), ``"us-west"`` (Oregon), or ``"auto"`` - the one
                nearest where this call is made. Fixed once it is first deployed.
            min_containers: Containers kept warm, 0-5. 0 scales to zero when no stream runs; more is billed
                continuously.
            max_containers: The most it runs under load, 1-10, four streams each.
            scaledown_window: Seconds a container stays after its last run or stream, 2-3600.
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
        """Pause a workflow (async twin of :meth:`pictograph.resources.workflows.Workflows.pause`)."""
        workflow_id = await _resolve.workflow_id(self._transport, workflow)
        response = await self._transport.request("POST", f"{_API}/{workflow_id}/pause")
        return self._parse(Workflow, response["workflow"])

    async def resume(self, workflow: str) -> Workflow:
        """Resume a paused workflow (async twin of :meth:`pictograph.resources.workflows.Workflows.resume`)."""
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
            scaledown_window: Seconds a container stays after its last run or stream, 2-3600.
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
        workflow: Workflow, *, token: str, timeout: float = DEFAULT_TIMEOUT_S
    ) -> DeploymentClient[Any]:
        """A direct (sync) client of a workflow's endpoint - its streams - with its endpoint token (see
        :meth:`pictograph.resources.workflows.Workflows.connect`).

        Args:
            workflow: The workflow, as ``get``, ``create`` or ``wait_until_live`` answer it: its ``endpoint`` names where its streams open.
            token: Its endpoint's ``pk_deploy_`` token (``endpoint_token`` from ``create``, or revealed in the app).
            timeout: Seconds a request to its endpoint may take.
        """
        return Workflows.connect(workflow, token=token, timeout=timeout)

    async def watch_run(
        self,
        run: str,
        *,
        token: str,
        timeout: float = _DEFAULT_WATCH_TIMEOUT,
        poll_interval: float = _DEFAULT_WATCH_POLL,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> DeploymentStream[Any]:
        """Watch a file run's live view as it runs (async twin of
        :meth:`pictograph.resources.workflows.Workflows.watch_run`): waits for its first frames, then answers the
        (sync) stream joined to its live view - its ``frames()``, ``whep_url``, ``set_stream`` / ``set_overlay``;
        ``stop()`` only stops watching."""
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else asyncio.sleep
        deadline = time.monotonic() + timeout
        while True:
            current = await self.get_run(run)
            session = watchable_session(current)
            if session is not None:
                break
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Workflow run {run}'s live view was not up within {timeout:.0f}s (its status: "
                    f"{current.status}; {current.dispatch_note or 'no frames yet'}). Watch it again with "
                    "client.workflows.watch_run(...)."
                )
            await sleep_fn(poll_interval)
        connection = self.connect(await self.get(current.workflow_id), token=token)
        try:
            live = await asyncio.to_thread(connection.join_stream, session)
        except BaseException:
            connection.close()
            raise
        live._owns(connection.close)
        return live

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
        track: str | int | None = None,
        branch: bool | None = None,
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
            track: Frames and events with this track.
            branch: Frames and events a Condition block routed down this branch (``True`` / ``False``).
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
            track_id=track,
            branch=branch,
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
        class_name: str | None = None,
        track: str | int | None = None,
        branch: bool | None = None,
        min_confidence: float | None = None,
        source: str | None = None,
        event_type: str | None = None,
        trigger: str | None = None,
    ) -> AsyncIterator[RunLogRecord]:
        """Every record of a workflow's live log stream, page by page (async; :meth:`live_log`'s filters too).

        Args:
            workflow: Workflow name or id.
            stream: ``"frames"`` (default), ``"events"`` or ``"stats"``.
            page_size: Records fetched per request, 1-200.
            order: ``"desc"`` (newest first, default) or ``"asc"``.
            full: Also each record's whole Pictograph record or message.
            max_total: Stop after this many records. ``None`` walks the whole stream.
            class_name, track, branch, min_confidence, source, event_type, trigger: :meth:`live_log`'s filters.
        """
        cursor: str | None = None
        seen = 0
        while True:
            page = await self.live_log(
                workflow,
                stream=stream,
                limit=page_size,
                cursor=cursor,
                order=order,
                full=full,
                class_name=class_name,
                track=track,
                branch=branch,
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

    async def live_log_summary(self, workflow: str) -> RunLogSummary:
        """What a workflow's live streams logged (async)."""
        workflow_id = await _resolve.workflow_id(self._transport, workflow)
        response = await self._transport.request(
            "GET", f"{_API}/{workflow_id}/endpoint/log/summary"
        )
        return self._parse(RunLogSummary, response)

    async def delete(self, workflow: str) -> None:
        """Delete a workflow and its run history (its live endpoint is stopped first)."""
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
            json={"workflows": await _resolve.workflow_ids(self._transport, workflows)},
        )
        return self._parse(BulkDeleteResult, response)

    @overload
    async def run(
        self,
        workflow: str,
        *,
        source: RunSource,
        sample_fps: float | None = ...,
        outputs: RunOutputs | None = ...,
        stream: bool | None = ...,
        overlay: bool | None = ...,
        wait: Literal[False] = ...,
        poll_interval: float = ...,
        timeout: float = ...,
    ) -> WorkflowRunCreated: ...

    @overload
    async def run(
        self,
        workflow: str,
        *,
        source: RunSource,
        sample_fps: float | None = ...,
        outputs: RunOutputs | None = ...,
        stream: bool | None = ...,
        overlay: bool | None = ...,
        wait: Literal[True],
        poll_interval: float = ...,
        timeout: float = ...,
    ) -> WorkflowRun: ...

    async def run(
        self,
        workflow: str,
        *,
        source: RunSource,
        sample_fps: float | None = None,
        outputs: RunOutputs | None = None,
        stream: bool | None = None,
        overlay: bool | None = None,
        wait: bool = False,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_RUN_TIMEOUT,
    ) -> WorkflowRunCreated | WorkflowRun:
        """Run a workflow on an input: a video or an image, a dataset, or an export.

        Async twin of :meth:`pictograph.resources.workflows.Workflows.run` - see it for the ``source`` forms (a local
        file's path, an ``http(s)`` link Pictograph downloads, bytes, a dataset, an export, a stored file's handle),
        ``sample_fps`` (a video's frames read a second), ``outputs`` (what it writes beside its log), ``stream`` /
        ``overlay`` (its live view's Stream and Overlay as it starts: a video's, a dataset's or an export's run) and
        what it raises. It runs on the workflow's endpoint, billed as its uptime; ``wait=True`` polls to the ended
        :class:`WorkflowRun`.
        """
        # Everything that can be refused here is, before the first request.
        out_body = outputs_body(outputs)
        prepared = _prepare_run_input(source)
        sniffed = _sniff_media(prepared) if isinstance(prepared, bytes) else None
        local_kind = _local_source_kind(prepared) if isinstance(prepared, Path) else None
        live = _live_view_fields(_prepared_kind(prepared, local_kind, sniffed), stream, overlay)
        if isinstance(prepared, WorkflowRunInput):
            prepared = _with_rate(prepared, sample_fps)
        elif (
            sample_fps is not None and (local_kind or (sniffed[0] if sniffed else None)) != "video"
        ):
            raise ValueError("sample_fps is a video's: this source is an image.")
        workflow_id = await _resolve.workflow_id(self._transport, workflow)
        if isinstance(prepared, Path) and local_kind is not None:
            src = _local_input(local_kind, await self.upload_source(prepared), sample_fps)
        elif isinstance(prepared, bytes) and sniffed is not None:
            kind, filename, content_type = sniffed
            src = _local_input(
                kind,
                await self._upload_data(prepared, filename=filename, content_type=content_type),
                sample_fps,
            )
        else:
            src = cast("WorkflowRunInput", prepared)
        body: dict[str, Any] = {"source": src.model_dump(exclude_none=True), **live}
        if out_body is not None:
            body["outputs"] = out_body
        response = await self._transport.request(
            "POST",
            f"{_API}/{workflow_id}/run",
            json=body,
            timeout=_run_timeout(src),
        )
        created = self._parse(WorkflowRunCreated, response)
        if not wait:
            return created
        return await self.wait_for_run(created.run_id, poll_interval=poll_interval, timeout=timeout)

    async def invoke(
        self,
        workflow: str,
        *,
        source: RunSource,
        sample_fps: float | None = None,
        outputs: RunOutputs | None = None,
        stream: bool | None = None,
        overlay: bool | None = None,
        wait: bool = False,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_RUN_TIMEOUT,
    ) -> WorkflowRunCreated | WorkflowRun:
        """The old name of :meth:`run` (it takes the same arguments and does the same)."""
        if wait:
            return await self.run(
                workflow,
                source=source,
                sample_fps=sample_fps,
                outputs=outputs,
                stream=stream,
                overlay=overlay,
                wait=True,
                poll_interval=poll_interval,
                timeout=timeout,
            )
        return await self.run(
            workflow,
            source=source,
            sample_fps=sample_fps,
            outputs=outputs,
            stream=stream,
            overlay=overlay,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    async def upload_source(self, file_path: str | Path, *, content_type: str | None = None) -> str:
        """Upload a local video or image to run a workflow on; returns its handle.

        Async twin of :meth:`pictograph.resources.workflows.Workflows.upload_source` - streams the file straight to
        object storage (signed PUT, no API hop) under your organization's prefix. :meth:`run` calls this for you when
        given a local path.
        """
        path = Path(file_path).expanduser()
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
        return info.handle

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
        return info.handle

    async def get_run(self, run: str) -> WorkflowRun:
        """Poll a run. When ``status == "completed"`` read ``artifacts`` (signed URLs)."""
        response = await self._transport.request("GET", f"{_API}/runs/{run}")
        return self._parse(WorkflowRun, response["run"])

    async def run_log(
        self,
        run: str,
        *,
        stream: RunLogStream = "events",
        limit: int = 50,
        cursor: str | None = None,
        order: RunLogOrder = "asc",
        full: bool = False,
        class_name: str | None = None,
        track: str | int | None = None,
        branch: bool | None = None,
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
            track_id=track,
            branch=branch,
            min_confidence=min_confidence,
            source=source,
            event_type=event_type,
            trigger=trigger,
        )
        response = await self._transport.request("GET", f"{_API}/runs/{run}/log", params=params)
        return self._parse(RunLogPage, response)

    async def iter_run_log(
        self,
        run: str,
        *,
        stream: RunLogStream = "events",
        page_size: int = 200,
        order: RunLogOrder = "asc",
        full: bool = False,
        max_total: int | None = None,
        class_name: str | None = None,
        track: str | int | None = None,
        branch: bool | None = None,
        min_confidence: float | None = None,
        source: str | None = None,
        event_type: str | None = None,
        trigger: str | None = None,
    ) -> AsyncIterator[RunLogRecord]:
        """Every record of a run's log stream, page by page::

        async for event in client.workflows.iter_run_log(run):
            print(event.ts, event.trigger)
        """
        cursor: str | None = None
        seen = 0
        while True:
            page = await self.run_log(
                run,
                stream=stream,
                limit=page_size,
                cursor=cursor,
                order=order,
                full=full,
                class_name=class_name,
                track=track,
                branch=branch,
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

    async def run_log_summary(self, run: str) -> RunLogSummary:
        """What a run logged, per stream (sync twin: :meth:`pictograph.resources.workflows.Workflows.run_log_summary`)."""
        response = await self._transport.request("GET", f"{_API}/runs/{run}/log/summary")
        return self._parse(RunLogSummary, response)

    async def list_runs(
        self, workflow: str, *, limit: int = 50, offset: int = 0, kind: WorkflowRunsKind = "all"
    ) -> Sequence[WorkflowRunSummary]:
        """A single page of a workflow's runs, newest first (B494).

        Each item is a slim :class:`WorkflowRunSummary`; call :meth:`get_run` for a
        single run's full :class:`WorkflowRun` (including signed artifact downloads).

        Args:
            workflow: Workflow name or id.
            limit: Page size, 1-100 (default 50).
            offset: Rows to skip (default 0). Past the last run returns an empty page.
            kind: ``"all"`` (default), ``"file"`` or ``"stream"`` (a stream session is a run).
        """
        workflow_id = await _resolve.workflow_id(self._transport, workflow)
        response = await self._transport.request(
            "GET", f"{_API}/{workflow_id}/runs", params=_runs_params(limit, offset, kind)
        )
        return self._parse_list(WorkflowRunSummary, response.get("data", []))

    async def iter_runs(
        self,
        workflow: str,
        *,
        page_size: int = 50,
        max_total: int | None = None,
        kind: WorkflowRunsKind = "all",
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
                    params=_runs_params(limit, offset, kind),
                ),
            )

        return AsyncOffsetPager(
            fetch,
            items_key="data",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(WorkflowRunSummary, raw),
        )

    async def cancel_run(self, run: str) -> None:
        """Cancel an in-flight run (stops the GPU job). The run is FREE - workflows
        bill on success only, so a cancelled run is never charged (no refund needed)."""
        await self._transport.request("POST", f"{_API}/runs/{run}/cancel")

    async def bulk_cancel_runs(self, runs: Sequence[str]) -> BulkActionResult:
        """Cancel many in-flight workflow runs in one call.

        Each run is billed as with :meth:`cancel_run`. Duplicate ids are collapsed; an
        id that is not in your organization, or a run already over, lands in
        :attr:`~pictograph.models.common.BulkActionResult.not_found` instead of
        raising. Needs member or above.
        """
        response = await self._transport.request(
            "POST", f"{_API}/runs/bulk-cancel", json={"runs": list(runs)}
        )
        return self._parse(BulkActionResult, response)

    async def wait_for_run(
        self,
        run: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_RUN_TIMEOUT,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> WorkflowRun:
        """Poll a current until it reaches a terminal status, then return it.

        Returns the current on ``completed``; raises :class:`ApiError` if it ends
        ``error``/``cancelled`` and :class:`PollTimeoutError` on ``timeout``.

        Args:
            run: The current UUID from :meth:`current`.
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
            current = await self.get_run(run)
            if current.status == "completed":
                return current
            if current.status in ("error", "cancelled"):
                raise ApiError(
                    f"Workflow run {run} ended with status '{current.status}': "
                    f"{current.error or 'no error message provided'}",
                    response=current.model_dump(mode="json"),
                    code=job_end_code(current.status),
                )
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Workflow run {run} did not complete within {timeout:.0f}s "
                    f"(last status: {current.status}). Fetch later via "
                    f"client.workflows.get_run(...)."
                )
            await sleep_fn(poll_interval)
