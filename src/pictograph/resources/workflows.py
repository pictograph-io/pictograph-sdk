"""Workflows resource - the node-graph feature, run over an image / video / dataset.

Build a graph in the app, then drive runs headlessly here: ``create`` → ``run`` →
poll ``get_run`` until ``status == "completed"`` and read its ``artifacts``.
"""

from __future__ import annotations

import time
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote, urlsplit

from pictograph._http.pagination import OffsetPager
from pictograph._image_input import is_url
from pictograph.exceptions import ApiError, PollTimeoutError
from pictograph.models.common import BulkActionResult, BulkDeleteResult
from pictograph.models.workflow import (
    RunLogOrder,
    RunLogPage,
    RunLogRecord,
    RunLogStream,
    RunLogSummary,
    Workflow,
    WorkflowInvokeKind,
    WorkflowInvokeSource,
    WorkflowRun,
    WorkflowRunCreated,
    WorkflowRunSummary,
    WorkflowStatus,
    WorkflowUploadInfo,
)
from pictograph.resources import _resolve
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

_API = "/api/v1/developer/workflows"
_DEFAULT_POLL_INTERVAL = 5.0
_DEFAULT_RUN_TIMEOUT = 3600.0  # 1 hour - matches the workflow runner's own timeout
# An invoke on a URL answers once Pictograph has DOWNLOADED the file (it may take up to
# ten minutes), so that one request waits longer than the client's default timeout.
URL_INVOKE_TIMEOUT_SECONDS = 720.0

# Local-file invoke (WF-D): infer the source kind + content-type from the extension.
_VIDEO_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".m4v": "video/x-m4v",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}
_IMAGE_CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}


def _local_source_kind(path: Path) -> WorkflowInvokeKind:
    """`video` or `image` from a local file's extension (WF-D local-file mode)."""
    ext = path.suffix.lower()
    if ext in _VIDEO_CONTENT_TYPES:
        return "video"
    if ext in _IMAGE_CONTENT_TYPES:
        return "image"
    raise ValueError(
        f"Can't infer a video/image source from {path.name!r}. Pass an explicit "
        f"source (a dict or WorkflowInvokeSource) for a dataset or a gcs_uri."
    )


def _content_type_for(path: Path) -> str:
    ext = path.suffix.lower()
    return (
        _VIDEO_CONTENT_TYPES.get(ext) or _IMAGE_CONTENT_TYPES.get(ext) or "application/octet-stream"
    )


def _url_source_kind(url: str) -> WorkflowInvokeKind:
    """`video` or `image` from the file name at the end of a link's path."""
    ext = PurePosixPath(unquote(urlsplit(url).path)).suffix.lower()
    if ext in _VIDEO_CONTENT_TYPES:
        return "video"
    if ext in _IMAGE_CONTENT_TYPES:
        return "image"
    raise ValueError(
        f"Can't tell whether {url!r} is a video or an image from its name. Say which: "
        f'source={{"kind": "video", "url": {url!r}}}.'
    )


# ISO BMFF brands (the `ftyp` box) that are still images, not video.
_STILL_BRANDS = (b"heic", b"heix", b"heim", b"heis", b"mif1", b"msf1", b"avif", b"avis")


def _sniff_media(data: bytes) -> tuple[WorkflowInvokeKind, str, str]:
    """``(kind, filename, content_type)`` of video or image bytes, read from their
    signature (stdin and in-memory bytes have no file name to go by)."""
    head = bytes(data[:64])
    if head.startswith(b"\xff\xd8\xff"):
        return "image", "upload.jpg", "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image", "upload.png", "image/png"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image", "upload.gif", "image/gif"
    if head.startswith(b"BM"):
        return "image", "upload.bmp", "image/bmp"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "image", "upload.webp", "image/webp"
    if head.startswith(b"RIFF") and head[8:12] == b"AVI ":
        return "video", "upload.avi", "video/x-msvideo"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        if b"webm" in head:
            return "video", "upload.webm", "video/webm"
        return "video", "upload.mkv", "video/x-matroska"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in _STILL_BRANDS:
            raise ValueError("HEIC and AVIF images are not read here. Convert it to JPEG or PNG.")
        if brand == b"qt  ":
            return "video", "upload.mov", "video/quicktime"
        if brand.startswith(b"M4V"):
            return "video", "upload.m4v", "video/x-m4v"
        return "video", "upload.mp4", "video/mp4"
    raise ValueError(
        "These bytes are not a video or an image this reads (MP4, MOV, M4V, AVI, MKV, "
        "WebM, JPEG, PNG, WebP, BMP or GIF)."
    )


def _invoke_timeout(source: WorkflowInvokeSource) -> float | None:
    """A URL source answers only once the file is downloaded: wait for it."""
    return URL_INVOKE_TIMEOUT_SECONDS if source.url else None


def _prepare_invoke_source(
    source: WorkflowInvokeSource | Mapping[str, Any] | str | Path | bytes,
) -> WorkflowInvokeSource | Path | bytes:
    """Normalize an invoke ``source`` argument. Returns a ready
    :class:`WorkflowInvokeSource` for a dataset / gcs-uri / URL source, a local
    ``Path`` or the ``bytes`` the caller must upload first. An ``http(s)://`` string
    is a link Pictograph downloads (the kind read from its extension). Any other
    ``str``/``Path`` must be an existing file (a dataset ref / gcs_uri needs the
    explicit dict form, so a typo can't be silently treated as a missing upload)."""
    if isinstance(source, WorkflowInvokeSource):
        return source
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
        if not data:
            raise ValueError("The source bytes are empty.")
        return data
    if is_url(source):
        url = cast("str", source)
        return WorkflowInvokeSource(kind=_url_source_kind(url), url=url)
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"Source {str(source)!r} is not a local file. For a dataset or a "
                f"gs:// URI, pass a source dict, e.g. "
                f'{{"kind": "dataset", "dataset": "..."}}.'
            )
        return path
    return WorkflowInvokeSource.model_validate(dict(source))


def _log_params(
    *,
    stream: RunLogStream,
    limit: int,
    cursor: str | None,
    order: RunLogOrder,
    full: bool,
    class_name: str | None,
    track_id: str | int | None,
    min_confidence: float | None,
    source: str | None,
    event_type: str | None,
    trigger: str | None,
) -> dict[str, Any]:
    """The query of one page of a run's log (``GET .../runs/{run}/log``): only what is set."""
    params: dict[str, Any] = {"stream": stream, "limit": limit, "order": order}
    if full:
        params["full"] = "true"
    for key, value in (
        ("cursor", cursor),
        ("class", class_name),
        ("track_id", None if track_id is None else str(track_id)),
        ("min_confidence", min_confidence),
        ("source", source),
        ("type", event_type),
        ("trigger", trigger),
    ):
        if value is not None:
            params[key] = value
    return params


class Workflows(Resource):
    """Manage node-graph workflows and their runs."""

    def list(self) -> Sequence[Workflow]:
        """List every workflow in your organization."""
        response = self._transport.request("GET", f"{_API}/")
        return self._parse_list(Workflow, response.get("workflows", []))

    def get(self, workflow: str) -> Workflow:
        """Fetch a single workflow by NAME (includes a ``validation`` list).

        A UUID is accepted too.
        """
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        response = self._transport.request("GET", f"{_API}/{workflow_id}")
        return self._parse(Workflow, response["workflow"])

    def create(
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
        response = self._transport.request("POST", f"{_API}/", json=body)
        return self._parse(Workflow, response["workflow"])

    def update(
        self,
        workflow: str,
        *,
        name: str | None = None,
        readme: str | None = None,
        description: str | None = None,
        graph: dict[str, Any] | None = None,
        status: WorkflowStatus | None = None,
    ) -> Workflow:
        """Update a workflow's name / readme / graph / status.

        ``readme`` is the card the UI renders; ``description`` is deprecated in
        its favour and kept so existing callers keep working.
        """
        body: dict[str, Any] = {}
        if readme is not None:
            body["readme"] = readme
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if graph is not None:
            body["graph"] = graph
        if status is not None:
            body["status"] = status
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        response = self._transport.request("PATCH", f"{_API}/{workflow_id}", json=body)
        return self._parse(Workflow, response["workflow"])

    def delete(self, workflow: str) -> None:
        """Delete a workflow, by NAME, and its run history."""
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        self._transport.request("DELETE", f"{_API}/{workflow_id}")

    def bulk_delete(self, workflows: Sequence[str]) -> BulkDeleteResult:
        """Delete many workflows in one atomic, org-scoped server-side call.

        Issues a single request the backend resolves with chunked,
        organization-scoped deletes, so it never fans out N calls. Requires the
        ``member``, ``admin``, or ``owner`` role (same as :meth:`delete`).

        Args:
            workflows: Names to delete (UUIDs are accepted too).
                Duplicates are ignored; ids that don't resolve in your
                organization are reported in
                :attr:`~pictograph.models.common.BulkDeleteResult.not_found`
                rather than raising, so a re-run still succeeds.

        Returns:
            A :class:`~pictograph.models.common.BulkDeleteResult`.

        Raises:
            ForbiddenError: Your API key role cannot manage workflows.
            ValidationError: ``workflows`` is empty.
        """
        response = self._transport.request(
            "POST",
            f"{_API}/bulk-delete",
            json={"workflow_ids": _resolve.workflow_ids(self._transport, workflows)},
        )
        return self._parse(BulkDeleteResult, response)

    def run(self, workflow: str) -> WorkflowRunCreated:
        """Validate + start a run. Raises ValidationError (400) if the graph is invalid,
        PaymentRequiredError (402) if there's not enough compute credit."""
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        response = self._transport.request("POST", f"{_API}/{workflow_id}/run")
        return self._parse(WorkflowRunCreated, response)

    def upload_source(self, local_path: str | Path, *, content_type: str | None = None) -> str:
        """Upload a local video/image as a workflow source; returns its ``gcs_uri``.

        Streams the file straight to object storage (a signed PUT URL - no API hop, so
        a multi-GB video never passes through the API pod), under your org's prefix, and
        returns the ``gcs_uri`` to hand to :meth:`invoke`. :meth:`invoke` calls this for
        you when given a local path; use it directly to upload once and invoke several
        times. Requires a ``member``+ API key.
        """
        path = Path(local_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Source file not found: {path}")
        ct = content_type or _content_type_for(path)
        info = self._parse(
            WorkflowUploadInfo,
            self._transport.request(
                "POST", f"{_API}/upload-url", json={"filename": path.name, "content_type": ct}
            ),
        )
        self._transport.upload_external(info.upload_url, path, content_type=ct)
        return info.gcs_uri

    def _upload_data(self, data: bytes, *, filename: str, content_type: str) -> str:
        """:meth:`upload_source` for bytes already in memory (stdin, a buffer)."""
        info = self._parse(
            WorkflowUploadInfo,
            self._transport.request(
                "POST",
                f"{_API}/upload-url",
                json={"filename": filename, "content_type": content_type},
            ),
        )
        self._transport.upload_external(info.upload_url, data, content_type=content_type)
        return info.gcs_uri

    def invoke(
        self,
        workflow: str,
        *,
        source: WorkflowInvokeSource | Mapping[str, Any] | str | Path | bytes,
        wait: bool = False,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_RUN_TIMEOUT,
    ) -> WorkflowRunCreated | WorkflowRun:
        """Run a SAVED workflow on-demand over a source given AT CALL TIME (WF-D).

        The workflow-DAG equivalent of a model deployment's ``/predict`` endpoint:
        one call runs a saved pipeline on *your* media without editing the workflow's
        stored source. Charge-on-success (a failed or cancelled run is free), exactly
        like :meth:`run`.

        Args:
            workflow: Workflow NAME (a UUID also works).
            source: What to run the workflow over for THIS run. One of:

                - **a local file path** (``str`` / :class:`~pathlib.Path`) - a video or
                  image on disk; uploaded via :meth:`upload_source` and run over.
                - **an** ``http(s)://`` **link** (``str``) to a video or an image -
                  Pictograph downloads it, then runs over its copy (nothing passes
                  through this machine). Whether it is a video or an image is read
                  from the link's extension; for a link without one, use the mapping
                  form below with ``"url"``.
                - **bytes** - a video or an image in memory (e.g. read from stdin);
                  its kind is read from the bytes, then it is uploaded like a file.
                - a :class:`~pictograph.models.workflow.WorkflowInvokeSource` or an
                  equivalent mapping:
                  ``{"kind": "dataset", "dataset": "<name-or-id>", "directory_path": "/f"}``
                  (a dataset/folder in your org),
                  ``{"kind": "video", "gcs_uri": "<uri>", "sample_fps": 2}`` /
                  ``{"kind": "image", "gcs_uri": "<uri>"}`` (a ``gcs_uri`` you own), or
                  ``{"kind": "video", "url": "https://...", "sample_fps": 2}`` (a link).

                A mistyped mapping field raises locally (``extra="forbid"``). A link is
                downloaded before the run starts (a video up to 1 GB, an image up to
                25 MB, within ten minutes), so that call can take a while to return.
            wait: If True, poll client-side until the run reaches a terminal state and
                return the full :class:`WorkflowRun` (raises on error/cancel/timeout,
                like :meth:`wait_for_run`). If False (default), return immediately.
            poll_interval: Seconds between checks when ``wait`` (default 5s).
            timeout: Max seconds to wait when ``wait`` (default 3600s = 1h).

        Returns:
            A :class:`WorkflowRunCreated` (``wait=False``) with the ``run_id`` +
            pre-run estimate + a ``status_url``, or the terminal :class:`WorkflowRun`
            (``wait=True``).

        Raises:
            NotFoundError: The workflow, or a ``dataset`` source, doesn't resolve in
                your org.
            ValidationError: The source is invalid for this graph (400) - e.g. an
                image under a tracking pipeline, a ``gcs_uri`` outside your org, or a
                link that answered with a web page instead of the file. A link larger
                than its cap is a 413.
            PaymentRequiredError: Not enough compute credit for the run estimate.
            ValueError: A link or bytes whose kind can't be read (raised locally).
        """
        # Everything that can be refused locally is, before the first request.
        prepared = _prepare_invoke_source(source)
        sniffed = _sniff_media(prepared) if isinstance(prepared, bytes) else None
        local_kind = _local_source_kind(prepared) if isinstance(prepared, Path) else None
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        if isinstance(prepared, Path) and local_kind is not None:
            src = WorkflowInvokeSource(kind=local_kind, gcs_uri=self.upload_source(prepared))
        elif isinstance(prepared, bytes) and sniffed is not None:
            kind, filename, content_type = sniffed
            src = WorkflowInvokeSource(
                kind=kind,
                gcs_uri=self._upload_data(prepared, filename=filename, content_type=content_type),
            )
        else:
            src = cast("WorkflowInvokeSource", prepared)
        response = self._transport.request(
            "POST",
            f"{_API}/{workflow_id}/invoke",
            json={"source": src.model_dump(exclude_none=True)},
            timeout=_invoke_timeout(src),
        )
        created = self._parse(WorkflowRunCreated, response)
        if not wait:
            return created
        return self.wait_for_run(created.run_id, poll_interval=poll_interval, timeout=timeout)

    def get_run(self, run_id: str) -> WorkflowRun:
        """Poll a run. When ``status == "completed"`` read ``artifacts`` (signed URLs)."""
        response = self._transport.request("GET", f"{_API}/runs/{run_id}")
        return self._parse(WorkflowRun, response["run"])

    def run_log(
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
        """One page of a run's log: the frames, events or stats the engine logged as it ran.

        A run on PRT writes a log (its :attr:`~pictograph.models.workflow.WorkflowRun.has_log`).
        Its events (each trigger that fired, each capture and its upload) and stats are
        logged on every plan; every frame is logged on paid plans (``log_frames``).

        Args:
            run_id: The run's id.
            stream: ``"events"`` (default), ``"frames"`` or ``"stats"``.
            limit: Records per page, 1-200 (default 50).
            cursor: The previous page's ``next_cursor`` (None: the first page).
            order: ``"asc"`` (oldest first, default) or ``"desc"`` (newest first). A
                cursor pages only the order it was made for.
            full: Also each frame's whole Pictograph record (``pictograph``) / each event's
                and stats' whole message (``message``).
            class_name: Frames and events with an annotation of this class.
            track_id: Frames and events with this track.
            min_confidence: Frames whose best annotation is at least this confident.
            source: One source's records.
            event_type: Events of this type (``"event"``, ``"capture"``, ``"capture_uploaded"``).
            trigger: Events of this trigger.

        Returns:
            The page; ``next_cursor`` is None on the last one. :meth:`iter_run_log` pages
            for you.

        Raises:
            NotFoundError: The run is not in your organization, or has no log.
        """
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
        response = self._transport.request("GET", f"{_API}/runs/{run_id}/log", params=params)
        return self._parse(RunLogPage, response)

    def iter_run_log(
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
    ) -> Iterator[RunLogRecord]:
        """Every record of a run's log stream, page by page (:meth:`run_log`'s filters)::

            for event in client.workflows.iter_run_log(run_id, stream="events"):
                print(event.ts, event.trigger, event.rule_values)

        ``max_total`` stops after that many records.
        """
        cursor: str | None = None
        seen = 0
        while True:
            page = self.run_log(
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

    def run_log_summary(self, run_id: str) -> RunLogSummary:
        """What a run logged: each stream's record count (from the log itself), its first and
        last record's time, and whether the plan logs frames (``log_frames``).

        Raises:
            NotFoundError: The run is not in your organization, or has no log.
        """
        response = self._transport.request("GET", f"{_API}/runs/{run_id}/log/summary")
        return self._parse(RunLogSummary, response)

    def list_runs(
        self, workflow: str, *, limit: int = 50, offset: int = 0
    ) -> Sequence[WorkflowRunSummary]:
        """A single page of a workflow's runs, newest first (B494).

        Each item is a slim :class:`WorkflowRunSummary` (status, progress, frame
        counts, a settled/estimated ``cost_micro_usd``, ``artifact_count``, error)
        - not the bulky per-run record. Call :meth:`get_run` for a single run's
        full :class:`WorkflowRun` (including signed artifact download URLs).

        Args:
            workflow: Workflow NAME (a UUID also works).
            limit: Page size, 1-100 (default 50).
            offset: Rows to skip (default 0). A request past the last run returns
                an empty page, not an error.

        Returns:
            The page of runs. Use :meth:`iter_runs` to page automatically.
        """
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        response = self._transport.request(
            "GET", f"{_API}/{workflow_id}/runs", params={"limit": limit, "offset": offset}
        )
        return self._parse_list(WorkflowRunSummary, response.get("data", []))

    def iter_runs(
        self, workflow: str, *, page_size: int = 50, max_total: int | None = None
    ) -> OffsetPager[WorkflowRunSummary]:
        """Auto-paging iterator over every run of a workflow, newest first.

        Resolves ``workflow`` once, then pages ``list_runs`` for you::

            for run in client.workflows.iter_runs("Line counter"):
                print(run.id, run.status, run.cost_micro_usd)
        """
        workflow_id = _resolve.workflow_id(self._transport, workflow)

        def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            return cast(
                "Mapping[str, Any]",
                self._transport.request(
                    "GET",
                    f"{_API}/{workflow_id}/runs",
                    params={"offset": offset, "limit": limit},
                ),
            )

        return OffsetPager(
            fetch,
            items_key="data",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(WorkflowRunSummary, raw),
        )

    def cancel_run(self, run_id: str) -> None:
        """Cancel an in-flight run (stops the GPU job). The run is FREE - workflows
        bill on success only, so a cancelled run is never charged (no refund needed)."""
        self._transport.request("POST", f"{_API}/runs/{run_id}/cancel")

    def bulk_cancel_runs(self, run_ids: Sequence[str]) -> BulkActionResult:
        """Cancel many in-flight workflow runs in one org-scoped server-side call.

        One request the backend resolves per-item (each run stops its GPU job; a
        cancelled run is free, like :meth:`cancel_run`) instead of fanning out N
        calls. Idempotent: duplicate ids are collapsed, and any id that doesn't
        resolve in your org OR is already terminal is reported in
        :attr:`~pictograph.models.common.BulkActionResult.not_found` rather than
        raising. Requires the same role as :meth:`cancel_run` (member+).
        """
        response = self._transport.request(
            "POST", f"{_API}/runs/bulk-cancel", json={"run_ids": list(run_ids)}
        )
        return self._parse(BulkActionResult, response)

    def wait_for_run(
        self,
        run_id: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_RUN_TIMEOUT,
        sleep: Callable[[float], None] | None = None,
    ) -> WorkflowRun:
        """Poll a run until it reaches a terminal status, then return it.

        Returns the run on ``completed``. Raises :class:`ApiError` if the run ends
        ``error`` or ``cancelled``, and :class:`PollTimeoutError` if ``timeout``
        elapses first (the run keeps going server-side - re-poll via :meth:`get_run`).

        Args:
            run_id: The run UUID from :meth:`run`.
            poll_interval: Seconds between checks (default 5s).
            timeout: Max seconds to wait (default 3600 = 1h, the runner's cap).
            sleep: Injectable sleep (tests pass a no-op); defaults to ``time.sleep``.
        """
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else time.sleep
        deadline = time.monotonic() + timeout
        while True:
            run = self.get_run(run_id)
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
            sleep_fn(poll_interval)
