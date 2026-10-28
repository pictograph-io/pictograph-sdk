"""Workflows resource - the node-graph feature, run over an image / video / dataset, or LIVE.

Build a graph in the app, then drive runs headlessly here: ``create`` → ``run`` →
poll ``get_run`` until ``status == "completed"`` and read its ``artifacts``.

A workflow's endpoint is made WITH it (its ``endpoint``: in the region and with the autoscaling ``create`` is
given) and both kinds of work run there: its file and dataset runs - queued until it has room, billed as its uptime -
and its live streams. ``wait_until_live``, then ``connect`` with its endpoint token and ``open_stream`` a camera
through it - every block a frame passes (models, Filter, tracker, lines and zones, Condition) runs on each frame, its
results back as it is processed::

    wf = client.workflows.create("Gate counter", template_key="line_counter", region="auto")
    token = wf.endpoint_token  # on the create's answer once: store it (revealable again in the app)
    wf = client.workflows.wait_until_live(wf)
    with client.workflows.connect(wf, token).open_stream(
        "rtsp://camera.example.com/live"
    ) as stream:
        for frame in stream.frames():
            print(frame.index, frame.measures)  # each line's and zone's counts on this frame
"""

from __future__ import annotations

import time
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote, urlsplit

from pictograph._http.pagination import OffsetPager
from pictograph._image_input import is_url
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
    WorkflowInvokeKind,
    WorkflowInvokeSource,
    WorkflowNodeCatalog,
    WorkflowRun,
    WorkflowRunCreated,
    WorkflowRunsKind,
    WorkflowRunSummary,
    WorkflowStatus,
    WorkflowTemplate,
    WorkflowUploadInfo,
    WorkflowValidation,
)
from pictograph.resources import _resolve
from pictograph.resources._base import Resource
from pictograph.resources._deployment_client import DEFAULT_TIMEOUT_S, DeploymentClient
from pictograph.resources.jobs import check_wait_args

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

_API = "/api/v1/developer/workflows"


def _runs_params(limit: int, offset: int, kind: WorkflowRunsKind) -> dict[str, Any]:
    """A runs page's query: its page, and its kind when it keeps only streams or only file runs."""
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if kind != "all":
        params["kind"] = kind
    return params


_DEFAULT_POLL_INTERVAL = 5.0
# How long waiting for a workflow's endpoint takes at most by default: its deploy (a minute or so), after a model's
# first TensorRT engine build on its GPU (a few minutes).
_DEFAULT_LIVE_TIMEOUT = 900.0
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
    branch: bool | None,
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
        # Spelled out: the route reads "true" / "false" (and False is a filter, never "no filter").
        ("branch", None if branch is None else ("true" if branch else "false")),
        ("min_confidence", min_confidence),
        ("source", source),
        ("type", event_type),
        ("trigger", trigger),
    ):
        if value is not None:
            params[key] = value
    return params


def _compute_body(compute: WorkflowCompute | Mapping[str, Any]) -> dict[str, Any]:
    """A workflow's compute as the API takes it: ``{"gpu"}``."""
    if isinstance(compute, WorkflowCompute):
        return compute.model_dump()
    return {k: v for k, v in dict(compute).items() if v is not None}


# What a workflow's endpoint is made with unless the caller says otherwise (the server's defaults too).
_ENDPOINT_DEFAULTS: dict[str, Any] = {
    "region": "auto",
    "min_containers": 0,
    "max_containers": 1,
    "scaledown_window": 60,
}


def endpoint_body(
    *, region: str, min_containers: int, max_containers: int, scaledown_window: int
) -> dict[str, Any]:
    """Where a workflow's endpoint runs and its autoscaling, as ``create`` and ``deploy`` send them: only what
    differs from the defaults (the server's own)."""
    given = {
        "region": region,
        "min_containers": min_containers,
        "max_containers": max_containers,
        "scaledown_window": scaledown_window,
    }
    return {k: v for k, v in given.items() if v != _ENDPOINT_DEFAULTS[k]}


def scaling_body(
    min_containers: int | None, max_containers: int | None, scaledown_window: int | None
) -> dict[str, Any]:
    """An endpoint's autoscaling change: only what is given. ValueError for nothing."""
    body = {
        k: v
        for k, v in (
            ("min_containers", min_containers),
            ("max_containers", max_containers),
            ("scaledown_window", scaledown_window),
        )
        if v is not None
    }
    if not body:
        raise ValueError("Give min_containers, max_containers or scaledown_window to change.")
    return body


def with_token(response: Mapping[str, Any]) -> dict[str, Any]:
    """The workflow of an answer, with the endpoint token it carries once (the ``create`` or ``deploy`` that made its
    endpoint)."""
    out = dict(response["workflow"])
    if response.get("endpoint_token"):
        out["endpoint_token"] = response["endpoint_token"]
    return out


def not_coming_live(workflow: Workflow) -> ApiError | None:
    """What :meth:`Workflows.wait_until_live` raises when waiting cannot make ``workflow``'s endpoint live - it has
    none, it has nothing to run, it failed, it was paused - else None. The workflow is the error's ``response``."""
    endpoint = workflow.endpoint
    if endpoint is None:
        return ApiError(
            f"Workflow '{workflow.name}' has no endpoint yet.",
            response=workflow.model_dump(mode="json"),
            code="not_deployed",
            fix="Deploy it (client.workflows.deploy(...)) - or run it: its first run makes one - then wait again.",
        )
    if endpoint.status == "failed":
        return ApiError(
            f"Workflow '{workflow.name}' failed to deploy: {endpoint.error_message or 'no message'}",
            response=workflow.model_dump(mode="json"),
            code=job_end_code("failed"),
            fix="Fix what the error names and save the workflow (or deploy it again).",
        )
    if endpoint.status == "draft" and endpoint.error_message:
        return ApiError(
            f"Workflow '{workflow.name}' has nothing to run: {endpoint.error_message}",
            response=workflow.model_dump(mode="json"),
            code="workflow_not_runnable",
            fix="Fix what the message names and save the workflow: it deploys when it has something to run.",
        )
    if endpoint.status in ("paused", "terminated"):
        return ApiError(
            f"Workflow '{workflow.name}' is {endpoint.status}.",
            response=workflow.model_dump(mode="json"),
            code=job_end_code("cancelled"),
            fix="Resume it (client.workflows.resume(...)), then wait again.",
        )
    return None


class Workflows(Resource):
    """Manage node-graph workflows, their runs and their live endpoint."""

    def compute_options(self, region: str | None = None) -> WorkflowComputeOptions:
        """The compute a workflow may run with, priced in ``region`` (``"us-east"``, ``"us-west"``; None: the one
        nearest you): each GPU (``t4``, ``l4``) with its memory (``vram_gb``) and its endpoint's container a minute and
        an hour there, each precision a Model block may take (``fp32``, ``fp16`` - set in the block's config), the
        default and every region (``regions``)."""
        params = {"region": region} if region else None
        response = self._transport.request("GET", f"{_API}/compute-options", params=params)
        return self._parse(WorkflowComputeOptions, response)

    def nodes(self) -> WorkflowNodeCatalog:
        """Every block a workflow graph can hold, to write one without guessing: each block's
        ports, its settings (the values each may hold, what the run does without it) and the
        rules the validator holds it to, with how a graph is written. ``.card("measure")`` is one
        block's card."""
        response = self._transport.request("GET", f"{_API}/nodes")
        return self._parse(WorkflowNodeCatalog, response)

    def templates(self) -> Sequence[WorkflowTemplate]:
        """The use cases a workflow starts from - the app's "New workflow" list - each with its
        graph. ``create(name, template_key=key)`` starts one from it."""
        response = self._transport.request("GET", f"{_API}/templates")
        return self._parse_list(WorkflowTemplate, response.get("templates", []))

    def validate(self, graph: Mapping[str, Any]) -> WorkflowValidation:
        """Whether a run of ``graph`` would start, saving and starting nothing: every issue the
        validator, the engine and a run's start would raise - its source, models and annotations
        sinks are checked in your organization - each on its block (``node_id``). Only the credit
        balance is not asked. ``.valid`` is True when no issue is an error."""
        response = self._transport.request("POST", f"{_API}/validate", json={"graph": dict(graph)})
        return self._parse(WorkflowValidation, response)

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
        graph: dict[str, Any] | None = None,
        *,
        readme: str | None = None,
        description: str | None = None,
        template_key: str | None = None,
        compute: WorkflowCompute | Mapping[str, Any] | None = None,
        region: str = "auto",
        min_containers: int = 0,
        max_containers: int = 1,
        scaledown_window: int = 60,
    ) -> Workflow:
        """Create a workflow - and its endpoint with it, where its file runs and its streams run - from a graph
        (``{version, nodes, edges}``), or from a template: ``template_key`` without a graph starts from that template's
        (see :meth:`templates`); with neither, an empty canvas. The workflow's ``validation`` lists what a run would
        still need.

        Its endpoint is deployed once the workflow has something to run (``endpoint.status``), and follows every
        save. Its token is on this answer once (``endpoint_token``): store it to stream from code (it is revealable
        again in the app).

        Args:
            name: Its name (deduplicated: ``"Name (2)"``).
            graph: The node graph.
            compute: The GPU its endpoint runs on - ``{"gpu": "l4"}``, see :meth:`compute_options`; without one, a T4.
                A Model block's precision is its own, in the graph: its config's ``"precision": "fp16"`` runs the
                model at half precision (faster, accuracy-equivalent).
            region: Where its endpoint runs: ``"us-east"`` (Virginia), ``"us-west"`` (Oregon), or ``"auto"`` - the
                one nearest where this call is made. Fixed once it is first deployed.
            min_containers: Containers kept warm for its streams, 0-5. 0 scales to zero when idle; more is billed
                continuously. Its runs always start from zero.
            max_containers: The most it runs under load, 1-10: four streams a container, one run a container.
            scaledown_window: Seconds a container stays after its last run or stream, 2-3600.

        Raises:
            PaymentRequiredError: a warm minimum, and no compute credits to spend.
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
        response = self._transport.request("POST", f"{_API}/", json=body)
        return self._parse(Workflow, with_token(response))

    def update(
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
        """Update a workflow's name / readme / graph / status / compute.

        ``readme`` is the card the UI renders; ``description`` is deprecated in
        its favour and kept so existing callers keep working. ``compute`` sets the GPU its endpoint
        runs on (the keys given; the rest stay the default): a change deploys its endpoint again on it.
        """
        body: dict[str, Any] = {}
        if compute is not None:
            body["compute"] = _compute_body(compute)
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

    def deploy(
        self,
        workflow: str,
        *,
        region: str = "auto",
        min_containers: int = 0,
        max_containers: int = 1,
        scaledown_window: int = 60,
    ) -> Workflow:
        """Deploy a workflow's endpoint. Its create makes one (deployed once it has something to run); this makes
        one for a workflow made before endpoints were made with workflows - in ``region`` (``"auto"``: the one nearest
        you) with its autoscaling, its token on the answer once (``endpoint_token``): store it, its streams open with
        it - and deploys a failed or draft one again, in its own region.

        Args:
            workflow: Workflow name or id.
            region: Where it is served: ``"us-east"`` (Virginia), ``"us-west"`` (Oregon), or ``"auto"`` - the one
                nearest where this call is made. Fixed once it is first deployed.
            min_containers: Containers kept warm, 0-5. 0 scales to zero when no stream runs; more is billed
                continuously.
            max_containers: The most it runs under load, 1-10, four streams each.
            scaledown_window: Seconds a container stays after its last stream, 2-3600.

        Raises:
            ConflictError: ``already_deployed`` - pause, resume or :meth:`update_endpoint` it instead.
            PaymentRequiredError: the organization has no compute credits to spend.
        """
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        body = endpoint_body(
            region=region,
            min_containers=min_containers,
            max_containers=max_containers,
            scaledown_window=scaledown_window,
        )
        response = self._transport.request("POST", f"{_API}/{workflow_id}/deploy", json=body)
        return self._parse(Workflow, with_token(response))

    def pause(self, workflow: str) -> Workflow:
        """Pause a workflow by NAME: its endpoint stopped (nothing held, nothing billed), its streams and runs refused
        until it is resumed."""
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        response = self._transport.request("POST", f"{_API}/{workflow_id}/pause")
        return self._parse(Workflow, response["workflow"])

    def resume(self, workflow: str) -> Workflow:
        """Resume a paused workflow by NAME: its endpoint deployed again on its graph as it is now - wait with
        :meth:`wait_until_live`."""
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        response = self._transport.request("POST", f"{_API}/{workflow_id}/resume")
        return self._parse(Workflow, response["workflow"])

    def update_endpoint(
        self,
        workflow: str,
        *,
        min_containers: int | None = None,
        max_containers: int | None = None,
        scaledown_window: int | None = None,
    ) -> Workflow:
        """Change a workflow's endpoint's autoscaling (only what is given): a live one takes it at once. Raising the
        minimum keeps containers warm - billed whether or not a stream runs.

        Args:
            workflow: Workflow name or id.
            min_containers: Containers kept warm, 0-5 (0 scales to zero).
            max_containers: The most it runs under load, 1-10, four streams each.
            scaledown_window: Seconds a container stays after its last stream, 2-3600.
        """
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        response = self._transport.request(
            "PATCH",
            f"{_API}/{workflow_id}/endpoint",
            json=scaling_body(min_containers, max_containers, scaledown_window),
        )
        return self._parse(Workflow, response["workflow"])

    def wait_until_live(
        self,
        workflow: str | Workflow,
        *,
        timeout: float = _DEFAULT_LIVE_TIMEOUT,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        sleep: Callable[[float], None] | None = None,
    ) -> Workflow:
        """Poll a workflow until its endpoint is ``active`` - its runs and streams start without waiting for a deploy
        (an asleep one starts a container for each, about 20 seconds). One that runs its files but cannot stream as
        saved is live for its runs: ``endpoint.streams`` is False and ``error_message`` says why.

        Raises:
            ApiError: it has no endpoint (``not_deployed``), nothing to run as saved (``workflow_not_runnable``),
                failed to deploy, or was paused. The workflow is the error's ``response``.
            PollTimeoutError: ``timeout`` elapsed first; it keeps deploying.
        """
        check_wait_args(poll_interval, timeout)
        workflow_id = (
            workflow.id
            if isinstance(workflow, Workflow)
            else _resolve.workflow_id(self._transport, workflow)
        )
        sleep_fn = sleep if sleep is not None else time.sleep
        deadline = time.monotonic() + timeout
        while True:
            current = self._parse(
                Workflow, self._transport.request("GET", f"{_API}/{workflow_id}")["workflow"]
            )
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
            sleep_fn(poll_interval)

    @staticmethod
    def connect(
        workflow: Workflow, token: str, *, timeout: float = DEFAULT_TIMEOUT_S
    ) -> DeploymentClient[Any]:
        """A direct client of a workflow's endpoint - its streams, with its endpoint token (``endpoint_token`` from
        :meth:`deploy`, or revealed in the app): ``.open_stream(source)``, exactly as a
        deployment's. Each frame's ``result`` is typed by the frame's own model (a workflow's first Model block's
        task); ``frame.measures`` is each line's and zone's counts, ``frame.message`` the whole message.

        A workflow's endpoint takes streams, never a single image (``infer`` is refused: ``predict_unavailable``).

        Args:
            workflow: The workflow, as ``get``, ``deploy`` or ``wait_until_live`` answer it: its ``endpoint`` names where its streams open.
            token: Its endpoint's ``pk_deploy_`` token (``endpoint_token`` from ``deploy``, or revealed in the app).
            timeout: Seconds a request to its endpoint may take.
        """
        endpoint = workflow.endpoint
        if endpoint is None or not endpoint.endpoint_url:
            raise ValueError(
                "The workflow has no endpoint to connect to: deploy it (client.workflows.deploy(...))."
            )
        return DeploymentClient(endpoint.endpoint_url, token, timeout=timeout)

    def live_log(
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
        branch: bool | None = None,
        min_confidence: float | None = None,
        source: str | None = None,
        event_type: str | None = None,
        trigger: str | None = None,
    ) -> RunLogPage:
        """One page of a workflow's LIVE log - what its streams' frames found, their events and stats - newest first
        by default, with :meth:`run_log`'s filters and page shape (its current deployment's: each deploy logs anew).

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
            branch: Frames and events a Condition block routed down this branch (``True`` / ``False``).
            min_confidence: Frames whose best annotation is at least this confident.
            source: One stream's records (each stream logs as its own source).
            event_type: Events of this type (``"event"``, ``"capture"``, ``"capture_uploaded"``).
            trigger: Events of this trigger.

        Raises:
            NotFoundError: the workflow is not deployed (``not_deployed``) or has no live log yet (``no_log``).
        """
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        params = _log_params(
            stream=stream,
            limit=limit,
            cursor=cursor,
            order=order,
            full=full,
            class_name=class_name,
            track_id=track_id,
            branch=branch,
            min_confidence=min_confidence,
            source=source,
            event_type=event_type,
            trigger=trigger,
        )
        response = self._transport.request(
            "GET", f"{_API}/{workflow_id}/endpoint/log", params=params
        )
        return self._parse(RunLogPage, response)

    def iter_live_log(
        self,
        workflow: str,
        *,
        stream: RunLogStream = "frames",
        page_size: int = 200,
        order: RunLogOrder = "desc",
        full: bool = False,
        max_total: int | None = None,
        class_name: str | None = None,
        track_id: str | int | None = None,
        branch: bool | None = None,
        min_confidence: float | None = None,
        source: str | None = None,
        event_type: str | None = None,
        trigger: str | None = None,
    ) -> Iterator[RunLogRecord]:
        """Every record of a workflow's live log stream, page by page (:meth:`live_log`, its filters too);
        ``max_total`` stops after that many.

        Args:
            workflow: Workflow name or id.
            stream: ``"frames"`` (default), ``"events"`` or ``"stats"``.
            page_size: Records fetched per request, 1-200.
            order: ``"desc"`` (newest first, default) or ``"asc"``.
            full: Also each record's whole Pictograph record or message.
            max_total: Stop after this many records. ``None`` walks the whole stream.
            class_name, track_id, branch, min_confidence, source, event_type, trigger: :meth:`live_log`'s filters.
        """
        cursor: str | None = None
        seen = 0
        while True:
            page = self.live_log(
                workflow,
                stream=stream,
                limit=page_size,
                cursor=cursor,
                order=order,
                full=full,
                class_name=class_name,
                track_id=track_id,
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

    def live_log_summary(self, workflow: str) -> RunLogSummary:
        """What a workflow's live streams logged: each stream's count, its first and last record's time."""
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        response = self._transport.request("GET", f"{_API}/{workflow_id}/endpoint/log/summary")
        return self._parse(RunLogSummary, response)

    def delete(self, workflow: str) -> None:
        """Delete a workflow, by NAME, and its run history (its live endpoint is stopped first)."""
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
        """Validate + start a run on the workflow's endpoint: queued until the endpoint is up and has room (the run's
        ``dispatch_note`` says what it waits for), on the workflow's GPU, each model at its block's precision. The run
        records what it ran with (:attr:`WorkflowRun.compute`). It is billed as its endpoint's uptime while it runs -
        a failed run too - and a SAM3 block's seconds once, at its end; nothing is charged at the start.

        Raises:
            ValidationError: the graph is invalid (400).
            ConflictError: its endpoint cannot take a run (409: ``workflow_paused`` - resume it; ``endpoint_failed``
                - deploy it again; ``workflow_not_runnable`` - nothing to run as saved).
            PaymentRequiredError: the organization cannot spend at all (402).
        """
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
        stored source. It runs on the workflow's endpoint and is billed exactly like
        :meth:`run` (its endpoint's uptime).

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
            A :class:`WorkflowRunCreated` (``wait=False``) with the ``run_id`` + a
            ``status_url``, or the terminal :class:`WorkflowRun` (``wait=True``).

        Raises:
            NotFoundError: The workflow, or a ``dataset`` source, doesn't resolve in
                your org.
            ValidationError: The source is invalid for this graph (400) - e.g. an
                image under a tracking pipeline, a ``gcs_uri`` outside your org, or a
                link that answered with a web page instead of the file. A link larger
                than its cap is a 413.
            ConflictError: The workflow's endpoint cannot take a run (paused, failed,
                nothing to run) - as :meth:`run`.
            PaymentRequiredError: The organization cannot spend at all.
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
        body: dict[str, Any] = {"source": src.model_dump(exclude_none=True)}
        response = self._transport.request(
            "POST",
            f"{_API}/{workflow_id}/invoke",
            json=body,
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
        branch: bool | None = None,
        min_confidence: float | None = None,
        source: str | None = None,
        event_type: str | None = None,
        trigger: str | None = None,
    ) -> RunLogPage:
        """One page of a run's log: the frames, events or stats the engine logged as it ran.

        Every run writes a log (its :attr:`~pictograph.models.workflow.WorkflowRun.has_log`).
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
            branch: Frames and events a Condition block routed down this branch (``True`` / ``False``).
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
            branch=branch,
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
        branch: bool | None = None,
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

    def run_log_summary(self, run_id: str) -> RunLogSummary:
        """What a run logged: each stream's record count (from the log itself), its first and
        last record's time, and whether the plan logs frames (``log_frames``).

        Raises:
            NotFoundError: The run is not in your organization, or has no log.
        """
        response = self._transport.request("GET", f"{_API}/runs/{run_id}/log/summary")
        return self._parse(RunLogSummary, response)

    def list_runs(
        self, workflow: str, *, limit: int = 50, offset: int = 0, kind: WorkflowRunsKind = "all"
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
            kind: ``"all"`` (default), ``"file"`` (a video, image or dataset run) or
                ``"stream"`` - each stream session on the workflow's endpoint is a run of
                it (its ``source.kind`` is ``"stream"``).

        Returns:
            The page of runs. Use :meth:`iter_runs` to page automatically.
        """
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        response = self._transport.request(
            "GET", f"{_API}/{workflow_id}/runs", params=_runs_params(limit, offset, kind)
        )
        return self._parse_list(WorkflowRunSummary, response.get("data", []))

    def iter_runs(
        self,
        workflow: str,
        *,
        page_size: int = 50,
        max_total: int | None = None,
        kind: WorkflowRunsKind = "all",
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
                    params=_runs_params(limit, offset, kind),
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
                    code=job_end_code(run.status),
                )
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Workflow run {run_id} did not complete within {timeout:.0f}s "
                    f"(last status: {run.status}). Fetch later via "
                    f"client.workflows.get_run(...)."
                )
            sleep_fn(poll_interval)
