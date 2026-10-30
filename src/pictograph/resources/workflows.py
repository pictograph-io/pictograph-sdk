"""Workflows resource - the node-graph feature, run over an image / video / dataset, or LIVE.

Build a graph in the app, then drive runs headlessly here: ``create`` → ``run`` on any input (a video or an image -
a file, bytes or a link - a dataset or an export: a workflow's Source block holds none, each run names its own) →
poll ``get_run`` until ``status == "completed"`` (or ``run(..., wait=True)``) and read its ``artifacts``.

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
from typing import TYPE_CHECKING, Any, Literal, Union, cast, overload
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
    WorkflowNodeCatalog,
    WorkflowRun,
    WorkflowRunCreated,
    WorkflowRunInput,
    WorkflowRunInputKind,
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

#: What :meth:`Workflows.run` takes as its ``source``: a :class:`WorkflowRunInput` (or the same as a dict), a local
#: file's path, an ``http(s)`` link, or the bytes of a video or an image.
RunSource = Union[WorkflowRunInput, "Mapping[str, Any]", str, Path, bytes]


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
# A run on a URL answers once Pictograph has DOWNLOADED the file (it may take up to ten minutes), so that one request
# waits longer than the client's default timeout.
URL_RUN_TIMEOUT_SECONDS = 720.0
#: The old name of :data:`URL_RUN_TIMEOUT_SECONDS`.
URL_INVOKE_TIMEOUT_SECONDS = URL_RUN_TIMEOUT_SECONDS

# A local file's kind and content type, from its extension.
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


def _local_source_kind(path: Path) -> WorkflowRunInputKind:
    """`video` or `image` from a local file's extension."""
    ext = path.suffix.lower()
    if ext in _VIDEO_CONTENT_TYPES:
        return "video"
    if ext in _IMAGE_CONTENT_TYPES:
        return "image"
    raise ValueError(
        f"Can't tell whether {path.name!r} is a video or an image from its name. For a dataset or an export, "
        f'pass source={{"kind": "dataset", "dataset": "..."}} (or a WorkflowRunInput).'
    )


def _content_type_for(path: Path) -> str:
    ext = path.suffix.lower()
    return (
        _VIDEO_CONTENT_TYPES.get(ext) or _IMAGE_CONTENT_TYPES.get(ext) or "application/octet-stream"
    )


def _url_source_kind(url: str) -> WorkflowRunInputKind:
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


def _sniff_media(data: bytes) -> tuple[WorkflowRunInputKind, str, str]:
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


def _run_timeout(source: WorkflowRunInput) -> float | None:
    """A URL source answers only once the file is downloaded: wait for it."""
    return URL_RUN_TIMEOUT_SECONDS if source.url else None


def _prepare_run_input(source: RunSource) -> WorkflowRunInput | Path | bytes:
    """Normalize a run's ``source`` argument. Returns a ready :class:`WorkflowRunInput` for a dataset, an export, a
    link or a stored handle; or a local ``Path`` / the ``bytes`` the caller must upload first. An ``http(s)://``
    string is a link Pictograph downloads (the kind read from its extension). Any other ``str`` / ``Path`` must be an
    existing file (a dataset or an export needs the explicit dict form, so a typo is never taken for a missing
    upload)."""
    if isinstance(source, WorkflowRunInput):
        return source
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
        if not data:
            raise ValueError("The source bytes are empty.")
        return data
    if is_url(source):
        url = cast("str", source)
        return WorkflowRunInput(kind=_url_source_kind(url), url=url)
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"Source {str(source)!r} is not a local file. For a dataset or an export, pass a source dict, e.g. "
                f'{{"kind": "dataset", "dataset": "..."}}.'
            )
        return path
    return WorkflowRunInput.model_validate(dict(source))


def _with_rate(src: WorkflowRunInput, sample_fps: float | None) -> WorkflowRunInput:
    """``src`` read at ``sample_fps`` frames a second (a video's alone; given once)."""
    if sample_fps is None:
        return src
    if src.kind != "video":
        raise ValueError(
            f"sample_fps is a video's: this source is {'an' if src.kind[0] in 'aeiou' else 'a'} {src.kind}."
        )
    if src.sample_fps is not None and src.sample_fps != sample_fps:
        raise ValueError("Give sample_fps once: in the source, or as run(..., sample_fps=).")
    return src.model_copy(update={"sample_fps": float(sample_fps)})


def _local_input(
    kind: WorkflowRunInputKind, handle: str, sample_fps: float | None
) -> WorkflowRunInput:
    """An uploaded file as a run's input (its rate when it is a video)."""
    return _with_rate(
        WorkflowRunInput(kind=kind, video=handle)
        if kind == "video"
        else WorkflowRunInput(kind=kind, image=handle),
        sample_fps,
    )


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
        ("track", None if track_id is None else str(track_id)),
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

    def validate(
        self,
        graph: Mapping[str, Any],
        *,
        source: WorkflowRunInput | Mapping[str, Any] | None = None,
    ) -> WorkflowValidation:
        """Whether a run of ``graph`` would start, saving and starting nothing: every issue the
        validator, the engine and a run's start would raise - its models and annotations sinks are
        checked in your organization - each on its block (``node_id``). Only the credit balance is
        not asked. ``.valid`` is True when no issue is an error.

        ``source``: the input a run would name (a :class:`WorkflowRunInput` or the same as a
        mapping) - what depends on it is checked too: a single image under a Track or Measure
        block, an annotations output with no dataset on an input that has none. Without one, the
        graph is checked as a save checks it."""
        body: dict[str, Any] = {"graph": dict(graph)}
        if source is not None:
            src = (
                source
                if isinstance(source, WorkflowRunInput)
                else WorkflowRunInput.model_validate(dict(source))
            )
            body["source"] = src.model_dump(exclude_none=True)
        response = self._transport.request("POST", f"{_API}/validate", json=body)
        return self._parse(WorkflowValidation, response)

    def list(self) -> Sequence[Workflow]:
        """List every workflow in your organization."""
        response = self._transport.request("GET", f"{_API}/")
        return self._parse_list(Workflow, response.get("workflows", []))

    def get(self, workflow: str) -> Workflow:
        """Fetch one workflow, by name or id (includes a ``validation`` list)."""
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        response = self._transport.request("GET", f"{_API}/{workflow_id}")
        return self._parse(Workflow, response["workflow"])

    def create(
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
            scaledown_window: Seconds a container stays after its last run or stream, 2-3600.

        Raises:
            ConflictError: ``not_failed`` - a live endpoint: pause, resume or :meth:`update_endpoint` it instead.
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
        """Pause a workflow: its endpoint stopped (nothing held, nothing billed), its streams and runs refused
        until it is resumed."""
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        response = self._transport.request("POST", f"{_API}/{workflow_id}/pause")
        return self._parse(Workflow, response["workflow"])

    def resume(self, workflow: str) -> Workflow:
        """Resume a paused workflow: its endpoint deployed again on its graph as it is now - wait with
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
            scaledown_window: Seconds a container stays after its last run or stream, 2-3600.
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
        workflow: Workflow, *, token: str, timeout: float = DEFAULT_TIMEOUT_S
    ) -> DeploymentClient[Any]:
        """A direct client of a workflow's endpoint - its streams, with its endpoint token (``endpoint_token`` from
        :meth:`deploy`, or revealed in the app): ``.open_stream(source)``, exactly as a
        deployment's. Each frame's ``result`` is typed by the frame's own model (a workflow's first Model block's
        task); ``frame.measures`` is each line's and zone's counts, ``frame.message`` the whole message.

        A workflow's endpoint takes streams, never a single image (``infer`` is refused: ``predict_unavailable``).

        Args:
            workflow: The workflow, as ``get``, ``deploy`` or ``wait_until_live`` answer it: its ``endpoint`` names where its streams open.
            token: Its endpoint's ``pk_deploy_`` token (``endpoint_token`` from ``create``, or revealed in the app).
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
        track: str | int | None = None,
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
            track: Frames and events with this track.
            branch: Frames and events a Condition block routed down this branch (``True`` / ``False``).
            min_confidence: Frames whose best annotation is at least this confident.
            source: One stream's records (each stream logs as its own source).
            event_type: Events of this type (``"event"``, ``"capture"``, ``"capture_uploaded"``).
            trigger: Events of this trigger.

        Raises:
            NotFoundError: the workflow is not deployed (``not_deployed``) or has no live log yet (``no_log``):
                its endpoint's ``logged`` says beforehand (a draft never deployed, or a paused endpoint, has none).
        """
        workflow_id = _resolve.workflow_id(self._transport, workflow)
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
        track: str | int | None = None,
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
            class_name, track, branch, min_confidence, source, event_type, trigger: :meth:`live_log`'s filters.
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

    def live_log_summary(self, workflow: str) -> RunLogSummary:
        """What a workflow's live streams logged: each stream's count, its first and last record's time.

        Raises:
            NotFoundError: the workflow is not deployed (``not_deployed``) or has no live log yet (``no_log``):
                its endpoint's ``logged`` says beforehand.
        """
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        response = self._transport.request("GET", f"{_API}/{workflow_id}/endpoint/log/summary")
        return self._parse(RunLogSummary, response)

    def delete(self, workflow: str) -> None:
        """Delete a workflow and its run history (its live endpoint is stopped first)."""
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
            json={"workflows": _resolve.workflow_ids(self._transport, workflows)},
        )
        return self._parse(BulkDeleteResult, response)

    @overload
    def run(
        self,
        workflow: str,
        *,
        source: RunSource,
        sample_fps: float | None = ...,
        wait: Literal[False] = ...,
        poll_interval: float = ...,
        timeout: float = ...,
    ) -> WorkflowRunCreated: ...

    @overload
    def run(
        self,
        workflow: str,
        *,
        source: RunSource,
        sample_fps: float | None = ...,
        wait: Literal[True],
        poll_interval: float = ...,
        timeout: float = ...,
    ) -> WorkflowRun: ...

    def run(
        self,
        workflow: str,
        *,
        source: RunSource,
        sample_fps: float | None = None,
        wait: bool = False,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_RUN_TIMEOUT,
    ) -> WorkflowRunCreated | WorkflowRun:
        """Run a workflow on an input: a video or an image, a dataset, or an export.

        A workflow's Source block holds no input - every run names its own. It runs on the
        workflow's endpoint (queued until the endpoint is up and has room: the run's
        ``dispatch_note`` says what it waits for), on the workflow's GPU, each model at its
        block's precision, and is billed as the endpoint's uptime - a failed run too - with a
        SAM3 block's seconds charged once at its end.

        Args:
            workflow: Workflow name or id.
            source: What it runs on. One of:

                - **a local file's path** (``str`` / :class:`~pathlib.Path`) - a video or an
                  image on disk, uploaded (:meth:`upload_source`), then run.
                - **an** ``http(s)://`` **link** (``str``) to a video or an image - Pictograph
                  downloads it, then runs its copy (nothing passes through this machine). Its
                  kind is read from the link's extension; for a link without one, use the
                  mapping form with ``"url"``.
                - **bytes** - a video or an image in memory (stdin, a buffer); its kind is read
                  from the bytes, then it is uploaded like a file.
                - a :class:`~pictograph.models.workflow.WorkflowRunInput` or the same as a
                  mapping: ``{"kind": "dataset", "dataset": "<name-or-id>", "directory": "/f"}``
                  (a dataset, or one folder of it), ``{"kind": "export", "export": "<name-or-id>"}``
                  (the images an export captured), ``{"kind": "video", "video": "<handle>"}`` /
                  ``{"kind": "image", "image": "<handle>"}`` (a file :meth:`upload_source`
                  stored), or ``{"kind": "video", "url": "https://..."}``.

                A mistyped mapping field raises here (``extra="forbid"``). A link is downloaded
                before the run starts (a video up to 1 GB, an image up to 25 MB, within ten
                minutes), so that call can take a while to return.
            sample_fps: A video's frames read a second - absent, every frame at the video's own
                rate: for a video given as a file, bytes or a link (or in the mapping, as
                ``sample_fps``).
            wait: Poll until the run ends and return the whole :class:`WorkflowRun` (raises on
                an error, a cancel or a timeout, as :meth:`wait_for_run`). Else return at once.
            poll_interval: Seconds between checks when ``wait`` (default 5).
            timeout: Most seconds to wait when ``wait`` (default 3600).

        Returns:
            A :class:`WorkflowRunCreated` (``run_id`` + ``status_url``), or with ``wait=True``
            the ended :class:`WorkflowRun`.

        Raises:
            NotFoundError: The workflow, or a dataset or an export it names, is not in your
                organization.
            ValidationError: The input cannot run this graph (400) - a single image under a
                Track or Measure block, an annotations output with no dataset on an input that
                has none - or a handle outside your organization, or a link that answered with
                a web page instead of the file. A link larger than its cap is a 413.
            ConflictError: Its endpoint cannot take a run (409: ``workflow_paused`` - resume
                it; ``endpoint_failed`` - deploy it again; ``workflow_not_runnable`` - nothing
                to run as saved).
            PaymentRequiredError: The organization cannot spend at all (402).
            ValueError: A file, link or bytes whose kind can't be read, or a ``sample_fps``
                given for something that is no video (raised here, before any request).
        """
        # Everything that can be refused here is, before the first request.
        prepared = _prepare_run_input(source)
        sniffed = _sniff_media(prepared) if isinstance(prepared, bytes) else None
        local_kind = _local_source_kind(prepared) if isinstance(prepared, Path) else None
        if isinstance(prepared, WorkflowRunInput):
            prepared = _with_rate(prepared, sample_fps)
        elif (
            sample_fps is not None and (local_kind or (sniffed[0] if sniffed else None)) != "video"
        ):
            raise ValueError("sample_fps is a video's: this source is an image.")
        workflow_id = _resolve.workflow_id(self._transport, workflow)
        if isinstance(prepared, Path) and local_kind is not None:
            src = _local_input(local_kind, self.upload_source(prepared), sample_fps)
        elif isinstance(prepared, bytes) and sniffed is not None:
            kind, filename, content_type = sniffed
            src = _local_input(
                kind,
                self._upload_data(prepared, filename=filename, content_type=content_type),
                sample_fps,
            )
        else:
            src = cast("WorkflowRunInput", prepared)
        response = self._transport.request(
            "POST",
            f"{_API}/{workflow_id}/run",
            json={"source": src.model_dump(exclude_none=True)},
            timeout=_run_timeout(src),
        )
        created = self._parse(WorkflowRunCreated, response)
        if not wait:
            return created
        return self.wait_for_run(created.run_id, poll_interval=poll_interval, timeout=timeout)

    def invoke(
        self,
        workflow: str,
        *,
        source: RunSource,
        sample_fps: float | None = None,
        wait: bool = False,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_RUN_TIMEOUT,
    ) -> WorkflowRunCreated | WorkflowRun:
        """The old name of :meth:`run` (it takes the same arguments and does the same)."""
        if wait:
            return self.run(
                workflow,
                source=source,
                sample_fps=sample_fps,
                wait=True,
                poll_interval=poll_interval,
                timeout=timeout,
            )
        return self.run(
            workflow,
            source=source,
            sample_fps=sample_fps,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    def upload_source(self, file_path: str | Path, *, content_type: str | None = None) -> str:
        """Upload a local video or image to run a workflow on; returns its handle.

        Streams the file straight to object storage (a signed PUT URL - no API hop, so a
        multi-GB video never passes through the API), under your organization's prefix, and
        returns the handle a run's input names (``{"kind": "video", "video": handle}``).
        :meth:`run` calls this for you when given a local path; use it to upload once and run
        several times. Needs a ``member``+ API key.
        """
        path = Path(file_path).expanduser()
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
        return info.handle

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
        return info.handle

    def get_run(self, run: str) -> WorkflowRun:
        """Poll a run. When ``status == "completed"`` read ``artifacts`` (signed URLs)."""
        response = self._transport.request("GET", f"{_API}/runs/{run}")
        return self._parse(WorkflowRun, response["run"])

    def run_log(
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
        """One page of a run's log: the frames, events or stats the engine logged as it ran.

        Every run writes a log (its :attr:`~pictograph.models.workflow.WorkflowRun.has_log`).
        Its events (each trigger that fired, each capture and its upload) and stats are
        logged on every plan; every frame is logged on paid plans (``log_frames``).

        Args:
            run: The run's id.
            stream: ``"events"`` (default), ``"frames"`` or ``"stats"``.
            limit: Records per page, 1-200 (default 50).
            cursor: The previous page's ``next_cursor`` (None: the first page).
            order: ``"asc"`` (oldest first, default) or ``"desc"`` (newest first). A
                cursor pages only the order it was made for.
            full: Also each frame's whole Pictograph record (``pictograph``) / each event's
                and stats' whole message (``message``).
            class_name: Frames and events with an annotation of this class.
            track: Frames and events with this track.
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
            track_id=track,
            branch=branch,
            min_confidence=min_confidence,
            source=source,
            event_type=event_type,
            trigger=trigger,
        )
        response = self._transport.request("GET", f"{_API}/runs/{run}/log", params=params)
        return self._parse(RunLogPage, response)

    def iter_run_log(
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
    ) -> Iterator[RunLogRecord]:
        """Every record of a run's log stream, page by page (:meth:`run_log`'s filters)::

            for event in client.workflows.iter_run_log(run, stream="events"):
                print(event.ts, event.trigger, event.rule_values)

        ``max_total`` stops after that many records.
        """
        cursor: str | None = None
        seen = 0
        while True:
            page = self.run_log(
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

    def run_log_summary(self, run: str) -> RunLogSummary:
        """What a run logged: each stream's record count (from the log itself), its first and
        last record's time, and whether the plan logs frames (``log_frames``).

        Raises:
            NotFoundError: The run is not in your organization, or has no log.
        """
        response = self._transport.request("GET", f"{_API}/runs/{run}/log/summary")
        return self._parse(RunLogSummary, response)

    def list_runs(
        self, workflow: str, *, limit: int = 50, offset: int = 0, kind: WorkflowRunsKind = "all"
    ) -> Sequence[WorkflowRunSummary]:
        """A single page of a workflow's runs, newest first.

        Each item is a slim :class:`WorkflowRunSummary` (status, progress, frame
        counts, a settled/estimated ``cost_micro_usd``, ``artifact_count``, error)
        - not the bulky per-run record. Call :meth:`get_run` for a single run's
        full :class:`WorkflowRun` (including signed artifact download URLs).

        Args:
            workflow: Workflow name or id.
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

    def cancel_run(self, run: str) -> None:
        """Cancel an in-flight run (stops the GPU job). The run is FREE - workflows
        bill on success only, so a cancelled run is never charged (no refund needed)."""
        self._transport.request("POST", f"{_API}/runs/{run}/cancel")

    def bulk_cancel_runs(self, runs: Sequence[str]) -> BulkActionResult:
        """Cancel many in-flight workflow runs in one call.

        Each run is billed as with :meth:`cancel_run`. Duplicate ids are collapsed; an
        id that is not in your organization, or a run already over, lands in
        :attr:`~pictograph.models.common.BulkActionResult.not_found` instead of
        raising. Needs member or above.
        """
        response = self._transport.request(
            "POST", f"{_API}/runs/bulk-cancel", json={"runs": list(runs)}
        )
        return self._parse(BulkActionResult, response)

    def wait_for_run(
        self,
        run: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_RUN_TIMEOUT,
        sleep: Callable[[float], None] | None = None,
    ) -> WorkflowRun:
        """Poll a current until it reaches a terminal status, then return it.

        Returns the current on ``completed``. Raises :class:`ApiError` if the current ends
        ``error`` or ``cancelled``, and :class:`PollTimeoutError` if ``timeout``
        elapses first (the current keeps going server-side - re-poll via :meth:`get_run`).

        Args:
            run: The current UUID from :meth:`current`.
            poll_interval: Seconds between checks (default 5s).
            timeout: Max seconds to wait (default 3600, a run's longest).
            sleep: Injectable sleep (tests pass a no-op); defaults to ``time.sleep``.
        """
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else time.sleep
        deadline = time.monotonic() + timeout
        while True:
            current = self.get_run(run)
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
            sleep_fn(poll_interval)
