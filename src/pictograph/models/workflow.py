"""Workflow Pydantic models - node-graph pipelines and their runs.

"Workflow" means exactly one thing in this SDK: the graph-editor feature modelled
here - a saved DAG (source → model → filter → track → measure → sink) run over an
image, video, or dataset, accessed via ``client.workflows``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

WorkflowStatus = Literal["draft", "ready", "archived"]
WorkflowRunStatus = Literal["queued", "processing", "completed", "error", "cancelled"]


class WorkflowCompute(BaseModel):
    """What a workflow runs with: its runner's GPU and its model's precision.

    ``gpu`` is ``"t4"`` (16 GB, the default) or ``"l4"`` (24 GB) - the inference GPUs with a video
    encoder; a run is billed per measured second at its GPU's rate. ``precision`` is ``"fp32"`` (the model as
    trained; the default) or ``"fp16"`` - Pictograph's realtime engine's fast mode, accuracy-
    equivalent and not output-identical; a run on the ONNX runner stays at fp32 and says so in
    its warnings. Strings, not literals, so a new tier never breaks an older SDK.
    """

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    gpu: str = "t4"
    precision: str = "fp32"


class WorkflowComputeGpu(BaseModel):
    """A GPU a workflow may run on, as :meth:`~pictograph.resources.workflows.Workflows.compute_options` lists it."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    gpu: str
    label: str
    # Its memory in GB (16 on a T4): what decides whether a model fits. ``description`` says the
    # same in words ("16 GB VRAM").
    vram_gb: int | None = None
    description: str = ""
    # The price of a minute on it (µUSD), as the server prices it.
    rate_per_min_micro_usd: int = 0


class WorkflowComputePrecision(BaseModel):
    """A precision a workflow's model may run at."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    precision: str
    label: str
    description: str = ""


class WorkflowComputeOptions(BaseModel):
    """The compute a workflow may run with: each GPU with its memory and rate, each precision, the
    default."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    gpus: list[WorkflowComputeGpu] = Field(default_factory=list)
    precisions: list[WorkflowComputePrecision] = Field(default_factory=list)
    default: WorkflowCompute = Field(default_factory=WorkflowCompute)


class Workflow(BaseModel):
    """A saved node-graph workflow."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    id: str
    organization_id: str
    name: str
    description: str | None = None
    graph: dict[str, Any] = Field(default_factory=dict)
    template_key: str | None = None
    status: WorkflowStatus = "draft"
    # What its runs run with, unless a run names its own (a server that does not say: the default).
    compute: WorkflowCompute = Field(default_factory=WorkflowCompute)
    last_run_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class WorkflowRun(BaseModel):
    """One execution of a workflow over a source."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    id: str
    organization_id: str
    workflow_id: str
    status: WorkflowRunStatus
    progress: float = 0.0
    frames_total: int | None = None
    frames_done: int = 0
    sample_fps: float | None = None
    step_results: dict[str, Any] = Field(default_factory=dict)
    # B473 - per condition node {passed, failed, combinator, rules}: how an if/else
    # condition split the run. Empty when the graph has no condition node.
    condition_results: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # Un-charged pre-run estimate (µUSD); the settled charge is ``final_micro_usd``
    # once the run completes (charge-on-success from measured GPU time).
    deposit_micro_usd: int = 0
    final_micro_usd: int | None = None
    error: str | None = None
    # Which engine ran it: "onnx" (the ONNX runner) or "prt" (PRT, Pictograph's realtime engine,
    # on the GPU end to end). A string, not a Literal, so a new engine never breaks an older SDK;
    # a server that does not say ran it on ONNX.
    engine: str = "onnx"
    # Whether the run wrote a log - its frames, events and stats, read with
    # :meth:`~pictograph.resources.workflows.Workflows.run_log`. A run on PRT does; an
    # ONNX run does not.
    has_log: bool = False
    # What it ran with: its GPU and its model's precision (fp16 only on PRT).
    compute: WorkflowCompute = Field(default_factory=WorkflowCompute)
    created_at: datetime | None = None
    completed_at: datetime | None = None


class WorkflowRunSummary(BaseModel):
    """A slim row from ``client.workflows.list_runs`` - the run-LIST projection.

    This is what a run looks like in a listing: status, progress, frame counts, a
    single settled/estimated ``cost_micro_usd``, and the artifact COUNT. It
    deliberately omits the bulky ``step_results`` and the per-artifact download
    URLs - call :meth:`~pictograph.resources.workflows.Workflows.get_run` for a
    single run's full :class:`WorkflowRun` record (including signed artifact
    downloads).
    """

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    id: str
    status: WorkflowRunStatus
    created_at: datetime | None = None
    completed_at: datetime | None = None
    # Wall-clock seconds from created_at to completed_at, when both are known.
    runtime_seconds: float | None = None
    # The SETTLED measured charge (µUSD) once terminal (``cost_settled`` True),
    # else the un-charged pre-run estimate. Workflows bill once, on success.
    cost_micro_usd: int = 0
    cost_settled: bool = False
    frames_done: int = 0
    frames_total: int | None = None
    progress: float = 0.0
    artifact_count: int = 0
    # Which engine ran it ("onnx" or "prt"), as on :class:`WorkflowRun`.
    engine: str = "onnx"
    # Whether the run wrote a log, as on :class:`WorkflowRun`.
    has_log: bool = False
    # What it ran with, as on :class:`WorkflowRun`.
    compute: WorkflowCompute = Field(default_factory=WorkflowCompute)
    error: str | None = None


class WorkflowRunCreated(BaseModel):
    """Run response - the new run id + ``deposit_micro_usd``, which is the
    un-charged pre-run ESTIMATE. Workflows bill ONCE, on success, from measured GPU
    time; a failed or cancelled run is free. The field name is kept for wire-compat."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    run_id: str
    # Un-charged pre-run estimate (µUSD). The settled charge lands on the run's
    # ``final_micro_usd`` once it completes; see :class:`WorkflowRun`.
    deposit_micro_usd: int = 0
    # A relative URL to poll for this run (returned by the on-demand ``invoke``
    # endpoint; absent from the plain ``run`` response).
    status_url: str | None = None


WorkflowInvokeKind = Literal["dataset", "video", "image"]


class WorkflowInvokeSource(BaseModel):
    """A source OVERRIDE for :meth:`~pictograph.resources.workflows.Workflows.invoke`.

    Runs a SAVED workflow over a source supplied AT CALL TIME, without touching the
    workflow's stored source node. Discriminated on ``kind``:

    - ``"dataset"``: a dataset (``dataset`` = its name or uuid) in your organization,
      optionally narrowed to one ``directory_path`` (absent = the whole dataset).
    - ``"video"`` / ``"image"``: a ``gcs_uri`` you own (the storage URI returned by
      the normal image/video upload flow), under your org's prefix, OR a ``url``: a
      public ``http(s)`` link to the file itself, which Pictograph downloads into your
      organization's storage before the run starts (a video up to 1 GB, an image up to
      25 MB). ``sample_fps`` and ``duration_seconds`` apply to ``"video"`` only.

    ``extra="forbid"`` so a mistyped field is caught locally rather than silently
    dropped into an empty override (which would run the workflow's own source).
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    kind: WorkflowInvokeKind
    dataset: str | None = None
    directory_path: str | None = None
    gcs_uri: str | None = None
    sample_fps: float | None = None
    duration_seconds: float | None = None
    url: str | None = None


class WorkflowUploadInfo(BaseModel):
    """Response from the workflow source upload-url endpoint (WF-D local-file mode):
    a signed PUT URL and the ``gcs_uri`` the media will live at once uploaded."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    upload_url: str
    gcs_uri: str


RunLogStream = Literal["frames", "events", "stats"]
RunLogOrder = Literal["asc", "desc"]


class RunLogRecord(BaseModel):
    """One record of a run's log: a frame, an event (a trigger firing, a capture and its upload),
    or the engine's stats - read with :meth:`~pictograph.resources.workflows.Workflows.run_log`.

    Every record carries its time (``ts``) and its identity in the log - ``engine_run`` (the
    engine process that wrote it), ``seq`` (its batch) and ``record_index`` (its place there).
    The rest depends on the stream; a field a stream does not have is ``None``. ``pictograph``
    (a frame's record as Pictograph JSON) and ``message`` (an event's or stats' whole message)
    come only with ``full=True``.
    """

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    ts: datetime
    engine_run: str
    seq: int
    record_index: int
    # events and stats: "event" | "capture" | "capture_uploaded" | "stats"
    type: str | None = None
    source: str | None = None
    frame: int | None = None
    pts: float | None = None
    image: str | None = None
    image_id: str | None = None
    image_index: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    annotation_count: int | None = None
    classes: list[str] = Field(default_factory=list)
    track_ids: list[str] = Field(default_factory=list)
    min_confidence: float | None = None
    max_confidence: float | None = None
    latency_total_ms: float | None = None
    latency_engine_ms: float | None = None
    latency_inference_ms: float | None = None
    model_name: str | None = None
    model_task: str | None = None
    event_id: str | None = None
    trigger: str | None = None
    sequence: int | None = None
    capture_kind: str | None = None
    capture_ok: bool | None = None
    capture_detail: str | None = None
    capture_name: str | None = None
    capture_uri: str | None = None
    capture_bytes: int | None = None
    # A record whose clock the log could not trust is dated by its arrival instead.
    clock_suspect: bool | None = None
    rule_values: dict[str, Any] | None = None
    sources: list[dict[str, Any]] | None = None
    pictograph: dict[str, Any] | None = None
    tag_confidences: Any = None
    measures: Any = None
    extra: dict[str, Any] | None = None
    message: dict[str, Any] | None = None


class RunLogPage(BaseModel):
    """One page of a run's log: its ``records``, and ``next_cursor`` until the last page."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    stream: RunLogStream
    order: RunLogOrder = "asc"
    records: list[RunLogRecord] = Field(default_factory=list)
    next_cursor: str | None = None
    # False on a plan that logs events and stats only (Free): the frames stream is empty.
    log_frames: bool = True


class RunLogStreamSummary(BaseModel):
    """One stream of a run's log: how many records, the first's and the last's time."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    count: int = 0
    first_ts: datetime | None = None
    last_ts: datetime | None = None


class RunLogSummary(BaseModel):
    """What a run logged, per stream (``frames`` / ``events`` / ``stats``), counted from the log."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    log_frames: bool = True
    streams: dict[str, RunLogStreamSummary] = Field(default_factory=dict)
