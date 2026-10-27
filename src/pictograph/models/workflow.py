"""Workflow Pydantic models - node-graph pipelines and their runs.

"Workflow" means exactly one thing in this SDK: the graph-editor feature modelled
here - a saved DAG (source → model → filter → track → measure → sink) run over an
image, video, or dataset, accessed via ``client.workflows``.

A workflow is also served LIVE (its :class:`WorkflowEndpoint`): a stream - a camera, a video you send over WebRTC -
runs through every block a frame passes (its models, Filter, tracker, lines and zones, Condition), each frame's
results coming back as it is processed, the annotated video watchable as it plays.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# A run's engine builds: the record a provisioning deployment carries too.
from pictograph.models.deployment import DeploymentRegion, DeploymentRuntime, DeploymentStatus
from pictograph.models.engine_build import EngineBuild

WorkflowStatus = Literal["draft", "ready", "archived"]
WorkflowRunStatus = Literal["queued", "processing", "completed", "error", "cancelled"]


class WorkflowCompute(BaseModel):
    """What a workflow's runs run on: the runner's GPU.

    ``gpu`` is ``"t4"`` (16 GB, the default) or ``"l4"`` (24 GB) - the inference GPUs with a video
    encoder; a run is billed per measured second at its GPU's rate. Each Model block holds its own
    precision, in its config: ``{"precision": "fp16"}`` runs the model at half precision (faster,
    accuracy-equivalent, not output-identical); ``"fp32"``, the default, runs it as trained.
    A string, not a literal, so a new tier never breaks an older SDK.
    """

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    gpu: str = "t4"


class WorkflowRunCompute(BaseModel):
    """What a run ran with: its GPU and the precision its model ran at."""

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
    # Its endpoint's container a minute and an hour on it (µUSD) in the options' region, as the server prices it - a
    # workflow's runs and streams are its endpoint's held compute (B579 slice 5).
    rate_per_min_micro_usd: int = 0
    cost_per_hour_micro_usd: int = 0


class WorkflowComputePrecision(BaseModel):
    """A precision a Model block may take (its config's ``precision``)."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    precision: str
    label: str
    description: str = ""


class WorkflowComputeOptions(BaseModel):
    """The compute a workflow may run with, priced in one region: each GPU with its memory and its
    endpoint's rate there, each precision a Model block may take, the default, and every region its
    endpoint can run in."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    gpus: list[WorkflowComputeGpu] = Field(default_factory=list)
    precisions: list[WorkflowComputePrecision] = Field(default_factory=list)
    default: WorkflowCompute = Field(default_factory=WorkflowCompute)
    # The region these prices are for, and every region (``id``, ``label``, ``location``).
    region: str | None = None
    regions: list[DeploymentRegion] = Field(default_factory=list)
    # The price of a GiB of its streams' data transfer (µUSD).
    transfer_per_gib_micro_usd: int | None = None


class WorkflowIssue(BaseModel):
    """One thing a run of a graph would refuse (``severity == "error"``) or wants you to know
    (``"warning"``): its ``code``, its ``message`` and the block it is on (``node_id``; None for
    the graph as a whole)."""

    model_config = ConfigDict(extra="ignore")

    code: str
    message: str
    node_id: str | None = None
    severity: str = "error"


class WorkflowValidation(BaseModel):
    """Whether a run of a graph would start (:meth:`~pictograph.resources.workflows.Workflows.validate`):
    ``valid`` when no issue is an error, and every issue - the validator's, the engine's and what a
    run checks before it starts (its source, models and annotations sinks)."""

    model_config = ConfigDict(extra="ignore")

    valid: bool
    issues: list[WorkflowIssue] = Field(default_factory=list)

    @property
    def errors(self) -> list[WorkflowIssue]:
        """The issues that stop a run."""
        return [i for i in self.issues if i.severity == "error"]


class WorkflowNodeField(BaseModel):
    """One setting of a block's config: its ``key``, its JSON ``type``, whether a run needs it
    (``True``, or the condition under which it does, in words), its ``default`` (what the run does
    without it), the ``values`` it may hold, a ``note``, and the settings inside it (``fields``: an
    object's, or each item's of a list)."""

    model_config = ConfigDict(extra="ignore")

    key: str
    type: str
    required: bool | str = False
    default: Any = None
    values: list[Any] | None = None
    note: str = ""
    fields: list[WorkflowNodeField] | None = None


class WorkflowNodeCard(BaseModel):
    """One block type a workflow graph can hold: the port it takes (``inputs``) and the one it emits
    (``output``; None for a sink), a new block's config as the editor makes it (``new_block``), its
    settings (``fields``) and the rules the validator holds it to (``rules``)."""

    model_config = ConfigDict(extra="ignore")

    type: str
    label: str
    summary: str
    inputs: list[str] = Field(default_factory=list)
    output: str | None = None
    new_block: dict[str, Any] = Field(default_factory=dict)
    fields: list[WorkflowNodeField] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)


class WorkflowNodeCatalog(BaseModel):
    """Every block type (``nodes``, in the editor's order) and how a graph is written (``graph``)."""

    model_config = ConfigDict(extra="ignore")

    graph: dict[str, Any] = Field(default_factory=dict)
    nodes: list[WorkflowNodeCard] = Field(default_factory=list)

    def card(self, block: str) -> WorkflowNodeCard:
        """The card of the block type ``block`` (``"measure"``, ...); ``KeyError`` when there is none."""
        for node in self.nodes:
            if node.type == block:
                return node
        raise KeyError(f"No block type {block!r}. Blocks: {', '.join(n.type for n in self.nodes)}.")


class WorkflowTemplate(BaseModel):
    """A use case a workflow starts from - the app's "New workflow" list - with its graph.
    ``client.workflows.create(name, template_key=key)`` creates one from it."""

    model_config = ConfigDict(extra="ignore")

    key: str
    title: str
    description: str
    graph: dict[str, Any] = Field(default_factory=dict)


class WorkflowLiveMeasure(BaseModel):
    """A line or zone of a workflow's live stream, as each frame's ``measures`` names it (by ``id``)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    # ``"line"`` or ``"zone"``; a zone counts ``"occupancy"`` or ``"dwell"``, a line ``"line_cross"``.
    kind: str
    measure: str
    # How the workflow labels it: L1, L2, ... and P1, P2, ...
    label: str
    # The colour its overlay draws it in (``"#38bdf8"``).
    color: str | None = None


class WorkflowLiveModel(BaseModel):
    """A model a workflow's live stream runs: one per Model block's model at its precision."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    model_id: str
    name: str
    # ``"fp32"`` or ``"fp16"``: the Model block's own.
    precision: str = "fp32"
    model_type: str | None = None


class WorkflowLiveEvent(BaseModel):
    """An event a workflow's live outputs act on while it streams - the same events its log records, named as the app
    names them. A webhook output's selected events go out in ``workflow_stream.events`` deliveries; an annotations
    output saves the frame of each it selected (``captured``), and its passive samples (kind ``"passive"``)."""

    model_config = ConfigDict(extra="ignore")

    # Its trigger in the workflow's log (each event record's ``trigger``): ``"<kind>-<8 hex>"``.
    trigger: str
    # What a live config selects it by: ``"crossing:<measure node>:<geometry>"``, ``"entry:..."``, ``"exit:..."``,
    # ``"dwell:..."``, ``"appearance:<class>"`` / ``"appearance:*"``, ``"condition"``; ``"passive:<output node>"``.
    key: str
    # ``"crossing"``, ``"entry"``, ``"exit"``, ``"dwell"``, ``"appearance"``, ``"condition"`` or ``"passive"``.
    kind: str
    # What a person reads (``"Crossing · Gate"``, ``"Appearance · person"``, ``"Passive capture"``).
    name: str
    # Whether its frame is saved into a dataset (an annotations output selected it).
    captured: bool = False


class WorkflowLiveOutputs(BaseModel):
    """What a workflow endpoint's live outputs did since they first acted: every outcome counted, none silent."""

    model_config = ConfigDict(extra="ignore")

    # The events its streams sent.
    events: int = 0
    # The ``workflow_stream.*`` webhook deliveries enqueued (each retried and logged as every webhook is).
    deliveries: int = 0
    # Frames saved into a dataset, and frames that were not (``last_not_saved_reason`` says why).
    saved: int = 0
    not_saved: int = 0
    # Events whose frame was not captured: the stream already saved a frame that second (at most one a second, an
    # event's before a passive sample's).
    skipped: int = 0
    since: datetime | None = None
    last_not_saved_reason: str | None = None
    last_not_saved_at: datetime | None = None


class WorkflowLive(BaseModel):
    """What a workflow's live stream runs, reports and draws: its models, its lines and zones, its classes - each in
    the colour it is drawn in on the live view, and its annotation type (``"bbox"``, ``"polygon"``, ``"keypoint"``,
    ``"tag"``) - and the events its live outputs act on."""

    model_config = ConfigDict(extra="ignore")

    models: list[WorkflowLiveModel] = Field(default_factory=list)
    measures: list[WorkflowLiveMeasure] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    class_colors: dict[str, str] = Field(default_factory=dict)
    class_types: dict[str, str] = Field(default_factory=dict)
    events: list[WorkflowLiveEvent] = Field(default_factory=list)


class WorkflowEndpoint(BaseModel):
    """A workflow's endpoint: where its file and dataset runs and its live streams run, in its region.

    Made with the workflow (:meth:`~pictograph.resources.workflows.Workflows.create`; one made before endpoints were
    made with workflows gets one at its first run, or from
    :meth:`~pictograph.resources.workflows.Workflows.deploy`). Its ``status``:

    * ``"draft"`` - nothing running yet: the workflow has nothing to run (``error_message`` says why - no model);
      the save that makes it runnable deploys it;
    * ``"provisioning"`` - being deployed (``engine_build`` while a model's TensorRT engine is built);
    * ``"active"`` - deployed; ``runtime.state`` says what it is doing now (``"live"``, ``"starting"``,
      ``"waiting"``, ``"asleep"``);
    * ``"paused"`` - stopped by you (or at a zero balance): its streams and runs are refused until it is resumed;
    * ``"failed"`` - its deploy failed (``error_message``).

    A save is live at the endpoint's next stream and run; one that needs another model, precision or GPU deploys it
    again. A workflow that cannot stream as saved (a SAM3 block) runs its files there (``streams`` False) - or keeps
    streaming its last live version - ``error_message`` saying why. It is billed as a deployment is: the compute it
    holds while a run or a stream runs.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    status: DeploymentStatus
    region: str | None = None
    gpu_type: str | None = None
    min_containers: int = 0
    max_containers: int = 1
    # Seconds its last container stays up after its last stream ends.
    scaledown_window: int = 60
    # Where its streams open: its ``/sessions`` on its region's gateway - it takes streams, never a predict
    # (connect with :meth:`~pictograph.resources.workflows.Workflows.connect`).
    endpoint_url: str | None = None
    auth_token_prefix: str | None = None
    error_message: str | None = None
    engine_build: EngineBuild | None = None
    # The streams it runs at once (0 until it is deployed): its maximum containers x the streams each holds.
    stream_limit: int = 0
    # Its file and dataset runs at once: one a container, up to its maximum containers.
    run_limit: int = 0
    # Whether it takes streams as saved (False: it runs its files only - ``error_message`` says why).
    streams: bool = True
    # Whether it has ever been deployed (its region is fixed from then on).
    deployed: bool = False
    live: WorkflowLive = Field(default_factory=WorkflowLive)
    # What its live outputs did (webhook deliveries, frames saved into a dataset): None until they first act.
    live_outputs: WorkflowLiveOutputs | None = None
    # What it is doing now (``active`` only): its containers serve its streams.
    runtime: DeploymentRuntime | None = None
    cost_per_hour_micro_usd: int = 0
    max_cost_per_hour_micro_usd: int = 0
    transfer_per_gib_micro_usd: int | None = None
    accrued_cost_micro_usd: int = 0
    uptime_seconds: int = 0
    transfer_bytes: int = 0
    transfer_cost_micro_usd: int = 0
    created_at: datetime | None = None
    started_at: datetime | None = None
    paused_at: datetime | None = None
    updated_at: datetime | None = None


class Workflow(BaseModel):
    """A saved node-graph workflow."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    id: str
    organization_id: str
    name: str
    description: str | None = None
    readme: str | None = None
    graph: dict[str, Any] = Field(default_factory=dict)
    template_key: str | None = None
    status: WorkflowStatus = "draft"
    # What its runs run on, unless a run names its own (a server that does not say: the default).
    compute: WorkflowCompute = Field(default_factory=WorkflowCompute)
    last_run_id: str | None = None
    # What a run of its graph would still be refused for (get / create / update): the validator's
    # errors, then the engine's. Empty when it could run.
    validation: list[WorkflowIssue] = Field(default_factory=list)
    # Its live endpoint (None: never deployed - deploy it with ``workflows.deploy``).
    endpoint: WorkflowEndpoint | None = None
    # The endpoint's bearer token, in plaintext - ONLY on the answer that made the endpoint (its first ``deploy``).
    # Store it: streams open with it (a member can reveal it again in the app).
    endpoint_token: str | None = None
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
    # Its usage once it ends (µUSD): its own seconds at its endpoint's rate, and a SAM3 block's
    # charge. What is billed is its endpoint's uptime while it runs, a failed run too.
    final_micro_usd: int | None = None
    error: str | None = None
    # Why a queued run waits ("Waiting: its endpoint is starting."); None once it runs.
    dispatch_note: str | None = None
    # The endpoint it was dispatched onto (None while queued, and for a run of before endpoints): a
    # build it shows (``engine_build``) is that endpoint's, made when it was deployed.
    endpoint_id: str | None = None
    # Whether the run wrote a log - its frames, events and stats, read with
    # :meth:`~pictograph.resources.workflows.Workflows.run_log`. Every run does, unless its log
    # could not be opened (its ``warnings`` say so); a run from before PRT ran every workflow has none.
    has_log: bool = False
    # What it ran with: its GPU and its Model blocks' precision.
    compute: WorkflowRunCompute = Field(default_factory=WorkflowRunCompute)
    # The TensorRT engines its endpoint built for it - a model's first deploy on its GPU (None:
    # every engine was stored). While a run is "queued" with ``engine_build.state == "building"``,
    # its endpoint is building them; frames follow.
    engine_build: EngineBuild | None = None
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
    # Whether the run wrote a log, as on :class:`WorkflowRun`.
    has_log: bool = False
    # What it ran with, as on :class:`WorkflowRun`.
    compute: WorkflowRunCompute = Field(default_factory=WorkflowRunCompute)
    # The TensorRT engines it built, as on :class:`WorkflowRun`.
    engine_build: EngineBuild | None = None
    error: str | None = None


class WorkflowRunCreated(BaseModel):
    """Run response - the new run id. The run is queued onto its workflow's endpoint and billed
    as that endpoint's uptime while it runs; its usage lands on the run's ``final_micro_usd``
    once it ends (see :class:`WorkflowRun`)."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    run_id: str
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
    # The branch a Workflow's Condition block routed the frame down (a frame and its events say it): ``True`` /
    # ``False``; None on a workflow without one. Filter by it with ``branch=``.
    condition: bool | None = None
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
    # An event's rule values as words, named by the server ("1 crossed", "Lot: 3 inside · car: 4 in view"); None for
    # a record without values. ``rule_values`` keeps PRT's keys.
    values_text: str | None = None
    sources: list[dict[str, Any]] | None = None
    pictograph: dict[str, Any] | None = None
    tag_confidences: Any = None
    measures: Any = None
    extra: dict[str, Any] | None = None
    message: dict[str, Any] | None = None


class RunLogEventName(BaseModel):
    """What a person reads for a log record's trigger: its event's name and kind (``WorkflowLiveEvent``'s)."""

    model_config = ConfigDict(extra="ignore")

    name: str
    kind: str


class RunLogPage(BaseModel):
    """One page of a run's log: its ``records``, and ``next_cursor`` until the last page."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    stream: RunLogStream
    order: RunLogOrder = "asc"
    records: list[RunLogRecord] = Field(default_factory=list)
    next_cursor: str | None = None
    # False on a plan that logs events and stats only (Free): the frames stream is empty.
    log_frames: bool = True
    # An events page: each event's name by its record's ``trigger`` (``{"crossing-0a1b2c3d": {"name": "Crossing ·
    # Gate", "kind": "crossing"}}``) - the run's own graph's, or the live streams' workflow's.
    event_names: dict[str, RunLogEventName] = Field(default_factory=dict)


class RunLogBranchCounts(BaseModel):
    """How a Workflow's Condition block split a stream's records: ``passed`` went down its true branch, ``failed``
    down its false one."""

    model_config = ConfigDict(extra="ignore")

    passed: int = 0
    failed: int = 0


class RunLogStreamSummary(BaseModel):
    """One stream of a run's log: how many records, the first's and the last's time, and - for a workflow with a
    Condition block - how many went down each branch (``condition``; None without one)."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    count: int = 0
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    condition: RunLogBranchCounts | None = None


class RunLogSummary(BaseModel):
    """What a run logged, per stream (``frames`` / ``events`` / ``stats``), counted from the log."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    log_frames: bool = True
    streams: dict[str, RunLogStreamSummary] = Field(default_factory=dict)
