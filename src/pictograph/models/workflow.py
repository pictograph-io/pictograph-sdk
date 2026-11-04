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

from pydantic import BaseModel, ConfigDict, Field, model_validator

# A run's engine builds: the record a provisioning deployment carries too.
from pictograph.models.deployment import DeploymentRegion, DeploymentRuntime, DeploymentStatus
from pictograph.models.engine_build import EngineBuild

WorkflowStatus = Literal["draft", "ready", "archived"]
WorkflowRunStatus = Literal["queued", "processing", "completed", "error", "cancelled"]
# Which runs a listing keeps: every run, a file run (a video, an image, a dataset), or a stream session.
WorkflowRunsKind = Literal["all", "file", "stream"]


class WorkflowCompute(BaseModel):
    """What a workflow's runs run on: its GPU.

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


class WorkflowRunSource(BaseModel):
    """What a run ran over: its input's kind (``"video"``, ``"image"``, ``"dataset"``, ``"export"``) - or ``"stream"``, a
    stream session on the workflow's endpoint (each one is a run of it: the app's Run view, :meth:`connect`, an RTSP
    camera), with how its video came (``via``): ``"whip"`` (a browser's camera or a video played as one, any WHIP
    publisher), ``"rtsp"`` (a camera the engine pulled) or ``"whep"`` (a WebRTC sender it pulled). Strings, so a kind
    the platform adds later reads as itself.

    A file run also says its input: ``name`` (a file's own, a dataset's, an export's) and ``input``, the input as a run
    names it (:class:`WorkflowRunInput`'s fields) - give it to :meth:`~pictograph.resources.workflows.Workflows.run` to
    run it again. Both ``None`` for a stream (and a run from before inputs were recorded)."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    kind: str
    via: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None


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
    """An event a workflow offers - what a run's Event capture and webhook pick from (by ``key``), and what its log's
    events are named by (their ``trigger``) - named as the app names them. Its Passive capture is one too (kind
    ``"passive"``)."""

    model_config = ConfigDict(extra="ignore")

    # Its trigger in the workflow's log (each event record's ``trigger``): ``"<kind>-<8 hex>"``.
    trigger: str
    # What a run's outputs pick it by: ``"crossing:<measure node>:<geometry>"``, ``"entry:..."``, ``"exit:..."``,
    # ``"dwell:..."``, ``"appearance:<class>"`` / ``"appearance:*"``, ``"condition"``; ``"passive"``.
    key: str
    # ``"crossing"``, ``"entry"``, ``"exit"``, ``"dwell"``, ``"appearance"``, ``"condition"`` or ``"passive"``.
    kind: str
    # What a person reads (``"Crossing · Gate"``, ``"Appearance · person"``, ``"Passive capture"``).
    name: str


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
    ``"tag"``) - and every event it offers (``events``: what a run's Event capture and webhook pick from, and what
    its log's events are named by; its Passive capture too)."""

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
    # Seconds its last container stays up after its last run or stream ends.
    scaledown_window: int = 60
    # Where its streams open: its ``/sessions`` on its region's endpoint - it takes streams, never a predict
    # (connect with :meth:`~pictograph.resources.workflows.Workflows.connect`).
    endpoint_url: str | None = None
    auth_token_prefix: str | None = None
    error_message: str | None = None
    engine_build: EngineBuild | None = None
    # The streams it runs at once (0 until it is deployed): its maximum containers x the streams each holds.
    stream_limit: int = 0
    # Its file and dataset runs at once: one a container, up to its maximum containers.
    run_limit: int = 0
    # Whether a stream or a run that finds no GPU free in its region for 30 seconds is also offered
    # the rest of the US (and runs wherever a GPU is found first): every endpoint has this once it is
    # next deployed.
    region_fallback: bool = False
    # Whether it takes streams as saved (False: it runs its files only - ``error_message`` says why).
    streams: bool = True
    # Whether it has ever been deployed (its region is fixed from then on).
    deployed: bool = False
    # Whether its live streams have a log now: its current deployment's (a draft never deployed, or a
    # paused endpoint, has none - ``live_log`` raises ``no_log``). None from a server of before 1.69.164's.
    logged: bool | None = None
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


class WorkflowOutputFolder(BaseModel):
    """Where a run's output saves: a folder of one of your organization's datasets."""

    model_config = ConfigDict(extra="forbid")

    #: The dataset, by name or id. None: the dataset the run runs on (a dataset or an export run).
    dataset: str | None = None
    #: One folder of it, e.g. ``"/captures"`` (None: its top).
    directory: str | None = None
    #: Give each run a folder of its own inside this one, named after the workflow and the run.
    new_folder_each_run: bool = False


class WorkflowAnnotationsOutput(WorkflowOutputFolder):
    """Write annotations: each frame (a video's), each image (a dataset's, an export's) or the image a run runs on,
    with its annotations, into a dataset folder - at most 2,000 a run."""

    #: Only frames or images with an annotation or a classifier's label.
    only_with_detections: bool = True
    #: A video's: at most one frame every this many seconds of it (None: every frame it reads).
    min_interval_s: float | None = Field(None, gt=0, le=3600)


class WorkflowPassiveCaptureOutput(WorkflowOutputFolder):
    """Passive capture: a frame every so many seconds (a video's, a stream's), with or without detections."""

    #: Seconds between frames, 1 to 3600.
    every_s: float = Field(10.0, ge=1, le=3600)


class WorkflowEventCaptureOutput(WorkflowOutputFolder):
    """Event capture: the frame of each event picked (a video's, a stream's), with its annotations."""

    #: The events whose frames it saves, by their keys (:attr:`WorkflowLiveEvent.key`).
    events: list[str] = Field(..., min_length=1, max_length=64)
    #: At most one capture every this many seconds - passive ones included, an event's first (at least 1).
    min_interval_s: float = Field(1.0, ge=1, le=3600)


class WorkflowWebhookOutput(BaseModel):
    """Send to a webhook: the run's events as they happen (a video's, a stream's) and its end."""

    model_config = ConfigDict(extra="forbid")

    #: The webhook endpoint's id (``client.webhooks``).
    endpoint: str
    #: The events it is sent while the run goes, by their keys (a video's or a stream's; none: its end alone).
    events: list[str] = Field(default_factory=list, max_length=64)


class WorkflowRunOutputs(BaseModel):
    """What a run writes beside its log, given when it starts - a file run's
    :meth:`~pictograph.resources.workflows.Workflows.run`, a stream's ``open_stream``. A workflow's Sink block holds no
    settings: a run's outputs are its own. Absent, or every output off: its log alone.

    Which inputs take which: Write annotated video - a video, a dataset, an export, a stream (saved as it streams, its
    video on its run once it ends: ``WorkflowRun.recording``); Write annotations - an image, a video, a dataset, an
    export (a stream writes its frames through its captures); Event and Passive capture - a video, a stream; a webhook
    - any (its events from a video or a stream). ``extra="forbid"``: a mistyped field is refused here.
    """

    model_config = ConfigDict(extra="forbid")

    #: Write the run as a video, drawn as its blocks draw it - a stream's saved as it streams, on its run once it ends.
    annotated_video: bool = False
    annotations: WorkflowAnnotationsOutput | None = None
    passive_capture: WorkflowPassiveCaptureOutput | None = None
    event_capture: WorkflowEventCaptureOutput | None = None
    webhook: WorkflowWebhookOutput | None = None


class WorkflowWrittenFolder(BaseModel):
    """A folder a run writes into, as its record says: its dataset by id with its name, the folder chosen and the
    folder it writes (the chosen one, or the run's own inside it)."""

    model_config = ConfigDict(extra="ignore")

    dataset: str
    dataset_name: str = ""
    directory: str = "/"
    new_folder_each_run: bool = False
    folder: str = "/"


class WorkflowWrittenAnnotations(WorkflowWrittenFolder):
    only_with_detections: bool = True
    min_interval_s: float | None = None


class WorkflowWrittenPassiveCapture(WorkflowWrittenFolder):
    every_s: float = 10.0


class WorkflowWrittenEventCapture(WorkflowWrittenFolder):
    events: list[str] = Field(default_factory=list)
    min_interval_s: float = 1.0


class WorkflowWrittenWebhook(BaseModel):
    model_config = ConfigDict(extra="ignore")

    endpoint: str
    events: list[str] = Field(default_factory=list)


class WorkflowRunWrites(BaseModel):
    """What a run writes beside its log, as its record says (its outputs as resolved when it started)."""

    model_config = ConfigDict(extra="ignore")

    annotated_video: bool = False
    annotations: WorkflowWrittenAnnotations | None = None
    passive_capture: WorkflowWrittenPassiveCapture | None = None
    event_capture: WorkflowWrittenEventCapture | None = None
    webhook: WorkflowWrittenWebhook | None = None


class WorkflowRunLiveView(BaseModel):
    """A file run's live view - a video's, a dataset's or an export's run: its frames shown as it processes them, drawn
    in the workflow's look. ``stream`` and ``overlay``: its Stream and Overlay as it started (``run(stream=,
    overlay=)``; switched while it runs on the stream :meth:`~pictograph.resources.workflows.Workflows.watch_run`
    answers). ``session``: its session on the workflow's endpoint, once the run is dispatched (None while it is
    queued)."""

    model_config = ConfigDict(extra="ignore")

    stream: bool = False
    overlay: bool = True
    session: str | None = None


class WorkflowRunRecording(BaseModel):
    """A stream run's annotated video - a stream that asked for one (its outputs' ``annotated_video``): saved in parts
    as it streamed, made into one video on its run once it ended (its ``annotated_video`` artifact). ``state``:
    ``"saved"`` (the whole of it), ``"partial"`` (the video up to where it stopped: its container was lost, or a part
    could not be saved - ``detail`` says which) or ``"none"`` (nothing could be made of it; ``detail`` says why).
    ``bytes`` and ``parts``: the video's size and the parts it was made of; ``at``: when it was made."""

    model_config = ConfigDict(extra="ignore")

    state: Literal["saved", "partial", "none"]
    bytes: int | None = None
    parts: int | None = None
    at: datetime | None = None
    detail: str | None = None


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
    # could not be opened (its ``warnings`` say so), or an older run.
    has_log: bool = False
    # What it ran with: its GPU and its Model blocks' precision.
    compute: WorkflowRunCompute = Field(default_factory=WorkflowRunCompute)
    # The TensorRT engines its endpoint built for it - a model's first deploy on its GPU (None:
    # every engine was stored). While a run is "queued" with ``engine_build.state == "building"``,
    # its endpoint is building them; frames follow.
    engine_build: EngineBuild | None = None
    # What it ran over (None: unknown) - a stream session is a run too, its session's id beside it.
    source: WorkflowRunSource | None = None
    stream_session: str | None = None
    # What it writes beside its log - its outputs as resolved when it started (None: its log alone).
    outputs: WorkflowRunWrites | None = None
    # Its live view - a video's, a dataset's or an export's run (None: an image's, a stream's - a stream is its own).
    live_view: WorkflowRunLiveView | None = None
    # A stream run's annotated video, once its parts are made into one video (None while it streams, until then, and
    # for a run that asked for none).
    recording: WorkflowRunRecording | None = None
    # A single image's answer: its frame (``{"frames": [{"frame", "t", "annotations"}]}``); None for any other run.
    result: dict[str, Any] | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None


class WorkflowRunSummary(BaseModel):
    """A run as ``client.workflows.list_runs`` lists it.

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
    # What it ran over, and a stream's session, as on :class:`WorkflowRun`.
    source: WorkflowRunSource | None = None
    stream_session: str | None = None
    # What it writes beside its log, as on :class:`WorkflowRun`.
    outputs: WorkflowRunWrites | None = None
    # Its live view, as on :class:`WorkflowRun`.
    live_view: WorkflowRunLiveView | None = None
    # A stream run's annotated video, as on :class:`WorkflowRun`.
    recording: WorkflowRunRecording | None = None
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


WorkflowRunInputKind = Literal["video", "image", "dataset", "export"]
#: The old name of :data:`WorkflowRunInputKind`.
WorkflowInvokeKind = WorkflowRunInputKind


class WorkflowRunInput(BaseModel):
    """What a run of a workflow runs on, for :meth:`~pictograph.resources.workflows.Workflows.run`.

    A workflow's Source block holds no input: every run names its own. Discriminated on
    ``kind``, the input named by the kind's own field:

    - ``"video"`` / ``"image"``: ``video`` / ``image``, the handle
      :meth:`~pictograph.resources.workflows.Workflows.upload_source` returns, OR ``url``: a
      public ``http(s)`` link to the file itself, which Pictograph downloads into your
      organization's storage before the run starts (a video up to 1 GB, an image up to
      25 MB). A video also takes ``sample_fps`` (frames read a second; absent, every frame at the
      video's own rate) and
      ``duration_seconds`` (its length: the run's frame total until it reads the video).
    - ``"dataset"``: ``dataset`` - a dataset in your organization, by name or id -
      optionally one ``directory`` of it (absent: the whole dataset).
    - ``"export"``: ``export`` - an export in your organization, by name or id: the
      images it captured.

    ``extra="forbid"``: a mistyped field is refused here, never dropped. ``gcs_uri`` (a
    handle's old name) and ``directory_path`` (``directory``'s) still read.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    kind: WorkflowRunInputKind
    video: str | None = None
    image: str | None = None
    url: str | None = None
    dataset: str | None = None
    directory: str | None = None
    export: str | None = None
    sample_fps: float | None = None
    duration_seconds: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _old_names(cls, data: Any) -> Any:
        """``gcs_uri`` read as the kind's own field, ``directory_path`` as ``directory``."""
        if not isinstance(data, dict) or not ({"gcs_uri", "directory_path"} & set(data)):
            return data
        out = dict(data)
        if "gcs_uri" in out:
            kind = out.get("kind")
            if kind not in ("video", "image"):
                raise ValueError("gcs_uri is a video's or an image's handle: use video= or image=")
            if out.get(kind) is not None:
                raise ValueError(f"gcs_uri is {kind}'s old name: give one of them")
            out[kind] = out.pop("gcs_uri")
        if "directory_path" in out:
            if out.get("directory") is not None:
                raise ValueError("directory_path is directory's old name: give one of them")
            out["directory"] = out.pop("directory_path")
        return out

    @property
    def handle(self) -> str | None:
        """A video's or an image's upload handle (``None`` for a link or another kind)."""
        return self.video if self.kind == "video" else self.image if self.kind == "image" else None


#: The old name of :class:`WorkflowRunInput`.
WorkflowInvokeSource = WorkflowRunInput


class WorkflowUploadInfo(BaseModel):
    """Response from the workflow upload-url endpoint: a signed PUT URL, and the file's handle once
    uploaded - under its kind's own name (``video`` / ``image``, what a run's input names), and as
    ``gcs_uri`` (its old name)."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    upload_url: str
    gcs_uri: str
    video: str | None = None
    image: str | None = None

    @property
    def handle(self) -> str:
        """The file's handle: what :class:`WorkflowRunInput` names as ``video`` / ``image``."""
        return self.video or self.image or self.gcs_uri


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
    # a record without values. ``rule_values`` keeps the engine's keys.
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
