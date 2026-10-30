"""Deployment Pydantic models - an org-level inference deployment.

A deployment serves one trained model behind a token-guarded endpoint, on a GPU (a T4, L4 or
A100), on PRT, Pictograph's realtime engine: a TensorRT engine built from the model's ONNX
graph, at the deployment's precision. Created via
:meth:`pictograph.resources.deployments.Deployments.create`, it is ``provisioning`` until its
server is up - about a minute when the model's engine is already built for that GPU and
precision; a model's first deployment on a GPU at a precision builds the engine first, a few
minutes, reported on :attr:`Deployment.engine_build` - and then ``active``. Wait for that with
:meth:`~pictograph.resources.deployments.Deployments.wait_until_active`, then call the endpoint
directly with :class:`pictograph.resources._deployment_client.DeploymentClient`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pictograph.models.engine_build import EngineBuild

# What a deployment REPORTS it runs on. Every deployment runs on a GPU; ``"cpu"`` still parses
# because a deployment made before the GPU-only engine reports it (it cannot be resumed: deploy
# the model again on a GPU).
ComputeType = Literal["cpu", "gpu"]
# The GPU tiers a deployment can be CREATED on (``compute_options().options`` lists them with
# their prices). There is no CPU tier: the engine is TensorRT, which needs a GPU.
DeploymentGpuType = Literal["t4", "l4", "a100"]
# The precision a deployment's TensorRT engine is built at. ``"fp32"`` (the default) serves the
# model as trained; ``"fp16"`` is faster and accuracy-equivalent, not output-identical.
# ``compute_options().precisions`` lists them in the server's words.
DeploymentPrecision = Literal["fp32", "fp16"]
# ``"draft"`` is a workflow's endpoint's alone (no app yet: :attr:`pictograph.models.workflow.WorkflowEndpoint.status`);
# a model's deployment is never in it.
DeploymentStatus = Literal["draft", "provisioning", "active", "paused", "failed", "terminated"]


class DeploymentMetrics(BaseModel):
    """The requests a deployment has served.

    Advisory and fail-soft: zeros when nothing is recorded yet, or the counts cannot be read.
    """

    model_config = ConfigDict(extra="ignore")

    requests: int = 0
    errors: int = 0
    # Mean gateway-side latency of a request, in milliseconds.
    avg_latency_ms: float = 0.0
    # ``errors / requests`` (0.0 with no requests).
    error_rate: float = 0.0
    last_request_at: datetime | None = None


class DeploymentRuntime(BaseModel):
    """What an ``active`` deployment is doing right now, as Pictograph read it at ``checked_at``.

    ``state`` is one of:

    * ``"live"`` - a container is up and serving (``containers`` of them);
    * ``"starting"`` - a container is being created (``creating`` of them): its engine is loading;
    * ``"waiting"`` - a request is waiting and no GPU has been found for it yet, since
      ``waiting_since`` (or a warm minimum has no container yet);
    * ``"asleep"`` - scaled to zero: the next request wakes it, and pays the cold start;
    * ``"unknown"`` - the compute platform did not answer in time.

    Reading it never wakes the deployment: it is read from the platform, never by calling the
    deployment. ``state`` is a ``str``, so a state added later still parses - compare against
    the words above rather than assuming they are all there is.
    """

    model_config = ConfigDict(extra="ignore")

    state: str
    # Containers up and serving, and containers being created (their engine loading).
    containers: int = 0
    creating: int = 0
    # Containers serving its stream sessions - never counted in ``containers`` or ``state``, which
    # are its requests'.
    streams: int = 0
    # Of ``containers``, those running elsewhere in the US because its region had no GPU free for
    # 30 seconds (``Deployment.region_fallback``).
    fallback: int = 0
    # When a request began waiting for a container (``"waiting"``), else None.
    waiting_since: datetime | None = None
    # When Pictograph read this (UTC).
    checked_at: datetime | None = None


class Deployment(BaseModel):
    """A model inference deployment: provisioning, active, paused, failed or terminated."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    id: str
    organization_id: str
    model_id: str
    # The model version it serves: the model's current version when it was created or last resumed.
    version_id: str | None = None
    name: str
    # Its README (Markdown), as written in its settings or with ``deployments.update``; None
    # when it has none.
    readme: str | None = None
    status: DeploymentStatus
    # Where it runs - ``"us-east"`` or ``"us-west"`` (``compute_options().regions``): its endpoint
    # takes requests there and its containers run there. Set at create, never changed.
    region: str | None = None
    # ``"gpu"`` for every deployment; ``"cpu"`` only on one made before the GPU-only engine.
    compute_type: ComputeType = "gpu"
    # The tier it runs on. A ``str``, not ``DeploymentGpuType``: a deployment made before
    # a tier was retired still reports it, and a tier added later must not break a read.
    gpu_type: str | None = None
    # The precision its TensorRT engine is built at (``"fp32"`` or ``"fp16"``). A ``str`` for
    # the same reason as ``gpu_type``.
    precision: str = "fp32"
    min_containers: int
    max_containers: int
    scaledown_window: int
    # Its request ceiling: requests a minute past which its endpoint answers 429 (``RateLimitError``,
    # with ``Retry-After``). None - the default - is none: what it costs is bounded by
    # ``max_containers`` and your budget. Set with ``create(rate_limit_rpm=...)`` or ``update``.
    rate_limit_rpm: int | None = None
    endpoint_url: str | None = None
    auth_token_prefix: str | None = None
    inference_config: dict[str, Any] = Field(default_factory=dict)
    # The TensorRT engine build a provisioning deployment waits on: ``state`` ``"building"``
    # while the model's first engine for this GPU and precision is built (with an estimate when
    # there is history), ``"built"`` once it is. None when the engine was already stored.
    engine_build: EngineBuild | None = None
    # Why a deployment is ``failed`` (its provisioning could not start, or its engine could
    # not be built); None otherwise.
    error_message: str | None = None
    # One container's price per minute, in micro-USD.
    cost_rate_per_min: int = 0
    # What it costs per hour while it is up, in micro-USD: its warm containers' (one
    # container's when it scales to zero).
    cost_per_hour_micro_usd: int | None = None
    # What it costs per hour with every container up to ``max_containers`` running.
    max_cost_per_hour_micro_usd: int | None = None
    # A GiB of its requests' data transfer (each image sent through its endpoint and its
    # answer), billed on top of the uptime.
    transfer_per_gib_micro_usd: int | None = None
    # What it has cost so far, in micro-USD: its uptime and its data transfer.
    accrued_cost_micro_usd: int = 0
    uptime_seconds: int = 0
    # Its data transfer so far: the bytes its answered requests sent through its endpoint (each
    # image and its answer), and their part of ``accrued_cost_micro_usd``. Billed per GiB
    # (``quote().transfer_per_gib_micro_usd``), a few minutes behind the requests.
    transfer_bytes: int = 0
    transfer_cost_micro_usd: int = 0
    # The requests the gateway has served for it.
    metrics: DeploymentMetrics | None = None
    # What it is doing right now - live, starting, waiting for a GPU, asleep - when it is
    # ``active``; None otherwise. Reading it never wakes the deployment.
    runtime: DeploymentRuntime | None = None
    # The streams it runs at once: its ``max_containers`` times the stream sessions each container
    # holds. 0 for a deployment made before streams, until it is paused and resumed.
    stream_limit: int = 0
    # Whether a request or a stream that finds no GPU free in its region for 30 seconds is also
    # offered the rest of the US (and served from wherever a GPU is found first): every deployment
    # has this once it is next deployed.
    region_fallback: bool = False
    # Whether it has a log now: its current deployment's (each deploy logs anew; a pause ends it).
    # None from a server of before 1.69.164's.
    logged: bool | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    paused_at: datetime | None = None


class CreatedDeployment(BaseModel):
    """Create response - carries the one-time plaintext bearer token."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    deployment: Deployment
    auth_token: str = Field(
        description=(
            "Per-deployment bearer token. The API returns it only here - store it "
            "securely (a member can reveal it again from the app's deployment settings)."
        )
    )


# Where a stream's video comes from: ``"whip"`` - you send it (a browser's camera, OBS, GStreamer,
# any WHIP client) - or a camera the deployment connects to: ``"rtsp"`` (``rtsp://`` / ``rtsps://``)
# or ``"whep"`` (a WebRTC camera's ``https://`` WHEP URL).
StreamKind = Literal["whip", "rtsp", "whep"]


class StreamLatency(BaseModel):
    """How long one frame of a stream took, in milliseconds (None when not reported)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    # From the frame's capture to its result, on the camera's clock.
    total: float | None = None
    # From the frame's arrival at the deployment to its result.
    engine: float | None = None
    # The model's share of ``engine``.
    inference: float | None = None


class StreamPaths(BaseModel):
    """A stream's URL paths, on its deployment's gateway (the host of its ``endpoint_url``)."""

    model_config = ConfigDict(extra="ignore")

    session: str
    whep: str
    overlay: str
    results: str
    # Only a ``"whip"`` stream has one: where its video is sent.
    whip: str | None = None


class OpenedStream(BaseModel):
    """What opening a stream answers (``POST {sessions}``)."""

    model_config = ConfigDict(extra="ignore")

    session: str
    kind: str
    overlay: bool = True
    # The streams the deployment runs at once.
    stream_limit: int = 0
    # Seconds a stream lasts with nothing reading or sending.
    idle_timeout_s: int = 0
    opened_at: datetime | None = None
    # When the stream ends on its own (four hours after it opened).
    expires_at: datetime | None = None
    # The ICE servers a WebRTC client connects with.
    ice_servers: list[dict[str, Any]] = Field(default_factory=list)
    # Whether it runs elsewhere in the US: its deployment's region had no GPU free for 30 seconds.
    fallback: bool = False
    paths: StreamPaths


class StreamState(BaseModel):
    """A stream's state, as its deployment reports it."""

    model_config = ConfigDict(extra="ignore")

    session: str
    kind: str
    opened_at: datetime | None = None
    expires_at: datetime | None = None
    # Whether it runs elsewhere in the US: its deployment's region had no GPU free for 30 seconds.
    fallback: bool = False
    # The engine's own account of the stream: ``status``, ``uptime_s``, ``frame_messages``, the video
    # it receives (``whip``), the people watching it (``whep``) and ``overlay``.
    engine: dict[str, Any] = Field(default_factory=dict)


class DeploymentQuote(BaseModel):
    """What a deployment costs while it is up, before creating it. All amounts are
    the prices you pay, in micro-USD (1 USD = 1_000_000 µUSD).

    ``min_containers`` containers stay up and are billed whether or not requests arrive; up to
    ``max_containers`` run while load needs them. Scale to zero (``min_containers=0``) bills
    only while a container is up.
    """

    model_config = ConfigDict(extra="ignore")

    # One container's price per minute.
    rate_per_min_micro_usd: int
    # One container's price per hour.
    container_cost_per_hour_micro_usd: int | None = None
    # Its warm cost per hour and per day: the containers kept up (one container's when it
    # scales to zero).
    cost_per_hour_micro_usd: int
    cost_per_day_micro_usd: int
    # The containers kept up, exactly: 0 when it scales to zero.
    warm_cost_per_hour_micro_usd: int | None = None
    warm_cost_per_day_micro_usd: int | None = None
    # Every container up to ``max_containers`` running.
    max_cost_per_hour_micro_usd: int | None = None
    # A GiB of its requests' data transfer (each image sent and its answer), on top of the
    # uptime - the same in every region and on every GPU.
    transfer_per_gib_micro_usd: int | None = None
    scale_to_zero: bool
    billing_note: str


class DeploymentComputeOption(BaseModel):
    """A GPU tier a deployment runs on: its container and its price (one container)."""

    model_config = ConfigDict(extra="ignore")

    key: str
    label: str
    compute_type: ComputeType = "gpu"
    gpu_type: str | None = None  # a str for the same reason as ``Deployment.gpu_type``
    is_gpu: bool = True
    description: str = ""
    vram_gb: int | None = None  # the GPU's memory in GB (16 on a T4)
    # The container's CPU cores and memory beside its GPU.
    cpu_cores: float | None = None
    memory_gib: float | None = None
    rate_per_min_micro_usd: int
    cost_per_hour_micro_usd: int | None = None


class DeploymentComputePrecision(BaseModel):
    """A precision a deployment's TensorRT engine is built at, in the server's words."""

    model_config = ConfigDict(extra="ignore")

    # ``"fp32"`` or ``"fp16"`` - a string, so a precision added later never breaks a read.
    precision: str
    label: str
    description: str = ""


class DeploymentRegion(BaseModel):
    """A region a deployment can run in, in the server's words."""

    model_config = ConfigDict(extra="ignore")

    id: str  # ``"us-east"`` / ``"us-west"`` - what ``create(region=...)`` takes
    label: str  # ``"US East"``
    location: str = ""  # where its requests enter: ``"Virginia"``, ``"Oregon"``
    # Its gateway's ping, reached as a request to a deployment there is: time it to find the region nearest you (the
    # app does). None when the server has no gateway host configured.
    ping_url: str | None = None


class DeploymentComputeOptions(BaseModel):
    """What a deployment may run with: the regions (and the one these prices are for), each GPU
    tier (its container and price there) and each precision its engine is built at, the default
    first."""

    model_config = ConfigDict(extra="ignore")

    region: str | None = None
    regions: list[DeploymentRegion] = Field(default_factory=list)
    options: list[DeploymentComputeOption] = Field(default_factory=list)
    precisions: list[DeploymentComputePrecision] = Field(default_factory=list)
    # A GiB of a deployment's data transfer, in micro-USD: the same on every tier and region.
    transfer_per_gib_micro_usd: int | None = None
