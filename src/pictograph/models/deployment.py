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
DeploymentStatus = Literal["provisioning", "active", "paused", "failed", "terminated"]


class DeploymentMetrics(BaseModel):
    """The requests the inference gateway has served for a deployment.

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
    # What its uptime has cost so far, in micro-USD.
    accrued_cost_micro_usd: int = 0
    uptime_seconds: int = 0
    # The requests the gateway has served for it.
    metrics: DeploymentMetrics | None = None
    # What it is doing right now - live, starting, waiting for a GPU, asleep - when it is
    # ``active``; None otherwise. Reading it never wakes the deployment.
    runtime: DeploymentRuntime | None = None
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


class DeploymentQuote(BaseModel):
    """What a deployment costs while it is up, before creating it. All amounts are
    already-marked-up micro-USD (1 USD = 1_000_000 µUSD).

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


class DeploymentComputeOptions(BaseModel):
    """What a deployment may run with: each GPU tier (its container and price) and each
    precision its engine is built at, the default first."""

    model_config = ConfigDict(extra="ignore")

    options: list[DeploymentComputeOption] = Field(default_factory=list)
    precisions: list[DeploymentComputePrecision] = Field(default_factory=list)
