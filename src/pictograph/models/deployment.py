"""Deployment Pydantic models - an org-level inference deployment.

A deployment serves one trained model behind a token-guarded endpoint. Created
via :meth:`pictograph.resources.deployments.Deployments.create`; call it
directly with :class:`pictograph.resources._deployment_client.DeploymentClient`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ComputeType = Literal["cpu", "gpu"]
# The GPU tiers a deployment can be CREATED on (``compute_options()`` lists them with
# their prices). A10G was retired from deployments: L4 replaces it at that tier.
DeploymentGpuType = Literal["t4", "l4", "a100"]
DeploymentStatus = Literal["provisioning", "active", "paused", "failed", "terminated"]


class Deployment(BaseModel):
    """A live (or provisioning) model inference deployment."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    id: str
    organization_id: str
    model_id: str
    name: str
    status: DeploymentStatus
    compute_type: ComputeType
    # The tier it runs on. A ``str``, not ``DeploymentGpuType``: a deployment made before
    # a tier was retired still reports it, and a tier added later must not break a read.
    gpu_type: str | None = None
    min_containers: int
    max_containers: int
    scaledown_window: int
    endpoint_url: str | None = None
    auth_token_prefix: str | None = None
    inference_config: dict[str, Any] = Field(default_factory=dict)
    cost_rate_per_min: int = 0
    cost_per_hour: int | None = None
    accrued_cost_credits: int = 0
    uptime_seconds: int = 0
    created_at: datetime | None = None
    started_at: datetime | None = None


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
    """Cost quote for a deployment, before creating it. All amounts are
    already-marked-up micro-USD (1 USD = 1_000_000 µUSD)."""

    model_config = ConfigDict(extra="ignore")

    rate_per_min_micro_usd: int
    cost_per_hour_micro_usd: int
    cost_per_day_micro_usd: int
    scale_to_zero: bool
    billing_note: str


class DeploymentComputeOption(BaseModel):
    """A selectable deployment compute tier + its per-minute rate (one warm container)."""

    model_config = ConfigDict(extra="ignore")

    key: str
    label: str
    compute_type: ComputeType
    gpu_type: str | None = None  # a str for the same reason as ``Deployment.gpu_type``
    is_gpu: bool
    description: str = ""
    vram_gb: int | None = None  # a GPU's memory in GB (16 on a T4); None on CPU
    rate_per_min_micro_usd: int
