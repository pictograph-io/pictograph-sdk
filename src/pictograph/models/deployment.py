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
DeploymentGpuType = Literal["t4", "l4", "a10g", "a100"]
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
    gpu_type: DeploymentGpuType | None = None
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
        description="Per-deployment bearer token. Shown once - store it securely."
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
    gpu_type: DeploymentGpuType | None = None
    is_gpu: bool
    description: str = ""
    rate_per_min_micro_usd: int
