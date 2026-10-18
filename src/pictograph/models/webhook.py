"""Webhook Pydantic models - outbound webhook endpoints + their deliveries.

Register an endpoint with :meth:`pictograph.resources.webhooks.Webhooks.create`;
Pictograph then POSTs signed events (e.g. ``workflow_run.completed``) to your URL.
Verify each delivery's ``X-Pictograph-Signature`` (``t=<ts>,v1=<hmac>``) with the
signing secret returned once at create time, using HMAC-SHA256 over ``"{t}.{body}"``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

WebhookDeliveryStatus = Literal["pending", "delivered", "failed", "dead_letter"]


class WebhookEndpoint(BaseModel):
    """A registered outbound webhook destination."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    id: str
    organization_id: str
    url: str
    description: str | None = None
    event_types: list[str] = Field(default_factory=list)
    enabled: bool = True
    secret_version: int = 1
    secret_prefix: str | None = None
    consecutive_failures: int = 0
    disabled_reason: str | None = None
    #: Names of any custom auth headers configured on this endpoint. The
    #: VALUES are never returned here - set them on create/update, reveal them once
    #: per login in the app. ``None``/empty means no custom headers.
    auth_header_names: list[str] | None = None
    last_delivery_at: datetime | None = None
    created_at: datetime | None = None


class CreatedWebhookEndpoint(BaseModel):
    """Create / rotate response - carries the one-time signing secret."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    endpoint: WebhookEndpoint
    secret: str = Field(description="HMAC signing secret. Shown once - store it securely.")


class WebhookDelivery(BaseModel):
    """One delivery attempt-set for an event to an endpoint (durable, retried)."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    id: str
    endpoint_id: str
    organization_id: str
    event_type: str
    delivery_id: str
    status: WebhookDeliveryStatus
    attempts: int = 0
    last_status_code: int | None = None
    last_error: str | None = None
    next_retry_at: datetime | None = None
    created_at: datetime | None = None
    delivered_at: datetime | None = None
