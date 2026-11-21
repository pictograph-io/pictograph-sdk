"""Webhook Pydantic models - outbound webhook endpoints + their deliveries.

Register an endpoint with :meth:`pictograph.resources.webhooks.Webhooks.create`;
Pictograph then POSTs signed events to your URL: ``workflow_run.completed`` /
``workflow_run.failed`` (a run ended), ``workflow_run.events`` (the events a run's
webhook output selects, during a run on a video), ``workflow_stream.events`` (the
same while its workflow streams; both a batch at most every second),
``workflow_stream.ended`` (a stream's end, each line's and zone's final counts), and the
Live log's ``deployment.frames`` (a deployment's frames, a delivery for each second of a
stream), ``capture.saved`` (an image a stream captured, saved into its dataset) and
``session.status`` (an edge stream's state: running, paused, stalled, ended) - each
carrying :class:`~pictograph.models.live_log.LogRecord` records, and sent only to an
endpoint that names them, under its :class:`WebhookFilters`.

Verify each delivery's ``X-Pictograph-Signature`` (``t=<ts>,v1=<hmac>``) with the signing
secret (returned at create, and shown again in the app under Settings → Webhooks), using
HMAC-SHA256 over ``"{t}.{body}"``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

WebhookDeliveryStatus = Literal["pending", "delivered", "failed", "dead_letter"]


class WebhookFilters(BaseModel):
    """What a Live log webhook (``deployment.frames``, ``capture.saved``, ``session.status``) sends - a log
    read's filters: only these deployments' or workflows' records (by name; none named: every one), only
    annotations of these classes and at least this confident (each record carrying those alone; one left with
    none is not sent), every Nth frame. A ``session.status`` record carries no annotations: only the
    deployments and workflows choose it."""

    model_config = ConfigDict(extra="ignore")

    deployments: list[str] | None = Field(
        default=None, description="Only these deployments' records, by name."
    )
    workflows: list[str] | None = Field(
        default=None, description="Only these workflows' records, by name."
    )
    classes: list[str] | None = Field(
        default=None, description="Only annotations of these classes."
    )
    min_confidence: float | None = Field(
        default=None, description="Only annotations at least this confident."
    )
    every: int | None = Field(default=None, description="Every Nth frame of each stream.")


class WebhookEndpoint(BaseModel):
    """A registered outbound webhook destination."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    id: str
    organization_id: str
    url: str
    description: str | None = None
    event_types: list[str] = Field(
        default_factory=list, description="The events it's sent; empty means every event."
    )
    enabled: bool = Field(
        default=True, description="Whether it's sent deliveries; turned off after many failures."
    )
    secret_version: int = Field(default=1, description="Goes up by one at each rotation.")
    secret_prefix: str | None = Field(default=None, description="The secret's first characters.")
    consecutive_failures: int = Field(
        default=0, description="Failed deliveries in a row; 0 after a success."
    )
    disabled_reason: str | None = Field(default=None, description="Why it was turned off.")
    auth_header_names: list[str] | None = Field(
        default=None,
        description="The names of its custom headers (their values are never returned).",
    )
    filters: WebhookFilters | None = Field(
        default=None,
        description="Its Live log webhooks' filters (deployment.frames, capture.saved, session.status).",
    )
    last_delivery_at: datetime | None = Field(
        default=None, description="When it was last sent a delivery."
    )
    created_at: datetime | None = None


class CreatedWebhookEndpoint(BaseModel):
    """An endpoint and its signing secret (from create or a rotation)."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    endpoint: WebhookEndpoint
    secret: str = Field(
        description="The signing secret; the app shows it again under Settings → Webhooks."
    )


class WebhookDelivery(BaseModel):
    """One event's delivery to an endpoint, with its retries."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    id: str
    endpoint_id: str
    organization_id: str
    event_type: str
    delivery_id: str = Field(description="Sent to your endpoint as X-Pictograph-Delivery-Id.")
    status: WebhookDeliveryStatus = Field(
        description="pending (being tried), delivered, failed or dead_letter (out of retries)."
    )
    attempts: int = Field(default=0, description="Tries so far.")
    last_status_code: int | None = Field(default=None, description="What the last try got back.")
    last_error: str | None = Field(default=None, description="The last try's error.")
    next_retry_at: datetime | None = Field(default=None, description="When it's tried next.")
    created_at: datetime | None = None
    delivered_at: datetime | None = Field(default=None, description="When it was delivered.")
