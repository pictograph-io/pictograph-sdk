"""Async Webhooks resource - register outbound sinks + inspect / replay deliveries.

Async twin of :class:`pictograph.resources.webhooks.Webhooks`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pictograph.aio.resources import _resolve
from pictograph.models.webhook import (
    CreatedWebhookEndpoint,
    WebhookDelivery,
    WebhookDeliveryStatus,
    WebhookEndpoint,
)
from pictograph.resources._base import AsyncResource

if TYPE_CHECKING:
    from collections.abc import Sequence

_ENDPOINTS = "/api/v1/developer/webhooks/endpoints"
_DELIVERIES = "/api/v1/developer/webhooks/deliveries"
_EVENT_TYPES = "/api/v1/developer/webhooks/event-types"


class AsyncWebhooks(AsyncResource):
    """Manage outbound webhook endpoints and their delivery log (async)."""

    async def create(
        self,
        url: str,
        *,
        description: str | None = None,
        event_types: list[str] | None = None,
        auth_headers: dict[str, str] | None = None,
    ) -> CreatedWebhookEndpoint:
        """Register an endpoint. Returns the signing secret ONCE - store it securely.

        ``url`` must be https and publicly routable (private / loopback /
        cloud-metadata targets are rejected).

        ``auth_headers`` are optional custom request headers (e.g.
        ``{"Authorization": "Bearer …"}``) sent on every delivery so the endpoint can
        sit behind your own gateway. Stored encrypted; never returned by
        :meth:`list` / :meth:`get` (only their names are, as ``auth_header_names``).
        """
        body: dict[str, Any] = {"url": url}
        if description is not None:
            body["description"] = description
        if event_types is not None:
            body["event_types"] = event_types
        if auth_headers is not None:
            body["auth_headers"] = auth_headers
        response = await self._transport.request("POST", _ENDPOINTS, json=body)
        return self._parse(CreatedWebhookEndpoint, response)

    async def event_types(self) -> list[str]:
        """The canonical event types you can subscribe an endpoint to. An endpoint
        with an EMPTY subscription receives all of them (including ones added later)."""
        response = await self._transport.request("GET", _EVENT_TYPES)
        types: list[str] = response.get("event_types", [])
        return types

    async def list(self) -> Sequence[WebhookEndpoint]:
        """List every webhook endpoint in your organization."""
        response = await self._transport.request("GET", _ENDPOINTS)
        return self._parse_list(WebhookEndpoint, response.get("endpoints", []))

    async def get(self, endpoint: str) -> WebhookEndpoint:
        """Fetch a single endpoint by its registered URL (an id also works)."""
        endpoint_id = await _resolve.webhook_endpoint_id(self._transport, endpoint)
        response = await self._transport.request("GET", f"{_ENDPOINTS}/{endpoint_id}")
        return self._parse(WebhookEndpoint, response["endpoint"])

    async def update(
        self,
        endpoint: str,
        *,
        url: str | None = None,
        description: str | None = None,
        event_types: Sequence[str] | None = None,
        enabled: bool | None = None,
        auth_headers: dict[str, str] | None = None,
    ) -> WebhookEndpoint:
        """Update an endpoint's url / description / event_types / enabled state.

        ``auth_headers`` replaces the custom headers wholesale; pass ``{}`` to clear
        them, omit to leave them unchanged.
        """
        body: dict[str, Any] = {}
        if url is not None:
            body["url"] = url
        if description is not None:
            body["description"] = description
        if event_types is not None:
            body["event_types"] = list(event_types)
        if enabled is not None:
            body["enabled"] = enabled
        if auth_headers is not None:
            body["auth_headers"] = auth_headers
        if not body:
            raise ValueError("update() requires at least one field to change")
        endpoint_id = await _resolve.webhook_endpoint_id(self._transport, endpoint)
        response = await self._transport.request("PATCH", f"{_ENDPOINTS}/{endpoint_id}", json=body)
        return self._parse(WebhookEndpoint, response["endpoint"])

    async def delete(self, endpoint: str) -> None:
        """Delete an endpoint, by its registered URL, and its delivery history."""
        endpoint_id = await _resolve.webhook_endpoint_id(self._transport, endpoint)
        await self._transport.request("DELETE", f"{_ENDPOINTS}/{endpoint_id}")

    async def test(self, endpoint: str) -> dict[str, Any]:
        """Send a synthetic signed test event; returns the immediate delivery result."""
        endpoint_id = await _resolve.webhook_endpoint_id(self._transport, endpoint)
        result: dict[str, Any] = await self._transport.request(
            "POST", f"{_ENDPOINTS}/{endpoint_id}/test"
        )
        return result

    async def rotate_secret(self, endpoint: str) -> CreatedWebhookEndpoint:
        """Mint a new signing secret for an endpoint. Returns the new secret ONCE -
        store it securely. The previous secret stays valid during a grace window
        (deliveries carry both signatures), so you can swap receivers with no downtime."""
        endpoint_id = await _resolve.webhook_endpoint_id(self._transport, endpoint)
        response = await self._transport.request(
            "POST", f"{_ENDPOINTS}/{endpoint_id}/rotate-secret"
        )
        return self._parse(CreatedWebhookEndpoint, response)

    async def deliveries(
        self,
        *,
        endpoint: str | None = None,
        status: WebhookDeliveryStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[WebhookDelivery]:
        """List webhook deliveries, optionally filtered by endpoint and/or status."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if endpoint is not None:
            params["endpoint_id"] = await _resolve.webhook_endpoint_id(self._transport, endpoint)
        if status is not None:
            params["status"] = status
        response = await self._transport.request("GET", _DELIVERIES, params=params)
        return self._parse_list(WebhookDelivery, response.get("deliveries", []))

    async def replay(self, delivery_id: str) -> None:
        """Re-queue a failed / dead-letter delivery with a fresh retry budget."""
        await self._transport.request("POST", f"{_DELIVERIES}/{delivery_id}/replay")
