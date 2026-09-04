"""Tests for ``pictograph.resources.webhooks.Webhooks``.

Coverage: create (typed CreatedWebhookEndpoint carrying the one-time secret +
request body), list, get, delete, test (raw result), deliveries (filters), replay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.models.webhook import CreatedWebhookEndpoint, WebhookDelivery, WebhookEndpoint
from pictograph.resources.webhooks import Webhooks

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
_ENDPOINTS = f"{BASE}/api/v1/developer/webhooks/endpoints"
_DELIVERIES = f"{BASE}/api/v1/developer/webhooks/deliveries"
_EVENT_TYPES = f"{BASE}/api/v1/developer/webhooks/event-types"


@pytest.fixture
def transport() -> Transport:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def webhooks(transport: Transport) -> Webhooks:
    return Webhooks(transport)


def _endpoint(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "abcd1234-0000-1111-2222-333344445555",
        "organization_id": "org-1",
        "url": "https://hooks.example.com/x",
        "description": None,
        "event_types": ["workflow_run.completed"],
        "enabled": True,
        "secret_version": 1,
        "secret_prefix": "whsec_ab12",
        "consecutive_failures": 0,
        "disabled_reason": None,
        "last_delivery_at": None,
        "created_at": "2026-06-03T00:00:00Z",
    }
    base.update(overrides)
    return base


def _delivery(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "d-1",
        "endpoint_id": "abcd1234-0000-1111-2222-333344445555",
        "organization_id": "org-1",
        "event_type": "workflow_run.completed",
        "delivery_id": "dl-1",
        "status": "delivered",
        "attempts": 1,
        "last_status_code": 200,
        "last_error": None,
        "next_retry_at": None,
        "created_at": "2026-06-03T00:00:00Z",
        "delivered_at": "2026-06-03T00:00:01Z",
    }
    base.update(overrides)
    return base


def test_create_returns_secret_and_endpoint(httpx_mock: HTTPXMock, webhooks: Webhooks) -> None:
    httpx_mock.add_response(
        method="POST",
        url=_ENDPOINTS,
        json={"endpoint": _endpoint(), "secret": "whsec_live_secret", "message": "shown once"},
    )
    created = webhooks.create("https://hooks.example.com/x", event_types=["workflow_run.completed"])
    assert isinstance(created, CreatedWebhookEndpoint)
    assert created.secret == "whsec_live_secret"  # noqa: S105 - test fixture, not a real secret
    assert created.endpoint.url == "https://hooks.example.com/x"
    body = httpx_mock.get_requests()[-1].read().decode()
    assert "hooks.example.com" in body and "workflow_run.completed" in body


def test_create_sends_auth_headers_and_parses_names(
    httpx_mock: HTTPXMock, webhooks: Webhooks
) -> None:
    """Create forwards custom auth headers; the response carries only names."""
    httpx_mock.add_response(
        method="POST",
        url=_ENDPOINTS,
        json={
            "endpoint": _endpoint(auth_header_names=["Authorization"]),
            "secret": "whsec_x",
            "message": "shown once",
        },
    )
    created = webhooks.create(
        "https://hooks.example.com/x", auth_headers={"Authorization": "Bearer tok"}
    )
    body = httpx_mock.get_requests()[-1].read().decode()
    assert '"auth_headers"' in body and "Bearer tok" in body  # forwarded on the wire
    assert created.endpoint.auth_header_names == ["Authorization"]  # names surfaced back


def test_create_omits_auth_headers_when_not_passed(
    httpx_mock: HTTPXMock, webhooks: Webhooks
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=_ENDPOINTS,
        json={"endpoint": _endpoint(), "secret": "whsec_x", "message": "m"},
    )
    webhooks.create("https://hooks.example.com/x")
    assert "auth_headers" not in httpx_mock.get_requests()[-1].read().decode()


def test_update_sends_auth_headers(httpx_mock: HTTPXMock, webhooks: Webhooks) -> None:
    httpx_mock.add_response(  # get() resolves the endpoint id first
        method="GET",
        url=_ENDPOINTS,
        json={"endpoints": [_endpoint()]},
    )
    httpx_mock.add_response(
        method="PATCH",
        url=f"{_ENDPOINTS}/abcd1234-0000-1111-2222-333344445555",
        json={"endpoint": _endpoint(auth_header_names=["X-Api-Key"])},
    )
    updated = webhooks.update("https://hooks.example.com/x", auth_headers={"X-Api-Key": "k"})
    assert updated.auth_header_names == ["X-Api-Key"]
    assert '"auth_headers"' in httpx_mock.get_requests()[-1].read().decode()


def test_list_returns_typed(httpx_mock: HTTPXMock, webhooks: Webhooks) -> None:
    httpx_mock.add_response(
        method="GET", url=_ENDPOINTS, json={"endpoints": [_endpoint(), _endpoint(id="wh-2")]}
    )
    result = webhooks.list()
    assert len(result) == 2 and all(isinstance(e, WebhookEndpoint) for e in result)


def test_get_returns_typed(httpx_mock: HTTPXMock, webhooks: Webhooks) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_ENDPOINTS}/abcd1234-0000-1111-2222-333344445555",
        json={"endpoint": _endpoint()},
    )
    e = webhooks.get("abcd1234-0000-1111-2222-333344445555")
    assert (
        isinstance(e, WebhookEndpoint)
        and e.id == "abcd1234-0000-1111-2222-333344445555"
        and e.secret_prefix == "whsec_ab12"  # noqa: S105 - fixture value
    )


def test_update_patches_and_returns_typed(httpx_mock: HTTPXMock, webhooks: Webhooks) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{_ENDPOINTS}/abcd1234-0000-1111-2222-333344445555",
        json={"success": True, "endpoint": _endpoint(description="updated", enabled=False)},
    )
    e = webhooks.update(
        "abcd1234-0000-1111-2222-333344445555", description="updated", enabled=False
    )
    assert isinstance(e, WebhookEndpoint) and e.id == "abcd1234-0000-1111-2222-333344445555"
    body = httpx_mock.get_requests()[-1].read().decode().replace(" ", "")
    assert '"description":"updated"' in body and '"enabled":false' in body


def test_update_no_fields_raises(webhooks: Webhooks) -> None:
    with pytest.raises(ValueError):
        webhooks.update("abcd1234-0000-1111-2222-333344445555")


def test_delete(httpx_mock: HTTPXMock, webhooks: Webhooks) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{_ENDPOINTS}/abcd1234-0000-1111-2222-333344445555",
        json={"success": True},
    )
    webhooks.delete("abcd1234-0000-1111-2222-333344445555")
    req = httpx_mock.get_requests()[0]
    assert req.method == "DELETE"
    assert req.url.path.endswith("/endpoints/abcd1234-0000-1111-2222-333344445555")


def test_test_returns_result(httpx_mock: HTTPXMock, webhooks: Webhooks) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{_ENDPOINTS}/abcd1234-0000-1111-2222-333344445555/test",
        json={"success": True, "delivered": True, "status_code": 200, "error": None},
    )
    result = webhooks.test("abcd1234-0000-1111-2222-333344445555")
    assert result["delivered"] is True and result["status_code"] == 200


def test_deliveries_passes_filters(httpx_mock: HTTPXMock, webhooks: Webhooks) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_DELIVERIES}?limit=20&offset=0&endpoint_id=abcd1234-0000-1111-2222-333344445555&status=failed",
        json={"deliveries": [_delivery(status="failed")]},
    )
    result = webhooks.deliveries(
        endpoint="abcd1234-0000-1111-2222-333344445555", status="failed", limit=20
    )
    assert len(result) == 1 and isinstance(result[0], WebhookDelivery)


def test_replay(httpx_mock: HTTPXMock, webhooks: Webhooks) -> None:
    httpx_mock.add_response(method="POST", url=f"{_DELIVERIES}/d-1/replay", json={"success": True})
    webhooks.replay("d-1")
    req = httpx_mock.get_requests()[0]
    assert req.method == "POST"
    assert req.url.path.endswith("/deliveries/d-1/replay")


def test_event_types_returns_list(httpx_mock: HTTPXMock, webhooks: Webhooks) -> None:
    httpx_mock.add_response(
        method="GET",
        url=_EVENT_TYPES,
        json={"event_types": ["workflow_run.completed", "workflow_run.failed"]},
    )
    types = webhooks.event_types()
    assert types == ["workflow_run.completed", "workflow_run.failed"]


def test_rotate_secret_returns_new_secret(httpx_mock: HTTPXMock, webhooks: Webhooks) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{_ENDPOINTS}/abcd1234-0000-1111-2222-333344445555/rotate-secret",
        json={
            "endpoint": _endpoint(secret_version=2),
            "secret": "whsec_rotated",
            "message": "rotated",
        },
    )
    created = webhooks.rotate_secret("abcd1234-0000-1111-2222-333344445555")
    assert isinstance(created, CreatedWebhookEndpoint)
    assert created.secret == "whsec_rotated"  # noqa: S105 - test fixture, not a real secret
    assert created.endpoint.secret_version == 2
