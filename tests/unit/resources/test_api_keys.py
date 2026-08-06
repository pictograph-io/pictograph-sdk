"""Tests for ``pictograph.resources.api_keys.ApiKeys``."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import ForbiddenError, NotFoundError, ValidationError
from pictograph.models.api_key import ApiKey, CreatedApiKey
from pictograph.resources.api_keys import ApiKeys

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"


@pytest.fixture
def transport() -> Transport:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def api_keys(transport: Transport) -> ApiKeys:
    return ApiKeys(transport)


def _key_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "key-uuid-1",
        "organization_id": "0a111111-2222-3333-4444-555566667777",
        "name": "Production Key",
        "key_prefix": "pk_live_abcd",
        "role": "member",
        "rate_limit": 5000,
        "is_active": True,
        "last_used_at": "2026-01-01T12:00:00Z",
        "expires_at": None,
        "created_at": "2025-12-01T00:00:00Z",
    }
    base.update(overrides)
    return base


# ───────────── list ─────────────


def test_list_without_org_id_omits_query_param(httpx_mock: HTTPXMock, api_keys: ApiKeys) -> None:
    # Defaults to backend's API-key auth context resolution.
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/api-keys/",
        json={"success": True, "api_keys": [_key_payload()]},
    )
    result = api_keys.list()
    assert len(result) == 1
    assert isinstance(result[0], ApiKey)
    sent = httpx_mock.get_request()
    assert sent is not None
    assert "organization_id" not in str(sent.url)


def test_list_with_explicit_org_id_passes_param(httpx_mock: HTTPXMock, api_keys: ApiKeys) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/api-keys/?organization_id=0a111111-2222-3333-4444-555566667777",
        json={"api_keys": []},
    )
    api_keys.list(organization="0a111111-2222-3333-4444-555566667777")
    sent = httpx_mock.get_request()
    assert sent is not None
    assert "organization_id=0a111111-2222-3333-4444-555566667777" in str(sent.url)


def test_list_empty_result(httpx_mock: HTTPXMock, api_keys: ApiKeys) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/api-keys/",
        json={"api_keys": []},
    )
    assert api_keys.list() == []


def test_list_missing_api_keys_field_returns_empty(
    httpx_mock: HTTPXMock, api_keys: ApiKeys
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/api-keys/",
        json={"success": True},
    )
    assert api_keys.list() == []


# ───────────── create ─────────────


def test_create_returns_typed_created_api_key_with_secret(
    httpx_mock: HTTPXMock, api_keys: ApiKeys
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/api-keys/",
        status_code=201,
        json={
            "success": True,
            "api_key": "pk_live_THE_FULL_SECRET",
            "key_id": "key-uuid-1",
            "key_prefix": "pk_live_THE_",
            "name": "CI Pipeline",
            "role": "member",
            "rate_limit": 5000,
            "expires_at": None,
            "created_at": "2026-01-01T00:00:00Z",
            "warning": "Store this API key securely. It will not be shown again.",
        },
    )
    result = api_keys.create("CI Pipeline", organization="0a111111-2222-3333-4444-555566667777")
    assert isinstance(result, CreatedApiKey)
    assert result.api_key == "pk_live_THE_FULL_SECRET"
    assert result.key_id == "key-uuid-1"


def test_create_serialises_default_role_member(httpx_mock: HTTPXMock, api_keys: ApiKeys) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/api-keys/",
        status_code=201,
        json={
            "api_key": "pk_live_x",
            "key_id": "k",
            "key_prefix": "pk_live_x",
            "name": "n",
            "role": "member",
            "rate_limit": 5000,
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    api_keys.create("n", organization="0a111111-2222-3333-4444-555566667777")
    import json as _json

    sent = httpx_mock.get_request()
    assert sent is not None
    body = _json.loads(sent.read())
    assert body["role"] == "member"


@pytest.mark.parametrize("role", ["viewer", "member", "admin", "owner"])
def test_create_passes_explicit_role(httpx_mock: HTTPXMock, api_keys: ApiKeys, role: str) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/api-keys/",
        status_code=201,
        json={
            "api_key": "pk_live_x",
            "key_id": "k",
            "key_prefix": "pk_live_x",
            "name": "n",
            "role": role,
            "rate_limit": 5000,
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    api_keys.create("n", organization="0a111111-2222-3333-4444-555566667777", role=role)  # type: ignore[arg-type]
    import json as _json

    sent = httpx_mock.get_request()
    assert sent is not None
    body = _json.loads(sent.read())
    assert body["role"] == role


def test_create_omits_optional_fields_when_default(
    httpx_mock: HTTPXMock, api_keys: ApiKeys
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/api-keys/",
        status_code=201,
        json={
            "api_key": "pk_live_x",
            "key_id": "k",
            "key_prefix": "pk_live_x",
            "name": "n",
            "role": "member",
            "rate_limit": 5000,
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    api_keys.create("n", organization="0a111111-2222-3333-4444-555566667777")
    import json as _json

    body = _json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert "rate_limit" not in body
    assert "expires_at" not in body


def test_create_serialises_rate_limit_when_provided(
    httpx_mock: HTTPXMock, api_keys: ApiKeys
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/api-keys/",
        status_code=201,
        json={
            "api_key": "pk_live_x",
            "key_id": "k",
            "key_prefix": "pk_live_x",
            "name": "n",
            "role": "member",
            "rate_limit": 50000,
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    api_keys.create("n", organization="0a111111-2222-3333-4444-555566667777", rate_limit=50000)
    import json as _json

    body = _json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body["rate_limit"] == 50000


def test_create_serialises_expires_at_datetime(httpx_mock: HTTPXMock, api_keys: ApiKeys) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/api-keys/",
        status_code=201,
        json={
            "api_key": "pk_live_x",
            "key_id": "k",
            "key_prefix": "pk_live_x",
            "name": "n",
            "role": "member",
            "rate_limit": 5000,
            "expires_at": "2026-12-31T23:59:59+00:00",
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    expires = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    api_keys.create("n", organization="0a111111-2222-3333-4444-555566667777", expires_at=expires)
    import json as _json

    body = _json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body["expires_at"].startswith("2026-12-31T23:59:59")


def test_create_serialises_expires_at_string_passthrough(
    httpx_mock: HTTPXMock, api_keys: ApiKeys
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/api-keys/",
        status_code=201,
        json={
            "api_key": "pk_live_x",
            "key_id": "k",
            "key_prefix": "pk_live_x",
            "name": "n",
            "role": "member",
            "rate_limit": 5000,
            "expires_at": "2026-06-01T00:00:00Z",
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    api_keys.create(
        "n", organization="0a111111-2222-3333-4444-555566667777", expires_at="2026-06-01T00:00:00Z"
    )
    import json as _json

    body = _json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body["expires_at"] == "2026-06-01T00:00:00Z"


def test_create_403_propagates(httpx_mock: HTTPXMock, api_keys: ApiKeys) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/api-keys/",
        status_code=403,
        json={"detail": "Insufficient permissions"},
    )
    with pytest.raises(ForbiddenError):
        api_keys.create("n", organization="0a111111-2222-3333-4444-555566667777")


def test_create_400_invalid_role_propagates(httpx_mock: HTTPXMock, api_keys: ApiKeys) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/api-keys/",
        status_code=400,
        json={"detail": "Invalid role"},
    )
    with pytest.raises(ValidationError):
        api_keys.create("n", organization="0a111111-2222-3333-4444-555566667777", role="member")


# ───────────── get ─────────────


def test_get_returns_typed_api_key(httpx_mock: HTTPXMock, api_keys: ApiKeys) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/api-keys/key-uuid-1",
        json={"success": True, "api_key": _key_payload()},
    )
    result = api_keys.get("key-uuid-1")
    assert isinstance(result, ApiKey)
    assert result.id == "key-uuid-1"


def test_get_404_propagates(httpx_mock: HTTPXMock, api_keys: ApiKeys) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/api-keys/missing",
        status_code=404,
        json={"detail": "API key not found"},
    )
    with pytest.raises(NotFoundError):
        api_keys.get("missing")


# ───────────── update ─────────────


def test_update_with_no_fields_raises_value_error(api_keys: ApiKeys) -> None:
    with pytest.raises(ValueError, match="At least one"):
        api_keys.update("key-uuid-1")


def test_update_serialises_only_provided_fields(httpx_mock: HTTPXMock, api_keys: ApiKeys) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE}/api/v1/api-keys/key-uuid-1",
        json={"success": True, "api_key": _key_payload(name="renamed")},
    )
    api_keys.update("key-uuid-1", name="renamed")
    import json as _json

    body = _json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {"name": "renamed"}


def test_update_disable_key_via_is_active_false(httpx_mock: HTTPXMock, api_keys: ApiKeys) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE}/api/v1/api-keys/key-uuid-1",
        json={"success": True, "api_key": _key_payload(is_active=False)},
    )
    result = api_keys.update("key-uuid-1", is_active=False)
    assert result.is_active is False


def test_update_combines_multiple_fields(httpx_mock: HTTPXMock, api_keys: ApiKeys) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE}/api/v1/api-keys/key-uuid-1",
        json={
            "success": True,
            "api_key": _key_payload(name="x", rate_limit=99, is_active=False),
        },
    )
    api_keys.update("key-uuid-1", name="x", rate_limit=99, is_active=False)
    import json as _json

    body = _json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {"name": "x", "rate_limit": 99, "is_active": False}


# ───────────── delete ─────────────


def test_delete_round_trip(httpx_mock: HTTPXMock, api_keys: ApiKeys) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/api-keys/key-uuid-1",
        json={"success": True, "message": "API key deleted successfully"},
    )
    api_keys.delete("key-uuid-1")
    # `delete` returns None, so state the request itself. Without this the test rests
    # entirely on pytest-httpx's `assert_all_responses_were_requested` default - a real
    # assertion, but an INVISIBLE one that a single fixture option would remove here and
    # in ~16 sibling tests at once, with nothing in any of them saying so.
    req = httpx_mock.get_requests()[0]
    assert req.method == "DELETE"
    assert req.url.path == "/api/v1/api-keys/key-uuid-1"


def test_delete_404_propagates(httpx_mock: HTTPXMock, api_keys: ApiKeys) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/api-keys/missing",
        status_code=404,
        json={"detail": "API key not found"},
    )
    with pytest.raises(NotFoundError):
        api_keys.delete("missing")
