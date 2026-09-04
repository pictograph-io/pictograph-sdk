"""API keys resource - list, create, get, update, delete.

Note the URL prefix differs from other developer endpoints: this resource
lives at ``/api/v1/api-keys/`` (not ``/developer/...``) because the route
predates the developer API split. Auth is dual: both JWT (web app) and API
key (this SDK) work via :func:`get_auth_context` server-side.

The :meth:`ApiKeys.create` response uniquely contains the secret string -
all subsequent calls reveal only metadata (``key_prefix``, ``role``, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pictograph.models.api_key import ApiKey, ApiKeyRole, CreatedApiKey
from pictograph.resources import _resolve
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from datetime import datetime

_API_PATH = "/api/v1/api-keys/"


class ApiKeys(Resource):
    """Manage API keys for the authenticated organization."""

    def list(self, organization: str | None = None) -> list[ApiKey]:
        """List API keys.

        ``organization`` defaults to the calling key's own organization, which
        is the only one an API key may address - naming a different org is a 403
        from the API. A NAME or SLUG is accepted as well as an id.
        """
        params: dict[str, Any] = {}
        if organization is not None:
            params["organization_id"] = _resolve.organization_id(self._transport, organization)
        response = self._transport.request("GET", _API_PATH, params=params or None)
        return self._parse_list(ApiKey, response.get("api_keys", []))

    def create(
        self,
        name: str,
        *,
        organization: str | None = None,
        role: ApiKeyRole = "member",
        rate_limit: int | None = None,
        expires_at: datetime | str | None = None,
    ) -> CreatedApiKey:
        """Create a new API key. Requires admin or owner role.

        The returned :class:`CreatedApiKey` includes the secret string - this
        is the **only** time the secret is exposed. Persist it immediately.

        Args:
            organization: Your organization, by NAME or SLUG (an id also
                works). Defaults to the calling key's own organization -
                the only one an API key may create keys for.
            name: Human-readable label (1-100 chars).
            role: Role assigned to the new key. Defaults to ``"member"``.
            rate_limit: Custom requests-per-hour cap. Defaults to the org's
                tier-based limit (Free: 1k, Core: 5k, Pro: 20k, Team: 100k).
            expires_at: Optional expiry - accepts a datetime or ISO 8601
                string. ``None`` means the key never expires.
        """
        body: dict[str, Any] = {
            "organization_id": _resolve.organization_id(self._transport, organization),
            "name": name,
            "role": role,
        }
        if rate_limit is not None:
            body["rate_limit"] = rate_limit
        if expires_at is not None:
            body["expires_at"] = (
                expires_at.isoformat() if hasattr(expires_at, "isoformat") else str(expires_at)
            )
        response = self._transport.request("POST", _API_PATH, json=body)
        return self._parse(CreatedApiKey, response)

    def get(self, key_id: str) -> ApiKey:
        """Fetch metadata for a single API key (no secret returned)."""
        response = self._transport.request("GET", f"{_API_PATH}{key_id}")
        return self._parse(ApiKey, response["api_key"])

    def update(
        self,
        key_id: str,
        *,
        name: str | None = None,
        rate_limit: int | None = None,
        is_active: bool | None = None,
    ) -> ApiKey:
        """Patch a key's mutable fields. Requires admin or owner role.

        At least one of ``name`` / ``rate_limit`` / ``is_active`` must be
        provided. The role of an existing key is immutable - create a new
        key with a different role and rotate.
        """
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if rate_limit is not None:
            body["rate_limit"] = rate_limit
        if is_active is not None:
            body["is_active"] = is_active
        if not body:
            raise ValueError("At least one of name / rate_limit / is_active must be provided")
        response = self._transport.request(
            "PATCH",
            f"{_API_PATH}{key_id}",
            json=body,
        )
        return self._parse(ApiKey, response["api_key"])

    def delete(self, key_id: str) -> None:
        """Permanently revoke a key. Cannot be undone."""
        self._transport.request("DELETE", f"{_API_PATH}{key_id}")
