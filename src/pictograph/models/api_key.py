"""API-key Pydantic models - listed key metadata + freshly-created key reveal."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ApiKeyRole = Literal["viewer", "member", "admin", "owner"]
"""Roles an API key can carry. Higher roles include lower role permissions."""


class ApiKey(BaseModel):
    """API key metadata returned by list/get/update endpoints.

    The actual secret string is never returned after creation - see
    :class:`CreatedApiKey` for the create response that reveals it once.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    organization_id: str
    name: str
    key_prefix: str = Field(
        description="First 12 characters (e.g., 'pk_live_abc1') - safe to log.",
    )
    role: ApiKeyRole
    rate_limit: int
    is_active: bool
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    created_at: datetime


class CreatedApiKey(BaseModel):
    """Response from :meth:`ApiKeys.create` - includes the secret one time only.

    Treat ``api_key`` like a password: store it in a secret manager, never
    in version control. The backend hashes it with bcrypt and discards the
    plaintext on insert; there is no recovery path.
    """

    model_config = ConfigDict(extra="ignore")

    api_key: str = Field(
        description="The full secret. Store securely - only shown once.",
    )
    key_id: str
    key_prefix: str
    name: str
    role: ApiKeyRole
    rate_limit: int
    expires_at: datetime | None = None
    created_at: datetime
