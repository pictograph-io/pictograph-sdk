"""Organization Pydantic models - info, members, invites.

Triplet of read-side models exposed by the ``client.organizations`` resource:

- :class:`Organization` - what tier you're on, how many credits remain,
  what the per-tier caps are.
- :class:`OrganizationMember` - one row per user in the org, with their
  role and joined date.
- :class:`OrganizationInvite` - pending / accepted / expired / revoked
  invites. The token itself is never exposed (invitees follow the email).

All three use ``extra="ignore"`` so backend additions don't break parsing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

OrganizationRole = Literal["owner", "admin", "member", "viewer"]
"""Roles a member can hold. Strict superset hierarchy: owner > admin > member > viewer."""

InviteRole = Literal["admin", "member", "viewer"]
"""Roles assignable on an invite. Owners cannot be invited - they must be promoted."""

InviteStatus = Literal["pending", "accepted", "expired", "revoked"]
"""Lifecycle of an organization invite."""

SubscriptionTier = Literal["free", "core", "pro", "team", "enterprise"]
"""Billing tiers. Determines per-org caps (members, images, storage)."""


class Organization(BaseModel):
    """Organization metadata + tier limits + credit balance.

    Combines the org row with denormalized counts (``member_count``,
    ``pending_invite_count``) so an agent can decide whether to invite more
    users without a follow-up call.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    slug: str
    description: str | None = None
    is_public: bool | None = None
    subscription_tier: SubscriptionTier
    credits_remaining: int = Field(ge=0)
    credits_monthly_allowance: int = Field(ge=0)
    credits_reset_at: datetime | None = None
    max_users: int = Field(ge=1)
    max_images: int = Field(ge=0)
    max_storage_bytes: int = Field(ge=0)
    member_count: int = Field(ge=0)
    pending_invite_count: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime


class OrganizationMember(BaseModel):
    """A single member's row.

    ``email`` and ``full_name`` are pulled from the user's profile server-side
    and may be ``None`` if the linked profile row is missing (extremely rare,
    indicates a soft inconsistency).
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Membership row UUID - pass to update_role / remove.")
    user_id: str
    email: str | None = None
    full_name: str | None = None
    role: OrganizationRole
    joined_at: datetime


class OrganizationInvite(BaseModel):
    """A pending / accepted / expired / revoked invite.

    The invite ``token`` is intentionally not exposed - invitees receive it
    via email and use it through the web app's accept-invite flow.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    organization_id: str
    email: str
    role: InviteRole
    status: InviteStatus
    invited_by: str | None = None
    expires_at: datetime
    created_at: datetime
