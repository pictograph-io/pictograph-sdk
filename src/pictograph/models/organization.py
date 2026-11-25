"""Organization Pydantic models - info, members, invites.

Triplet of read-side models exposed by the ``client.organizations`` resource:

- :class:`Organization` - what tier you're on, how many credits remain,
  what the per-tier caps are.
- :class:`OrganizationMember` - one row per user in the org, with their
  role and joined date.
- :class:`OrganizationInvite` - pending / accepted / expired / revoked / declined
  invites. The token itself is never exposed (invitees follow the email).

All three use ``extra="ignore"`` so backend additions don't break parsing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

OrganizationRole = Literal["owner", "admin", "member", "annotator", "viewer"]
"""Roles a member can hold. Hierarchy: owner > admin > member > annotator > viewer.

``annotator`` is a member-minus role: it can annotate + soft-archive + download
within its granted datasets/directories, but cannot create/delete datasets, manage
API keys, upload, or permanently delete. ``annotator`` and ``viewer`` are scoped to
specific datasets/directories set by an owner/admin."""

InviteRole = Literal["admin", "member", "annotator", "viewer"]
"""Roles assignable on an invite. Owners cannot be invited - they must be promoted."""

InviteStatus = Literal["pending", "accepted", "expired", "revoked", "declined"]
"""Lifecycle of an organization invite. ``declined``: the invitee said no."""

SubscriptionTier = Literal["free", "core", "pro", "team", "enterprise"]
"""The standard plans. A plan sets the organization's limits (members, images, storage).

``Organization.subscription_tier`` also takes any other string - a custom Enterprise plan, or a
plan added later - so it reads as itself instead of failing to parse."""


class Organization(BaseModel):
    """An organization: its plan, its plan's limits, and its member and invite counts.

    ``member_count`` + ``pending_invite_count`` against ``max_users`` says whether
    another invite fits, with no second call.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    slug: str
    description: str | None = None
    is_public: bool | None = None
    subscription_tier: SubscriptionTier | str = Field(description="Its plan.")
    credits_remaining: int = Field(
        ge=0,
        description="DEPRECATED: whole dollars left this period. Use credits.balance().",
    )
    credits_monthly_allowance: int = Field(
        ge=0,
        description="DEPRECATED: the monthly allowance in whole dollars. Use credits.balance().",
    )
    credits_reset_at: datetime | None = None
    max_users: int = Field(ge=1, description="Members its plan allows, pending invites included.")
    max_images: int = Field(ge=0, description="Images its plan allows.")
    max_storage_bytes: int = Field(ge=0, description="Storage its plan allows, in bytes.")
    member_count: int = Field(ge=0)
    pending_invite_count: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime


class OrganizationMember(BaseModel):
    """One member of the organization.

    ``email`` and ``full_name`` come from the member's profile, and are ``None`` only if
    that profile is missing.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Membership id: update_member_role and remove_member take it.")
    user_id: str
    email: str | None = None
    full_name: str | None = None
    role: OrganizationRole
    joined_at: datetime


class OrganizationInvite(BaseModel):
    """A pending / accepted / expired / revoked / declined invite.

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
