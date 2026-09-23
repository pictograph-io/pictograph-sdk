"""Async Organizations resource - info, members, invites for the calling key's org.

Async twin of :class:`pictograph.resources.organizations.Organizations`. Every
method is implicitly scoped to the API key's organization (no ``organization_id``
parameter - cross-org access is impossible by construction).
"""

from __future__ import annotations

from typing import Any

from pictograph.models.organization import (
    InviteRole,
    InviteStatus,
    Organization,
    OrganizationInvite,
    OrganizationMember,
    OrganizationRole,
)
from pictograph.resources._base import AsyncResource

_API_PATH = "/api/v1/developer/organizations"


class AsyncOrganizations(AsyncResource):
    """Read org info; manage members and invites for the calling key's org (async)."""

    async def me(self) -> Organization:
        """Get the calling API key's organization (with member/invite counts)."""
        response = await self._transport.request("GET", f"{_API_PATH}/me")
        return self._parse(Organization, response["organization"])

    async def update(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        is_public: bool | None = None,
    ) -> Organization:
        """Update your organization's profile - name / description / is_public
        (only the fields you pass change). Requires an admin/owner key."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if is_public is not None:
            body["is_public"] = is_public
        if not body:
            raise ValueError("update() requires at least one field to change")
        response = await self._transport.request("PATCH", f"{_API_PATH}/me", json=body)
        return self._parse(Organization, response["organization"])

    # ───────────── members ─────────────

    async def list_members(self) -> list[OrganizationMember]:
        """List every member of the org with role and email."""
        response = await self._transport.request("GET", f"{_API_PATH}/members")
        return self._parse_list(OrganizationMember, response.get("members", []))

    async def update_member_role(self, member_id: str, *, role: OrganizationRole) -> dict[str, Any]:
        """Update a member's role (cannot demote the last owner).

        Returns ``{"id": str, "role": str}`` - a thin acknowledgement.

        Raises:
            NotFoundError: ``member_id`` doesn't exist (or is in another org).
            ForbiddenError: API key lacks admin/owner role.
            ValidationError: Demoting the last owner.
        """
        response = await self._transport.request(
            "PATCH",
            f"{_API_PATH}/members/{member_id}",
            json={"role": role},
        )
        return dict(response["member"])

    async def remove_member(self, member_id: str) -> None:
        """Remove a member from the org (cannot remove the last owner)."""
        await self._transport.request("DELETE", f"{_API_PATH}/members/{member_id}")

    # ───────────── invites ─────────────

    async def list_invites(self, *, status: InviteStatus | None = None) -> list[OrganizationInvite]:
        """List invites in the org, optionally filtered by status."""
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        response = await self._transport.request(
            "GET", f"{_API_PATH}/invites", params=params or None
        )
        return self._parse_list(OrganizationInvite, response.get("invites", []))

    async def invite(self, email: str, *, role: InviteRole = "member") -> OrganizationInvite:
        """Invite a new member by email (sends an invite email server-side).

        Args:
            email: Address to invite. Lowercased + trimmed server-side.
            role: ``"admin"``, ``"member"`` (default), or ``"viewer"``.

        Raises:
            ForbiddenError: API key lacks admin/owner role.
            ConflictError: Email is already a member or has a pending invite.
            PaymentRequiredError: Member-cap reached for the org's tier.
            ValidationError: Email is malformed.
        """
        body = {"email": email, "role": role}
        response = await self._transport.request("POST", f"{_API_PATH}/invites", json=body)
        return self._parse(OrganizationInvite, response["invite"])

    async def revoke_invite(self, invite_id: str) -> None:
        """Revoke a pending invite (no-op if already accepted or expired)."""
        await self._transport.request("DELETE", f"{_API_PATH}/invites/{invite_id}")
