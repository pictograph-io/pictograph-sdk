"""Organizations resource - info, members, invites for the calling key's org.

Every method is implicitly scoped to the API key's organization - there
is no ``organization_id`` parameter anywhere on this surface. That's a
deliberate security property: cross-org access is impossible by
construction.

Common agent flows:

- :meth:`Organizations.me` once at startup to know the tier, credit
  balance, and member cap.
- :meth:`Organizations.list_members` to surface "who's on this team" for
  attribution.
- :meth:`Organizations.invite` + :meth:`Organizations.list_invites` to
  onboard a colleague to the workspace.

Member / invite mutations require ``admin`` or ``owner`` role on the API
key - surfaced as :class:`ForbiddenError` if the role is too low.
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
from pictograph.resources._base import Resource

_API_PATH = "/api/v1/developer/organizations"


class Organizations(Resource):
    """Read org info; manage members and invites for the calling key's org."""

    def me(self) -> Organization:
        """Get the calling API key's organization.

        Includes denormalized ``member_count`` and ``pending_invite_count``
        so an agent can decide whether to invite more users without a
        second round-trip.
        """
        response = self._transport.request("GET", f"{_API_PATH}/me")
        return self._parse(Organization, response["organization"])

    def update(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        is_public: bool | None = None,
    ) -> Organization:
        """Update your organization's profile - ``name`` / ``description`` /
        ``is_public`` (only the fields you pass change). Requires an admin/owner key."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if is_public is not None:
            body["is_public"] = is_public
        if not body:
            raise ValueError("update() requires at least one field to change")
        response = self._transport.request("PATCH", f"{_API_PATH}/me", json=body)
        return self._parse(Organization, response["organization"])

    # ───────────── members ─────────────

    def list_members(self) -> list[OrganizationMember]:
        """List every member of the org with role and email."""
        response = self._transport.request("GET", f"{_API_PATH}/members")
        return self._parse_list(OrganizationMember, response.get("members", []))

    def update_member_role(self, member_id: str, *, role: OrganizationRole) -> dict[str, Any]:
        """Update a member's role. Cannot demote the last owner.

        Returns ``{"id": str, "role": str}`` - a thin acknowledgement.
        Re-fetch via :meth:`list_members` if you need the full row.

        Raises:
            NotFoundError: ``member_id`` doesn't exist (or is in another org).
            ForbiddenError: API key lacks admin/owner role.
            ValidationError: Demoting the last owner.
        """
        response = self._transport.request(
            "PATCH",
            f"{_API_PATH}/members/{member_id}",
            json={"role": role},
        )
        return dict(response["member"])

    def remove_member(self, member_id: str) -> None:
        """Remove a member from the org. Cannot remove the last owner.

        Raises:
            NotFoundError: ``member_id`` doesn't exist (or is in another org).
            ForbiddenError: API key lacks admin/owner role.
            ValidationError: Removing the last owner.
        """
        self._transport.request("DELETE", f"{_API_PATH}/members/{member_id}")

    # ───────────── invites ─────────────

    def list_invites(self, *, status: InviteStatus | None = None) -> list[OrganizationInvite]:
        """List invites in the org, optionally filtered by status."""
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        response = self._transport.request("GET", f"{_API_PATH}/invites", params=params or None)
        return self._parse_list(OrganizationInvite, response.get("invites", []))

    def invite(self, email: str, *, role: InviteRole = "member") -> OrganizationInvite:
        """Invite a new member by email. Sends an invite email server-side.

        Args:
            email: Address to invite. Lowercased + trimmed server-side.
            role: ``"admin"``, ``"member"`` (default), ``"annotator"``, or
                ``"viewer"``. ``annotator``/``viewer`` are scoped to specific
                datasets/directories set by an owner/admin in the web app.
                Owners cannot be invited - they must be promoted post-acceptance.

        Returns:
            The newly created :class:`OrganizationInvite` (status ``"pending"``).

        Raises:
            ForbiddenError: API key lacks admin/owner role.
            ConflictError: Email is already a member or has a pending invite.
            PaymentRequiredError: Member-cap (members + pending invites) reached
                for the org's tier.
            ValidationError: Email is malformed.
        """
        body = {"email": email, "role": role}
        response = self._transport.request("POST", f"{_API_PATH}/invites", json=body)
        return self._parse(OrganizationInvite, response["invite"])

    def revoke_invite(self, invite_id: str) -> None:
        """Revoke a pending invite. No-op if already accepted or expired.

        Raises:
            NotFoundError: ``invite_id`` doesn't exist (or is in another org).
            ForbiddenError: API key lacks admin/owner role.
            ValidationError: Invite is not in ``pending`` status.
        """
        self._transport.request("DELETE", f"{_API_PATH}/invites/{invite_id}")
