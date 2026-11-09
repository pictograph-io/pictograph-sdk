"""Live: organizations.me + list_members + list_invites."""

from __future__ import annotations

import pytest

from pictograph import Client
from pictograph.models.organization import (
    Organization,
    OrganizationInvite,
    OrganizationMember,
)

pytestmark = pytest.mark.live


def test_me_returns_current_org(client: Client) -> None:
    org = client.organizations.me()
    assert isinstance(org, Organization)
    assert org.id
    assert org.name
    assert org.subscription_tier
    assert org.member_count >= 1


def test_list_members(client: Client) -> None:
    members = client.organizations.list_members()
    assert isinstance(members, list)
    assert len(members) >= 1
    for m in members:
        assert isinstance(m, OrganizationMember)
        assert m.user_id
        assert m.role in {"owner", "admin", "member", "annotator", "viewer"}


def test_list_invites(client: Client) -> None:
    invites = client.organizations.list_invites()
    assert isinstance(invites, list)
    for i in invites:
        assert isinstance(i, OrganizationInvite)


def test_list_invites_with_status_filter(client: Client) -> None:
    # All supported statuses return a list (may be empty).
    for status in ("pending", "accepted", "expired", "revoked"):
        invites = client.organizations.list_invites(status=status)
        assert isinstance(invites, list)
