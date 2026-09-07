"""Tests for ``pictograph.resources.organizations.Organizations``.

Coverage targets:
- ``me`` returns typed Organization with member/invite counts.
- ``list_members`` parses email/full_name into typed members.
- ``update_member_role`` happy path + 400 last-owner + 404 missing.
- ``remove_member`` happy + 404 cross-org.
- ``list_invites`` filter forwarding.
- ``invite`` happy + 409 dup + 402 cap-exceeded + 403 no-write-role.
- ``revoke_invite`` happy + 400 non-pending status.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    PaymentRequiredError,
    ValidationError,
)
from pictograph.models.organization import (
    Organization,
    OrganizationInvite,
    OrganizationMember,
)
from pictograph.resources.organizations import Organizations

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
def organizations(transport: Transport) -> Organizations:
    return Organizations(transport)


def _org_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "org-uuid",
        "name": "Pictograph Inc",
        "slug": "pictograph",
        "subscription_tier": "pro",
        "credits_remaining": 850,
        "credits_monthly_allowance": 1000,
        "credits_reset_at": "2026-05-01T00:00:00Z",
        "max_users": 25,
        "max_images": 50000,
        "max_storage_bytes": 53687091200,
        "member_count": 5,
        "pending_invite_count": 2,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-04-19T00:00:00Z",
    }
    base.update(overrides)
    return base


def _member_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "membership-uuid-1",
        "user_id": "user-uuid-1",
        "email": "alice@example.com",
        "full_name": "Alice Anderson",
        "role": "admin",
        "joined_at": "2026-02-01T00:00:00Z",
    }
    base.update(overrides)
    return base


def _invite_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "invite-uuid-1",
        "organization_id": "org-uuid",
        "email": "bob@example.com",
        "role": "member",
        "status": "pending",
        "invited_by": "user-uuid-1",
        "expires_at": "2026-04-26T00:00:00Z",
        "created_at": "2026-04-19T00:00:00Z",
    }
    base.update(overrides)
    return base


# ───────────── me ─────────────


def test_me_returns_typed_organization(httpx_mock: HTTPXMock, organizations: Organizations) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/organizations/me",
        json={"organization": _org_payload()},
    )
    org = organizations.me()
    assert isinstance(org, Organization)
    assert org.id == "org-uuid"
    assert org.subscription_tier == "pro"
    assert org.credits_remaining == 850
    assert org.member_count == 5
    assert org.pending_invite_count == 2


def test_me_handles_null_credits_reset_at(
    httpx_mock: HTTPXMock, organizations: Organizations
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/organizations/me",
        json={"organization": _org_payload(credits_reset_at=None)},
    )
    org = organizations.me()
    assert org.credits_reset_at is None


# ───────────── list_members ─────────────


def test_list_members_returns_typed_rows(
    httpx_mock: HTTPXMock, organizations: Organizations
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/organizations/members",
        json={
            "members": [
                _member_payload(),
                _member_payload(
                    id="m2",
                    user_id="u2",
                    email="bob@example.com",
                    full_name="Bob Brown",
                    role="member",
                ),
            ]
        },
    )
    members = organizations.list_members()
    assert len(members) == 2
    assert all(isinstance(m, OrganizationMember) for m in members)
    assert {m.role for m in members} == {"admin", "member"}


def test_list_members_handles_null_profile_fields(
    httpx_mock: HTTPXMock, organizations: Organizations
) -> None:
    """Profile rows can be missing - email/full_name surface as None, not crash."""
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/organizations/members",
        json={"members": [_member_payload(email=None, full_name=None)]},
    )
    members = organizations.list_members()
    assert members[0].email is None
    assert members[0].full_name is None


def test_list_members_empty(httpx_mock: HTTPXMock, organizations: Organizations) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/organizations/members",
        json={"members": []},
    )
    assert organizations.list_members() == []


# ───────────── update_member_role ─────────────


def test_update_member_role_happy_path(httpx_mock: HTTPXMock, organizations: Organizations) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE}/api/v1/developer/organizations/members/m1",
        json={"member": {"id": "m1", "role": "admin"}},
    )
    result = organizations.update_member_role("m1", role="admin")
    sent = httpx_mock.get_request()
    assert sent is not None
    body = json.loads(sent.read())
    assert body == {"role": "admin"}
    assert result == {"id": "m1", "role": "admin"}


def test_update_member_role_400_demote_last_owner(
    httpx_mock: HTTPXMock, organizations: Organizations
) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE}/api/v1/developer/organizations/members/m1",
        status_code=400,
        json={"detail": "Cannot demote the last owner of the organization."},
    )
    with pytest.raises(ValidationError, match="last owner"):
        organizations.update_member_role("m1", role="admin")


def test_update_member_role_404_missing_or_cross_org(
    httpx_mock: HTTPXMock, organizations: Organizations
) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE}/api/v1/developer/organizations/members/missing",
        status_code=404,
        json={"detail": "Member not found"},
    )
    with pytest.raises(NotFoundError):
        organizations.update_member_role("missing", role="admin")


def test_update_member_role_403_no_write_role(
    httpx_mock: HTTPXMock, organizations: Organizations
) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE}/api/v1/developer/organizations/members/m1",
        status_code=403,
        json={"detail": "Insufficient permissions. Requires admin or owner role."},
    )
    with pytest.raises(ForbiddenError):
        organizations.update_member_role("m1", role="admin")


# ───────────── remove_member ─────────────


def test_remove_member_happy_path(httpx_mock: HTTPXMock, organizations: Organizations) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/organizations/members/m1",
        json={"success": True, "member_id": "m1"},
    )
    organizations.remove_member("m1")
    req = httpx_mock.get_requests()[0]
    assert req.method == "DELETE"
    assert req.url.path == "/api/v1/developer/organizations/members/m1"


def test_remove_member_404_cross_org(httpx_mock: HTTPXMock, organizations: Organizations) -> None:
    """Cross-org reads return 404 (not 403) - no existence leak."""
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/organizations/members/other-org-member",
        status_code=404,
        json={"detail": "Member not found"},
    )
    with pytest.raises(NotFoundError):
        organizations.remove_member("other-org-member")


def test_remove_member_400_last_owner(httpx_mock: HTTPXMock, organizations: Organizations) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/organizations/members/m1",
        status_code=400,
        json={"detail": "Cannot remove the last owner of the organization."},
    )
    with pytest.raises(ValidationError, match="last owner"):
        organizations.remove_member("m1")


# ───────────── list_invites ─────────────


def test_list_invites_no_filter(httpx_mock: HTTPXMock, organizations: Organizations) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/organizations/invites",
        json={
            "invites": [
                _invite_payload(),
                _invite_payload(id="i2", email="charlie@example.com", status="accepted"),
            ]
        },
    )
    invites = organizations.list_invites()
    assert len(invites) == 2
    assert all(isinstance(i, OrganizationInvite) for i in invites)


def test_list_invites_status_filter_forwarded(
    httpx_mock: HTTPXMock, organizations: Organizations
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/organizations/invites?status=pending",
        json={"invites": [_invite_payload()]},
    )
    invites = organizations.list_invites(status="pending")
    assert len(invites) == 1
    assert invites[0].status == "pending"


# ───────────── invite ─────────────


def test_invite_serialises_body(httpx_mock: HTTPXMock, organizations: Organizations) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/organizations/invites",
        json={"invite": _invite_payload(email="alice@new.com", role="admin")},
    )
    invite = organizations.invite("alice@new.com", role="admin")
    sent = httpx_mock.get_request()
    assert sent is not None
    body = json.loads(sent.read())
    assert body == {"email": "alice@new.com", "role": "admin"}
    assert invite.email == "alice@new.com"
    assert invite.role == "admin"


def test_invite_annotator_role(httpx_mock: HTTPXMock, organizations: Organizations) -> None:
    """The annotator role (2026-08-20) is a valid invite role and round-trips."""
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/organizations/invites",
        json={"invite": _invite_payload(email="ann@new.com", role="annotator")},
    )
    invite = organizations.invite("ann@new.com", role="annotator")
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {"email": "ann@new.com", "role": "annotator"}
    assert invite.role == "annotator"


def test_update_member_role_to_annotator(
    httpx_mock: HTTPXMock, organizations: Organizations
) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE}/api/v1/developer/organizations/members/m1",
        json={"member": {"id": "m1", "role": "annotator"}},
    )
    result = organizations.update_member_role("m1", role="annotator")
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {"role": "annotator"}
    assert result == {"id": "m1", "role": "annotator"}


def test_invite_default_role_is_member(httpx_mock: HTTPXMock, organizations: Organizations) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/organizations/invites",
        json={"invite": _invite_payload()},
    )
    organizations.invite("bob@example.com")
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body["role"] == "member"


def test_invite_409_duplicate(httpx_mock: HTTPXMock, organizations: Organizations) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/organizations/invites",
        status_code=409,
        json={"detail": "A pending invite already exists for this email."},
    )
    with pytest.raises(ConflictError):
        organizations.invite("dup@example.com")


def test_invite_409_already_member(httpx_mock: HTTPXMock, organizations: Organizations) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/organizations/invites",
        status_code=409,
        json={"detail": "User is already a member of this organization."},
    )
    with pytest.raises(ConflictError, match="already a member"):
        organizations.invite("existing@example.com")


def test_invite_402_cap_exceeded_carries_credit_context(
    httpx_mock: HTTPXMock, organizations: Organizations
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/organizations/invites",
        status_code=402,
        json={
            "detail": {
                "error": "limit_exceeded",
                "limit_type": "users",
                "current": 25,
                "pending": 0,
                "limit": 25,
                "upgrade_url": "/settings?tab=billing",
            }
        },
    )
    with pytest.raises(PaymentRequiredError) as exc:
        organizations.invite("over-cap@example.com")
    assert exc.value.upgrade_url == "/settings?tab=billing"


def test_invite_403_no_write_role(httpx_mock: HTTPXMock, organizations: Organizations) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/organizations/invites",
        status_code=403,
        json={"detail": "Insufficient permissions. Requires admin or owner role."},
    )
    with pytest.raises(ForbiddenError):
        organizations.invite("nope@example.com")


# ───────────── revoke_invite ─────────────


def test_revoke_invite_happy_path(httpx_mock: HTTPXMock, organizations: Organizations) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/organizations/invites/i1",
        json={"success": True, "invite_id": "i1"},
    )
    organizations.revoke_invite("i1")
    req = httpx_mock.get_requests()[0]
    assert req.method == "DELETE"
    assert req.url.path == "/api/v1/developer/organizations/invites/i1"


def test_revoke_invite_400_already_accepted(
    httpx_mock: HTTPXMock, organizations: Organizations
) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/organizations/invites/i1",
        status_code=400,
        json={"detail": "Cannot revoke invite with status 'accepted'."},
    )
    with pytest.raises(ValidationError, match="accepted"):
        organizations.revoke_invite("i1")


def test_revoke_invite_404(httpx_mock: HTTPXMock, organizations: Organizations) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/organizations/invites/missing",
        status_code=404,
        json={"detail": "Invite not found"},
    )
    with pytest.raises(NotFoundError):
        organizations.revoke_invite("missing")


# ───────────── update ─────────────


def test_update_patches_profile(httpx_mock: HTTPXMock, organizations: Organizations) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE}/api/v1/developer/organizations/me",
        json={"organization": _org_payload(name="Renamed", description="hi", is_public=True)},
    )
    org = organizations.update(name="Renamed", description="hi", is_public=True)
    assert org.name == "Renamed" and org.description == "hi" and org.is_public is True
    body = httpx_mock.get_requests()[-1].read().decode().replace(" ", "")
    assert '"name":"Renamed"' in body and '"is_public":true' in body


def test_update_no_fields_raises(organizations: Organizations) -> None:
    with pytest.raises(ValueError):
        organizations.update()
