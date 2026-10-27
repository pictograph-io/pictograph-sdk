"""Tests for the ``pictograph organizations`` command group.

Self-contained: each command is exercised via ``typer.testing.CliRunner``
against a ``MagicMock`` client patched at the command module's import
boundary, with config redirected to a tmp dir so no real ``~/.pictograph``
or network access is touched.
"""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pictograph.cli._app import app
from pictograph.cli._config import write_config
from pictograph.models.organization import (
    Organization,
    OrganizationInvite,
    OrganizationMember,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ~/.pictograph/* to tmp_path so tests don't touch real config."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("PICTOGRAPH_API_KEY", raising=False)
    monkeypatch.setattr(
        "pictograph.cli._config.CONFIG_DIR",
        fake_home / ".pictograph",
    )
    monkeypatch.setattr(
        "pictograph.cli._config.CONFIG_PATH",
        fake_home / ".pictograph" / "config.toml",
    )
    return fake_home


def _patch_get_client(client: MagicMock) -> AbstractContextManager[MagicMock]:
    """Patch ``get_client`` in the organizations command namespace."""
    return patch(
        "pictograph.cli.commands.organizations.get_client",
        return_value=client,
    )


def _organization() -> Organization:
    now = datetime.now(timezone.utc)
    return Organization(
        id="org-1",
        name="Acme",
        slug="acme",
        subscription_tier="core",
        credits_remaining=1000,
        credits_monthly_allowance=5000,
        credits_reset_at=None,
        max_users=10,
        max_images=100_000,
        max_storage_bytes=10_000_000_000,
        member_count=3,
        pending_invite_count=1,
        created_at=now,
        updated_at=now,
    )


def _member(member_id: str = "mem-1") -> OrganizationMember:
    return OrganizationMember(
        id=member_id,
        user_id="user-1",
        email="alice@acme.com",
        full_name="Alice",
        role="member",
        joined_at=datetime.now(timezone.utc),
    )


def _invite(invite_id: str = "inv-1") -> OrganizationInvite:
    now = datetime.now(timezone.utc)
    return OrganizationInvite(
        id=invite_id,
        organization_id="org-1",
        email="bob@acme.com",
        role="member",
        status="pending",
        invited_by="user-1",
        expires_at=now,
        created_at=now,
    )


def test_me_prints_organization(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.organizations.me.return_value = _organization()
    with _patch_get_client(client):
        res = runner.invoke(app, ["organizations", "me"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["id"] == "org-1"
    assert payload["subscription_tier"] == "core"
    client.organizations.me.assert_called_once_with()


def test_members_renders_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.organizations.list_members.return_value = [_member("a"), _member("b")]
    with _patch_get_client(client):
        res = runner.invoke(app, ["organizations", "members", "--json"])
    assert res.exit_code == 0, res.stdout
    assert len(json.loads(res.stdout)) == 2
    client.organizations.list_members.assert_called_once_with()


def test_member_role_updates(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.organizations.update_member_role.return_value = {"id": "mem-1", "role": "admin"}
    with _patch_get_client(client):
        res = runner.invoke(app, ["organizations", "member-role", "mem-1", "--role", "admin"])
    assert res.exit_code == 0, res.stdout
    assert json.loads(res.stdout)["role"] == "admin"
    client.organizations.update_member_role.assert_called_once_with("mem-1", role="admin")


def test_member_remove_with_yes(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.organizations.remove_member.return_value = None
    with _patch_get_client(client):
        res = runner.invoke(app, ["organizations", "member-remove", "mem-1", "--yes"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload == {"id": "mem-1", "removed": True}
    client.organizations.remove_member.assert_called_once_with("mem-1")


def test_invites_renders_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.organizations.list_invites.return_value = [_invite("a"), _invite("b")]
    with _patch_get_client(client):
        res = runner.invoke(app, ["organizations", "invites", "--status", "pending", "--json"])
    assert res.exit_code == 0, res.stdout
    assert len(json.loads(res.stdout)) == 2
    client.organizations.list_invites.assert_called_once_with(status="pending")


def test_invite_creates(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.organizations.invite.return_value = _invite("inv-1")
    with _patch_get_client(client):
        res = runner.invoke(app, ["organizations", "invite", "bob@acme.com", "--role", "admin"])
    assert res.exit_code == 0, res.stdout
    assert json.loads(res.stdout)["id"] == "inv-1"
    client.organizations.invite.assert_called_once_with("bob@acme.com", role="admin")


def test_invite_revoke_with_yes(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.organizations.revoke_invite.return_value = None
    with _patch_get_client(client):
        res = runner.invoke(app, ["organizations", "invite-revoke", "inv-1", "--yes"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload == {"id": "inv-1", "revoked": True}
    client.organizations.revoke_invite.assert_called_once_with("inv-1")


def test_update_edits_profile(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    org = _organization()
    org.name = "Renamed"
    org.is_public = True
    client.organizations.update.return_value = org
    with _patch_get_client(client):
        res = runner.invoke(app, ["organizations", "update", "--name", "Renamed", "--public"])
    assert res.exit_code == 0, res.stdout
    assert json.loads(res.stdout)["name"] == "Renamed"
    client.organizations.update.assert_called_once_with(
        name="Renamed", description=None, is_public=True
    )
