"""Tests for the ``pictograph webhooks`` command group.

Self-contained: each command is invoked through its own ``typer.Typer`` app
(``pictograph.cli.commands.webhooks.app``) via ``CliRunner``, with the shared
``get_client`` factory patched at the command module's import boundary so no
network / auth resolution happens. One happy-path test per command.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pictograph.cli._config import write_config
from pictograph.cli.commands.webhooks import app
from pictograph.models.webhook import (
    CreatedWebhookEndpoint,
    WebhookDelivery,
    WebhookEndpoint,
)

_PATCH_TARGET = "pictograph.cli.commands.webhooks.get_client"


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
    # The module-level paths were computed at import - patch them too.
    monkeypatch.setattr(
        "pictograph.cli._config.CONFIG_DIR",
        fake_home / ".pictograph",
    )
    monkeypatch.setattr(
        "pictograph.cli._config.CONFIG_PATH",
        fake_home / ".pictograph" / "config.toml",
    )
    return fake_home


@pytest.fixture(autouse=True)
def _config(isolated_config: Path) -> None:
    """Write a throwaway API key so ``get_client`` resolution never short-circuits."""
    write_config(api_key="pk_live_x")


def _endpoint(endpoint_id: str = "abcd1234-0000-1111-2222-333344445555") -> WebhookEndpoint:
    return WebhookEndpoint(
        id=endpoint_id,
        organization_id="org-1",
        url="https://example.com/hook",
        event_types=["workflow_run.completed"],
    )


def _delivery(delivery_id: str = "del-1") -> WebhookDelivery:
    return WebhookDelivery(
        id=delivery_id,
        endpoint_id="abcd1234-0000-1111-2222-333344445555",
        organization_id="org-1",
        event_type="workflow_run.completed",
        delivery_id="evt-1",
        status="failed",
        attempts=3,
        last_status_code=500,
    )


def test_create_prints_secret_once(runner: CliRunner) -> None:
    signing_secret = "whsec_super_secret"  # noqa: S105 - fixture value, not a real credential
    client = MagicMock()
    client.webhooks.create.return_value = CreatedWebhookEndpoint(
        endpoint=_endpoint(), secret=signing_secret
    )
    with patch(_PATCH_TARGET, return_value=client):
        res = runner.invoke(
            app,
            ["create", "https://example.com/hook", "--event", "workflow_run.completed"],
        )
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["secret"] == signing_secret
    assert payload["id"] == "abcd1234-0000-1111-2222-333344445555"
    client.webhooks.create.assert_called_once_with(
        "https://example.com/hook",
        description=None,
        event_types=["workflow_run.completed"],
        auth_headers=None,
    )


def test_create_forwards_auth_headers(runner: CliRunner) -> None:
    """--auth-header 'Name: Value' (repeatable) → the create call's auth_headers."""
    client = MagicMock()
    client.webhooks.create.return_value = CreatedWebhookEndpoint(
        endpoint=WebhookEndpoint(id="e", organization_id="o", url="https://example.com/hook"),
        secret="whsec_x",  # noqa: S106 - test fixture
    )
    with patch(_PATCH_TARGET, return_value=client):
        res = runner.invoke(
            app,
            [
                "create",
                "https://example.com/hook",
                "--auth-header",
                "Authorization: Bearer tok",
                "--auth-header",
                "X-Api-Key=k",
            ],
        )
    assert res.exit_code == 0, res.stdout
    client.webhooks.create.assert_called_once_with(
        "https://example.com/hook",
        description=None,
        event_types=None,
        auth_headers={"Authorization": "Bearer tok", "X-Api-Key": "k"},
    )


def test_list_renders_json(runner: CliRunner) -> None:
    client = MagicMock()
    client.webhooks.list.return_value = [_endpoint("a"), _endpoint("b")]
    with patch(_PATCH_TARGET, return_value=client):
        res = runner.invoke(app, ["list", "--json"])
    assert res.exit_code == 0, res.stdout
    assert len(json.loads(res.stdout)) == 2
    client.webhooks.list.assert_called_once_with()


def test_get_renders_json(runner: CliRunner) -> None:
    client = MagicMock()
    client.webhooks.get.return_value = _endpoint("abcd1234-0000-1111-2222-333344445555")
    with patch(_PATCH_TARGET, return_value=client):
        res = runner.invoke(app, ["get", "abcd1234-0000-1111-2222-333344445555"])
    assert res.exit_code == 0, res.stdout
    assert json.loads(res.stdout)["id"] == "abcd1234-0000-1111-2222-333344445555"
    client.webhooks.get.assert_called_once_with("abcd1234-0000-1111-2222-333344445555")


def test_delete_with_yes_skips_confirm(runner: CliRunner) -> None:
    client = MagicMock()
    with patch(_PATCH_TARGET, return_value=client):
        res = runner.invoke(app, ["delete", "abcd1234-0000-1111-2222-333344445555", "--yes"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload == {"endpoint": "abcd1234-0000-1111-2222-333344445555", "deleted": True}
    client.webhooks.delete.assert_called_once_with("abcd1234-0000-1111-2222-333344445555")


def test_test_sends_event(runner: CliRunner) -> None:
    client = MagicMock()
    client.webhooks.test.return_value = {"status": "delivered", "status_code": 200}
    with patch(_PATCH_TARGET, return_value=client):
        res = runner.invoke(app, ["test", "abcd1234-0000-1111-2222-333344445555"])
    assert res.exit_code == 0, res.stdout
    assert json.loads(res.stdout)["status"] == "delivered"
    client.webhooks.test.assert_called_once_with("abcd1234-0000-1111-2222-333344445555")


def test_deliveries_renders_json_with_filters(runner: CliRunner) -> None:
    client = MagicMock()
    client.webhooks.deliveries.return_value = [_delivery("d1"), _delivery("d2")]
    with patch(_PATCH_TARGET, return_value=client):
        res = runner.invoke(
            app,
            [
                "deliveries",
                "--endpoint",
                "abcd1234-0000-1111-2222-333344445555",
                "--status",
                "failed",
                "--json",
            ],
        )
    assert res.exit_code == 0, res.stdout
    assert len(json.loads(res.stdout)) == 2
    client.webhooks.deliveries.assert_called_once_with(
        endpoint="abcd1234-0000-1111-2222-333344445555",
        status="failed",
        limit=50,
        offset=0,
    )


def test_replay_requeues(runner: CliRunner) -> None:
    client = MagicMock()
    with patch(_PATCH_TARGET, return_value=client):
        res = runner.invoke(app, ["replay", "del-1"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload == {"id": "del-1", "replayed": True}
    client.webhooks.replay.assert_called_once_with("del-1")


def test_event_types_lists(runner: CliRunner) -> None:
    client = MagicMock()
    client.webhooks.event_types.return_value = ["workflow_run.completed", "workflow_run.failed"]
    with patch(_PATCH_TARGET, return_value=client):
        res = runner.invoke(app, ["event-types"])
    assert res.exit_code == 0, res.stdout
    assert json.loads(res.stdout)["event_types"] == [
        "workflow_run.completed",
        "workflow_run.failed",
    ]
    client.webhooks.event_types.assert_called_once_with()


def test_rotate_secret_prints_new_secret(runner: CliRunner) -> None:
    client = MagicMock()
    ep = WebhookEndpoint(
        id="abcd1234-0000-1111-2222-333344445555",
        organization_id="org-1",
        url="https://example.com/hook",
        event_types=[],
        secret_version=2,
    )
    new_secret = "whsec_new"  # noqa: S105 - fixture value, not a real credential
    client.webhooks.rotate_secret.return_value = CreatedWebhookEndpoint(
        endpoint=ep, secret=new_secret
    )
    with patch(_PATCH_TARGET, return_value=client):
        res = runner.invoke(app, ["rotate-secret", "abcd1234-0000-1111-2222-333344445555"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["secret"] == new_secret and payload["secret_version"] == 2
    client.webhooks.rotate_secret.assert_called_once_with("abcd1234-0000-1111-2222-333344445555")
