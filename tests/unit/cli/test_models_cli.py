"""Tests for the ``pictograph models`` command group.

Self-contained: a :class:`~typer.testing.CliRunner` invokes the command group
in-process, the SDK client is a :class:`~unittest.mock.MagicMock` patched at the
``models`` command-module boundary, and ``~/.pictograph`` is redirected to a
``tmp_path`` so the suite never touches real config (mirrors ``test_app.py``).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pictograph.cli._config import write_config
from pictograph.cli.commands.models import app
from pictograph.models.model import Model


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


def _model(name: str = "Swift Falcon", *, forked_from: str | None = None) -> Model:
    return Model(
        id="model-uuid-1",
        organization_id="org-uuid",
        name=name,
        description=None,
        model_type="object_detection",
        architecture="yolox-s",
        visibility="private",
        status="ready",
        metrics={"mAP": 0.85},
        class_mapping={"0": "car"},
        version="1.0.0",
        parent_model_id=None,
        forked_from_model_id=forked_from,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _patch_client(client: MagicMock) -> Any:
    """Patch ``get_client`` in the models command-module namespace."""
    return patch("pictograph.cli.commands.models.get_client", return_value=client)


# ───────────── get ─────────────


def test_get_renders_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    # The CLI accepts a name OR id and resolves via get_by_name.
    client.models.get_by_name.return_value = _model("Swift Falcon")
    with _patch_client(client):
        res = runner.invoke(app, ["get", "Swift Falcon"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["name"] == "Swift Falcon"
    client.models.get_by_name.assert_called_once_with("Swift Falcon")


# ───────────── delete ─────────────


def test_delete_with_yes(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.models.get_by_name.return_value = _model("Swift Falcon")
    with _patch_client(client):
        res = runner.invoke(app, ["delete", "model-uuid-1", "--yes"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["deleted"] is True
    assert payload["model_id"] == "model-uuid-1"
    client.models.delete.assert_called_once_with(model_id="model-uuid-1")


def test_delete_aborts_without_confirmation(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    with _patch_client(client):
        res = runner.invoke(app, ["delete", "model-uuid-1"], input="n\n")
    assert res.exit_code != 0
    client.models.delete.assert_not_called()


def test_delete_multiple_uses_bulk_delete(runner: CliRunner, isolated_config: Path) -> None:
    from pictograph.models.common import BulkDeleteResult

    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.models.bulk_delete.return_value = BulkDeleteResult(
        succeeded=["m1", "m2"], not_found=["m3"], count=2
    )
    with _patch_client(client):
        res = runner.invoke(app, ["delete", "m1", "m2", "m3", "--yes"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["succeeded"] == ["m1", "m2"]
    assert payload["not_found"] == ["m3"]
    assert payload["count"] == 2
    client.models.bulk_delete.assert_called_once_with(["m1", "m2", "m3"])
    client.models.delete.assert_not_called()
