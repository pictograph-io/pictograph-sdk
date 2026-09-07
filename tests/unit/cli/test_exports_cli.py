"""Tests for the ``pictograph exports`` command group.

Self-contained: a :class:`~typer.testing.CliRunner` invokes the command group
in-process, the SDK client is a :class:`~unittest.mock.MagicMock` patched at the
``exports`` command-module boundary, and ``~/.pictograph`` is redirected to a
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
from pictograph.cli.commands.exports import app
from pictograph.models.export import Export


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


def _export(name: str = "nightly", *, status: str = "completed") -> Export:
    return Export(
        id="exp-uuid",
        project_id="proj-uuid",
        dataset_name="ds",
        name=name,
        format="coco",
        status=status,  # type: ignore[arg-type]
        created_at=datetime.now(timezone.utc),
        image_count=4,
        file_size=2048,
    )


def _patch_client(client: MagicMock) -> Any:
    """Patch ``get_client`` in the exports command-module namespace."""
    return patch("pictograph.cli.commands.exports.get_client", return_value=client)


# ───────────── list ─────────────


def test_list_renders_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.exports.list.return_value = [_export("a"), _export("b")]
    with _patch_client(client):
        res = runner.invoke(app, ["list", "--json", "--dataset", "ds"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert len(payload) == 2
    client.exports.list.assert_called_once_with(dataset_name="ds", status=None, limit=100)


# ───────────── get ─────────────


def test_get(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.exports.get.return_value = _export("nightly")
    with _patch_client(client):
        res = runner.invoke(app, ["get", "ds", "nightly"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["name"] == "nightly"
    client.exports.get.assert_called_once_with("ds", "nightly")


# ───────────── create ─────────────


def test_create_waits_for_completion(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.exports.create.return_value = _export("nightly")
    with _patch_client(client):
        res = runner.invoke(
            app,
            ["create", "ds", "--name", "nightly", "--format", "coco", "--class-filter", "car,dog"],
        )
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["status"] == "completed"
    client.exports.create.assert_called_once_with(
        "ds",
        "nightly",
        format="coco",
        include_images=False,
        class_filter=["car", "dog"],
        status_filter=None,
        organize_by_split=False,
        wait=True,
        poll_interval=2.0,
        timeout=300.0,
    )


def test_create_no_wait(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.exports.create.return_value = _export("nightly", status="pending")
    with _patch_client(client):
        res = runner.invoke(app, ["create", "ds", "--name", "nightly", "--no-wait"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["status"] == "pending"
    assert client.exports.create.call_args.kwargs["wait"] is False


# ───────────── download ─────────────


def test_download_writes_summary(runner: CliRunner, isolated_config: Path, tmp_path: Path) -> None:
    write_config(api_key="pk_live_x")
    out = tmp_path / "nightly.zip"
    out.write_bytes(b"zipbytes")  # the resource is mocked; simulate the landed file
    client = MagicMock()
    client.exports.download.return_value = out
    with _patch_client(client):
        res = runner.invoke(app, ["download", "ds", "nightly", "-o", str(out)])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["path"] == str(out)
    assert payload["bytes"] == 8
    client.exports.download.assert_called_once_with("ds", "nightly", out)


# ───────────── delete ─────────────


def test_delete_with_yes(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    with _patch_client(client):
        res = runner.invoke(app, ["delete", "ds", "nightly", "--yes"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["deleted"] is True
    client.exports.delete.assert_called_once_with("ds", "nightly")


def test_bulk_delete_with_yes(runner: CliRunner, isolated_config: Path) -> None:
    from pictograph.models.common import BulkDeleteResult

    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.exports.bulk_delete.return_value = BulkDeleteResult(
        succeeded=["e1", "e2"], not_found=[], count=2
    )
    with _patch_client(client):
        res = runner.invoke(app, ["bulk-delete", "e1", "e2", "--yes"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["succeeded"] == ["e1", "e2"]
    assert payload["count"] == 2
    client.exports.bulk_delete.assert_called_once_with(["e1", "e2"])
