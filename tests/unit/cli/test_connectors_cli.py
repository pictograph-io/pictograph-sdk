"""Tests for the ``pictograph connectors`` command group.

Self-contained: each test drives the connectors Typer app via
``CliRunner``, patches ``get_client`` at the connectors command-module
boundary, and asserts on the JSON written to stdout. One happy-path test
per command.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pictograph.cli._config import write_config
from pictograph.cli.commands.connectors import app
from pictograph.models.connector import (
    ImportJob,
    LimitCheckResult,
    RemoteDataset,
    ValidationResult,
)

_PATCH_TARGET = "pictograph.cli.commands.connectors.get_client"


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


def _remote_dataset(ds_id: str = "ds-1", name: str = "cats") -> RemoteDataset:
    return RemoteDataset(id=ds_id, name=name, slug=name, image_count=42, version=None)


def _validation(valid: bool = True) -> ValidationResult:
    return ValidationResult(
        valid=valid,
        workspace="my-workspace",
        datasets=[_remote_dataset("ds-1", "cats"), _remote_dataset("ds-2", "dogs")]
        if valid
        else [],
        error=None if valid else "Invalid API key",
    )


def _limit_check() -> LimitCheckResult:
    return LimitCheckResult(
        allowed=True,
        current_images=10,
        image_limit=1000,
        images_after_import=110,
        current_storage_bytes=1000,
        storage_limit_bytes=10_000_000,
        storage_after_import_bytes=2_000_000,
        exceeded=None,
    )


def _import_job(status: str = "completed", import_id: str = "imp-1") -> ImportJob:
    return ImportJob(
        import_id=import_id,
        status=status,  # type: ignore[arg-type]
        progress=100.0 if status == "completed" else 0.0,
        total_images=42,
        imported_images=42 if status == "completed" else 0,
        failed_images=0,
    )


# ───────────── validate ─────────────


def test_validate_renders_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.connectors.validate.return_value = _validation()
    with patch(_PATCH_TARGET, return_value=client):
        res = runner.invoke(app, ["validate", "v7", "--key", "v7-token", "--json"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["valid"] is True
    assert len(payload["datasets"]) == 2
    client.connectors.validate.assert_called_once_with("v7", "v7-token")


# ───────────── check-limits ─────────────


def test_check_limits_renders_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.connectors.check_limits.return_value = _limit_check()
    with patch(_PATCH_TARGET, return_value=client):
        res = runner.invoke(app, ["check-limits", "--images", "100", "--bytes", "2000000"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["allowed"] is True
    client.connectors.check_limits.assert_called_once_with(
        total_images=100, estimated_size_bytes=2_000_000
    )


# ───────────── import ─────────────


def test_import_resolves_dataset_ids_and_waits(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.connectors.validate.return_value = _validation()
    client.connectors.import_.return_value = _import_job("completed")
    with patch(_PATCH_TARGET, return_value=client):
        res = runner.invoke(
            app,
            ["import", "v7", "--key", "v7-token", "--dataset", "ds-1"],
        )
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["status"] == "completed"
    # Resolved the id against validate(), then imported the full RemoteDataset spec.
    client.connectors.validate.assert_called_once_with("v7", "v7-token")
    args, kwargs = client.connectors.import_.call_args
    assert args[0] == "v7"
    assert args[1] == "v7-token"
    assert [d.id for d in args[2]] == ["ds-1"]
    assert kwargs["wait"] is True


def test_import_no_wait_skips_poll(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.connectors.validate.return_value = _validation()
    client.connectors.import_.return_value = _import_job("processing")
    with patch(_PATCH_TARGET, return_value=client):
        res = runner.invoke(
            app,
            ["import", "v7", "--key", "v7-token", "--dataset", "ds-1", "--no-wait"],
        )
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["status"] == "processing"
    assert client.connectors.import_.call_args.kwargs["wait"] is False


def test_import_with_dataset_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.connectors.import_.return_value = _import_job("completed")
    spec = json.dumps([{"id": "ds-9", "name": "n", "slug": "s"}])
    with patch(_PATCH_TARGET, return_value=client):
        res = runner.invoke(
            app,
            ["import", "roboflow", "--key", "rf-key", "--dataset-json", spec],
        )
    assert res.exit_code == 0, res.stdout
    # --dataset-json path does not call validate; spec passed straight through.
    client.connectors.validate.assert_not_called()
    args, _ = client.connectors.import_.call_args
    assert args[0] == "roboflow"
    assert args[2] == [{"id": "ds-9", "name": "n", "slug": "s"}]


def test_import_requires_a_dataset(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    with patch(_PATCH_TARGET, return_value=client):
        res = runner.invoke(app, ["import", "v7", "--key", "v7-token"])
    assert res.exit_code != 0
    client.connectors.import_.assert_not_called()


# ───────────── status ─────────────


def test_status_renders_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.connectors.get_import.return_value = _import_job("processing", "imp-7")
    with patch(_PATCH_TARGET, return_value=client):
        res = runner.invoke(app, ["status", "imp-7"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["import_id"] == "imp-7"
    client.connectors.get_import.assert_called_once_with("imp-7")


# ───────────── cancel ─────────────


def test_cancel_with_yes_flag(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.connectors.cancel_import.return_value = _import_job("cancelled", "imp-3")
    with patch(_PATCH_TARGET, return_value=client):
        res = runner.invoke(app, ["cancel", "imp-3", "--yes"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["status"] == "cancelled"
    client.connectors.cancel_import.assert_called_once_with("imp-3")
