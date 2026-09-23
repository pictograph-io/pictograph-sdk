"""Tests for ``pictograph auto-annotate {get,cancel-batch}`` (CLI parity)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pictograph.cli._config import write_config
from pictograph.cli.commands.auto_annotate import app
from pictograph.models.auto_annotate import BatchJob


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("PICTOGRAPH_API_KEY", raising=False)
    monkeypatch.setattr("pictograph.cli._config.CONFIG_DIR", fake_home / ".pictograph")
    monkeypatch.setattr(
        "pictograph.cli._config.CONFIG_PATH", fake_home / ".pictograph" / "config.toml"
    )
    return fake_home


def _job(status: str = "completed") -> BatchJob:
    return BatchJob(
        job_id="job-uuid-1",
        status=status,
        progress=100 if status == "completed" else 0,
        total_images=2,
        processed_images=2 if status == "completed" else 0,
        total_annotations_added=3,
        failed_images=0,
        error_message=None,
        estimated_credits=None,
        completed_at=datetime.now(timezone.utc) if status == "completed" else None,
    )


def _patch_client(client: MagicMock) -> Any:
    return patch("pictograph.cli.commands.auto_annotate.get_client", return_value=client)


def test_get_renders_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.auto_annotate.get_batch.return_value = _job()
    with _patch_client(client):
        res = runner.invoke(app, ["get", "job-uuid-1"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["job_id"] == "job-uuid-1"
    assert payload["status"] == "completed"
    client.auto_annotate.get_batch.assert_called_once_with("job-uuid-1")


def test_cancel_batch_with_yes(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.auto_annotate.cancel_batch.return_value = _job("cancelled")
    with _patch_client(client):
        res = runner.invoke(app, ["cancel-batch", "job-uuid-1", "--yes"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["status"] == "cancelled"
    client.auto_annotate.cancel_batch.assert_called_once_with("job-uuid-1")


def test_cancel_batch_aborts_without_confirmation(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    with _patch_client(client):
        res = runner.invoke(app, ["cancel-batch", "job-uuid-1"], input="n\n")
    assert res.exit_code != 0
    client.auto_annotate.cancel_batch.assert_not_called()
