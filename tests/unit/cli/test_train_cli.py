"""Tests for the ``pictograph train {list,wait}`` commands.

Self-contained: a :class:`~typer.testing.CliRunner` invokes the command group
in-process, the SDK client is a :class:`~unittest.mock.MagicMock` patched at the
``train`` command-module boundary, and ``~/.pictograph`` is redirected to a
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
from pictograph.cli.commands.train import app
from pictograph.models.training import TrainingRun


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


def _run(name: str = "yolox-run", *, status: str = "completed") -> TrainingRun:
    return TrainingRun(
        id="run-uuid-1",
        organization_id="org-uuid",
        name=name,
        dataset_id="proj-uuid-1",
        export_id="export-uuid-1",
        model_id="model-uuid-1" if status == "completed" else None,
        pipeline_type="yolox",
        gpu_type="a10g",
        status=status,  # type: ignore[arg-type]
        progress=100 if status == "completed" else 50,
        current_epoch=10,
        total_epochs=10,
        metrics={"mAP": 0.85},
        config={},
        created_at=datetime.now(timezone.utc),
    )


def _patch_client(client: MagicMock) -> Any:
    """Patch ``get_client`` in the train command-module namespace."""
    return patch("pictograph.cli.commands.train.get_client", return_value=client)


# ───────────── list ─────────────


def test_list_renders_json_with_filters(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.training.list.return_value = [_run("a"), _run("b", status="running")]
    with _patch_client(client):
        res = runner.invoke(app, ["list", "--json", "--dataset", "ds", "--status", "running"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert len(payload) == 2
    client.training.list.assert_called_once_with(dataset_name="ds", status="running", limit=50)


def test_list_renders_table(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.training.list.return_value = [_run("nightly")]
    with _patch_client(client):
        res = runner.invoke(app, ["list"])
    assert res.exit_code == 0, res.stdout
    assert "nightly" in res.stdout
    client.training.list.assert_called_once_with(dataset_name=None, status=None, limit=50)


# ───────────── wait ─────────────


def test_wait_polls_to_completion(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.training.wait_for_completion.return_value = _run("nightly")
    with _patch_client(client):
        res = runner.invoke(app, ["wait", "run-uuid-1", "--poll-interval", "1", "--timeout", "30"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["status"] == "completed"
    client.training.wait_for_completion.assert_called_once_with(
        "run-uuid-1", poll_interval=1.0, timeout=30.0
    )
