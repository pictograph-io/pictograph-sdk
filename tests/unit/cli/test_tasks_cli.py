"""Tests for ``pictograph tasks {list,contributions}``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pictograph.cli._config import write_config
from pictograph.cli.commands.tasks import app
from pictograph.models.task import Task, TaskContribution, TaskContributions


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


def _patch_client(client: MagicMock) -> Any:
    return patch("pictograph.cli.commands.tasks.get_client", return_value=client)


def _task(**overrides: Any) -> Task:
    base: dict[str, Any] = {
        "id": "t1",
        "project_id": "p1",
        "project": "Demo",
        "title": "Label the cars",
        "kind": "annotate",
        "status": "open",
        "created_at": "2026-08-05T00:00:00Z",
        "image_count": 8,
        "assignee_count": 2,
    }
    base.update(overrides)
    return Task(**base)


def _contrib() -> TaskContributions:
    return TaskContributions(
        task_id="t1",
        contributors=[
            TaskContribution(
                user_id="u1",
                full_name="Ann",
                email=None,
                avatar_url=None,
                is_assignee=True,
                active_seconds=90,
                images_worked=4,
                images_completed=2,
                annotations_added=12,
            )
        ],
        contributor_count=1,
        total_images=8,
        images_complete=2,
        total_active_seconds=90,
    )


def test_list_json_pages_once(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.tasks.list.return_value = [_task()]
    with _patch_client(client):
        res = runner.invoke(app, ["list", "--json"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload[0]["title"] == "Label the cars"
    client.tasks.list.assert_called_once_with(limit=50, offset=0)
    client.tasks.iter.assert_not_called()


def test_list_all_uses_iter(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.tasks.iter.return_value = [_task(), _task(id="t2")]
    with _patch_client(client):
        res = runner.invoke(app, ["list", "--all", "--json", "--max-total", "5"])
    assert res.exit_code == 0, res.stdout
    assert len(json.loads(res.stdout)) == 2
    client.tasks.iter.assert_called_once_with(max_total=5)
    client.tasks.list.assert_not_called()


def test_contributions_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.tasks.contributions.return_value = _contrib()
    with _patch_client(client):
        res = runner.invoke(app, ["contributions", "t1", "--json"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["task_id"] == "t1"
    assert payload["contributors"][0]["full_name"] == "Ann"
    client.tasks.contributions.assert_called_once_with("t1")


def test_contributions_table_renders(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.tasks.contributions.return_value = _contrib()
    with _patch_client(client):
        res = runner.invoke(app, ["contributions", "t1"])
    assert res.exit_code == 0, res.stdout
    assert "Ann" in res.stdout  # annotator name renders in the table
    client.tasks.contributions.assert_called_once_with("t1")
