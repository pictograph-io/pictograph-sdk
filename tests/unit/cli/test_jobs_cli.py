"""``pictograph jobs list / get / wait / cancel`` (B533 #14).

The SDK client is a :class:`~unittest.mock.MagicMock` patched at the command module's
boundary; what it returns is the REAL :class:`Job` parsed from the payloads the resource
tests use, so what a command prints is what an agent reads.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pictograph.cli._app import app as root
from pictograph.cli._config import write_config
from pictograph.cli._errors import ExitCode
from pictograph.exceptions import ApiError, PollTimeoutError, ValidationError
from pictograph.models.job import Job
from tests.unit.resources.test_jobs import TRAIN_ID, job_row

if TYPE_CHECKING:
    from pathlib import Path


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
    write_config(api_key="pk_live_x")
    return fake_home


def _client(**methods: Any) -> MagicMock:
    client = MagicMock()
    for name, value in methods.items():
        setattr(client.jobs, name, value)
    return client


def _run(runner: CliRunner, client: MagicMock, *args: str) -> Any:
    with patch("pictograph.cli.commands.jobs.get_client", return_value=client):
        # A terminal as wide as a real one: the table's columns narrow to fit 80.
        return runner.invoke(root, ["jobs", *args], env={"COLUMNS": "160"})


def _job(**over: Any) -> Job:
    return Job.model_validate(job_row(**over))


def test_list_prints_every_job_and_passes_the_filters(
    runner: CliRunner, isolated_config: Path
) -> None:
    client = _client(
        list=MagicMock(return_value=iter([_job(), _job(type="video_extract", cancel_path=None)]))
    )
    res = _run(runner, client, "list", "--status", "all", "--type", "training", "-n", "5", "--json")
    assert res.exit_code == 0, res.output
    rows = json.loads(res.stdout)
    assert [r["type"] for r in rows] == ["training", "video_extract"]
    assert rows[0]["cancel_path"].endswith("/cancel") and rows[1]["cancel_path"] is None
    client.jobs.list.assert_called_once_with(
        status="all", type="training", page_size=5, max_total=5
    )


def test_list_table_marks_what_can_be_cancelled(runner: CliRunner, isolated_config: Path) -> None:
    client = _client(list=MagicMock(return_value=iter([_job()])))
    res = _run(runner, client, "list")
    assert res.exit_code == 0, res.output
    assert TRAIN_ID in res.stdout and "Swift Falcon" in res.stdout and "35%" in res.stdout
    client.jobs.list.assert_called_once_with(status="active", type=None, page_size=50, max_total=50)


def test_list_refuses_an_unknown_status_before_any_request(
    runner: CliRunner, isolated_config: Path
) -> None:
    client = _client(list=MagicMock())
    res = _run(runner, client, "list", "--status", "done")
    assert res.exit_code == ExitCode.INVALID
    client.jobs.list.assert_not_called()


def test_get_and_cancel_print_the_job(runner: CliRunner, isolated_config: Path) -> None:
    client = _client(
        get=MagicMock(return_value=_job(status="failed", error="out of memory", cancel_path=None)),
        cancel=MagicMock(return_value=_job(status="cancelled", cancel_path=None)),
    )
    res = _run(runner, client, "get", TRAIN_ID, "--json")
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["error"] == "out of memory"
    res = _run(runner, client, "cancel", TRAIN_ID, "--json")
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["status"] == "cancelled"
    client.jobs.cancel.assert_called_once_with(TRAIN_ID)


def test_cancel_of_a_job_that_cannot_be_cancelled_is_a_usage_failure(
    runner: CliRunner, isolated_config: Path
) -> None:
    client = _client(
        cancel=MagicMock(side_effect=ValidationError("A video_extract job cannot be cancelled"))
    )
    res = _run(runner, client, "cancel", TRAIN_ID, "--json")
    assert res.exit_code != 0
    assert "cannot be cancelled" in res.output


def test_wait_succeeds_fails_and_times_out_with_distinct_exit_codes(
    runner: CliRunner, isolated_config: Path
) -> None:
    done = _client(wait=MagicMock(return_value=_job(status="completed", cancel_path=None)))
    res = _run(runner, done, "wait", TRAIN_ID, "--timeout", "30", "--poll-interval", "1", "--json")
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["status"] == "completed"
    done.jobs.wait.assert_called_once_with(TRAIN_ID, poll_interval=1.0, timeout=30.0)

    slow = _client(wait=MagicMock(side_effect=PollTimeoutError("still running")))
    res = _run(runner, slow, "wait", TRAIN_ID, "--timeout", "5")
    assert res.exit_code == ExitCode.STILL_RUNNING
    assert f"pictograph jobs wait {TRAIN_ID}" in " ".join(res.output.split())

    failed = _client(
        wait=MagicMock(
            side_effect=ApiError("training job ended with status 'failed'", code="job_failed")
        )
    )
    res = _run(runner, failed, "wait", TRAIN_ID)
    assert res.exit_code not in (0, ExitCode.STILL_RUNNING)
