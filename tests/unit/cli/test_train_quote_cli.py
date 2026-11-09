"""``pictograph train quote`` and the ``--gpu auto`` / ``--config <file>`` it shares with
``train start`` (B533 #10).

The SDK client is a :class:`~unittest.mock.MagicMock` patched at the ``train``
command-module boundary; the quote it returns is the REAL model parsed from the payload
the resource tests use, so what the command prints is what an agent reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pictograph.cli._config import write_config
from pictograph.cli.commands.train import app
from pictograph.models.training import TrainingQuote
from tests.unit.resources.test_training import _quote_payload, _run_payload


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


def _client(**quote_overrides: Any) -> MagicMock:
    client = MagicMock()
    client.training.quote.return_value = TrainingQuote.model_validate(
        _quote_payload(**quote_overrides)
    )
    return client


def _patch(client: MagicMock) -> Any:
    return patch("pictograph.cli.commands.train.get_client", return_value=client)


def test_quote_prints_the_quote_and_passes_the_plan(
    runner: CliRunner, isolated_config: Path
) -> None:
    client = _client()
    with _patch(client):
        result = runner.invoke(
            app,
            [
                "quote",
                "road-signs",
                "v1",
                "--pipeline",
                "rfdetr_detection",
                "--gpu",
                "auto",
                "--config",
                '{"epochs": 12}',
            ],
        )
    assert result.exit_code == 0, result.output
    printed = json.loads(result.stdout)
    assert printed["gpu_type"] == "a10g" and printed["estimate"]["total_micro_usd"] == 880_000
    assert printed["can_start"] is True
    client.training.quote.assert_called_once_with(
        "road-signs", "v1", pipeline_type="rfdetr_detection", gpu_type="auto", config={"epochs": 12}
    )
    assert "cannot start" not in result.stderr


def test_a_blocked_quote_still_succeeds_and_says_why_on_stderr(
    runner: CliRunner, isolated_config: Path
) -> None:
    blocked = [{"code": "insufficient_credits", "message": "Training needs $2.00 to start."}]
    with _patch(_client(can_start=False, blocking=blocked)):
        result = runner.invoke(app, ["quote", "road-signs", "v1", "-p", "yolox"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["can_start"] is False  # stdout stays one JSON object
    assert "insufficient_credits" in result.stderr


def test_quote_reads_a_config_file(
    runner: CliRunner, isolated_config: Path, tmp_path: Path
) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"epochs": 3}), encoding="utf-8")
    client = _client()
    with _patch(client):
        result = runner.invoke(
            app, ["quote", "road-signs", "v1", "-p", "yolox", "--config", str(cfg)]
        )
    assert result.exit_code == 0, result.output
    # The path goes to the SDK, which reads it the way create reads a config.json.
    assert client.training.quote.call_args.kwargs["config"] == cfg


@pytest.mark.parametrize(
    ("value", "says"),
    [
        ('{"epochs": ', "not valid JSON"),
        ("no/such/config.json", "neither a JSON object nor a file"),
    ],
)
def test_a_bad_config_is_a_usage_error_before_any_request(
    runner: CliRunner, isolated_config: Path, value: str, says: str
) -> None:
    client = _client()
    with _patch(client):
        result = runner.invoke(
            app, ["quote", "road-signs", "v1", "-p", "yolox", "--config", value, "--json"]
        )
    assert result.exit_code == 6, result.output
    error = json.loads(result.stderr.strip().splitlines()[-1])["error"]
    assert error["code"] == "usage_error" and says in error["message"]
    assert error["details"] == {"param": "--config"}
    client.training.quote.assert_not_called()


def test_quote_accepts_json(runner: CliRunner, isolated_config: Path) -> None:
    """The quote is JSON already; ``--json`` is accepted so an agent can always pass it."""
    with _patch(_client()):
        plain = runner.invoke(app, ["quote", "road-signs", "v1", "-p", "yolox"])
        flagged = runner.invoke(app, ["quote", "road-signs", "v1", "-p", "yolox", "--json"])
    assert plain.exit_code == flagged.exit_code == 0
    assert plain.stdout == flagged.stdout


def test_start_takes_auto_and_a_config_file(
    runner: CliRunner, isolated_config: Path, tmp_path: Path
) -> None:
    from pictograph.models.training import TrainingRun

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"epochs": 3}), encoding="utf-8")
    client = MagicMock()
    client.training.create.return_value = TrainingRun.model_validate(_run_payload())
    with _patch(client):
        result = runner.invoke(
            app,
            [
                "start",
                "road-signs",
                "v1",
                "-p",
                "yolox",
                "--gpu",
                "auto",
                "--config",
                str(cfg),
                "--no-wait",
            ],
        )
    assert result.exit_code == 0, result.output
    kwargs = client.training.create.call_args.kwargs
    assert kwargs["gpu_type"] == "auto" and kwargs["config"] == cfg


def test_the_gpu_help_names_auto_and_the_quote(runner: CliRunner) -> None:
    for command in ("start", "quote"):
        result = runner.invoke(app, [command, "--help"], env={"COLUMNS": "200"})
        assert result.exit_code == 0
        assert "auto" in result.stdout and "train quote" in result.stdout, command
