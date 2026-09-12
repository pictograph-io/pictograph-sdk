"""Tests for ``pictograph credits history`` incl. the ``--all`` iter path
(CLI parity)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pictograph.cli._config import write_config
from pictograph.cli.commands.credits import app
from pictograph.models.credit import CreditLedgerEntry


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


def _entry(op: str = "training_a10g", amount: int = -1_000_000) -> CreditLedgerEntry:
    return CreditLedgerEntry(
        id="ledger-uuid-1",
        operation=op,
        amount=amount,
        balance_after=5_000_000,
        description=None,
        metadata=None,
        created_at=datetime.now(timezone.utc),
    )


def _patch_client(client: MagicMock) -> Any:
    return patch("pictograph.cli.commands.credits.get_client", return_value=client)


def test_history_default_pages_once(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.credits.history.return_value = [_entry()]
    with _patch_client(client):
        res = runner.invoke(app, ["history", "--json", "--limit", "10"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload[0]["operation"] == "training_a10g"
    client.credits.history.assert_called_once_with(limit=10, offset=0)
    client.credits.iter.assert_not_called()


def test_history_all_uses_iter(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    # The command does ``list(client.credits.iter(...))`` - return a real list.
    client.credits.iter.return_value = [_entry("a"), _entry("b")]
    with _patch_client(client):
        res = runner.invoke(app, ["history", "--all", "--json", "--max-total", "50"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert len(payload) == 2
    client.credits.iter.assert_called_once_with(max_total=50)
    client.credits.history.assert_not_called()
