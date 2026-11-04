"""Tests for ``pictograph tile dataset``.

Drives the standalone ``tile`` Typer app with a real CliRunner. The pipeline
(``tile_dataset``) and the client factory are patched at the command-module
boundary - the flag→arg mapping and validation are what's under test here.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pictograph.cli.commands.tile import app
from pictograph.resources.images import TileReport


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _report() -> TileReport:
    return TileReport(source="src", target="tgt", source_images=2, tiles_created=8)


def _client() -> MagicMock:
    """A Client whose images.tile returns the canned report."""
    client = MagicMock()
    client.images.tile.return_value = _report()
    return client


def test_passes_grid_args(runner: CliRunner) -> None:
    with (
        patch("pictograph.cli.commands.tile.get_client", return_value=_client()) as gc,
    ):
        result = runner.invoke(
            app,
            [
                "dataset",
                "src",
                "--into",
                "tgt",
                "--rows",
                "3",
                "--cols",
                "2",
                "--overlap",
                "0.1",
            ],
        )
    assert result.exit_code == 0, result.output
    _args, kwargs = gc.return_value.images.tile.call_args
    assert kwargs["rows"] == 3
    assert kwargs["cols"] == 2
    assert kwargs["overlap"] == pytest.approx(0.1)
    assert kwargs["into"] == "tgt"
    assert kwargs["include_empty"] is True
    assert "tiles_created" in result.output


def test_exclude_empty_flag(runner: CliRunner) -> None:
    with (
        patch("pictograph.cli.commands.tile.get_client", return_value=_client()) as gc,
    ):
        result = runner.invoke(app, ["dataset", "src", "--exclude-empty"])
    assert result.exit_code == 0, result.output
    _args, kwargs = gc.return_value.images.tile.call_args
    assert kwargs["include_empty"] is False


def test_rejects_zero_rows(runner: CliRunner) -> None:
    with patch("pictograph.cli.commands.tile.get_client", return_value=MagicMock()):
        result = runner.invoke(app, ["dataset", "src", "--rows", "0"])
    assert result.exit_code != 0
    assert "must both be >= 1" in result.output
