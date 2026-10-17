"""Tests for ``pictograph augment dataset``.

Drives the standalone ``augment`` Typer app with a real CliRunner. The pipeline
(``augment_dataset``) and the client factory are patched at the command-module
boundary - the flag→op mapping and validation are what's under test here.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pictograph.cli.commands.augment import app
from pictograph.resources.images import AugmentReport


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _report() -> AugmentReport:
    return AugmentReport(source="src", target="tgt", source_images=2, variants_created=6)


def _client() -> MagicMock:
    """A Client whose images.augment returns the canned report."""
    client = MagicMock()
    client.images.augment.return_value = _report()
    return client


def test_builds_ops_from_flags(runner: CliRunner) -> None:
    with (
        patch("pictograph.cli.commands.augment.get_client", return_value=_client()) as gc,
    ):
        result = runner.invoke(
            app,
            [
                "dataset",
                "src",
                "--into",
                "tgt",
                "-m",
                "3",
                "--flip",
                "--rotate",
                "15",
                "--brightness",
                "0.2",
                "--resize",
                "640x480",
                "--seed",
                "7",
            ],
        )
    assert result.exit_code == 0, result.output
    args, kwargs = gc.return_value.images.augment.call_args
    op_names = [op.name for op in args[1]]
    # resize is applied before flip/rotate; photometric last
    assert op_names == ["resize", "horizontal_flip", "rotate", "brightness"]
    assert kwargs["multiplier"] == 3
    assert kwargs["into"] == "tgt"
    assert kwargs["seed"] == 7
    assert "variants_created" in result.output


def test_no_ops_is_error(runner: CliRunner) -> None:
    with patch("pictograph.cli.commands.augment.get_client", return_value=MagicMock()):
        result = runner.invoke(app, ["dataset", "src", "--into", "tgt"])
    assert result.exit_code != 0
    assert "at least one augmentation" in result.output


def test_bad_resize_is_error(runner: CliRunner) -> None:
    with (
        patch("pictograph.cli.commands.augment.get_client", return_value=_client()) as gc,
    ):
        result = runner.invoke(app, ["dataset", "src", "--resize", "not-a-size"])
    assert result.exit_code != 0
    assert "WIDTHxHEIGHT" in result.output


def test_in_place_passes_no_original(runner: CliRunner) -> None:
    with (
        patch("pictograph.cli.commands.augment.get_client", return_value=_client()) as gc,
    ):
        result = runner.invoke(app, ["dataset", "src", "--grayscale", "--no-original"])
    assert result.exit_code == 0, result.output
    args, kwargs = gc.return_value.images.augment.call_args
    assert kwargs["include_original"] is False
    assert [op.name for op in args[1]] == ["grayscale"]
