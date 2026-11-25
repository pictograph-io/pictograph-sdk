"""A CLI choice is a ``str`` checked in its command, never a ``typing.Literal`` parameter (``cli/_choices.py``).

typer builds every command when the CLI starts, and a ``Literal`` parameter needs a typer newer than the SDK accepts
(``typer>=0.12``): on an older one (0.17.4, the landing's docs gate, 2026-11-25) the WHOLE CLI fails to start -
"RuntimeError: Type not yet supported: typing.Literal['png', 'float16']" - every command with it, for every user
whose environment holds such a typer. This suite runs a newer typer, which builds a Literal happily, so the rule is
checked on the annotations themselves.
"""

from __future__ import annotations

import json
import typing
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from pictograph.cli._app import app
from pictograph.cli._choices import choice
from pictograph.cli._config import write_config
from pictograph.cli._errors import ExitCode


def _commands(t: typer.Typer, path: str = "pictograph") -> Iterator[tuple[str, Callable[..., Any]]]:
    """Every command's callback in the app, groups walked (their own callbacks included)."""
    if t.registered_callback is not None and t.registered_callback.callback is not None:
        yield f"{path} (callback)", t.registered_callback.callback
    for info in t.registered_commands:
        if info.callback is not None:
            yield f"{path} {info.name or info.callback.__name__}", info.callback
    for group in t.registered_groups:
        sub = group.typer_instance
        if sub is None:
            continue
        name = group.name or (sub.info.name if sub.info is not None else None) or "?"
        yield from _commands(sub, f"{path} {name}")


def _literal_in(annotation: object) -> bool:
    if typing.get_origin(annotation) is typing.Literal:
        return True
    return any(_literal_in(arg) for arg in typing.get_args(annotation))


def test_no_command_takes_a_literal_parameter() -> None:
    commands = list(_commands(app))
    assert len(commands) > 100, "the walk must reach the CLI's commands"
    offenders = []
    for name, callback in commands:
        hints = typing.get_type_hints(callback, include_extras=True)
        offenders += [
            f"{name}: {param}"
            for param, hint in hints.items()
            if param != "return" and _literal_in(hint)
        ]
    assert offenders == [], (
        f"a Literal parameter breaks the whole CLI on typer<0.19 - use cli/_choices.py: {offenders}"
    )


def test_a_choice_is_its_word_lowercased_or_refused_naming_the_flag() -> None:
    allowed: tuple[typing.Literal["png", "float16"], ...] = ("png", "float16")
    assert choice("--anomaly-map", None, allowed) is None
    assert choice("--anomaly-map", " PNG ", allowed) == "png"
    assert choice("--anomaly-map", "float16", allowed) == "float16"
    with pytest.raises(typer.BadParameter) as e:
        choice("--anomaly-map", "jpeg", allowed)
    assert e.value.param_hint == "--anomaly-map" and "must be one of png, float16" in str(e.value)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """~/.pictograph/* in tmp_path, as test_app.py keeps it."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("PICTOGRAPH_API_KEY", raising=False)
    monkeypatch.setattr("pictograph.cli._config.CONFIG_DIR", fake_home / ".pictograph")
    monkeypatch.setattr(
        "pictograph.cli._config.CONFIG_PATH", fake_home / ".pictograph" / "config.toml"
    )
    return fake_home


def test_predict_checks_the_map_encoding_before_any_request(
    runner: CliRunner, isolated_config: Path
) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    with patch("pictograph.cli.commands.deployments.get_client", return_value=client) as get_client:
        res = runner.invoke(app, ["deployments", "predict", "dep-1", "https://img.test/x.jpg",
                                  "--token", "pk_deploy_t", "--anomaly-map", "jpeg"])  # fmt: skip
    assert res.exit_code == ExitCode.INVALID, (
        res.output
    )  # a bad flag value: the CLI's own exit code
    assert "must be one of png, float16" in res.output
    get_client.assert_not_called()


def test_predict_asks_for_the_map_in_its_canonical_spelling(
    runner: CliRunner, isolated_config: Path
) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    conn = MagicMock()
    conn.infer_raw.return_value = {"model_type": "anomaly_detection", "anomaly_score": 0.7}
    client.deployments.connect.return_value = conn
    with patch("pictograph.cli.commands.deployments.get_client", return_value=client):
        res = runner.invoke(app, ["deployments", "predict", "dep-1", "https://img.test/x.jpg",
                                  "--token", "pk_deploy_t", "--anomaly-map", "Float16"])  # fmt: skip
    assert res.exit_code == 0, res.output
    assert conn.infer_raw.call_args.kwargs["anomaly_map"] == "float16"
    assert json.loads(res.stdout)["anomaly_score"] == 0.7
