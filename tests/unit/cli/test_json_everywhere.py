"""Every command accepts ``--json`` (B533): an agent can pass it anywhere.

Two kinds of command, and both are held here:

* a command that prints a table or a sentence has a REAL ``--json`` of its own (its
  callback reads it and prints JSON instead);
* a command that only ever prints JSON is given an accepted-and-ignored ``--json`` by
  ``cli/_errors.py::accept_json_everywhere`` - click used to refuse it there ("No such
  option: --json", exit 6), the failure an agent hit on ``images list --json``.

The second kind is only honest while the command really prints nothing but JSON on
stdout, so every command that gets the injected flag is READ: its callback, and every
function of its own module it calls, may not print text to stdout (``typer.echo`` without
``err=True``, ``print_table``, the stdout ``console``, ``print``). A command that starts
printing text must take a real ``--json`` first - this test names it.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.main import get_command
from typer.testing import CliRunner

from pictograph.cli._app import app
from pictograph.cli._config import write_config
from pictograph.cli._errors import JSON_ALWAYS_HELP, accept_json_everywhere, has_json_flag


def _leaves(command: Any, path: tuple[str, ...] = ()) -> list[tuple[str, Any]]:
    subcommands = getattr(command, "commands", None)
    if subcommands:
        return [leaf for name, sub in subcommands.items() for leaf in _leaves(sub, (*path, name))]
    return [(" ".join(path), command)]


@pytest.fixture(scope="module")
def root() -> Any:
    command = get_command(app)
    declared = {name for name, cmd in _leaves(command) if has_json_flag(cmd)}
    accept_json_everywhere(command)
    command.declared_json = declared  # what each command declared before the injection
    return command


def _injected(command: Any) -> bool:
    return any(
        "--json" in (p.opts or ()) and getattr(p, "help", None) == JSON_ALWAYS_HELP
        for p in command.params
    )


def test_every_command_accepts_json(root: Any) -> None:
    leaves = _leaves(root)
    assert len(leaves) > 150, f"only {len(leaves)} commands found - the walk is not looking"
    missing = [name for name, cmd in leaves if not has_json_flag(cmd)]
    assert missing == [], f"commands that still refuse --json: {missing}"


def test_the_injection_is_idempotent(root: Any) -> None:
    accept_json_everywhere(root)
    doubled = [
        name
        for name, cmd in _leaves(root)
        if sum("--json" in (p.opts or ()) for p in cmd.params) != 1
    ]
    assert doubled == []


# ── a command given the flag prints only JSON ──────────────────────────────────

_STDOUT_TEXT = ("print_table", "print")


def _prints_text(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name) and func.id in _STDOUT_TEXT:
        return func.id
    if isinstance(func, ast.Attribute):
        owner = func.value
        if func.attr == "echo" and isinstance(owner, ast.Name) and owner.id == "typer":
            to_stderr = any(
                k.arg == "err" and isinstance(k.value, ast.Constant) and k.value.value is True
                for k in call.keywords
            )
            return None if to_stderr else "typer.echo"
        if func.attr == "print" and isinstance(owner, ast.Name) and owner.id == "console":
            return "console.print"
        if func.attr == "write" and isinstance(owner, ast.Attribute) and owner.attr == "stdout":
            return "sys.stdout.write"
    return None


def _text_output(function: Any) -> list[str]:
    """Stdout text calls in ``function`` and in every function of its own module it
    calls (transitively) - a helper like ``finish_import`` counts."""
    module: ModuleType | None = inspect.getmodule(function)
    assert module is not None
    local = {
        name: obj
        for name, obj in vars(module).items()
        if inspect.isfunction(obj) and obj.__module__ == module.__name__
    }
    found: list[str] = []
    seen: set[str] = set()
    pending = [function]
    while pending:
        current = pending.pop()
        if current.__name__ in seen:
            continue
        seen.add(current.__name__)
        tree = ast.parse(textwrap.dedent(inspect.getsource(current)))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            kind = _prints_text(node)
            if kind:
                found.append(f"{current.__name__}: {kind}")
            if isinstance(node.func, ast.Name) and node.func.id in local:
                pending.append(local[node.func.id])
    return found


def test_a_command_given_the_flag_prints_only_json(root: Any) -> None:
    injected = [(name, cmd) for name, cmd in _leaves(root) if _injected(cmd)]
    assert len(injected) > 80, f"only {len(injected)} commands got the flag - not looking"
    offenders = {name: found for name, cmd in injected if (found := _text_output(cmd.callback))}
    assert offenders == {}, (
        "these commands print text to stdout but have no --json of their own; give each a "
        f"real --json before it prints anything but JSON: {offenders}"
    )


def test_no_declared_flag_is_shadowed(root: Any) -> None:
    """A command with its own --json keeps it - the injection never adds a second one."""
    declared = root.declared_json
    assert len(declared) > 40
    for name, cmd in _leaves(root):
        if name in declared:
            assert not _injected(cmd), name


def test_the_text_check_sees_text() -> None:
    """The control: the reader flags every way a command prints text to stdout."""

    def sneaky() -> None:
        typer.echo("done")  # stdout

    def table() -> None:
        from pictograph.cli._format import print_table

        print_table([], title="x")

    def quiet() -> None:
        typer.echo("progress", err=True)

    assert _text_output(sneaky) == ["sneaky: typer.echo"]
    assert _text_output(table) == ["table: print_table"]
    assert _text_output(quiet) == []


# ── the flag changes nothing, and an agent sees it ─────────────────────────────


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


def test_images_list_takes_json_and_prints_the_same(isolated_config: Path) -> None:
    """The command the gap was found on (B533 follow-up c)."""
    client = MagicMock()
    client.images.iter.return_value.all.return_value = []
    runner = CliRunner()
    with patch("pictograph.cli.commands.images.get_client", return_value=client):
        plain = runner.invoke(app, ["images", "list", "demo"])
        flagged = runner.invoke(app, ["images", "list", "demo", "--json"])
    assert plain.exit_code == 0, plain.output
    assert flagged.exit_code == 0, flagged.output
    assert flagged.stdout == plain.stdout
    json.loads(flagged.stdout)


def test_help_shows_the_flag() -> None:
    result = CliRunner().invoke(app, ["annotations", "get", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "--json" in result.stdout and "always prints JSON" in result.stdout


@pytest.mark.parametrize(
    "argv",
    [
        ["notifications", "unread-count", "--json"],
        ["augment", "ops", "--json"],
        ["agents", "list-tools", "--json"],
    ],
)
def test_text_commands_print_real_json(argv: list[str], isolated_config: Path) -> None:
    client = MagicMock()
    client.notifications.unread_count.return_value = 3
    with patch("pictograph.cli.commands.notifications.get_client", return_value=client):
        result = CliRunner().invoke(app, argv)
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    if argv[0] == "notifications":
        assert parsed == {"unread": 3}
    else:
        assert isinstance(parsed, list) and parsed and isinstance(parsed[0], dict)
