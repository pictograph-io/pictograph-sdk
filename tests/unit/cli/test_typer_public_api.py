"""The CLI builds on Typer's public API only, so a Typer release cannot break it at startup.

Typer 0.29 (2026-11-09) added a required keyword (``type_descriptor``) to the constructor
of its internal ``TyperOption``. The CLI built one ``--json`` flag by hand, so every
``pictograph`` command - ``--help`` included - crashed at startup on every new install
(``pip install 'pictograph[cli]'`` resolves the newest Typer). The same release changed
``Parameter.get_error_hint`` to take no ``ctx``, which silently dropped ``details.param``
from every JSON usage error.

These tests hold the two seams: no Typer parameter class is constructed in the CLI's own
code (options and arguments come from ``typer.Option`` / ``typer.Argument``, which Typer
turns into parameters with whatever its version needs), and the usage-error hint is read
from a parameter of either shape.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from pictograph.cli import _errors
from pictograph.cli._errors import JSON_ALWAYS_HELP, _accepted_json_flag, _usage_param

CLI = Path(_errors.__file__).resolve().parent

# Typer's parameter classes: their constructors are internal (0.29 changed one).
_INTERNAL_PARAMETER_CLASSES = {"TyperOption", "TyperArgument"}


def _cli_modules() -> list[Path]:
    modules = sorted(CLI.rglob("*.py"))
    assert len(modules) > 20, f"only {len(modules)} CLI modules found - the walk is not looking"
    return modules


def test_no_cli_module_builds_a_typer_parameter_by_hand() -> None:
    offenders: list[str] = []
    for path in _cli_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("typer"):
                for alias in node.names:
                    if alias.name in _INTERNAL_PARAMETER_CLASSES:
                        offenders.append(f"{path.name}:{node.lineno} imports {alias.name}")
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if name in _INTERNAL_PARAMETER_CLASSES:
                    offenders.append(f"{path.name}:{node.lineno} calls {name}(...)")
    assert offenders == [], (
        "build options and arguments with typer.Option / typer.Argument (see "
        f"cli/_errors.py::_accepted_json_flag), never Typer's own classes: {offenders}"
    )


def test_the_accepted_json_flag_is_a_flag_the_command_never_receives() -> None:
    flag = _accepted_json_flag()
    assert flag.opts == ["--json"]
    assert flag.name == "json"
    assert flag.is_flag is True
    assert flag.expose_value is False
    assert flag.default is False
    assert flag.help == JSON_ALWAYS_HELP


class _Ctx:
    command_path = "pictograph datasets list"


class _ClickStyleParam:
    """click's ``Parameter.get_error_hint(ctx)``."""

    def __init__(self) -> None:
        self.asked_with: list[Any] = []

    def get_error_hint(self, ctx: Any) -> str:
        self.asked_with.append(ctx)
        return "'--limit' / '-n'"


class _TyperStyleParam:
    """Typer 0.29's own parameters: ``get_error_hint()``, no ``ctx``."""

    def get_error_hint(self) -> str:
        return "'--limit' / '-n'"


class _BadParameterError(Exception):
    def __init__(self, param: Any) -> None:
        super().__init__("Input should be greater than or equal to 1")
        self.param = param
        self.param_hint = None


def test_the_usage_hint_is_read_from_a_click_parameter() -> None:
    param = _ClickStyleParam()
    ctx = _Ctx()
    assert _usage_param(_BadParameterError(param), ctx) == "'--limit' / '-n'"
    assert param.asked_with == [ctx]


def test_the_usage_hint_is_read_from_a_typer_029_parameter() -> None:
    assert _usage_param(_BadParameterError(_TyperStyleParam()), _Ctx()) == "'--limit' / '-n'"
    # It needs no context to name itself.
    assert _usage_param(_BadParameterError(_TyperStyleParam()), None) == "'--limit' / '-n'"


def test_a_click_parameter_without_a_context_gives_no_hint() -> None:
    assert _usage_param(_BadParameterError(_ClickStyleParam()), None) is None
