"""A flag's choice, checked in the command - never a ``typing.Literal`` parameter.

typer builds every command when the CLI starts, and a ``typing.Literal`` parameter needs a newer typer than the SDK
accepts (``typer>=0.12``): 0.12 - 0.18 raise "Type not yet supported" and the WHOLE CLI fails to start, every command with
it (2026-11-25: ``predict --anomaly-map``, caught by the landing's docs gate on typer 0.17.4). So a choice is a plain
``str`` option checked here, before any request, so a typo costs none (``tests/unit/test_cli_choices.py`` holds every
command's parameters to it).
"""

from __future__ import annotations

from typing import TypeVar

import typer

_Word = TypeVar("_Word", bound=str)


def choice(flag: str, value: str | None, allowed: tuple[_Word, ...]) -> _Word | None:
    """A flag's value, lowercased, when it is one of ``allowed`` (checked here so a typo costs no request)."""
    if value is None:
        return None
    word = value.strip().lower()
    for option in allowed:
        if option == word:
            return option
    raise typer.BadParameter(f"must be one of {', '.join(allowed)}", param_hint=flag)
