"""Every bracketed word a command's help carries reaches the rendered ``--help``.

Typer renders ``--help`` through Rich, and Rich reads ``[word]`` as a style tag and
drops it. ``models predict --help`` said ``pip install 'pictograph'`` - the wrong
install - because ``[inference]`` vanished, and ``metrics evaluate`` described its
files as ``{image: }``. A help string escapes the bracket as ``\\[``.

This renders the real help of EVERY command, so a new help string with a bare
``[word]`` fails here instead of quietly printing something else.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from pictograph.cli._app import app

# A bracketed word as written in a help string, escaped or not.
_TOKEN = re.compile(r"\\?\[([A-Za-z_][A-Za-z0-9_,\- ]*)\]")


def _leaf_commands() -> list[tuple[list[str], Any]]:
    out: list[tuple[list[str], Any]] = []

    def walk(command: Any, path: list[str]) -> None:
        subcommands = getattr(command, "commands", None)
        if subcommands:
            for name, sub in subcommands.items():
                walk(sub, [*path, name])
        else:
            out.append((path, command))

    walk(get_command(app), [])
    return out


def _tokens(command: Any) -> dict[str, int]:
    """Each bracketed word in the command's help sources, with how often it occurs.

    Counted, not just collected: the same word in two help strings (the command's
    own and an option's) must survive in BOTH - one surviving copy must not hide
    the other one's loss.
    """
    texts = [command.help or ""] + [getattr(p, "help", None) or "" for p in command.params]
    counts: dict[str, int] = {}
    for text in texts:
        for token in _TOKEN.findall(text):
            counts[token] = counts.get(token, 0) + 1
    return counts


CASES = [
    (path, token, count)
    for path, command in _leaf_commands()
    for token, count in sorted(_tokens(command).items())
]


def test_the_check_sees_every_command_and_the_known_cases() -> None:
    commands = _leaf_commands()
    assert len(commands) > 100, "the walk reached the whole command tree"
    seen = {(" ".join(path), token): count for path, token, count in CASES}
    # The command's own help AND the --video option's.
    assert seen[("models predict", "inference")] == 2
    assert seen[("deployments predict", "inference")] == 1
    assert seen[("metrics evaluate", "annotations")] == 2


@pytest.mark.parametrize(
    ("path", "token", "count"), CASES, ids=[f"{' '.join(p)}:{t}" for p, t, _c in CASES]
)
def test_a_bracketed_word_survives_the_rendered_help(
    path: list[str], token: str, count: int
) -> None:
    result = CliRunner().invoke(app, [*path, "--help"])
    assert result.exit_code == 0, result.output
    rendered = result.output.count(f"[{token}]")
    assert rendered >= count, (
        f"`pictograph {' '.join(path)} --help` shows [{token}] {rendered} time(s); "
        f"its help strings carry it {count} time(s) - escape it as \\[{token}]"
    )
