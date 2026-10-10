"""The AGENTS.md that ``pictograph init`` writes must run as written.

It is the whole on-ramp for Codex and every other agent that reads AGENTS.md
rather than a Claude skill, so an agent copies its commands verbatim. The
2026-10 landing audit found it stale in three ways at once: ``train start
<dataset> --pipeline`` with no export (the command takes the export as its
second argument), a ``python -m`` path through the hyphenated skill package
(not importable), and a ``references/pictograph-json-schema.md`` that never
existed. Every check here resolves against the INSTALLED CLI and skill, so the
template cannot drift past either again.
"""

from __future__ import annotations

import re
import shlex
from typing import Any

import pytest
import typer.main

from pictograph.cli._app import app
from pictograph.cli.commands.init import _AGENTS_MD_TEMPLATE
from pictograph.skills import skill_path

_CONTINUATION = re.compile(r"\\\s*\n\s*")
_FENCE = re.compile(r"```bash\n(.*?)```", re.S)
_INLINE = re.compile(r"`(pictograph [^`]+)`")


def _invocations() -> list[str]:
    """Every ``pictograph ...`` command in the template: fenced and inline."""
    found: list[str] = []
    for block in _FENCE.findall(_AGENTS_MD_TEMPLATE):
        for line in _CONTINUATION.sub(" ", block).splitlines():
            line = line.split(" #", 1)[0].strip()
            if line.startswith("pictograph "):
                found.append(line)
    prose = _FENCE.sub("", _AGENTS_MD_TEMPLATE)
    found.extend(" ".join(m.split()) for m in _INLINE.findall(prose))
    return list(dict.fromkeys(found))  # each distinct command once


def _is_group(cmd: Any) -> bool:
    # Duck-typed: under typer 0.27 a TyperGroup does not subclass click.Group.
    return callable(getattr(cmd, "get_command", None)) and callable(
        getattr(cmd, "list_commands", None)
    )


def _parse(line: str) -> tuple[list[str], dict[str, Any]]:
    """Resolve the command path and PARSE its arguments with the real CLI.

    ``make_context`` runs Click's own parser - unknown options, a missing
    required argument or option, an extra positional and a bad option value all
    raise - without invoking the command, so nothing touches an organization.
    """
    argv = shlex.split(line)[1:]
    cmd: Any = typer.main.get_command(app)
    path: list[str] = []
    while argv and _is_group(cmd) and not argv[0].startswith("-"):
        sub = cmd.get_command(cmd.make_context("pictograph", [], resilient_parsing=True), argv[0])
        assert sub is not None, f"`{line}`: no command {argv[0]!r} under {path or ['pictograph']}"
        cmd, path = sub, [*path, argv.pop(0)]
    assert not _is_group(cmd), f"`{line}`: {' '.join(path)!r} is a group, not a command"
    ctx = cmd.make_context(" ".join(["pictograph", *path]), argv)
    return path, dict(ctx.params)


def test_the_template_has_commands_to_check() -> None:
    # Floor, so a change to the scan (a ```sh fence, say) cannot make the
    # per-command test below vacuous.
    assert len(_invocations()) >= 10, _invocations()


@pytest.mark.parametrize("line", _invocations())
def test_every_command_parses_against_the_real_cli(line: str) -> None:
    _parse(line)


def test_training_runs_on_the_export_the_template_creates() -> None:
    """A model trains on an EXPORT (an extract of the dataset), never the dataset itself.

    The stale template ran ``train start <dataset> --pipeline`` with no export. The loop
    it teaches is draft -> review -> export -> train, and the export is a plain extract:
    no stage filter (there is no "approved images" training concept - Noah, 2026-10-09).
    """
    parsed = {" ".join(path): params for path, params in map(_parse, _invocations())}

    export = parsed["exports create"]
    assert export["status_filter"] is None, export

    train = parsed["train start"]
    assert train["dataset"] == export["dataset_name"], (train, export)
    assert train["export"] == export["name"], (train, export)


def test_it_names_only_files_the_skill_ships() -> None:
    skill = skill_path("pictograph-cv")
    for ref in re.findall(r"references/[\w.-]+\.md", _AGENTS_MD_TEMPLATE):
        assert (skill / ref).is_file(), f"the template points at {ref}, which the skill lacks"
    assert "SKILL.md" in _AGENTS_MD_TEMPLATE
    assert (skill / "SKILL.md").is_file()


def test_it_never_runs_the_skill_as_a_module() -> None:
    # `pictograph-cv` has a hyphen: `python -m pictograph.skills.pictograph-cv...`
    # cannot import. The CLI and `pictograph agents install-skill` are the way in.
    assert "python -m" not in _AGENTS_MD_TEMPLATE


def test_it_installs_the_cli_extra() -> None:
    # The `pictograph` command ships in the [cli] extra, not the base package.
    assert "pip install 'pictograph[cli]'" in _AGENTS_MD_TEMPLATE
