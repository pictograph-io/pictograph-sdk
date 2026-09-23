"""``pictograph init`` - drops an AGENTS.md template into the cwd."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

_AGENTS_MD_TEMPLATE = """\
# AGENTS.md - agent on-ramp for this project

This project uses [Pictograph](https://pictograph.io) for computer-vision
annotation and model training. Agents working in this repo should:

## Setup
- `pip install pictograph` (CLI: `pip install 'pictograph[cli]'`)
- Set `PICTOGRAPH_API_KEY` from app.pictograph.io → Settings → API Keys.

## Common workflows
- **Upload a directory of images**: `pictograph datasets create <name>` then
  `python -m pictograph.skills.pictograph-cv.scripts.upload_and_annotate
  --directory <path> --dataset <name>`.
- **Auto-annotate via SAM3**: see `pictograph/skills/pictograph-cv/SKILL.md`.
- **Train a model**: `pictograph train start <dataset> --pipeline yolox`.

## Annotation format
- Class label field is **`name`** (not `class`).
- Polygons use multi-ring `paths`, not flat coordinate arrays.
- Full schema in the bundled Skill at `references/pictograph-json-schema.md`.

## Helpful commands
- `pictograph datasets list` - see what's in your org.
- `pictograph credits balance` - check spend before paid ops.
- `pictograph agents install-skill` - install the Pictograph Skill into
  `~/.claude/skills/` for Claude Code.
- `pictograph agents export-tools -o tools.json` - get the agent tool
  registry as JSON Schema (for Vercel AI SDK / LangChain / etc.).

## When things fail
- 401 → `PICTOGRAPH_API_KEY` missing or revoked.
- 402 → out of credits. Check `pictograph credits balance` and the
  organization's plan tier.
- 404 → dataset name is case-sensitive; verify with `pictograph datasets list`.
- Rate-limit (429) → SDK auto-retries with backoff; see `pictograph credits history`.
"""


def command(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Where to write AGENTS.md. Defaults to ./AGENTS.md.",
        ),
    ] = Path("./AGENTS.md"),
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite an existing AGENTS.md.",
        ),
    ] = False,
) -> None:
    """Drop an AGENTS.md template into ``output``."""
    target = output.expanduser().resolve()
    if target.is_file() and not force:
        typer.echo(
            f"{target} already exists. Pass --force to overwrite.",
            err=True,
        )
        raise typer.Exit(1)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_AGENTS_MD_TEMPLATE, encoding="utf-8")
    typer.echo(f"Wrote AGENTS.md template → {target}")
