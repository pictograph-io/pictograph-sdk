"""``pictograph init`` - drops an AGENTS.md template into the cwd."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

_AGENTS_MD_TEMPLATE = """\
# AGENTS.md - Pictograph in this repo

This project uses [Pictograph](https://pictograph.io) to label images, train
small vision models and deploy them. Drive it with the `pictograph` CLI.

## Setup

- `pip install 'pictograph[cli]'` - the CLI is the `[cli]` extra.
- Authenticate with `pictograph login`, or set `PICTOGRAPH_API_KEY` (from
  app.pictograph.io > Settings > API Keys).
- The full guide is the pictograph-cv skill.
  `pictograph agents install-skill --output ./skills` copies it here; read
  `skills/pictograph-cv/SKILL.md`.

## Draft, review, train

- Auto-annotations (SAM3, or a model trained here) are **drafts**.
- Before you export and train, have a person review the drafts in the app.
  `pictograph images list shelf --min-confidence-lt 0.5` lists the ones the model
  was least sure of - start the review there.
- Never approve your own drafts unless the user says so.
- After the review, export the dataset and train on the export.

## The loop, command by command

```bash
pictograph images upload-directory shelf ./shelf
pictograph auto-annotate batch shelf --images aisle-01.jpg,aisle-02.jpg --classes "cereal box:bbox"
pictograph images list shelf --min-confidence-lt 0.5
pictograph exports create shelf --name v1 --format coco
pictograph train start shelf v1 --pipeline rfdetr_detection --name cereal-detector
pictograph deployments create cereal-detector
```

- Between the third and fourth commands, the user reviews the drafts in the app.
- Training runs on an EXPORT, never on a dataset: `train start` takes the dataset
  and the export's name.
- `pictograph images review shelf aisle-01.jpg` records a review decision. Run it
  only when the user tells you to.

## Call a vision tool

SAM3, each ready model, each active deployment and each workflow is a tool. Ask the
platform how to call one before you call it:

```bash
pictograph tools list
pictograph tools describe sam3 --json
pictograph tools describe "model/cereal-detector" --json
```

The card says which inputs each way of running the tool takes (file, URL, bytes,
video, dataset), whether it writes anything, what a prediction looks like, the exact
command, and what it costs.

A model or a deployment reads an image as a path, a glob, a URL or `-` (stdin), and
takes several at once, one JSON line per image:

```bash
pictograph models predict cereal-detector 'frames/*.jpg' https://example.com/a.jpg --remote
```

## Annotation format

- The class label field is **`name`** (never `class`).
- Polygons use multi-ring `paths`, not flat coordinate arrays.
- Full schema: `references/annotations.md` in the skill.

## Before you spend

Auto-annotation, training and deployments spend compute credit.
`pictograph credits balance` shows what is left, and
`pictograph auto-annotate quote --dataset shelf --classes "cereal box:bbox"`
prices a labelling job before it runs.

## Helpful commands

- `pictograph datasets list` - what is in the organization.
- `pictograph agents export-tools -o tools.json` - the agent tool registry as
  JSON Schema.

## When things fail

- 401: `PICTOGRAPH_API_KEY` is missing or revoked.
- 402: out of compute credit; check `pictograph credits balance`.
- 404: names are case-sensitive; check `pictograph datasets list`.
- 429: rate limited; short waits are retried automatically.
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
