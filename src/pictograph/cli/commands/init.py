"""``pictograph init`` - drops an AGENTS.md template into the cwd."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pictograph.cli._errors import CliError, ExitCode
from pictograph.cli._format import print_json

_AGENTS_MD_TEMPLATE = """\
# AGENTS.md - Pictograph in this repo

This project uses [Pictograph](https://pictograph.io) to label images, train
small vision models and deploy them. Drive it with the `pictograph` CLI.

## Setup

- `pip install 'pictograph[cli]'` - the CLI is the `[cli]` extra.
- Authenticate with `pictograph login`, or set `PICTOGRAPH_API_KEY` (from
  app.pictograph.io > Settings > API Keys).
- Better, run as a key of your own: with the user's OK,
  `pictograph api-keys create agent --role member --save-profile agent` mints one
  (`--role viewer` if the work only reads) and keeps it in the CLI profile `agent`;
  `pictograph --profile agent datasets list` then runs as it.
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
pictograph auto-annotate dataset shelf --classes "cereal box:bbox"
pictograph images list shelf --min-confidence-lt 0.5
pictograph exports create shelf --name v1 --format coco
pictograph train quote shelf v1 --pipeline rfdetr_detection --gpu auto
pictograph train start shelf v1 --pipeline rfdetr_detection --gpu auto --name cereal-detector
pictograph deployments create cereal-detector --wait
```

- Between the third and fourth commands, the user reviews the drafts in the app.
- Training runs on an EXPORT, never on a dataset: `train start` takes the dataset
  and the export's name.
- `train quote` runs the checks `train start` would and prints the GPU, each
  class's training data, the estimated cost and `can_start`, without starting
  anything. Show the user the estimate before a paid run.
- `pictograph images review shelf aisle-01.jpg` records a review decision. Run it
  only when the user tells you to.
- For an object SAM3 cannot find by name (a part, a defect, a product), have the user
  draw one polygon around an example in the app, then
  `pictograph auto-annotate similar shelf aisle-01.jpg --class "cereal box" --quote`
  prices labelling every object like it in that folder (`--scope dataset`: the whole
  dataset), and the same command without `--quote` runs it. Those are drafts too.
- `deployments create` prints the one-time `pk_deploy_` token; keep it. A deployment
  runs on a GPU and provisions before it is active (a model's first deployment on a GPU
  builds its TensorRT engine first, a few minutes): `--wait` returns once it is.
- Before you deploy, score the model on an export it did not train on:
  `pictograph models evaluate cereal-detector --dataset shelf --export holdout --wait`
  prints precision, recall, F1 and mAP per class. It runs on Pictograph and spends
  compute credit.

## Start from a public dataset

Datasets and models other organizations published can be a starting point:

```bash
pictograph explore datasets --search cereal
pictograph explore dataset pictograph-research/coco-5k
pictograph datasets import-public pictograph-research/coco-5k --name shelf-start
```

- A public dataset or model is named `org/name` - the REF column `explore` prints.
- The copy is private to this organization and keeps the source's classes, license
  and attribution. Check the license fits what the user will do with it.
- `pictograph explore models --task object_detection` lists public models;
  `pictograph models fork <ref>` copies one.

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
takes several at once, one JSON line per image. It also runs on a video's frames and on
a dataset's images (read-only; nothing is saved):

```bash
pictograph models predict cereal-detector 'frames/*.jpg' https://example.com/a.jpg --remote
pictograph models predict cereal-detector --video aisle.mp4 --sample-fps 2
pictograph models predict cereal-detector --dataset shelves --directory /val --limit 100
```

SAM3 needs no model at all: it finds what you name in any image, with one prompt
(`--text`, `--box x,y,w,h` or `--point x,y`), and saves nothing:

```bash
pictograph sam3 predict aisle-01.jpg --text "cereal box" --json
pictograph sam3 predict 'frames/*.jpg' --box 120,80,200,160 --name cereal --jsonl
```

A workflow reads a whole video on Pictograph (tracking, counting). Give it a file, a
link Pictograph downloads itself, or `-` for bytes piped in:

```bash
pictograph workflows invoke shelf-counter --file aisle.mp4
pictograph workflows invoke shelf-counter --file https://example.com/aisle.mp4
```

## Annotation format

- The class label field is **`name`** (never `class`).
- Polygons use multi-ring `paths`, not flat coordinate arrays.
- Full schema: `references/annotations.md` in the skill.

## Before you spend

Auto-annotation, training and deployments spend compute credit.
`pictograph credits balance` shows what is left,
`pictograph auto-annotate quote --dataset shelf --classes "cereal box:bbox"`
prices a labelling job before it runs;
`pictograph auto-annotate similar shelf aisle-01.jpg --class "cereal box" --quote` prices
an "Annotate similar" job; `pictograph train quote shelf v1 --pipeline rfdetr_detection`
prices a training run.

## Helpful commands

- `pictograph datasets list` - what is in the organization.
- `pictograph jobs list` - what is running (training, imports, exports, auto-annotation,
  workflow runs); `jobs wait <id>` blocks until one finishes, `jobs cancel <id>` stops it.
- `pictograph classes list shelf` - a dataset's classes; `classes add` / `update` set
  a class's type, colour, attributes or keypoint skeleton.
- `pictograph images copy shelf --to-dataset shelf-night --directory /night` - copy
  images into another dataset (create it first: `pictograph datasets create shelf-night`). The
  copies share the files and keep their annotations, tags, split and stage.
- `pictograph agents export-tools -o tools.json` - the agent tool registry as
  JSON Schema.

## When things fail

Set `PICTOGRAPH_OUTPUT=json` (or pass `--json`; every command takes it): every command
then prints JSON, and every failure is one line of JSON on stderr (`{"error": {"code", "message", "fix", "details", ...}}`). Branch
on the exit code:

- 2: no API key, or it was refused - ask the user for one.
- 3: the key's role or the plan does not allow it - tell the user.
- 4: not found - names are case-sensitive; check `pictograph datasets list`.
- 5: conflicts with the current state - read `code` (e.g. `dataset_cold`).
- 6: invalid input - fix the arguments; `--help` lists them.
- 7: out of compute credit - check `pictograph credits balance`, ask the user.
- 8: rate limited - wait `retry_after` seconds.
- 9: server or network - retry later.
- 10: a `--wait` ran out of time; the job is still running - wait again, never
  start it twice.
- 1: anything else (a job that ended failed, a declined confirmation).
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
    json_output: Annotated[
        bool, typer.Option("--json", help="Print {written} as JSON instead of a sentence.")
    ] = False,
) -> None:
    """Drop an AGENTS.md template into ``output``."""
    target = output.expanduser().resolve()
    if target.is_file() and not force:
        raise CliError(
            f"{target} already exists.",
            exit_code=ExitCode.CONFLICT,
            code="file_exists",
            fix="Pass --force to overwrite it.",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_AGENTS_MD_TEMPLATE, encoding="utf-8")
    if json_output:
        print_json({"written": str(target)})
        return
    typer.echo(f"Wrote AGENTS.md template → {target}")
