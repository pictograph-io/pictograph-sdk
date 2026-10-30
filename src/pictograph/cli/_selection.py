"""Which images a bulk ``pictograph images`` command acts on, and running it in chunks.

A selection is exactly one of:

- ``--images a,b,...`` - image UUIDs (``pictograph images list`` prints them);
- ``--from-file ids.txt`` - one UUID per line (``-`` reads stdin; blank lines and
  ``#`` comments are skipped);
- filters - ``--directory`` (exactly that folder), ``--status``, ``--split`` - which
  list the dataset's images that match all of them;
- ``--all`` - every image of the dataset.

Nothing is selected by default: a bulk command with no selection is refused, so a
typo can never turn into "every image". The batch API takes at most 10,000 images
per request, so a bigger selection runs as several requests; their results are added
up into one report.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import typer

from pictograph.resources._resolve import looks_like_id

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pictograph import Client
    from pictograph.models.batch import BatchResult
    from pictograph.models.image import ImageSplit, ImageStatus

#: The batch API's per-request ceiling.
CHUNK = 10_000

STATUSES = ("new", "annotate", "review", "complete")
SPLITS = ("train", "val", "test")

ImagesOption = Annotated[
    str | None,
    typer.Option(
        "--images", help="Comma-separated image UUIDs (pictograph images list prints them)."
    ),
]
FromFileOption = Annotated[
    str | None,
    typer.Option("--from-file", help="A file of image UUIDs, one per line ('-' reads stdin)."),
]
DirectoryOption = Annotated[
    str | None,
    typer.Option(
        "--directory",
        "-f",
        help="Select the images in this folder ('/val'; '/' is the root). Exactly this folder.",
    ),
]
StatusFilterOption = Annotated[
    str | None,
    typer.Option(
        "--status", "-s", help="Select images at this stage: new, annotate, review, complete."
    ),
]
SplitOption = Annotated[
    str | None, typer.Option("--split", help="Select images in this split: train, val, test.")
]
AllOption = Annotated[bool, typer.Option("--all", help="Select every image of the dataset.")]
DryRunOption = Annotated[
    bool,
    typer.Option(
        "--dry-run", help="Print how many images are selected (and the first ids); change nothing."
    ),
]


@dataclass
class Selection:
    """The images a command acts on, and how they were chosen (for the report)."""

    image_ids: list[str]
    how: str


@dataclass
class Totals:
    """The batch results of every chunk, added up."""

    processed: int = 0
    renamed: int = 0
    failed: list[dict[str, str]] = field(default_factory=list)
    affected_directories: list[str] = field(default_factory=list)
    operation: str | None = None


def _ids_from_text(text: str, source: str) -> list[str]:
    ids: list[str] = []
    for raw in text.replace(",", "\n").splitlines():
        value = raw.split("#", 1)[0].strip()
        if not value:
            continue
        if not looks_like_id(value):
            raise typer.BadParameter(
                f"{value!r} is not an image UUID - pictograph images list prints them.",
                param_hint=source,
            )
        ids.append(value)
    return list(dict.fromkeys(ids))  # keep order, drop repeats


def select(
    client: Client,
    dataset: str,
    *,
    images: str | None,
    from_file: str | None,
    directory: str | None,
    status: str | None,
    split: str | None,
    select_all: bool,
    archived: bool = False,
) -> Selection:
    """Resolve the one selection the flags make (see the module docstring).

    ``archived`` selects archived images instead of live ones (``images restore``).
    """
    by_ids = images is not None or from_file is not None
    filters = [f for f in (directory, status, split) if f is not None]
    kinds = sum([by_ids, bool(filters), select_all])
    if kinds == 0:
        raise typer.BadParameter(
            "select the images: --images, --from-file, a filter (--directory, "
            "--status, --split) or --all.",
            param_hint="--images",
        )
    if kinds > 1 or (images is not None and from_file is not None):
        raise typer.BadParameter(
            "give one selection: image ids (--images or --from-file), filters, or --all.",
            param_hint="--images",
        )
    if status is not None and status not in STATUSES:
        raise typer.BadParameter(f"use one of {', '.join(STATUSES)}.", param_hint="--status")
    if split is not None and split not in SPLITS:
        raise typer.BadParameter(f"use one of {', '.join(SPLITS)}.", param_hint="--split")
    if images is not None:
        return Selection(_ids_from_text(images, "--images"), "--images")
    if from_file is not None:
        if from_file == "-":
            text = sys.stdin.read()
        else:
            try:
                text = Path(from_file).read_text(encoding="utf-8")
            except FileNotFoundError:
                raise typer.BadParameter(
                    f"No such file: {from_file}", param_hint="--from-file"
                ) from None
        return Selection(_ids_from_text(text, "--from-file"), "--from-file")
    listed = client.images.iter(
        dataset,
        directory=directory,
        status=cast("ImageStatus | None", status),
        split=cast("ImageSplit | None", split),
        include_archived=archived,
        page_size=1000,
    )
    ids = [img.id for img in listed if img.is_archived == archived]
    parts = [
        f"{k}={v}" for k, v in (("directory", directory), ("status", status), ("split", split)) if v
    ]
    return Selection(ids, ", ".join(parts) or "--all")


def run_in_chunks(selection: Sequence[str], call: Callable[[list[str]], BatchResult]) -> Totals:
    """Run ``call`` over ``selection`` 10,000 ids at a time and add the results up."""
    totals = Totals()
    for start in range(0, len(selection), CHUNK):
        result = call(list(selection[start : start + CHUNK]))
        totals.processed += result.processed
        totals.renamed += result.renamed
        totals.failed.extend({"image_id": f.id, "reason": f.reason} for f in result.failed)
        for directory in result.affected_directories:
            if directory not in totals.affected_directories:
                totals.affected_directories.append(directory)
        totals.operation = getattr(result, "operation", None) or totals.operation
    return totals


def dry_run_report(
    operation: str, dataset: str, selection: Selection, **extra: Any
) -> dict[str, Any]:
    return {
        "operation": operation,
        "dataset": dataset,
        **extra,
        "selected_by": selection.how,
        "selected": len(selection.image_ids),
        "image_ids": selection.image_ids[:20],
        "dry_run": True,
    }


def report(
    operation: str, dataset: str, selection: Selection, totals: Totals, **extra: Any
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "operation": operation,
        "dataset": dataset,
        "selected_by": selection.how,
        "selected": len(selection.image_ids),
        "processed": totals.processed,
        "failed": totals.failed,
        "affected_directories": totals.affected_directories,
        **extra,
    }
    return out


Kind = Literal["move", "copy", "set-status", "archive", "restore"]

__all__ = [
    "CHUNK",
    "SPLITS",
    "STATUSES",
    "AllOption",
    "DirectoryOption",
    "DryRunOption",
    "FromFileOption",
    "ImagesOption",
    "Kind",
    "Selection",
    "SplitOption",
    "StatusFilterOption",
    "Totals",
    "dry_run_report",
    "report",
    "run_in_chunks",
    "select",
]
