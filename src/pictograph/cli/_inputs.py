"""What the ``predict`` commands accept as an image: paths, globs, URLs and stdin.

One reading for ``models predict`` and ``deployments predict``, so the same line
works against a model on this machine, a model on Pictograph and a deployment:

- a **path** to an image file;
- a **glob** (``frames/*.jpg``, ``shots/**/*.png``) - expanded HERE, sorted, so a
  quoted pattern behaves the same in every shell and on Windows;
- an **http(s) URL**, fetched on this machine;
- **``-``**, the image bytes on standard input (``curl ... | pictograph models
  predict "Shelf Detector" -``).

Everything is checked before the first prediction runs: a pattern that matches
nothing, or a path that does not exist, stops the command up front instead of half
way through a batch.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from pictograph._image_input import is_url

if TYPE_CHECKING:
    from collections.abc import Callable

STDIN = "-"
_GLOB_CHARS = frozenset("*?[")

HELP = (
    "One or more images: a file path, a glob ('frames/*.jpg'), an http(s) URL, "
    "or - for image bytes on stdin."
)


def expand_inputs(values: list[str]) -> list[str]:
    """The images to run, in order: globs expanded and sorted, the rest as typed.

    Raises:
        typer.BadParameter: Nothing was given, ``-`` appears twice, a glob matches
            no file, or a path does not exist.
    """
    if not values:
        raise typer.BadParameter("Give at least one image.", param_hint="IMAGES")
    if values.count(STDIN) > 1:
        raise typer.BadParameter(
            "- (stdin) can only be given once: there is one stdin.", param_hint="IMAGES"
        )
    out: list[str] = []
    for value in values:
        if value == STDIN or is_url(value):
            out.append(value)
        elif _GLOB_CHARS & set(value):
            matches = sorted(
                p
                # glob.glob, not Path.glob: the pattern may be absolute or start with ~.
                for p in glob.glob(str(Path(value).expanduser()), recursive=True)  # noqa: PTH207
                if Path(p).is_file()
            )
            if not matches:
                raise typer.BadParameter(f"No file matches {value!r}.", param_hint="IMAGES")
            out.extend(matches)
        else:
            if not Path(value).expanduser().is_file():
                raise typer.BadParameter(f"No such file: {value}", param_hint="IMAGES")
            out.append(value)
    return out


def read_stdin() -> bytes:
    """The image bytes piped to this command."""
    data = sys.stdin.buffer.read()
    if not data:
        raise typer.BadParameter(
            "- was given but nothing arrived on stdin. Pipe an image in: "
            "`cat photo.jpg | pictograph models predict <model> -`.",
            param_hint="IMAGES",
        )
    return data


def load(value: str, stdin: bytes | None) -> str | Path | bytes:
    """What to hand the SDK for one expanded input."""
    if value == STDIN:
        if stdin is None:  # pragma: no cover - expand_inputs + the caller guarantee it
            raise typer.BadParameter("- was given but stdin was not read.", param_hint="IMAGES")
        return stdin
    return value if is_url(value) else Path(value).expanduser()


def emit_jsonl(inputs: list[str], run: Callable[[str], Any]) -> None:
    """Run ``run`` over every input and print one JSON object per line.

    ``{"input": ..., "result": ...}`` for an image that worked and
    ``{"input": ..., "error": ..., "error_type": ...}`` for one that did not, in
    the order given, each line written as soon as its image is done. A batch never
    stops at the first bad image; the exit code is 1 when any failed.
    """
    failed = 0
    for value in inputs:
        line: dict[str, Any]
        try:
            line = {"input": value, "result": run(value)}
        except Exception as exc:
            failed += 1
            line = {"input": value, "error": str(exc), "error_type": type(exc).__name__}
        typer.echo(json.dumps(line, default=str))
    if failed:
        raise typer.Exit(code=1)


__all__ = ["HELP", "STDIN", "emit_jsonl", "expand_inputs", "load", "read_stdin"]
