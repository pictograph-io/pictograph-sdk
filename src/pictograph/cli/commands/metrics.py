"""``pictograph metrics evaluate`` - offline model evaluation.

The CLI face of :mod:`pictograph.metrics`: read predicted and ground-truth
annotations from Pictograph-JSON files and compute detection metrics entirely on
your machine - no server round-trip, no credits (handy in CI: fail a build when
mAP drops). Each input file is a JSON object mapping an image key to its list of
annotations::

    {"img1.jpg": [{"id": "a", "name": "car", "type": "bbox",
                   "bounding_box": {"x": 0, "y": 0, "w": 10, "h": 10},
                   "confidence": 0.8}], ...}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from pydantic import TypeAdapter, ValidationError

from pictograph.cli._format import print_json, print_table
from pictograph.metrics import evaluate_detections
from pictograph.models.annotation import Annotation

if TYPE_CHECKING:
    from collections.abc import Sequence

app = typer.Typer(no_args_is_help=True)


@app.callback()
def _metrics() -> None:
    """Offline model evaluation.

    This no-op callback keeps ``metrics`` a command GROUP. Typer collapses a
    single-command app into the group itself, which would silently move the
    documented ``pictograph metrics evaluate`` to ``pictograph metrics`` now
    that the ``rank`` subcommand is gone. Keep it when adding/removing commands.
    """


_ann_list_adapter: TypeAdapter[list[Annotation]] = TypeAdapter(list[Annotation])


def _load(path: Path) -> dict[str, Sequence[Annotation]]:
    """Parse a ``{image_key: [annotation, ...]}`` Pictograph-JSON file."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"could not read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise typer.BadParameter(f"{path} must be a JSON object mapping image key -> [annotations]")
    try:
        return {str(k): _ann_list_adapter.validate_python(v) for k, v in raw.items()}
    except ValidationError as exc:
        raise typer.BadParameter(f"{path} has an invalid annotation: {exc}") from exc


@app.command("evaluate", help="Detection P/R/F1 + mAP of predictions vs ground truth (by IoU).")
def evaluate(
    predictions: Annotated[
        Path, typer.Argument(help="Pictograph-JSON {image: [annotations]} of predictions.")
    ],
    ground_truth: Annotated[
        Path, typer.Argument(help="Pictograph-JSON {image: [annotations]} of ground truth.")
    ],
    iou: Annotated[float, typer.Option("--iou", help="IoU match threshold (0-1].")] = 0.5,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        result = evaluate_detections(_load(predictions), _load(ground_truth), iou_threshold=iou)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        print_json(
            {
                "iou_threshold": result.iou_threshold,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "macro_f1": result.macro_f1,
                "mean_average_precision": result.mean_average_precision,
                "per_class": {
                    name: {
                        "precision": m.precision,
                        "recall": m.recall,
                        "f1": m.f1,
                        "average_precision": m.average_precision,
                        "support": m.support,
                    }
                    for name, m in result.per_class.items()
                },
            }
        )
        return
    rows = [
        {
            "class": name,
            "P": f"{m.precision:.3f}",
            "R": f"{m.recall:.3f}",
            "F1": f"{m.f1:.3f}",
            "AP": f"{m.average_precision:.3f}",
            "support": m.support,
        }
        for name, m in sorted(result.per_class.items())
    ]
    print_table(
        rows,
        columns=["class", "P", "R", "F1", "AP", "support"],
        title=f"Detection metrics @ IoU {iou}",
    )
    typer.echo(
        f"Overall  P {result.precision:.3f}  R {result.recall:.3f}  F1 {result.f1:.3f}  "
        f"macro-F1 {result.macro_f1:.3f}  mAP {result.mean_average_precision:.3f}"
    )
