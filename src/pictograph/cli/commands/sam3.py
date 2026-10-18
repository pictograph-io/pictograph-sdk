"""``pictograph sam3 predict`` - SAM3 on any image. Nothing is saved.

The image is SENT: a file, a glob, an http(s) URL, ``-`` (bytes on stdin), a video's
frames, or a dataset's images (read only). One prompt per run - a phrase, a box, or
points - and every image gets the same one::

    pictograph sam3 predict photo.jpg --text "red car"
    pictograph sam3 predict https://example.com/street.jpg --box 120,80,200,160 --name car
    curl -s https://example.com/a.jpg | pictograph sam3 predict - --point 320,240 --point 300,400,neg
    pictograph sam3 predict 'shots/*.jpg' --text forklift --output bbox --jsonl
    pictograph sam3 predict --video clip.mp4 --sample-fps 1 --text person
    pictograph sam3 predict --dataset my-dataset --directory /val --text car --limit 20

For one image already in a dataset, ``pictograph auto-annotate text|box|point`` runs
on it in place (nothing to upload); ``pictograph auto-annotate batch`` saves drafts.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer

from pictograph._image_input import read_image
from pictograph.cli import _inputs
from pictograph.cli._client import get_client
from pictograph.cli._format import print_json, print_table
from pictograph.resources.sam3 import prompt_form

app = typer.Typer(no_args_is_help=True)

_OUTPUTS = ("polygon", "bbox")


def _numbers(value: str, count: int, flag: str, shape: str) -> list[float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != count:
        raise typer.BadParameter(f"{flag} is {shape}, got {value!r}.", param_hint=flag)
    try:
        return [float(p) for p in parts]
    except ValueError:
        raise typer.BadParameter(f"{flag} is {shape}, got {value!r}.", param_hint=flag) from None


def parse_box(value: str, flag: str = "--box") -> tuple[float, float, float, float]:
    """``x,y,w,h`` -> ``(x, y, w, h)``."""
    x, y, w, h = _numbers(value, 4, flag, "x,y,w,h (absolute pixels)")
    return x, y, w, h


def parse_point(value: str) -> tuple[tuple[float, float], bool]:
    """``x,y`` (on the object) or ``x,y,neg`` (excluded) -> ``((x, y), positive)``."""
    parts = [p.strip() for p in value.split(",")]
    positive = True
    if len(parts) == 3:
        label = parts.pop().lower()
        if label not in ("pos", "neg"):
            raise typer.BadParameter(
                f"--point {value!r}: the third part is 'neg' (exclude) or 'pos'.",
                param_hint="--point",
            )
        positive = label == "pos"
    x, y = _numbers(",".join(parts), 2, "--point", "x,y or x,y,neg (absolute pixels)")
    return (x, y), positive


@app.command(
    "predict",
    help=(
        "Run SAM3 on images you send - files, globs, URLs, - (stdin), a video's frames "
        "(--video) or a dataset's images (--dataset) - with ONE prompt: --text a phrase, "
        "--box x,y,w,h, or --point x,y (repeatable). Prints the objects it finds; nothing "
        "is saved. Charged the SAM3 session fee once per image (repeat prompts on the same "
        "image are free); `pictograph tools describe sam3` shows the price and every way "
        "to run it."
    ),
)
def predict(
    images: Annotated[list[str] | None, typer.Argument(help=_inputs.HELP)] = None,
    text: Annotated[
        str | None,
        typer.Option("--text", help="A short noun phrase ('red car'). Every match comes back."),
    ] = None,
    box: Annotated[
        str | None,
        typer.Option(
            "--box",
            help="x,y,w,h in absolute pixels around one object: it and every object like it come back.",
        ),
    ] = None,
    point: Annotated[
        list[str] | None,
        typer.Option(
            "--point",
            help="x,y on the object; repeat to add points, x,y,neg to exclude a region. One mask comes back.",
        ),
    ] = None,
    negative_box: Annotated[
        list[str] | None,
        typer.Option("--negative-box", help="With --box: x,y,w,h to exclude (repeatable)."),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(
            "--name", help="Class name for the results (default: the phrase, or 'object')."
        ),
    ] = None,
    output: Annotated[
        str, typer.Option("--output", help="polygon (mask outlines, default) or bbox.")
    ] = "polygon",
    confidence: Annotated[
        float | None,
        typer.Option(
            "--confidence",
            "-c",
            min=0.0,
            max=1.0,
            help="Minimum score (0-1). Default by prompt: 0.3 text, 0.5 box, 0.75 point.",
        ),
    ] = None,
    max_detections: Annotated[
        int,
        typer.Option(
            "--max-detections", min=1, max=100, help="--text: the most matches to return."
        ),
    ] = 50,
    video: Annotated[list[str] | None, typer.Option("--video", help=_inputs.VIDEO_HELP)] = None,
    sample_fps: Annotated[
        float | None,
        typer.Option("--sample-fps", min=_inputs.MIN_SAMPLE_FPS, help=_inputs.SAMPLE_FPS_HELP),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option(
            "--dataset",
            help=(
                "Run on the images of one of your datasets, read-only: results are "
                "printed, nothing is saved (to save drafts: pictograph auto-annotate "
                "batch). A dataset in cold storage must be restored first."
            ),
        ),
    ] = None,
    directory: Annotated[
        str | None, typer.Option("--directory", help=_inputs.DIRECTORY_HELP)
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", min=1, help=_inputs.LIMIT_HELP)] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    jsonl: Annotated[
        bool,
        typer.Option(
            "--jsonl",
            help="One JSON object per image or frame, one per line: {input, result} or "
            "{input, error}. Always used for several images, a video or a dataset.",
        ),
    ] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if output not in _OUTPUTS:
        raise typer.BadParameter("--output is polygon or bbox.", param_hint="--output")
    given = [
        flag for flag, value in (("--text", text), ("--box", box), ("--point", point)) if value
    ]
    if len(given) != 1:
        raise typer.BadParameter(
            "Give exactly one prompt: --text a phrase, --box x,y,w,h, or --point x,y "
            f"(repeatable). Got {' and '.join(given) if given else 'none'}.",
            param_hint="--text / --box / --point",
        )
    if negative_box and not box:
        raise typer.BadParameter("--negative-box goes with --box.", param_hint="--negative-box")
    positives: list[tuple[float, float]] = []
    negatives: list[tuple[float, float]] = []
    for value in point or ():
        xy, positive = parse_point(value)
        (positives if positive else negatives).append(xy)
    if point and not positives:
        raise typer.BadParameter(
            "A point prompt needs at least one point ON the object (x,y); x,y,neg only excludes.",
            param_hint="--point",
        )
    try:
        form = prompt_form(
            text=text,
            box=parse_box(box) if box is not None else None,
            points=positives or None,
            negative_points=negatives or None,
            negative_boxes=[parse_box(b, "--negative-box") for b in negative_box or ()] or None,
            name=name,
            output=output,  # type: ignore[arg-type]  # checked against _OUTPUTS above
            confidence=confidence,
            max_detections=max_detections,
        )
    except ValueError as exc:  # a blank --text: the shape checks above caught the rest
        raise typer.BadParameter(str(exc), param_hint="--text / --box / --point") from None

    source = _inputs.parse_source(
        images,
        video=video,
        sample_fps=sample_fps,
        dataset=dataset,
        directory=directory,
        limit=limit,
    )
    client = get_client(api_key)
    items = _inputs.open_items(source, client, frames="jpeg")

    def run(image: Any) -> Any:
        data, filename = read_image(image)
        return client.sam3._predict_form(data, filename, form)

    if jsonl or not source.single_image:
        _inputs.emit(items, lambda image: run(image).model_dump(mode="json", exclude_none=True))
        return

    (item,) = list(items)
    result = run(item.load())
    if json_output:
        print_json(result.model_dump(mode="json", exclude_none=True))
        return
    rows = [
        {
            "name": a.name,
            "type": a.type,
            "confidence": round(float(getattr(a, "confidence", 1.0)), 4),
        }
        for a in result.annotations
    ]
    title = (
        f"SAM3 {result.prompt} - {len(rows)} object(s) on the {result.width}x{result.height} image"
        if rows
        else f"SAM3 {result.prompt} - {result.status.replace('_', ' ')} on the "
        f"{result.width}x{result.height} image"
    )
    print_table(rows, title=title)
