"""``pictograph auto-annotate {point,box,text,batch}`` - SAM3 / trained-model inference.

SAM3 prompts over images already uploaded to a dataset::

    pictograph auto-annotate point road-signs frame_001.jpg --x 320 --y 240 --name stop_sign
    pictograph auto-annotate box   road-signs frame_001.jpg --box 100,100,200,200 --name stop_sign
    pictograph auto-annotate text  road-signs frame_001.jpg --prompt "stop sign"
    pictograph auto-annotate batch road-signs --images a.jpg,b.jpg --classes "person:bbox,car:polygon"
"""

from __future__ import annotations

from typing import Annotated

import typer

from pictograph.cli._client import get_client
from pictograph.cli._format import print_json

app = typer.Typer(no_args_is_help=True)


@app.command("point", help="SAM3 point prompt → one polygon annotation.")
def point(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    image: Annotated[str, typer.Argument(help="Image filename within the dataset.")],
    x: Annotated[int, typer.Option("--x", help="Anchor X in absolute pixels.")],
    y: Annotated[int, typer.Option("--y", help="Anchor Y in absolute pixels.")],
    name: Annotated[str, typer.Option("--name", help="Class label for the result.")] = "object",
    score_threshold: Annotated[float, typer.Option("--score-threshold")] = 0.75,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    result = client.auto_annotate.point(
        dataset, image, x=x, y=y, name=name, score_threshold=score_threshold
    )
    print_json(result.model_dump(mode="json", exclude_none=True))


@app.command("box", help="SAM3 box prompt → bbox + optional polygon.")
def box(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    image: Annotated[str, typer.Argument(help="Image filename within the dataset.")],
    box: Annotated[str, typer.Option("--box", help="Box as 'x,y,w,h' in absolute pixels.")],
    name: Annotated[str, typer.Option("--name", help="Class label.")],
    confidence: Annotated[float, typer.Option("--confidence")] = 0.5,
    no_polygon: Annotated[
        bool, typer.Option("--no-polygon", help="Return only the refined bbox.")
    ] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    parts = [p.strip() for p in box.split(",")]
    if len(parts) != 4:
        raise typer.BadParameter("--box must be 'x,y,w,h'", param_hint="--box")
    try:
        x, y, w, h = (float(p) for p in parts)
    except ValueError as exc:
        raise typer.BadParameter("--box values must be numbers", param_hint="--box") from exc
    client = get_client(api_key)
    result = client.auto_annotate.box(
        dataset,
        image,
        box={"x": x, "y": y, "w": w, "h": h},
        name=name,
        confidence_threshold=confidence,
        return_polygon=not no_polygon,
    )
    print_json(result.model_dump(mode="json", exclude_none=True))


@app.command("text", help="SAM3 text prompt → detected annotations (phrase grounding).")
def text(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    image: Annotated[str, typer.Argument(help="Image filename within the dataset.")],
    prompt: Annotated[
        str, typer.Option("--prompt", help="Natural-language phrase, e.g. 'red car'.")
    ],
    output_type: Annotated[
        str, typer.Option("--output-type", help="polygon (default) or bbox.")
    ] = "polygon",
    confidence: Annotated[float, typer.Option("--confidence")] = 0.3,
    max_detections: Annotated[int, typer.Option("--max-detections")] = 50,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    result = client.auto_annotate.text(
        dataset,
        image,
        text_prompt=prompt,
        output_type=output_type,
        confidence_threshold=confidence,
        max_detections=max_detections,
    )
    print_json(result.model_dump(mode="json", exclude_none=True))


@app.command("batch", help="Batch auto-annotate many images (SAM3 or a trained model).")
def batch(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    images: Annotated[
        str, typer.Option("--images", help="Comma-separated image filenames (1-500).")
    ],
    classes: Annotated[
        str | None,
        typer.Option(
            "--classes",
            help="Comma-separated 'name:output_type' (e.g. 'person:bbox,car:polygon'). Required for SAM3.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Trained model by name (a UUID works); omit for SAM3."),
    ] = None,
    confidence: Annotated[float, typer.Option("--confidence")] = 0.5,
    sahi: Annotated[
        bool,
        typer.Option(
            "--sahi",
            help="SAHI sliced inference (SAM3 only): tile each image for small-object recall.",
        ),
    ] = False,
    sahi_slice_size: Annotated[
        int,
        typer.Option(
            "--sahi-slice-size", min=256, max=1024, help="SAHI tile edge in pixels (256-1024)."
        ),
    ] = 640,
    no_wait: Annotated[
        bool, typer.Option("--no-wait", help="Return the kick-off snapshot + exit.")
    ] = False,
    timeout: Annotated[float, typer.Option("--timeout")] = 1800.0,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    image_list = [s.strip() for s in images.split(",") if s.strip()]
    if not image_list:
        raise typer.BadParameter("--images must list at least one filename", param_hint="--images")
    class_specs: list[dict[str, str]] = []
    if classes:
        for spec in classes.split(","):
            spec = spec.strip()
            if not spec:
                continue
            name, _, otype = spec.partition(":")
            class_specs.append({"name": name.strip(), "output_type": (otype.strip() or "polygon")})
    client = get_client(api_key)
    job = client.auto_annotate.batch(
        dataset,
        image_list,
        class_specs,
        confidence_threshold=confidence,
        model=model,
        sahi=sahi,
        sahi_slice_size=sahi_slice_size,
        wait=not no_wait,
        timeout=timeout,
    )
    print_json(job.model_dump(mode="json", exclude_none=True))


@app.command("quote", help="Price a batch auto-annotate job WITHOUT running it.")
def quote(
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", help="Dataset name. Required to price existing images."),
    ] = None,
    images: Annotated[
        str | None,
        typer.Option("--images", help="Comma-separated image filenames to price."),
    ] = None,
    frames: Annotated[
        int,
        typer.Option(
            "--frames",
            min=0,
            help=(
                "Price N images that DON'T EXIST YET - e.g. the frames a video is about "
                "to be cut into (floor(duration x fps)). A video is one file but hundreds "
                "of frames, and the frames are what you pay for."
            ),
        ),
    ] = 0,
    width: Annotated[
        int | None,
        typer.Option("--width", help="Pixel width of the projected images (SAHI pricing)."),
    ] = None,
    height: Annotated[int | None, typer.Option("--height")] = None,
    classes: Annotated[
        str | None,
        typer.Option("--classes", help="Comma-separated 'name:output_type'."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Trained model by name (a UUID works); omit for SAM3."),
    ] = None,
    sahi: Annotated[bool, typer.Option("--sahi")] = False,
    sahi_slice_size: Annotated[int, typer.Option("--sahi-slice-size", min=256, max=1024)] = 640,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    image_list = [s.strip() for s in (images or "").split(",") if s.strip()]
    class_specs: list[dict[str, str]] = []
    if classes:
        for spec in classes.split(","):
            spec = spec.strip()
            if not spec:
                continue
            name, _, otype = spec.partition(":")
            class_specs.append({"name": name.strip(), "output_type": (otype.strip() or "polygon")})

    projected: list[dict[str, int]] = []
    if frames > 0:
        group: dict[str, int] = {"count": frames}
        if width:
            group["width"] = width
        if height:
            group["height"] = height
        projected.append(group)

    if not image_list and not projected:
        raise typer.BadParameter(
            "give --images (existing images) and/or --frames (images that don't exist yet)",
            param_hint="--images/--frames",
        )

    client = get_client(api_key)
    result = client.auto_annotate.quote(
        dataset_name=dataset,
        image_filenames=image_list,
        projected=projected,
        classes=class_specs,
        model=model,
        sahi=sahi,
        sahi_slice_size=sahi_slice_size,
    )
    print_json(result.model_dump(mode="json", exclude_none=True))


@app.command("get", help="Fetch a batch auto-annotation job's status by UUID.")
def get_batch(
    job_id: Annotated[str, typer.Argument(help="Batch job UUID.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    job = client.auto_annotate.get_batch(job_id)
    print_json(job.model_dump(mode="json", exclude_none=True))


@app.command("cancel-batch", help="Request cancellation of a running batch auto-annotation job.")
def cancel_batch(
    job_id: Annotated[str, typer.Argument(help="Batch job UUID.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(f"Cancel batch job {job_id!r}?"):
        raise typer.Abort()
    client = get_client(api_key)
    job = client.auto_annotate.cancel_batch(job_id)
    print_json(job.model_dump(mode="json", exclude_none=True))
