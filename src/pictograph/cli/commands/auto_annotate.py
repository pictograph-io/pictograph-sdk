"""``pictograph auto-annotate {point,box,text,batch,dataset,similar}`` - SAM3 / trained-model inference.

SAM3 prompts over images already uploaded to a dataset::

    pictograph auto-annotate point road-signs frame_001.jpg --x 320 --y 240 --name stop_sign
    pictograph auto-annotate box   road-signs frame_001.jpg --box 100,100,200,200 --name stop_sign
    pictograph auto-annotate text  road-signs frame_001.jpg --prompt "stop sign"
    pictograph auto-annotate batch road-signs --images a.jpg,b.jpg --classes "person:bbox,car:polygon"
    pictograph auto-annotate dataset road-signs --classes "stop sign:bbox" --directory /val
    pictograph auto-annotate similar road-signs val/frame_001.jpg --class stop_sign --scope dataset
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer
from rich.markup import escape

from pictograph.cli._client import get_client
from pictograph.cli._errors import CliError, ExitCode, command_group, exit_code_for
from pictograph.cli._format import console, err_console, print_json

if TYPE_CHECKING:
    from pictograph.resources.auto_annotate import AnnotateReport

app = command_group(no_args_is_help=True)


def _class_specs(classes: str | None) -> list[dict[str, str]]:
    """``"person:bbox,car:polygon"`` -> the batch endpoint's class list (polygon by default)."""
    specs: list[dict[str, str]] = []
    for spec in (classes or "").split(","):
        spec = spec.strip()
        if not spec:
            continue
        name, _, otype = spec.partition(":")
        specs.append({"name": name.strip(), "output_type": (otype.strip() or "polygon")})
    return specs


@app.command("point", help="SAM3 point prompt → one polygon annotation.")
def point(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    image: Annotated[
        str,
        typer.Argument(
            help="The image: its filename (the dataset root first), its path in the dataset (val/x.jpg), or its id."
        ),
    ],
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
    image: Annotated[
        str,
        typer.Argument(
            help="The image: its filename (the dataset root first), its path in the dataset (val/x.jpg), or its id."
        ),
    ],
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
    image: Annotated[
        str,
        typer.Argument(
            help="The image: its filename (the dataset root first), its path in the dataset (val/x.jpg), or its id."
        ),
    ],
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
        str,
        typer.Option(
            "--images",
            help="Comma-separated images (up to 5,000): an id or a path in the dataset is exactly that image; a bare filename is every image of that name.",
        ),
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
    class_specs = _class_specs(classes)
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


_DATASET_HELP = (
    "Draft labels for a whole dataset, or one folder of it: SAM3 finds the --classes you "
    "name, or --model runs a trained model. Images that already have annotations are "
    "skipped unless --overwrite. Prints a report: images drafted, skipped and held back, "
    "annotations added, the batch job's id. The labels are DRAFTS: have a person review "
    "them (pictograph images list <dataset> --min-confidence-lt 0.5 lists the least "
    "certain first), then export the dataset and train on the export. Spends compute "
    "credit; price it first with pictograph auto-annotate quote."
)


@app.command("dataset", help=_DATASET_HELP)
def dataset_cmd(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    classes: Annotated[
        str | None,
        typer.Option(
            "--classes",
            help=(
                "Comma-separated 'name:output_type' (polygon, bbox or tag; polygon by "
                "default). Required for SAM3; optional with --model."
            ),
        ),
    ] = None,
    directory: Annotated[
        str | None,
        typer.Option(
            "--directory",
            help="Only the images in this folder ('/val'; '/' is the root). Exactly this folder, not its subfolders.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Trained model by name (a UUID works); omit for SAM3."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help=(
                "Also draft images that already have annotations. The new drafts are added "
                "beside them; one that duplicates an existing annotation of its class is dropped."
            ),
        ),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-n", min=1, help="Draft at most this many images."),
    ] = None,
    confidence: Annotated[
        float,
        typer.Option("--confidence", min=0.0, max=1.0, help="Minimum confidence to keep (0-1)."),
    ] = 0.5,
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
        bool,
        typer.Option(
            "--no-wait",
            help=(
                "Submit the job and exit; follow it with pictograph auto-annotate get <job>. "
                "Do not run this again on the same images until it finishes, or they are drafted twice."
            ),
        ),
    ] = False,
    timeout: Annotated[
        float, typer.Option("--timeout", min=1.0, help="Max seconds to wait for the job.")
    ] = 1800.0,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the report as JSON instead of a summary.")
    ] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    class_specs = _class_specs(classes)
    if not class_specs and model is None:
        raise typer.BadParameter(
            "name what SAM3 should find, e.g. --classes 'car:bbox,person:polygon' "
            "(or pass --model to use a trained model)",
            param_hint="--classes",
        )
    if sahi and model is not None:
        raise typer.BadParameter("--sahi is for SAM3; drop it with --model.", param_hint="--sahi")
    client = get_client(api_key)
    report = client.auto_annotate.dataset(
        dataset,
        class_specs,
        directory=directory,
        model=model,
        confidence_threshold=confidence,
        overwrite=overwrite,
        max_images=limit,
        sahi=sahi,
        sahi_slice_size=sahi_slice_size,
        wait=not no_wait,
        timeout=timeout,
    )
    if not (report.images_attempted or report.images_skipped or report.images_capped):
        # Nothing at all, not "everything is annotated": a typo in --directory reads as
        # an empty folder, and drafting nothing must not look like a finished job.
        where = f" in {directory}" if directory else ""
        raise CliError(
            f"Dataset {dataset!r} has no images{where}.",
            exit_code=ExitCode.INVALID,
            code="no_images",
            fix="Check the folder with `pictograph directories list "
            f"{_quoted(dataset)}` (a folder is exact: its subfolders are not included).",
        )
    _emit_dataset_report(report, dataset=dataset, json_output=json_output, waited=not no_wait)


def _emit_dataset_report(
    report: AnnotateReport, *, dataset: str, json_output: bool, waited: bool
) -> None:
    """Print the report, say what is left, and fail with the batch's own exit code."""
    failures = [f for f in report.failures if f.image_filename != "<batch-cap>"]
    held_back = any(f.image_filename == "<batch-cap>" for f in report.failures)
    if json_output:
        payload = report.model_dump(mode="json")
        payload["success"] = not failures and (report.images_processed > 0 or not waited)
        print_json(payload)
    else:
        if waited:
            console.print(
                f"Drafted {report.images_processed} of {report.images_attempted} image(s) in "
                f"{escape(dataset)}: {report.annotations_added} annotation(s) added.",
                soft_wrap=True,
            )
        else:
            console.print(
                f"Submitted {report.images_attempted} image(s) of {escape(dataset)} as job "
                f"{report.job_id} ({report.job_status}).",
                soft_wrap=True,
            )
        console.print(
            f"Skipped {report.images_skipped} image(s) that already have annotations; "
            f"held back {report.images_capped}.",
            soft_wrap=True,
        )
    if held_back:
        err_console.print(
            f"[dim]{report.images_capped} image(s) were held back by the 5,000-image job "
            "limit: run the same command again after this job finishes to draft them.[/dim]",
            soft_wrap=True,
        )
    if not waited and report.job_id:
        err_console.print(
            f"[dim]Job {report.job_id} keeps running: pictograph auto-annotate get "
            f"{report.job_id}[/dim]",
            soft_wrap=True,
        )
    elif report.annotations_added and not failures:
        err_console.print(
            "[dim]These are drafts: review them before you train, least certain first - "
            f"pictograph images list {escape(_quoted(dataset))} --min-confidence-lt 0.5[/dim]",
            soft_wrap=True,
        )
    if failures:
        first = failures[0]
        raise CliError(
            f"Drafting {dataset} did not finish: {first.reason}",
            exit_code=exit_code_for(first.error_type),
            code=first.code or "batch_failed",
            details={"job_id": report.job_id, "failures": len(failures)},
        )


def _quoted(value: str) -> str:
    return f"'{value}'" if any(c.isspace() for c in value) else value


_SIMILAR_HELP = (
    "Label every object like your example polygon(s), across the image's folder or the "
    'whole dataset (the app\'s "Annotate similar"): draw a polygon around an example '
    "object on one image, name it with --class (every polygon of that class on the image) "
    "or --annotation <id>, and SAM3 finds every object like it in the OTHER images and "
    "saves each as an annotation. The labels are DRAFTS: review them before you export and "
    "train. Charged its measured GPU time once, when the job completes (a failed or "
    "cancelled job is free); --quote checks and prices it without running it."
)


@app.command("similar", help=_SIMILAR_HELP)
def similar(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    image: Annotated[
        str,
        typer.Argument(
            help=(
                "The image the example polygons are drawn on: its filename (the dataset "
                "root), its path in the dataset (val/x.jpg), or its id."
            )
        ),
    ],
    reference_class: Annotated[
        str | None,
        typer.Option(
            "--class",
            help="Use every polygon of this class on the image as an example.",
        ),
    ] = None,
    annotation: Annotated[
        list[str] | None,
        typer.Option(
            "--annotation",
            "-a",
            help=(
                "An example polygon by its annotation id (repeat for several; "
                "pictograph annotations get <dataset> <image> shows them). Instead of --class."
            ),
        ),
    ] = None,
    scope: Annotated[
        str,
        typer.Option(
            "--scope",
            help="directory (the other images in the image's folder) or dataset (every other image).",
        ),
    ] = "directory",
    target: Annotated[
        str | None,
        typer.Option(
            "--target",
            help="The class to label the matches as - one of the dataset's classes. The examples' class by default.",
        ),
    ] = None,
    output_type: Annotated[
        str,
        typer.Option(
            "--output-type",
            help="polygon (default; needs a polygon class), bbox, or obb (a rotated box); both box kinds need a box class.",
        ),
    ] = "polygon",
    confidence: Annotated[
        float | None,
        typer.Option(
            "--confidence",
            min=0.2,
            max=0.8,
            help="SAM3's detection confidence (0.2-0.8; the app's default if omitted). Raise it if unrelated objects slip in; lower it to catch more.",
        ),
    ] = None,
    quote_only: Annotated[
        bool,
        typer.Option(
            "--quote",
            help="Check and price the job WITHOUT running it: prints what it would label and cost.",
        ),
    ] = False,
    no_wait: Annotated[
        bool,
        typer.Option(
            "--no-wait",
            help="Start the job and exit; follow it with pictograph jobs wait <job>.",
        ),
    ] = False,
    timeout: Annotated[
        float, typer.Option("--timeout", min=1.0, help="Max seconds to wait for the job.")
    ] = 1800.0,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the job (or the quote) as JSON.")
    ] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if scope not in ("directory", "dataset"):
        raise typer.BadParameter("use directory or dataset", param_hint="--scope")
    if output_type not in ("polygon", "bbox", "obb"):
        raise typer.BadParameter("use polygon, bbox or obb", param_hint="--output-type")
    ids = [a for a in (annotation or []) if a.strip()]
    if ids and reference_class:
        raise typer.BadParameter(
            "name the examples one way: --class or --annotation, not both", param_hint="--class"
        )
    if not ids and not reference_class:
        raise typer.BadParameter(
            "name the example polygon(s): --class <name> (every polygon of that class on the "
            "image) or --annotation <id>",
            param_hint="--class",
        )
    client = get_client(api_key)
    kwargs = {
        "annotation_ids": ids,
        "reference_class": reference_class,
        "scope": scope,
        "target_class": target,
        "output_type": output_type,
        "confidence": confidence,
    }
    if quote_only:
        quote = client.auto_annotate.quote_similar(dataset, image, **kwargs)  # type: ignore[arg-type]
        if json_output:
            print_json(quote.model_dump(mode="json"))
        else:
            where = f"{quote.directory}" if quote.scope == "directory" else "the dataset"
            console.print(
                f"Would label {quote.total_images} image(s) in {escape(where)} as "
                f"{escape(quote.target_class)} ({quote.output_type}) from {quote.examples} "
                f"example(s): about {_usd(quote.estimated_credits)}, charged when it completes.",
                soft_wrap=True,
            )
            if quote.capped:
                err_console.print(
                    f"[dim]{quote.total_in_scope} image(s) are in scope; one job labels the "
                    f"first {quote.max_images}.[/dim]",
                    soft_wrap=True,
                )
        if not quote.sufficient:
            # The quote succeeded (exit 0); say on stderr why a start would not.
            err_console.print(
                f"[yellow]This job cannot start: it needs about {_usd(quote.estimated_credits)} "
                f"and {_usd(quote.remaining_credits)} remains.[/yellow]",
                soft_wrap=True,
            )
        return
    job = client.auto_annotate.similar(
        dataset,
        image,
        wait=not no_wait,
        timeout=timeout,
        **kwargs,  # type: ignore[arg-type]
    )
    if json_output:
        print_json(job.model_dump(mode="json", exclude_none=True))
        return
    if no_wait:
        console.print(
            f"Started job {job.job_id}: labelling {job.total_images} image(s) of "
            f"{escape(dataset)} (estimated {_usd(job.estimated_credits or 0)}, charged when "
            "it completes).",
            soft_wrap=True,
        )
        err_console.print(
            f"[dim]Follow it: pictograph jobs wait {job.job_id}[/dim]", soft_wrap=True
        )
        return
    console.print(
        f"Labelled {job.processed_images - job.failed_images} of {job.total_images} image(s) in "
        f"{escape(dataset)}: {job.total_annotations_added} annotation(s) added"
        + (f", {job.failed_images} image(s) failed." if job.failed_images else "."),
        soft_wrap=True,
    )
    if job.total_annotations_added:
        err_console.print(
            "[dim]These are drafts: review them before you train, least certain first - "
            f"pictograph images list {escape(_quoted(dataset))} --min-confidence-lt 0.5[/dim]",
            soft_wrap=True,
        )


def _usd(micro_usd: int) -> str:
    """µUSD as dollars a person reads: $0.04, $12.30."""
    return f"${micro_usd / 1_000_000:,.2f}"


@app.command("quote", help="Price a batch auto-annotate job WITHOUT running it.")
def quote(
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", help="Dataset name. Required to price existing images."),
    ] = None,
    images: Annotated[
        str | None,
        typer.Option("--images", help="Comma-separated images to price (as in batch)."),
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
    class_specs = _class_specs(classes)

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
        dataset=dataset,
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
