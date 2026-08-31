"""``pictograph augment dataset`` - generate an augmented version of a dataset."""

from __future__ import annotations

from typing import Annotated

import typer

from pictograph.augment import build_ops
from pictograph.cli._client import get_client
from pictograph.cli._format import print_json, print_table

app = typer.Typer(no_args_is_help=True)

# Catalog for `pictograph augment ops` (flag → what it does). Keeps discovery in
# the CLI itself rather than only the docs.
_OP_CATALOG: list[tuple[str, str, str]] = [
    ("--flip", "geometric", "Random horizontal (left-right) flip."),
    ("--vflip", "geometric", "Random vertical (top-bottom) flip."),
    (
        "--rotate DEG",
        "geometric",
        "Rotate by up to ±DEG degrees; canvas expands so nothing is cropped.",
    ),
    ("--rotate90", "geometric", "Random lossless 90/180/270° rotation."),
    ("--crop SCALE", "geometric", "Random crop keeping SCALE-1.0 of each side; clips geometry."),
    ("--resize WxH", "geometric", "Resize to WIDTHxHEIGHT; scales geometry."),
    ("--shear DEG", "geometric", "Horizontal shear by up to ±DEG degrees; clips geometry."),
    ("--brightness F", "photometric", "Brightness jitter of ±F (e.g. 0.2 = ±20%)."),
    ("--contrast F", "photometric", "Contrast jitter of ±F."),
    ("--saturation F", "photometric", "Saturation jitter of ±F."),
    ("--hue DEG", "photometric", "Rotate hue by up to ±DEG degrees."),
    ("--grayscale", "photometric", "Convert to grayscale."),
    ("--blur R", "photometric", "Gaussian blur up to R pixels."),
    ("--noise A", "photometric", "Additive noise up to amount A (0-1)."),
    ("--cutout SIZE", "photometric", "Erase random rectangles up to SIZE of each side."),
]


@app.command("ops", help="List the available augmentation ops and their flags.")
def ops() -> None:
    """Print the augmentation catalog (geometric ops remap annotations; photometric don't)."""
    print_table(
        [{"flag": f, "kind": k, "description": d} for f, k, d in _OP_CATALOG],
        title="Augmentation ops",
    )


def _parse_resize(value: str) -> tuple[int, int]:
    try:
        w_str, h_str = value.lower().split("x")
        return int(w_str), int(h_str)
    except (ValueError, AttributeError) as exc:
        raise typer.BadParameter("--resize must look like WIDTHxHEIGHT, e.g. 640x480") from exc


@app.command("dataset", help="Generate an augmented version of a dataset (Roboflow-style).")
def dataset(
    source: Annotated[str, typer.Argument(help="Source dataset name.")],
    into: Annotated[
        str | None,
        typer.Option(
            "--into",
            help="Target dataset name (created if missing). Omit to append into the source.",
        ),
    ] = None,
    multiplier: Annotated[
        int, typer.Option("--multiplier", "-m", help="Augmented variants per source image.")
    ] = 3,
    flip: Annotated[bool, typer.Option("--flip", help="Random horizontal flip (p=0.5).")] = False,
    vflip: Annotated[bool, typer.Option("--vflip", help="Random vertical flip (p=0.5).")] = False,
    rotate: Annotated[
        float | None,
        typer.Option("--rotate", help="Rotate by up to ±DEG degrees (canvas expands)."),
    ] = None,
    rotate90: Annotated[
        bool, typer.Option("--rotate90", help="Random lossless 90/180/270° rotation.")
    ] = False,
    brightness: Annotated[
        float | None, typer.Option("--brightness", help="Brightness jitter ±F (e.g. 0.2 = ±20%).")
    ] = None,
    contrast: Annotated[
        float | None, typer.Option("--contrast", help="Contrast jitter ±F.")
    ] = None,
    saturation: Annotated[
        float | None, typer.Option("--saturation", help="Saturation jitter ±F.")
    ] = None,
    blur: Annotated[
        float | None, typer.Option("--blur", help="Gaussian blur up to R pixels.")
    ] = None,
    noise: Annotated[
        float | None, typer.Option("--noise", help="Additive noise up to amount A (0-1).")
    ] = None,
    grayscale: Annotated[bool, typer.Option("--grayscale", help="Convert to grayscale.")] = False,
    crop: Annotated[
        float | None,
        typer.Option("--crop", help="Random crop keeping SCALE-1.0 of each side (e.g. 0.8)."),
    ] = None,
    resize: Annotated[
        str | None, typer.Option("--resize", help="Resize to WIDTHxHEIGHT (e.g. 640x480).")
    ] = None,
    shear: Annotated[
        float | None, typer.Option("--shear", help="Horizontal shear up to ±DEG degrees.")
    ] = None,
    hue: Annotated[
        float | None, typer.Option("--hue", help="Rotate hue up to ±DEG degrees.")
    ] = None,
    cutout: Annotated[
        float | None,
        typer.Option("--cutout", help="Erase random rectangles up to SIZE of each side."),
    ] = None,
    no_original: Annotated[
        bool, typer.Option("--no-original", help="Don't copy the source images into a new dataset.")
    ] = False,
    max_images: Annotated[
        int | None, typer.Option("--max", help="Only process the first N source images.")
    ] = None,
    drop_class: Annotated[
        list[str] | None,
        typer.Option("--drop-class", help="Preprocessing: drop this class (repeatable)."),
    ] = None,
    skip_empty: Annotated[
        bool,
        typer.Option("--skip-empty", help="Preprocessing: skip images left with no annotations."),
    ] = False,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Seed for reproducible variants.")
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    """Build the op pipeline from the flags and materialize the augmented dataset."""
    # Build JSON-friendly specs (geometric first, then photometric - a stable order)
    # and hand them to the shared `build_ops`, the single source of flag/spec -> op truth.
    specs: list[dict[str, object]] = []
    if resize is not None:
        w, h = _parse_resize(resize)
        specs.append({"op": "resize", "width": w, "height": h})
    if crop is not None:
        specs.append({"op": "crop", "scale": crop})
    if flip:
        specs.append({"op": "flip"})
    if vflip:
        specs.append({"op": "vflip"})
    if rotate90:
        specs.append({"op": "rotate90"})
    if rotate is not None:
        specs.append({"op": "rotate", "degrees": rotate})
    if shear is not None:
        specs.append({"op": "shear", "degrees": shear})
    if brightness is not None:
        specs.append({"op": "brightness", "factor": brightness})
    if contrast is not None:
        specs.append({"op": "contrast", "factor": contrast})
    if saturation is not None:
        specs.append({"op": "saturation", "factor": saturation})
    if hue is not None:
        specs.append({"op": "hue_shift", "degrees": hue})
    if grayscale:
        specs.append({"op": "grayscale"})
    if blur is not None:
        specs.append({"op": "blur", "radius": blur})
    if noise is not None:
        specs.append({"op": "noise", "amount": noise})
    if cutout is not None:
        specs.append({"op": "cutout", "size": cutout})

    if not specs:
        raise typer.BadParameter(
            "Specify at least one augmentation, e.g. --flip --rotate 15 --brightness 0.2"
        )
    ops = build_ops(specs)

    client = get_client(api_key)
    report = client.images.augment(
        source,
        ops,
        multiplier=multiplier,
        into=into,
        include_original=not no_original,
        seed=seed,
        max_source_images=max_images,
        drop_classes=drop_class or None,
        skip_empty=skip_empty,
        on_progress=lambda done, total: typer.echo(
            f"\r  augmenting {done}/{total} source images…", nl=False, err=True
        ),
    )
    typer.echo("", err=True)  # newline after the progress line
    print_json(
        {
            "source": report.source,
            "target": report.target,
            "source_images": report.source_images,
            "originals_copied": report.originals_copied,
            "variants_created": report.variants_created,
            "annotations_written": report.annotations_written,
            "skipped_empty": report.skipped_empty,
            "failures": [{"image_id": f.image_id, "reason": f.reason} for f in report.failures],
            "ops": [op.name for op in ops],
        }
    )
