"""``pictograph tile dataset`` - slice a dataset into a tiled version."""

from __future__ import annotations

from typing import Annotated

import typer

from pictograph.cli._client import get_client
from pictograph.cli._format import print_json

app = typer.Typer(no_args_is_help=True)


@app.callback()
def _tile() -> None:
    """Slice a dataset into an N x M grid of tiles (small-object-detection preprocessing).

    Keeps ``dataset`` an explicit subcommand (a lone command would otherwise flatten
    the group), so the CLI reads ``pictograph tile dataset <name>``.
    """


@app.command("dataset", help="Slice every image into a rowsxcols grid of tiles (Roboflow-style).")
def dataset(
    source: Annotated[str, typer.Argument(help="Source dataset name.")],
    into: Annotated[
        str | None,
        typer.Option(
            "--into",
            help="Target dataset name (created if missing). Omit to append into the source.",
        ),
    ] = None,
    rows: Annotated[int, typer.Option("--rows", "-r", help="Grid rows per image.")] = 2,
    cols: Annotated[int, typer.Option("--cols", "-c", help="Grid columns per image.")] = 2,
    overlap: Annotated[
        float,
        typer.Option("--overlap", help="Fractional overlap added to each tile edge (0.0-0.9)."),
    ] = 0.0,
    min_visibility: Annotated[
        float,
        typer.Option(
            "--min-visibility",
            help="Drop an annotation from a tile when less than this fraction of its area survives.",
        ),
    ] = 0.1,
    exclude_empty: Annotated[
        bool,
        typer.Option("--exclude-empty", help="Don't upload tiles that have no annotations."),
    ] = False,
    max_images: Annotated[
        int | None, typer.Option("--max", help="Only process the first N source images.")
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    """Materialize a tiled version of the dataset."""
    if rows < 1 or cols < 1:
        raise typer.BadParameter("--rows and --cols must both be >= 1")

    client = get_client(api_key)
    report = client.images.tile(
        source,
        rows=rows,
        cols=cols,
        overlap=overlap,
        min_visibility=min_visibility,
        include_empty=not exclude_empty,
        into=into,
        max_source_images=max_images,
        on_progress=lambda done, total: typer.echo(
            f"\r  tiling {done}/{total} source images…", nl=False, err=True
        ),
    )
    typer.echo("", err=True)  # newline after the progress line
    print_json(
        {
            "source": report.source,
            "target": report.target,
            "source_images": report.source_images,
            "tiles_created": report.tiles_created,
            "empty_tiles": report.empty_tiles,
            "annotations_written": report.annotations_written,
            "failures": [{"image_id": f.image_id, "reason": f.reason} for f in report.failures],
            "grid": f"{rows}x{cols}",
        }
    )
