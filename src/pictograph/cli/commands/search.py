"""``pictograph search {similar,tags}`` - SigLIP visual + auto-tag search.

Two query modes, mirroring the SDK's :class:`pictograph.resources.search.Search`::

    pictograph search similar <image-id>                 # nearest-neighbour by embedding
    pictograph search similar <image-id> --threshold 0.8 --limit 20
    pictograph search tags --object car --scene outdoor  # JSONB auto-tag containment
    pictograph search tags --object car --dataset road-signs --json

``similar`` runs pgvector cosine similarity against the source image's
SigLIP-1152 embedding (scope defaults to the image's own directory; override
with ``--directory``). ``tags`` ANDs the supplied object/scene/attribute tags
against an image's ``image_auto_tags`` and requires at least one tag.
"""

from __future__ import annotations

from typing import Annotated

import typer

from pictograph.cli._client import get_client
from pictograph.cli._format import print_json, print_table

app = typer.Typer(no_args_is_help=True)


@app.command("similar", help="Find images visually similar to a source image (SigLIP embedding).")
def similar(
    dataset_name: Annotated[str, typer.Argument(help="The dataset's name.")],
    image: Annotated[str, typer.Argument(help="The source image's filename.")],
    threshold: Annotated[
        float,
        typer.Option("--threshold", "-t", help="Minimum cosine similarity in [0, 1]."),
    ] = 0.6,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max results (cap 500).")] = 50,
    directory_path: Annotated[
        str | None,
        typer.Option(
            "--directory",
            help="Override the source image's directory. Pass '/' for the dataset root.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    results = client.search.by_similarity(
        dataset_name,
        image,
        threshold=threshold,
        limit=limit,
        directory_path=directory_path,
    )
    if json_output:
        print_json([r.model_dump(mode="json", exclude_none=True) for r in results])
        return
    rows = [
        {
            "id": r.id,
            "filename": r.filename,
            "directory": r.virtual_directory_path,
            "status": r.status,
            "annotations": r.annotation_count,
            "similarity": f"{r.similarity:.3f}",
        }
        for r in results
    ]
    print_table(rows, title=f"Similar images ({len(rows)})")


@app.command(
    "tags", help="Search images by their SigLIP auto-tags (objects / scenes / attributes)."
)
def tags(
    objects: Annotated[
        list[str] | None,
        typer.Option("--object", help="Require an objects tag (repeatable; ANDed)."),
    ] = None,
    scenes: Annotated[
        list[str] | None,
        typer.Option("--scene", help="Require a scenes tag (repeatable; ANDed)."),
    ] = None,
    attributes: Annotated[
        list[str] | None,
        typer.Option("--attribute", help="Require an attributes tag (repeatable; ANDed)."),
    ] = None,
    dataset_name: Annotated[
        str | None,
        typer.Option("--dataset", help="Restrict to one dataset (omit for the whole org)."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max results (cap 500).")] = 50,
    offset: Annotated[int, typer.Option("--offset", help="Pagination offset.")] = 0,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    results = client.search.by_tag(
        objects=objects,
        scenes=scenes,
        attributes=attributes,
        dataset_name=dataset_name,
        limit=limit,
        offset=offset,
    )
    if json_output:
        print_json([r.model_dump(mode="json", exclude_none=True) for r in results])
        return
    rows = [
        {
            "id": r.id,
            "dataset": r.dataset_id,
            "filename": r.filename,
            "directory": r.virtual_directory_path,
            "status": r.status,
            "annotations": r.annotation_count,
        }
        for r in results
    ]
    print_table(rows, title=f"Tagged images ({len(rows)})")
