"""``pictograph annotations {get,save,bulk-save,delete}``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import TypeAdapter

from pictograph.cli._client import get_client
from pictograph.cli._format import print_json
from pictograph.models.annotation import Annotation

app = typer.Typer(no_args_is_help=True)


@app.command("get", help="Fetch annotations attached to an image.")
def get_annotations(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    image: Annotated[str, typer.Argument(help="Image filename (an id also works).")],
    directory_path: Annotated[
        str | None,
        typer.Option("--directory", help="Directory, when the filename is in more than one."),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    annotations = client.annotations.get(dataset, image, directory_path=directory_path)
    print_json([a.model_dump(mode="json", exclude_none=True) for a in annotations])


@app.command("save", help="Replace annotations on an image (full overwrite).")
def save_annotations(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    image: Annotated[str, typer.Argument(help="Image filename (an id also works).")],
    file: Annotated[
        Path,
        typer.Option(
            "--file",
            "-f",
            help="Path to a JSON file containing a list of annotations.",
        ),
    ],
    directory_path: Annotated[
        str | None,
        typer.Option("--directory", help="Directory, when the filename is in more than one."),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    raw = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        typer.echo(
            "JSON file must contain a list of annotations at the top level.",
            err=True,
        )
        raise typer.Exit(2)
    adapter: TypeAdapter[list[Annotation]] = TypeAdapter(list[Annotation])
    parsed = adapter.validate_python(raw)
    client = get_client(api_key)
    result = client.annotations.save(dataset, image, parsed, directory_path=directory_path)
    print_json(
        {
            "image_id": result.image_id,
            "previous_count": result.previous_count,
            "new_count": result.new_count,
            "status": result.status,
        }
    )


@app.command("bulk-save", help="Replace annotations on many images in one server-side call.")
def bulk_save_annotations(
    file: Annotated[
        Path,
        typer.Option(
            "--file",
            "-f",
            help='JSON object mapping image UUID -> list of annotations, e.g. {"img-1": [...]}.',
        ),
    ],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    raw = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        typer.echo(
            "JSON file must be an object mapping image_id -> [annotations].",
            err=True,
        )
        raise typer.Exit(2)
    adapter: TypeAdapter[list[Annotation]] = TypeAdapter(list[Annotation])
    saves = {image_id: adapter.validate_python(anns) for image_id, anns in raw.items()}
    client = get_client(api_key)
    result = client.annotations.bulk_save(saves)
    print_json(
        {
            "saved": [
                {
                    "image_id": s.image_id,
                    "previous_count": s.previous_count,
                    "new_count": s.new_count,
                    "status": s.status,
                }
                for s in result.saved
            ],
            "failed": [{"image_id": f.image_id, "error": f.error} for f in result.failed],
        }
    )


@app.command("delete", help="Remove every annotation from an image.")
def delete_annotations(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    image: Annotated[str, typer.Argument(help="Image filename (an id also works).")],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    directory_path: Annotated[
        str | None,
        typer.Option("--directory", help="Directory, when the filename is in more than one."),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(f"Delete all annotations on {image!r} in {dataset!r}?"):
        raise typer.Abort()
    client = get_client(api_key)
    result = client.annotations.delete(dataset, image, directory_path=directory_path)
    print_json(
        {
            "image_id": result.image_id,
            "deleted_count": result.deleted_count,
        }
    )


@app.command(
    "rename-class",
    help="Rename a class across a whole dataset - ontology + every annotation.",
)
def rename_class(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    old_name: Annotated[str, typer.Argument(help="Current class name.")],
    new_name: Annotated[str, typer.Argument(help="New class name.")],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(
        f"Rename class {old_name!r} to {new_name!r} across dataset {dataset!r}?"
    ):
        raise typer.Abort()
    client = get_client(api_key)
    project = client.datasets.get(dataset)
    result = client.annotations.rename_class(project.id, old_name, new_name)
    print_json(
        {
            "dataset_id": result.dataset_id,
            "old_name": result.old_name,
            "new_name": result.new_name,
            "images_updated": result.images_updated,
            "annotations_updated": result.annotations_updated,
            "config_updated": result.config_updated,
        }
    )


@app.command(
    "merge-class",
    help="Merge one class into another across a dataset - reassign annotations + drop the source.",
)
def merge_class(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    source_name: Annotated[str, typer.Argument(help="Class merged away.")],
    target_name: Annotated[str, typer.Argument(help="Class kept.")],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(
        f"Merge class {source_name!r} into {target_name!r} across dataset {dataset!r}?"
    ):
        raise typer.Abort()
    client = get_client(api_key)
    project = client.datasets.get(dataset)
    result = client.annotations.merge_class(project.id, source_name, target_name)
    print_json(
        {
            "dataset_id": result.dataset_id,
            "source_name": result.source_name,
            "target_name": result.target_name,
            "images_updated": result.images_updated,
            "annotations_updated": result.annotations_updated,
            "config_updated": result.config_updated,
        }
    )


@app.command(
    "delete-class",
    help="Delete a class from a dataset's ontology, optionally its annotations too.",
)
def delete_class(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    name: Annotated[str, typer.Argument(help="Class name to delete.")],
    with_annotations: Annotated[
        bool,
        typer.Option("--with-annotations", help="Also delete every annotation of this class."),
    ] = False,
    class_type: Annotated[
        str | None,
        typer.Option("--type", help="Narrow removal to one (name, type) ontology entry."),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    scope = "and its annotations " if with_annotations else ""
    if not yes and not typer.confirm(f"Delete class {name!r} {scope}from dataset {dataset!r}?"):
        raise typer.Abort()
    client = get_client(api_key)
    project = client.datasets.get(dataset)
    result = client.annotations.delete_class(
        project.id, name, class_type=class_type, delete_annotations=with_annotations
    )
    print_json(
        {
            "dataset_id": result.dataset_id,
            "name": result.name,
            "config_updated": result.config_updated,
            "images_updated": result.images_updated,
            "annotations_removed": result.annotations_removed,
        }
    )
