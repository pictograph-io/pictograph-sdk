"""``pictograph images {list,upload,bulk-upload,get,download,delete,tag}``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import typer

from pictograph.cli._client import get_client
from pictograph.cli._format import print_json

if TYPE_CHECKING:
    from pictograph.models.image import ImageSplit, ImageStatus

app = typer.Typer(no_args_is_help=True)


@app.command(
    "list", help="List a dataset's images (filter by --directory / --status), newest first."
)
def list_images(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    directory: Annotated[
        str | None, typer.Option("--directory", "-f", help="Virtual directory, e.g. /train.")
    ] = None,
    status: Annotated[
        str | None,
        typer.Option("--status", "-s", help="Stage: new | annotate | review | complete."),
    ] = None,
    split: Annotated[
        str | None,
        typer.Option("--split", help="Dataset split: train | val | test."),
    ] = None,
    include_archived: Annotated[
        bool, typer.Option("--archived", help="Include archived (soft-deleted) images.")
    ] = False,
    min_confidence_lt: Annotated[
        float | None,
        typer.Option(
            "--min-confidence-lt",
            help="Active-learning: only images with model confidence below this (0..1).",
        ),
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", "-n", help="Max images to return (pages as needed).")
    ] = 100,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    project = client.datasets.get(dataset)
    imgs = client.images.iter(
        project.id,
        directory_path=directory,
        status=cast("ImageStatus | None", status),
        split=cast("ImageSplit | None", split),
        include_archived=include_archived,
        min_confidence_lt=min_confidence_lt,
        max_total=limit,
    ).all()
    print_json([i.model_dump(mode="json", exclude_none=True) for i in imgs])


@app.command("upload", help="Upload a single image to a dataset.")
def upload_image(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    file_path: Annotated[Path, typer.Argument(help="Local image path.")],
    directory: Annotated[
        str, typer.Option("--directory", "-f", help="Virtual directory (e.g. /cars).")
    ] = "/",
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    project = client.datasets.get(dataset)
    image = client.images.upload(
        # already resolved above - a uuid short-circuits _resolve, so this
        # does not repeat the name lookup per file.
        dataset_name=project.id,
        file_path=file_path,
        directory_path=directory,
    )
    print_json(image.model_dump(mode="json", exclude_none=True))


@app.command("bulk-upload", help="Upload many images to one directory in two server round-trips.")
def bulk_upload_images(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    files: Annotated[list[Path], typer.Argument(help="Local image paths (up to 500).")],
    directory: Annotated[
        str, typer.Option("--directory", "-f", help="Virtual directory (e.g. /cars).")
    ] = "/",
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    project = client.datasets.get(dataset)
    result = client.images.bulk_upload(project.id, files, directory_path=directory)
    print_json(
        {
            "succeeded": [
                {"filename": u.filename, "id": u.id, "image_url": u.image_url}
                for u in result.succeeded
            ],
            "failed": [{"filename": f.filename, "error": f.error} for f in result.failed],
            "count": result.count,
        }
    )


@app.command(
    "upload-directory",
    help="Upload a local directory, recreating its subdirectory tree on the dataset.",
)
def upload_directory(
    dataset: Annotated[str, typer.Argument(help="Dataset name (created if missing).")],
    directory: Annotated[Path, typer.Argument(help="Local directory to walk (recursive).")],
    flat: Annotated[
        bool,
        typer.Option("--flat", help="Put every image at the dataset root instead."),
    ] = False,
    by_class: Annotated[
        bool,
        typer.Option(
            "--by-class",
            help="ImageFolder mode: use only the FIRST subdirectory level as the directory "
            "(cars/red/x.jpg -> /cars). Deeper nesting collapses.",
        ),
    ] = False,
    workers: Annotated[int, typer.Option("--workers", "-w", help="Concurrent uploads.")] = 8,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    """The CLI twin of the web app's "Add -> Directory".

    Default is full structure preservation: ``cars/red/0001.jpg`` lands in ``/cars/red``,
    and the same basename in two subdirectories stays two distinct images.
    """

    client = get_client(api_key)
    report = client.images.upload_from_directory(
        dataset,
        directory,
        organize_by_class=by_class,
        preserve_structure=not flat and not by_class,
        max_workers=workers,
    )
    print_json({**report.model_dump(mode="json"), "success": report.success})


@app.command("get", help="Fetch a single image's metadata.")
def get_image(
    dataset_name: Annotated[str, typer.Argument(help="The dataset's name.")],
    image: Annotated[str, typer.Argument(help="The image's filename (a UUID also works).")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    found = client.images.get(dataset_name, image)
    print_json(found.model_dump(mode="json", exclude_none=True))


@app.command("download", help="Download a single image to a local file.")
def download_image(
    dataset_name: Annotated[str, typer.Argument(help="The dataset's name.")],
    image: Annotated[str, typer.Argument(help="The image's filename (a UUID also works).")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    output.parent.mkdir(parents=True, exist_ok=True)
    client.images.download(dataset_name, image, output_path=output)
    print_json({"dataset": dataset_name, "image": image, "output_path": str(output)})


@app.command(
    "download-bundle",
    help="Download an image's data bundle (image + depth map + annotations) as a zip.",
)
def download_image_bundle(
    dataset_name: Annotated[str, typer.Argument(help="The dataset's name.")],
    image: Annotated[str, typer.Argument(help="The image's filename (a UUID also works).")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Zip path. Defaults to ./<image-stem>.zip."),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    # Default to the image's own stem in the CURRENT directory, which is what the
    # editor's "Image data" button produces - the two are meant to be equivalent,
    # so the no-flag invocation must land the same file.
    out = output or Path(f"{Path(image).stem or 'image'}.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    client.images.download_bundle(dataset_name, image, output_path=out)
    print_json({"dataset": dataset_name, "image": image, "output_path": str(out)})


@app.command(
    "delete",
    help="Permanently delete a single image (use --archive to soft-delete instead).",
)
def delete_image(
    dataset_name: Annotated[str, typer.Argument(help="The dataset's name.")],
    image: Annotated[str, typer.Argument(help="The image's filename (a UUID also works).")],
    archive: Annotated[
        bool,
        typer.Option(
            "--archive",
            help="Archive (soft-delete, restorable) instead of permanently deleting.",
        ),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    verb = "Archive" if archive else "Permanently delete"
    if not yes and not typer.confirm(f"{verb} image {image!r}?"):
        raise typer.Abort()
    client = get_client(api_key)
    client.images.delete(dataset_name, image, permanent=not archive)
    typer.echo(f"{'Archived' if archive else 'Deleted'} image {image!r}.")


@app.command("tag", help="Add (or with --remove, remove) user tags across many images.")
def tag_images(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    image_ids: Annotated[list[str], typer.Argument(help="Image UUIDs to tag.")],
    tag: Annotated[
        list[str] | None,
        typer.Option("--tag", "-t", help="User tag to apply (repeatable)."),
    ] = None,
    remove: Annotated[
        bool, typer.Option("--remove", help="Remove the given tags instead of adding.")
    ] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not tag:
        raise typer.BadParameter("Provide at least one --tag.")
    client = get_client(api_key)
    processed = client.images.bulk_tag(dataset, image_ids, tag, add=not remove)
    print_json({"processed": processed, "tags": tag, "added": not remove})


@app.command("review", help="Approve or request changes on an image (review workflow).")
def review_image(
    dataset_name: Annotated[str, typer.Argument(help="The dataset's name.")],
    image: Annotated[str, typer.Argument(help="The image's filename (a UUID also works).")],
    request_changes: Annotated[
        bool,
        typer.Option(
            "--request-changes",
            help="Send the image back to 'annotate' (default is approve → complete).",
        ),
    ] = False,
    note: Annotated[
        str | None,
        typer.Option("--note", "-n", help="Note for the annotator on --request-changes."),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    if request_changes:
        status = client.images.review(dataset_name, image, "request_changes", note=note)
        action = "request_changes"
    else:
        status = client.images.review(dataset_name, image, "approve")
        action = "approve"
    print_json({"image": image, "action": action, "status": status})


@app.command("split", help="Assign an image to a train/val/test split (or 'none' to clear).")
def set_image_split(
    dataset_name: Annotated[str, typer.Argument(help="The dataset's name.")],
    image: Annotated[str, typer.Argument(help="The image's filename (a UUID also works).")],
    split: Annotated[str, typer.Argument(help="train | val | test | none (clear)")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    value = None if split.lower() in ("none", "clear", "") else split.lower()
    if value is not None and value not in ("train", "val", "test"):
        raise typer.BadParameter("split must be train, val, test, or none")
    client = get_client(api_key)
    result = client.images.set_split(dataset_name, image, cast("ImageSplit | None", value))
    print_json({"image": image, "split": result})


@app.command(
    "rebalance",
    help="One-click Rebalance: assign a whole dataset a train/val/test split by ratio.",
)
def rebalance_splits(
    dataset_id: Annotated[str, typer.Argument(help="Dataset UUID.")],
    train: Annotated[int, typer.Option("--train", help="Train weight (%).")] = 70,
    val: Annotated[int, typer.Option("--val", help="Validation weight (%).")] = 20,
    test: Annotated[int, typer.Option("--test", help="Test weight (%).")] = 10,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic shuffle seed.")] = 42,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    result = client.images.assign_splits(dataset_id, train=train, val=val, test=test, seed=seed)
    print_json(result)
