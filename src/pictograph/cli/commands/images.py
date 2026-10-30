"""``pictograph images {list,upload,bulk-upload,get,download,delete,tag,...}``.

Bulk changes over a selection of images - ``move``, ``copy``, ``set-status``,
``archive``, ``restore`` - take the selection described in
:mod:`pictograph.cli._selection` (ids, a file of ids, filters, or ``--all``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, cast

import typer

from pictograph.cli import _selection as sel
from pictograph.cli._client import get_client
from pictograph.cli._errors import CliError, ExitCode, command_group
from pictograph.cli._format import err_console, print_json

if TYPE_CHECKING:
    from pictograph.models.batch import BatchResult, DuplicateHandling
    from pictograph.models.image import ImageSplit, ImageStatus

app = command_group(no_args_is_help=True)


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
        directory=directory,
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
        dataset=project.id,
        file_path=file_path,
        directory=directory,
    )
    print_json(image.model_dump(mode="json", exclude_none=True))


@app.command("bulk-upload", help="Upload many images to one directory in two server round-trips.")
def bulk_upload_images(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    files: Annotated[list[Path], typer.Argument(help="Local image paths (up to 500).")],
    directory: Annotated[
        str, typer.Option("--directory", "-f", help="Virtual directory (e.g. /cars).")
    ] = "/",
    no_embed: Annotated[
        bool,
        typer.Option(
            "--no-embed",
            help="Skip the SigLIP embedding / auto-tag pass; run `pictograph datasets embed` later.",
        ),
    ] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    project = client.datasets.get(dataset)
    result = client.images.bulk_upload(project.id, files, directory=directory, embed=not no_embed)
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
    dataset: Annotated[str, typer.Argument(help="The dataset's name.")],
    image: Annotated[
        str, typer.Argument(help="The image's filename (unique in the dataset), or its UUID.")
    ],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    found = client.images.get(dataset, image)
    print_json(found.model_dump(mode="json", exclude_none=True))


@app.command("download", help="Download a single image to a local file.")
def download_image(
    dataset: Annotated[str, typer.Argument(help="The dataset's name.")],
    image: Annotated[
        str,
        typer.Argument(
            help="The image's path: directory/filename (a bare filename means the dataset root), or its UUID."
        ),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    output.parent.mkdir(parents=True, exist_ok=True)
    client.images.download(dataset, image, output_path=output)
    print_json({"dataset": dataset, "image": image, "output_path": str(output)})


@app.command(
    "download-bundle",
    help="Download an image's data bundle (image + depth map + annotations) as a zip.",
)
def download_image_bundle(
    dataset: Annotated[str, typer.Argument(help="The dataset's name.")],
    image: Annotated[
        str,
        typer.Argument(
            help="The image's path: directory/filename (a bare filename means the dataset root), or its UUID."
        ),
    ],
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
    client.images.download_bundle(dataset, image, output_path=out)
    print_json({"dataset": dataset, "image": image, "output_path": str(out)})


@app.command(
    "delete",
    help="Permanently delete a single image (use --archive to soft-delete instead).",
)
def delete_image(
    dataset: Annotated[str, typer.Argument(help="The dataset's name.")],
    image: Annotated[
        str,
        typer.Argument(
            help="The image's path: directory/filename (a bare filename means the dataset root), or its UUID."
        ),
    ],
    archive: Annotated[
        bool,
        typer.Option(
            "--archive",
            help="Archive (soft-delete, restorable) instead of permanently deleting.",
        ),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    verb = "Archive" if archive else "Permanently delete"
    if not yes and not typer.confirm(f"{verb} image {image!r}?"):
        raise typer.Abort()
    client = get_client(api_key)
    client.images.delete(dataset, image, permanent=not archive)
    if json_output:
        print_json(
            {"dataset": dataset, "image": image, "archived": archive, "deleted": not archive}
        )
        return
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
    dataset: Annotated[str, typer.Argument(help="The dataset's name.")],
    image: Annotated[
        str,
        typer.Argument(
            help="The image's path: directory/filename (a bare filename means the dataset root), or its UUID."
        ),
    ],
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
        status = client.images.review(dataset, image, "request_changes", note=note)
        action = "request_changes"
    else:
        status = client.images.review(dataset, image, "approve")
        action = "approve"
    print_json({"image": image, "action": action, "status": status})


@app.command("split", help="Assign an image to a train/val/test split (or 'none' to clear).")
def set_image_split(
    dataset: Annotated[str, typer.Argument(help="The dataset's name.")],
    image: Annotated[
        str,
        typer.Argument(
            help="The image's path: directory/filename (a bare filename means the dataset root), or its UUID."
        ),
    ],
    split: Annotated[str, typer.Argument(help="train | val | test | none (clear)")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    value = None if split.lower() in ("none", "clear", "") else split.lower()
    if value is not None and value not in ("train", "val", "test"):
        raise typer.BadParameter("split must be train, val, test, or none")
    client = get_client(api_key)
    result = client.images.set_split(dataset, image, cast("ImageSplit | None", value))
    print_json({"image": image, "split": result})


@app.command(
    "rebalance",
    help="One-click Rebalance: assign a whole dataset a train/val/test split by ratio.",
)
def rebalance_splits(
    dataset: Annotated[str, typer.Argument(help="The dataset to rebalance (its name or UUID).")],
    train: Annotated[
        int | None,
        typer.Option("--train", help="Train weight (%). Omitted: the dataset's saved split."),
    ] = None,
    val: Annotated[
        int | None,
        typer.Option("--val", help="Validation weight (%). Omitted: the dataset's saved split."),
    ] = None,
    test: Annotated[
        int | None,
        typer.Option("--test", help="Test weight (%). Omitted: the dataset's saved split."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic shuffle seed.")] = 42,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="random (one shuffle of the dataset) or embedding (the ratio within each embedding cluster).",
        ),
    ] = "random",
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    """A weight you leave out is the dataset's saved split percentage (70/20/10 unless changed)."""
    if mode not in ("random", "embedding"):
        raise typer.BadParameter("use random or embedding.", param_hint="--mode")
    client = get_client(api_key)
    result = client.images.assign_splits(
        dataset,
        train=train,
        val=val,
        test=test,
        seed=seed,
        mode=cast("Literal['random', 'embedding']", mode),
    )
    print_json(result)


# ───────────── bulk changes over a selection ─────────────


_DONE = {
    "move": "moved",
    "copy": "copied",
    "set-status": "updated",
    "archive": "archived",
    "restore": "restored",
}


def _finish(payload: dict[str, object], failed: int) -> None:
    """Print the report; some images failing is exit 1, each one named in it."""
    print_json(payload)
    if failed:
        raise CliError(
            f"{failed} of {payload['selected']} image(s) were not "
            f"{_DONE[str(payload['operation'])]} - each is listed under `failed` with the reason.",
            exit_code=ExitCode.FAILED,
            code="some_images_failed",
            details={"failed": failed},
        )


def _nothing_selected(dataset: str, selection: sel.Selection) -> None:
    raise CliError(
        f"No images of {dataset!r} match {selection.how}.",
        exit_code=ExitCode.NOT_FOUND,
        code="no_images",
        fix="Check the selection with pictograph images list (same filters).",
    )


@app.command(
    "move",
    help="Move images to another folder of the dataset. A name already in that folder "
    "is suffixed -1, -2, ... (the image there is never touched). Changes no file.",
)
def move_images(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    to: Annotated[str, typer.Option("--to", help="Destination folder ('/val'; '/' is the root).")],
    images: sel.ImagesOption = None,
    from_file: sel.FromFileOption = None,
    directory: sel.DirectoryOption = None,
    status: sel.StatusFilterOption = None,
    split: sel.SplitOption = None,
    select_all: sel.AllOption = False,
    dry_run: sel.DryRunOption = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    chosen = sel.select(
        client,
        dataset,
        images=images,
        from_file=from_file,
        directory=directory,
        status=status,
        split=split,
        select_all=select_all,
    )
    if dry_run:
        print_json(sel.dry_run_report("move", dataset, chosen))
        return
    if not chosen.image_ids:
        _nothing_selected(dataset, chosen)
    totals = sel.run_in_chunks(
        chosen.image_ids,
        lambda ids: client.batch.move(dataset, ids, target_directory=to),
    )
    _finish(
        sel.report("move", dataset, chosen, totals, to=to, renamed=totals.renamed),
        len(totals.failed),
    )


@app.command(
    "copy",
    help="Copy images into another folder of the dataset (a server-side copy: instant, "
    "no transfer; copies start without annotations unless --with-annotations), or with "
    "--to-dataset into another dataset of your organization: the copies share the files and "
    "keep their annotations, tags, split and stage, and the classes they use are added to it.",
)
def copy_images(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    to: Annotated[
        str | None,
        typer.Option(
            "--to",
            help="Destination folder ('/val'; '/' is the root). With --to-dataset, the folder "
            "in that dataset (default: its root).",
        ),
    ] = None,
    to_dataset: Annotated[
        str | None,
        typer.Option(
            "--to-dataset",
            help="Copy into this dataset instead (it must exist: pictograph datasets create). "
            "Names already in the folder are renamed -1, -2, ...",
        ),
    ] = None,
    duplicates: Annotated[
        str,
        typer.Option(
            "--duplicates",
            help="A name already in the folder: rename (add -2, -3, ...), skip, or overwrite.",
        ),
    ] = "rename",
    with_annotations: Annotated[
        bool, typer.Option("--with-annotations", help="Copy annotations and stage too.")
    ] = False,
    images: sel.ImagesOption = None,
    from_file: sel.FromFileOption = None,
    directory: sel.DirectoryOption = None,
    status: sel.StatusFilterOption = None,
    split: sel.SplitOption = None,
    select_all: sel.AllOption = False,
    dry_run: sel.DryRunOption = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if duplicates not in ("rename", "skip", "overwrite"):
        raise typer.BadParameter("use rename, skip or overwrite.", param_hint="--duplicates")
    if to is None and to_dataset is None:
        raise typer.BadParameter(
            "name the destination folder ('/val'; '/' is the root), or a dataset with "
            "--to-dataset.",
            param_hint="--to",
        )
    if to_dataset is not None and duplicates != "rename":
        raise typer.BadParameter(
            "a copy into another dataset renames a name already in the folder (-1, -2, ...); "
            "it cannot skip or overwrite.",
            param_hint="--duplicates",
        )
    client = get_client(api_key)
    chosen = sel.select(
        client,
        dataset,
        images=images,
        from_file=from_file,
        directory=directory,
        status=status,
        split=split,
        select_all=select_all,
    )
    folder = to or "/"
    if dry_run:
        extra = {"to_dataset": to_dataset, "to": folder} if to_dataset is not None else {}
        print_json(sel.dry_run_report("copy", dataset, chosen, **extra))
        return
    if not chosen.image_ids:
        _nothing_selected(dataset, chosen)
    if to_dataset is not None:
        target = to_dataset
        classes_added: list[str] = []

        def copy_chunk(ids: list[str]) -> BatchResult:
            result = client.batch.copy_to_dataset(dataset, ids, target, target_directory=folder)
            classes_added.extend(c for c in result.classes_added if c not in classes_added)
            return result

        totals = sel.run_in_chunks(chosen.image_ids, copy_chunk)
        _finish(
            sel.report(
                "copy",
                dataset,
                chosen,
                totals,
                to_dataset=target,
                to=folder,
                renamed=totals.renamed,
                classes_added=classes_added,
            ),
            len(totals.failed),
        )
        return
    totals = sel.run_in_chunks(
        chosen.image_ids,
        lambda ids: client.batch.copy(
            dataset,
            ids,
            target_directory=folder,
            duplicate_handling=cast("DuplicateHandling", duplicates),
            copy_annotations=with_annotations,
        ),
    )
    _finish(sel.report("copy", dataset, chosen, totals, to=folder), len(totals.failed))


@app.command(
    "set-status",
    help="Move images to a stage: new, annotate, review or complete. To record a "
    "review decision with a note, use pictograph images review.",
)
def set_status(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    to: Annotated[str, typer.Option("--to", help="The stage: new, annotate, review or complete.")],
    images: sel.ImagesOption = None,
    from_file: sel.FromFileOption = None,
    directory: sel.DirectoryOption = None,
    status: sel.StatusFilterOption = None,
    split: sel.SplitOption = None,
    select_all: sel.AllOption = False,
    dry_run: sel.DryRunOption = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if to not in sel.STATUSES:
        raise typer.BadParameter(f"use one of {', '.join(sel.STATUSES)}.", param_hint="--to")
    client = get_client(api_key)
    chosen = sel.select(
        client,
        dataset,
        images=images,
        from_file=from_file,
        directory=directory,
        status=status,
        split=split,
        select_all=select_all,
    )
    if dry_run:
        print_json(sel.dry_run_report("set-status", dataset, chosen))
        return
    if not chosen.image_ids:
        _nothing_selected(dataset, chosen)
    totals = sel.run_in_chunks(
        chosen.image_ids, lambda ids: client.batch.update(dataset, ids, status=to)
    )
    _finish(sel.report("set-status", dataset, chosen, totals, to=to), len(totals.failed))


@app.command(
    "archive",
    help="Archive images: they move to the dataset's Archive and out of new exports; "
    "pictograph images restore brings them back. The files are kept.",
)
def archive_images(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    images: sel.ImagesOption = None,
    from_file: sel.FromFileOption = None,
    directory: sel.DirectoryOption = None,
    status: sel.StatusFilterOption = None,
    split: sel.SplitOption = None,
    select_all: sel.AllOption = False,
    dry_run: sel.DryRunOption = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    chosen = sel.select(
        client,
        dataset,
        images=images,
        from_file=from_file,
        directory=directory,
        status=status,
        split=split,
        select_all=select_all,
    )
    if dry_run:
        print_json(sel.dry_run_report("archive", dataset, chosen))
        return
    if not chosen.image_ids:
        _nothing_selected(dataset, chosen)
    if not yes and not typer.confirm(
        f"Archive {len(chosen.image_ids)} image(s) of {dataset!r} ({chosen.how})?"
    ):
        raise typer.Abort()
    totals = sel.run_in_chunks(
        chosen.image_ids, lambda ids: client.batch.delete(dataset, ids, permanent=False)
    )
    _finish(sel.report("archive", dataset, chosen, totals), len(totals.failed))


@app.command(
    "restore",
    help="Bring archived images back. Filters (or --all) select among the archived images.",
)
def restore_images(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    images: sel.ImagesOption = None,
    from_file: sel.FromFileOption = None,
    directory: sel.DirectoryOption = None,
    status: sel.StatusFilterOption = None,
    split: sel.SplitOption = None,
    select_all: sel.AllOption = False,
    dry_run: sel.DryRunOption = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    chosen = sel.select(
        client,
        dataset,
        images=images,
        from_file=from_file,
        directory=directory,
        status=status,
        split=split,
        select_all=select_all,
        archived=True,
    )
    if dry_run:
        print_json(sel.dry_run_report("restore", dataset, chosen))
        return
    if not chosen.image_ids:
        _nothing_selected(dataset, chosen)
    totals = sel.run_in_chunks(
        chosen.image_ids, lambda ids: client.batch.update(dataset, ids, is_archived=False)
    )
    if totals.processed:
        err_console.print(
            f"[dim]Restored {totals.processed} image(s) to {dataset}.[/dim]", soft_wrap=True
        )
    _finish(sel.report("restore", dataset, chosen, totals), len(totals.failed))
