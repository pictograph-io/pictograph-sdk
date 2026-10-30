"""``pictograph datasets {list,get,create,update,delete,archive,unarchive,insights,duplicates,download,export,import-coco,import-public,import-status,storage,freeze,restore}``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, cast, get_args

import typer

from pictograph.cli._client import get_client
from pictograph.cli._errors import command_group
from pictograph.cli._format import console, print_json, print_table
from pictograph.cli.commands import annotations as annotations_cmd
from pictograph.models.export import ExportFormat

app = command_group(no_args_is_help=True)


@app.command("list", help="List datasets in your organization.")
def list_datasets(
    limit: Annotated[
        int,
        typer.Option(
            "--limit", "-n", min=1, help="How many to return (pages through the API as needed)."
        ),
    ] = 100,
    json_output: Annotated[bool, typer.Option("--json", help="Emit raw JSON.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    # Paged: one request takes at most 1000 rows, and a larger --limit used to 422.
    datasets = list(client.datasets.iter(page_size=min(limit, 1000), max_total=limit))
    rows = [
        {
            "name": d.name,
            "id": d.id,
            "images": d.image_count,
            "completed": d.completed_image_count,
            "size": d.total_size,
        }
        for d in datasets
    ]
    if json_output:
        print_json([d.model_dump(mode="json", exclude_none=True) for d in datasets])
    else:
        print_table(rows, title=f"Datasets ({len(rows)})")


@app.command("get", help="Fetch a dataset by name.")
def get_dataset(
    name: Annotated[str, typer.Argument(help="Dataset name.")],
    include_images: Annotated[bool, typer.Option("--include-images", "-I")] = False,
    images_limit: Annotated[int, typer.Option("--images-limit")] = 100,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    dataset = client.datasets.get(name, include_images=include_images, images_limit=images_limit)
    print_json(dataset.model_dump(mode="json", exclude_none=True))


@app.command(
    "insights", help="Dataset Health / Insights - class balance, labeling progress, dimensions."
)
def dataset_insights(
    name: Annotated[str, typer.Argument(help="Dataset name.")],
    json_out: Annotated[
        bool, typer.Option("--json", help="Print the raw JSON instead of a summary.")
    ] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    insights = client.datasets.insights(name)
    if json_out:
        print_json(insights.model_dump(mode="json", exclude_none=True))
        return

    console.print(f"[bold]{name}[/bold] - Dataset Insights")
    console.print(
        f"  {insights.total_images:,} images · {insights.total_annotations:,} annotations "
        f"· {insights.avg_annotations_per_image:g} avg/image · "
        f"{insights.annotated_images:,} annotated ({insights.unannotated_images:,} unannotated)"
    )
    sc = insights.status_counts
    console.print(
        f"  Stages: new {sc.new:,} · annotate {sc.annotate:,} · "
        f"review {sc.review:,} · complete {sc.complete:,}"
    )
    if insights.type_counts:
        console.print(
            "  Types: "
            + " · ".join(
                f"{k} {v:,}" for k, v in sorted(insights.type_counts.items(), key=lambda kv: -kv[1])
            )
        )
    d = insights.dimensions
    if d.images_with_dimensions:
        console.print(
            f"  Dimensions: W {d.min_width}-{d.max_width} · H {d.min_height}-{d.max_height} "
            f"· landscape {d.orientation.landscape:,} · portrait {d.orientation.portrait:,} "
            f"· square {d.orientation.square:,} ({d.distinct_size_count:,} distinct sizes)"
        )

    balance = sorted(insights.class_annotation_counts.items(), key=lambda kv: -kv[1])
    if balance:
        print_table(
            [
                {
                    "class": name,
                    "annotations": count,
                    "images": insights.class_image_counts.get(name, 0),
                }
                for name, count in balance[:25]
            ],
            columns=["class", "annotations", "images"],
            title="Class balance",
        )


@app.command(
    "duplicates",
    help="Find near-duplicate images (keep one per cluster, archive the rest).",
)
def dataset_duplicates(
    name: Annotated[str, typer.Argument(help="Dataset name.")],
    threshold: Annotated[
        float | None,
        typer.Option("--threshold", "-t", help="Min cosine similarity 0.5-0.9999 (default 0.92)."),
    ] = None,
    sample: Annotated[
        int | None,
        typer.Option("--sample", help="Max source images to scan (default 1000, cap 2000)."),
    ] = None,
    directory: Annotated[
        str | None,
        typer.Option(
            "--directory", "-f", help="Scope the scan to one virtual directory (e.g. /train)."
        ),
    ] = None,
    json_out: Annotated[
        bool, typer.Option("--json", help="Print the raw JSON instead of a summary.")
    ] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    dup = client.datasets.near_duplicates(
        name, threshold=threshold, sample=sample, directory=directory
    )
    if json_out:
        print_json(dup.model_dump(mode="json", exclude_none=True))
        return

    console.print(f"[bold]{name}[/bold] - Near-duplicate images")
    console.print(
        f"  {dup.group_count:,} duplicate groups · {dup.redundant_count:,} redundant "
        f"· {dup.analyzed:,} of {dup.total_images:,} images analyzed "
        f"(>= {dup.threshold:g} similarity)"
    )
    if dup.sample_capped:
        console.print(
            f"  [yellow]Analyzed the first {dup.sample_limit:,} images - "
            f"raise --sample to scan more.[/yellow]"
        )
    if dup.groups:
        print_table(
            [
                {
                    "group": i + 1,
                    "size": g.size,
                    "max match": f"{round(g.max_similarity * 100)}%",
                    "keep": g.members[0].filename if g.members else "",
                }
                for i, g in enumerate(dup.groups[:25])
            ],
            columns=["group", "size", "max match", "keep"],
            title="Near-duplicate clusters",
        )
    else:
        console.print("  No near-duplicates found - your dataset looks clean.")


@app.command("create", help="Create a new dataset.")
def create_dataset(
    name: Annotated[str, typer.Argument(help="Dataset name.")],
    description: Annotated[str | None, typer.Option("--description", "-d")] = None,
    annotation_types: Annotated[
        list[str] | None,
        typer.Option(
            "--type",
            "-t",
            help="Allowed annotation type (repeatable). Defaults to bbox. "
            "Values: bbox/box, polygon, polyline, keypoint.",
        ),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    dataset = client.datasets.create(
        name, description=description, annotation_types=annotation_types
    )
    print_json(dataset.model_dump(mode="json", exclude_none=True))


@app.command(
    "update",
    help="Patch a dataset's metadata (rename / README / retype). Classes: pictograph classes.",
)
def update_dataset(
    name: Annotated[str, typer.Argument(help="Current dataset name.")],
    new_name: Annotated[
        str | None, typer.Option("--new-name", help="Rename to this (unique within the org).")
    ] = None,
    readme: Annotated[
        str | None,
        typer.Option(
            "--readme",
            help="Replace the dataset card with this markdown file ('-' reads stdin). "
            "The card is then yours: the platform stops regenerating it.",
        ),
    ] = None,
    description: Annotated[
        str | None, typer.Option("--description", "-d", help="Deprecated: use --readme.")
    ] = None,
    annotation_types: Annotated[
        list[str] | None,
        typer.Option(
            "--type",
            "-t",
            help="Replace the allowed annotation types (repeatable). "
            "Values: bbox/box, polygon, polyline, keypoint.",
        ),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    card = None
    if readme is not None:
        card = sys.stdin.read() if readme == "-" else _read_text(Path(readme), "--readme")
    client = get_client(api_key)
    dataset = client.datasets.update(
        name,
        new_name=new_name,
        readme=card,
        description=description,
        annotation_types=annotation_types,
    )
    print_json(dataset.model_dump(mode="json", exclude_none=True))


def _read_text(path: Path, flag: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise typer.BadParameter(f"No such file: {path}", param_hint=flag) from None


@app.command(
    "archive", help="Archive a dataset - hide it from the default list without deleting it."
)
def archive_dataset(
    name: Annotated[str, typer.Argument(help="Dataset name.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    dataset = client.datasets.archive(name)
    print_json(dataset.model_dump(mode="json", exclude_none=True))


@app.command("unarchive", help="Bring an archived dataset back into the default list.")
def unarchive_dataset(
    name: Annotated[str, typer.Argument(help="Dataset name.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    dataset = client.datasets.unarchive(name)
    print_json(dataset.model_dump(mode="json", exclude_none=True))


@app.command("delete", help="Permanently delete a dataset and its images.")
def delete_dataset(
    name: Annotated[str, typer.Argument(help="Dataset name.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(f"Permanently delete dataset {name!r}? This is irreversible."):
        raise typer.Abort()
    client = get_client(api_key)
    client.datasets.delete(name)
    if json_output:
        print_json({"dataset": name, "deleted": True})
        return
    typer.echo(f"Deleted dataset {name!r}.")


@app.command("download", help="Bulk-download a dataset's images and annotations.")
def download_dataset(
    name: Annotated[str, typer.Argument(help="Dataset name.")],
    output_dir: Annotated[Path, typer.Option("--output", "-o")] = Path("./dataset"),
    mode: Annotated[str, typer.Option(help="full / images_only / annotations_only")] = "full",
    workers: Annotated[int, typer.Option("--workers", "-w")] = 10,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    report = client.datasets.download(
        name,
        output_path=output_dir,
        mode=mode,  # type: ignore[arg-type]
        max_workers=workers,
    )
    print_json(
        {
            "dataset_id": report.dataset_id,
            "images_downloaded": report.images_downloaded,
            "annotations_downloaded": report.annotations_downloaded,
            "failures": len(report.failures),
            "output_dir": str(output_dir),
        }
    )


@app.command(
    "export",
    help="Export a dataset to a downloadable ZIP in any Pictograph format "
    "(pictograph / coco / yolo / pascal_voc / darwin / cvat / datumaro / labelme / csv), "
    "built server-side by Pictograph's own converters.",
)
def export_dataset(
    name: Annotated[str, typer.Argument(help="Dataset name.")],
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="pictograph / coco / yolo / pascal_voc / darwin / cvat / datumaro / labelme / csv",
        ),
    ] = "coco",
    output_dir: Annotated[
        Path, typer.Option("--output", "-o", help="Directory to write the export ZIP into.")
    ] = Path("./export"),
    include_images: Annotated[
        bool, typer.Option("--include-images", help="Bundle the source images into the ZIP.")
    ] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    fmt_norm = fmt.lower().replace("-", "_")
    valid = get_args(ExportFormat)
    if fmt_norm not in valid:
        raise typer.BadParameter(
            f"Unsupported format {fmt!r}. Choose one of: {', '.join(valid)}.",
            param_hint="--format",
        )

    client = get_client(api_key)
    # Server-side export via Pictograph's OWN converters (utils/export_formats) -
    # no third-party dependency. create() waits for completion by default.
    export = client.exports.create(
        name,
        f"{name}-{fmt_norm}",
        format=cast("ExportFormat", fmt_norm),
        include_images=include_images,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{export.name}.zip"
    client.exports.download(name, export.name, dest)

    print_json(
        {
            "dataset": name,
            "format": fmt_norm,
            "export": export.name,
            "output": str(dest),
        }
    )


@app.command(
    "import-coco",
    help="Same as `pictograph annotations import-coco` (kept for scripts that call it here).",
)
def import_coco(
    name: Annotated[str, typer.Argument(help="Dataset name (it must hold the images).")],
    coco_file: Annotated[Path, typer.Argument(help="The COCO JSON file.")],
    create_classes: annotations_cmd.CreateClassesOption = True,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    annotations_cmd.import_coco(name, coco_file, create_classes=create_classes, api_key=api_key)


@app.command(
    "import-public",
    help=(
        "Import (fork) a public dataset into your organization as a private copy. REF is "
        "org/dataset (find one with `pictograph explore datasets`). Waits until the copy "
        "is ready unless --no-wait."
    ),
)
def import_public(
    ref: Annotated[
        str,
        typer.Argument(
            help="org/dataset - the publishing organization's slug and the dataset's slug or name."
        ),
    ],
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Name for the copy (default: the source's; a name you use already gets '(imported)').",
        ),
    ] = None,
    no_wait: Annotated[
        bool,
        typer.Option(
            "--no-wait",
            help="Start it and print its import_id; follow with `datasets import-status <id> --wait`.",
        ),
    ] = False,
    timeout: Annotated[
        float, typer.Option("--timeout", min=1.0, help="Max seconds to wait for the copy.")
    ] = 1800.0,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    job = client.datasets.import_public(ref, name=name, wait=not no_wait, timeout=timeout)
    print_json(job.model_dump(mode="json"))


@app.command(
    "import-status",
    help="Show a public-dataset import's progress (the import_id `import-public --no-wait` printed).",
)
def import_status(
    import_id: Annotated[str, typer.Argument(help="The import's id.")],
    wait: Annotated[bool, typer.Option("--wait", help="Wait until it finishes.")] = False,
    timeout: Annotated[
        float, typer.Option("--timeout", min=1.0, help="Max seconds to wait (with --wait).")
    ] = 1800.0,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    if wait:
        job = client.datasets.wait_for_public_import(import_id, timeout=timeout)
    else:
        job = client.datasets.get_public_import(import_id)
    print_json(job.model_dump(mode="json"))


@app.command(
    "embed",
    help="Queue SigLIP embeddings + auto-tags for up to --limit images that have none.",
)
def dataset_embed(
    name: Annotated[str, typer.Argument(help="Dataset name.")],
    limit: Annotated[
        int,
        typer.Option("--limit", min=0, max=500, help="Images to queue (0 only counts)."),
    ] = 500,
    json_output: Annotated[bool, typer.Option("--json", help="Emit raw JSON.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    """The paced half of a large import: images registered with ``--no-embed`` (or
    whose embedding pass never finished) get their vectors here, one batch per call."""
    client = get_client(api_key)
    result = client.datasets.backfill_embeddings(name, limit=limit)
    if json_output:
        print_json(result.model_dump(mode="json"))
        return
    print_table(
        [
            {
                "dataset": name,
                "queued": result.queued,
                "missing": result.missing,
                "remaining": result.remaining,
            }
        ],
        title="Embedding backfill",
    )


@app.command("storage", help="Show a dataset's cold-storage state (+ restore quote).")
def dataset_storage(
    name: Annotated[str, typer.Argument(help="Dataset name.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit raw JSON.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    status = client.datasets.storage_status(name)
    if json_output:
        print_json(status.model_dump(mode="json", exclude_none=True))
        return
    rows = [
        {
            "dataset": name,
            "storage_class": status.storage_class,
            "state": status.storage_state,
            "cold_images": status.cold_image_count,
            "cold_bytes": status.cold_bytes,
            "restore_usd": (
                f"${status.restore_estimate.total_micro_usd / 1_000_000:.4f}"
                if status.restore_estimate
                else "-"
            ),
        }
    ]
    print_table(rows, title="Dataset storage")


@app.command(
    "freeze", help="Move a dataset to cold storage (free; images count half toward quota)."
)
def freeze_dataset(
    name: Annotated[str, typer.Argument(help="Dataset name.")],
    wait: Annotated[bool, typer.Option("--wait/--no-wait", help="Block until done.")] = True,
    json_output: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    job = client.datasets.freeze(name)
    if not json_output:
        typer.echo(f"Freeze started (job {job.job_id})")
    status = client.datasets.wait_for_storage(name) if wait else None
    if json_output:
        print_json(
            {
                "dataset": name,
                "job": job.model_dump(mode="json"),
                "storage": status.model_dump(mode="json") if status is not None else None,
            }
        )
        return
    if status is not None:
        typer.echo(
            f"Done: {status.cold_image_count} images / {status.cold_bytes} bytes "
            f"now in {status.storage_class}"
        )


@app.command(
    "restore", help="Restore a cold dataset to standard storage (charges compute credits)."
)
def restore_dataset(
    name: Annotated[str, typer.Argument(help="Dataset name.")],
    wait: Annotated[bool, typer.Option("--wait/--no-wait", help="Block until done.")] = True,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the price confirmation.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    status = client.datasets.storage_status(name)
    est = status.restore_estimate
    if est is not None and not yes:
        typer.confirm(
            f"Restoring '{name}' costs ${est.total_micro_usd / 1_000_000:.4f} "
            f"({est.cold_image_count} images). Continue?",
            abort=True,
        )
    job = client.datasets.restore(name)
    if not json_output:
        quoted = (job.quoted_micro_usd or 0) / 1_000_000
        typer.echo(f"Restore started (job {job.job_id}, ${quoted:.4f} charged on success)")
    final = client.datasets.wait_for_storage(name) if wait else None
    if json_output:
        print_json(
            {
                "dataset": name,
                "job": job.model_dump(mode="json"),
                "storage": final.model_dump(mode="json") if final is not None else None,
            }
        )
        return
    if final is not None:
        typer.echo("Done: dataset is back on standard storage")
