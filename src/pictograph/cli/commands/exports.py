"""``pictograph exports {list,get,create,download,delete}``.

Generate, fetch, and download dataset exports from the command line::

    pictograph exports create my-dataset --format coco --name nightly
    pictograph exports list --dataset my-dataset
    pictograph exports download my-dataset nightly -o nightly.zip

``create`` is asynchronous on the backend; by default it polls until the
export finishes (status ``completed``) and prints the populated row. Pass
``--no-wait`` to fire-and-forget and poll later via ``exports get``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pictograph.cli._client import get_client
from pictograph.cli._format import print_json, print_table

app = typer.Typer(no_args_is_help=True)


@app.command("list", help="List dataset exports in your organization.")
def list_exports(
    dataset: Annotated[
        str | None, typer.Option("--dataset", help="Filter to one dataset by name.")
    ] = None,
    status: Annotated[
        str | None, typer.Option("--status", help="pending/processing/completed/failed.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 100,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    exports = client.exports.list(dataset_name=dataset, status=status, limit=limit)
    if json_output:
        print_json([e.model_dump(mode="json", exclude_none=True) for e in exports])
        return
    rows = [
        {
            "name": e.name,
            "dataset": e.dataset_name,
            "format": e.format,
            "status": e.status,
            "images": e.image_count,
            "size": e.file_size,
        }
        for e in exports
    ]
    print_table(rows, title=f"Exports ({len(rows)})")


@app.command("get", help="Fetch a single export by (dataset, export name).")
def get_export(
    dataset_name: Annotated[str, typer.Argument(help="Dataset the export belongs to.")],
    export_name: Annotated[str, typer.Argument(help="Export name.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    export = client.exports.get(dataset_name, export_name)
    print_json(export.model_dump(mode="json", exclude_none=True))


@app.command("get-by-id", help="Fetch a single export by its UUID.")
def get_export_by_id(
    export_id: Annotated[str, typer.Argument(help="Export UUID.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    export = client.exports.get_by_id(export_id)
    print_json(export.model_dump(mode="json", exclude_none=True))


@app.command("create", help="Create an export; waits for completion unless --no-wait.")
def create_export(
    dataset_name: Annotated[str, typer.Argument(help="Dataset to export.")],
    name: Annotated[str, typer.Option("--name", help="Export name (≤100 chars).")],
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="pictograph / darwin / coco / yolo / pascal_voc / cvat / datumaro / labelme / csv.",
        ),
    ] = "pictograph",
    include_images: Annotated[
        bool,
        typer.Option("--include-images", help="Bundle the original image bytes (larger ZIP)."),
    ] = False,
    class_filter: Annotated[
        str | None,
        typer.Option("--class-filter", help="Comma-separated class names to keep (default: all)."),
    ] = None,
    status_filter: Annotated[
        str | None,
        typer.Option(
            "--status-filter", help="Restrict to images with this status (e.g. complete)."
        ),
    ] = None,
    organize_by_split: Annotated[
        bool,
        typer.Option(
            "--organize-by-split",
            help="Organize the ZIP into train/valid/test directories by each image's split "
            "(unassigned → train). Yields a directly-trainable YOLO/COCO layout.",
        ),
    ] = False,
    wait: Annotated[
        bool, typer.Option("--wait/--no-wait", help="Poll until the export completes.")
    ] = True,
    poll_interval: Annotated[
        float, typer.Option("--poll-interval", help="Seconds between polls.")
    ] = 2.0,
    timeout: Annotated[float, typer.Option("--timeout", help="Max seconds to wait.")] = 300.0,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    classes = [c.strip() for c in class_filter.split(",") if c.strip()] if class_filter else None
    export = client.exports.create(
        dataset_name,
        name,
        format=format,  # type: ignore[arg-type]
        include_images=include_images,
        class_filter=classes,
        status_filter=status_filter,
        organize_by_split=organize_by_split,
        wait=wait,
        poll_interval=poll_interval,
        timeout=timeout,
    )
    print_json(export.model_dump(mode="json", exclude_none=True))


@app.command("download", help="Download an export ZIP to disk.")
def download_export(
    dataset_name: Annotated[str, typer.Argument(help="Dataset the export belongs to.")],
    export_name: Annotated[str, typer.Argument(help="Export name.")],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Local destination path for the ZIP.")
    ],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    path = client.exports.download(dataset_name, export_name, output)
    print_json({"path": str(path), "bytes": path.stat().st_size})


@app.command("download-by-id", help="Download an export ZIP to disk by its UUID.")
def download_export_by_id(
    export_id: Annotated[str, typer.Argument(help="Export UUID.")],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Local destination path for the ZIP.")
    ],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    path = client.exports.download_by_id(export_id, output)
    print_json({"path": str(path), "bytes": path.stat().st_size})


@app.command("delete", help="Delete an export (DB row + stored file). Requires admin/owner.")
def delete_export(
    dataset_name: Annotated[str, typer.Argument(help="Dataset the export belongs to.")],
    export_name: Annotated[str, typer.Argument(help="Export name.")],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(f"Delete export {export_name!r} from {dataset_name!r}?"):
        raise typer.Abort()
    client = get_client(api_key)
    client.exports.delete(dataset_name, export_name)
    print_json({"dataset_name": dataset_name, "export_name": export_name, "deleted": True})


@app.command(
    "bulk-delete",
    help="Delete many exports by UUID in one server-side call. Requires admin/owner.",
)
def bulk_delete_exports(
    export_ids: Annotated[list[str], typer.Argument(help="One or more export UUIDs.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(f"Delete {len(export_ids)} export(s)? This cannot be undone."):
        raise typer.Abort()
    client = get_client(api_key)
    result = client.exports.bulk_delete(export_ids)
    print_json(result.model_dump(mode="json"))
