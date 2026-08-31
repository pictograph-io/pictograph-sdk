"""``pictograph connectors {validate,check-limits,import,status,cancel}``.

Import datasets from V7 (Darwin) or Roboflow into Pictograph. Three-stage flow::

    pictograph connectors validate v7 --key <v7-token>        # list importable datasets
    pictograph connectors check-limits --images 5000 --bytes 2000000000
    pictograph connectors import v7 --key <v7-token> --dataset <id> --dataset <id>

``import`` waits for the job to reach a terminal state by default; pass
``--no-wait`` to fire-and-forget and poll later with ``status``.

NOTE on the two distinct keys:

* ``--key`` is the **source provider's** API key (a V7 token / Roboflow key).
  It is sent only on the call that needs it and is never persisted.
* ``--api-key`` is your **Pictograph** auth key (``pk_live_...``). It resolves
  the same way as every other command: flag > ``PICTOGRAPH_API_KEY`` env >
  ``~/.pictograph/config.toml``.
"""

from __future__ import annotations

import json as json_module
from typing import Annotated, Any

import typer

from pictograph.cli._client import get_client
from pictograph.cli._format import print_json, print_table

app = typer.Typer(no_args_is_help=True)


@app.command("validate", help="Verify a source provider's key and list importable datasets.")
def validate(
    provider: Annotated[str, typer.Argument(help="Source provider: v7 / roboflow.")],
    key: Annotated[
        str,
        typer.Option(
            "--key",
            help=(
                "The SOURCE provider's API key (V7 token / Roboflow key). Prefer the "
                "PICTOGRAPH_SOURCE_KEY environment variable: a key passed on the "
                "command line lands in shell history and is readable by any user "
                "via `ps`."
            ),
            envvar="PICTOGRAPH_SOURCE_KEY",
        ),
    ],
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key", help="Your Pictograph auth key (pk_live_...) - not the source key."
        ),
    ] = None,
) -> None:
    client = get_client(api_key)
    result = client.connectors.validate(provider, key)  # type: ignore[arg-type]
    if json_output or not result.valid:
        print_json(result.model_dump(mode="json", exclude_none=True))
        return
    rows = [
        {
            "id": d.id,
            "name": d.name,
            "slug": d.slug,
            "images": d.image_count,
            "version": d.version,
        }
        for d in result.datasets
    ]
    print_table(
        rows,
        title=f"{provider} datasets - workspace {result.workspace!r} ({len(rows)})",
    )


@app.command("check-limits", help="Preflight whether an import fits under your org's tier caps.")
def check_limits(
    images: Annotated[int, typer.Option("--images", help="Total images about to be imported.")],
    size_bytes: Annotated[
        int, typer.Option("--bytes", help="Estimated total size of the import, in bytes.")
    ],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    result = client.connectors.check_limits(total_images=images, estimated_size_bytes=size_bytes)
    print_json(result.model_dump(mode="json", exclude_none=True))


@app.command("import", help="Import datasets from the source provider (waits unless --no-wait).")
def import_datasets(
    provider: Annotated[str, typer.Argument(help="Source provider: v7 / roboflow.")],
    key: Annotated[
        str,
        typer.Option(
            "--key",
            help=(
                "The SOURCE provider's API key (V7 token / Roboflow key). Prefer the "
                "PICTOGRAPH_SOURCE_KEY environment variable: a key passed on the "
                "command line lands in shell history and is readable by any user "
                "via `ps`."
            ),
            envvar="PICTOGRAPH_SOURCE_KEY",
        ),
    ],
    dataset: Annotated[
        list[str] | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Dataset id to import (repeat for several). Resolved against `validate`.",
        ),
    ] = None,
    dataset_json: Annotated[
        str | None,
        typer.Option(
            "--dataset-json",
            help='Raw JSON dataset spec list, e.g. \'[{"id":"x","name":"n","slug":"s"}]\'. '
            "Mutually exclusive with --dataset.",
        ),
    ] = None,
    wait: Annotated[
        bool,
        typer.Option("--wait/--no-wait", help="Poll until the import finishes (default: wait)."),
    ] = True,
    poll_interval: Annotated[
        float, typer.Option("--poll-interval", help="Seconds between status polls.")
    ] = 3.0,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Max seconds to wait (default 1h).")
    ] = 3600.0,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key", help="Your Pictograph auth key (pk_live_...) - not the source key."
        ),
    ] = None,
) -> None:
    if dataset and dataset_json:
        raise typer.BadParameter("pass either --dataset or --dataset-json, not both")
    client = get_client(api_key)

    datasets: list[Any]
    if dataset_json:
        parsed = json_module.loads(dataset_json)
        if not isinstance(parsed, list):
            raise typer.BadParameter(
                "--dataset-json must be a JSON array", param_hint="--dataset-json"
            )
        datasets = parsed
    elif dataset:
        # Resolve the supplied ids against the provider's listing so we send the
        # full dataset spec (name/slug/image_count) the importer expects.
        result = client.connectors.validate(provider, key)  # type: ignore[arg-type]
        if not result.valid:
            raise typer.BadParameter(
                f"source key rejected by {provider}: {result.error or 'invalid key'}",
                param_hint="--key",
            )
        by_id = {d.id: d for d in result.datasets}
        missing = [d for d in dataset if d not in by_id]
        if missing:
            available = ", ".join(sorted(by_id)) or "<none>"
            raise typer.BadParameter(
                f"dataset id(s) not found in {provider}: {', '.join(missing)} "
                f"(available: {available})",
                param_hint="--dataset",
            )
        datasets = [by_id[d] for d in dataset]
    else:
        raise typer.BadParameter("specify at least one --dataset or --dataset-json")

    job = client.connectors.import_(
        provider,  # type: ignore[arg-type]
        key,
        datasets,
        wait=wait,
        poll_interval=poll_interval,
        timeout=timeout,
    )
    print_json(job.model_dump(mode="json", exclude_none=True))


@app.command("status", help="Fetch the current state of an import job.")
def status(
    import_id: Annotated[str, typer.Argument(help="Import job id.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    job = client.connectors.get_import(import_id)
    print_json(job.model_dump(mode="json", exclude_none=True))


@app.command("cancel", help="Soft-cancel an in-flight import (already-imported images are kept).")
def cancel(
    import_id: Annotated[str, typer.Argument(help="Import job id.")],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(f"Cancel import {import_id!r}?"):
        raise typer.Abort()
    client = get_client(api_key)
    job = client.connectors.cancel_import(import_id)
    print_json(job.model_dump(mode="json", exclude_none=True))
