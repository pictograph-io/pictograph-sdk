"""``pictograph train {start,list,status,wait,cancel,logs}``."""

from __future__ import annotations

import json as json_module
from datetime import datetime, timezone
from typing import Annotated

import typer

from pictograph.cli._client import get_client
from pictograph.cli._format import print_json, print_table


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


app = typer.Typer(no_args_is_help=True)


@app.command("start", help="Kick off a training run on a completed EXPORT.")
def start_training(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    export: Annotated[
        str,
        typer.Argument(help="Name of the COMPLETED export to train on."),
    ],
    pipeline: Annotated[
        str,
        typer.Option(
            "--pipeline",
            "-p",
            help="yolox / sm_pytorch / classification / rfdetr_detection / rfdetr_segmentation / rfdetr_keypoint",
        ),
    ],
    gpu: Annotated[str, typer.Option("--gpu", help="a10g / a100 / h100")] = "a10g",
    name: Annotated[str | None, typer.Option("--name")] = None,
    config: Annotated[
        str | None,
        typer.Option(
            "--config",
            help="JSON dict of pipeline-specific hyperparameters.",
        ),
    ] = None,
    no_wait: Annotated[
        bool, typer.Option("--no-wait", help="Fire-and-forget; print run_id and exit.")
    ] = False,
    timeout: Annotated[float, typer.Option("--timeout")] = 7200.0,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    config_dict = json_module.loads(config) if config else None
    client = get_client(api_key)
    # Training runs off an EXPORT, never off a dataset (owner, 2026-07-31).
    # The dataset-driven entry point created an export behind the caller's back
    # and could hand training an EMPTY one - which failed on the GPU, after the
    # deposit, saying "has no image_ids".
    run = client.training.create(
        dataset_name=dataset,
        export_name=export,
        pipeline_type=pipeline,  # type: ignore[arg-type]
        gpu_type=gpu,  # type: ignore[arg-type]
        name=name or f"{pipeline}-run-{_stamp()}",
        config=config_dict,
        wait=not no_wait,
        timeout=timeout,
    )
    model = None
    if run.model_id:
        model = client.models.get(model_id=run.model_id)
    print_json(
        {
            "run_id": run.id,
            "status": run.status,
            "progress": run.progress,
            "model_id": model.id if model else None,
        }
    )


@app.command("list", help="List training runs in your organization.")
def list_runs(
    dataset: Annotated[
        str | None, typer.Option("--dataset", "-d", help="Filter by dataset name.")
    ] = None,
    status_filter: Annotated[
        str | None,
        typer.Option(
            "--status",
            "-s",
            help="Filter by status: pending / running / completed / failed / cancelled.",
        ),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 50,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    runs = client.training.list(
        dataset_name=dataset,
        status=status_filter,  # type: ignore[arg-type]
        limit=limit,
    )
    if json_output:
        print_json([r.model_dump(mode="json", exclude_none=True) for r in runs])
        return
    rows = [
        {
            "name": r.name,
            "id": r.id,
            "pipeline": r.pipeline_type,
            "status": r.status,
            "progress": f"{r.progress}%",
            "model_id": r.model_id,
        }
        for r in runs
    ]
    print_table(rows, title=f"Training runs ({len(rows)})")


@app.command("status", help="Fetch a training run's status.")
def status(
    run_id: Annotated[str, typer.Argument(help="Training run UUID.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    run = client.training.get(run_id=run_id)
    print_json(run.model_dump(mode="json", exclude_none=True))


@app.command("wait", help="Poll a training run until it reaches a terminal status.")
def wait(
    run_id: Annotated[str, typer.Argument(help="Training run UUID.")],
    poll_interval: Annotated[
        float, typer.Option("--poll-interval", help="Seconds between status checks.")
    ] = 5.0,
    timeout: Annotated[float, typer.Option("--timeout", help="Max seconds to wait.")] = 7200.0,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    run = client.training.wait_for_completion(run_id, poll_interval=poll_interval, timeout=timeout)
    print_json(run.model_dump(mode="json", exclude_none=True))


@app.command("cancel", help="Cancel an in-flight training run.")
def cancel(
    run_id: Annotated[str, typer.Argument(help="Training run UUID.")],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(f"Cancel training run {run_id!r}?"):
        raise typer.Abort()
    client = get_client(api_key)
    run = client.training.cancel(run_id=run_id)
    print_json({"run_id": run.id, "status": run.status})


@app.command("cancel-batch", help="Cancel many training runs in one server-side call.")
def cancel_batch(
    run_ids: Annotated[list[str], typer.Argument(help="Training run UUIDs to cancel.")],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(f"Cancel {len(run_ids)} training run(s)?"):
        raise typer.Abort()
    client = get_client(api_key)
    # One server-side bulk call - never N fan-out requests.
    print_json(client.training.bulk_cancel(run_ids).model_dump(mode="json"))


@app.command("logs", help="Tail training logs (placeholder - backend SSE pending).")
def logs(
    run_id: Annotated[str, typer.Argument(help="Training run UUID.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    """Streaming logs require the backend SSE endpoint shipping in Phase 9."""
    client = get_client(api_key)
    run = client.training.get(run_id=run_id)
    typer.echo(
        f"Run {run_id}: status={run.status}, progress={run.progress}%, "
        f"epoch={run.current_epoch}/{run.total_epochs}"
    )
    if run.error_message:
        typer.echo(f"error: {run.error_message}", err=True)
    typer.echo(
        "Live log streaming arrives with the SSE endpoint in v1.1 - "
        "use `pictograph train status` to poll for now.",
        err=True,
    )
