"""``pictograph train {start,quote,list,status,wait,cancel,cancel-batch,logs}``."""

from __future__ import annotations

import json as json_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import typer

from pictograph.cli._client import get_client
from pictograph.cli._errors import command_group
from pictograph.cli._format import err_console, print_json, print_table


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


_PIPELINE_HELP = (
    "yolox / sm_pytorch / classification / rfdetr_detection / rfdetr_segmentation / rfdetr_keypoint"
)
_GPU_HELP = (
    "a10g (default) / a100 / h100, or auto: the cheapest tier whose memory fits the "
    "config's predicted peak. `pictograph train quote` shows what auto picks."
)
_CONFIG_HELP = (
    "Pipeline hyperparameters: a JSON object ('{\"epochs\": 20}'), or the path to a JSON "
    "file (a config.json downloaded with `pictograph models download-file` works)."
)


def _config_arg(value: str | None) -> dict[str, Any] | Path | None:
    """``--config``: a JSON object, or a path to a JSON file the SDK reads."""
    if value is None:
        return None
    text = value.strip()
    if text.startswith("{"):
        try:
            parsed = json_module.loads(text)
        except json_module.JSONDecodeError as exc:
            raise typer.BadParameter(f"not valid JSON: {exc.msg}", param_hint="--config") from exc
        if not isinstance(parsed, dict):  # pragma: no cover - a '{' always parses to a dict
            raise typer.BadParameter("must be a JSON object", param_hint="--config")
        return parsed
    path = Path(value).expanduser()
    if not path.is_file():
        raise typer.BadParameter(
            f"neither a JSON object nor a file: {value!r}", param_hint="--config"
        )
    return path


app = command_group(no_args_is_help=True)


@app.command("start", help="Kick off a training run on a completed EXPORT.")
def start_training(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    export: Annotated[
        str,
        typer.Argument(help="Name of the COMPLETED export to train on."),
    ],
    pipeline: Annotated[str, typer.Option("--pipeline", "-p", help=_PIPELINE_HELP)],
    gpu: Annotated[str, typer.Option("--gpu", help=_GPU_HELP)] = "a10g",
    name: Annotated[str | None, typer.Option("--name")] = None,
    config: Annotated[str | None, typer.Option("--config", help=_CONFIG_HELP)] = None,
    no_wait: Annotated[
        bool, typer.Option("--no-wait", help="Fire-and-forget; print run_id and exit.")
    ] = False,
    timeout: Annotated[float, typer.Option("--timeout")] = 7200.0,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    config_dict = _config_arg(config)
    client = get_client(api_key)
    # Training runs off an EXPORT, never off a dataset (owner, 2026-07-31).
    # The dataset-driven entry point created an export behind the caller's back
    # and could hand training an EMPTY one - which failed on the GPU, after the
    # deposit, saying "has no image_ids".
    run = client.training.create(
        dataset=dataset,
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


@app.command(
    "quote",
    help=(
        "Check and price a training run without starting it: the checks `train start` "
        "runs, the GPU (and what --gpu auto picks), each class's training data, the "
        "estimate, and whether the run can start. Writes nothing, charges nothing."
    ),
)
def quote_training(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    export: Annotated[
        str,
        typer.Argument(help="Name of the COMPLETED export to train on."),
    ],
    pipeline: Annotated[str, typer.Option("--pipeline", "-p", help=_PIPELINE_HELP)],
    gpu: Annotated[str, typer.Option("--gpu", help=_GPU_HELP)] = "a10g",
    config: Annotated[str | None, typer.Option("--config", help=_CONFIG_HELP)] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    config_value = _config_arg(config)
    client = get_client(api_key)
    quote = client.training.quote(
        dataset,
        export,
        pipeline_type=pipeline,  # type: ignore[arg-type]
        gpu_type=gpu,  # type: ignore[arg-type]
        config=config_value,
    )
    print_json(quote.model_dump(mode="json"))
    if not quote.can_start:
        # The quote itself succeeded (exit 0); say on stderr why a start would not.
        reasons = "; ".join(f"{b.code}: {b.message}" for b in quote.blocking)
        err_console.print(f"[yellow]This run cannot start yet - {reasons}[/yellow]")


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
    limit: Annotated[
        int,
        typer.Option(
            "--limit", "-n", min=1, help="How many to return (pages through the API as needed)."
        ),
    ] = 50,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    # Paged: one request takes at most 100 runs, and a larger --limit used to 422.
    runs = list(
        client.training.iter(
            dataset=dataset,
            status=status_filter,  # type: ignore[arg-type]
            page_size=min(limit, 100),
            max_total=limit,
        )
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
    json_output: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    """Streaming logs require the backend SSE endpoint shipping in Phase 9."""
    client = get_client(api_key)
    run = client.training.get(run_id=run_id)
    if json_output:
        print_json(
            {
                "run_id": run_id,
                "status": run.status,
                "progress": run.progress,
                "current_epoch": run.current_epoch,
                "total_epochs": run.total_epochs,
                "error_message": run.error_message,
            }
        )
        return
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
