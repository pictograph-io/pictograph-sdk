"""``pictograph jobs {list,get,wait,cancel}`` - one view of every job (B533 #14).

Training runs, batch auto-annotation, workflow runs, model evaluations, exports, imports,
video extraction and bulk image operations each have their own status command. ``jobs``
answers "what am I waiting on?" across all of them: one status vocabulary (queued,
running, completed, failed, cancelled) and one way to wait for or cancel any of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

import typer

from pictograph.cli._client import get_client
from pictograph.cli._errors import CliError, ExitCode, command_group
from pictograph.cli._format import err_console, print_json, print_table
from pictograph.exceptions import PollTimeoutError

if TYPE_CHECKING:
    from pictograph.models.job import Job

app = command_group(no_args_is_help=True)

_STATUSES = ("active", "finished", "all", "queued", "running", "completed", "failed", "cancelled")

_JobArg = Annotated[
    str,
    typer.Argument(
        help="The job's id: what the work answered when it started, or the ID column of `jobs list`."
    ),
]


def _when(job: Job) -> str | None:
    return job.created_at.isoformat(timespec="seconds") if job.created_at else None


def _row(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "type": job.type,
        "status": job.status,
        "progress": "-" if job.progress is None else f"{job.progress:.0f}%",
        "name": job.name or "-",
        "created": _when(job),
        "cancel": "yes" if job.cancellable else "-",
    }


def _emit(job: Job, json_output: bool) -> None:
    if json_output:
        print_json(job.model_dump(mode="json"))
        return
    print_table([_row(job)], title=f"{job.type} job")
    if job.error:
        err_console.print(f"[red]error:[/red] {job.error}", soft_wrap=True)


@app.command(
    "list",
    help=(
        "List your organization's jobs, newest first: training, auto-annotation, workflow "
        "runs, evaluations, exports, imports, video extraction, bulk image operations. "
        "Queued and running jobs by default; --status all for history."
    ),
)
def list_jobs(
    status: Annotated[
        str,
        typer.Option(
            "--status",
            "-s",
            help="active (default), finished, all, or one of queued, running, completed, "
            "failed, cancelled.",
        ),
    ] = "active",
    job_type: Annotated[
        str | None,
        typer.Option(
            "--type",
            "-t",
            help="Only this kind: training, auto_annotate, workflow_run, evaluation, export, "
            "or a background job type (connector_import, video_extract, batch_move, ...).",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", min=1, max=1000, help="How many to return (newest first)."),
    ] = 50,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if status not in _STATUSES:
        raise typer.BadParameter(f"must be one of: {', '.join(_STATUSES)}", param_hint="--status")
    client = get_client(api_key)
    jobs = list(
        client.jobs.list(
            status=status,  # type: ignore[arg-type]
            type=job_type,
            page_size=min(limit, 100),
            max_total=limit,
        )
    )
    if json_output:
        print_json([job.model_dump(mode="json") for job in jobs])
        return
    print_table(
        [_row(job) for job in jobs],
        title=f"Jobs ({len(jobs)}, {status})",
        no_wrap=("id",),
    )


@app.command("get", help="Show one job: its type, status, progress and error.")
def get_job(
    job_id: _JobArg,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    _emit(get_client(api_key).jobs.get(job_id), json_output)


@app.command(
    "wait",
    help=(
        "Block until a job finishes. Exits 0 when it completed, non-zero when it failed, "
        "was cancelled or is still running at --timeout (it keeps running)."
    ),
)
def wait_for_job(
    job_id: _JobArg,
    timeout: Annotated[
        float,
        typer.Option("--timeout", min=1.0, help="Seconds to wait. The job keeps running after it."),
    ] = 3600.0,
    poll_interval: Annotated[
        float, typer.Option("--poll-interval", min=0.5, help="Seconds between status checks.")
    ] = 3.0,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    err_console.print(
        f"[dim]Waiting for job {job_id} (up to {timeout:.0f}s)...[/dim]", soft_wrap=True
    )
    try:
        job = client.jobs.wait(job_id, poll_interval=poll_interval, timeout=timeout)
    except PollTimeoutError:
        # The SDK's message names the Python call; on the CLI the next step is a command.
        raise CliError(
            f"Job {job_id} did not finish within {timeout:.0f}s. It keeps running on "
            f"Pictograph: wait again with `pictograph jobs wait {job_id}`.",
            exit_code=ExitCode.STILL_RUNNING,
            code="wait_timeout",
        ) from None
    _emit(job, json_output)


@app.command(
    "cancel",
    help=(
        "Cancel a queued or running job, through the route that cancels its kind (a training "
        "run is refunded what it did not use). `jobs list` marks the jobs that can be cancelled."
    ),
)
def cancel_job(
    job_id: _JobArg,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    _emit(get_client(api_key).jobs.cancel(job_id), json_output)
