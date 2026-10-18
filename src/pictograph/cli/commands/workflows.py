"""``pictograph workflows {list,get,create,update,run,run-status,compute-options,cancel,delete}``.

Drive node-graph Workflows headlessly. Build the graph in the app, then::

    pictograph workflows list
    pictograph workflows get <workflow-id> > graph.json   # export the current graph
    pictograph workflows update <workflow-id> --graph graph.json   # edit the graph
    pictograph workflows run <workflow-id>            # runs + waits, prints artifacts
    pictograph workflows run <workflow-id> --no-wait  # fire-and-forget, prints run id
    pictograph workflows run-status <run-id>          # poll a run + its artifacts
    pictograph workflows run <workflow-id> --gpu l4   # this run's GPU
    pictograph workflows compute-options              # the GPUs (memory, price), a model's precisions

A Model block's precision is its own, in the graph: its config's ``"precision": "fp16"`` runs the
model at half precision (``--graph`` to set it here, or the app's Model block).
"""

from __future__ import annotations

import json as json_module
from pathlib import Path
from typing import Annotated, Any

import typer

from pictograph._image_input import is_url
from pictograph.cli._client import get_client
from pictograph.cli._format import print_json, print_table
from pictograph.cli._inputs import STDIN, read_stdin
from pictograph.models.workflow import RunLogStream, WorkflowInvokeKind, WorkflowInvokeSource

app = typer.Typer(no_args_is_help=True)

# This run's (or this workflow's) compute: the runner's GPU.
GpuOption = Annotated[
    str | None,
    typer.Option("--gpu", help="The runner's GPU: t4 (default) or l4. See compute-options."),
]


def _vram(gb: int | None) -> str:
    return f"{gb} GB" if gb else ""


def _compute(gpu: str | None) -> dict[str, str] | None:
    return {"gpu": gpu} if gpu is not None else None


@app.command(
    "compute-options",
    help="The GPUs a workflow runs on (memory, price) and the precisions a Model block may take.",
)
def compute_options(
    as_json: Annotated[bool, typer.Option("--json", help="Print the raw JSON.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    options = get_client(api_key).workflows.compute_options()
    if as_json:
        print_json(options.model_dump(mode="json"))
        return
    print_table(
        [
            {
                "gpu": g.gpu,
                "label": g.label,
                "vram": _vram(g.vram_gb),
                "per minute": f"${g.rate_per_min_micro_usd / 1_000_000:.4f}",
                "default": "yes" if g.gpu == options.default.gpu else "",
            }
            for g in options.gpus
        ],
        title="GPUs",
    )
    print_table(
        [
            {
                "precision": p.precision,
                "label": p.label,
                "default": "yes" if p.precision == "fp32" else "",
                "description": p.description,
            }
            for p in options.precisions
        ],
        title="A Model block's precision (its config's precision)",
    )


@app.command("list", help="List node-graph workflows in your organization.")
def list_workflows(
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    workflows = client.workflows.list()
    if json_output:
        print_json([w.model_dump(mode="json", exclude_none=True) for w in workflows])
        return
    rows = [
        {
            "name": w.name,
            "id": w.id,
            "status": w.status,
            "blocks": len(w.graph.get("nodes", [])) if isinstance(w.graph, dict) else 0,
            "last_run_id": w.last_run_id,
        }
        for w in workflows
    ]
    print_table(rows, title=f"Workflows ({len(rows)})")


@app.command("get", help="Fetch a single workflow (graph + validation).")
def get_workflow(
    workflow: Annotated[str, typer.Argument(help="Workflow name (a UUID also works).")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    found = client.workflows.get(workflow)
    print_json(found.model_dump(mode="json", exclude_none=True))


@app.command("create", help="Create a workflow from a graph JSON file.")
def create_workflow(
    name: Annotated[str, typer.Argument(help="Workflow name.")],
    graph: Annotated[
        Path,
        typer.Option("--graph", "-g", help="Path to a graph JSON file ({version,nodes,edges})."),
    ],
    description: Annotated[str | None, typer.Option("--description")] = None,
    template_key: Annotated[
        str | None,
        typer.Option("--template", help="line_counter / dwell_time / custom."),
    ] = None,
    gpu: GpuOption = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not graph.is_file():
        raise typer.BadParameter(f"graph file not found: {graph}", param_hint="--graph")
    graph_dict = json_module.loads(graph.read_text(encoding="utf-8"))
    client = get_client(api_key)
    workflow = client.workflows.create(
        name,
        graph_dict,
        description=description,
        template_key=template_key,
        compute=_compute(gpu),
    )
    print_json(
        {
            "id": workflow.id,
            "name": workflow.name,
            "status": workflow.status,
            "compute": workflow.compute.model_dump(),
        }
    )


@app.command("update", help="Update a workflow's graph / name / readme / status.")
def update_workflow(
    workflow: Annotated[str, typer.Argument(help="Workflow id or slug.")],
    graph: Annotated[
        Path | None,
        typer.Option(
            "--graph",
            "-g",
            help="Path to a NEW graph JSON file ({version,nodes,edges}); replaces the graph.",
        ),
    ] = None,
    name: Annotated[str | None, typer.Option("--name")] = None,
    readme: Annotated[
        str | None, typer.Option("--readme", help="Markdown card shown in the app.")
    ] = None,
    description: Annotated[str | None, typer.Option("--description")] = None,
    status: Annotated[
        str | None, typer.Option("--status", help="draft / ready / archived.")
    ] = None,
    gpu: GpuOption = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    graph_dict = None
    if graph is not None:
        if not graph.is_file():
            raise typer.BadParameter(f"graph file not found: {graph}", param_hint="--graph")
        graph_dict = json_module.loads(graph.read_text(encoding="utf-8"))
    if (
        graph_dict is None
        and name is None
        and readme is None
        and description is None
        and status is None
        and gpu is None
    ):
        raise typer.BadParameter(
            "nothing to update: pass at least one of "
            "--graph/--name/--readme/--description/--status/--gpu"
        )
    client = get_client(api_key)
    wf = client.workflows.update(
        workflow,
        name=name,
        readme=readme,
        description=description,
        graph=graph_dict,
        status=status,  # type: ignore[arg-type]
        compute=_compute(gpu),
    )
    print_json(
        {"id": wf.id, "name": wf.name, "status": wf.status, "compute": wf.compute.model_dump()}
    )


@app.command("run", help="Validate + start a run; waits for completion unless --no-wait.")
def run_workflow(
    workflow: Annotated[str, typer.Argument(help="Workflow name (a UUID also works).")],
    no_wait: Annotated[
        bool,
        typer.Option("--no-wait", help="Fire-and-forget; print run_id + cost estimate and exit."),
    ] = False,
    poll_interval: Annotated[
        float, typer.Option("--poll-interval", help="Seconds between polls.")
    ] = 5.0,
    timeout: Annotated[float, typer.Option("--timeout", help="Max seconds to wait.")] = 3600.0,
    gpu: GpuOption = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    created = client.workflows.run(workflow, gpu=gpu)
    if no_wait:
        print_json({"run_id": created.run_id, "deposit_micro_usd": created.deposit_micro_usd})
        return
    run = client.workflows.wait_for_run(
        created.run_id, poll_interval=poll_interval, timeout=timeout
    )
    print_json(run.model_dump(mode="json", exclude_none=True))


@app.command(
    "invoke",
    help="Run a saved workflow on-demand over a source given at call time; waits unless --no-wait.",
)
def invoke_workflow(
    workflow: Annotated[str, typer.Argument(help="Workflow name (a UUID also works).")],
    file: Annotated[
        str | None,
        typer.Option(
            "--file",
            help="A video or image to run over: a file on disk (uploaded), an http(s) "
            "link Pictograph downloads, or - to read it from stdin.",
        ),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", help="Run over this dataset (name or uuid) in your org."),
    ] = None,
    directory: Annotated[
        str | None, typer.Option("--directory", help="Restrict --dataset to one folder path.")
    ] = None,
    video: Annotated[
        str | None,
        typer.Option(
            "--video",
            help="Run over a video: the storage URI an upload returned (under your org) "
            "or an http(s) link Pictograph downloads.",
        ),
    ] = None,
    image: Annotated[
        str | None,
        typer.Option(
            "--image",
            help="Run over an image: the storage URI an upload returned (under your org) "
            "or an http(s) link Pictograph downloads.",
        ),
    ] = None,
    sample_fps: Annotated[
        float | None, typer.Option("--sample-fps", help="Frames/sec to sample (with --video).")
    ] = None,
    duration: Annotated[
        float | None,
        typer.Option(
            "--duration", help="Video length in seconds, for the pre-run estimate (with --video)."
        ),
    ] = None,
    no_wait: Annotated[
        bool, typer.Option("--no-wait", help="Fire-and-forget; print run_id + status_url and exit.")
    ] = False,
    poll_interval: Annotated[
        float, typer.Option("--poll-interval", help="Seconds between polls.")
    ] = 5.0,
    timeout: Annotated[float, typer.Option("--timeout", help="Max seconds to wait.")] = 3600.0,
    gpu: GpuOption = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if sum(bool(x) for x in (file, dataset, video, image)) != 1:
        raise typer.BadParameter("Provide exactly one of --file / --dataset / --video / --image.")
    source: WorkflowInvokeSource | Path | str | bytes
    if file:
        if file == STDIN:
            # invoke() reads whether the bytes are a video or an image, then uploads them.
            source = read_stdin(
                "--file",
                example=f"`cat clip.mp4 | pictograph workflows invoke {workflow!r} --file -`",
            )
        elif is_url(file):
            source = file  # invoke() hands the link to Pictograph, which downloads it
        else:
            source = Path(file).expanduser()  # invoke() uploads the file and runs over it
    elif dataset:
        source = WorkflowInvokeSource(kind="dataset", dataset=dataset, directory_path=directory)
    elif video:
        source = _stored_or_linked("video", video, sample_fps=sample_fps, duration=duration)
    else:
        source = _stored_or_linked("image", image or "", sample_fps=None, duration=None)
    client = get_client(api_key)
    try:
        result = client.workflows.invoke(
            workflow,
            source=source,
            wait=not no_wait,
            poll_interval=poll_interval,
            timeout=timeout,
            gpu=gpu,
        )
    except ValueError as exc:  # a link or bytes whose kind can't be read, before any request
        raise typer.BadParameter(str(exc), param_hint="--file") from None
    print_json(result.model_dump(mode="json", exclude_none=True))


def _stored_or_linked(
    kind: WorkflowInvokeKind, value: str, *, sample_fps: float | None, duration: float | None
) -> WorkflowInvokeSource:
    """``--video`` / ``--image``: a storage URI under your org, or a link Pictograph downloads."""
    if is_url(value):
        return WorkflowInvokeSource(
            kind=kind, url=value, sample_fps=sample_fps, duration_seconds=duration
        )
    if value.startswith("gs://"):
        return WorkflowInvokeSource(
            kind=kind, gcs_uri=value, sample_fps=sample_fps, duration_seconds=duration
        )
    raise typer.BadParameter(
        f"{value!r} is neither the storage URI an upload returned nor an http(s) link. For a "
        "file on disk use --file.",
        param_hint=f"--{kind}",
    )


@app.command("run-status", help="Poll a run's status + artifacts (signed URLs).")
def run_status(
    run_id: Annotated[str, typer.Argument(help="Workflow run UUID.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    run = client.workflows.get_run(run_id)
    print_json(run.model_dump(mode="json", exclude_none=True))


@app.command("runs", help="List a workflow's runs, newest first (paginated).")
def list_runs(
    workflow: Annotated[str, typer.Argument(help="Workflow name (a UUID also works).")],
    limit: Annotated[int, typer.Option("--limit", min=1, max=100, help="Page size (1-100).")] = 50,
    offset: Annotated[int, typer.Option("--offset", min=0, help="Rows to skip.")] = 0,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    runs = client.workflows.list_runs(workflow, limit=limit, offset=offset)
    if json_output:
        print_json([r.model_dump(mode="json", exclude_none=True) for r in runs])
        return
    rows = [
        {
            "id": r.id,
            "status": r.status,
            "progress": f"{r.progress:.0f}%",
            "frames": f"{r.frames_done}/{r.frames_total if r.frames_total is not None else '?'}",
            "cost_uUSD": r.cost_micro_usd,
            "artifacts": r.artifact_count,
            "engine": r.engine,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in runs
    ]
    print_table(rows, title=f"Runs of {workflow} ({len(rows)})")


@app.command("log", help="Read a run's log: its events, frames or stats as the engine logged them.")
def run_log(
    run_id: Annotated[str, typer.Argument(help="Workflow run UUID.")],
    stream: Annotated[
        str, typer.Option("--stream", help="events (default), frames or stats.")
    ] = "events",
    limit: Annotated[
        int, typer.Option("--limit", min=1, help="Records to read (pages of up to 200).")
    ] = 50,
    newest_first: Annotated[
        bool, typer.Option("--newest-first", help="Newest record first.")
    ] = False,
    full: Annotated[
        bool, typer.Option("--full", help="Each frame's Pictograph record / each event's message.")
    ] = False,
    class_name: Annotated[
        str | None, typer.Option("--class", help="Only records with this class.")
    ] = None,
    track_id: Annotated[
        str | None, typer.Option("--track", help="Only records with this track.")
    ] = None,
    min_confidence: Annotated[
        float | None,
        typer.Option("--min-confidence", min=0.0, max=1.0, help="Frames: best confidence."),
    ] = None,
    source: Annotated[
        str | None, typer.Option("--source", help="Only this source's records.")
    ] = None,
    event_type: Annotated[
        str | None, typer.Option("--type", help="Events: event, capture or capture_uploaded.")
    ] = None,
    trigger: Annotated[
        str | None, typer.Option("--trigger", help="Events: only this trigger's.")
    ] = None,
    summary: Annotated[
        bool, typer.Option("--summary", help="Each stream's count and time span instead.")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    streams: dict[str, RunLogStream] = {"events": "events", "frames": "frames", "stats": "stats"}
    if stream not in streams:
        raise typer.BadParameter("--stream must be events, frames or stats")
    client = get_client(api_key)
    if summary:
        s = client.workflows.run_log_summary(run_id)
        if json_output:
            print_json(s.model_dump(mode="json"))
            return
        spans: list[dict[str, Any]] = [
            {
                "stream": name,
                "records": st.count,
                "first": st.first_ts.isoformat() if st.first_ts else None,
                "last": st.last_ts.isoformat() if st.last_ts else None,
            }
            for name, st in s.streams.items()
        ]
        note = "" if s.log_frames else " - frames are logged on paid plans"
        print_table(spans, title=f"Log of run {run_id}{note}")
        return
    records = list(
        client.workflows.iter_run_log(
            run_id,
            stream=streams[stream],
            page_size=min(limit, 200),
            order="desc" if newest_first else "asc",
            full=full,
            max_total=limit,
            class_name=class_name,
            track_id=track_id,
            min_confidence=min_confidence,
            source=source,
            event_type=event_type,
            trigger=trigger,
        )
    )
    if json_output:
        print_json([r.model_dump(mode="json", exclude_none=True) for r in records])
        return
    rows: list[dict[str, Any]]
    if stream == "frames":
        rows = [
            {
                "time": r.ts.isoformat(),
                "source": r.source,
                "frame": r.frame,
                "annotations": r.annotation_count,
                "classes": ", ".join(r.classes),
                "best": r.max_confidence,
                "latency_ms": r.latency_total_ms,
            }
            for r in records
        ]
    elif stream == "events":
        rows = [
            {
                "time": r.ts.isoformat(),
                "type": r.type,
                "trigger": r.trigger,
                "source": r.source,
                "frame": r.frame,
                "values": r.rule_values,
                "capture": r.capture_name or r.capture_kind,
            }
            for r in records
        ]
    else:
        rows = [{"time": r.ts.isoformat(), "type": r.type, "sources": r.sources} for r in records]
    print_table(rows, title=f"{stream.capitalize()} of run {run_id} ({len(rows)})")


@app.command("cancel", help="Cancel an in-flight run (free - workflows bill on success only).")
def cancel_run(
    run_id: Annotated[str, typer.Argument(help="Workflow run UUID.")],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(f"Cancel workflow run {run_id!r}?"):
        raise typer.Abort()
    client = get_client(api_key)
    client.workflows.cancel_run(run_id)
    print_json({"run_id": run_id, "status": "cancelled"})


@app.command("cancel-batch", help="Cancel many in-flight workflow runs in one server-side call.")
def cancel_runs_batch(
    run_ids: Annotated[list[str], typer.Argument(help="Workflow run UUIDs to cancel.")],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(f"Cancel {len(run_ids)} workflow run(s)?"):
        raise typer.Abort()
    client = get_client(api_key)
    # One server-side bulk call - never N fan-out requests.
    print_json(client.workflows.bulk_cancel_runs(run_ids).model_dump(mode="json"))


@app.command(
    "delete",
    help="Delete one or more workflows (and their run history). "
    "Pass multiple names for a single server-side bulk delete.",
)
def delete_workflow(
    workflows: Annotated[
        list[str], typer.Argument(help="One or more workflow names (UUIDs also work).")
    ],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    n = len(workflows)
    prompt = (
        f"Delete workflow {workflows[0]!r} and its run history?"
        if n == 1
        else f"Delete {n} workflows and their run history?"
    )
    if not yes and not typer.confirm(prompt):
        raise typer.Abort()
    client = get_client(api_key)
    if n == 1:
        client.workflows.delete(workflows[0])
        print_json({"workflow": workflows[0], "deleted": True})
        return
    result = client.workflows.bulk_delete(workflows)
    print_json(result.model_dump(mode="json"))
