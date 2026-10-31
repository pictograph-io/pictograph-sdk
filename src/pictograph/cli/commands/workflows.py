"""``pictograph workflows {list,get,create,update,run,run-status,compute-options,cancel,delete,deploy,pause,resume,
scale,stream,live-log}``.

Write a graph without guessing - start from a template, read each block's card, check it::

    pictograph workflows templates                     # the use cases a workflow starts from
    pictograph workflows templates traffic -o graph.json
    pictograph workflows nodes measure                 # a block's settings and rules
    pictograph workflows validate --graph graph.json   # would a run start? (exit 6 if not)
    pictograph workflows create counter --template traffic --region us-west   # its endpoint, its token once

Or build the graph in the app, then drive it headlessly::

    pictograph workflows list
    pictograph workflows get <workflow-id> > graph.json   # export the current graph
    pictograph workflows update <workflow-id> --graph graph.json   # edit the graph
    pictograph workflows run <workflow> --file clip.mp4 --sample-fps 10   # runs a video + waits, prints artifacts
    pictograph workflows run <workflow> --dataset my-dataset --directory /night   # a dataset, or one folder of it
    pictograph workflows run <workflow> --export my-export --no-wait   # an export; prints the run id at once
    pictograph workflows run-status <run-id>          # poll a run + its artifacts
    pictograph workflows update <workflow-id> --gpu l4   # the GPU its endpoint runs on
    pictograph workflows compute-options --region us-west   # the GPUs (memory, price), a model's precisions

A Model block's precision is its own, in the graph: its config's ``"precision": "fp16"`` runs the
model at half precision (``--graph`` to set it here, or the app's Model block).

A workflow's endpoint is made with it (B579 slice 5): its runs run there - queued until it has room, billed as its
uptime - and its live streams, a camera streamed through every block a frame passes::

    pictograph workflows stream <workflow> rtsp://camera.example.com/live --token pk_deploy_...
    pictograph workflows deploy <workflow> --wait     # a workflow made before endpoints: its endpoint, its token once
    pictograph workflows pause <workflow>              # nothing held, nothing billed
    pictograph workflows live-log <workflow>           # what its streams found
"""

from __future__ import annotations

import json as json_module
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer
from pydantic import ValidationError as PydanticValidationError
from rich.markup import escape

from pictograph import DeploymentClient
from pictograph._image_input import is_url
from pictograph.cli._client import get_client
from pictograph.cli._endpoint import (
    MaxChange,
    MaxContainers,
    MinChange,
    MinContainers,
    Region,
    ScaledownChange,
    ScaledownWindow,
    build_state,
    live_status,
    run_stream,
    say_token_once,
    say_waiting,
)
from pictograph.cli._errors import CliError, ExitCode, command_group
from pictograph.cli._format import err_console, print_json, print_table
from pictograph.cli._inputs import STDIN, read_stdin
from pictograph.models.workflow import (
    RunLogStream,
    Workflow,
    WorkflowIssue,
    WorkflowRunInput,
    WorkflowRunInputKind,
    WorkflowRunOutputs,
    WorkflowRunsKind,
)

if TYPE_CHECKING:
    from pictograph import Client

app = command_group(no_args_is_help=True)

# A workflow's compute: the GPU its endpoint runs on (its runs and its streams).
GpuOption = Annotated[
    str | None,
    typer.Option(
        "--gpu", help="The GPU its endpoint runs on: t4 (default) or l4. See compute-options."
    ),
]
_WAIT_HELP = (
    "Wait until it is live (a model's first deployment on a GPU at a precision builds its TensorRT engine "
    "first, a few minutes)."
)


def _wait_live(client: Client, workflow: Workflow, timeout: float) -> Workflow:
    """Wait for a workflow's endpoint to go live, saying on stderr what it waits on."""
    ep = workflow.endpoint
    say_waiting(
        workflow.name,
        ep.engine_build if ep else None,
        building=f"its models' TensorRT engines for {ep.gpu_type if ep else 'its GPU'}",
        until="go live",
    )
    return client.workflows.wait_until_live(workflow, timeout=timeout)


def _endpoint_out(workflow: Workflow) -> dict[str, Any] | None:
    """A workflow's endpoint as the CLI prints it (its token once, when the answer carries it)."""
    ep = workflow.endpoint
    if ep is None:
        return None
    out: dict[str, Any] = {
        "status": ep.status,
        "region": ep.region,
        "gpu": ep.gpu_type,
        "endpoint_url": ep.endpoint_url,
        "stream_limit": ep.stream_limit,
        "run_limit": ep.run_limit,
        "streams": ep.streams,
    }
    if ep.runtime is not None:
        out["state"] = ep.runtime.state
    if ep.error_message:
        out["error"] = ep.error_message
    if ep.live.events:
        # What its live outputs act on while it streams, as the app names them.
        out["live_events"] = [e.name for e in ep.live.events]
    if ep.live_outputs is not None:
        lo = ep.live_outputs
        out["live_outputs"] = {
            "events": lo.events,
            "deliveries": lo.deliveries,
            "saved": lo.saved,
            "not_saved": lo.not_saved,
            "skipped": lo.skipped,
        }
        if lo.last_not_saved_reason:
            out["live_outputs"]["last_not_saved_reason"] = lo.last_not_saved_reason
    return out


def _vram(gb: int | None) -> str:
    return f"{gb} GB" if gb else ""


def _compute(gpu: str | None) -> dict[str, str] | None:
    return {"gpu": gpu} if gpu is not None else None


GraphOption = Annotated[
    str | None,
    typer.Option(
        "--graph", "-g", help="A graph JSON file ({version,nodes,edges}), or - for stdin."
    ),
]


def _read_graph(source: str, param_hint: str = "--graph") -> dict[str, Any]:
    """A graph JSON file (or ``-`` for stdin) as a dict. A missing file, or text that is not a
    JSON object, is invalid input (exit 6) before any request."""
    if source == STDIN:
        text = sys.stdin.read()
    else:
        path = Path(source)
        if not path.is_file():
            raise typer.BadParameter(f"graph file not found: {source}", param_hint=param_hint)
        text = path.read_text(encoding="utf-8")
    try:
        graph = json_module.loads(text)
    except ValueError as exc:
        raise typer.BadParameter(f"not JSON ({exc})", param_hint=param_hint) from None
    if not isinstance(graph, dict):
        raise typer.BadParameter(
            "a graph is a JSON object: {version, nodes, edges}", param_hint=param_hint
        )
    return graph


def _issue_line(issue: WorkflowIssue) -> str:
    where = f" (block {issue.node_id})" if issue.node_id else ""
    return f"[{issue.code}] {issue.message}{where}"


@app.command(
    "nodes",
    help="Every block a workflow graph can hold, or one block's card: its ports, settings "
    "(values, defaults) and rules.",
)
def nodes(
    block: Annotated[
        str | None, typer.Argument(help="One block type (source, model, measure, ...).")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    catalog = get_client(api_key).workflows.nodes()
    if block is not None:
        try:
            card = catalog.card(block)
        except KeyError:
            raise CliError(
                f"No block type {block!r}.",
                exit_code=ExitCode.NOT_FOUND,
                code="not_found",
                fix="Blocks: " + ", ".join(n.type for n in catalog.nodes) + ".",
            ) from None
        print_json(card.model_dump(mode="json", exclude_none=True))
        return
    if json_output:
        print_json(catalog.model_dump(mode="json", exclude_none=True))
        return
    print_table(
        [
            {
                "block": n.type,
                "takes": ", ".join(n.inputs) or "-",
                "emits": n.output or "-",
                "settings": ", ".join(f.key for f in n.fields),
                "what it does": n.summary,
            }
            for n in catalog.nodes
        ],
        title=f"Workflow blocks ({len(catalog.nodes)})",
        no_wrap=("block",),
    )
    err_console.print(
        "`pictograph workflows nodes <block>` shows a block's settings, their values and "
        "defaults, and its rules."
    )


@app.command(
    "templates",
    help="The use cases a workflow starts from, or one template's graph (to a file with -o).",
)
def templates(
    key: Annotated[str | None, typer.Argument(help="One template (traffic, parking, ...).")] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the template's graph to this file.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if output is not None and key is None:
        raise typer.BadParameter("name the template whose graph to write", param_hint="--output")
    found = get_client(api_key).workflows.templates()
    if key is None:
        if json_output:
            print_json([t.model_dump(mode="json") for t in found])
            return
        print_table(
            [{"template": t.key, "title": t.title, "description": t.description} for t in found],
            title=f"Workflow templates ({len(found)})",
            no_wrap=("template",),
        )
        return
    template = next((t for t in found if t.key == key), None)
    if template is None:
        raise CliError(
            f"No template {key!r}.",
            exit_code=ExitCode.NOT_FOUND,
            code="not_found",
            fix="Templates: " + ", ".join(t.key for t in found) + ".",
        )
    if output is None:
        print_json(template.graph)
        return
    output.write_text(json_module.dumps(template.graph, indent=2) + "\n", encoding="utf-8")
    print_json({"template": template.key, "graph_path": str(output)})


@app.command(
    "validate",
    help="Whether a run of a graph would start, saving and starting nothing: every issue, on its "
    "block, checked against the input a run would name when you pass one. Exit 6 when it would not.",
)
def validate(
    workflow: Annotated[
        str | None, typer.Argument(help="A saved workflow (its name) - or pass --graph.")
    ] = None,
    graph: GraphOption = None,
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", help="Check it against a run on this dataset (its name or id)."),
    ] = None,
    directory: Annotated[
        str | None,
        typer.Option("--directory", help="With --dataset: one folder of it, e.g. /night."),
    ] = None,
    export: Annotated[
        str | None,
        typer.Option("--export", help="Check it against a run on this export (its name or id)."),
    ] = None,
    video: Annotated[
        str | None,
        typer.Option(
            "--video",
            help="Check it against a run on a video: an upload's handle or an http(s) link.",
        ),
    ] = None,
    image: Annotated[
        str | None,
        typer.Option(
            "--image",
            help="Check it against a run on an image: an upload's handle or an http(s) link.",
        ),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if (workflow is None) == (graph is None):
        raise typer.BadParameter("pass a saved workflow's name, or --graph <file>, not both")
    if sum(bool(x) for x in (dataset, export, video, image)) > 1:
        raise typer.BadParameter(
            "Name at most one input: --dataset / --export / --video / --image."
        )
    if directory and not dataset:
        raise typer.BadParameter("--directory is a folder of --dataset.", param_hint="--directory")
    source: WorkflowRunInput | None = None
    if dataset:
        source = WorkflowRunInput(kind="dataset", dataset=dataset, directory=directory)
    elif export:
        source = WorkflowRunInput(kind="export", export=export)
    elif video:
        source = _stored_or_linked("video", video, duration=None)
    elif image:
        source = _stored_or_linked("image", image, duration=None)
    graph_dict = _read_graph(graph) if graph is not None else None
    client = get_client(api_key)
    if graph_dict is None:
        graph_dict = client.workflows.get(workflow or "").graph
    report = client.workflows.validate(graph_dict, source=source)
    if not report.valid:
        lines = "\n".join("  " + _issue_line(i) for i in report.issues)
        raise CliError(
            f"This graph would not run - {len(report.errors)} error(s):\n{lines}",
            exit_code=ExitCode.INVALID,
            code="invalid_workflow",
            fix="Each issue names its block (node_id); `pictograph workflows nodes <block>` "
            "shows what a block takes.",
            details={"issues": [i.model_dump(mode="json") for i in report.issues]},
        )
    print_json(report.model_dump(mode="json"))
    for issue in report.issues:
        err_console.print(f"[yellow]warning: {escape(_issue_line(issue))}[/yellow]")


@app.command(
    "compute-options",
    help="The GPUs a workflow's endpoint runs on (memory, price in a region) and the precisions "
    "a Model block may take.",
)
def compute_options(
    region: Annotated[
        str | None,
        typer.Option("--region", help="Price them in this region (default: the one nearest you)."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the raw JSON.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    options = get_client(api_key).workflows.compute_options(region)
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
                "per hour": f"${g.cost_per_hour_micro_usd / 1_000_000:.2f}",
                "default": "yes" if g.gpu == options.default.gpu else "",
            }
            for g in options.gpus
        ],
        title=f"GPUs (in {options.region})" if options.region else "GPUs",
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
            # Its live endpoint, as `deployments list` says a deployment's: what it is doing right now.
            "live": live_status(w.endpoint.status, w.endpoint.runtime) if w.endpoint else "",
            "region": (w.endpoint.region or "") if w.endpoint else "",
            "engine": build_state(w.endpoint.engine_build) if w.endpoint else "",
            "last_run_id": w.last_run_id,
        }
        for w in workflows
    ]
    # The name is what the next command takes: never wrapped into two halves.
    print_table(rows, title=f"Workflows ({len(rows)})", no_wrap=("name",))


@app.command("get", help="Fetch a single workflow (graph + validation).")
def get_workflow(
    workflow: Annotated[str, typer.Argument(help="Workflow name or id.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    found = client.workflows.get(workflow)
    print_json(found.model_dump(mode="json", exclude_none=True))


@app.command(
    "create", help="Create a workflow from a graph JSON file, or from a template (--template)."
)
def create_workflow(
    name: Annotated[str, typer.Argument(help="Workflow name.")],
    graph: GraphOption = None,
    description: Annotated[str | None, typer.Option("--description")] = None,
    template_key: Annotated[
        str | None,
        typer.Option(
            "--template",
            help="Start from this template's graph (`pictograph workflows templates`); "
            "with --graph, only recorded.",
        ),
    ] = None,
    gpu: GpuOption = None,
    region: Region = "auto",
    min_containers: MinContainers = 0,
    max_containers: MaxContainers = 1,
    scaledown_window: ScaledownWindow = 60,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if graph is None and template_key is None:
        raise typer.BadParameter(
            "pass --graph <file> or --template <key> (`pictograph workflows templates` lists them)"
        )
    graph_dict = _read_graph(graph) if graph is not None else None
    client = get_client(api_key)
    workflow = client.workflows.create(
        name,
        graph=graph_dict,
        description=description,
        template_key=template_key,
        compute=_compute(gpu),
        region=region,
        min_containers=min_containers,
        max_containers=max_containers,
        scaledown_window=scaledown_window,
    )
    token = workflow.endpoint_token
    if token:
        say_token_once("the workflow's Analytics")
    print_json(
        {
            "id": workflow.id,
            "name": workflow.name,
            "status": workflow.status,
            "compute": workflow.compute.model_dump(),
            # Its endpoint, made with it: where its runs and streams run.
            "endpoint": _endpoint_out(workflow),
            **({"endpoint_token": token} if token else {}),
            # What a run of it would still be refused for (a template's model, line or zone, ...).
            "validation": [i.model_dump(mode="json") for i in workflow.validation],
        }
    )
    err_console.print(f"Run it: pictograph workflows run {escape(shlex.quote(workflow.name))}")


@app.command(
    "update",
    help="Update a workflow's graph / name / readme / status / GPU (a GPU change deploys its "
    "endpoint again on it).",
)
def update_workflow(
    workflow: Annotated[str, typer.Argument(help="Workflow name or id.")],
    graph: Annotated[
        str | None,
        typer.Option(
            "--graph",
            "-g",
            help="A NEW graph JSON file ({version,nodes,edges}), or - for stdin; replaces the graph.",
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
    graph_dict = _read_graph(graph) if graph is not None else None
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
        {
            "id": wf.id,
            "name": wf.name,
            "status": wf.status,
            "compute": wf.compute.model_dump(),
            "validation": [i.model_dump(mode="json") for i in wf.validation],
        }
    )


def _run_command(
    workflow: Annotated[str, typer.Argument(help="Workflow name or id.")],
    file: Annotated[
        str | None,
        typer.Option(
            "--file",
            help="A video or an image to run on: a file on disk (uploaded), an http(s) link Pictograph downloads, "
            "or - to read it from stdin.",
        ),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", help="Run on this dataset's images (its name or id)."),
    ] = None,
    directory: Annotated[
        str | None,
        typer.Option("--directory", help="With --dataset: one folder of it, e.g. /night."),
    ] = None,
    export: Annotated[
        str | None,
        typer.Option("--export", help="Run on the images this export captured (its name or id)."),
    ] = None,
    video: Annotated[
        str | None,
        typer.Option(
            "--video",
            help="Run on a video: the handle an upload returned (under your organization) or an http(s) link "
            "Pictograph downloads.",
        ),
    ] = None,
    image: Annotated[
        str | None,
        typer.Option(
            "--image",
            help="Run on an image: the handle an upload returned (under your organization) or an http(s) link "
            "Pictograph downloads.",
        ),
    ] = None,
    sample_fps: Annotated[
        float | None,
        typer.Option(
            "--sample-fps",
            help="A video's frames read a second (absent: every frame, at its own rate).",
        ),
    ] = None,
    duration: Annotated[
        float | None,
        typer.Option(
            "--duration",
            help="With --video: its length in seconds, the run's frame total until it reads the video.",
        ),
    ] = None,
    outputs: Annotated[
        str | None,
        typer.Option(
            "--outputs",
            help="What the run writes beside its log, as JSON - or @path to a JSON file: e.g. "
            '\'{"annotations": {"dataset": "Street scenes", "directory": "/drafts"}}\'. Its keys: '
            "annotated_video, annotations, event_capture, passive_capture, webhook (absent: its log alone).",
        ),
    ] = None,
    write_video: Annotated[
        bool,
        typer.Option(
            "--write-video",
            help="Write the run as an annotated video, drawn as its blocks draw it (annotated_video in --outputs).",
        ),
    ] = False,
    no_wait: Annotated[
        bool, typer.Option("--no-wait", help="Fire-and-forget; print run_id + status_url and exit.")
    ] = False,
    poll_interval: Annotated[
        float, typer.Option("--poll-interval", help="Seconds between polls.")
    ] = 5.0,
    timeout: Annotated[float, typer.Option("--timeout", help="Max seconds to wait.")] = 3600.0,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if sum(bool(x) for x in (file, dataset, export, video, image)) != 1:
        raise typer.BadParameter(
            "Name what it runs on: exactly one of --file / --dataset / --export / --video / --image "
            "(a workflow's Source block holds no input)."
        )
    if directory and not dataset:
        raise typer.BadParameter("--directory is a folder of --dataset.", param_hint="--directory")
    if duration is not None and not video:
        raise typer.BadParameter("--duration goes with --video.", param_hint="--duration")
    run_outputs = outputs_option(outputs, write_video=write_video)
    source: WorkflowRunInput | Path | str | bytes
    if file:
        if file == STDIN:
            # run() reads whether the bytes are a video or an image, then uploads them.
            source = read_stdin(
                "--file",
                example=f"`cat clip.mp4 | pictograph workflows run {workflow!r} --file -`",
            )
        elif is_url(file):
            source = file  # run() hands the link to Pictograph, which downloads it
        else:
            source = Path(file).expanduser()  # run() uploads the file and runs it
    elif dataset:
        source = WorkflowRunInput(kind="dataset", dataset=dataset, directory=directory)
    elif export:
        source = WorkflowRunInput(kind="export", export=export)
    elif video:
        source = _stored_or_linked("video", video, duration=duration)
    else:
        source = _stored_or_linked("image", image or "", duration=None)
    client = get_client(api_key)
    try:
        if no_wait:
            result: Any = client.workflows.run(
                workflow,
                source=source,
                sample_fps=sample_fps,
                outputs=run_outputs,
                poll_interval=poll_interval,
                timeout=timeout,
            )
        else:
            result = client.workflows.run(
                workflow,
                source=source,
                sample_fps=sample_fps,
                outputs=run_outputs,
                wait=True,
                poll_interval=poll_interval,
                timeout=timeout,
            )
    except (
        ValueError
    ) as exc:  # a file, link or bytes whose kind can't be read; a rate for what is no video
        hint = "--sample-fps" if "sample_fps" in str(exc) else "--file"
        raise typer.BadParameter(str(exc), param_hint=hint) from None
    print_json(result.model_dump(mode="json", exclude_none=True))


# `run`: the input it runs on named at the call (B606: a workflow's Source block holds none); waits unless --no-wait.
app.command(
    "run",
    help="Run a workflow on an input - a video or an image (a file, a link, stdin), a dataset or one of its folders, "
    "or an export - on its endpoint (billed as its uptime); waits for it unless --no-wait.",
)(_run_command)
# `invoke`: run's old name (the same options), kept working and out of the help.
app.command("invoke", hidden=True, help="The old name of `run`.")(_run_command)


def outputs_option(text: str | None, *, write_video: bool = False) -> dict[str, Any] | None:
    """``--outputs`` (JSON, or @path to a JSON file) and ``--write-video``: a run's outputs, checked here - a mistyped
    key or value is a usage error before any request."""
    given: dict[str, Any] = {}
    if text:
        raw = text
        if text.startswith("@"):
            try:
                raw = Path(text[1:]).expanduser().read_text()
            except OSError as e:
                raise typer.BadParameter(
                    f"could not read {text[1:]}: {e}", param_hint="--outputs"
                ) from None
        try:
            parsed = json_module.loads(raw)
        except ValueError as e:
            raise typer.BadParameter(f"not JSON: {e}", param_hint="--outputs") from None
        if not isinstance(parsed, dict):
            raise typer.BadParameter("a JSON object of the run's outputs.", param_hint="--outputs")
        given = parsed
    if write_video:
        given["annotated_video"] = True
    if not given:
        return None
    try:
        return WorkflowRunOutputs.model_validate(given).model_dump(exclude_none=True)
    except PydanticValidationError as e:
        errors = e.errors()
        where = ".".join(str(part) for part in errors[0]["loc"]) if errors else ""
        said = str(errors[0]["msg"]) if errors else "not outputs a run takes"
        raise typer.BadParameter(
            f"{where}: {said}" if where else said, param_hint="--outputs"
        ) from None


def _stored_or_linked(
    kind: WorkflowRunInputKind, value: str, *, duration: float | None
) -> WorkflowRunInput:
    """``--video`` / ``--image``: a stored file's handle (under your organization), or a link Pictograph downloads."""
    if is_url(value):
        return WorkflowRunInput(kind=kind, url=value, duration_seconds=duration)
    if value.startswith("gs://"):
        stored = {"video": value} if kind == "video" else {"image": value}
        return WorkflowRunInput.model_validate(
            {"kind": kind, **stored, "duration_seconds": duration}
        )
    raise typer.BadParameter(
        f"{value!r} is neither the handle an upload returned nor an http(s) link. For a file on disk use --file.",
        param_hint=f"--{kind}",
    )


@app.command("run-status", help="Poll a run's status + artifacts (signed URLs).")
def run_status(
    run: Annotated[str, typer.Argument(help="Workflow run UUID.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    result = client.workflows.get_run(run)
    print_json(result.model_dump(mode="json", exclude_none=True))


_RUN_KINDS = ("all", "file", "stream")


def _run_source(source: Any) -> str | None:
    """What a run ran over, as the app's Runs table says it: video, image, dataset - or a stream by how it came."""
    if source is None:
        return None
    if source.kind == "stream":
        return {"whip": "stream", "rtsp": "rtsp", "whep": "webrtc"}.get(source.via or "", "stream")
    return str(source.kind)


@app.command(
    "runs",
    help="List a workflow's runs, newest first (paginated). Each stream session is a run too.",
)
def list_runs(
    workflow: Annotated[str, typer.Argument(help="Workflow name or id.")],
    limit: Annotated[int, typer.Option("--limit", min=1, max=100, help="Page size (1-100).")] = 50,
    offset: Annotated[int, typer.Option("--offset", min=0, help="Rows to skip.")] = 0,
    kind: Annotated[
        str,
        typer.Option(
            "--kind", help="all (default), file (a video, image or dataset run) or stream."
        ),
    ] = "all",
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if kind not in _RUN_KINDS:
        raise typer.BadParameter("--kind is all, file or stream")
    client = get_client(api_key)
    runs = client.workflows.list_runs(
        workflow,
        limit=limit,
        offset=offset,
        kind=cast("WorkflowRunsKind", kind),
    )
    if json_output:
        print_json([r.model_dump(mode="json", exclude_none=True) for r in runs])
        return
    rows = [
        {
            "id": r.id,
            "source": _run_source(r.source),
            "status": r.status,
            "progress": f"{r.progress:.0f}%",
            "frames": f"{r.frames_done}/{r.frames_total if r.frames_total is not None else '?'}",
            "cost_uUSD": r.cost_micro_usd,
            "artifacts": r.artifact_count,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in runs
    ]
    print_table(rows, title=f"Runs of {workflow} ({len(rows)})")


def _branch(value: str | None) -> bool | None:
    """``--branch true|false``: a Condition block's branch (None: every record)."""
    if value is None:
        return None
    word = value.strip().lower()
    if word not in ("true", "false"):
        raise typer.BadParameter("--branch is true or false")
    return word == "true"


def _branch_word(condition: bool | None) -> str | None:
    return None if condition is None else ("true" if condition else "false")


@app.command("log", help="Read a run's log: its events, frames or stats as the engine logged them.")
def run_log(
    run: Annotated[str, typer.Argument(help="Workflow run UUID.")],
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
    track: Annotated[
        str | None, typer.Option("--track", help="Only records with this track.")
    ] = None,
    branch: Annotated[
        str | None,
        typer.Option(
            "--branch",
            help="Only records a Condition block routed down this branch: true or false.",
        ),
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
        s = client.workflows.run_log_summary(run)
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
        print_table(spans, title=f"Log of run {run}{note}")
        return
    records = list(
        client.workflows.iter_run_log(
            run,
            stream=streams[stream],
            page_size=min(limit, 200),
            order="desc" if newest_first else "asc",
            full=full,
            max_total=limit,
            class_name=class_name,
            track=track,
            branch=_branch(branch),
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
    branched = any(r.condition is not None for r in records)
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
                **({"branch": _branch_word(r.condition)} if branched else {}),
            }
            for r in records
        ]
    elif stream == "events":
        # Each event named as the app and the webhooks name it (one more page read: its names).
        names = (
            client.workflows.run_log(run, stream="events", limit=1).event_names if records else {}
        )
        rows = [
            {
                "time": r.ts.isoformat(),
                "type": r.type,
                "event": names[r.trigger].name if r.trigger in names else r.trigger,
                "source": r.source,
                "frame": r.frame,
                # The values as words (the server's); PRT's keys only from a server that names none.
                "values": r.values_text if r.values_text is not None else r.rule_values,
                "capture": r.capture_name or r.capture_kind,
                **({"branch": _branch_word(r.condition)} if branched else {}),
            }
            for r in records
        ]
    else:
        rows = [{"time": r.ts.isoformat(), "type": r.type, "sources": r.sources} for r in records]
    print_table(rows, title=f"{stream.capitalize()} of run {run} ({len(rows)})")


@app.command(
    "cancel",
    help="Cancel an in-flight run: its outputs so far are kept; its time on its endpoint until it "
    "stops is billed as uptime.",
)
def cancel_run(
    run: Annotated[str, typer.Argument(help="Workflow run UUID.")],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(f"Cancel workflow run {run!r}?"):
        raise typer.Abort()
    client = get_client(api_key)
    client.workflows.cancel_run(run)
    print_json({"run_id": run, "status": "cancelled"})


@app.command("cancel-batch", help="Cancel many in-flight workflow runs in one server-side call.")
def cancel_runs_batch(
    runs: Annotated[list[str], typer.Argument(help="Workflow run UUIDs to cancel.")],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(f"Cancel {len(runs)} workflow run(s)?"):
        raise typer.Abort()
    client = get_client(api_key)
    # One server-side bulk call - never N fan-out requests.
    print_json(client.workflows.bulk_cancel_runs(runs).model_dump(mode="json"))


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


# ── its live endpoint (B579) ──────────────────────────────────────────────────


@app.command(
    "deploy",
    help="Deploy a workflow's endpoint: one made before endpoints were made with workflows gets one "
    "(its token shown once); a failed or draft one is deployed again.",
)
def deploy_workflow(
    workflow: Annotated[str, typer.Argument(help="Workflow name or id.")],
    region: Region = "auto",
    min_containers: MinContainers = 0,
    max_containers: MaxContainers = 1,
    scaledown_window: ScaledownWindow = 60,
    wait: Annotated[bool, typer.Option("--wait/--no-wait", help=_WAIT_HELP)] = False,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Max seconds to wait (with --wait).")
    ] = 900.0,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    deployed = client.workflows.deploy(
        workflow,
        region=region,
        min_containers=min_containers,
        max_containers=max_containers,
        scaledown_window=scaledown_window,
    )
    token = deployed.endpoint_token
    if token:
        say_token_once("the workflow's Analytics")

    def printed(current: Workflow) -> dict[str, Any]:
        return {
            "id": current.id,
            "name": current.name,
            "endpoint": _endpoint_out(current),
            **({"endpoint_token": token} if token else {}),
        }

    if not wait:
        print_json(printed(deployed))
        return
    try:
        live = _wait_live(client, deployed, timeout)
    except BaseException:
        # The token is shown once: a wait that fails (or is interrupted) still prints it.
        print_json(printed(deployed))
        raise
    print_json(printed(live))


@app.command(
    "pause",
    help="Pause a workflow: its endpoint stopped, its streams and runs refused until resumed.",
)
def pause_workflow(
    workflow: Annotated[str, typer.Argument(help="Workflow name or id.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    paused = get_client(api_key).workflows.pause(workflow)
    print_json({"id": paused.id, "endpoint": _endpoint_out(paused)})


@app.command(
    "resume",
    help="Resume a paused workflow: its endpoint deployed again on its graph as it is now.",
)
def resume_workflow(
    workflow: Annotated[str, typer.Argument(help="Workflow name or id.")],
    wait: Annotated[bool, typer.Option("--wait/--no-wait", help=_WAIT_HELP)] = False,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Max seconds to wait (with --wait).")
    ] = 900.0,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    resumed = client.workflows.resume(workflow)
    if wait:
        resumed = _wait_live(client, resumed, timeout)
    print_json({"id": resumed.id, "endpoint": _endpoint_out(resumed)})


@app.command(
    "scale",
    help="Change a workflow's endpoint's autoscaling (only what is given; a live one at once).",
)
def scale_workflow(
    workflow: Annotated[str, typer.Argument(help="Workflow name or id.")],
    min_containers: MinChange = None,
    max_containers: MaxChange = None,
    scaledown_window: ScaledownChange = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    scaled = get_client(api_key).workflows.update_endpoint(
        workflow,
        min_containers=min_containers,
        max_containers=max_containers,
        scaledown_window=scaledown_window,
    )
    ep = scaled.endpoint
    print_json(
        {
            "id": scaled.id,
            "min_containers": ep.min_containers if ep else None,
            "max_containers": ep.max_containers if ep else None,
            "scaledown_window": ep.scaledown_window if ep else None,
        }
    )


@app.command(
    "stream",
    help="Stream live video through a workflow - every block a frame passes - and print each frame's results "
    "(its lines' and zones' counts under measures) as it is processed, one JSON line per frame. SOURCE is a camera "
    "the workflow connects to (rtsp://, rtsps://, or a WebRTC camera's https:// WHEP URL), or whip to send the "
    "video yourself. Ctrl-C stops it; it is billed while it runs.",
)
def stream_workflow(
    workflow: Annotated[str, typer.Argument(help="Workflow name or id.")],
    source: Annotated[
        str, typer.Argument(help="rtsp://..., rtsps://..., an https:// WHEP URL, or whip.")
    ],
    token: Annotated[
        str, typer.Option("--token", help="The workflow's endpoint token (pk_deploy_...).")
    ],
    endpoint: Annotated[
        str | None,
        typer.Option(
            "--endpoint",
            help="Its endpoint's URL (its endpoint_url, ending in /sessions). With this, --token is the only credential.",
        ),
    ] = None,
    frames: Annotated[
        int | None, typer.Option("--frames", min=1, help="Stop after this many frames.")
    ] = None,
    seconds: Annotated[
        float | None, typer.Option("--seconds", min=1, help="Stop after this many seconds.")
    ] = None,
    overlay: Annotated[
        bool, typer.Option("--overlay/--no-overlay", help="Draw its look on the live view.")
    ] = True,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream/--no-stream",
            help="Send the live view's pictures. --no-stream: none are drawn or encoded; each "
            "frame's result, its events and its outputs still come.",
        ),
    ] = True,
    all_messages: Annotated[
        bool, typer.Option("--all", help="Print every message, not only its frames.")
    ] = False,
    idle_timeout: Annotated[
        float | None,
        typer.Option(
            "--idle-timeout",
            min=0.1,
            help="Fail after this many seconds without a frame (exit 9, stream_stalled): its camera "
            "stalled or its connection dropped. The stream stops with it.",
        ),
    ] = None,
    transport: Annotated[
        str | None,
        typer.Option(
            "--transport",
            help="An rtsp:// camera's: tcp (the default: over the RTSP connection - a camera's "
            "UDP seldom reaches a cloud container), auto (UDP first, then TCP) or udp.",
        ),
    ] = None,
    buffer_mode: Annotated[
        str | None,
        typer.Option(
            "--buffer-mode",
            help="An rtsp:// camera's: auto smooths its clock onto the workflow's; none keeps "
            "the sender's own timestamps (a recording streamed).",
        ),
    ] = None,
    reconnect: Annotated[
        str | None,
        typer.Option(
            "--reconnect",
            help="An rtsp:// camera's: always (a camera that drops or ends is tried again) or "
            "never (a recording: the stream ends at its end, its last frame decoded).",
        ),
    ] = None,
    outputs: Annotated[
        str | None,
        typer.Option(
            "--outputs",
            help="What the stream writes beside its log, as JSON - or @path to a JSON file: event_capture, "
            "passive_capture, webhook (a stream writes its frames through its captures). E.g. "
            '\'{"passive_capture": {"dataset": "Street scenes", "directory": "/bg", "every_s": 10}}\'.',
        ),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    stream_outputs = outputs_option(outputs)
    if endpoint:
        conn: Any = DeploymentClient(endpoint=endpoint, api_key=token)
    else:
        account = get_client(api_key)
        conn = account.workflows.connect(account.workflows.get(workflow), token=token)
    run_stream(
        conn,
        source,
        overlay=overlay,
        stream=stream,
        frames=frames,
        seconds=seconds,
        all_messages=all_messages,
        idle_timeout=idle_timeout,
        transport=transport,
        buffer_mode=buffer_mode,
        reconnect=reconnect,
        outputs=stream_outputs,
    )


@app.command(
    "live-log",
    help="Read what a workflow's live streams logged: frames (default), events or stats.",
)
def live_log(
    workflow: Annotated[str, typer.Argument(help="Workflow name or id.")],
    stream: Annotated[str, typer.Option("--stream", help="frames, events or stats.")] = "frames",
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    branch: Annotated[
        str | None,
        typer.Option(
            "--branch",
            help="Only records a Condition block routed down this branch: true or false.",
        ),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    streams: dict[str, RunLogStream] = {"frames": "frames", "events": "events", "stats": "stats"}
    if stream not in streams:
        raise typer.BadParameter("--stream is frames, events or stats")
    page = get_client(api_key).workflows.live_log(
        workflow, stream=streams[stream], limit=limit, branch=_branch(branch)
    )
    print_json(page.model_dump(mode="json"))
