"""``pictograph workflows {list,get,create,run,run-status,cancel,delete}``.

Drive node-graph Workflows headlessly. Build the graph in the app, then::

    pictograph workflows list
    pictograph workflows run <workflow-id>            # runs + waits, prints artifacts
    pictograph workflows run <workflow-id> --no-wait  # fire-and-forget, prints run id
    pictograph workflows run-status <run-id>          # poll a run + its artifacts
"""

from __future__ import annotations

import json as json_module
from pathlib import Path
from typing import Annotated

import typer

from pictograph.cli._client import get_client
from pictograph.cli._format import print_json, print_table

app = typer.Typer(no_args_is_help=True)


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
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not graph.is_file():
        raise typer.BadParameter(f"graph file not found: {graph}", param_hint="--graph")
    graph_dict = json_module.loads(graph.read_text(encoding="utf-8"))
    client = get_client(api_key)
    workflow = client.workflows.create(
        name, graph_dict, description=description, template_key=template_key
    )
    print_json({"id": workflow.id, "name": workflow.name, "status": workflow.status})


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
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    created = client.workflows.run(workflow)
    if no_wait:
        print_json({"run_id": created.run_id, "deposit_micro_usd": created.deposit_micro_usd})
        return
    run = client.workflows.wait_for_run(
        created.run_id, poll_interval=poll_interval, timeout=timeout
    )
    print_json(run.model_dump(mode="json", exclude_none=True))


@app.command("run-status", help="Poll a run's status + artifacts (signed URLs).")
def run_status(
    run_id: Annotated[str, typer.Argument(help="Workflow run UUID.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    run = client.workflows.get_run(run_id)
    print_json(run.model_dump(mode="json", exclude_none=True))


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
