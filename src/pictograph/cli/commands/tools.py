"""``pictograph tools {list,describe}`` - the vision tools you can call, and how.

These are your organization's VISION tools: SAM3, your models, your deployments and
your workflows. (``pictograph agents list-tools`` is a different list: the SDK's own
platform operations, as tool definitions for an agent framework.)
"""

from __future__ import annotations

from typing import Annotated

import typer

from pictograph.cli._client import get_client
from pictograph.cli._format import print_json, print_table

app = typer.Typer(no_args_is_help=True)

_KINDS = ("sam3", "model", "deployment", "workflow")


@app.command(
    "list",
    help="List the vision tools you can call: SAM3, your ready models, active deployments, workflows.",
)
def list_tools(
    kind: Annotated[
        str | None,
        typer.Option("--kind", help="Only this kind: sam3, model, deployment or workflow."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if kind is not None and kind not in _KINDS:
        raise typer.BadParameter(f"--kind must be one of: {', '.join(_KINDS)}", param_hint="--kind")
    client = get_client(api_key)
    tools = client.tools.list(kind=kind)  # type: ignore[arg-type]
    if json_output:
        print_json([t.model_dump(mode="json") for t in tools])
        return
    print_table(
        [
            {"tool": t.tool, "task": (t.task or "").replace("_", " "), "summary": t.summary}
            for t in tools
        ],
        title=f"Vision tools ({len(tools)}) - `pictograph tools describe <tool>` for how to call one",
    )


@app.command(
    "describe",
    help=(
        "Show how to call one tool: its inputs, parameters, output, cost and the exact "
        "CLI / Python / HTTP calls. TOOL is a value from `tools list`: sam3, "
        "model/<name>, deployment/<name> or workflow/<name>."
    ),
)
def describe_tool(
    tool: Annotated[
        str,
        typer.Argument(help="sam3, model/<name>, deployment/<name> or workflow/<name>."),
    ],
    json_output: Annotated[
        bool, typer.Option("--json", help="The card as JSON (for an agent) instead of Markdown.")
    ] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    try:
        card = client.tools.describe(tool)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="TOOL") from exc
    if json_output:
        print_json(card.model_dump(mode="json", by_alias=True))
        return
    typer.echo(card.to_markdown(), nl=False)
