"""``pictograph tasks {list,contributions}``."""

from __future__ import annotations

from typing import Annotated

import typer

from pictograph.cli._client import get_client
from pictograph.cli._format import print_json, print_table

app = typer.Typer(no_args_is_help=True)


def _fmt_duration(seconds: int) -> str:
    """Compact active-editing duration ("2h 5m" / "3m 10s" / "45s")."""
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m" if m else f"{h}h"
    return f"{m}m {sec}s" if sec else f"{m}m"


@app.command("list", help="List the organization's annotation tasks (newest first).")
def list_tasks(
    limit: Annotated[int, typer.Option("--limit", "-n")] = 50,
    offset: Annotated[int, typer.Option("--offset")] = 0,
    all_tasks: Annotated[
        bool,
        typer.Option("--all", help="Auto-page ALL tasks (ignores --limit/--offset)."),
    ] = False,
    max_total: Annotated[
        int | None,
        typer.Option("--max-total", help="Cap on total tasks fetched when using --all."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    tasks = (
        list(client.tasks.iter(max_total=max_total))
        if all_tasks
        else client.tasks.list(limit=limit, offset=offset)
    )
    if json_output:
        print_json([t.model_dump(mode="json", exclude_none=True) for t in tasks])
        return
    rows = [
        {
            "id": t.id,
            "title": t.title,
            "dataset": t.dataset,
            "kind": t.kind,
            "status": t.status,
            "images": t.image_count,
            "assignees": t.assignee_count,
        }
        for t in tasks
    ]
    print_table(rows, title=f"Tasks ({len(rows)})")


@app.command("contributions", help="Per-annotator contribution breakdown for a task.")
def contributions(
    task_id: Annotated[str, typer.Argument(help="The task's id.")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    c = client.tasks.contributions(task_id)
    if json_output:
        print_json(c.model_dump(mode="json", exclude_none=True))
        return
    rows = [
        {
            "annotator": con.full_name,
            "assignee": "yes" if con.is_assignee else "no",
            "images": con.images_worked,
            "active": _fmt_duration(con.active_seconds),
            "completed": con.images_completed,
            "annotations": con.annotations_added,
        }
        for con in c.contributors
    ]
    print_table(
        rows,
        title=(
            f"Contributors ({c.contributor_count}) - "
            f"{c.images_complete}/{c.total_images} images complete, "
            f"{_fmt_duration(c.total_active_seconds)} total"
        ),
    )
