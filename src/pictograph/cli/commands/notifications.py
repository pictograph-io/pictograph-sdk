"""``pictograph notifications {list,unread-count,read,read-all,delete}``."""

from __future__ import annotations

from typing import Annotated

import typer

from pictograph.cli._client import get_client
from pictograph.cli._format import print_json, print_table

app = typer.Typer(no_args_is_help=True)


@app.command("list", help="List the organization's notifications (newest first).")
def list_cmd(
    unread: Annotated[bool, typer.Option("--unread", help="Only unread notifications.")] = False,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 50,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    items = client.notifications.list(unread_only=unread, limit=limit)
    if json_output:
        print_json([n.model_dump(mode="json", exclude_none=True) for n in items])
        return
    print_table(
        [
            {
                "id": n.id,
                "type": n.type,
                "title": n.title,
                "read": "•" if n.read else "unread",
                "created_at": str(n.created_at),
            }
            for n in items
        ],
        title="Notifications",
    )


@app.command("unread-count", help="Show the number of unread notifications.")
def unread_count(
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    typer.echo(str(client.notifications.unread_count()))


@app.command("read", help="Mark a notification read by id.")
def read(
    notification_id: Annotated[str, typer.Argument(help="Notification id.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    client.notifications.mark_read(notification_id)
    typer.echo(f"Marked {notification_id} read.")


@app.command("read-all", help="Mark every unread notification read.")
def read_all(
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    marked = client.notifications.mark_all_read()
    typer.echo(f"Marked {marked} notification(s) read.")


@app.command("delete", help="Delete a notification by id.")
def delete(
    notification_id: Annotated[str, typer.Argument(help="Notification id.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    client.notifications.delete(notification_id)
    typer.echo(f"Deleted {notification_id}.")
