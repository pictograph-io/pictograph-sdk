"""Rich rendering helpers for CLI output.

Two output modes:
- **table** (default) - Rich tables for human-friendly terminals.
- **json** - pretty-printed JSON for piping into ``jq`` / scripting.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table

console = Console()
err_console = Console(stderr=True)


def print_json(payload: Any) -> None:
    """Print ``payload`` as indented JSON to stdout. Used for ``--json`` output."""
    sys.stdout.write(json.dumps(payload, indent=2, default=str))
    sys.stdout.write("\n")


def print_table(
    rows: list[dict[str, Any]],
    *,
    columns: list[str] | None = None,
    title: str | None = None,
) -> None:
    """Render ``rows`` as a Rich table.

    Args:
        rows: List of dicts. Empty list prints "No results."
        columns: Column names. Defaults to keys of the first row.
        title: Optional table title.
    """
    if not rows:
        console.print("[dim]No results.[/dim]")
        return
    cols = columns or list(rows[0].keys())
    # Escape the title too: some callers build it from server data
    # (e.g. a connector workspace name, a usage range) that can contain
    # bracket sequences Rich would otherwise parse as markup. No title
    # uses intentional markup, so escaping is always safe here.
    table = Table(
        title=escape(title) if title is not None else None,
        show_lines=False,
        header_style="bold",
    )
    for col in cols:
        table.add_column(col)
    for row in rows:
        table.add_row(*[_render_cell(row.get(col, "")) for col in cols])
    console.print(table)


def _render_cell(value: Any) -> str:
    """Coerce arbitrary values into Rich-friendly strings.

    Server-controlled values (names, slugs, descriptions, error strings, paths
    like ``frame[01].jpg``) are escaped with :func:`rich.markup.escape` so a
    bracket sequence is rendered literally instead of being parsed as console
    markup - an unbalanced/closing tag (``[/]``) would raise ``MarkupError`` and
    crash the whole command, and a style-like tag (``[red]…``) would be silently
    consumed, dropping text. The styled literals below are our own constants
    (intentional markup) and stay unescaped.
    """
    if value is None:
        return "[dim]-[/dim]"
    if isinstance(value, bool):
        return "[green]yes[/green]" if value else "[red]no[/red]"
    if isinstance(value, list | tuple):
        return ", ".join(escape(str(v)) for v in value)
    if isinstance(value, dict):
        return escape(json.dumps(value, default=str))
    return escape(str(value))


def print_error(message: str) -> None:
    """Bold-red error to stderr - used by exception handlers.

    ``message`` is escaped: error strings routinely carry server/user text with
    brackets, which would otherwise be parsed as markup (crash on a bad tag, or
    silently garble the message). The ``[bold red]`` prefix is our own markup.
    """
    err_console.print(f"[bold red]error:[/bold red] {escape(message)}")
