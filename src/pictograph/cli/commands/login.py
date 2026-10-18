"""``pictograph login`` - interactive API-key setup."""

from __future__ import annotations

from typing import Annotated

import typer

from pictograph.cli._config import write_config


def command(
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            help="API key. If omitted, prompts interactively (input hidden).",
        ),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help="Override the API base URL (default: https://api.pictograph.io).",
        ),
    ] = None,
) -> None:
    """Write ``~/.pictograph/config.toml`` with the supplied API key."""
    key = (api_key or typer.prompt("Pictograph API key", hide_input=True)).strip()
    if not key.startswith("pk_"):
        typer.echo(
            "Warning: API keys typically start with 'pk_live_' - proceeding anyway.",
            err=True,
        )
    path = write_config(api_key=key, base_url=base_url)
    typer.echo(f"Saved API key to {path}")
