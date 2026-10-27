"""``pictograph login`` - interactive API-key setup, one profile per key."""

from __future__ import annotations

from typing import Annotated

import typer

from pictograph.cli._config import DEFAULT_PROFILE, active_profile, check_profile_name, write_config
from pictograph.cli._format import print_json


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
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile",
            help="Save the key as this profile (one per API key / organization); run as "
            "it with `pictograph --profile <name> ...`. Default: the active profile.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print {config_path, profile} as JSON instead of a sentence."),
    ] = False,
) -> None:
    """Write the supplied API key to ``~/.pictograph/config.toml``, as one profile."""
    name = profile or active_profile()[0]
    try:
        check_profile_name(name)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--profile") from None
    key = (api_key or typer.prompt("Pictograph API key", hide_input=True)).strip()
    if not key.startswith("pk_"):
        typer.echo(
            "Warning: API keys typically start with 'pk_live_' - proceeding anyway.",
            err=True,
        )
    path = write_config(api_key=key, base_url=base_url, profile=name)
    if json_output:
        print_json({"config_path": str(path), "profile": name})
        return
    if name == DEFAULT_PROFILE:
        typer.echo(f"Saved API key to {path}")
    else:
        typer.echo(
            f"Saved API key to {path} as profile {name!r} - run as it with "
            f"`pictograph --profile {name} ...` or PICTOGRAPH_PROFILE={name}."
        )
