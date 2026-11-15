"""``pictograph profiles list`` - the profiles of ``~/.pictograph/config.toml``.

An API key belongs to one organization; an agent working for several keeps one profile per
key and picks one per command::

    pictograph login --profile research
    pictograph --profile research datasets list
    PICTOGRAPH_PROFILE=research pictograph jobs list
"""

from __future__ import annotations

import os
from typing import Annotated

import typer

from pictograph.cli._config import active_profile, load_config, profile_names
from pictograph.cli._errors import command_group
from pictograph.cli._format import err_console, print_json, print_table

app = command_group(no_args_is_help=True)


def _masked(key: str | None) -> str | None:
    """A key's first 12 characters - its prefix, which the API itself treats as safe to show."""
    return f"{key[:12]}..." if key else None


@app.command("list", help="List the profiles (keys masked) and the one commands run as.")
def list_profiles(
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    active, source = active_profile()
    env_key = bool(os.environ.get("PICTOGRAPH_API_KEY")) and source != "--profile"
    rows = []
    for name in profile_names():
        cfg = load_config(name)
        rows.append(
            {
                "profile": name,
                "key": _masked(cfg.api_key),
                "base_url": cfg.base_url,
                "active": name == active and not env_key,
            }
        )
    if json_output:
        print_json(
            {
                "profiles": rows,
                "active": None if env_key else active,
                "source": "PICTOGRAPH_API_KEY" if env_key else source,
            }
        )
        return
    print_table(
        [
            {**r, "base_url": r["base_url"] or "production", "active": "yes" if r["active"] else ""}
            for r in rows
        ],
        title=f"Profiles ({len(rows)})",
    )
    if env_key:
        err_console.print(
            "PICTOGRAPH_API_KEY is set: commands run with it unless --profile names one."
        )
