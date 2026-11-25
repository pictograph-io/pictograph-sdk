"""``pictograph api-keys {list,get,create,update,revoke}`` - mint a narrowly-scoped key for an agent.

A key carries a role (``viewer`` < ``member`` < ``admin`` < ``owner``) and acts only in its
organization; the API never lets a key mint one above its own role. ``create`` shows the
secret ONCE - or, with ``--save-profile``, keeps it in a CLI profile without printing it::

    pictograph api-keys create labeller --role member --save-profile labeller
    pictograph --profile labeller datasets list
    pictograph api-keys revoke labeller --yes

Every command that takes a key takes its id, its prefix (``pk_live_...``, or the whole key)
or its name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, cast

import typer

from pictograph.cli._client import get_client
from pictograph.cli._config import (
    active_profile,
    check_profile_name,
    resolve_api_key,
    resolve_base_url,
    write_config,
)
from pictograph.cli._errors import CliError, ExitCode, command_group
from pictograph.cli._format import err_console, print_json, print_table
from pictograph.resources._resolve import looks_like_id

if TYPE_CHECKING:
    from pictograph.models.api_key import ApiKey, ApiKeyRole

app = command_group(no_args_is_help=True)

ROLES = ("viewer", "member", "admin", "owner")
_KEY_HELP = "The key's id, its prefix (pk_live_...), or its name."


@app.command("list", help="List your organization's API keys (never their secrets).")
def list_keys(
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    keys = client.api_keys.list()
    if json_output:
        print_json([k.model_dump(mode="json") for k in keys])
        return
    rows = [
        {
            "name": k.name,
            "prefix": k.key_prefix,
            "role": k.role,
            "per hour": k.rate_limit,
            "active": k.is_active,
            "last used": k.last_used_at,
            "expires": k.expires_at,
            "id": k.id,
        }
        for k in keys
    ]
    print_table(rows, title=f"API keys ({len(rows)})", no_wrap=("prefix", "id"))


@app.command("get", help="Show one API key's metadata (never its secret), by id, prefix or name.")
def get_key(
    key: Annotated[str, typer.Argument(help=_KEY_HELP)],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    target = _find_key(client.api_keys.list(), key)
    print_json(target.model_dump(mode="json", exclude_none=True))


@app.command(
    "create",
    help="Create an API key. Its secret is printed once - or kept in a CLI profile, unprinted, "
    "with --save-profile.",
)
def create_key(
    name: Annotated[str, typer.Argument(help="A label for the key (1-100 characters).")],
    role: Annotated[
        str,
        typer.Option(
            "--role",
            help="viewer (read), member (read + write), admin, or owner. Never above your own.",
        ),
    ] = "member",
    rate_limit: Annotated[
        int | None,
        typer.Option(
            "--rate-limit",
            min=1,
            help="Cap this key at N requests per hour (below the plan's limit, never above).",
        ),
    ] = None,
    expires: Annotated[
        str | None,
        typer.Option("--expires", help="When the key stops working (ISO 8601, e.g. 2026-12-31)."),
    ] = None,
    save_profile: Annotated[
        str | None,
        typer.Option(
            "--save-profile",
            help="Keep the new key as this CLI profile (~/.pictograph/config.toml, 0600) "
            "instead of printing it; run as it with `pictograph --profile <name> ...`.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if role not in ROLES:
        raise typer.BadParameter("use viewer, member, admin or owner.", param_hint="--role")
    if save_profile is not None:
        try:
            check_profile_name(save_profile)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--save-profile") from None
    client = get_client(api_key)
    created = client.api_keys.create(
        name,
        role=cast("ApiKeyRole", role),
        rate_limit=rate_limit,
        expires_at=expires,
    )
    summary = {
        "id": created.key_id,
        "name": created.name,
        "key_prefix": created.key_prefix,
        "role": created.role,
        "rate_limit": created.rate_limit,
    }
    if save_profile is not None:
        # The new key talks to the same API as the key that made it.
        path = write_config(
            api_key=created.api_key, base_url=resolve_base_url(None), profile=save_profile
        )
        if json_output:
            print_json({**summary, "profile": save_profile, "config_path": str(path)})
            return
        typer.echo(
            f"Created {created.name!r} ({created.key_prefix}..., {created.role}) and saved it as "
            f"profile {save_profile!r} in {path}. Run as it with "
            f"`pictograph --profile {save_profile} ...`."
        )
        return
    err_console.print(
        "Save this key now: it is printed once (Settings can reveal it again, once per "
        "sign-in). --save-profile keeps it in a CLI profile instead."
    )
    print_json({**summary, "api_key": created.api_key})


def _find_key(keys: list[ApiKey], ref: str) -> ApiKey:
    """The key ``ref`` names: its id, its prefix (or the whole key), or its name."""
    if looks_like_id(ref):
        hits = [k for k in keys if k.id == ref]
    elif ref.startswith("pk_"):
        hits = [k for k in keys if ref.startswith(k.key_prefix) or k.key_prefix.startswith(ref)]
    else:
        hits = [k for k in keys if k.name == ref]
    if not hits:
        raise CliError(
            f"No API key {ref!r} in your organization.",
            exit_code=ExitCode.NOT_FOUND,
            code="not_found",
            fix="`pictograph api-keys list` shows each key's name, prefix and id.",
        )
    if len(hits) > 1:
        raise CliError(
            f"{len(hits)} API keys match {ref!r} - pass the id of the one you mean.",
            exit_code=ExitCode.INVALID,
            code="ambiguous_key",
            details={"ids": [k.id for k in hits]},
            fix="`pictograph api-keys list` shows each key's id.",
        )
    return hits[0]


def _warn_if_own(api_key: str | None, target: ApiKey, consequence: str) -> bool:
    """Say so (on stderr) when ``target`` is the key this command runs with."""
    calling = resolve_api_key(api_key) or ""
    own = bool(calling) and calling.startswith(target.key_prefix)
    if own:
        name, _ = active_profile()
        err_console.print(
            f"This is the key this command runs with (profile {name!r}): {consequence}"
        )
    return own


@app.command(
    "update",
    help="Rename an API key, cap its requests per hour, or turn it off and on again "
    "(a key's role never changes: create a new key instead).",
)
def update_key(
    key: Annotated[str, typer.Argument(help=_KEY_HELP)],
    name: Annotated[str | None, typer.Option("--name", help="A new label for the key.")] = None,
    rate_limit: Annotated[
        int | None,
        typer.Option(
            "--rate-limit",
            min=1,
            help="Cap the key at N requests per hour (below its plan's limit, never above).",
        ),
    ] = None,
    clear_rate_limit: Annotated[
        bool,
        typer.Option(
            "--clear-rate-limit", help="Remove the key's cap: it follows its plan's limit again."
        ),
    ] = False,
    active: Annotated[
        bool | None,
        typer.Option("--active/--inactive", help="Turn the key on, or off without revoking it."),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if rate_limit is not None and clear_rate_limit:
        raise typer.BadParameter(
            "pass --rate-limit or --clear-rate-limit, not both", param_hint="--clear-rate-limit"
        )
    if name is None and rate_limit is None and not clear_rate_limit and active is None:
        raise typer.BadParameter(
            "nothing to update: pass at least one of "
            "--name/--rate-limit/--clear-rate-limit/--active/--inactive"
        )
    client = get_client(api_key)
    target = _find_key(client.api_keys.list(), key)
    if active is False:
        _warn_if_own(
            api_key,
            target,
            "every later command with it is refused until another admin key or the app turns "
            "it back on.",
        )
    updated = client.api_keys.update(
        target.id,
        name=name,
        rate_limit=rate_limit,
        clear_rate_limit=clear_rate_limit,
        is_active=active,
    )
    print_json(updated.model_dump(mode="json", exclude_none=True))


@app.command(
    "revoke",
    help="Revoke an API key for good (by id, prefix or name): it stops working at once.",
)
def revoke_key(
    key: Annotated[str, typer.Argument(help=_KEY_HELP)],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    target = _find_key(client.api_keys.list(), key)
    own = _warn_if_own(api_key, target, "every later command with it will be refused.")
    if not yes and not typer.confirm(
        f"Revoke API key {target.name!r} ({target.key_prefix}..., {target.role})? "
        "This cannot be undone."
    ):
        raise typer.Abort()
    client.api_keys.revoke(target.id)
    print_json(
        {"revoked": target.id, "name": target.name, "key_prefix": target.key_prefix, "own_key": own}
    )
