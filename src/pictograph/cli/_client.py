"""Lazy-resolved Client for CLI commands.

Centralised so every command shares the same auth resolution + error
handling path. The Client is built lazily so ``--help`` doesn't trigger
a network connect or env-var lookup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pictograph.cli._config import (
    DEFAULT_PROFILE,
    UnknownProfileError,
    active_profile,
    resolve_api_key,
    resolve_base_url,
)
from pictograph.cli._errors import CliError, ExitCode

if TYPE_CHECKING:
    from pictograph import Client


def get_client(api_key: str | None = None, base_url: str | None = None) -> Client:
    """Build and return a Client.

    Raises:
        CliError: No key is configured (exit code 2, ``no_api_key``), or the
            ``--profile`` flag / ``PICTOGRAPH_PROFILE`` names a profile the config file
            lacks (exit code 2, ``unknown_profile``).
    """
    from pictograph import Client

    try:
        resolved = resolve_api_key(api_key)
        # Resolve base_url from flag > env > config (mirrors the key); without this
        # a `login --base-url <staging>` stored in config.toml was silently ignored
        # and every command kept hitting the production default.
        url = resolve_base_url(base_url)
    except UnknownProfileError as exc:
        raise CliError(
            str(exc),
            exit_code=ExitCode.AUTH,
            code="unknown_profile",
            fix=f"Create it with `pictograph login --profile {exc.name}`, or pick one "
            "`pictograph profiles list` shows.",
            details={"profile": exc.name, "source": exc.source, "profiles": exc.known},
        ) from None
    if not resolved:
        name, source = active_profile()
        where = "" if name == DEFAULT_PROFILE else f" Profile {name!r} (from {source}) has none."
        raise CliError(
            "No API key. Run `pictograph login`, set PICTOGRAPH_API_KEY, or pass --api-key."
            + where,
            exit_code=ExitCode.AUTH,
            code="no_api_key",
        )
    return Client(api_key=resolved, base_url=url)
