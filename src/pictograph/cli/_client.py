"""Lazy-resolved Client for CLI commands.

Centralised so every command shares the same auth resolution + error
handling path. The Client is built lazily so ``--help`` doesn't trigger
a network connect or env-var lookup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pictograph.cli._config import resolve_api_key, resolve_base_url
from pictograph.cli._errors import CliError, ExitCode

if TYPE_CHECKING:
    from pictograph import Client


def get_client(api_key: str | None = None, base_url: str | None = None) -> Client:
    """Build and return a Client.

    Raises:
        CliError: No key is configured (exit code 2, ``no_api_key``).
    """
    from pictograph import Client

    resolved = resolve_api_key(api_key)
    if not resolved:
        raise CliError(
            "No API key. Run `pictograph login`, set PICTOGRAPH_API_KEY, or pass --api-key.",
            exit_code=ExitCode.AUTH,
            code="no_api_key",
        )
    # Resolve base_url from flag > env > config (mirrors the key); without this
    # a `login --base-url <staging>` stored in config.toml was silently ignored
    # and every command kept hitting the production default.
    return Client(api_key=resolved, base_url=resolve_base_url(base_url))
