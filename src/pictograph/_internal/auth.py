"""API-key resolution for the Pictograph SDK.

The single :func:`resolve_api_key` entry point implements the resolution
order documented on :class:`pictograph._internal.config.ClientConfig`. It
raises :class:`pictograph.exceptions.ConfigurationError` with an actionable
``.fix`` message when no key is found - designed to read well in tracebacks
*and* in LLM-agent error-handling logic.
"""

from __future__ import annotations

import os

from pictograph.exceptions import ConfigurationError

ENV_VAR_NAME = "PICTOGRAPH_API_KEY"
"""Canonical environment variable name read by :func:`resolve_api_key`."""


def resolve_api_key(explicit: str | None = None) -> str:
    """Resolve the API key from explicit input or the environment.

    Args:
        explicit: A key passed directly (e.g. ``Client(api_key=...)``). When
            non-empty, this wins over the environment.

    Returns:
        The resolved API key string.

    Raises:
        ConfigurationError: If no key is available from any source. The
            error's ``fix`` field tells the caller exactly what to do.
    """
    if explicit:
        candidate = explicit.strip()
        if candidate:
            return candidate

    env_value = os.environ.get(ENV_VAR_NAME)
    if env_value:
        candidate = env_value.strip()
        if candidate:
            return candidate

    raise ConfigurationError(
        "No Pictograph API key found.",
        fix=(
            f"Pass api_key=... to Client(), or set the {ENV_VAR_NAME} "
            "environment variable. Generate a key at Settings → API Keys."
        ),
    )
