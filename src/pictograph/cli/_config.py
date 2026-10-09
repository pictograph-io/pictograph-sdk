"""CLI config file - ``~/.pictograph/config.toml`` resolution.

Resolution order for the API key (highest wins):
1. ``--api-key`` flag on a command.
2. ``PICTOGRAPH_API_KEY`` environment variable.
3. ``~/.pictograph/config.toml`` ``[default].api_key``.
4. Nothing → :class:`pictograph.exceptions.ConfigurationError` from Client.

The base URL follows the same precedence (``--base-url`` > ``PICTOGRAPH_BASE_URL``
env > ``config.toml`` ``[default].base_url`` > the production default).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# tomllib is stdlib in 3.11+; tomli is a back-compat dep for 3.10.
# Resolution: pyproject.toml's [[tool.mypy.overrides]] for "tomli"
# silences the missing-import error on CI runtimes where tomli isn't
# installed (Python ≥3.11 - the cli-extras pin is conditional).
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

CONFIG_DIR = Path.home() / ".pictograph"
CONFIG_PATH = CONFIG_DIR / "config.toml"


@dataclass(frozen=True)
class CliConfig:
    """Resolved config - written by ``pictograph login``, read on every command."""

    api_key: str | None = None
    base_url: str | None = None


def load_config() -> CliConfig:
    """Read ``~/.pictograph/config.toml``. Returns empty config when missing."""
    if not CONFIG_PATH.is_file():
        return CliConfig()
    with CONFIG_PATH.open("rb") as fh:
        data = tomllib.load(fh)
    default = data.get("default", {})
    return CliConfig(
        api_key=default.get("api_key"),
        base_url=default.get("base_url"),
    )


def write_config(*, api_key: str, base_url: str | None = None) -> Path:
    """Write the API key (and optional base_url) to ``~/.pictograph/config.toml``.

    Creates the directory with mode 0700 and the file with mode 0600.
    Overwrites an existing file silently - interactive ``login`` is the
    expected caller, and the prior value is irrelevant after rotation.

    The file is created with owner-only permissions **before** the API key is
    written to it, so the secret is never momentarily world/group-readable. The
    earlier ``write_text()``-then-``chmod()`` sequence left a TOCTOU window (a
    new file lands at the umask default - typically 0644 - until the follow-up
    chmod) and never repaired loose permissions on a pre-existing file.
    """
    CONFIG_DIR.mkdir(mode=0o700, exist_ok=True)
    lines = ["[default]", f'api_key = "{api_key}"']
    if base_url:
        lines.append(f'base_url = "{base_url}"')
    body = "\n".join(lines) + "\n"

    # os.open's mode is umask-masked on creation and ignored entirely for an
    # existing file, so harden the open fd with fchmod (POSIX) before writing -
    # this both closes the umask gap on new files and repairs loose perms on an
    # existing one. Operating on the fd (not the path) is itself TOCTOU-safe.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(CONFIG_PATH, flags, 0o600)
    try:
        if hasattr(os, "fchmod"):  # POSIX only; chmod below covers Windows
            os.fchmod(fd, 0o600)
    except OSError:
        os.close(fd)
        raise
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(body)
    # Belt-and-suspenders for platforms without fchmod (Windows toggles the
    # read-only bit here); a no-op repeat of the fd perms on POSIX.
    CONFIG_PATH.chmod(0o600)
    return CONFIG_PATH


def resolve_api_key(override: str | None = None) -> str | None:
    """Apply the documented resolution order; returns None when nothing found."""
    if override:
        return override
    env = os.environ.get("PICTOGRAPH_API_KEY")
    if env:
        return env
    return load_config().api_key


def resolve_base_url(override: str | None = None) -> str | None:
    """Resolve the API base URL with the same precedence as the API key.

    ``--base-url`` flag > ``PICTOGRAPH_BASE_URL`` env > ``config.toml``
    ``[default].base_url`` > ``None`` (Client falls back to the production
    default). Mirrors :func:`resolve_api_key` so ``pictograph login
    --base-url <staging>`` actually points subsequent commands at staging
    instead of being silently ignored.
    """
    if override:
        return override
    env = os.environ.get("PICTOGRAPH_BASE_URL")
    if env:
        return env
    return load_config().base_url
