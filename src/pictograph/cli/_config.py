"""CLI config file - ``~/.pictograph/config.toml`` resolution, one table per PROFILE.

An API key belongs to one organization, so an agent working for several keeps one profile
per key: ``pictograph login --profile research`` writes a ``[research]`` table beside
``[default]``, and ``pictograph --profile research <command>`` (or
``PICTOGRAPH_PROFILE=research``) runs as it.

Resolution order for the API key (highest wins):
1. ``--api-key`` flag on a command.
2. the ``--profile`` flag (the root option): that profile's key.
3. ``PICTOGRAPH_API_KEY`` environment variable.
4. ``PICTOGRAPH_PROFILE`` environment variable: that profile's key.
5. ``~/.pictograph/config.toml`` ``[default].api_key``.
6. Nothing → :class:`pictograph.exceptions.ConfigurationError` from Client.

A profile named by the flag or the environment must exist (an unknown name is refused, never
read as "no key"). The base URL follows the same order (``--base-url`` > the ``--profile``
flag's profile > ``PICTOGRAPH_BASE_URL`` env > the ``PICTOGRAPH_PROFILE`` profile >
``[default].base_url``); a profile with no base URL means the production default. The SDK's
``Client`` reads none of this - it takes ``api_key=`` or ``PICTOGRAPH_API_KEY``
(``pictograph._internal.config``).
"""

from __future__ import annotations

import os
import re
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

DEFAULT_PROFILE = "default"
PROFILE_ENV = "PICTOGRAPH_PROFILE"
# A TOML bare key: a "." would make `[a.b]` a NESTED table, so it is refused.
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_FIELDS = ("api_key", "base_url")

# The profile the root `--profile` flag chose for this invocation (set by the root
# callback before any command runs; None = the flag was not given).
_flag_profile: str | None = None


@dataclass(frozen=True)
class CliConfig:
    """One profile's settings - written by ``pictograph login``, read on every command."""

    api_key: str | None = None
    base_url: str | None = None


class UnknownProfileError(LookupError):
    """A profile named by ``--profile`` / ``PICTOGRAPH_PROFILE`` that the config lacks."""

    def __init__(self, name: str, source: str, known: list[str]) -> None:
        self.name = name
        self.source = source
        self.known = known
        listed = ", ".join(known) if known else "none yet"
        super().__init__(
            f"No profile {name!r} (from {source}) in {CONFIG_PATH}. Profiles: {listed}."
        )


def check_profile_name(name: str) -> str:
    """``name`` when it can be a profile (letters, digits, ``-`` and ``_``), else ValueError."""
    if not _PROFILE_NAME.match(name):
        raise ValueError(
            f"{name!r} cannot be a profile name: use letters, digits, '-' and '_' "
            "(up to 64, starting with a letter or digit)."
        )
    return name


def set_flag_profile(name: str | None) -> None:
    """Record the root ``--profile`` flag for this invocation (``None`` clears it)."""
    global _flag_profile
    _flag_profile = check_profile_name(name) if name else None


def _read_profiles() -> dict[str, CliConfig]:
    if not CONFIG_PATH.is_file():
        return {}
    with CONFIG_PATH.open("rb") as fh:
        data = tomllib.load(fh)
    out: dict[str, CliConfig] = {}
    for name, table in data.items():
        if isinstance(table, dict):
            values = {k: table[k] for k in _FIELDS if isinstance(table.get(k), str)}
            out[str(name)] = CliConfig(**values)
    return out


def profile_names() -> list[str]:
    """The profiles in the config file, ``default`` first."""
    names = list(_read_profiles())
    return sorted(names, key=lambda n: (n != DEFAULT_PROFILE, n))


def load_config(profile: str | None = None) -> CliConfig:
    """One profile of ``~/.pictograph/config.toml`` (``default`` unless named); empty when
    the file or the profile is missing."""
    return _read_profiles().get(profile or DEFAULT_PROFILE, CliConfig())


def active_profile() -> tuple[str, str]:
    """``(name, source)`` of the profile a command runs as: the ``--profile`` flag, then
    ``PICTOGRAPH_PROFILE``, then ``default``. (An explicit key - ``--api-key`` or
    ``PICTOGRAPH_API_KEY`` - can still win over a profile from the environment; see
    :func:`resolve_api_key`.)"""
    if _flag_profile:
        return _flag_profile, "--profile"
    env = os.environ.get(PROFILE_ENV)
    if env:
        return env, PROFILE_ENV
    return DEFAULT_PROFILE, "default"


def _named(name: str, source: str) -> CliConfig:
    profiles = _read_profiles()
    if name not in profiles:
        raise UnknownProfileError(name, source, profile_names())
    return profiles[name]


def _toml_string(value: str) -> str:
    if any(ch in value for ch in "\n\r\t\x00"):
        raise ValueError("a config value cannot contain control characters")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_config(
    *, api_key: str, base_url: str | None = None, profile: str = DEFAULT_PROFILE
) -> Path:
    """Write ``profile``'s API key (and optional base_url) to ``~/.pictograph/config.toml``,
    keeping every other profile as it was.

    Creates the directory with mode 0700 and the file with mode 0600. A profile written
    again replaces its own values (a ``login`` without ``--base-url`` returns it to the
    production default).

    The file is created with owner-only permissions **before** the API key is
    written to it, so the secret is never momentarily world/group-readable. The
    earlier ``write_text()``-then-``chmod()`` sequence left a TOCTOU window (a
    new file lands at the umask default - typically 0644 - until the follow-up
    chmod) and never repaired loose permissions on a pre-existing file.
    """
    check_profile_name(profile)
    profiles = _read_profiles()
    profiles[profile] = CliConfig(api_key=api_key, base_url=base_url or None)
    blocks = []
    for name in sorted(profiles, key=lambda n: (n != DEFAULT_PROFILE, n)):
        entry = profiles[name]
        lines = [f"[{name}]"]
        if entry.api_key:
            lines.append(f"api_key = {_toml_string(entry.api_key)}")
        if entry.base_url:
            lines.append(f"base_url = {_toml_string(entry.base_url)}")
        blocks.append("\n".join(lines))
    body = "\n\n".join(blocks) + "\n"

    CONFIG_DIR.mkdir(mode=0o700, exist_ok=True)
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
    """Apply the documented resolution order; returns None when nothing found.

    Raises:
        UnknownProfileError: the ``--profile`` flag or ``PICTOGRAPH_PROFILE`` names a
            profile the config file lacks.
    """
    if override:
        return override
    if _flag_profile:
        return _named(_flag_profile, "--profile").api_key
    env = os.environ.get("PICTOGRAPH_API_KEY")
    if env:
        return env
    env_profile = os.environ.get(PROFILE_ENV)
    if env_profile:
        return _named(env_profile, PROFILE_ENV).api_key
    return load_config().api_key


def resolve_base_url(override: str | None = None) -> str | None:
    """Resolve the API base URL with the same precedence as the API key.

    ``--base-url`` flag > the ``--profile`` flag's profile > ``PICTOGRAPH_BASE_URL`` env >
    the ``PICTOGRAPH_PROFILE`` profile > ``config.toml`` ``[default].base_url`` > ``None``
    (Client falls back to the production default, as it does for a profile with no base
    URL). Mirrors :func:`resolve_api_key` so ``pictograph login --base-url <staging>``
    actually points subsequent commands at staging instead of being silently ignored.
    """
    if override:
        return override
    if _flag_profile:
        # The flag picks everything: its profile's URL or the production default, never
        # an ambient PICTOGRAPH_BASE_URL (that could send this profile's key elsewhere).
        return _named(_flag_profile, "--profile").base_url
    env = os.environ.get("PICTOGRAPH_BASE_URL")
    if env:
        return env
    env_profile = os.environ.get(PROFILE_ENV)
    if env_profile:
        return _named(env_profile, PROFILE_ENV).base_url
    return load_config().base_url
