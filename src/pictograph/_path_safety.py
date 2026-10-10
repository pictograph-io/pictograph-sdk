"""Turning server-supplied identifiers into safe filesystem names.

An id, a filename or a version string arriving from the API is DATA, not a path
component. Interpolating one straight into a path lets a malicious or compromised
server steer a write: ``../`` escapes the cache directory, a leading ``/`` makes
it absolute, and on Windows a name like ``CON`` or a trailing dot resolves
somewhere unintended.

These ids are normally well-formed UUIDs, which is exactly why this is easy to
skip and easy to get wrong later: the guard has to be at the point where the path
is BUILT, not at the point where the value is validated, because nothing
validates it today.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

#: Anything outside this set is replaced. Deliberately strict: identifiers we
#: build paths from are UUIDs, timestamps and short slugs, all of which survive.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")

#: Windows reserves these regardless of extension.
_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def safe_path_component(value: object, *, fallback: str = "unnamed") -> str:
    """Reduce a server-supplied value to ONE safe filesystem path component.

    Never returns an empty string, a path separator, a parent reference, or a
    Windows reserved name. The result is a single component: it can be joined to
    a directory without being able to leave it.
    """
    text = _UNSAFE.sub("_", str(value))
    text = text.strip(". ")  # a trailing dot or space is stripped by Windows
    if not text or set(text) <= {"."}:
        return fallback
    if text.split(".")[0].upper() in _RESERVED:
        text = f"_{text}"
    return text[:128]


def safe_download_name(value: object, *, fallback: str = "unnamed") -> str:
    """Reduce a server-supplied FILENAME to a safe single component.

    Distinct from :func:`safe_path_component`, which rewrites every character
    outside ``[A-Za-z0-9._-]``. That is right for ids and version strings, but
    wrong for a user's own filenames: it would rewrite ``my photo (1).jpg`` to
    ``my_photo__1_.jpg`` and silently rename files people recognise.

    So this strips the DIRECTORY part instead of rewriting characters. An
    absolute name (``/etc/cron.d/pwn``, which would otherwise discard the output
    directory entirely) and a relative escape (``../../../.bashrc``) both reduce
    to their final component, while every legitimate name survives byte for byte.
    Backslashes are folded first so a Windows-style separator cannot slip past a
    POSIX-only split.
    """
    name = PurePosixPath(str(value).replace("\\", "/")).name
    return name if name and name not in {".", ".."} else fallback


__all__ = ["safe_download_name", "safe_path_component"]
