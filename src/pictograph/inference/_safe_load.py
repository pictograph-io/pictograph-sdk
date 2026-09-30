"""The one place a PyTorch checkpoint is read.

``torch.load(weights_only=False)`` executes the file's own ``__reduce__``, so
loading an untrusted checkpoint is equivalent to running it. Models can be FORKED
between organizations, which means a ``.pth`` reaching this code is not
necessarily one the caller produced.

Every checkpoint read in the SDK, including the vendored RF-DETR code, goes
through :func:`safe_torch_load`. There were five separate ``weights_only=False``
call sites before this module existed and only one of them was even aware of the
problem, which is exactly why this is centralised: a rule enforced in one place
holds, a rule repeated in five places drifts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

_LOG = logging.getLogger("pictograph.inference")

#: Inert non-tensor types our checkpoints legitimately carry. RF-DETR stores its
#: training configuration as an ``argparse.Namespace``, which the safe unpickler
#: refuses by default even though it cannot execute anything.
#:
#: Allowlisting SPECIFIC types is categorically different from disabling the safe
#: unpickler: these classes have no ``__reduce__`` that runs code, whereas
#: ``weights_only=False`` will run ``__reduce__`` on ANY class the file names.
#: Only add a type here after confirming it cannot execute on construction.
SAFE_CHECKPOINT_GLOBALS: tuple[str, ...] = ("argparse.Namespace",)

_allowlisted = False


def _register_safe_globals() -> None:
    """Register the inert types once per process. Cheap and idempotent."""
    global _allowlisted
    if _allowlisted:
        return
    import torch

    add = getattr(torch.serialization, "add_safe_globals", None)
    if add is None:  # torch < 2.2 predates the allowlist API
        _allowlisted = True
        return

    resolved: list[Any] = []
    for dotted in SAFE_CHECKPOINT_GLOBALS:
        module_name, _, attr = dotted.rpartition(".")
        try:
            resolved.append(getattr(__import__(module_name, fromlist=[attr]), attr))
        except Exception as exc:
            # Not allowlisting a type is the SAFE direction: the load simply
            # refuses rather than executing anything. Worth a debug line so a
            # checkpoint that suddenly stops loading is traceable.
            _LOG.debug("Could not allowlist %s for safe loading: %s", dotted, exc)
            continue
    if resolved:
        add(resolved)
    _allowlisted = True


def safe_torch_load(
    path: str | Path,
    *,
    map_location: str = "cpu",
    allow_unsafe_pickle: bool = False,
) -> Any:
    """Read a checkpoint without executing code embedded in it.

    Raises :class:`~pictograph.exceptions.UnsafeCheckpointError` when the file
    cannot be read on the safe path, rather than silently falling back to the
    loader that would execute it. Pass ``allow_unsafe_pickle=True`` only for a
    file you produced yourself.
    """
    import torch

    from pictograph.exceptions import UnsafeCheckpointError

    _register_safe_globals()
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except OSError:
        # A missing, unreadable or not-a-file path is an I/O problem, NOT a
        # refusal to execute code. Wrapping it would tell the caller their
        # checkpoint is dangerous when it simply is not there, and would hide the
        # real errno. Retrying it under the unsafe loader cannot help either.
        raise
    except Exception as exc:
        if not allow_unsafe_pickle:
            raise UnsafeCheckpointError(path, exc) from exc
        _LOG.warning(
            "Loading %s with the full pickle unpickler because "
            "allow_unsafe_pickle=True. This executes code embedded in the file. "
            "Only do this for checkpoints you produced yourself.",
            path,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


__all__ = ["SAFE_CHECKPOINT_GLOBALS", "safe_torch_load"]
