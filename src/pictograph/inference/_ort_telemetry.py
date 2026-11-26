"""ONNX Runtime's telemetry stays off for everything the SDK runs.

ONNX Runtime 1.30 and newer report to Microsoft by default (``mobile.events.data.microsoft.com``). Importing the
runtime creates its environment, and that writes a persistent device id and an event store on the machine
(``~/.cache/Microsoft/DeveloperTools/.onnxruntime/`` on Linux, ``~/Library/Application Support/Microsoft/
DeveloperTools/.onnxruntime/`` on macOS) and queues a device event (CPU model, OS, memory, container type); sessions
report their own. Windows builds report through the operating system's event tracing instead. Local inference with
Pictograph sends none of it.

Two switches, both needed:

- ``ORT_DISABLE_TELEMETRY``, read once, when the runtime is IMPORTED. Set after that it changes nothing, and
  ``onnxruntime.disable_telemetry_events()`` called after the import does not undo what the import already wrote
  (measured on Linux, 1.31.0: the store and its device event exist either way). So importing this module sets the
  variable, and :mod:`pictograph.inference` imports it first: Python runs that package's init before any module of
  it, and the modules that import the runtime (the vendored wrappers do, at their top) all live in it.
- ``onnxruntime.disable_telemetry_events()``, the runtime's own off switch: for Windows, and for a process that
  imported the runtime before Pictograph (the variable is too late there). Every SDK entry to the runtime goes
  through :func:`import_onnxruntime`, which calls it.

Importing ``pictograph`` therefore sets ``ORT_DISABLE_TELEMETRY=1`` for the process (and the processes it starts).
Opting in to Microsoft's telemetry is explicit: ``ORT_DISABLE_TELEMETRY=0`` (or ``false`` / ``no`` / ``off``), set
before ``pictograph`` is imported. Nothing else turns it on.
"""

from __future__ import annotations

import logging
import os
from typing import Any

_LOG = logging.getLogger(__name__)

#: The variable ONNX Runtime reads when its environment is created. The runtime parses it as a boolean: ``1`` /
#: ``true`` / ``yes`` switch its telemetry off; ``0`` / ``false`` / blank / anything else leave it on (measured on
#: Linux, 1.31.0).
ENV = "ORT_DISABLE_TELEMETRY"

#: The values that mean the user asked for ONNX Runtime's telemetry (compared case-insensitively, stripped).
OPT_IN_VALUES = frozenset({"0", "false", "no", "off"})

_events_disabled = False


def opted_in() -> bool:
    """Whether the user explicitly asked for ONNX Runtime's telemetry (``ORT_DISABLE_TELEMETRY=0``)."""
    return os.environ.get(ENV, "").strip().lower() in OPT_IN_VALUES


def keep_off() -> None:
    """Set ``ORT_DISABLE_TELEMETRY=1`` unless the user opted in.

    Effective only before ``onnxruntime`` is imported. A value the runtime would read as "on" without being an
    explicit opt-in (blank, ``2``) is replaced too, so the only way telemetry runs is a user asking for it.
    """
    if not opted_in():
        os.environ[ENV] = "1"


def disable_events(ort: Any) -> None:
    """Call the runtime's own off switch, once per process (nothing, for a user who opted in).

    Never raises: switching telemetry off must not be able to break inference.
    """
    global _events_disabled
    if _events_disabled or opted_in():
        return
    switch = getattr(ort, "disable_telemetry_events", None)
    if not callable(switch):
        return
    try:
        switch()
    except Exception as exc:  # an odd build; the variable already covered the import
        _LOG.debug("onnxruntime.disable_telemetry_events() failed: %s", exc)
        return
    _events_disabled = True


def import_onnxruntime() -> Any:
    """``import onnxruntime`` with its telemetry off - the one way SDK code reaches the runtime."""
    keep_off()
    import onnxruntime

    disable_events(onnxruntime)
    return onnxruntime


# Importing this module is what switches the telemetry off: pictograph.inference imports it before any module of
# its own that imports the runtime.
keep_off()
