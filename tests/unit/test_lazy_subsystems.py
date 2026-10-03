"""`import pictograph` must not build the async client or the agent registry.

The saving is modest and honestly measured - ~8ms of ~201ms, because
nearly all of aio/agents' cost is the models, resources and httpx the sync
client pulls in regardless. The structural property is the durable part: a
script that only ever uses the sync Client should not construct an httpx async
transport and a 37-tool registry to do it.

Pinned because an eager `from pictograph.aio import AsyncClient` at the top of
__init__ is exactly the kind of line that gets added back for convenience.
"""

from __future__ import annotations

import subprocess
import sys


def _in_fresh_interpreter(code: str) -> str:
    """Run in a NEW process - `import pictograph` in THIS one already happened,
    so an in-process sys.modules check would pass no matter what."""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def test_importing_pictograph_does_not_load_aio_or_agents() -> None:
    loaded = _in_fresh_interpreter(
        "import sys, pictograph; "
        "print([m for m in ('pictograph.aio', 'pictograph.agents') if m in sys.modules])"
    )
    assert loaded == "[]", f"eagerly imported: {loaded}"


def test_the_lazy_attributes_still_resolve() -> None:
    """The public API is unchanged - these are re-exported from the package
    root and must keep working, just on first use."""
    out = _in_fresh_interpreter(
        "import pictograph; "
        "print(pictograph.AsyncClient.__name__, pictograph.Toolkit.__name__, len(pictograph.REGISTRY))"
    )
    assert out == "AsyncClient Toolkit 37"


def test_touching_a_lazy_attribute_loads_its_module() -> None:
    """Anti-vacuity twin for the first test: proves the module WOULD show up in
    sys.modules if it were loaded, so "[]" there means not-loaded rather than
    a check that can never fail."""
    out = _in_fresh_interpreter(
        "import sys, pictograph; pictograph.AsyncClient; print('pictograph.aio' in sys.modules)"
    )
    assert out == "True"


def test_a_direct_submodule_import_is_unaffected() -> None:
    out = _in_fresh_interpreter(
        "from pictograph.aio import AsyncClient; print(AsyncClient.__name__)"
    )
    assert out == "AsyncClient"


def test_an_unknown_attribute_still_raises_attribute_error() -> None:
    out = _in_fresh_interpreter(
        "import pictograph\n"
        "try:\n"
        "    pictograph.does_not_exist\n"
        "except AttributeError as e:\n"
        "    print('AttributeError')\n"
    )
    assert out == "AttributeError"
