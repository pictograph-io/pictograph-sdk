"""Every `client.<resource>.<method>(...)` call in `tests/live/` binds for real.

`tests/live/` is 22 modules of the most realistic coverage the SDK has - it is the
only tree that drives the shipped client against the real API. It is also
INVISIBLE to the documented gate: `pytest` skips the whole tree without
`PICTOGRAPH_TEST_KEY`, so a signature change lands green while every live caller
of it is already broken. That is exactly what happened on 2026-07-31 - the
names-not-ids sweep gave 53 methods a new required leading argument and 25 live
callers went stale under a fully green suite, discovered only when someone
happened to run the tree with a key.

CI cannot run the live tree: it needs a real key, bills real inference and mutates
a real org. So the guard has to be static. This is the same trade the landing's
`verify_doc_calls.py` makes for documentation, applied one layer up - a call is
bound against the real signature without being executed, which catches the entire
class of "this caller no longer matches the method" for free, on every run, with
no key.

Two deliberate choices, both learned the hard way in this codebase:

* **A full `bind`, never `bind_partial`.** `bind_partial` tolerates a missing
  required parameter BY DEFINITION, which is precisely the drift the sweep
  created. Both doc gates had this hole and reported OK across all 53 changes.
* **An unparseable live module is a FAILURE, not a skip.** `verify_doc_calls.py`
  shipped with `except SyntaxError: continue` and reported "OK, 271 calls bind"
  while two of the snippets it was checking were not valid Python.

This does NOT replace running the tree. It cannot see wrong VALUES, only wrong
SHAPES. `PICTOGRAPH_TEST_KEY=... pytest tests/live` is still the real thing before
calling an SDK signature change done.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

from pictograph import client as _client_mod

LIVE = Path(__file__).resolve().parents[1] / "live"


def _resource_map() -> dict[str, type]:
    """Map `client.<attr>` -> the resource CLASS, read off `Client.__init__`.

    Derived rather than hand-listed so a new resource is covered the day it is
    wired up, with nothing to remember.
    """
    src = inspect.getsource(_client_mod.Client.__init__)
    out: dict[str, type] = {}
    for m in re.finditer(r"self\.(\w+)\s*=\s*(\w+)\(", src):
        cls = getattr(_client_mod, m.group(2), None)
        if isinstance(cls, type):
            out[m.group(1)] = cls
    return out


RESOURCES = _resource_map()


def _chain(node: ast.Call) -> list[str]:
    parts, f = [], node.func
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return list(reversed(parts))


def _resolve(chain: list[str]):
    """`['client', 'images', 'upload']` -> the unbound `Images.upload`."""
    if len(chain) < 2 or chain[0] != "client":
        return None
    rest = chain[1:]
    if len(rest) >= 2 and rest[0] in RESOURCES:
        obj: object = RESOURCES[rest[0]]
        rest = rest[1:]
    elif len(rest) == 1:
        obj = _client_mod.Client
    else:
        return None
    for step in rest[:-1]:
        obj = getattr(obj, step, None)
        if obj is None:
            return None
    fn = getattr(obj, rest[-1], None)
    return fn if callable(fn) else None


def _live_modules() -> list[Path]:
    return sorted(p for p in LIVE.rglob("*.py") if p.name != "__init__.py")


def test_live_tree_is_present() -> None:
    """Guard the guard: a glob that matches nothing makes every check below vacuous."""
    mods = _live_modules()
    assert len(mods) >= 20, f"expected the live tree, found {len(mods)} modules under {LIVE}"


def test_every_live_call_binds_against_the_shipped_signature() -> None:
    failures: list[str] = []
    bound = 0

    for path in _live_modules():
        rel = path.relative_to(LIVE.parents[1])
        source = path.read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError as e:  # a live module that is not Python is a failure
            failures.append(f"  {rel}:{e.lineno}  NOT VALID PYTHON: {e.msg}")
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            chain = _chain(node)
            if not chain or chain[0] != "client":
                continue

            fn = _resolve(chain)
            if fn is None:
                # A known resource with an unknown method is a caller of something
                # that was renamed or removed - the most likely way this tree rots.
                if len(chain) >= 3 and chain[1] in RESOURCES:
                    failures.append(
                        f"  {rel}:{node.lineno}  {'.'.join(chain)}(...)  NO SUCH METHOD"
                    )
                continue

            # `*args` / `**kwargs` hide the real arity, so binding proves nothing.
            if any(isinstance(a, ast.Starred) for a in node.args) or any(
                k.arg is None for k in node.keywords
            ):
                continue

            try:
                sig = inspect.signature(fn)
            except (ValueError, TypeError):
                continue

            pos = [inspect.Parameter.empty] * len(node.args)
            if inspect.isfunction(fn) and "self" in sig.parameters:
                pos = [inspect.Parameter.empty, *pos]
            kw = {k.arg: inspect.Parameter.empty for k in node.keywords if k.arg}

            try:
                sig.bind(*pos, **kw)
                bound += 1
            except TypeError as e:
                failures.append(f"  {rel}:{node.lineno}  {'.'.join(chain)}(...)  {e}")

    # Report the real signal FIRST. The vacuity floor below is a guard on this
    # check, not a finding about the tree - ordering it first made a genuinely
    # broken tree report "the resolver is not matching" and hid the stale callers
    # it had correctly found.
    if failures:
        pytest.fail(
            f"{len(failures)} live-suite call(s) do not bind against the shipped SDK.\n"
            "These are already broken; running tests/live with a key would fail.\n\n"
            + "\n".join(failures),
            pytrace=False,
        )

    # A floor, not a target: with zero failures, a collapsed `bound` means the
    # resolver stopped matching (a renamed `Client` attribute, a changed
    # `__init__` shape) and the check passed by looking at nothing.
    assert bound >= 150, f"only {bound} live calls resolved - the resolver is not matching"
