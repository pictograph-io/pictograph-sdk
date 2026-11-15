"""Bind every SDK call in README.md to the real API.

The landing docs have thirteen gates checking every sample against the source.
README.md, which is the FIRST thing anyone who finds this project reads, had
none. It showed for months:

  * ``client.auto_annotate.batch(classes=[{"name": ..., "type": ...}])`` - the
    backend forbids extras and requires ``output_type``, so that request 422s;
  * ``for a in result.annotations: print(a.name)`` - predictions come back as
    dicts, so that is an ``AttributeError`` on the very first example.

Both survived a full green suite and a clean ``mypy --strict``, because nothing
read the README. This is that reader.

For every ``python`` block it checks that each ``client.<resource>.<method>(...)``
names a real resource and method, and that every keyword argument exists in that
method's signature. It deliberately does NOT execute anything - the samples spend
real credits.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from typing import Any

import pytest

from pictograph import AsyncClient, Client

README = Path(__file__).resolve().parents[2] / "README.md"

_PY_BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def _blocks() -> list[str]:
    return _PY_BLOCK.findall(README.read_text(encoding="utf-8"))


def _resolve(root: Any, dotted: list[str]) -> Any:
    """Walk `client.images.upload_from_directory` down to the bound method."""
    current = root
    for part in dotted:
        current = getattr(current, part)
    return current


def _calls(source: str) -> list[tuple[list[str], list[str], int]]:
    """Every `client.a.b(...)` call: its attribute path, kwarg names, and line."""
    try:
        tree = ast.parse(source)
    except SyntaxError:  # a deliberately partial snippet
        return []
    out: list[tuple[list[str], list[str], int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        parts: list[str] = []
        cursor: Any = node.func
        while isinstance(cursor, ast.Attribute):
            parts.append(cursor.attr)
            cursor = cursor.value
        # Unwrap `await client...` and plain `client...`
        if isinstance(cursor, ast.Name) and cursor.id == "client" and parts:
            out.append(
                (list(reversed(parts)), [k.arg for k in node.keywords if k.arg], node.lineno)
            )
    return out


def test_readme_has_python_examples() -> None:
    """Guard the guard: a README that stopped containing examples must not pass."""
    blocks = _blocks()
    assert len(blocks) >= 3, f"only {len(blocks)} python blocks found in README.md"
    assert any(_calls(b) for b in blocks), "no client.* calls found - is the regex still right?"


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_readme_calls_resolve(is_async: bool) -> None:
    """Every client call in the README exists, with the kwargs it is given."""
    client: Any = AsyncClient(api_key="pk_live_x") if is_async else Client(api_key="pk_live_x")
    problems: list[str] = []

    for block in _blocks():
        # Async blocks address AsyncClient; sync blocks address Client. Route each
        # block to the right one instead of checking both against both.
        block_is_async = "AsyncClient" in block or "await " in block
        if block_is_async != is_async:
            continue
        for path, kwargs, lineno in _calls(block):
            try:
                method = _resolve(client, path)
            except AttributeError:
                problems.append(f"line {lineno}: client.{'.'.join(path)} does not exist")
                continue
            if not callable(method):
                problems.append(f"line {lineno}: client.{'.'.join(path)} is not callable")
                continue
            try:
                sig = inspect.signature(method)
            except (TypeError, ValueError):  # pragma: no cover - builtins
                continue
            accepted = set(sig.parameters)
            takes_kwargs = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            for name in kwargs:
                if name not in accepted and not takes_kwargs:
                    problems.append(
                        f"line {lineno}: client.{'.'.join(path)}() got `{name}=`, "
                        f"which it does not accept. Valid: {sorted(accepted - {'self'})}"
                    )

    assert not problems, "README.md documents calls the SDK does not support:\n  " + "\n  ".join(
        problems
    )
