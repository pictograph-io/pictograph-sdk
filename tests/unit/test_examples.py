"""Bind the runnable ``examples/`` to the real SDK surface.

The examples ship in the source distribution and the public repo, so a stale
example is a broken first impression - and a prior set had methods that no
longer existed. String-matching cannot catch that: it moves with the code. This
gate parses each example and checks every
``client.<resource>.<method>`` call and every ``from pictograph import ...``
name against a REAL ``Client`` / ``AsyncClient`` instance, so renaming or
removing a method breaks the build here, not in a user's terminal.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from pictograph import AsyncClient, Client

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

# A dummy key: construction is offline, so this never touches the network.
_SYNC = Client(api_key="pk_live_dummy")
_ASYNC = AsyncClient(api_key="pk_live_dummy")


def _example_files() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("*.py"))


def _resource_calls(tree: ast.AST) -> set[tuple[str, str]]:
    """Every ``client.<resource>.<method>`` pair used in the file."""
    pairs: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        inner = node.value
        if (
            isinstance(inner, ast.Attribute)
            and isinstance(inner.value, ast.Name)
            and inner.value.id == "client"
        ):
            pairs.add((inner.attr, node.attr))
    return pairs


def _pictograph_imports(tree: ast.AST) -> set[tuple[str, str]]:
    """Every ``(module, symbol)`` imported from a ``pictograph`` module."""
    pairs: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("pictograph"):
            for alias in node.names:
                pairs.add((node.module or "", alias.name))
    return pairs


def test_examples_directory_is_present() -> None:
    # If examples/ is ever excluded again, this suite must not silently pass.
    assert _example_files(), f"no example scripts found under {EXAMPLES_DIR}"


@pytest.mark.parametrize("path", _example_files(), ids=lambda p: p.name)
def test_example_parses(path: Path) -> None:
    ast.parse(path.read_text(), filename=str(path))


@pytest.mark.parametrize("path", _example_files(), ids=lambda p: p.name)
def test_example_client_calls_exist(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    for resource, method in sorted(_resource_calls(tree)):
        # A call is valid if it exists on the sync OR the async client - the
        # examples use both, and the two surfaces mirror each other.
        sync_ok = hasattr(getattr(_SYNC, resource, None), method)
        async_ok = hasattr(getattr(_ASYNC, resource, None), method)
        assert sync_ok or async_ok, (
            f"{path.name}: client.{resource}.{method}() does not exist on the SDK"
        )


@pytest.mark.parametrize("path", _example_files(), ids=lambda p: p.name)
def test_example_imports_resolve(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    for module_name, symbol in sorted(_pictograph_imports(tree)):
        module = importlib.import_module(module_name)
        assert hasattr(module, symbol), (
            f"{path.name}: `from {module_name} import {symbol}` - not exported"
        )
