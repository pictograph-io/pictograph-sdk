"""Every call the SDK makes to a resource method names a method the resource has.

A renamed resource method leaves its old name dead at every internal call site, and nothing
imports-fails: SDK 1.69.176 renamed ``models.get_by_name`` to ``models.get``, and
``get_model`` kept calling the old name through 1.69.178 (``AttributeError`` on every call).
This reads the SDK's source (and its bundled skill scripts) and checks each
``<client>.<resource>.<method>(...)`` call - and, in the local-inference loader, calls on its
``models`` parameter - against the real sync and async resource classes.
"""

from __future__ import annotations

import ast
from pathlib import Path

from pictograph import Client
from pictograph.aio import AsyncClient

SRC = Path(__file__).resolve().parents[2] / "src" / "pictograph"


def _resources() -> dict[str, set[str]]:
    methods: dict[str, set[str]] = {}
    for client in (
        Client(api_key="pk_live_" + "x" * 32),
        AsyncClient(api_key="pk_live_" + "x" * 32),
    ):
        for name in dir(client):
            obj = getattr(client, name, None)
            if name.startswith("_") or callable(obj):
                continue
            if type(obj).__module__.startswith("pictograph."):
                methods.setdefault(name, set()).update(dir(type(obj)))
    return methods


def _is_client(node: ast.expr) -> bool:
    """A client value: ``client``, ``self._client``, ``async_client`` - a name ending in "client"."""
    if isinstance(node, ast.Name):
        return node.id.lower().endswith("client")
    return isinstance(node, ast.Attribute) and node.attr.lower().endswith("client")


def _calls(path: Path, resources: dict[str, set[str]]) -> list[tuple[int, str, str]]:
    """(line, resource, method) for each call this file makes on a resource object."""
    found = []
    in_loader = "inference" in path.parts
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        owner = node.func.value
        if isinstance(owner, ast.Attribute) and owner.attr in resources and _is_client(owner.value):
            found.append((node.lineno, owner.attr, node.func.attr))  # client.models.get(...)
        elif in_loader and isinstance(owner, ast.Name) and owner.id == "models":
            found.append((node.lineno, "models", node.func.attr))  # the loader's `models` parameter
    return found


def test_every_resource_method_the_sdk_calls_exists() -> None:
    resources = _resources()
    assert {"models", "datasets", "workflows"} <= set(resources)
    seen = 0
    dead = []
    for path in sorted(SRC.rglob("*.py")):
        for line, resource, method in _calls(path, resources):
            seen += 1
            if method not in resources[resource]:
                dead.append(f"{path.relative_to(SRC.parent)}:{line}: {resource}.{method}()")
    assert seen > 50, f"the scan saw only {seen} calls - it is no longer reading the SDK"
    assert not dead, "calls to resource methods that do not exist:\n" + "\n".join(dead)


def test_the_scan_catches_a_dead_method(tmp_path: Path) -> None:
    """The control: the exact call that shipped is reported."""
    loader = tmp_path / "inference" / "loader.py"
    loader.parent.mkdir()
    loader.write_text("def load(models, name):\n    return models.get_by_name(name)\n")
    resources = _resources()
    [(line, resource, method)] = _calls(loader, resources)
    assert (resource, method) == ("models", "get_by_name") and method not in resources["models"]
