"""Keep prototype residue out of the vendored inference wrappers.

These wrappers were lifted from prototype pipeline code, and they arrived with
its furniture attached. Found and removed on 2026-08-06, having survived at
least one previous round of "clean this up":

  * `PytorchClassifierConfig`, defaulting `model_path` to one developer's local
    file and `class_map` to a nine-class candy dataset;
  * `RFDETRSegConfig`, defaulting `class_map` to a customer's vehicle/windshield
    classes;
  * `RFDETRConfig`, `Detection`, `Metrics`, `SemanticSegmentationModel`,
    `process_detections`, `process_instance_segmentations`, `InstanceSegmentation`
    - none referenced by anything;
  * module docstrings whose usage example imports `from data_objects`, a module
    that exists only in the prototype workspaces and not in any shipped package.

None of it was reachable, so no test failed and no gate complained. It shipped to
PyPI and then to a public GitHub repo. A reviewer noticing it twice is not a
control; this is.

Two rules, both mechanical:

1. NO DEAD PUBLIC SYMBOL. Every public top-level class/function in a wrapper must
   be exported, referenced elsewhere in the SDK, or used inside the wrappers.
   References are collected via AST, not text search - `Detection` and `Metrics`
   both LOOK used if you grep, because those words appear in ordinary prose.
2. NO PROTOTYPE FURNITURE. No `./models/...` literal, no import from the
   prototype `data_objects` module.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SDK_SRC = Path(__file__).resolve().parents[2] / "src" / "pictograph"
WRAPPERS = SDK_SRC / "inference" / "_wrappers"

#: Local-path and prototype-import shapes that must never reappear.
_LOCAL_MODEL_PATH = re.compile(r"""["']\.{0,2}/?models/[^"']*\.(onnx|pth|engine|pte)["']""")
_PROTOTYPE_IMPORT = re.compile(r"\bfrom\s+data_objects\s+import\b")


#: These wrappers are VENDORED - the same files also ship inside the serving
#: image, and the two consumers do not use an identical subset. A symbol only the
#: serving side calls is live, but nothing in this repository can prove it, so it
#: is named here deliberately rather than weakening the rule for everyone.
#:
#: Adding a name here is a claim you must be able to back: point at the caller.
#:   keypoint_schema_from_config - imported by the deployment service to recover a
#:   model's keypoint schema from its config.json before serving pose predictions.
CONSUMED_BY_THE_SERVING_IMAGE_ONLY = frozenset({"keypoint_schema_from_config"})


def _wrapper_modules() -> list[Path]:
    return sorted(p for p in WRAPPERS.glob("*.py") if p.name != "__init__.py")


def _public_definitions() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in _wrapper_modules():
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and not node.name.startswith("_"):
                out[node.name] = path
    return out


def _identifiers_used(root: Path, *, skip: Path | None = None) -> set[str]:
    """Every identifier referenced in Python source under `root`, via AST.

    Deliberately NOT a text search: `Detection` appears in docstrings and comments
    all over this codebase, so grep reports it live when it is dead.
    """
    used: set[str] = set()
    for path in root.rglob("*.py"):
        if skip is not None and skip in path.parents:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)
            elif isinstance(node, ast.alias):
                used.add(node.name.split(".")[-1])
                if node.asname:
                    used.add(node.asname)
    return used


def test_the_gate_can_see() -> None:
    """Guard the guard: if this stops finding modules, everything below passes."""
    modules = _wrapper_modules()
    assert len(modules) >= 8, f"only {len(modules)} wrapper modules found"
    assert "YOLOXDetector" in _public_definitions(), "definition scan is not working"


def test_no_dead_public_symbols_in_wrappers() -> None:
    defined = _public_definitions()

    # Used anywhere in the SDK outside the wrapper package...
    used = _identifiers_used(SDK_SRC, skip=WRAPPERS)
    # ...or inside the wrappers themselves (helpers called by the detectors).
    for path in [*_wrapper_modules(), WRAPPERS / "__init__.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined_here = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)
            elif isinstance(node, ast.alias):
                used.add(node.name.split(".")[-1])
            # __all__ entries are strings, not Names.
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and (node.value in defined or node.value in defined_here)
            ):
                used.add(node.value)

    dead = sorted(
        f"{path.name}::{name}"
        for name, path in defined.items()
        if name not in used and name not in CONSUMED_BY_THE_SERVING_IMAGE_ONLY
    )
    assert not dead, (
        "Dead public symbols in the inference wrappers. Nothing references these, "
        "so nothing will notice when they rot:\n  " + "\n  ".join(dead)
    )


@pytest.mark.parametrize("path", _wrapper_modules(), ids=lambda p: p.name)
def test_no_prototype_furniture(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    local_paths = _LOCAL_MODEL_PATH.findall(text)
    assert not local_paths, (
        f"{path.name} hard-codes a local model file. Weights arrive from the "
        f"caller, never from a path baked into a wrapper."
    )

    assert not _PROTOTYPE_IMPORT.search(text), (
        f"{path.name} imports from `data_objects`, which exists only in the "
        f"prototype workspaces. A usage example that cannot run is worse than none."
    )
