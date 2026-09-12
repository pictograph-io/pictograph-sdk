"""A test module may not import an optional dependency at module scope unguarded.

`pytest` imports every `test_*.py` and every `conftest.py` during COLLECTION, before
a single test runs and before any module-scope skip can fire. So a module-scope
`import numpy` in a test file raises during collection on any environment that does
not have numpy - and the documented quality gate is exactly such an environment:
`pip install -e ".[dev,cli,agents,cache,telemetry]"` installs NONE of the
`[torch] / [inference] / [executorch] / [tensorrt]` extras, so numpy, torch, cv2,
onnxruntime and friends are absent under it.

On 2026-08-01 this bit for real: a module-level `import numpy` raised on a base venv
and the collection error took the ENTIRE run down - 2788 tests collected, 0 executed,
a fully green-looking gate that had run nothing. The fix is the `pytest.importorskip`
guard the parity suites already use: it turns a collection ERROR into a visible SKIP.

This gate makes the guard non-optional. For every module pytest imports eagerly
(`test_*.py` + `conftest.py`), a MODULE-SCOPE import of any dependency the documented
gate does not install must be preceded, in module-body order, by
`pytest.importorskip("<that package>")`. The same rule covers a stdlib module newer
than the SDK's floor Python (`tomllib`, added in 3.11, against a 3.10 floor and a CI
matrix that runs 3.10): guard it with a `sys.version_info` conditional, which nests the
import out of module scope. It is the static twin of `test_live_suite_binds`
and `test_examples`: it reads the file and would FAIL on the exact regression, on every
run, with none of the heavy extras installed - because it is pure AST, it needs none of
them to prove they are guarded.

It cannot see a WRONG guard value, only a MISSING one. `pip install -e .` then `pytest`
on a base venv is still the real proof that the tree collects. This keeps that proof
from silently rotting.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - the 3.10 back-compat path, exercised only on 3.10 CI
    import tomli as tomllib

_SDK_ROOT = Path(__file__).resolve().parents[2]
_TESTS = _SDK_ROOT / "tests"

# The extras the DOCUMENTED quality gate installs (CLAUDE.md / pyproject):
#   pip install -e ".[dev,cli,agents,cache,telemetry]"
# Anything in an extra OUTSIDE this set is NOT present under the gate, so a
# module-scope import of it must be importorskip-guarded. If the documented install
# ever changes, change this constant.
_GATE_EXTRAS = frozenset({"dev", "cli", "agents", "cache", "telemetry"})

# Distribution name (PEP 503, hyphenated) -> the top-level import root it provides.
# Only the non-gate optional dists need an entry; a dist that lands in a non-gate
# extra without one is a FAILURE below, so this map cannot silently miss a new one.
_DIST_TO_IMPORT_ROOT: dict[str, str] = {
    "torch": "torch",
    "torchvision": "torchvision",
    "safetensors": "safetensors",
    "onnxruntime": "onnxruntime",
    "onnx": "onnx",
    "opencv-python-headless": "cv2",
    "numpy": "numpy",
    "scikit-image": "skimage",
    "segmentation-models-pytorch": "segmentation_models_pytorch",
    "executorch": "executorch",
    "tensorrt": "tensorrt",
}


def _dist_name(requirement: str) -> str:
    """`"opencv-python-headless>=4.8"` -> `"opencv-python-headless"` (normalized)."""
    m = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement.strip())
    assert m, f"unparseable requirement: {requirement!r}"
    return m.group(0).lower().replace("_", "-")


def _pyproject() -> dict:
    return tomllib.loads((_SDK_ROOT / "pyproject.toml").read_text())


def _non_gate_optional_dists() -> set[str]:
    """Every dist in a non-gate extra, minus base deps and the gate extras' deps."""
    proj = _pyproject()["project"]
    base = {_dist_name(r) for r in proj.get("dependencies", [])}
    extras: dict[str, list[str]] = proj.get("optional-dependencies", {})
    gate = set(base)
    for name in _GATE_EXTRAS:
        gate |= {_dist_name(r) for r in extras.get(name, [])}
    out: set[str] = set()
    for name, reqs in extras.items():
        if name in _GATE_EXTRAS or name == "all":  # "all" only references other extras
            continue
        for r in reqs:
            out.add(_dist_name(r))
    return out - gate - {"pictograph"}


# Stdlib modules that a bare module-scope import breaks collection on for any
# interpreter OLDER than the given version. `tomllib` landed in 3.11; the SDK floor is
# 3.10 and the CI matrix runs 3.10, so a bare `import tomllib` in a collected test
# raises there - the same "silent 0-executed run" failure as a missing extra. The guard
# is a `sys.version_info` conditional (which nests the import out of module-body scope,
# so this top-level walk stops seeing it - see `cli/_config.py`) or `importorskip`.
_STDLIB_MIN_PY: dict[str, tuple[int, int]] = {"tomllib": (3, 11)}


def _python_floor() -> tuple[int, int]:
    """The minimum Python the SDK supports, read off `requires-python`."""
    spec = _pyproject()["project"]["requires-python"]
    m = re.search(r"(\d+)\.(\d+)", spec)
    assert m, f"could not parse requires-python: {spec!r}"
    return (int(m.group(1)), int(m.group(2)))


def _must_guard_roots() -> set[str]:
    roots = {_DIST_TO_IMPORT_ROOT[d] for d in _non_gate_optional_dists()}
    floor = _python_floor()
    roots |= {mod for mod, added in _STDLIB_MIN_PY.items() if floor < added}
    return roots


def _unguarded_module_scope_imports(source: str, must_guard: set[str]) -> list[tuple[int, str]]:
    """Module-scope imports of a must-guard root NOT preceded by its importorskip.

    Walks the module body IN ORDER, tracking which roots have been
    `pytest.importorskip(...)`-guarded so far, so an import that comes before its
    guard (or has no guard) is reported. Returns `(lineno, root)` pairs.
    """
    tree = ast.parse(source)
    guarded: set[str] = set()
    hits: list[tuple[int, str]] = []
    for node in tree.body:  # top-level statements only
        skipped = _importorskip_root(node)
        if skipped is not None:
            guarded.add(skipped)
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in must_guard and root not in guarded:
                    hits.append((node.lineno, root))
        elif isinstance(node, ast.ImportFrom) and not node.level:
            root = (node.module or "").split(".")[0]
            if root in must_guard and root not in guarded:
                hits.append((node.lineno, root))
    return hits


def _importorskip_root(node: ast.stmt) -> str | None:
    """If `node` is a top-level `[x = ]pytest.importorskip("pkg...")`, return `pkg`."""
    # Both a bare call (`pytest.importorskip(...)`) and an assignment
    # (`np = pytest.importorskip(...)`) carry the call on `.value`.
    call = node.value if isinstance(node, (ast.Expr, ast.Assign)) else None
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
        return None
    if call.func.attr != "importorskip" or not call.args:
        return None
    arg0 = call.args[0]
    if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
        return arg0.value.split(".")[0]
    return None


def _eagerly_imported_files() -> list[Path]:
    """Files pytest imports during collection: every `test_*.py` and `conftest.py`."""
    return sorted(
        p for p in _TESTS.rglob("*.py") if p.name == "conftest.py" or p.name.startswith("test_")
    )


# --------------------------------------------------------------------------- tests


def test_every_non_gate_optional_dist_has_a_known_import_root() -> None:
    """Guard the map: a new inference/torch dep cannot slip in unmapped and unchecked."""
    unmapped = sorted(d for d in _non_gate_optional_dists() if d not in _DIST_TO_IMPORT_ROOT)
    assert not unmapped, (
        "pyproject has non-gate optional dist(s) with no import root in "
        f"_DIST_TO_IMPORT_ROOT: {unmapped}. Add each so this gate can see it."
    )


def test_no_eagerly_imported_test_module_imports_an_optional_dep_unguarded() -> None:
    must_guard = _must_guard_roots()
    assert must_guard, "derived an empty must-guard set - the pyproject derivation broke"

    files = _eagerly_imported_files()
    assert len(files) >= 40, f"expected the test tree, scanned only {len(files)} files"

    failures: list[str] = []
    for path in files:
        for lineno, root in _unguarded_module_scope_imports(path.read_text(), must_guard):
            rel = path.relative_to(_SDK_ROOT)
            failures.append(
                f"  {rel}:{lineno}  module-scope `import {root}` with no preceding "
                f'`pytest.importorskip("{root}")`'
            )
    assert not failures, (
        f"{len(failures)} unguarded module-scope optional import(s) - these raise during "
        "collection on the documented gate env (no [inference]/[torch] extras):\n"
        + "\n".join(failures)
    )


def test_the_checker_flags_an_unguarded_import_and_clears_a_guarded_one() -> None:
    """Guard the guard: the checker must actually fail on the shape it exists to catch."""
    unguarded = "import pytest\nimport numpy as np\n"
    assert _unguarded_module_scope_imports(unguarded, {"numpy"}) == [(2, "numpy")]

    guarded = 'import pytest\npytest.importorskip("numpy")\nimport numpy as np\n'
    assert _unguarded_module_scope_imports(guarded, {"numpy"}) == []

    # Order matters: a guard AFTER the import does not protect collection.
    wrong_order = 'import numpy as np\nimport pytest\npytest.importorskip("numpy")\n'
    assert _unguarded_module_scope_imports(wrong_order, {"numpy"}) == [(1, "numpy")]

    # `from numpy import ...` and a non-guarded root are both caught; a guarded
    # sibling on the same file is not.
    mixed = (
        "import pytest\n"
        'pytest.importorskip("torch")\n'
        "import torch\n"  # guarded
        "from numpy.linalg import inv\n"  # unguarded root numpy
    )
    assert _unguarded_module_scope_imports(mixed, {"numpy", "torch"}) == [(4, "numpy")]

    # A stdlib module newer than the floor Python (tomllib, 3.11+) is caught the same
    # way. A `sys.version_info` conditional import NESTS it under an `if`, out of the
    # module body this walk inspects - which is exactly why a version-guarded import
    # clears and a bare top-level one does not.
    assert _unguarded_module_scope_imports("import tomllib\n", {"tomllib"}) == [(1, "tomllib")]
    conditional = (
        "import sys\n"
        "if sys.version_info >= (3, 11):\n"
        "    import tomllib\n"
        "else:\n"
        "    import tomli as tomllib\n"
    )
    assert _unguarded_module_scope_imports(conditional, {"tomllib"}) == []
