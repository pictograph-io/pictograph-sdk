"""Parity guard: the vendored ONNX wrappers stay faithful to their source.

``pictograph.inference._wrappers`` is a vendored copy of the canonical
server-side wrappers (synced by
``scripts/sync_inference_wrappers.sh``, which reformats + safe-autofixes them, so
they are NOT byte-identical to the canonical source verbatim). This file guards,
in increasing strength:

1. The file set matches (no wrapper added/removed on one side only).
2. Every top-level symbol - public AND private - in EVERY vendored module
   matches its canonical counterpart (previously only 4 of the ~10 files were
   checked, and only PUBLIC names within them - a private helper like
   ``_semantic_seg_to_annotations`` could drift unnoticed).
3. The strong guard: every vendored file's actual CONTENT must be exactly what
   running the sync script against the current canonical source produces right
   now. This is what catches real behavioral drift (a changed conditional, an
   added parameter, a rewritten branch) rather than just structural drift.

The canonical source does not ship in this repository, so the comparison is
opt-in and skips cleanly without it (see ``tests/conftest.py``).
"""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.conftest import ENV_WRAPPERS_SOURCE, companion_skip_reason, companion_source

_SDK_ROOT = Path(__file__).resolve().parents[2]
_VENDORED = _SDK_ROOT / "src" / "pictograph" / "inference" / "_wrappers"
_CANONICAL = companion_source(ENV_WRAPPERS_SOURCE)
_SYNC_SCRIPT = _SDK_ROOT / "scripts" / "sync_inference_wrappers.sh"

_requires_source = pytest.mark.skipif(
    not _CANONICAL.exists(),
    reason=companion_skip_reason(ENV_WRAPPERS_SOURCE),
)


def _resolve_ruff() -> str | None:
    """Mirror the sync script's own lookup: prefer the SDK's pinned ``.venv``
    copy, else whatever ``ruff`` resolves to on ``PATH``.

    Returns ``None`` if neither exists, so the content-fidelity test can skip
    rather than false-fail: without ruff, the sync script silently no-ops the
    reformatting step, so its raw output would legitimately not match the
    ruff-formatted committed copy - a tooling gap, not vendored-file drift.
    """
    venv_ruff = _SDK_ROOT / ".venv" / "bin" / "ruff"
    if venv_ruff.is_file():
        return str(venv_ruff)
    return shutil.which("ruff")


_requires_ruff = pytest.mark.skipif(
    _resolve_ruff() is None,
    reason="ruff not resolvable - the sync script needs it to reproduce vendored formatting",
)


def _top_level_names(path: Path, *, public_only: bool) -> set[str]:
    """Top-level function + class names defined in a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and (not public_only or not node.name.startswith("_"))
    }


@_requires_source
def test_file_set_matches_canonical() -> None:
    vendored = {p.name for p in _VENDORED.glob("*.py")}
    canonical = {p.name for p in _CANONICAL.glob("*.py")}
    # Two EMPTY sets are equal, so this passed unchanged if either directory
    # moved or emptied. `_requires_source` only checks that _CANONICAL exists,
    # not that it has anything in it, and nothing checks _VENDORED at all.
    assert len(canonical) >= 5, f"only {len(canonical)} canonical wrappers found at {_CANONICAL}"
    assert len(vendored) >= 5, f"only {len(vendored)} vendored wrappers found at {_VENDORED}"
    assert vendored == canonical, (
        f"vendored wrappers drifted from the canonical source; "
        f"run scripts/sync_inference_wrappers.sh. diff: {vendored ^ canonical}"
    )


@_requires_source
def test_all_symbols_match_canonical() -> None:
    """Every top-level symbol name in EVERY vendored module must match its
    canonical counterpart - public functions/classes AND private (``_``-
    prefixed) helpers alike.

    Previously this only checked 4 of the ~10 vendored files
    (``dispatch.py``, ``yolox_wrapper.py``, ``rfdetr_seg_wrapper.py``,
    ``classifier_wrapper.py``) and only their PUBLIC names, so e.g. a new
    private helper (``_semantic_seg_to_annotations``,
    ``_classification_to_result``) added to one copy but not the other would
    pass silently. This runs over every canonical module and every top-level
    def/class regardless of leading underscore.
    """
    canonical_files = sorted(p.name for p in _CANONICAL.glob("*.py"))
    mismatches: dict[str, set[str]] = {}
    for name in canonical_files:
        vendored_path = _VENDORED / name
        if not vendored_path.exists():
            continue  # test_file_set_matches_canonical already reports this
        vendored_names = _top_level_names(vendored_path, public_only=False)
        canonical_names = _top_level_names(_CANONICAL / name, public_only=False)
        if vendored_names != canonical_names:
            mismatches[name] = vendored_names ^ canonical_names
    assert not mismatches, (
        f"top-level symbols drifted between vendored and canonical - "
        f"run scripts/sync_inference_wrappers.sh. per-file symmetric diff: {mismatches}"
    )


@_requires_source
@_requires_ruff
def test_vendored_matches_fresh_sync(tmp_path: Path) -> None:
    """The strong guard: every vendored file's CONTENT - not just its symbol
    names - must be exactly what running ``scripts/sync_inference_wrappers.sh``
    against the current canonical source produces right now.

    The file-set and symbol-name checks above only catch STRUCTURAL drift (a
    module or a top-level def/class added, removed, or renamed). Real
    behavioral drift - a changed conditional, a rewritten branch, an added or
    reordered parameter, a docstring that quietly stopped matching the code -
    lives entirely inside function bodies and signatures, which those checks
    never look at. This test does: it re-runs the actual sync pipeline (copy +
    the dispatch.py import rewrite + ruff autofix + format - the same steps
    that produced the committed vendored copy) into a scratch directory, then
    diffs every file byte-for-byte against what's committed. Any drift, in
    either direction, fails this test.

    The sync script takes an optional destination-override argument for
    exactly this - a from-scratch render that never touches the real vendored
    copy.
    """
    result = subprocess.run(
        ["bash", str(_SYNC_SCRIPT), str(tmp_path)],
        cwd=_SDK_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"sync_inference_wrappers.sh failed against a scratch dir:\n"
        f"{result.stdout}\n{result.stderr}"
    )

    mismatches = []
    for canon_file in _CANONICAL.glob("*.py"):
        actual_path = _VENDORED / canon_file.name
        if not actual_path.exists():
            continue  # test_file_set_matches_canonical already reports this
        expected = (tmp_path / canon_file.name).read_text(encoding="utf-8")
        actual = actual_path.read_text(encoding="utf-8")
        if actual != expected:
            mismatches.append(canon_file.name)

    assert not mismatches, (
        f"vendored wrapper(s) drifted from a fresh sync of the canonical source: "
        f"{mismatches} - run scripts/sync_inference_wrappers.sh and commit the result."
    )


def test_vendored_dispatch_imports_are_relative() -> None:
    """The one absolute ``from inference_wrappers import`` the sync rewrites must
    be package-relative in the vendored copy (or it breaks when pip-installed)."""
    source = (_VENDORED / "dispatch.py").read_text(encoding="utf-8")
    assert "from inference_wrappers import" not in source
    assert "from . import" in source


def test_vendored_package_exposes_the_dispatch_contract() -> None:
    """The friendly layer relies on these three entry points existing.

    Checked statically (AST) rather than by importing the package - importing
    ``pictograph.inference._wrappers`` pulls in the heavy inference extras
    (cv2 / numpy / onnxruntime) that are deliberately absent from the base gate
    (see ``test_inference.py``, which ``importorskip``s them). Parsing the
    vendored ``dispatch.py`` keeps this contract check running in every
    environment, matching the other parity checks in this file.
    """
    names = _top_level_names(_VENDORED / "dispatch.py", public_only=True)
    for fn in ("build_wrapper", "infer_image", "infer_batch"):
        assert fn in names, f"dispatch.{fn} missing from the vendored wrappers"
