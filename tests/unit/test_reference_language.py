"""Reference-language gate - no user-facing "project" where the primitive is "dataset".

The developer surface calls the primitive a **dataset**. ``project`` is the leaked
internal name, surviving only in storage-layer identifiers such as ``project_id``. This
gate fails if any USER-FACING string in the SDK contains the standalone word
``project`` / ``projects``:

  * a docstring (module, class, or function - these render in ``help()`` / IDEs),
  * a ``help=`` or ``description=`` argument (Typer help, Pydantic ``Field`` and the
    agent tool-arg schemas served at ``/developer/tools.json``), or
  * the bundled Claude skill markdown (shipped in the wheel, installed into agents).

It sees ONLY user-facing text by construction: internal identifiers (``project_id``,
a local ``project = ...`` variable) and ``# code comments``
are *not* docstrings or ``help=``/``description=`` strings, so they never reach the
check. A span inside double backticks ````like this```` is a deliberate written
record of the internal name (e.g. the ``Project``/``ProjectClass`` history in
``models/dataset.py``) and is stripped first - the same convention the
folder->directory sweep used. ``this/your project`` (the user's own repo, in the
``pictograph init`` scaffold and skill prose) is allowed.

RED-proven: when it was written this fired on ~40 strings across resources/models/cli/agents
plus the bundled skill. ``test_gate_is_not_vacuous_and_catches_a_violation`` proves
the checker actually catches a planted violation and clears an allowed one, so a
future refactor cannot quietly turn it into a no-op.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SDK = Path(__file__).resolve().parents[2] / "src" / "pictograph"
PY_DIRS = ("resources", "models", "cli", "agents")
SKILLS = SDK / "skills"

_WORD = re.compile(r"\bprojects?\b", re.IGNORECASE)
# ``...`` (deliberate record of the internal name) - stripped before the check.
_TICKS = re.compile(r"``[^`]*``")
# `...` single-backtick inline code in markdown (e.g. `project_config`).
_CODE = re.compile(r"`[^`]*`")
# The user's OWN repo/codebase - not a Pictograph dataset.
_ALLOWED = re.compile(r"\b(this|your|the user'?s|their)\s+project\b", re.IGNORECASE)


def _clean(text: str, markdown: bool = False) -> str:
    text = _TICKS.sub("", text)
    if markdown:
        text = _CODE.sub("", text)
    return _ALLOWED.sub("", text)


def _hit(text: str, markdown: bool = False) -> bool:
    return bool(_WORD.search(_clean(text, markdown=markdown)))


def _user_facing_py_strings(path: Path) -> list[tuple[int, str]]:
    """Docstrings + ``help=``/``description=`` string literals in one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                out.append((getattr(node, "lineno", 1), doc))
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if (
                    kw.arg in ("help", "description")
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                ):
                    out.append((kw.value.lineno, kw.value.value))
    return out


def _scan() -> list[str]:
    problems: list[str] = []
    for d in PY_DIRS:
        for f in sorted((SDK / d).rglob("*.py")):
            for lineno, text in _user_facing_py_strings(f):
                for off, line in enumerate(text.splitlines()):
                    if _hit(line):
                        rel = f.relative_to(SDK)
                        problems.append(f"{rel} (~L{lineno + off}): {line.strip()}")
    for f in sorted(SKILLS.rglob("*.md")):
        for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if _hit(line, markdown=True):
                problems.append(f"{f.relative_to(SDK)}:{lineno}: {line.strip()}")
    return problems


def test_no_user_facing_project_alias_in_the_sdk() -> None:
    problems = _scan()
    assert not problems, (
        "User-facing 'project' found where the primitive is 'dataset'. "
        "Say 'dataset'; keep 'project' only in internal identifiers "
        "(project_id, the projects table) and ``-quoted history:\n  " + "\n  ".join(problems)
    )


def test_gate_is_not_vacuous_and_catches_a_violation() -> None:
    # Must FLAG the leaked internal name in user-facing prose...
    assert _hit("Dataset (project) name.")
    assert _hit("List a project's directories.")
    assert _hit("Add referenced classes the projects don't define.")
    # ...and must CLEAR the legitimate exceptions the scan relies on:
    assert not _hit("Exact dataset name (org-unique).")
    assert not _hit("dataset_id: The dataset UUID.")  # project_id-style ids are ok
    assert not _hit("The former ``Project``/``ProjectClass`` models were folded in.")
    assert not _hit("This project uses Pictograph for annotation.")  # user's own repo
    assert not _hit("Connectivity lives on the `project_config` table.", markdown=True)
    # The extractor must actually pull real docstrings/help from a known file
    # (guards a bad SDK path that would make test_no_user_facing... vacuously pass).
    strings = _user_facing_py_strings(SDK / "resources" / "images.py")
    assert len(strings) > 20, f"extractor saw only {len(strings)} strings in images.py"
