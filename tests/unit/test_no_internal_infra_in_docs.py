"""Guard: no internal infrastructure vendors in USER-FACING documentation.

Everything this test scans ships on PyPI and reaches users through ``help()``,
IDE tooltips, generated reference docs, ``--help`` output, and - for
``Field(description=...)`` / ``ToolDescriptor(description=...)`` - the JSON
Schema handed to LLM tool-callers.

The rule (repo owner's standing instruction for customer-facing docs): *clear
information, nothing else, no internal workings exposed*. Naming the compute
provider, the object-storage product, or the serving platform tells a reader
nothing they can act on. Describe the OBSERVABLE BEHAVIOUR instead - "a signed
storage URL", "object storage", "the training service" - and keep every fact a
caller must reason about (that a download is a second, unauthenticated request
to a different host; that a cancelled run is not charged; that a cold dataset
pauses byte-heavy operations).

What is scanned: docstrings (module / class / function / PEP-258 attribute) and
the string literals passed as ``description=`` or ``help=``. Ordinary in-body
``#`` comments are deliberately NOT scanned - they are not documentation and no
user surface renders them.

To fix a failure, rewrite the sentence to describe the behaviour. Do NOT delete
the behaviour to satisfy the grep, and do not add an allowlist entry for our own
infrastructure - the allowlist exists only for the CUSTOMER's cloud, which is a
product feature they configure by name.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "pictograph"

#: Our own infrastructure. Case-sensitive where a lowercase spelling is
#: legitimate: ``modal`` is a UI dialog, and ``gcs_path`` / ``gcs_uri`` /
#: ``gcs_image_path`` are backend WIRE FIELD NAMES the SDK cannot rename
#: without breaking the API contract, so both stay allowed.
_BANNED: dict[str, re.Pattern[str]] = {
    "compute/serving platform": re.compile(r"\bModal\b"),
    "object storage product": re.compile(r"\bGCS\b|Google Cloud Storage|\bColdline\b"),
    "storage host / URI scheme": re.compile(r"gs://|storage\.googleapis\.com"),
    "serving platform": re.compile(r"\bCloud Run\b"),
    "database platform": re.compile(r"\bSupabase\b"),
}

#: Modules whose SUBJECT is the customer's own cloud storage. There, "GCS" names
#: a provider the user picks and configures (``provider="gcs"``) - it is a
#: product capability, not a disclosure of our infrastructure.
_CUSTOMER_CLOUD_MODULES = {
    "resources/storage.py",
    "aio/resources/storage.py",
    "models/storage.py",
    "cli/commands/storage.py",
}


def _source_files() -> list[Path]:
    # skills/ is a separately-owned bundle with its own internal-detail guard.
    return sorted(p for p in _SRC.rglob("*.py") if "skills" not in p.parts)


def _doc_strings(tree: ast.AST) -> list[tuple[int, str]]:
    """Every user-visible documentation string in the module.

    Docstrings (module / class / def / async def), PEP-258 attribute docstrings
    (a bare string expression at module or class body level), and the literals
    passed as ``description=`` / ``help=``.
    """
    found: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                found.append((getattr(node, "lineno", 1), doc))

        # PEP-258 attribute docstrings sit as bare string statements in a
        # module or class body (e.g. the line under DEFAULT_CHUNK_SIZE). Skip
        # body[0] - that is the module/class docstring, already collected above.
        if isinstance(node, (ast.Module, ast.ClassDef)):
            for stmt in node.body[1:]:
                if (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                ):
                    found.append((stmt.lineno, stmt.value.value))

        # description= / help= keyword arguments: Pydantic Field descriptions
        # (which become JSON Schema for agent tool-callers), ToolDescriptor
        # descriptions, and typer --help text.
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in ("description", "help") and isinstance(
                    getattr(kw.value, "value", None), str
                ):
                    found.append((kw.value.lineno, kw.value.value))  # type: ignore[attr-defined]

    return found


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: str(p.relative_to(_SRC)))
def test_no_internal_infra_in_user_facing_docs(path: Path) -> None:
    rel = path.relative_to(_SRC).as_posix()
    if rel in _CUSTOMER_CLOUD_MODULES:
        pytest.skip("subject is the customer's own cloud storage, not our infrastructure")

    tree = ast.parse(path.read_text(encoding="utf-8"))

    violations: list[str] = []
    for lineno, text in _doc_strings(tree):
        for label, pattern in _BANNED.items():
            match = pattern.search(text)
            if match:
                line = text[: match.start()].count("\n") + lineno
                violations.append(f"  {rel}:~{line}  names the {label}: {match.group(0)!r}")

    assert not violations, (
        f"User-facing documentation in {rel} names internal infrastructure.\n"
        + "\n".join(sorted(set(violations)))
        + "\n\nRewrite it to describe the observable behaviour instead "
        "(see this module's docstring). Keep every fact the caller must reason "
        "about; only the vendor noun goes."
    )


def test_guard_actually_scans_something() -> None:
    """A silent no-op guard is worse than none - prove it reads real docstrings."""
    files = _source_files()
    assert len(files) > 50, "source tree not found"
    total = sum(len(_doc_strings(ast.parse(p.read_text(encoding="utf-8")))) for p in files)
    assert total > 500, f"only {total} documentation strings scanned - walker is broken"


def test_wire_field_names_are_not_flagged() -> None:
    """``gcs_path`` and friends are backend field names, not prose. Never banned."""
    sample = "The SDK uses ``upload_url`` to PUT bytes, then passes ``gcs_path`` and ``gcs_uri``."
    for pattern in _BANNED.values():
        assert not pattern.search(sample)


def test_ui_modal_word_is_not_flagged() -> None:
    """Lowercase ``modal`` means a UI dialog and must stay usable."""
    sample = "A user who hits this in the API, in the SDK and in the modal reads one sentence."
    assert not _BANNED["compute/serving platform"].search(sample)
