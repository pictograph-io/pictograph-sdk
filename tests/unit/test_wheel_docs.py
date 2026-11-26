"""The docs that ship INSIDE the wheel must describe the SDK that shipped with them.

``README.md`` (rendered on the PyPI project page) and the bundled ``pictograph-cv``
skill are what a ``pip install pictograph`` user and their agent actually read. They
travel in the same artifact as the code, so there is no cadence to lag behind and no
other repo to coordinate with - a mismatch here is unambiguously this package's bug,
and this gate is therefore BINDING.

Contrast ``test_local_inference_docs.py``, which checks the published documentation
site: that corpus lives elsewhere and deploys separately, so a keyword lag there is
reported as a warning rather than blocking an SDK release.

What is checked:

* every keyword passed to a loader in a snippet is a real parameter of it;
* every ``format="…"`` value is one the SDK accepts - the failure this exists for is
  a doc that keeps a working-name for a format after the vocabulary settles;
* no snippet calls a loader the package no longer exports.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

import pictograph
from pictograph.inference.runtime import WEIGHT_FORMATS
from pictograph.skills import skill_path

_REPO = Path(__file__).resolve().parents[2]
_README = _REPO / "README.md"

_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)

# The loaders whose call sites are resolved. Module-level functions only - the
# client-bound twins are covered by the skill's own `test_doc_sdk_calls_bind`.
_LOADERS = {
    "get_model": pictograph.get_model,
    "load_model": pictograph.load_model,
}

#: Loaders removed in 1.69.7, when the model-loading API unified on ``format=``.
#: A doc that still names one teaches an import that raises.
_REMOVED_LOADERS = ("load_pytorch",)


def _docs() -> list[Path]:
    """Every markdown doc that ships in the wheel or the sdist."""
    found = sorted(skill_path("pictograph-cv").glob("**/*.md"))
    if _README.exists():  # absent in an installed wheel; present in the repo + sdist
        found.append(_README)
    assert found, "no bundled docs found to check"
    return found


def _blocks(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    return [(text[: m.start()].count("\n") + 1, m.group(1)) for m in _PYTHON_BLOCK.finditer(text)]


def test_every_documented_loader_keyword_is_real() -> None:
    """A snippet that passes a keyword the loader does not take has never been run."""
    signatures = {name: inspect.signature(fn) for name, fn in _LOADERS.items()}
    problems: list[str] = []
    for doc in _docs():
        for lineno, block in _blocks(doc):
            for node in ast.walk(ast.parse(block)):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.id if isinstance(node.func, ast.Name) else None
                if name not in signatures:
                    continue
                for keyword in node.keywords:
                    if keyword.arg is None:
                        continue
                    if keyword.arg not in signatures[name].parameters:
                        problems.append(
                            f"{doc.name}:{lineno} {name}({keyword.arg}=…) - not a parameter "
                            f"of {name}{signatures[name]}"
                        )
    assert not problems, "bundled docs document keywords the SDK does not accept:\n" + "\n".join(
        problems
    )


def test_every_documented_format_value_is_accepted() -> None:
    """``format=`` is a closed vocabulary; a doc may not invent a sixth value.

    Checks the literal actually passed in a snippet, not prose, so a sentence
    ABOUT a format ("the ONNX graph") is not mistaken for a call.
    """
    problems: list[str] = []
    for doc in _docs():
        for lineno, block in _blocks(doc):
            for node in ast.walk(ast.parse(block)):
                if not isinstance(node, ast.Call):
                    continue
                for keyword in node.keywords:
                    if keyword.arg != "format" or not isinstance(keyword.value, ast.Constant):
                        continue
                    value = keyword.value.value
                    if not isinstance(value, str):
                        continue
                    name = node.func.id if isinstance(node.func, ast.Name) else None
                    # `client.models.download(format=…)` speaks the WIRE vocabulary
                    # on purpose (it is a raw file fetch, not a loader), so only
                    # loader calls are held to WEIGHT_FORMATS.
                    if name not in _LOADERS:
                        continue
                    if value not in WEIGHT_FORMATS:
                        problems.append(
                            f"{doc.name}:{lineno} {name}(format={value!r}) - the accepted "
                            f"values are {WEIGHT_FORMATS}"
                        )
    assert not problems, "bundled docs use a format the SDK does not accept:\n" + "\n".join(
        problems
    )


@pytest.mark.parametrize("removed", _REMOVED_LOADERS)
def test_no_bundled_doc_references_a_removed_loader(removed: str) -> None:
    """Prose counts here, not just code: "use ``load_pytorch``" is equally wrong."""
    assert not hasattr(pictograph, removed), (
        f"{removed} is back on the package - update _REMOVED_LOADERS rather than "
        f"leaving this test asserting the opposite of reality"
    )
    offenders = [doc.name for doc in _docs() if removed in doc.read_text(encoding="utf-8")]
    assert not offenders, (
        f"{offenders} still reference {removed}(), which the package no longer exports"
    )


# ───────────── house style: named arguments, one per line ─────────────
#
# The published documentation site has enforced both of these for a while. The
# BUNDLED skill - the docs an agent actually reads - had neither, so "every
# argument named in every sample, one argument per line" was a rule nothing
# checked. These are the same two rules applied to the wheel's own docs.

#: Calls whose single leading positional is self-evidently the subject, so
#: naming it adds nothing. Everything else names its arguments.
_SUBJECT_POSITIONAL_OK = {
    "get_model",
    "load_model",
    "Client",
    "print",
    "len",
    "open",
    "range",
    "sorted",
    "enumerate",
    "list",
    "str",
    "int",
    "float",
    "Path",
}


def _sdk_calls(block: str):
    """Attribute calls rooted at `client`, plus the top-level SDK functions."""
    for node in ast.walk(ast.parse(block)):
        if not isinstance(node, ast.Call):
            continue
        cursor, parts = node.func, []
        while isinstance(cursor, ast.Attribute):
            parts.append(cursor.attr)
            cursor = cursor.value
        if isinstance(cursor, ast.Name):
            parts.append(cursor.id)
            parts.reverse()
            if parts[0] == "client" and len(parts) == 3:
                yield node, ".".join(parts)
            elif len(parts) == 1 and parts[0] in _LOADERS:
                yield node, parts[0]


def test_bundled_docs_name_their_arguments() -> None:
    """A reader copies a snippet to learn the API.

    `annotations.save("road-signs", "img.jpg", [...])` teaches nothing about what
    those strings are, and cannot be adapted without going to look up the
    signature - the one thing a sample exists to save them.
    """
    problems: list[str] = []
    for doc in _docs():
        for lineno, block in _blocks(doc):
            for node, name in _sdk_calls(block):
                if name.split(".")[-1] in _SUBJECT_POSITIONAL_OK:
                    continue
                # One leading positional is allowed - it is the subject.
                if len(node.args) > 1:
                    problems.append(
                        f"{doc.name}:{lineno} {name}(...) - {len(node.args)} positional "
                        f"arguments; name them"
                    )
    assert not problems, "bundled docs pass unnamed arguments:\n" + "\n".join(problems)


def test_bundled_docs_put_one_argument_per_line() -> None:
    """A multi-argument call spread over one line is unreadable in a terminal and
    diffs badly.

    Scoped exactly as the documentation site's own rule is: attribute chains like
    `client.images.upload(...)`. A bare `get_model("Detector", task=...)` stays on
    one line - it is short, it is the idiom the published docs themselves use, and
    splitting it would make the single most-copied line in the SDK worse to read.
    """
    problems: list[str] = []
    for doc in _docs():
        for lineno, block in _blocks(doc):
            lines = block.splitlines()
            for node, name in _sdk_calls(block):
                if "." not in name:  # top-level loader, not an attribute chain
                    continue
                total = len(node.args) + len(node.keywords)
                if total < 2:
                    continue
                if node.end_lineno is not None and node.lineno == node.end_lineno:
                    source = lines[node.lineno - 1].strip()
                    problems.append(
                        f"{doc.name}:{lineno + node.lineno - 1} {name}(...) - "
                        f"{total} arguments on one line: {source[:70]}"
                    )
    assert not problems, "bundled docs put several arguments on one line:\n" + "\n".join(problems)
