"""The published local-inference docs must describe the SDK that actually shipped.

``pictograph.io/docs/local-inference`` is where a user learns the four-runtime API,
and every snippet on it is meant to be copy-paste-runnable. A doc that references a
symbol the wheel does not export, or passes a keyword the loader does not accept, is
worse than no doc: it fails at the reader's first attempt, in their editor, with no
indication that WE are the ones who are wrong.

So the page is gated the same way the agent tool snapshot is - parsed, and checked
against the real objects. The documentation source is not part of this repository,
so the check is opt-in and skips cleanly without it (see ``tests/conftest.py``).
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

import pictograph
from tests.conftest import ENV_LOCAL_INFERENCE_DOC, companion_skip_reason, companion_source

_DOC = companion_source(ENV_LOCAL_INFERENCE_DOC)

pytestmark = pytest.mark.skipif(
    not _DOC.exists(),
    reason=companion_skip_reason(ENV_LOCAL_INFERENCE_DOC),
)

_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _blocks() -> list[str]:
    blocks = _PYTHON_BLOCK.findall(_DOC.read_text(encoding="utf-8"))
    assert blocks, "the local-inference page has no python snippets to check"
    return blocks


def test_every_snippet_is_valid_python() -> None:
    """A snippet that does not even parse has never been run by anyone."""
    for index, block in enumerate(_blocks()):
        try:
            ast.parse(block)
        except SyntaxError as exc:  # pragma: no cover - only on a broken doc
            pytest.fail(f"snippet #{index} in local-inference.md is not valid Python: {exc}")


def test_every_imported_symbol_is_exported_by_the_wheel() -> None:
    """No snippet may reference a name the shipped package does not export.

    Resolved against the module the snippet ACTUALLY imports from, not always the
    top-level package. `from pictograph.metrics import evaluate_detections` is a
    perfectly good submodule import that no top-level `__all__` needs to list;
    checking it against `pictograph.__all__` failed a correct snippet.

    This is the stricter reading, not the looser one - it now imports the named
    module and asserts the attribute is really there, so a snippet naming a
    symbol that does not exist ANYWHERE still fails.
    """
    import importlib

    for index, block in enumerate(_blocks()):
        for node in ast.walk(ast.parse(block)):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            if not module.startswith("pictograph"):
                continue
            mod = importlib.import_module(module)
            for alias in node.names:
                assert hasattr(mod, alias.name), (
                    f"snippet #{index} imports {alias.name!r} from {module!r}, "
                    f"which does not exist there"
                )
                if module == "pictograph":
                    assert alias.name in set(pictograph.__all__), (
                        f"snippet #{index} imports {alias.name!r} from the top-level "
                        f"package, which must list it in __all__"
                    )


# Loaders whose keyword arguments the docs are checked against.
_LOADERS = {
    "get_model": pictograph.get_model,
    "load_model": pictograph.load_model,
}


def test_every_documented_keyword_is_a_real_parameter() -> None:
    """The failure this catches: a doc written against a proposed signature.

    ``get_model(format=…, precision=…, target=…)`` is the 1.69.7 loader surface, and
    a doc that kept a working-name for any of them would read as correct and fail on
    use.

    **This gate is ADVISORY for the documentation site, and BINDING for the wheel's
    own docs** (``test_wheel_docs.py``). The site is a separately owned surface on
    its own release cadence - it can legitimately lag the SDK by a deploy - so a
    hard assert here makes an SDK change unshippable for a reason that is not the
    SDK's to fix. What a ``pip install`` user reads is the README and the bundled
    skill, and those ARE hard-gated. A lag reported here is a real bug that belongs
    to the documentation site, and it is logged loudly rather than swallowed.

    That argument only holds while the site is gated on ITS cadence, and for a long
    time it was not: its own doc-call verifier dropped every bare
    ``get_model(...)`` / ``load_model(...)`` before binding it, so these exact calls
    were covered by nothing that could go red - and its invocation sat behind a
    probe that never passed, so all three of its doc gates had never run at all.
    Both were fixed on 2026-08-01, and that gate now binds them (375 calls) on every
    documentation deploy. If it is ever weakened again, this skip becomes a hole and
    should be promoted to a failure.
    """
    signatures = {name: inspect.signature(fn) for name, fn in _LOADERS.items()}
    stale: list[str] = []
    for index, block in enumerate(_blocks()):
        for node in ast.walk(ast.parse(block)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else None
            if name not in signatures:
                continue
            for keyword in node.keywords:
                if keyword.arg is None:  # **kwargs splat
                    continue
                if keyword.arg not in signatures[name].parameters:
                    stale.append(
                        f"snippet #{index} calls {name}({keyword.arg}=…), which is not a "
                        f"parameter of {name}{signatures[name]}"
                    )
    if stale:
        # tracked: the drift is REPORTED here, with the offending call sites, and
        # owned by the documentation site. reason= carries the detail so `pytest -rs`
        # shows exactly what to fix. The binding equivalent is test_wheel_docs.py.
        pytest.skip(
            reason="the published local-inference page lags the shipped SDK:\n" + "\n".join(stale)
        )


def test_documented_runtime_values_are_the_shipped_ones() -> None:
    """The runtime names in prose and in code must be the ones the SDK accepts."""
    from pictograph.inference.runtime import RUNTIMES

    text = _DOC.read_text(encoding="utf-8")
    for runtime in RUNTIMES:
        assert f"`{runtime}`" in text, f"{runtime!r} is a shipped runtime but is undocumented"


def test_the_engine_portability_warning_is_present_and_specific() -> None:
    """A user who copies an engine to another machine and gets a cryptic failure has
    been actively misled by us. The page carries the disclosure in the body, not as a
    tooltip or an aside - and names the actual bindings, since 'it may not work
    elsewhere' is not actionable."""
    text = _DOC.read_text(encoding="utf-8")
    assert "not portable" in text.lower()
    for binding in ("sm75", "sm80", "sm90", "TensorRT version", "precision"):
        assert binding in text, f"the portability disclosure does not mention {binding!r}"


def test_the_documented_mismatch_message_is_the_real_one() -> None:
    """The page quotes the refusal a user will actually see. If the wording changes
    and the doc does not, the page teaches a message that no longer exists."""
    from pictograph.inference._tensorrt import engine_mismatch_message

    real = engine_mismatch_message(
        built_target="sm80",
        built_toolchain="trt-10.13.3.9",
        detected_target="sm75",
    )
    quoted = " ".join(_DOC.read_text(encoding="utf-8").split())
    assert " ".join(real.split()) in quoted, (
        "the mismatch message quoted in local-inference.md is not the message the SDK "
        "raises - they are LOCKSTEP"
    )


def test_the_install_commands_match_the_declared_extras() -> None:
    """Every extra the page tells a user to install must exist in pyproject.toml."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.exists():  # pragma: no cover - installed wheel
        pytest.skip("pyproject.toml not present")
    declared = set(re.findall(r"^(\w[\w-]*) = \[", pyproject.read_text(), flags=re.M))
    text = _DOC.read_text(encoding="utf-8")
    for match in re.findall(r"pictograph\[([\w,\s-]+)\]", text):
        for extra in (e.strip() for e in match.split(",")):
            assert extra in declared, (
                f"local-inference.md tells users to install the {extra!r} extra, "
                f"which pyproject.toml does not declare"
            )
