"""Tests for the bundled ``pictograph-cv`` Claude Skill.

The skill is content (Markdown + Python wrapper scripts), not code, so
the tests verify it ships in the package, the SKILL.md frontmatter is
valid, and the references + scripts referenced from SKILL.md actually
exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pictograph.skills import list_skills, skill_path

SKILL_NAME = "pictograph-cv"


def test_skill_path_resolves() -> None:
    """``skill_path()`` returns an existing directory containing SKILL.md."""
    path = skill_path(SKILL_NAME)
    assert isinstance(path, Path)
    assert path.is_dir()
    assert (path / "SKILL.md").is_file()


def test_skill_path_unknown_raises() -> None:
    with pytest.raises(FileNotFoundError, match="not_a_real_skill"):
        skill_path("not_a_real_skill")


def test_list_skills_includes_pictograph_cv() -> None:
    skills = list_skills()
    assert SKILL_NAME in skills


def test_skill_md_has_yaml_frontmatter() -> None:
    """SKILL.md begins with `---` frontmatter - Claude requires it."""
    body = (skill_path(SKILL_NAME) / "SKILL.md").read_text(encoding="utf-8")
    assert body.startswith("---\n"), "SKILL.md missing frontmatter"
    end = body.find("\n---\n", 4)
    assert end > 4, "SKILL.md frontmatter not closed with --- on its own line"


def test_skill_md_frontmatter_has_required_fields() -> None:
    body = (skill_path(SKILL_NAME) / "SKILL.md").read_text(encoding="utf-8")
    frontmatter = body.split("\n---\n", 1)[0]
    assert "name: pictograph-cv" in frontmatter
    assert "description:" in frontmatter
    # The description should be at least 50 chars (Anthropic recommends 100+).
    desc_line = next(line for line in frontmatter.splitlines() if line.startswith("description:"))
    assert len(desc_line) > 60, "skill description too short"


def test_skill_md_under_token_budget() -> None:
    """SKILL.md should fit comfortably under 5000 tokens (~20k chars)."""
    body = (skill_path(SKILL_NAME) / "SKILL.md").read_text(encoding="utf-8")
    estimated_tokens = len(body) // 4
    assert estimated_tokens < 5000, f"SKILL.md is ~{estimated_tokens} tokens - keep under 5000."


def test_skill_references_directory_populated() -> None:
    refs = skill_path(SKILL_NAME) / "references"
    assert refs.is_dir()
    expected = {
        "annotations.md",
        "auto-annotation.md",
        "training.md",
        "inference.md",
        "bulk-operations.md",
    }
    actual = {p.name for p in refs.iterdir() if p.is_file()}
    assert expected.issubset(actual), f"Missing references: {expected - actual}"


def test_skill_scripts_directory_populated() -> None:
    scripts = skill_path(SKILL_NAME) / "scripts"
    assert scripts.is_dir()
    expected = {
        "upload_and_annotate.py",
        "train.py",
        "import_connector.py",
        "export.py",
    }
    actual = {p.name for p in scripts.iterdir() if p.is_file()}
    assert expected.issubset(actual), f"Missing scripts: {expected - actual}"


def test_skill_md_links_match_existing_files() -> None:
    """Every relative file path mentioned in SKILL.md must exist."""
    skill_dir = skill_path(SKILL_NAME)
    body = (skill_dir / "SKILL.md").read_text(encoding="utf-8")

    # Match patterns like `references/foo.md` or `scripts/bar.py` (in-text refs).
    import re

    matches = re.findall(r"`((?:references|scripts)/[a-z_\-]+\.(?:md|py))`", body)
    assert matches, "SKILL.md should reference its own files"
    for relative in set(matches):
        candidate = skill_dir / relative
        assert candidate.is_file(), f"SKILL.md references missing file: {relative}"


def test_skill_scripts_are_valid_python() -> None:
    """Each wrapper script must parse as Python (no syntax errors)."""
    import ast

    scripts = skill_path(SKILL_NAME) / "scripts"
    for script in scripts.glob("*.py"):
        source = script.read_text(encoding="utf-8")
        try:
            ast.parse(source, filename=str(script))
        except SyntaxError as exc:
            pytest.fail(f"{script.name}: {exc}")


# ── the checks that catch drift, not just presence ──
#
# The skill is bash-callable, so an agent WILL run these scripts and WILL copy
# these snippets. A stale method name is worse than a missing one: it produces a
# confusing traceback instead of a clean "not available". Everything below is
# resolved against the INSTALLED SDK, so the skill cannot silently rot past it.


def _markdown_files() -> list[Path]:
    """Every markdown that ships to a user - the skill AND the README.

    README.md was outside this list, and `test_wheel_docs` (which does include
    it) only checks loader keywords and format values. So the file PyPI renders
    as the SDK's front page was documenting four methods that do not exist -
    `client.annotations.create_bbox` and friends - and nothing said a word.
    """
    found = sorted(skill_path(SKILL_NAME).glob("**/*.md"))
    readme = Path(__file__).resolve().parents[3] / "README.md"
    if readme.exists():  # present in the repo + sdist, absent in an installed wheel
        found.append(readme)
    return found


def _code_blocks(path: Path, language: str) -> list[tuple[int, str]]:
    import re

    text = path.read_text(encoding="utf-8")
    found = []
    for match in re.finditer(rf"```{language}\n(.*?)```", text, re.S):
        found.append((text[: match.start()].count("\n") + 1, match.group(1)))
    return found


def test_skill_scripts_run() -> None:
    """Every wrapper must import and parse its arguments in a fresh interpreter.

    A wrapper that cannot even reach ``--help`` is worse than an absent one.
    """
    import subprocess

    for script in sorted((skill_path(SKILL_NAME) / "scripts").glob("*.py")):
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"{script.name} --help failed:\n{result.stderr}"


def test_the_doc_corpus_is_not_empty() -> None:
    """Guard the guards below.

    Six tests in this module are shaped ``for md in _markdown_files(): for lineno,
    code in _code_blocks(md, "python"): assert ...``, and two more loop over
    ``scripts/*.py``. Every one of them PASSES when its collection is empty, so a
    single change that stops the discovery matching - the content moving, or a
    doc switching to a ```py fence instead of ```python - would quietly turn all
    eight into no-ops while the suite stayed green.

    Floors, not targets: they exist to fail loudly if the corpus disappears, and
    the sibling suites already do this (``test_wheel_docs`` asserts ``found``,
    ``test_no_internal_infra_in_docs`` asserts ``len(files) > 50``).
    """
    docs = _markdown_files()
    assert len(docs) >= 5, f"only {len(docs)} markdown files discovered - the doc scan is broken"

    scripts = sorted((skill_path(SKILL_NAME) / "scripts").glob("*.py"))
    assert len(scripts) >= 3, f"only {len(scripts)} wrapper scripts found"

    blocks = sum(len(_code_blocks(md, "python")) for md in docs)
    assert blocks >= 20, (
        f"only {blocks} ```python blocks found across {len(docs)} files - a fence-style "
        "change would make every per-block check below vacuous"
    )


def test_doc_python_blocks_parse() -> None:
    import ast

    for md in _markdown_files():
        for lineno, code in _code_blocks(md, "python"):
            try:
                ast.parse(code)
            except SyntaxError as exc:
                pytest.fail(f"{md.name}:{lineno} SyntaxError: {exc}")


def test_doc_imports_exist() -> None:
    """Every ``from pictograph… import X`` in the docs must actually resolve."""
    import ast
    import importlib

    for md in _markdown_files():
        for lineno, code in _code_blocks(md, "python"):
            for node in ast.walk(ast.parse(code)):
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = node.module or ""
                if not module.startswith("pictograph"):
                    continue
                imported = importlib.import_module(module)
                for alias in node.names:
                    assert hasattr(imported, alias.name), (
                        f"{md.name}:{lineno} `from {module} import {alias.name}` - no such name"
                    )


def test_doc_sdk_calls_bind() -> None:
    """Resolve every documented SDK call and bind its arguments.

    Catches a renamed method, a dropped keyword, a resource that no longer hangs
    off ``Client``, AND a call that omits a now-required argument.

    That last one used to slip through. This bound with ``bind_partial``, which
    by definition tolerates missing required parameters - so when the
    names-not-ids sweep gave 53 methods a new required leading argument (an image
    is addressed by ``(dataset, filename)`` now, not ``image_id``), every stale
    snippet in this skill still "bound" and the test stayed green. A doc sample
    is supposed to be runnable, so a full ``bind`` is the honest check.
    """
    import ast
    import inspect

    from pictograph import Client

    client = Client(api_key="pk_live_" + "0" * 32)
    resources = {
        name: type(value) for name, value in vars(client).items() if not name.startswith("_")
    }
    modules = {"pictograph": pictograph_module()}

    problems: list[str] = []
    for md in _markdown_files():
        for lineno, code in _code_blocks(md, "python"):
            for node in ast.walk(ast.parse(code)):
                if not isinstance(node, ast.Call):
                    continue
                parts, cursor = [], node.func
                while isinstance(cursor, ast.Attribute):
                    parts.append(cursor.attr)
                    cursor = cursor.value
                if not isinstance(cursor, ast.Name):
                    continue
                parts.append(cursor.id)
                parts.reverse()
                where = f"{md.name}:{lineno} {'.'.join(parts)}(...)"

                if parts[0] == "client" and len(parts) == 3:
                    cls = resources.get(parts[1])
                    if cls is None:
                        problems.append(f"{where} - client.{parts[1]} is not a resource")
                        continue
                    target = getattr(cls, parts[2], None)
                    offset = 1  # bound self
                elif parts[0] in modules and len(parts) == 2:
                    target = getattr(modules[parts[0]], parts[1], None)
                    offset = 0
                else:
                    continue

                if target is None:
                    problems.append(f"{where} - no such method")
                    continue
                try:
                    signature = inspect.signature(target)
                except (TypeError, ValueError):
                    continue
                # A snippet using *args/**kwargs cannot be arity-checked from
                # the AST - the star hides the real count.
                if any(isinstance(a, ast.Starred) for a in node.args) or any(
                    kw.arg is None for kw in node.keywords
                ):
                    continue
                try:
                    signature.bind(
                        *[None] * (len(node.args) + offset),
                        **{kw.arg: None for kw in node.keywords if kw.arg},
                    )
                except TypeError as exc:
                    problems.append(f"{where} - {exc}")

    assert not problems, "Documented SDK calls no longer match the SDK:\n" + "\n".join(problems)


def test_doc_cli_commands_exist() -> None:
    """Every ``pictograph <group> <command>`` in the docs must be a real command."""
    from pictograph.cli._app import app

    known: set[tuple[str, str]] = set()
    for group in app.registered_groups:
        sub = group.typer_instance
        for command in sub.registered_commands:
            name = command.name or (command.callback.__name__ if command.callback else "")
            known.add((str(group.name), name.replace("_", "-")))

    for md in _markdown_files():
        for lineno, block in _code_blocks(md, "bash"):
            for line in block.splitlines():
                tokens = line.strip().split()
                if len(tokens) < 3 or tokens[0] != "pictograph":
                    continue
                group, command = tokens[1], tokens[2]
                if group.startswith("-") or command.startswith("-"):
                    continue
                assert (group, command) in known, (
                    f"{md.name}:{lineno} `pictograph {group} {command}` is not a CLI command"
                )


def test_skill_exposes_no_internal_details() -> None:
    """The skill documents the product, never how it is built.

    Naming the compute provider, internal storage paths, backend module paths or
    anything pricing-internal leaks implementation detail into a user-facing
    document. Keep this list in step with what the SDK's own public docs allow.
    """
    import re

    forbidden = {
        r"\bmodal\b": "the compute provider",
        r"\bGCS\b|google cloud storage": "internal storage",
        r"gs://": "an internal bucket path",
        r"markup": "internal pricing",
        r"\bsupabase\b": "the database",
        r"pictograph-core|pictograph-app|tier_limits|utils\.pricing": "backend internals",
    }
    for md in _markdown_files():
        body = md.read_text(encoding="utf-8")
        for pattern, why in forbidden.items():
            match = re.search(pattern, body, re.I)
            assert match is None, (
                f"{md.name} exposes {why}: {match.group(0)!r}"  # type: ignore[union-attr]
            )


def pictograph_module() -> object:
    import pictograph

    return pictograph
