"""Bundled Claude Skills.

Currently ships one skill, ``pictograph-cv`` (in
``src/pictograph/skills/pictograph-cv/``). Use :func:`skill_path` to
resolve the on-disk path for installer / packaging code.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def skill_path(skill_name: str = "pictograph-cv") -> Path:
    """Return the on-disk path of a bundled skill.

    Works for both editable installs (returns repo path) and wheel installs
    (returns site-packages path). Used by the CLI's ``install-skill`` command
    to copy the skill into ``~/.claude/skills/``.
    """
    package = resources.files("pictograph.skills")
    candidate = package / skill_name
    if not candidate.is_dir():
        raise FileNotFoundError(f"Skill {skill_name!r} not found in pictograph.skills package.")
    return Path(str(candidate))


def list_skills() -> list[str]:
    """Names of every bundled skill."""
    package = resources.files("pictograph.skills")
    return sorted(
        entry.name
        for entry in package.iterdir()
        if entry.is_dir() and (entry / "SKILL.md").is_file()
    )
