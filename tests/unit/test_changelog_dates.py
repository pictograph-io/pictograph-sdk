"""Every CHANGELOG version carries the day it was released.

A version's header is ``## [X.Y.Z] - YYYY-MM-DD``: the day it reached PyPI, in
Pacific time (the file's preamble). Release dates are read from these headers -
pictograph.io dates its Result models page by the release it documents - so a
header without its date leaves that page looking older than it is.

A version that was never published on its own stays undated and says where it
first shipped, so "no date" always means "never on PyPI", never "forgot".
"""

from __future__ import annotations

import re
from datetime import date
from itertools import pairwise
from pathlib import Path

import pictograph

CHANGELOG = Path(__file__).resolve().parents[2] / "CHANGELOG.md"
HEADER = re.compile(r"^## \[(\d+\.\d+\.\d+)\](?: - (\d{4}-\d{2}-\d{2}))?$")
UNPUBLISHED = re.compile(r"^_Not published on its own; first released in (\d+\.\d+\.\d+)\._$")


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def _sections() -> list[tuple[str, date | None, str]]:
    """(version, release date or None, first line of the body), newest first."""
    lines = CHANGELOG.read_text().split("\n")
    out: list[tuple[str, date | None, str]] = []
    for i, line in enumerate(lines):
        if not line.startswith("## "):
            continue
        m = HEADER.match(line)
        assert m, f"CHANGELOG line {i + 1} is not `## [X.Y.Z] - YYYY-MM-DD`: {line!r}"
        body = next((b for b in lines[i + 1 :] if b.strip()), "")
        out.append((m.group(1), date.fromisoformat(m.group(2)) if m.group(2) else None, body))
    return out


def test_versions_run_newest_first() -> None:
    versions = [v for v, _, _ in _sections()]
    assert versions, "CHANGELOG has no version sections"
    out_of_order = [(a, b) for a, b in pairwise(versions) if _key(a) <= _key(b)]
    assert not out_of_order, f"versions must run newest first: {out_of_order}"


def test_release_dates_never_go_backwards() -> None:
    dated = [(v, d) for v, d, _ in _sections() if d is not None]
    backwards = [(a, b) for a, b in pairwise(dated) if a[1] < b[1]]
    assert not backwards, f"a newer version is dated before an older one: {backwards}"


def test_an_undated_version_says_where_it_first_shipped() -> None:
    sections = _sections()
    dated = {v for v, d, _ in sections if d is not None}
    for version, released, body in sections:
        if released is not None:
            continue
        m = UNPUBLISHED.match(body)
        assert m, (
            f"{version} has no release date: write `## [{version}] - YYYY-MM-DD` (the day it "
            "reached PyPI, Pacific time), or, if it was never published on its own, open its "
            "section with `_Not published on its own; first released in X.Y.Z._`"
        )
        shipped_in = m.group(1)
        assert shipped_in in dated and _key(shipped_in) > _key(version), (
            f"{version} says it first shipped in {shipped_in}, which is not a newer dated version"
        )


def test_the_current_version_is_dated() -> None:
    by_version = {v: d for v, d, _ in _sections()}
    version = pictograph.__version__
    assert version in by_version, f"CHANGELOG has no `## [{version}]` section"
    assert by_version[version] is not None, (
        f"the {version} header has no release date: write `## [{version}] - YYYY-MM-DD`, "
        "the day it reaches PyPI (Pacific time)"
    )
