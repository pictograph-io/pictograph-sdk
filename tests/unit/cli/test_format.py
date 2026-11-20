"""Tests for the CLI Rich-rendering helpers (``pictograph.cli._format``).

Regression: ``_render_cell`` / ``print_error`` / the table title previously
passed server-controlled strings straight into Rich, which parses ``[...]`` as
console markup. A closing/unbalanced tag (``[/]``) raised ``MarkupError`` and
crashed the whole command; a style-like tag (``[red]…``) was silently consumed,
dropping text. All server data is now escaped, while our own styled literals
(``[green]ok``, ``[bold red]error:``) stay as intentional markup.

Note: ``rich.markup.escape`` only backslash-escapes *tag-like* brackets (``[``
followed by ``[a-z#/@]``), so a benign ``frame[01].jpg`` is left unchanged - it
was never markup. The dangerous forms (``[/]``, ``[red]…``) are what must be
escaped, and rendering must never crash regardless.
"""

from __future__ import annotations

from io import StringIO
from typing import Any

import pytest
from rich.console import Console

from pictograph.cli import _format


def _render(s: str) -> str:
    out = StringIO()
    Console(file=out, force_terminal=False, width=200).print(s)
    return out.getvalue()


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch):
    """Swap the module consoles for StringIO-backed (wide) ones."""
    out = StringIO()
    err = StringIO()
    monkeypatch.setattr(_format, "console", Console(file=out, force_terminal=False, width=200))
    monkeypatch.setattr(_format, "err_console", Console(file=err, force_terminal=False, width=200))
    return {"out": out, "err": err}


# ─── _render_cell escaping ─────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["[/bold]", "close[/]tag", "[red]danger[/red]", "[link=x]y"])
def test_render_cell_escapes_dangerous_markup(value: str) -> None:
    rendered = _format._render_cell(value)
    # Tag-like brackets are backslash-escaped so Rich renders them literally.
    assert "\\[" in rendered
    # Round-trip through a real Console: no crash, literal text survives.
    assert value in _render(rendered)


@pytest.mark.parametrize("value", ["frame[01].jpg", "model[v2]", "plain name"])
def test_render_cell_passes_benign_text_through(value: str) -> None:
    # Non-tag-like brackets aren't markup; they round-trip unchanged and never crash.
    assert value in _render(_format._render_cell(value))


def test_render_cell_keeps_styled_literals() -> None:
    # Our own intentional markup is preserved (not escaped).
    assert _format._render_cell(None) == "[dim]-[/dim]"
    assert _format._render_cell(True) == "[green]yes[/green]"
    assert _format._render_cell(False) == "[red]no[/red]"


def test_render_cell_escapes_list_and_dict_members() -> None:
    assert "\\[" in _format._render_cell(["a[/]", "b[bold]"])
    assert "\\[" in _format._render_cell({"k": "v[/red]"})


# ─── print_table does not crash on markup-like server data ──────────────────


def test_print_table_survives_markup_in_cells(captured: dict[str, Any]) -> None:
    rows = [{"name": "close[/]tag", "desc": "[red]x[/red]"}]
    _format.print_table(rows)  # pre-fix: MarkupError
    out = captured["out"].getvalue()
    assert "close[/]tag" in out
    assert "[red]x[/red]" in out  # literal, not consumed as a style tag


def test_print_table_survives_markup_in_title(captured: dict[str, Any]) -> None:
    # Wide row so the (short) title isn't wrapped by the table width.
    _format.print_table([{"col": "x" * 60}], title="ws [/] weird")  # pre-fix: MarkupError
    assert "ws [/] weird" in captured["out"].getvalue()


# ─── print_error does not crash on markup-like messages ─────────────────────


def test_print_error_survives_markup(captured: dict[str, Any]) -> None:
    _format.print_error("boom [/] at [bold]idx")  # pre-fix: MarkupError
    err = captured["err"].getvalue()
    assert "boom [/] at [bold]idx" in err
    assert "error:" in err
