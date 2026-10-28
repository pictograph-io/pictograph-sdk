"""Tests for ``pictograph.models.common``: Point, BoundingBox, NonBlankStr."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from pictograph.models.common import BoundingBox, NonBlankStr, Point

# ───────────── Point ─────────────


def test_point_construction_with_int_coerces_to_float() -> None:
    p = Point(x=10, y=20)
    assert p.x == 10.0
    assert p.y == 20.0
    assert isinstance(p.x, float)
    assert isinstance(p.y, float)


def test_point_accepts_negative_coordinates() -> None:
    # Negative coords appear when annotations reference off-canvas points
    # (e.g., during rotation or padded crops). The model permits them.
    p = Point(x=-50.0, y=-100.0)
    assert p.x == -50.0
    assert p.y == -100.0


def test_point_is_frozen() -> None:
    p = Point(x=1, y=2)
    with pytest.raises(ValidationError, match="frozen"):
        p.x = 99  # type: ignore[misc]


def test_point_is_hashable() -> None:
    p1 = Point(x=1, y=2)
    p2 = Point(x=1, y=2)
    p3 = Point(x=3, y=4)
    points = {p1, p2, p3}
    assert len(points) == 2  # p1 and p2 dedupe


def test_point_equality_uses_field_values() -> None:
    assert Point(x=1, y=2) == Point(x=1, y=2)
    assert Point(x=1, y=2) != Point(x=1, y=3)


def test_point_extra_field_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        Point.model_validate({"x": 1, "y": 2, "z": 3})
    assert exc.value.error_count() == 1
    assert exc.value.errors()[0]["type"] == "extra_forbidden"


def test_point_missing_field_raises_with_field_path() -> None:
    with pytest.raises(ValidationError) as exc:
        Point.model_validate({"x": 1})
    err = exc.value.errors()[0]
    assert err["loc"] == ("y",)
    assert err["type"] == "missing"


def test_point_round_trips_via_model_dump_and_model_validate() -> None:
    original = Point(x=1.5, y=2.5)
    dumped = original.model_dump()
    restored = Point.model_validate(dumped)
    assert restored == original


def test_point_canonical_dump_uses_x_y_keys_only() -> None:
    assert Point(x=1, y=2).model_dump(mode="json") == {"x": 1.0, "y": 2.0}


# ───────────── BoundingBox ─────────────


def test_bounding_box_construction() -> None:
    bb = BoundingBox(x=10, y=20, w=30, h=40)
    assert bb.x == 10.0
    assert bb.y == 20.0
    assert bb.w == 30.0
    assert bb.h == 40.0


@pytest.mark.parametrize("w", [0, -1, -0.5])
def test_bounding_box_rejects_non_positive_width(w: float) -> None:
    with pytest.raises(ValidationError) as exc:
        BoundingBox(x=0, y=0, w=w, h=10)
    err = exc.value.errors()[0]
    assert err["loc"] == ("w",)
    assert err["type"] == "greater_than"


@pytest.mark.parametrize("h", [0, -1, -0.5])
def test_bounding_box_rejects_non_positive_height(h: float) -> None:
    with pytest.raises(ValidationError) as exc:
        BoundingBox(x=0, y=0, w=10, h=h)
    err = exc.value.errors()[0]
    assert err["loc"] == ("h",)
    assert err["type"] == "greater_than"


def test_bounding_box_allows_negative_x_and_y() -> None:
    # Negative origin is OK; only w/h must be positive.
    bb = BoundingBox(x=-100, y=-50, w=10, h=10)
    assert bb.x == -100
    assert bb.y == -50


def test_bounding_box_is_frozen() -> None:
    bb = BoundingBox(x=0, y=0, w=10, h=10)
    with pytest.raises(ValidationError, match="frozen"):
        bb.x = 99  # type: ignore[misc]


def test_bounding_box_is_hashable_and_dedupes() -> None:
    a = BoundingBox(x=0, y=0, w=10, h=10)
    b = BoundingBox(x=0, y=0, w=10, h=10)
    c = BoundingBox(x=0, y=0, w=10, h=20)
    assert {a, b, c} == {a, c}


def test_bounding_box_extra_field_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        BoundingBox.model_validate({"x": 0, "y": 0, "w": 10, "h": 10, "rot": 45})
    assert exc.value.errors()[0]["type"] == "extra_forbidden"


def test_bounding_box_canonical_dump_uses_xywh_keys() -> None:
    dumped = BoundingBox(x=1, y=2, w=3, h=4).model_dump(mode="json")
    assert dumped == {"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0}
    # Order matters for wire format parity - assert key order, not just content.
    assert list(dumped.keys()) == ["x", "y", "w", "h"]


def test_bounding_box_int_inputs_coerced_to_float() -> None:
    bb = BoundingBox(x=1, y=2, w=3, h=4)
    assert all(isinstance(v, float) for v in (bb.x, bb.y, bb.w, bb.h))


# ───────────── NonBlankStr ─────────────


class _Wrapper(BaseModel):
    """Minimal model used to exercise NonBlankStr through Pydantic validation."""

    value: NonBlankStr


@pytest.mark.parametrize("ok", ["x", "abc", "  x  ", "x\n"])
def test_non_blank_str_accepts_strings_with_any_non_whitespace_char(ok: str) -> None:
    assert _Wrapper(value=ok).value == ok


@pytest.mark.parametrize("bad", ["", " ", "    ", "\t", "\n", "\r\n", " \t \n "])
def test_non_blank_str_rejects_blank_or_whitespace_only(bad: str) -> None:
    with pytest.raises(ValidationError) as exc:
        _Wrapper(value=bad)
    msg = str(exc.value)
    assert "blank" in msg or "whitespace" in msg


def test_non_blank_str_does_not_strip_whitespace() -> None:
    # We REJECT blanks but never silently strip - leading/trailing spaces are
    # preserved so callers see exactly what they sent. This is a deliberate
    # design choice (no surprise mutation of user input).
    assert _Wrapper(value="  x  ").value == "  x  "
