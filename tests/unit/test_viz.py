"""Tests for ``pictograph.viz.draw_annotations`` (B32c)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from pictograph import DEFAULT_PALETTE, draw_annotations
from pictograph.viz import _color_for


def _blank(size: tuple[int, int] = (200, 150)) -> Image.Image:
    return Image.new("RGB", size, color=(0, 0, 0))


def _bbox(name: str = "car") -> dict:
    return {
        "id": "a1",
        "name": name,
        "type": "bbox",
        "bounding_box": {"x": 10, "y": 20, "w": 50, "h": 40},
    }


def _polygon() -> dict:
    return {
        "id": "a2",
        "name": "road",
        "type": "polygon",
        "polygon": {"paths": [[{"x": 5, "y": 5}, {"x": 60, "y": 5}, {"x": 30, "y": 60}]]},
    }


def _polyline() -> dict:
    return {
        "id": "a3",
        "name": "lane",
        "type": "polyline",
        "polyline": {"path": [{"x": 0, "y": 0}, {"x": 100, "y": 100}]},
    }


def _keypoint() -> dict:
    return {"id": "a4", "name": "joint", "type": "keypoint", "keypoint": {"x": 80, "y": 90}}


def test_color_is_stable_and_in_palette() -> None:
    c1 = _color_for("car", None)
    c2 = _color_for("car", None)
    assert c1 == c2  # deterministic
    assert c1 in DEFAULT_PALETTE
    # Override wins.
    assert _color_for("car", {"car": "#123456"}) == "#123456"


def test_draw_all_types_returns_new_rgb_image() -> None:
    base = _blank()
    out = draw_annotations(base, [_bbox(), _polygon(), _polyline(), _keypoint()])
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    assert out.size == base.size
    # Something was actually drawn (the all-black input gained colored pixels).
    assert out.getextrema() != ((0, 0), (0, 0), (0, 0))


def test_input_image_not_mutated() -> None:
    base = _blank()
    before = base.tobytes()
    draw_annotations(base, [_bbox()])
    assert base.tobytes() == before  # the original is untouched


def test_accepts_path_input(tmp_path: Path) -> None:
    p = tmp_path / "img.png"
    _blank().save(p)
    out = draw_annotations(p, [_bbox()], show_labels=False)
    assert out.size == (200, 150)


def test_accepts_model_instances() -> None:
    from pydantic import TypeAdapter

    from pictograph.models.annotation import Annotation

    ann = TypeAdapter(Annotation).validate_python(_bbox())
    out = draw_annotations(_blank(), [ann])
    assert isinstance(out, Image.Image)


def test_color_override_applies() -> None:
    out = draw_annotations(_blank(), [_bbox("car")], colors={"car": "#ff0000"}, show_labels=False)
    # The red override (RGB 255,0,0) should appear among the drawn pixels.
    assert b"\xff\x00\x00" in out.tobytes(), "expected the overridden red outline to be drawn"


def test_invalid_annotation_raises() -> None:
    # A bbox annotation missing its `bounding_box` fails union validation.
    with pytest.raises(ValidationError):
        draw_annotations(_blank(), [{"id": "x", "name": "y", "type": "bbox"}])


def test_show_confidence_augments_the_label() -> None:
    base = _blank()
    ann = {**_bbox(), "confidence": 0.5}
    with_conf = draw_annotations(base, [ann], show_confidence=True)
    without = draw_annotations(base, [ann], show_confidence=False)
    # "car 50%" draws different pixels than "car"; both are valid non-blank images.
    assert with_conf.tobytes() != without.tobytes()
    assert with_conf.mode == "RGB"
    assert with_conf.getextrema() != ((0, 0), (0, 0), (0, 0))


def test_show_confidence_is_a_noop_without_labels() -> None:
    base = _blank()
    ann = {**_bbox(), "confidence": 0.5}
    a = draw_annotations(base, [ann], show_labels=False, show_confidence=True)
    b = draw_annotations(base, [ann], show_labels=False, show_confidence=False)
    assert a.tobytes() == b.tobytes()  # no label drawn → confidence flag has no effect
