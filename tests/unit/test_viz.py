"""Tests for ``pictograph.viz.draw_annotations`` (B32c)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from pictograph import DEFAULT_PALETTE, draw_annotations
from pictograph.inference.results import ClassificationResult, ClassScore
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


# -- a classifier's classes over its threshold: badges in the image's top-left corner (B600, B602) --


_RED, _GREEN = "#ff0000", "#00ff00"


def _classified(*scores: tuple[str, float]) -> ClassificationResult:
    return ClassificationResult(classes=[ClassScore(name=n, confidence=c) for n, c in scores])


def _rgb(hex_color: str) -> tuple[int, int, int]:
    return (int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16))


def _rows(img: Image.Image, x: int, color: str) -> list[int]:
    """The rows of column ``x`` painted ``color``."""
    return [y for y in range(img.height) if img.getpixel((x, y)) == _rgb(color)]


def _cols(img: Image.Image, y: int, color: str) -> list[int]:
    """The columns of row ``y`` painted ``color``."""
    return [x for x in range(img.width) if img.getpixel((x, y)) == _rgb(color)]


def test_a_classifiers_top_class_is_a_badge_inset_in_the_top_left_corner() -> None:
    """Its class over the threshold, in its colour, inset max(8 px, 1.5% of the shorter side): the
    corner badge Pictograph's live views and annotated videos draw (pictograph-prt D-085, D-089)."""
    colors = {"cat": _RED, "dog": _GREEN}
    out = draw_annotations(
        _blank((400, 300)), classification=_classified(("cat", 0.9), ("dog", 0.1)), colors=colors
    )
    assert out.getpixel((8, 8)) == _rgb(_RED), "inset 8 px (1.5% of 300 is under 8)"
    assert out.getpixel((7, 8)) == (0, 0, 0) and out.getpixel((8, 7)) == (0, 0, 0), (
        "nothing above or left of it"
    )
    assert not [y for y in range(300) if _cols(out, y, _GREEN)], "dog's 0.1 under the 0.5 threshold"
    big = draw_annotations(
        _blank((1600, 1200)), classification=_classified(("cat", 0.9)), colors=colors
    )
    assert big.getpixel((18, 18)) == _rgb(_RED), "1.5% of the shorter side (1200)"
    assert big.getpixel((17, 18)) == (0, 0, 0) and big.getpixel((18, 17)) == (0, 0, 0)


def test_the_badge_is_sized_by_the_images_longer_side() -> None:
    """Its words ~2% of the image's longer side (font 2.75%, at least 15 px), read at a glance at any
    size: the app's image badge and the engine's live view (D-089) - never a fixed size."""
    colors = {"cat": _RED}
    heights = {}
    for size in [(400, 300), (1600, 1200), (1200, 1600)]:
        out = draw_annotations(
            _blank(size), classification=_classified(("cat", 0.9)), colors=colors
        )
        inset = max(8, round(0.015 * min(size)))
        heights[size] = len(
            _rows(out, inset + 1, _RED)
        )  # in its chip's left padding (no text there)
    small, wide, tall = heights[(400, 300)], heights[(1600, 1200)], heights[(1200, 1600)]
    assert small > 0 and wide == tall, heights  # the longer side, landscape or portrait
    # Font 15 px on the small image (its floor), round(0.0275 x 1600) = 44 on the large ones: its words
    # (the chip less its 3 px padding each side) about three times as tall.
    assert 2.4 <= (wide - 6) / (small - 6) <= 3.4, heights


def test_every_class_over_min_confidence_is_a_badge_best_first() -> None:
    """Each of its classes at or over ``min_confidence`` (default 0.5), best first; a class under it
    is no badge - a classifier always reports its best class, a weak model's guess included."""
    colors = {"cat": _RED, "dog": _GREEN}
    result = _classified(("cat", 0.6), ("dog", 0.3))
    one = draw_annotations(_blank((400, 300)), classification=result, colors=colors)
    assert _rows(one, 9, _RED) and not _rows(one, 9, _GREEN), "dog's 0.3 under 0.5"
    both = draw_annotations(
        _blank((400, 300)), classification=result, colors=colors, min_confidence=0.25
    )
    red, green = _rows(both, 9, _RED), _rows(both, 9, _GREEN)
    assert red and green and green[0] > red[-1], "dog stacked below cat"
    weak = draw_annotations(
        _blank((400, 300)), classification=_classified(("cat", 0.02)), colors=colors
    )
    assert weak.tobytes() == _blank((400, 300)).tobytes(), "a 2% guess draws nothing"


def test_several_classifiers_stack_below_one_another_each_class_once() -> None:
    colors = {"cat": _RED, "indoor": _GREEN}
    out = draw_annotations(
        _blank((400, 300)),
        classification=[
            _classified(("cat", 0.9)),
            ClassScore(name="indoor", confidence=0.7),
            {"name": "cat", "confidence": 0.6},
        ],
        colors=colors,
    )
    red, green = _rows(out, 9, _RED), _rows(out, 9, _GREEN)
    assert red[0] == 8 and red == list(range(red[0], red[-1] + 1)), (
        "one badge for cat, however often it is given"
    )
    assert green and green[0] > red[-1] + 1, "the next one below it, a gap between"


def test_the_badge_follows_show_labels_and_show_confidence() -> None:
    colors = {"cat": _RED}
    plain = draw_annotations(
        _blank((400, 300)), classification=_classified(("cat", 0.9)), colors=colors
    )
    scored = draw_annotations(
        _blank((400, 300)),
        classification=_classified(("cat", 0.9)),
        colors=colors,
        show_confidence=True,
    )
    assert _cols(scored, 9, _RED)[-1] > _cols(plain, 9, _RED)[-1], "'cat 90%' is wider than 'cat'"
    off = draw_annotations(
        _blank((400, 300)),
        classification=_classified(("cat", 0.9)),
        colors=colors,
        show_labels=False,
    )
    assert off.tobytes() == _blank((400, 300)).tobytes()


def test_an_image_with_nothing_to_draw_is_returned_as_it_was() -> None:
    assert draw_annotations(_blank()).tobytes() == _blank().tobytes()


def test_a_classification_that_is_not_one_is_refused() -> None:
    with pytest.raises(TypeError, match="classification"):
        draw_annotations(_blank(), classification=[42])  # type: ignore[list-item]
    with pytest.raises(ValidationError):
        draw_annotations(_blank(), classification={"name": "cat"})
