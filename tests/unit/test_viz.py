"""Tests for ``pictograph.viz.draw_annotations`` - THE annotation look (B613).

Pixel-exact where the look is exact (a box's crisp line, its fill, a label's color), and
pinned to the app where the platform is one (the class-color hash, the label-ink rule).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError

from pictograph import DEFAULT_PALETTE, draw_annotations
from pictograph._look.render import (
    INK,
    INK_INVERSE,
    PALETTE,
    PAPER,
    WHITE,
    _words,
    contrast,
    label_ink,
    mix,
    shape_center,
    sizes_for,
)
from pictograph.inference.results import ClassificationResult, ClassScore
from pictograph.viz import LOOK_PRESETS, Look, class_color

GREY = (90, 96, 104)
GREEN = "#30A941"  # THE palette's Green: dark words
GOLD = "#ffd54f"  # THE palette's Gold: dark words
PURPLE = "#722f8f"  # THE palette's Product Purple: white words


def _img(size: tuple[int, int] = (400, 300), color: tuple[int, int, int] = GREY) -> Image.Image:
    return Image.new("RGB", size, color=color)


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _over(bg: tuple[int, int, int], color: str, alpha: float) -> tuple[int, int, int]:
    """``color`` at ``alpha`` over ``bg``."""
    c = _rgb(color)
    a = round(alpha * 255) / 255
    return tuple(round(b + (x - b) * a) for b, x in zip(bg, c, strict=True))  # type: ignore[return-value]


def _near(px: Any, rgb: tuple[int, int, int], tol: int = 3) -> bool:
    return all(abs(int(p) - int(q)) <= tol for p, q in zip(px, rgb, strict=False))


def _box(
    x: float = 100, y: float = 100, w: float = 120, h: float = 80, name: str = "car", **extra: Any
) -> dict[str, Any]:
    return {
        "id": "b",
        "name": name,
        "type": "bbox",
        "bounding_box": {"x": x, "y": y, "w": w, "h": h},
        **extra,
    }


def _polygon(name: str = "person", hole: bool = False, **extra: Any) -> dict[str, Any]:
    paths = [
        [{"x": 100, "y": 100}, {"x": 300, "y": 100}, {"x": 300, "y": 260}, {"x": 100, "y": 260}]
    ]
    if hole:
        paths.append(
            [{"x": 170, "y": 150}, {"x": 230, "y": 150}, {"x": 230, "y": 210}, {"x": 170, "y": 210}]
        )
    return {"id": "p", "name": name, "type": "polygon", "polygon": {"paths": paths}, **extra}


def _chip(
    img: Image.Image, glass: str, region: tuple[int, int, int, int] | None = None
) -> tuple[int, int, int, int]:
    """The box (x0, y0, x1, y1) of the pixels exactly ``glass`` in ``region``: a label's chip."""
    x0, y0, x1, y1 = region or (0, 0, img.width, img.height)
    want = _rgb(glass)
    xs, ys = [], []
    px = img.load()
    assert px is not None
    for y in range(y0, y1):
        for x in range(x0, x1):
            if px[x, y] == want:
                xs.append(x)
                ys.append(y)
    assert xs, f"no {glass} label in {region}"
    return min(xs), min(ys), max(xs), max(ys)


def _glass(color: str) -> str:
    """A label's color for a class color in THE look (85% class color, 15% paper)."""
    return label_ink(mix(color, PAPER, 0.15))[0]


# ── The palette: THE Pictograph palette, by the app's hash ─────────────────────────────────────


APP_VECTORS: list[tuple[str, str, dict[str, str] | None]] = [
    # The app's own vectors (its class-color test pins the same names), so the SDK hashes as the app does.
    ("car", "#ed3e49", None),
    ("truck", "#ff9800", None),
    ("person", "#00e5ff", None),
    ("bus", "#ed3e49", None),
    ("bicycle", "#4dd0e1", None),
    ("traffic light", "#30A941", None),
    ("forklift", "#00e5ff", None),
    ("pallet", "#26c6da", None),
    ("Car", "#5AE29E", None),
    ("a", "#722f8f", None),
    ("z", "#ff5e24", None),
    ("has_pineapple", "#00e5ff", None),
    ("left_eye", "#722f8f", None),
    ("Défaut", "#ff9800", None),
    ("überholt", "#42a5f5", None),
    ("猫", "#ff9800", None),
    ("dog \U0001f415", "#ffab91", None),
    ("\U0001f697", "#64ffda", None),
    ("a-very-long-class-name-with-many-parts_v2", "#26AD81", None),
    ("stop sign", "#4dd0e1", None),
    ("0", "#5AE29E", None),
    ("Person", "#ec407a", None),
    ("nose", "#30A941", None),
    ("Grüße", "#4dd0e1", None),
    ("person", "#2c5aa0", {"person": "#2c5aa0", "forklift": "#722f8f"}),
    ("forklift", "#722f8f", {"person": "#2c5aa0", "forklift": "#722f8f"}),
    ("car", "#ed3e49", {"person": "#2c5aa0", "forklift": "#722f8f"}),
]


def test_the_palette_is_the_pictograph_palette() -> None:
    assert DEFAULT_PALETTE == PALETTE
    assert len(PALETTE) == 20
    assert PALETTE[0] == "#ed3e49" and PALETTE[13] == "#00e5ff" and PALETTE[19] == "#ba68c8"


@pytest.mark.parametrize(("name", "color", "stored"), APP_VECTORS)
def test_a_class_is_the_color_the_app_draws_it_in(
    name: str, color: str, stored: dict[str, str] | None
) -> None:
    assert class_color(name, stored) == color


# ── The label's words: ink or white, whichever has the higher contrast ratio ─────────────────


@pytest.mark.parametrize("color", PALETTE)
def test_every_palette_label_reads(color: str) -> None:
    glass, ink = label_ink(mix(color, PAPER, 0.15))
    assert glass == mix(color, PAPER, 0.15)  # the label keeps its color
    assert contrast(ink, glass) >= 4.0
    assert contrast(ink, glass) >= contrast(INK if ink == INK_INVERSE else INK_INVERSE, glass)


# The rule's answers on THE palette: white words on Product Blue and Product Purple only (chosen 2026-10-31).
WHITE_WORDS = {"#3e62b0", "#722f8f"}


@pytest.mark.parametrize("color", PALETTE)
def test_dark_words_on_every_palette_color_but_two(color: str) -> None:
    _, ink = label_ink(mix(color, PAPER, 0.15))
    assert ink == (INK_INVERSE if color in WHITE_WORDS else INK)


def test_a_labels_words_name_its_track_and_score_independently() -> None:
    look = Look()
    # THE look's label: its name and its track id - no score (Noah, 2026-10-31).
    assert (_words(look, "person", 0.91, 7).name, _words(look, "person", 0.91, 7).value) == (
        "person",
        "7",
    )
    scored = Look(label={"parts": {"confidence": True}})  # type: ignore[arg-type]
    scored_no_id = Look(label={"parts": {"confidence": True, "track": False}})  # type: ignore[arg-type]
    assert _words(scored, "person", 0.91, 7).value == "7 91%"
    assert _words(scored_no_id, "person", 0.91, 7).value == "91%"
    assert _words(look, "person", None, None).value == ""


# ── Drawing: a new image, every type ──────────────────────────────────────────────────────────


def test_every_type_draws_on_a_new_rgb_image_and_the_input_is_untouched() -> None:
    base = _img()
    before = base.tobytes()
    anns = [
        _box(),
        _polygon(hole=True),
        {
            "id": "l",
            "name": "lane",
            "type": "polyline",
            "polyline": {"path": [{"x": 10, "y": 290}, {"x": 390, "y": 200}]},
        },
        {"id": "k", "name": "head", "type": "keypoint", "keypoint": {"x": 350, "y": 60}},
    ]
    out = draw_annotations(base, anns)
    assert out is not base and out.mode == "RGB" and out.size == base.size
    assert base.tobytes() == before
    assert out.tobytes() != before


def test_a_path_and_model_instances_are_accepted(tmp_path: Path) -> None:
    from pictograph.models.annotation import BBoxAnnotation

    path = tmp_path / "img.png"
    _img().save(path)
    ann = BBoxAnnotation.model_validate(_box())
    assert draw_annotations(path, [ann]).size == (400, 300)
    assert draw_annotations(str(path), [_box()]).size == (400, 300)


def test_an_invalid_annotation_is_refused() -> None:
    with pytest.raises(ValidationError):
        draw_annotations(_img(), [{"id": "x", "name": "car", "type": "bbox"}])


def test_nothing_to_draw_returns_the_image_as_it_was() -> None:
    base = _img()
    assert draw_annotations(base, []).tobytes() == base.tobytes()


# ── A box: a crisp line, filled 20% - no keyline ─────────────────────────────────────────────


def test_a_box_is_a_crisp_line_filled_with_its_color() -> None:
    out = draw_annotations(_img(), [_box()], colors={"car": "#ed3e49"}, show_labels=False)
    # 400 x 300: the 2 px line straddles x = 100 (pixels 99 and 100), straight on the photo (no keyline).
    assert out.getpixel((99, 140)) == _rgb("#ed3e49")
    assert out.getpixel((100, 140)) == _rgb("#ed3e49")
    assert out.getpixel((98, 140)) == GREY
    # Inside: the class color lifted 15% toward white, at 20%.
    assert _near(out.getpixel((101, 140)), _over(GREY, mix("#ed3e49", WHITE, 0.15), 0.2))
    assert _near(out.getpixel((160, 140)), _over(GREY, mix("#ed3e49", WHITE, 0.15), 0.2))


def test_a_box_fill_and_keyline_follow_the_look() -> None:
    out = draw_annotations(
        _img(),
        [_box()],
        colors={"car": "#ed3e49"},
        show_labels=False,
        look={"box": {"fill": 0, "keyline": {"alpha": 0.42}}},
    )
    assert out.getpixel((160, 140)) == GREY
    # A keyline given: ink at its alpha, a hairline either side of the line.
    assert _near(out.getpixel((98, 140)), _over(GREY, INK, 0.42))
    assert out.getpixel((97, 140)) == GREY
    assert out.getpixel((99, 140)) == _rgb("#ed3e49")


def test_a_width_sets_every_line_in_pixels() -> None:
    out = draw_annotations(
        _img(),
        [_box()],
        colors={"car": "#ed3e49"},
        show_labels=False,
        width=4,
        look={"box": {"keyline": None, "fill": 0}},
    )
    reds = [x for x in range(90, 110) if out.getpixel((x, 140)) == _rgb("#ed3e49")]
    assert reds == [98, 99, 100, 101]


def test_a_rotated_box_is_drawn_rotated() -> None:
    ann = _box(
        80, 80, 240, 140, oriented_box={"cx": 200, "cy": 150, "w": 200, "h": 60, "angle": 30}
    )
    out = draw_annotations(
        _img(), [ann], colors={"car": "#3e62b0"}, show_labels=False, look={"box": {"fill": 0}}
    )
    # The middle of its top edge (rotated 30 degrees about the center) is on the line; the enclosure's corner is not.
    import math

    a = math.radians(30)
    mx, my = 200 + 30 * math.sin(a), 150 - 30 * math.cos(a)
    assert _near(out.getpixel((round(mx), round(my))), _rgb("#3e62b0"), tol=40)
    assert out.getpixel((82, 82)) == GREY


def test_brackets_draw_heavy_corners_over_a_hairline() -> None:
    square = draw_annotations(_img(), [_box()], show_labels=False, look={"box": {"fill": 0}})
    brackets = draw_annotations(
        _img(), [_box()], show_labels=False, look={"box": {"fill": 0, "corners": "brackets"}}
    )
    assert square.tobytes() != brackets.tobytes()


# ── A polygon: filled 28% under a crisp rim, holes stay open ─────────────────────────────────


def test_a_polygon_is_filled_under_its_rim_and_its_hole_stays_open() -> None:
    out = draw_annotations(
        _img(), [_polygon(hole=True)], colors={"person": "#00e5ff"}, show_labels=False
    )
    assert _near(out.getpixel((130, 180)), _over(GREY, mix("#00e5ff", WHITE, 0.15), 0.28))
    assert out.getpixel((200, 180)) == GREY  # the hole
    assert _near(out.getpixel((200, 100)), _rgb("#00e5ff"), tol=12)  # on its top edge


def test_polygon_fills_never_stack_where_they_overlap() -> None:
    a = _polygon()
    b = {**_polygon(), "id": "p2"}
    one = draw_annotations(_img(), [a], show_labels=False)
    two = draw_annotations(_img(), [a, b], show_labels=False)
    assert one.getpixel((150, 200)) == two.getpixel((150, 200))


def test_a_polyline_and_a_keypoint() -> None:
    lane = {
        "id": "l",
        "name": "lane",
        "type": "polyline",
        "polyline": {"path": [{"x": 20, "y": 150}, {"x": 380, "y": 150}]},
    }
    head = {"id": "k", "name": "head", "type": "keypoint", "keypoint": {"x": 200, "y": 60}}
    out = draw_annotations(
        _img(), [lane, head], colors={"lane": "#ffc540", "head": "#ec407a"}, show_labels=False
    )
    assert out.getpixel((200, 150)) == _rgb("#ffc540")
    assert out.getpixel((200, 60)) == _rgb("#ec407a")  # a solid point: its color at its center
    assert (
        out.getpixel((200, 63)) != GREY
    )  # its keyline disk around it (400 x 300: radius 3, the keyline to 4)
    assert out.getpixel((200, 66)) == GREY


def test_a_point_off_the_image_space_is_skipped() -> None:
    nan = {"id": "k", "name": "head", "type": "keypoint", "keypoint": {"x": float("nan"), "y": 5}}
    base = _img()
    assert draw_annotations(base, [nan]).tobytes() == base.tobytes()


# ── Poses: joints connected per person, one palette color each ───────────────────────────────

TEMPLATE = {"person": {"nodes": [{"name": "head"}, {"name": "hip"}], "edges": [[0, 1]]}}


def _joint(name: str, x: float, y: float, instance: int) -> dict[str, Any]:
    return {
        "id": f"{name}{instance}",
        "name": name,
        "type": "keypoint",
        "keypoint": {"x": x, "y": y},
        "instance_id": instance,
    }


def test_a_template_connects_one_persons_joints_in_their_color() -> None:
    anns = [
        _joint("head", 100, 60, 1),
        _joint("hip", 100, 200, 1),
        _joint("head", 300, 60, 2),
        _joint("hip", 300, 200, 2),
    ]
    out = draw_annotations(_img(), anns, show_labels=False, keypoint_templates=TEMPLATE)
    assert out.getpixel((100, 130)) == _rgb(PALETTE[0])  # person 1's limb
    assert out.getpixel((300, 130)) == _rgb(PALETTE[1])  # person 2's
    assert out.getpixel((200, 130)) == GREY  # never joined across people
    assert out.getpixel((100, 60)) == _rgb(PALETTE[0])


# ── Labels: the class color as an opaque label at the top middle ─────────────────────────────


def test_a_label_sits_at_the_top_middle_of_its_box() -> None:
    out = draw_annotations(_img(), [_box()], colors={"car": GREEN})
    x0, y0, x1, y1 = _chip(out, _glass(GREEN), (0, 0, 400, 100))
    assert abs((x0 + x1) / 2 - 160) <= 2  # centered over the box (x 100 - 220)
    assert y1 < 100 and y1 >= 95  # just above its top edge
    assert y1 - y0 + 1 == 18  # 400 x 300: the 12 px floor, 1.5 em tall


def test_a_label_reads_dark_on_green_and_gold_white_on_purple() -> None:
    for color, ink in ((GREEN, INK), (GOLD, INK), (PURPLE, INK_INVERSE)):
        out = draw_annotations(_img(), [_box()], colors={"car": color})
        x0, y0, x1, y1 = _chip(out, _glass(color), (0, 0, 400, 100))
        inked = [
            out.getpixel((x, y))
            for x in range(x0, x1)
            for y in range(y0, y1)
            if _near(out.getpixel((x, y)), _rgb(ink), 24)
        ]
        assert inked, f"{color}: no words in {ink}"


def test_a_label_follows_the_image_size() -> None:
    def height(size: tuple[int, int]) -> int:
        w, h = size
        k = w / 400
        out = draw_annotations(
            _img(size), [_box(100 * k, 100 * k + 40, 120 * k, 80 * k)], colors={"car": GREEN}
        )
        _, y0, _, y1 = _chip(out, _glass(GREEN))
        return y1 - y0 + 1

    assert height((1920, 1080)) == 30  # 20 px text at 1080p
    assert height((1280, 720)) == 24  # 15.7 px at 720p


def test_font_size_sets_the_label_text() -> None:
    out = draw_annotations(_img(), [_box()], colors={"car": GREEN}, font_size=30)
    _, y0, _, y1 = _chip(out, _glass(GREEN))
    assert y1 - y0 + 1 == 45


def test_with_no_room_above_a_label_sits_inside() -> None:
    out = draw_annotations(_img(), [_box(10, 5, 120, 80)], colors={"car": GREEN})
    _, y0, _, _ = _chip(out, _glass(GREEN))
    assert y0 >= 5


def test_show_labels_false_draws_no_label() -> None:
    out = draw_annotations(_img(), [_box()], colors={"car": GREEN}, show_labels=False)
    assert all(out.getpixel((x, 88)) != _rgb(_glass(GREEN)) for x in range(400))


def _label_width(**kwargs: Any) -> int:
    ann = kwargs.pop("ann", _box())
    out = draw_annotations(_img(), [ann], colors={"car": GREEN}, **kwargs)
    x0, _, x1, _ = _chip(out, _glass(GREEN), (0, 0, 400, 100))
    return x1 - x0


def test_a_score_shows_only_when_asked_for() -> None:
    person = _label_width(ann=_box())
    assert _label_width(ann=_box(confidence=0.73)) == person  # THE look: no score
    assert _label_width(ann=_box(confidence=0.73), show_confidence=True) > person
    assert _label_width(ann=_box(), show_confidence=True) > person  # "car 100%"
    # The look's own switch shows a prediction's score, not a person's label (1.0).
    scored = {"label": {"parts": {"confidence": True}}}
    assert _label_width(ann=_box(confidence=0.73), look=scored) > person
    assert _label_width(ann=_box(), look=scored) == person


def test_a_tracked_detection_shows_its_id() -> None:
    assert _label_width(ann=_box(attributes={"track_id": 7})) > _label_width(ann=_box())
    assert _label_width(ann=_box(attributes={"track_id": "12"})) > _label_width(ann=_box())
    assert _label_width(ann=_box(attributes={"track_id": True})) == _label_width(ann=_box())
    assert _label_width(ann=_box(attributes=["track_id"])) == _label_width(ann=_box())


def test_track_ids_and_scores_switch_independently() -> None:
    both = _box(confidence=0.73, attributes={"track_id": 7})
    full = _label_width(ann=both, show_confidence=True)
    no_id = _label_width(ann=both, show_track_ids=False, show_confidence=True)
    no_score = _label_width(ann=both)
    neither = _label_width(ann=both, show_track_ids=False)
    assert full > no_id > neither and full > no_score > neither
    assert _label_width(ann=both, show_confidence=False) == no_score
    assert _label_width(ann=both, look={"label": {"parts": {"track": False}}}) == neither


def test_a_polygon_label_sits_over_its_top_or_at_its_center() -> None:
    out = draw_annotations(_img(), [_polygon()], colors={"person": GREEN})
    x0, _, x1, y1 = _chip(out, _glass(GREEN))
    assert abs((x0 + x1) / 2 - 200) <= 2 and y1 < 100
    out = draw_annotations(
        _img(), [_polygon()], colors={"person": GREEN}, look={"label": {"mask_position": "cc"}}
    )
    x0, y0, x1, y1 = _chip(out, _glass(GREEN))
    assert abs((x0 + x1) / 2 - 200) <= 2 and abs((y0 + y1) / 2 - 180) <= 2


def test_a_shapes_center_is_inside_it() -> None:
    # A C shape: its box's middle is outside it; its center is not.
    ring = [
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 20.0),
        (20.0, 20.0),
        (20.0, 80.0),
        (100.0, 80.0),
        (100.0, 100.0),
        (0.0, 100.0),
    ]
    x, y = shape_center([ring])
    assert x < 20 and 0 < y < 100


def test_a_class_can_drop_its_label_or_take_its_own_color_and_place() -> None:
    off = draw_annotations(_img(), [_box()], look={"classes": {"car": {"label": False}}})
    with pytest.raises(AssertionError):
        _chip(off, _glass(class_color("car")), (0, 0, 400, 100))
    own = draw_annotations(
        _img(), [_box()], look={"classes": {"car": {"color": GOLD, "position": "tl"}}}
    )
    x0, _, _, _ = _chip(own, _glass(GOLD), (0, 0, 400, 100))
    assert x0 == 100  # its left edge on the box's


def test_where_labels_meet_hide_leaves_the_weaker_out() -> None:
    a = _box(100, 100, 60, 60, confidence=0.9)
    b = {**_box(110, 102, 60, 60, confidence=0.6), "id": "b2"}
    drawn = draw_annotations(_img(), [a, b], colors={"car": GREEN})
    hidden = draw_annotations(
        _img(), [a, b], colors={"car": GREEN}, look={"label": {"overlap": "hide"}}
    )
    shifted = draw_annotations(
        _img(), [a, b], colors={"car": GREEN}, look={"label": {"overlap": "shift"}}
    )
    glass = _rgb(_glass(GREEN))

    def labelled(img: Image.Image) -> int:
        return sum(1 for x in range(400) for y in range(300) if img.getpixel((x, y)) == glass)

    assert labelled(hidden) < labelled(drawn)
    assert labelled(shifted) > labelled(hidden)


@pytest.mark.parametrize("style", ["tab", "inset", "split", "leader", "text"])
def test_every_label_form_draws(style: str) -> None:
    # A score on the label, so a split label has its two parts.
    chip = draw_annotations(
        _img(), [_box(confidence=0.8)], colors={"car": GREEN}, show_confidence=True
    )
    other = draw_annotations(
        _img(),
        [_box(confidence=0.8)],
        colors={"car": GREEN},
        show_confidence=True,
        look={"label": {"style": style}},
    )
    assert other.tobytes() != chip.tobytes()


@pytest.mark.parametrize("typeface", ["pair", "mono"])
def test_the_mono_typefaces_draw(typeface: str) -> None:
    # A score on the label: "pair" sets only the numbers in mono.
    sans = draw_annotations(
        _img(), [_box(confidence=0.8)], colors={"car": GREEN}, show_confidence=True
    )
    other = draw_annotations(
        _img(),
        [_box(confidence=0.8)],
        colors={"car": GREEN},
        show_confidence=True,
        look={"label": {"typeface": typeface}},
    )
    assert other.tobytes() != sans.tobytes()


# ── The look: presets and partial configs ────────────────────────────────────────────────────


def test_the_default_look_is_color_frost() -> None:
    assert Look() == LOOK_PRESETS["color"]
    assert Look().box.fill == 0.2 and Look().fill.opacity == 0.28 and Look().point.style == "solid"
    assert (
        Look().label.alpha == 1.0
        and Look().label.position == "tc"
        and Look().label.overlap == "draw"
    )


def test_a_preset_by_name_and_a_partial_config() -> None:
    smoke = draw_annotations(_img(), [_box()], colors={"car": GREEN}, look="smoke")
    # Smoke's label: graphite with 22% of the class color, at 80% over the photo.
    want = _over(GREY, label_ink(mix(INK, GREEN, 0.22))[0], 0.8)
    hits = sum(1 for x in range(400) for y in range(100) if _near(smoke.getpixel((x, y)), want, 2))
    assert hits > 100
    as_model = draw_annotations(_img(), [_box()], look=Look(box={"fill": 0}))  # type: ignore[arg-type]
    as_dict = draw_annotations(_img(), [_box()], look={"box": {"fill": 0}})
    assert as_model.tobytes() == as_dict.tobytes()


def test_an_unknown_preset_or_field_is_refused() -> None:
    with pytest.raises(ValueError, match="no look named"):
        draw_annotations(_img(), [_box()], look="neon")
    with pytest.raises(ValidationError):
        draw_annotations(_img(), [_box()], look={"box": {"glow": 1}})
    with pytest.raises(ValidationError):
        draw_annotations(_img(), [_box()], look={"label": {"typeface": "hershey"}})


def test_point_styles_and_skeleton_color_by_class() -> None:
    head = {"id": "k", "name": "head", "type": "keypoint", "keypoint": {"x": 200, "y": 150}}

    def styled(style: str, keyline: dict[str, Any] | None) -> bytes:
        look = {"point": {"style": style, "radius": 12}, "box": {"keyline": keyline}}
        return draw_annotations(_img(), [head], show_labels=False, look=look).tobytes()

    keyed = {s: styled(s, {"alpha": 0.42}) for s in ("solid", "paper", "glass", "ring", "dot")}
    assert len(set(keyed.values())) == 5
    # Without a keyline (THE look) a solid point is its disk alone: a dot.
    assert styled("solid", None) == styled("dot", None)
    anns = [_joint("head", 100, 60, 1), _joint("hip", 100, 200, 1)]
    out = draw_annotations(
        _img(),
        anns,
        show_labels=False,
        keypoint_templates=TEMPLATE,
        look={"bone": {"color": "class"}},
        colors={"person": GOLD},
    )
    assert out.getpixel((100, 130)) == _rgb(GOLD)


# ── A classifier's classes: badges in the corner ─────────────────────────────────────────────


def _scores(*pairs: tuple[str, float]) -> ClassificationResult:
    return ClassificationResult(classes=[ClassScore(name=n, confidence=c) for n, c in pairs])


def test_a_classifiers_classes_are_badges_stacked_in_the_top_left_corner() -> None:
    out = draw_annotations(
        _img(),
        classification=_scores(("cat", 0.9), ("dog", 0.6), ("bird", 0.2)),
        colors={"cat": GREEN, "dog": GOLD},
    )
    cx0, cy0, _, cy1 = _chip(out, _glass(GREEN))
    dx0, dy0, _, _ = _chip(out, _glass(GOLD))
    assert (cx0, cy0) == (8, 8)  # inset 8 px
    assert cy1 - cy0 + 1 == 23  # a label's size x 1.25
    assert dx0 == 8 and dy0 > cy1  # stacked below, best first
    s = sizes_for(400, 300, Look())
    assert s.badge_em == 15.0


def test_badges_follow_min_confidence_show_labels_and_corner() -> None:
    result = _scores(("cat", 0.9))
    with pytest.raises(AssertionError):
        _chip(
            draw_annotations(
                _img(), classification=result, colors={"cat": GREEN}, min_confidence=0.95
            ),
            _glass(GREEN),
        )
    with pytest.raises(AssertionError):
        _chip(
            draw_annotations(
                _img(), classification=result, colors={"cat": GREEN}, show_labels=False
            ),
            _glass(GREEN),
        )
    out = draw_annotations(
        _img(), classification=result, colors={"cat": GREEN}, look={"badge": {"corner": "br"}}
    )
    _, _, x1, y1 = _chip(out, _glass(GREEN))
    assert x1 > 300 and y1 > 250


def test_several_classifiers_each_class_once_and_mappings() -> None:
    out = draw_annotations(
        _img(),
        classification=[
            _scores(("cat", 0.9)),
            ClassScore(name="cat", confidence=0.8),
            {"name": "dog", "confidence": 0.7},
        ],
        colors={"cat": GREEN, "dog": GOLD},
    )
    cats = _chip(out, _glass(GREEN))
    dogs = _chip(out, _glass(GOLD))
    assert dogs[1] > cats[3]


def test_a_classification_that_is_not_one_is_refused() -> None:
    with pytest.raises(TypeError, match="classification takes"):
        draw_annotations(_img(), classification=[42])  # type: ignore[list-item]


# ── Big images and the fonts ─────────────────────────────────────────────────────────────────


def test_a_4k_image_draws() -> None:
    out = draw_annotations(
        _img((3840, 2160)), [_box(1000, 1000, 600, 400, confidence=0.9)], colors={"car": GREEN}
    )
    _, y0, _, y1 = _chip(out, _glass(GREEN), (900, 800, 1800, 1000))
    assert y1 - y0 + 1 == 45  # 30.3 px text at 4K


def test_the_fonts_ship_with_the_sdk() -> None:
    from importlib import resources

    fonts = resources.files("pictograph._look") / "fonts"
    names = {p.name for p in fonts.iterdir()}
    assert {
        "OpenSans-SemiBold.ttf",
        "OpenSans-Regular.ttf",
        "OxygenMono-Regular.ttf",
        "OpenSans-OFL.txt",
        "OxygenMono-OFL.txt",
    } <= names
