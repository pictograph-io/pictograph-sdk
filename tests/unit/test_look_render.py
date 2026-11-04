"""THE annotation look's renderer, below ``draw_annotations``: the shapes no annotation type makes on its own (a
semantic region), every label place and form, and the edges (B613)."""

from __future__ import annotations

import pytest
from PIL import Image

from pictograph._look.config import Look
from pictograph._look.render import (
    INK,
    INK_INVERSE,
    PAPER,
    Badge,
    Bone,
    Rect,
    Shape,
    _hex,
    _local_mask,
    _paint_for,
    contrast,
    mask_top,
    mix,
    render,
    shade_on,
    shape_center,
    sizes_for,
)

GREY = (90, 96, 104)


def _img(size: tuple[int, int] = (400, 300)) -> Image.Image:
    return Image.new("RGB", size, GREY)


SQUARE = [[(100.0, 100.0), (300.0, 100.0), (300.0, 260.0), (100.0, 260.0)]]


def test_hex_reads_short_forms_and_refuses_the_rest() -> None:
    assert _hex("#fff") == (255, 255, 255)
    assert _hex(" 0a0b0c ") == (10, 11, 12)
    with pytest.raises(ValueError, match="not a hex color"):
        _hex("#12345")


def test_shade_words_meet_contrast_on_light_and_dark_glass() -> None:
    for glass in (PAPER, INK, mix(PAPER, "#ffd54f", 0.18)):
        for color in ("#ffd54f", "#3e62b0", "#ed3e49"):
            assert contrast(shade_on(color, glass), glass) >= 4.5
    # A color that reaches it in no shade falls back to the pole: white on a mid grey (4.48:1).
    assert shade_on("#ffffff", "#777777") == "#ffffff"


def test_shade_ink_and_smoke_paint() -> None:
    frost = Look(label={"ink": "shade", "tint": 0.18})  # type: ignore[arg-type]
    p = _paint_for(frost, "#30A941")
    assert p.name not in (INK, INK_INVERSE) and contrast(p.name, p.glass) >= 4.5
    smoke = Look(label={"glass": "smoke", "tint": 0.22})  # type: ignore[arg-type]
    assert _paint_for(smoke, "#30A941").name == INK_INVERSE


def test_sizes_floor_and_zero_widths() -> None:
    s = sizes_for(640, 360, Look())
    assert s.em == 12 and s.px(0) == 0 and s.rad(0) == 0 and s.px(2) == 2 and s.rad(4.5) == 3
    assert sizes_for(640, 360, Look(), stroke=5).px(2) == 5
    linear = sizes_for(3840, 2160, Look(scale={"rule": "linear"}))  # type: ignore[arg-type]
    assert linear.em == 40


def test_shape_center_and_mask_top_on_degenerate_shapes() -> None:
    assert shape_center([]) == (0.0, 0.0)
    assert shape_center([[(5.0, 5.0), (5.0, 9.0), (5.0, 12.0)]]) == (5.0, 5.0)  # no area
    box = Rect(0, 0, 10, 10)
    assert mask_top([], box) is box
    assert _local_mask([[(1.0, 1.0), (2.0, 2.0)]], 2) is None  # a ring of two points is no shape


def test_a_semantic_region_is_filled_and_edged_and_labelled_over_its_top() -> None:
    region = Shape("region", "road", rings=SQUARE, box=(100, 100, 200, 160))
    out = render(_img(), [region], cfg=Look(), stored_colors={"road": "#7986cb"})
    inside = out.getpixel((150, 200))
    assert inside != GREY  # filled
    assert out.getpixel((150, 99)) != GREY or out.getpixel((150, 100)) != GREY  # its edge
    no_edge = render(
        _img(), [region], cfg=Look(region={"outline": 0}), stored_colors={"road": "#7986cb"}
    )  # type: ignore[arg-type]
    assert no_edge.tobytes() != out.tobytes()
    unfilled = render(
        _img(), [region], cfg=Look(region={"opacity": 0}), stored_colors={"road": "#7986cb"}
    )  # type: ignore[arg-type]
    assert unfilled.getpixel((150, 200)) == GREY


@pytest.mark.parametrize("place", ["tl", "tc", "tr", "cl", "cc", "cr", "bl", "bc", "br"])
def test_every_label_place_draws(place: str) -> None:
    box = Shape("box", "car", 0.9, box=(150, 120, 100, 80))
    poly = Shape("polygon", "person", 0.8, rings=SQUARE, box=(100, 100, 200, 160))
    look = Look(label={"position": place, "mask_position": place})  # type: ignore[arg-type]
    out = render(_img(), [box, poly], cfg=look)
    assert out.tobytes() != _img().tobytes()


def test_a_label_beside_a_box_at_the_frames_edge_turns_inside() -> None:
    near_left = Shape("box", "car", 0.9, box=(2, 120, 100, 80))
    near_right = Shape("box", "car", 0.9, box=(298, 120, 100, 80))
    for shape, place in ((near_left, "cl"), (near_right, "cr")):
        out = render(_img(), [shape], cfg=Look(label={"position": place}))  # type: ignore[arg-type]
        assert out.size == (400, 300)


def test_bottom_labels_flip_inside_at_the_frames_bottom() -> None:
    low = Shape("box", "car", 0.9, box=(150, 250, 100, 48))
    out = render(_img(), [low], cfg=Look(label={"position": "bc"}))  # type: ignore[arg-type]
    inset = render(_img(), [low], cfg=Look(label={"position": "bc", "style": "inset"}))  # type: ignore[arg-type]
    # No room under it: the label sits inside its bottom edge - where an inset label always sits.
    assert out.tobytes() != _img().tobytes() and inset.tobytes() == out.tobytes()


def test_tab_labels_sit_on_the_top_or_bottom_edge() -> None:
    box = Shape("box", "car", 0.9, box=(150, 120, 100, 80))
    top = render(_img(), [box], cfg=Look(label={"style": "tab"}))  # type: ignore[arg-type]
    bottom = render(_img(), [box], cfg=Look(label={"style": "tab", "position": "bc"}))  # type: ignore[arg-type]
    assert top.tobytes() != bottom.tobytes()


def test_rotated_boxes_with_brackets_keylines_and_fills() -> None:
    corners = [(120.0, 80.0), (300.0, 140.0), (270.0, 230.0), (90.0, 170.0)]
    obb = Shape("obb", "car", 0.7, box=(90, 80, 210, 150), corners=corners)
    plain = render(_img(), [obb], cfg=Look())
    brackets = render(_img(), [obb], cfg=Look(box={"corners": "brackets"}))  # type: ignore[arg-type]
    bare = render(_img(), [obb], cfg=Look(box={"keyline": None, "fill": 0}))  # type: ignore[arg-type]
    assert len({plain.tobytes(), brackets.tobytes(), bare.tobytes()}) == 3
    off_frame = Shape(
        "obb",
        "car",
        corners=[(-900.0, -900.0), (-800.0, -900.0), (-800.0, -850.0), (-900.0, -850.0)],
    )
    assert render(_img(), [off_frame], cfg=Look(), stored_colors=None).size == (400, 300)


def test_box_brackets_with_and_without_a_keyline() -> None:
    box = Shape("box", "car", box=(150, 120, 100, 80))
    keyed = {"corners": "brackets", "keyline": {"alpha": 0.42}}
    a = render(_img(), [box], cfg=Look(box=keyed))  # type: ignore[arg-type]
    b = render(_img(), [box], cfg=Look(box={"corners": "brackets"}))  # type: ignore[arg-type]
    assert a.tobytes() != b.tobytes()
    assert Look().box.keyline is None  # THE look: no keyline


def test_shapes_without_geometry_draw_nothing() -> None:
    empty = [
        Shape("box", "car"),
        Shape("polygon", "p", rings=[]),
        Shape("obb", "o"),
        Shape("polyline", "l", path=[(1.0, 1.0)]),
    ]
    base = _img()
    assert render(base, empty, cfg=Look(label={"show": False})).tobytes() == base.tobytes()  # type: ignore[arg-type]


def test_bones_take_their_class_color_and_width_and_skip_when_thin() -> None:
    bone = Bone((100.0, 100.0), (300.0, 100.0), "person", 1)
    a = render(_img(), [], cfg=Look(), bones=[bone])
    b = render(_img(), [], cfg=Look(bone={"color": "class", "keyline": False}), bones=[bone])  # type: ignore[arg-type]
    c = render(_img(), [], cfg=Look(bone={"width": 0}), bones=[bone])  # type: ignore[arg-type]
    assert a.tobytes() != b.tobytes()
    assert c.tobytes() == _img().tobytes()


def test_polylines_without_ends_and_keylines() -> None:
    lane = Shape("polyline", "lane", path=[(20.0, 150.0), (380.0, 160.0)])
    a = render(_img(), [lane], cfg=Look())
    b = render(
        _img(),
        [lane],
        cfg=Look(polyline={"ends": False}, box={"keyline": None}, outline={"keyline": None}),
    )  # type: ignore[arg-type]
    assert a.tobytes() != b.tobytes()


def test_a_ring_point_without_a_keyline() -> None:
    p = Shape("keypoint", "head", point=(200.0, 150.0))
    ring = {"style": "ring", "radius": 10}
    keyed = Look(point=ring, box={"keyline": {"alpha": 0.42}})  # type: ignore[arg-type]
    a = render(_img(), [p], cfg=keyed)
    b = render(_img(), [p], cfg=Look(point=ring))  # type: ignore[arg-type]
    assert a.tobytes() != b.tobytes()
    assert Look().outline.keyline is None


def test_a_label_with_no_words_and_badges_in_every_corner() -> None:
    nameless = Shape("box", "car", None, box=(150, 120, 100, 80))
    look = Look(label={"parts": {"name": False}})  # type: ignore[arg-type]
    assert (
        render(_img(), [nameless], cfg=look).tobytes()
        == render(_img(), [nameless], cfg=Look(label={"show": False})).tobytes()
    )  # type: ignore[arg-type]
    for corner in ("tl", "tr", "bl", "br"):
        out = render(
            _img(), [], cfg=Look(badge={"corner": corner}), badges=[Badge("cat", 0.9), Badge("dog")]
        )  # type: ignore[arg-type]
        assert out.tobytes() != _img().tobytes()
    text_badges = render(_img(), [], cfg=Look(label={"style": "text"}), badges=[Badge("cat", 0.9)])  # type: ignore[arg-type]
    assert text_badges.tobytes() != _img().tobytes()


def test_a_class_width_and_fill_override() -> None:
    box = Shape("box", "car", box=(150, 120, 100, 80))
    poly = Shape("polygon", "person", rings=SQUARE)
    base = render(_img(), [box, poly], cfg=Look(label={"show": False}))  # type: ignore[arg-type]
    over = render(
        _img(),
        [box, poly],
        cfg=Look(
            label={"show": False}, classes={"car": {"width": 3, "fill": 0.5}, "person": {"fill": 0}}
        ),  # type: ignore[arg-type]
    )
    assert base.tobytes() != over.tobytes()
    assert over.getpixel((120, 240)) == GREY  # the person's fill turned off
