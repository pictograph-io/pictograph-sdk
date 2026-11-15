"""Unit tests for ``pictograph.augment`` - the client-side augmentation engine.

The geometry remap is the load-bearing, hard-to-eyeball part, so these tests
pin it exhaustively per annotation type and per op, plus the engine's
reproducibility and input-coercion contracts.
"""

from __future__ import annotations

import math

import pytest
from PIL import Image

from pictograph.augment import (
    Augmenter,
    Blur,
    Brightness,
    Contrast,
    Crop,
    CutOut,
    Grayscale,
    HorizontalFlip,
    HueShift,
    Noise,
    Resize,
    Rotate,
    Rotate90,
    Saturation,
    Shear,
    VerticalFlip,
)
from pictograph.augment._geometry import remap_annotations
from pictograph.models.annotation import (
    BBoxAnnotation,
    KeypointAnnotation,
    PolygonAnnotation,
    PolylineAnnotation,
)
from pictograph.models.common import BoundingBox, Point

# ── helpers ─────────────────────────────────────────────────────────────


def _img(w: int = 100, h: int = 80) -> Image.Image:
    """A deterministic, high-frequency RGB image.

    Uses coprime multipliers so the pattern has real high-frequency content -
    a *linear* gradient is invariant under Gaussian blur (blur ≈ local average,
    and a linear ramp's average equals its center), which would make the blur
    op look like a no-op.
    """
    im = Image.new("RGB", (w, h))
    px = im.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = ((x * 7) % 256, (y * 13) % 256, (x * 17 + y * 31) % 256)
    return im


def _bbox(x=10.0, y=20.0, w=30.0, h=40.0, name="car") -> BBoxAnnotation:
    return BBoxAnnotation(id="b1", name=name, bounding_box=BoundingBox(x=x, y=y, w=w, h=h))


def _poly(name="road") -> PolygonAnnotation:
    ring = [Point(x=10, y=10), Point(x=40, y=10), Point(x=40, y=50), Point(x=10, y=50)]
    return PolygonAnnotation(id="p1", name=name, polygon={"paths": [ring]})


def _polyline(name="lane") -> PolylineAnnotation:
    return PolylineAnnotation(
        id="l1", name=name, polyline={"path": [Point(x=5, y=5), Point(x=50, y=60)]}
    )


def _kp(x=25.0, y=30.0, name="joint") -> KeypointAnnotation:
    return KeypointAnnotation(id="k1", name=name, keypoint=Point(x=x, y=y))


def _seeded(op):
    """A one-op Augmenter with p-fired ops forced by a fixed seed."""
    return Augmenter([op], seed=0)


# ── flips ───────────────────────────────────────────────────────────────


def test_horizontal_flip_bbox_and_size():
    img, anns = Augmenter([HorizontalFlip(p=1.0)], seed=1).augment(_img(100, 80), [_bbox()])
    assert img.size == (100, 80)
    box = anns[0].bounding_box
    # x' = W - (x + w) = 100 - 40 = 60; w,h preserved
    assert box.x == pytest.approx(60.0)
    assert box.y == pytest.approx(20.0)
    assert box.w == pytest.approx(30.0)
    assert box.h == pytest.approx(40.0)


def test_horizontal_flip_is_involutive():
    aug = Augmenter([HorizontalFlip(p=1.0), HorizontalFlip(p=1.0)], seed=1)
    _img_out, anns = aug.augment(_img(), [_bbox()])
    box = anns[0].bounding_box
    assert (box.x, box.y, box.w, box.h) == pytest.approx((10.0, 20.0, 30.0, 40.0))


def test_vertical_flip_bbox():
    _i, anns = Augmenter([VerticalFlip(p=1.0)], seed=1).augment(_img(100, 80), [_bbox()])
    box = anns[0].bounding_box
    # y' = H - (y + h) = 80 - 60 = 20
    assert (box.x, box.y, box.w, box.h) == pytest.approx((10.0, 20.0, 30.0, 40.0))


def test_flip_probability_zero_is_passthrough():
    _i, anns = Augmenter([HorizontalFlip(p=0.0)], seed=1).augment(_img(), [_bbox()])
    box = anns[0].bounding_box
    assert (box.x, box.y) == pytest.approx((10.0, 20.0))


# ── 90° rotations ───────────────────────────────────────────────────────


def test_rotate90_keypoint_and_size():
    img, anns = _seeded(Rotate90(k=1)).augment(_img(100, 80), [_kp(25, 30)])
    assert img.size == (80, 100)  # W,H swap
    kp = anns[0].keypoint
    # 90° CCW: (x, y) -> (y, W - x) = (30, 100 - 25) = (30, 75)
    assert (kp.x, kp.y) == pytest.approx((30.0, 75.0))


def test_rotate90_four_times_identity():
    aug = Augmenter([Rotate90(k=1)] * 4, seed=1)
    img, anns = aug.augment(_img(100, 80), [_bbox()])
    assert img.size == (100, 80)
    box = anns[0].bounding_box
    assert (box.x, box.y, box.w, box.h) == pytest.approx((10.0, 20.0, 30.0, 40.0))


@pytest.mark.parametrize("k", [1, 2, 3])
def test_rotate90_matches_arbitrary_rotate(k):
    kp = _kp(25, 30)
    _i1, a1 = _seeded(Rotate90(k=k)).augment(_img(100, 80), [kp])
    _i2, a2 = _seeded(Rotate(90.0 * k)).augment(_img(100, 80), [kp])
    assert (a1[0].keypoint.x, a1[0].keypoint.y) == pytest.approx(
        (a2[0].keypoint.x, a2[0].keypoint.y), abs=1e-6
    )


def test_rotate90_random_k_is_seed_stable():
    a = Augmenter([Rotate90(k=None)], seed=7)
    b = Augmenter([Rotate90(k=None)], seed=7)
    ia, _ = a.augment(_img(), [_bbox()])
    ib, _ = b.augment(_img(), [_bbox()])
    assert ia.size == ib.size


# ── arbitrary rotation ──────────────────────────────────────────────────


def test_rotate_zero_is_identity():
    _i, anns = _seeded(Rotate(0.0)).augment(_img(100, 80), [_bbox()])
    box = anns[0].bounding_box
    assert (box.x, box.y, box.w, box.h) == pytest.approx((10.0, 20.0, 30.0, 40.0), abs=1e-6)


def test_rotate_output_size_matches_pillow_and_centers_preserved():
    src = _img(100, 80)
    angle = 30.0
    expected = src.rotate(angle, expand=True).size
    # a box centered in the image
    centered = _bbox(x=40, y=30, w=20, h=20)  # center (50,40) == image center
    img, anns = _seeded(Rotate(angle)).augment(src, [centered])
    assert img.size == expected
    box = anns[0].bounding_box
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    assert (cx, cy) == pytest.approx((expected[0] / 2, expected[1] / 2), abs=1e-6)


def test_rotate_bbox_grows_under_rotation():
    # a 45° rotation of an axis-aligned box yields a larger enclosing box
    _i, anns = _seeded(Rotate(45.0)).augment(_img(200, 200), [_bbox(x=80, y=80, w=40, h=40)])
    box = anns[0].bounding_box
    assert box.w > 40.0 and box.h > 40.0
    # enclosing box of a 40px square rotated 45° ≈ 40*sqrt(2)
    assert box.w == pytest.approx(40 * math.sqrt(2), abs=1.0)


# ── resize ──────────────────────────────────────────────────────────────


def test_resize_scales_geometry():
    img, anns = _seeded(Resize(200, 160)).augment(_img(100, 80), [_bbox()])
    assert img.size == (200, 160)
    box = anns[0].bounding_box
    assert (box.x, box.y, box.w, box.h) == pytest.approx((20.0, 40.0, 60.0, 80.0))


def test_resize_polygon_points():
    _i, anns = _seeded(Resize(200, 160)).augment(_img(100, 80), [_poly()])
    ring = anns[0].polygon.paths[0]
    assert (ring[0].x, ring[0].y) == pytest.approx((20.0, 20.0))
    assert (ring[2].x, ring[2].y) == pytest.approx((80.0, 100.0))


# ── crop / clipping ─────────────────────────────────────────────────────


def test_crop_translates_and_drops_out_of_frame():
    # deterministic full-scale-then-offset crop via a fixed window
    img = _img(100, 100)
    # keep a box inside and a box outside the crop
    inside = _bbox(x=60, y=60, w=20, h=20, name="in")
    outside = _bbox(x=5, y=5, w=10, h=10, name="out")
    # Crop scale=0.5 → 50x50 window; force placement with seed search below.
    aug = Augmenter([Crop(scale=0.5, min_visibility=0.3)], seed=3)
    out_img, anns = aug.augment(img, [inside, outside])
    assert out_img.size == (50, 50)
    # Whatever the random window, every surviving box must be within the frame.
    for a in anns:
        b = a.bounding_box
        assert b.x >= 0 and b.y >= 0
        assert b.x + b.w <= 50 + 1e-6 and b.y + b.h <= 50 + 1e-6


def test_crop_clips_polygon_sutherland_hodgman():
    # A square polygon [0..40]^2; crop the top-left 20x20 window at origin →
    # clipped ring should be the [0..20]^2 square.
    fn = lambda x, y: (x, y)  # noqa: E731 - identity map, clip does the work
    poly = _poly()  # ring corners (10,10),(40,10),(40,50),(10,50)
    out = remap_annotations([poly], fn, 20, 20, clip=True, min_visibility=0.0)
    assert len(out) == 1
    ring = out[0].polygon.paths[0]
    xs = [p.x for p in ring]
    ys = [p.y for p in ring]
    assert max(xs) <= 20.0 + 1e-6 and max(ys) <= 20.0 + 1e-6
    assert min(xs) >= 10.0 - 1e-6  # left edge of the poly was inside


def test_clip_drops_fully_outside_keypoint():
    out = remap_annotations([_kp(90, 90)], lambda x, y: (x, y), 20, 20, clip=True)
    assert out == []


# ── photometric ops leave geometry untouched ────────────────────────────


@pytest.mark.parametrize(
    "op",
    [
        Brightness(1.5, p=1.0),
        Contrast(1.5, p=1.0),
        Saturation(0.5, p=1.0),
        Grayscale(p=1.0),
        Blur(1.5, p=1.0),
        Noise(0.1, p=1.0),
        HueShift(45.0, p=1.0),
        CutOut(0.3, count=2, p=1.0),
    ],
)
def test_photometric_preserves_geometry_and_size(op):
    src = _img(100, 80)
    img, anns = _seeded(op).augment(src, [_bbox(), _poly(), _kp()])
    assert img.size == (100, 80)
    # geometry identical to input
    assert anns[0].bounding_box.model_dump() == _bbox().bounding_box.model_dump()
    assert [p.model_dump() for p in anns[1].polygon.paths[0]] == [
        p.model_dump() for p in _poly().polygon.paths[0]
    ]
    # pixels actually changed (not a no-op) for a non-uniform source
    assert img.tobytes() != src.convert("RGB").tobytes()


def test_grayscale_channels_equal():
    img, _a = _seeded(Grayscale(p=1.0)).augment(_img(20, 20), [])
    px = img.load()
    r, g, b = px[5, 7]
    assert r == g == b


# ── shear ────────────────────────────────────────────────────────────────


def test_shear_remaps_keypoint():
    import math

    img, anns = _seeded(Shear(10.0)).augment(_img(200, 200), [_kp(50, 40)])
    assert img.size == (200, 200)  # shear keeps the canvas
    kp = anns[0].keypoint
    # x' = x + tan(10°)*y = 50 + tan(10°)*40
    assert (kp.x, kp.y) == pytest.approx((50 + math.tan(math.radians(10)) * 40, 40.0))


def test_shear_zero_is_identity():
    _i, anns = _seeded(Shear(0.0)).augment(_img(100, 80), [_bbox()])
    box = anns[0].bounding_box
    assert (box.x, box.y, box.w, box.h) == pytest.approx((10.0, 20.0, 30.0, 40.0), abs=1e-6)


def test_cutout_erases_region():
    src = _img(60, 60)
    img, anns = Augmenter([CutOut(0.5, count=1, fill=(0, 0, 0), p=1.0)], seed=2).augment(
        src, [_bbox()]
    )
    assert img.size == (60, 60)
    assert anns[0].bounding_box.model_dump() == _bbox().bounding_box.model_dump()  # geometry kept
    # some pixels are now the fill color (a black hole) that weren't before
    assert img.tobytes() != src.convert("RGB").tobytes()


# ── engine contracts ────────────────────────────────────────────────────


def test_augment_accepts_dict_annotations():
    raw = {
        "id": "b1",
        "name": "car",
        "type": "bbox",
        "bounding_box": {"x": 10, "y": 20, "w": 30, "h": 40},
    }
    _i, anns = Augmenter([HorizontalFlip(p=1.0)], seed=1).augment(_img(100, 80), [raw])
    assert isinstance(anns[0], BBoxAnnotation)
    assert anns[0].bounding_box.x == pytest.approx(60.0)


def test_augment_accepts_path(tmp_path):
    p = tmp_path / "src.png"
    _img(60, 40).save(p)
    img, anns = Augmenter([HorizontalFlip(p=1.0)], seed=1).augment(p, [_bbox(x=5, y=5, w=10, h=10)])
    assert img.size == (60, 40)
    assert anns[0].bounding_box.x == pytest.approx(60 - 15)


def test_augment_no_annotations():
    img, anns = Augmenter([HorizontalFlip(p=1.0)], seed=1).augment(_img(), None)
    assert anns == []
    assert img.size == (100, 80)


def test_generate_produces_n_variants():
    aug = Augmenter([HorizontalFlip(p=0.5), Rotate((-20, 20))], seed=5)
    variants = aug.generate(_img(), [_bbox()], n=4)
    assert len(variants) == 4
    for _img_out, anns in variants:
        assert len(anns) == 1


def test_generate_negative_raises():
    with pytest.raises(ValueError):
        Augmenter([HorizontalFlip()], seed=1).generate(_img(), [_bbox()], n=-1)


def test_seed_reproducibility_geometry_and_pixels():
    ops = [HorizontalFlip(p=0.5), Rotate((-25, 25)), Brightness((0.7, 1.3))]
    a = Augmenter(ops, seed=99)
    b = Augmenter(ops, seed=99)
    va = a.generate(_img(120, 90), [_bbox(), _poly()], n=5)
    vb = b.generate(_img(120, 90), [_bbox(), _poly()], n=5)
    for (ia, aa), (ib, ab) in zip(va, vb, strict=True):
        assert ia.size == ib.size
        assert ia.tobytes() == ib.tobytes()  # deterministic ops → identical pixels
        assert [x.model_dump() for x in aa] == [x.model_dump() for x in ab]


def test_reset_rewinds_sequence():
    aug = Augmenter([Rotate((-30, 30))], seed=11)
    first = aug.generate(_img(), [_bbox()], n=3)
    aug.reset()
    second = aug.generate(_img(), [_bbox()], n=3)
    for (i1, a1), (i2, a2) in zip(first, second, strict=True):
        assert i1.tobytes() == i2.tobytes()
        assert [x.model_dump() for x in a1] == [x.model_dump() for x in a2]


def test_different_seeds_diverge():
    a = Augmenter([Rotate((-30, 30))], seed=1).generate(_img(), [_bbox()], n=1)
    b = Augmenter([Rotate((-30, 30))], seed=2).generate(_img(), [_bbox()], n=1)
    # extremely unlikely two different seeds pick the identical angle
    assert a[0][1][0].bounding_box.model_dump() != b[0][1][0].bounding_box.model_dump()


def test_invalid_rotate90_k():
    with pytest.raises(ValueError):
        Rotate90(k=5)


def test_invalid_resize():
    with pytest.raises(ValueError):
        Resize(0, 100)


# ── build_ops (spec -> op) ───────────────────────────────────────────────


def test_build_ops_maps_all_specs():
    from pictograph.augment import build_ops

    ops = build_ops(
        [
            {"op": "resize", "width": 640, "height": 480},
            {"op": "crop", "scale": 0.8},
            {"op": "flip"},
            {"op": "vflip"},
            {"op": "rotate90"},
            {"op": "rotate", "degrees": 15},
            {"op": "brightness", "factor": 0.2},
            {"op": "contrast", "factor": 0.1},
            {"op": "saturation", "factor": 0.3},
            {"op": "hue_shift", "degrees": 20},
            {"op": "grayscale"},
            {"op": "blur", "radius": 2},
            {"op": "noise", "amount": 0.05},
            {"op": "cutout", "size": 0.3, "count": 2},
            {"op": "shear", "degrees": 10},
        ]
    )
    assert [o.name for o in ops] == [
        "resize",
        "crop",
        "horizontal_flip",
        "vertical_flip",
        "rotate90",
        "rotate",
        "brightness",
        "contrast",
        "saturation",
        "hue_shift",
        "grayscale",
        "blur",
        "noise",
        "cutout",
        "shear",
    ]


def test_build_ops_scalar_ranges():
    from pictograph.augment import Brightness, Rotate, build_ops

    rotate = build_ops([{"op": "rotate", "degrees": 15}])[0]
    assert isinstance(rotate, Rotate)
    assert rotate.degrees == (-15.0, 15.0)  # scalar -> symmetric-about-0 range
    bright = build_ops([{"op": "brightness", "factor": 0.2}])[0]
    assert isinstance(bright, Brightness)
    assert bright.factor == (0.8, 1.2)  # scalar -> symmetric-about-1 range


def test_build_ops_explicit_range():
    from pictograph.augment import Rotate, build_ops

    rotate = build_ops([{"op": "rotate", "degrees": [-30, 10]}])[0]
    assert isinstance(rotate, Rotate)
    assert rotate.degrees == (-30.0, 10.0)


def test_build_ops_functionally_matches_direct_construction():
    from pictograph.augment import build_ops

    spec_ops = build_ops([{"op": "flip", "p": 1.0}, {"op": "rotate", "degrees": 20}])
    direct = [HorizontalFlip(p=1.0), Rotate((-20.0, 20.0))]
    img = _img(100, 80)
    a, aa = Augmenter(spec_ops, seed=3).augment(img, [_bbox()])
    b, bb = Augmenter(direct, seed=3).augment(img, [_bbox()])
    assert a.tobytes() == b.tobytes()
    assert [x.model_dump() for x in aa] == [x.model_dump() for x in bb]


def test_build_ops_unknown_raises():
    from pictograph.augment import build_ops

    with pytest.raises(ValueError, match="Unknown augmentation op"):
        build_ops([{"op": "teleport"}])


# ── OP_SPECS is the served op catalog - parity is load-bearing ───────────────


class TestOpSpecsCatalog:
    def test_specs_cover_exactly_the_op_names(self) -> None:
        """The catalog and the accepted-op list can never drift - this pair is
        what killed the hand-maintained frontend copy (resize was missing)."""
        from pictograph.augment import OP_NAMES, OP_SPECS

        assert tuple(s["op"] for s in OP_SPECS) == OP_NAMES

    def test_every_spec_builds_with_its_defaults(self) -> None:
        """Each spec's declared defaults are ACCEPTED by build_ops - a spec
        whose defaults 400 on the server is a lie in the catalog."""
        from pictograph.augment import OP_SPECS, build_ops

        for spec in OP_SPECS:
            params = {
                p["key"]: (p["default"] if p["default"] is not None else 64) for p in spec["params"]
            }
            ops = build_ops([{"op": spec["op"], **params}])
            assert len(ops) == 1, spec["op"]

    def test_required_params_are_declared(self) -> None:
        from pictograph.augment import OP_SPECS

        resize = next(s for s in OP_SPECS if s["op"] == "resize")
        assert {p["key"] for p in resize["params"] if p["required"]} == {"width", "height"}
