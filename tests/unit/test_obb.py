"""Oriented bounding boxes in the SDK.

An oriented (rotated) box is NOT its own annotation type. It is a plain ``bbox`` that
additionally carries an optional :class:`OrientedBoxGeometry` under ``oriented_box``;
``bounding_box`` stays the axis-aligned enclosure that training and every OBB-unaware
consumer reads. A non-rotated box carries no oriented metadata (``oriented_box is None``).

Two things are being protected here.

1. **The wire contract.** `_obb.py` is the third copy of this geometry (the editor's
   `obb.ts` and the server's `obb.py` are the other two). The corner ORDER and the angle
   SIGN must agree across all three, or a box drawn in the editor lands mirrored once
   it is exported - and nothing else in the system notices. The reference vectors below
   are the same ones both twins pin.

2. **That a rotated box survives the pipelines.** `augment` and `tile` transform the
   `oriented_box` alongside `bounding_box`. The tests that matter are the ones that push
   a rotated box through a real flip / rotate / tile and check the ANGLE moved correctly -
   a case a plain "does it typecheck" test would sail past.
"""

import math

import pytest

from pictograph._obb import normalize_angle, obb_aabb, obb_corners, obb_from_corners
from pictograph.models.annotation import BBoxAnnotation, OrientedBoxGeometry

# The reference box. MUST match the web app's and the API's own oriented-box
# reference fixtures - the three are one shape, checked in three places.
REF = OrientedBoxGeometry(cx=100.0, cy=100.0, w=40.0, h=20.0, angle=0.0)


def _obb(**kw: float) -> OrientedBoxGeometry:
    return REF.model_copy(update=kw)


class TestGeometryLockstep:
    def test_unrotated_corners_are_clockwise_from_top_left(self) -> None:
        c = obb_corners(REF)
        assert [(p.x, p.y) for p in c] == [
            (80.0, 90.0),
            (120.0, 90.0),
            (120.0, 110.0),
            (80.0, 110.0),
        ]

    def test_positive_angle_turns_clockwise_on_screen(self) -> None:
        """The SIGN contract: at 90° the top-right corner swings DOWN (y grows down)."""
        c = obb_corners(_obb(angle=90))
        assert c[1].x == pytest.approx(110.0)
        assert c[1].y == pytest.approx(120.0)

    @pytest.mark.parametrize("angle", [0, 15, 45, 90, 137, 180, 271, 359])
    def test_from_corners_round_trips(self, angle: float) -> None:
        src = OrientedBoxGeometry(cx=250.5, cy=133.25, w=80.0, h=30.0, angle=float(angle))
        back = obb_from_corners(obb_corners(src))
        assert back is not None
        assert back.cx == pytest.approx(src.cx)
        assert back.cy == pytest.approx(src.cy)
        assert back.w == pytest.approx(src.w)
        assert back.h == pytest.approx(src.h)
        assert back.angle == pytest.approx(src.angle)

    def test_aabb_of_a_rotated_box_is_bigger_than_the_box(self) -> None:
        x0, y0, x1, y1 = obb_aabb(_obb(angle=45))
        assert (x1 - x0) == pytest.approx(60 / math.sqrt(2))
        assert (x1 - x0) > REF.w  # the whole reason OBB exists

    def test_normalize_angle(self) -> None:
        assert normalize_angle(370) == pytest.approx(10)
        assert normalize_angle(-10) == pytest.approx(350)

    def test_from_corners_rejects_a_bad_ring(self) -> None:
        assert obb_from_corners([]) is None
        assert obb_from_corners(obb_corners(REF)[:3]) is None


def _aabb_box(obb: OrientedBoxGeometry) -> dict[str, float]:
    """The axis-aligned enclosure of a rotated box, as a bounding_box dict."""
    x0, y0, x1, y1 = obb_aabb(obb)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


class TestModel:
    def test_a_rotated_box_parses_as_a_bbox_with_oriented_box(self) -> None:
        obb = OrientedBoxGeometry(cx=100, cy=100, w=40, h=20, angle=30)
        ann = BBoxAnnotation.model_validate(
            {
                "id": "a1",
                "name": "ship",
                "type": "bbox",
                "bounding_box": _aabb_box(obb),
                "oriented_box": {"cx": 100, "cy": 100, "w": 40, "h": 20, "angle": 30},
            }
        )
        assert ann.type == "bbox"
        assert ann.oriented_box is not None
        assert ann.oriented_box.angle == 30
        # bounding_box is the axis-aligned enclosure - always present, never None.
        assert ann.bounding_box is not None

    def test_a_plain_box_has_no_oriented_box(self) -> None:
        ann = BBoxAnnotation.model_validate(
            {
                "id": "b",
                "name": "car",
                "type": "bbox",
                "bounding_box": {"x": 0, "y": 0, "w": 4, "h": 4},
            }
        )
        assert ann.type == "bbox"
        assert ann.oriented_box is None

    def test_a_rotated_box_has_no_polygon_key(self) -> None:
        """The canonical wire shape carries NO derived polygon on a rotated box -
        `extra="forbid"` means a stray `polygon` key on a bbox is a HARD failure."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BBoxAnnotation.model_validate(
                {
                    "id": "a1",
                    "name": "ship",
                    "type": "bbox",
                    "bounding_box": {"x": 77.7, "y": 81.3, "w": 44.6, "h": 37.4},
                    "oriented_box": {"cx": 100, "cy": 100, "w": 40, "h": 20, "angle": 30},
                    "polygon": {
                        "paths": [
                            [
                                {"x": 87.7, "y": 81.3},
                                {"x": 122.3, "y": 101.3},
                                {"x": 112.3, "y": 118.7},
                                {"x": 77.7, "y": 98.7},
                            ]
                        ]
                    },
                }
            )

    def test_both_rotated_and_plain_boxes_parse_in_the_discriminated_union(self) -> None:
        from pydantic import TypeAdapter

        from pictograph.models.annotation import Annotation

        parsed = TypeAdapter(list[Annotation]).validate_python(
            [
                {
                    "id": "a",
                    "name": "ship",
                    "type": "bbox",
                    "bounding_box": {"x": 0, "y": 0, "w": 4, "h": 4},
                    "oriented_box": {"cx": 1, "cy": 1, "w": 2, "h": 2, "angle": 10},
                },
                {
                    "id": "b",
                    "name": "car",
                    "type": "bbox",
                    "bounding_box": {"x": 0, "y": 0, "w": 4, "h": 4},
                },
            ]
        )
        assert isinstance(parsed[0], BBoxAnnotation)
        assert parsed[0].oriented_box is not None
        assert isinstance(parsed[1], BBoxAnnotation)
        assert parsed[1].oriented_box is None

    def test_obb_is_no_longer_a_type(self) -> None:
        """The old `type:"obb"` shape must be REJECTED - there is no ObbAnnotation."""
        from pydantic import TypeAdapter

        from pictograph.models.annotation import Annotation

        with pytest.raises(Exception):
            TypeAdapter(Annotation).validate_python(
                {
                    "id": "a",
                    "name": "ship",
                    "type": "obb",
                    "oriented_box": {"cx": 1, "cy": 1, "w": 2, "h": 2, "angle": 10},
                }
            )

    def test_rejects_a_zero_extent_box(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OrientedBoxGeometry(cx=0, cy=0, w=0, h=10, angle=0)

    def test_rejects_an_out_of_range_angle(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OrientedBoxGeometry(cx=0, cy=0, w=1, h=1, angle=360)


class TestAugment:
    """The tests that actually matter: does the ANGLE move correctly?"""

    def _ann(self, angle: float = 30.0) -> BBoxAnnotation:
        obb = OrientedBoxGeometry(cx=100, cy=100, w=40, h=20, angle=angle)
        return BBoxAnnotation.model_validate(
            {
                "id": "a1",
                "name": "ship",
                "type": "bbox",
                "bounding_box": _aabb_box(obb),
                "oriented_box": obb.model_dump(),
            }
        )

    def test_a_horizontal_flip_mirrors_the_angle(self) -> None:
        """A 30° box in a 200-wide image flips to 150°. Get the sign wrong and every
        augmented rotated box is subtly, invisibly incorrect in the training set."""
        from pictograph.augment._geometry import remap_annotations

        out = remap_annotations([self._ann(30)], lambda x, y: (200 - x, y), 200, 200)
        assert len(out) == 1
        result = out[0]
        assert isinstance(result, BBoxAnnotation)
        assert result.oriented_box is not None
        assert result.oriented_box.angle == pytest.approx(150.0, abs=1e-6)
        assert result.oriented_box.cx == pytest.approx(100.0)  # centred → stays put
        assert result.oriented_box.w == pytest.approx(40.0)  # extents are invariant

    def test_a_vertical_flip_mirrors_the_angle_the_other_way(self) -> None:
        from pictograph.augment._geometry import remap_annotations

        out = remap_annotations([self._ann(30)], lambda x, y: (x, 200 - y), 200, 200)
        result = out[0]
        assert isinstance(result, BBoxAnnotation)
        assert result.oriented_box is not None
        assert result.oriented_box.angle == pytest.approx(330.0, abs=1e-6)

    def test_a_90_degree_rotation_adds_to_the_angle(self) -> None:
        from pictograph.augment._geometry import remap_annotations

        # Rotate the image 90° clockwise about the origin, into a 200x200 frame.
        out = remap_annotations([self._ann(30)], lambda x, y: (200 - y, x), 200, 200)
        result = out[0]
        assert isinstance(result, BBoxAnnotation)
        assert result.oriented_box is not None
        assert result.oriented_box.angle == pytest.approx(120.0, abs=1e-6)

    def test_a_translation_moves_only_the_centre(self) -> None:
        from pictograph.augment._geometry import remap_annotations

        out = remap_annotations([self._ann(30)], lambda x, y: (x + 10, y - 5), 200, 200)
        result = out[0]
        assert isinstance(result, BBoxAnnotation)
        assert result.oriented_box is not None
        assert result.oriented_box.cx == pytest.approx(110.0)
        assert result.oriented_box.cy == pytest.approx(95.0)
        assert result.oriented_box.angle == pytest.approx(30.0, abs=1e-6)

    def test_the_derived_bounding_box_is_the_moved_aabb_not_stale(self) -> None:
        from pictograph.augment._geometry import remap_annotations

        out = remap_annotations([self._ann(30)], lambda x, y: (x + 50, y), 200, 200)
        result = out[0]
        assert isinstance(result, BBoxAnnotation)
        assert result.oriented_box is not None
        assert result.bounding_box is not None
        # bounding_box must enclose the MOVED oriented box's corners, not the original's.
        x0, y0, x1, y1 = obb_aabb(result.oriented_box)
        assert result.bounding_box.x == pytest.approx(x0)
        assert result.bounding_box.y == pytest.approx(y0)
        assert result.bounding_box.w == pytest.approx(x1 - x0)
        assert result.bounding_box.h == pytest.approx(y1 - y0)

    def test_a_plain_box_stays_plain_through_a_flip(self) -> None:
        """A non-rotated box has no oriented_box before OR after a transform."""
        from pictograph.augment._geometry import remap_annotations

        plain = BBoxAnnotation.model_validate(
            {
                "id": "p",
                "name": "car",
                "type": "bbox",
                "bounding_box": {"x": 80, "y": 90, "w": 40, "h": 20},
            }
        )
        out = remap_annotations([plain], lambda x, y: (200 - x, y), 200, 200)
        result = out[0]
        assert isinstance(result, BBoxAnnotation)
        assert result.oriented_box is None
        assert result.bounding_box.w == pytest.approx(40.0)

    def test_a_crop_that_cuts_the_box_away_drops_it(self) -> None:
        from pictograph.augment._geometry import remap_annotations

        # Shift the box far off the left edge; almost nothing is visible.
        out = remap_annotations(
            [self._ann(30)],
            lambda x, y: (x - 500, y),
            200,
            200,
            clip=True,
            min_visibility=0.5,
        )
        assert out == []

    def test_a_crop_that_keeps_the_box_does_not_shear_it(self) -> None:
        """A clipped rectangle is not a rectangle. The box must come through whole (or
        not at all) rather than have its corners clamped into a shear."""
        from pictograph.augment._geometry import remap_annotations

        out = remap_annotations(
            [self._ann(30)],
            lambda x, y: (x, y),
            200,
            200,
            clip=True,
            min_visibility=0.5,
        )
        result = out[0]
        assert isinstance(result, BBoxAnnotation)
        assert result.oriented_box is not None
        assert result.oriented_box.w == pytest.approx(40.0)
        assert result.oriented_box.h == pytest.approx(20.0)
        assert result.oriented_box.angle == pytest.approx(30.0, abs=1e-6)


class TestTile:
    def test_a_rotated_box_lands_in_the_tile_it_overlaps(self) -> None:
        from PIL import Image

        from pictograph.tile import tile_image

        img = Image.new("RGB", (200, 200))
        obb = OrientedBoxGeometry(cx=50, cy=50, w=40, h=20, angle=30)
        ann = BBoxAnnotation.model_validate(
            {
                "id": "a1",
                "name": "ship",
                "type": "bbox",
                "bounding_box": _aabb_box(obb),
                "oriented_box": obb.model_dump(),
            }
        )
        tiles = tile_image(img, [ann], rows=2, cols=2)
        # The box sits wholly inside the top-left tile.
        top_left = next(t for t in tiles if t.row == 0 and t.col == 0)
        assert len(top_left.annotations) == 1
        got = top_left.annotations[0]
        assert isinstance(got, BBoxAnnotation)
        assert got.oriented_box is not None
        assert got.oriented_box.angle == pytest.approx(30.0, abs=1e-6)
        # ...and it does NOT appear in the far tile.
        bottom_right = next(t for t in tiles if t.row == 1 and t.col == 1)
        assert bottom_right.annotations == []

    def test_the_tile_uses_the_rotated_footprint(self) -> None:
        """`_source_rect` must enclose the rotated corners, not w x h. With w x h a
        turned box under-reports its footprint by up to sqrt(2), so a tile it really
        does overlap would decide that it doesn't and drop it."""
        from pictograph.tile._tiler import _source_rect

        obb = OrientedBoxGeometry(cx=100, cy=100, w=40, h=20, angle=45)
        ann = BBoxAnnotation.model_validate(
            {
                "id": "a1",
                "name": "ship",
                "type": "bbox",
                "bounding_box": _aabb_box(obb),
                "oriented_box": obb.model_dump(),
            }
        )
        rect = _source_rect(ann)
        assert rect is not None
        x0, y0, x1, y1 = rect
        assert (x1 - x0) == pytest.approx(60 / math.sqrt(2))  # NOT 40
