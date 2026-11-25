"""Ground-truth masks and COCO's many-ring polygons (SDK 1.69.170).

* A COCO polygon ``segmentation`` lists ONE object's pieces: it becomes ONE polygon
  annotation with every ring (the platform's convention, and what the platform's own
  COCO export writes) - it used to become one annotation per ring, so an occluded car
  came back as two cars and the SDK did not round-trip its own ``to_coco``.
* An RLE mask is traced into one polygon (holes included) by ``trace_mask``, the
  tracer every ground-truth mask import uses, or reported in ``unconverted`` - never
  replaced by its box.

The RLE strings below were written by pycocotools (``mask.encode``) from the masks
``_masks()`` rebuilds, so the decoder is checked against the reference encoder.
"""

from __future__ import annotations

from typing import Any

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from PIL import Image, ImageDraw  # noqa: E402

from pictograph.formats import decode_rle, from_coco, to_coco, trace_mask  # noqa: E402
from pictograph.models.annotation import PolygonAnnotation, PolygonGeometry  # noqa: E402
from pictograph.models.common import Point  # noqa: E402

PYCOCOTOOLS_RLE = {
    "holed": {
        "size": [40, 60],
        "counts": "e<i0?000000000000000000^OI37MI37MI37MI37MI37MI37MI37MI37MI37MI37MI37MI37MI37"
        "MI37MI37?0000000000000000000000000000[<",
    },
    "two_pieces_at_border": {
        "size": [32, 48],
        "counts": "0:f0000000000000000000000db02Z]O000000000000000000000000000000000",
    },
    "speck_and_line": {"size": [16, 16], "counts": "Z1180N090000000000000000000f0"},
}


def _masks() -> dict[str, Any]:
    holed = np.zeros((40, 60), np.uint8)
    holed[5:30, 10:50] = 1
    holed[12:20, 20:35] = 0
    two = np.zeros((32, 48), np.uint8)
    two[0:10, 0:12] = 1
    two[20:32, 30:48] = 1
    speck = np.zeros((16, 16), np.uint8)
    speck[3, 3] = 1
    speck[10, 2:14] = 1
    return {"holed": holed, "two_pieces_at_border": two, "speck_and_line": speck}


def _even_odd(rings: list[list[Point]], w: int, h: int) -> Any:
    """Rasterize as training reads a polygon (PIL fill per ring, XOR)."""
    out = np.zeros((h, w), dtype=bool)
    for ring in rings:
        layer = Image.new("1", (w, h), 0)
        ImageDraw.Draw(layer).polygon([(p.x, p.y) for p in ring], fill=1)
        out ^= np.asarray(layer, dtype=bool)
    return out


def _iou(a: Any, b: Any) -> float:
    return float((a & b).sum()) / float((a | b).sum())


# ───────────── decode_rle ─────────────


@pytest.mark.parametrize("name", sorted(PYCOCOTOOLS_RLE))
def test_decode_rle_matches_pycocotools(name: str) -> None:
    assert (decode_rle(PYCOCOTOOLS_RLE[name]) == _masks()[name].astype(bool)).all()


def test_decode_rle_reads_uncompressed_counts_and_bytes() -> None:
    m = _masks()["holed"]
    flat = m.flatten(order="F")
    counts, run, cur = [], 0, 0
    for v in flat:
        if v != cur:
            counts.append(run)
            run, cur = 0, v
        run += 1
    counts.append(run)
    assert (decode_rle({"size": [40, 60], "counts": counts}) == m.astype(bool)).all()
    raw = PYCOCOTOOLS_RLE["holed"]
    assert (
        decode_rle({"size": raw["size"], "counts": raw["counts"].encode()}) == m.astype(bool)
    ).all()


def test_decode_rle_refuses_runs_that_do_not_cover_the_mask() -> None:
    with pytest.raises(ValueError, match="cover 9 pixels"):
        decode_rle({"size": [4, 4], "counts": [4, 5]})


# ───────────── trace_mask ─────────────


def test_trace_mask_keeps_holes_as_rings_of_one_polygon() -> None:
    m = _masks()["holed"]
    traced = trace_mask(m)
    assert len(traced.rings) == 2 and traced.dropped == 0
    outer, hole = traced.rings
    assert {(p.x, p.y) for p in outer} == {(10, 5), (49, 5), (49, 29), (10, 29)}
    assert {(p.x, p.y) for p in hole} >= {(20, 11), (34, 11)} or len(hole) == 4
    assert _iou(_even_odd(traced.rings, 60, 40), m.astype(bool)) > 0.9


def test_trace_mask_never_clears_the_frame_border() -> None:
    traced = trace_mask(_masks()["two_pieces_at_border"])
    assert len(traced.rings) == 2
    # Largest piece first: the 18x12 piece, then the 12x10 one touching (0, 0).
    assert min(p.x for p in traced.rings[0]) == 30
    assert (0.0, 0.0) in {(p.x, p.y) for p in traced.rings[1]}


def test_trace_mask_counts_what_no_polygon_can_hold() -> None:
    traced = trace_mask(_masks()["speck_and_line"])
    assert traced.rings == [] and traced.dropped == 2


def test_trace_mask_offsets_a_crop_and_takes_the_frames_diagonal() -> None:
    m = _masks()["holed"]
    whole = trace_mask(m)
    crop = trace_mask(m[4:31, 9:51], offset=(9, 4), diagonal=(40**2 + 60**2) ** 0.5)
    assert crop.rings == whole.rings


# ───────────── from_coco ─────────────


def _coco(**ann: Any) -> dict[str, Any]:
    return {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 60, "height": 40}],
        "categories": [{"id": 1, "name": "car"}],
        "annotations": [{"id": 7, "image_id": 1, "category_id": 1, **ann}],
    }


def test_from_coco_keeps_an_objects_pieces_in_one_annotation() -> None:
    seg = [[0, 0, 10, 0, 10, 10, 0, 10], [20, 0, 30, 0, 30, 10]]
    imp = from_coco(_coco(segmentation=seg, bbox=[0, 0, 30, 10]))
    (ann,) = imp.annotations["a.jpg"]
    assert isinstance(ann, PolygonAnnotation) and ann.id == "7"
    assert [len(r) for r in ann.polygon.paths] == [4, 3]
    assert ann.bounding_box is not None and (ann.bounding_box.w, ann.bounding_box.h) == (30, 10)


def test_to_coco_then_from_coco_round_trips_a_polygon_in_pieces() -> None:
    poly = PolygonAnnotation(
        name="car",
        polygon=PolygonGeometry(
            paths=[
                [Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=10)],
                [Point(x=20, y=0), Point(x=30, y=0), Point(x=30, y=10)],
            ]
        ),
    )
    back = from_coco(to_coco({"a.jpg": [poly]})).annotations["a.jpg"]
    assert len(back) == 1
    assert isinstance(back[0], PolygonAnnotation)
    assert back[0].polygon.paths == poly.polygon.paths


def test_from_coco_traces_an_rle_mask_with_its_hole() -> None:
    imp = from_coco(_coco(segmentation=PYCOCOTOOLS_RLE["holed"], bbox=[10, 5, 40, 25], iscrowd=1))
    (ann,) = imp.annotations["a.jpg"]
    assert isinstance(ann, PolygonAnnotation) and ann.id == "7"
    assert len(ann.polygon.paths) == 2
    assert imp.unconverted == []


def test_from_coco_reports_an_rle_mask_it_cannot_trace_and_never_boxes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def no_cv2(name: str, *a: Any, **k: Any) -> Any:
        if name == "cv2":
            raise ImportError("No module named 'cv2'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_cv2)
    imp = from_coco(_coco(segmentation=PYCOCOTOOLS_RLE["holed"], bbox=[10, 5, 40, 25]))
    assert imp.annotations == {}
    (line,) = imp.unconverted
    assert line.startswith("a.jpg: annotation 7 (car): RLE mask not converted")
    assert "pictograph[inference]" in line


def test_from_coco_reports_an_rle_mask_that_holds_no_polygon() -> None:
    imp = from_coco(_coco(segmentation=PYCOCOTOOLS_RLE["speck_and_line"], bbox=[2, 3, 12, 8]))
    assert imp.annotations == {}
    assert imp.unconverted == ["a.jpg: annotation 7 (car): RLE mask holds no polygon"]


# ───────────── a polygon in pieces is boxed and measured whole ─────────────


def _ring(*xy: tuple[float, float]) -> list[Point]:
    return [Point(x=x, y=y) for x, y in xy]


def test_annotation_bbox_encloses_every_ring_as_the_platform_does() -> None:
    from pictograph.formats._shared import annotation_bbox

    poly = PolygonAnnotation(
        name="car",
        polygon=PolygonGeometry(
            paths=[_ring((0, 0), (10, 0), (10, 10)), _ring((20, 5), (30, 5), (30, 25))]
        ),
    )
    box = annotation_bbox(poly)
    assert box is not None and (box.x, box.y, box.w, box.h) == (0, 0, 30, 25)


def test_to_coco_writes_the_whole_box_and_the_even_odd_area() -> None:
    outer = _ring((0, 0), (10, 0), (10, 10), (0, 10))  # 100
    hole = _ring((2, 2), (4, 2), (4, 4), (2, 4))  # 4, inside the outer
    piece = _ring((20, 0), (30, 0), (30, 10), (20, 10))  # 100, separate
    poly = PolygonAnnotation(name="car", polygon=PolygonGeometry(paths=[outer, hole, piece]))
    (entry,) = to_coco({"a.jpg": [poly]})["annotations"]
    assert entry["bbox"] == [0, 0, 30, 10]
    assert entry["area"] == pytest.approx(196.0)
    assert len(entry["segmentation"]) == 3
