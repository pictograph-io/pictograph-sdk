"""Tests for ``pictograph.formats`` - the client-side COCO/YOLO converters.

Covers both directions, round-trip fidelity (bbox is exact; polygon within FP
tolerance), and the documented edge cases (RLE skip, unknown category/image
skip, malformed YOLO lines, class-name validation).
"""

from __future__ import annotations

import json

import pytest

from pictograph.formats import (
    CocoImport,
    from_coco,
    from_pascal_voc,
    from_yolo,
    to_coco,
    to_pascal_voc,
    to_yolo,
)
from pictograph.formats._shared import (
    bbox_from_points,
    points_from_flat,
    polygon_area,
)
from pictograph.models.annotation import (
    BBoxAnnotation,
    KeypointAnnotation,
    PolygonAnnotation,
    PolygonGeometry,
    PolylineAnnotation,
    PolylineGeometry,
)
from pictograph.models.common import BoundingBox, Point


def _coco_min(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "road"}],
        "annotations": [],
    }
    base.update(over)
    return base


# ───────────── shared helpers ─────────────


def test_bbox_from_points_degenerate_is_none() -> None:
    assert bbox_from_points([]) is None
    assert bbox_from_points([Point(x=5, y=5), Point(x=5, y=9)]) is None  # zero width
    box = bbox_from_points([Point(x=0, y=0), Point(x=10, y=4)])
    assert box is not None and (box.x, box.y, box.w, box.h) == (0, 0, 10, 4)


def test_polygon_area_shoelace() -> None:
    square = [Point(x=0, y=0), Point(x=4, y=0), Point(x=4, y=4), Point(x=0, y=4)]
    assert polygon_area(square) == 16.0
    assert polygon_area([Point(x=0, y=0)]) == 0.0


def test_points_from_flat_drops_odd_trailing() -> None:
    pts = points_from_flat([1, 2, 3, 4, 5])  # trailing 5 dropped
    assert pts == [Point(x=1, y=2), Point(x=3, y=4)]


# ───────────── COCO import ─────────────


def test_from_coco_bbox_and_polygon() -> None:
    coco = _coco_min(
        annotations=[
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]},
            {
                "id": 11,
                "image_id": 1,
                "category_id": 2,
                "segmentation": [[0, 0, 50, 0, 50, 50, 0, 50]],
            },
        ]
    )
    imp = from_coco(coco)
    assert isinstance(imp, CocoImport)
    assert imp.class_names == ["car", "road"]
    anns = imp.annotations["a.jpg"]
    assert [a.type for a in anns] == ["bbox", "polygon"]
    box = anns[0]
    assert isinstance(box, BBoxAnnotation)
    assert (box.bounding_box.x, box.bounding_box.w) == (10, 30)
    poly = anns[1]
    assert isinstance(poly, PolygonAnnotation)
    assert len(poly.polygon.paths[0]) == 4


def test_from_coco_score_becomes_confidence() -> None:
    coco = _coco_min(
        annotations=[{"image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2], "score": 0.42}]
    )
    imp = from_coco(coco)
    assert imp.annotations["a.jpg"][0].confidence == pytest.approx(0.42)


def test_coco_attributes_round_trip() -> None:
    # COCO per-annotation ontology attributes survive import → Pictograph → export.
    coco = _coco_min(
        annotations=[
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 1, 2, 2],
                "attributes": {"occluded": "true", "pose": "standing"},
            }
        ]
    )
    imp = from_coco(coco)
    ann = imp.annotations["a.jpg"][0]
    assert ann.attributes == {"occluded": "true", "pose": "standing"}

    out = to_coco({"a.jpg": [ann]})
    assert out["annotations"][0]["attributes"] == {"occluded": "true", "pose": "standing"}


def test_to_coco_omits_empty_or_list_attributes() -> None:
    # Absent / legacy-list attributes are not emitted (byte-unchanged output).
    ann = BBoxAnnotation(name="car", bounding_box=BoundingBox(x=0, y=0, w=5, h=5))
    out = to_coco({"a.jpg": [ann]})
    assert "attributes" not in out["annotations"][0]


def test_from_coco_keypoints_expand_visible_only() -> None:
    coco = _coco_min(
        annotations=[
            {
                "id": 5,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "keypoints": [3, 4, 2, 7, 8, 0, 9, 9, 1],  # middle one v=0 → skipped
            }
        ]
    )
    imp = from_coco(coco)
    anns = imp.annotations["a.jpg"]
    assert all(isinstance(a, KeypointAnnotation) for a in anns)
    assert len(anns) == 2  # 2 visible, 1 skipped
    assert isinstance(anns[0], KeypointAnnotation)
    assert (anns[0].keypoint.x, anns[0].keypoint.y) == (3, 4)


def test_from_coco_skips_rle_and_unknown_refs() -> None:
    coco = _coco_min(
        annotations=[
            {
                "image_id": 1,
                "category_id": 1,
                "segmentation": {"counts": "abc", "size": [100, 100]},
            },
            {"image_id": 99, "category_id": 1, "bbox": [1, 1, 2, 2]},  # unknown image
            {"image_id": 1, "category_id": 99, "bbox": [1, 1, 2, 2]},  # unknown category
        ]
    )
    imp = from_coco(coco)
    assert imp.annotations == {}  # all three skipped


def test_from_coco_from_path_and_string(tmp_path: object) -> None:
    from pathlib import Path

    coco = _coco_min(annotations=[{"image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]}])
    p = Path(str(tmp_path)) / "c.json"
    p.write_text(json.dumps(coco), encoding="utf-8")
    assert from_coco(p).annotations["a.jpg"]
    assert from_coco(str(p)).annotations["a.jpg"]


def test_from_coco_non_object_raises(tmp_path: object) -> None:
    from pathlib import Path

    p = Path(str(tmp_path)) / "bad.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="COCO JSON"):
        from_coco(p)


# ───────────── COCO export ─────────────


def test_to_coco_structure_and_skips_polyline() -> None:
    anns = [
        BBoxAnnotation(name="car", bounding_box=BoundingBox(x=1, y=2, w=3, h=4)),
        PolygonAnnotation(
            name="road",
            polygon=PolygonGeometry(paths=[[Point(x=0, y=0), Point(x=2, y=0), Point(x=2, y=2)]]),
        ),
        PolylineAnnotation(
            name="lane", polyline=PolylineGeometry(path=[Point(x=0, y=0), Point(x=5, y=5)])
        ),
    ]
    coco = to_coco({"a.jpg": anns}, image_sizes={"a.jpg": (100, 80)})
    assert coco["images"] == [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 80}]
    assert {c["name"] for c in coco["categories"]} == {"car", "road"}  # lane skipped (polyline)
    assert len(coco["annotations"]) == 2
    poly_entry = next(a for a in coco["annotations"] if "segmentation" in a)
    assert poly_entry["segmentation"] == [[0, 0, 2, 0, 2, 2]]
    assert poly_entry["iscrowd"] == 0


def test_to_coco_missing_size_defaults_zero() -> None:
    anns = [BBoxAnnotation(name="car", bounding_box=BoundingBox(x=1, y=2, w=3, h=4))]
    coco = to_coco({"a.jpg": anns})
    assert coco["images"][0]["width"] == 0 and coco["images"][0]["height"] == 0


def test_to_coco_keypoint_entry() -> None:
    anns = [KeypointAnnotation(name="nose", keypoint=Point(x=5, y=6))]
    coco = to_coco({"a.jpg": anns})
    entry = coco["annotations"][0]
    assert entry["keypoints"] == [5, 6, 2]
    assert entry["num_keypoints"] == 1


# ───────────── COCO round-trip ─────────────


def test_coco_bbox_roundtrip_exact() -> None:
    coco = _coco_min(
        annotations=[{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]}]
    )
    imp = from_coco(coco)
    back = to_coco(imp.annotations, image_sizes={"a.jpg": (100, 100)})
    assert back["annotations"][0]["bbox"] == [10, 20, 30, 40]
    assert back["categories"][0]["name"] == "car"


# ───────────── YOLO import ─────────────


def test_from_yolo_bbox_and_segmentation() -> None:
    text = "0 0.25 0.40 0.30 0.40\n1 0.0 0.0 0.5 0.0 0.5 0.5\n"
    anns = from_yolo(text, ["car", "road"], 100, 100)
    assert [a.type for a in anns] == ["bbox", "polygon"]
    box = anns[0]
    assert isinstance(box, BBoxAnnotation)
    assert box.bounding_box.x == pytest.approx(10.0)
    assert box.bounding_box.w == pytest.approx(30.0)


def test_from_yolo_skips_blank_malformed_and_oob_class() -> None:
    text = "\n  \n5 0.1 0.1 0.1 0.1\nnot a number\n0 0.5 0.5 0.2 0.2\n"
    anns = from_yolo(text, ["car"], 100, 100)  # class 5 out of range, junk line skipped
    assert len(anns) == 1 and anns[0].name == "car"


def test_from_yolo_bad_dims_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        from_yolo("0 0.5 0.5 0.2 0.2", ["car"], 0, 100)


# ───────────── YOLO export ─────────────


def test_to_yolo_bbox_normalized() -> None:
    box = BBoxAnnotation(name="car", bounding_box=BoundingBox(x=10, y=20, w=30, h=40))
    txt = to_yolo([box], ["car", "road"], 100, 100)
    assert txt == "0 0.250000 0.400000 0.300000 0.400000"


def test_to_yolo_polygon_as_box_vs_segmentation() -> None:
    poly = PolygonAnnotation(
        name="road",
        polygon=PolygonGeometry(paths=[[Point(x=0, y=0), Point(x=50, y=0), Point(x=50, y=50)]]),
    )
    as_box = to_yolo([poly], ["road"], 100, 100)
    assert as_box.startswith("0 ") and len(as_box.split()) == 5  # bbox line
    as_seg = to_yolo([poly], ["road"], 100, 100, segmentation=True)
    assert as_seg.split()[0] == "0" and len(as_seg.split()) == 7  # 3 points x 2 + class


def test_to_yolo_unknown_class_raises() -> None:
    box = BBoxAnnotation(name="ufo", bounding_box=BoundingBox(x=1, y=1, w=2, h=2))
    with pytest.raises(ValueError, match="not in class_names"):
        to_yolo([box], ["car"], 100, 100)


def test_to_yolo_skips_polyline_and_keypoint() -> None:
    anns = [
        PolylineAnnotation(
            name="l", polyline=PolylineGeometry(path=[Point(x=0, y=0), Point(x=1, y=1)])
        ),
        KeypointAnnotation(name="k", keypoint=Point(x=1, y=1)),
    ]
    assert to_yolo(anns, ["l", "k"], 100, 100) == ""


def test_yolo_bbox_roundtrip_exact() -> None:
    box = BBoxAnnotation(name="car", bounding_box=BoundingBox(x=10, y=20, w=30, h=40))
    txt = to_yolo([box], ["car"], 200, 160)
    rt = from_yolo(txt, ["car"], 200, 160)
    b = rt[0].bounding_box
    assert b is not None
    assert (b.x, b.y, b.w, b.h) == pytest.approx((10, 20, 30, 40))


# ───────────── Pascal VOC ─────────────


def test_from_pascal_voc_bbox() -> None:
    xml = """
    <annotation>
      <filename>a.jpg</filename>
      <size><width>100</width><height>100</height></size>
      <object><name>car</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>40</xmax><ymax>60</ymax></bndbox></object>
      <object><name>sign</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>
    </annotation>
    """
    anns = from_pascal_voc(xml)
    assert [a.name for a in anns] == ["car", "sign"]
    car = anns[0]
    assert isinstance(car, BBoxAnnotation)
    assert (car.bounding_box.x, car.bounding_box.w) == (10, 30)


def test_from_pascal_voc_skips_invalid_objects() -> None:
    xml = """
    <annotation>
      <object><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax></bndbox></object>
      <object><name>x</name></object>
      <object><name>y</name><bndbox><xmin>5</xmin><ymin>5</ymin><xmax>5</xmax><ymax>9</ymax></bndbox></object>
      <object><name>ok</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>3</xmax><ymax>3</ymax></bndbox></object>
    </annotation>
    """
    anns = from_pascal_voc(xml)  # no-name, no-box, zero-width all skipped
    assert [a.name for a in anns] == ["ok"]


def test_from_pascal_voc_malformed_raises() -> None:
    with pytest.raises(ValueError, match="Invalid Pascal VOC XML"):
        from_pascal_voc("<annotation><object>")


def test_to_pascal_voc_structure_and_skips() -> None:
    anns = [
        BBoxAnnotation(name="car", bounding_box=BoundingBox(x=10, y=20, w=30, h=40)),
        PolygonAnnotation(
            name="road",
            polygon=PolygonGeometry(paths=[[Point(x=0, y=0), Point(x=50, y=0), Point(x=50, y=50)]]),
        ),
        PolylineAnnotation(
            name="lane", polyline=PolylineGeometry(path=[Point(x=0, y=0), Point(x=5, y=5)])
        ),
    ]
    xml = to_pascal_voc(anns, filename="a.jpg", image_width=100, image_height=80)
    assert "<filename>a.jpg</filename>" in xml
    assert "<width>100</width>" in xml and "<height>80</height>" in xml
    # car (bbox) + road (polygon -> box); lane (polyline) skipped.
    assert xml.count("<object>") == 2
    assert "<name>car</name>" in xml and "<name>road</name>" in xml and "lane" not in xml


def test_pascal_voc_bbox_roundtrip_exact() -> None:
    box = BBoxAnnotation(name="car", bounding_box=BoundingBox(x=10, y=20, w=30, h=40))
    xml = to_pascal_voc([box], filename="a.jpg", image_width=200, image_height=160)
    rt = from_pascal_voc(xml)
    b = rt[0].bounding_box
    assert b is not None
    assert (b.x, b.y, b.w, b.h) == (10, 20, 30, 40)
