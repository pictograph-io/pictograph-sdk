"""Tests for ``pictograph.models.annotation``.

These tests pin the wire format the backend stores. Any drift between
SDK-emitted JSON and the canonical Pictograph JSON is treated as a bug;
the round-trip and explicit-dump tests below catch it.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from pictograph.models.annotation import (
    Annotation,
    AnnotationType,
    BBoxAnnotation,
    KeypointAnnotation,
    PolygonAnnotation,
    PolygonGeometry,
    PolylineAnnotation,
    PolylineGeometry,
)
from pictograph.models.common import BoundingBox, Point

ANNOTATION_ADAPTER: TypeAdapter[Annotation] = TypeAdapter(Annotation)


# ───────────── PolygonGeometry ─────────────


def test_polygon_geometry_single_path_three_points() -> None:
    pg = PolygonGeometry(paths=[[Point(x=0, y=0), Point(x=10, y=0), Point(x=5, y=10)]])
    assert len(pg.paths) == 1
    assert len(pg.paths[0]) == 3


def test_polygon_geometry_multi_path_with_hole() -> None:
    outer = [Point(x=0, y=0), Point(x=100, y=0), Point(x=100, y=100), Point(x=0, y=100)]
    hole = [Point(x=20, y=20), Point(x=40, y=20), Point(x=40, y=40), Point(x=20, y=40)]
    pg = PolygonGeometry(paths=[outer, hole])
    assert len(pg.paths) == 2


@pytest.mark.parametrize("ring_size", [0, 1, 2])
def test_polygon_geometry_rejects_ring_with_fewer_than_three_points(ring_size: int) -> None:
    ring = [Point(x=i, y=i) for i in range(ring_size)]
    with pytest.raises(ValidationError) as exc:
        PolygonGeometry(paths=[ring])
    err = exc.value.errors()[0]
    assert err["loc"][0] == "paths"
    assert "at least 3 points" in str(err["msg"])


def test_polygon_geometry_validates_each_ring_independently() -> None:
    good_ring = [Point(x=0, y=0), Point(x=1, y=0), Point(x=0, y=1)]
    bad_ring = [Point(x=0, y=0), Point(x=1, y=0)]
    with pytest.raises(ValidationError) as exc:
        PolygonGeometry(paths=[good_ring, bad_ring])
    msg = str(exc.value)
    # Error message identifies the offending ring index (1).
    assert "paths[1]" in msg


def test_polygon_geometry_rejects_zero_paths() -> None:
    with pytest.raises(ValidationError) as exc:
        PolygonGeometry(paths=[])
    err = exc.value.errors()[0]
    assert err["type"] == "too_short"
    assert err["loc"] == ("paths",)


def test_polygon_geometry_extra_field_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        PolygonGeometry.model_validate(
            {"paths": [[{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 0, "y": 1}]], "rotation": 45}
        )
    assert exc.value.errors()[0]["type"] == "extra_forbidden"


# ───────────── PolylineGeometry ─────────────


def test_polyline_geometry_minimum_two_points() -> None:
    pl = PolylineGeometry(path=[Point(x=0, y=0), Point(x=10, y=10)])
    assert len(pl.path) == 2


@pytest.mark.parametrize("size", [0, 1])
def test_polyline_geometry_rejects_fewer_than_two_points(size: int) -> None:
    with pytest.raises(ValidationError) as exc:
        PolylineGeometry(path=[Point(x=i, y=i) for i in range(size)])
    err = exc.value.errors()[0]
    assert err["type"] == "too_short"


# ───────────── BBoxAnnotation ─────────────


def _bbox_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "ann-1",
        "name": "person",
        "type": "bbox",
        "bounding_box": {"x": 100, "y": 200, "w": 50, "h": 80},
    }
    base.update(overrides)
    return base


def test_bbox_annotation_construction_from_dict() -> None:
    ann = BBoxAnnotation.model_validate(_bbox_payload())
    assert ann.id == "ann-1"
    assert ann.name == "person"
    assert ann.type == "bbox"
    assert ann.bounding_box == BoundingBox(x=100, y=200, w=50, h=80)
    # Defaults present and correct.
    assert ann.confidence == 1.0
    assert ann.created_by is None
    assert ann.attributes == []


def test_bbox_annotation_type_field_defaults_to_bbox_when_omitted() -> None:
    payload = _bbox_payload()
    payload.pop("type")
    ann = BBoxAnnotation.model_validate(payload)
    assert ann.type == "bbox"


def test_bbox_annotation_type_field_rejects_other_literals() -> None:
    with pytest.raises(ValidationError) as exc:
        BBoxAnnotation.model_validate(_bbox_payload(type="polygon"))
    err = exc.value.errors()[0]
    assert err["loc"] == ("type",)


def test_bbox_annotation_round_trip_preserves_canonical_wire_format() -> None:
    payload = _bbox_payload(confidence=0.85, created_by="user-abc", attributes=[{"k": "v"}])
    ann = BBoxAnnotation.model_validate(payload)
    dumped = ann.model_dump(mode="json")
    # Field order matches the canonical spec from the audit:
    # id, name, type, bounding_box, oriented_box, confidence, created_by, attributes.
    # oriented_box is present only on a ROTATED box; a plain box dumps it as null.
    assert list(dumped.keys()) == [
        "id",
        "name",
        "type",
        "bounding_box",
        "oriented_box",
        "confidence",
        "created_by",
        "attributes",
    ]
    assert dumped["oriented_box"] is None  # a plain, non-rotated box carries no OBB
    # Re-parse and equality holds.
    assert BBoxAnnotation.model_validate(dumped) == ann


def test_bbox_annotation_has_no_oriented_box_by_default() -> None:
    """A plain, non-rotated box carries no oriented metadata - the common case stays
    minimal. An oriented (rotated) box is a bbox that ADDITIONALLY sets oriented_box."""
    ann = BBoxAnnotation.model_validate(_bbox_payload())
    assert ann.oriented_box is None


def test_bbox_annotation_carries_optional_oriented_box() -> None:
    """A rotated box parses as a BBoxAnnotation with oriented_box set; bounding_box is
    still present as the axis-aligned enclosure (what training / OBB-unaware consumers
    read). There is no separate `type:"obb"` and no derived `polygon` key."""
    from pictograph.models.annotation import OrientedBoxGeometry

    payload = _bbox_payload(
        bounding_box={"x": 77.7, "y": 81.3, "w": 44.6, "h": 37.4},
        oriented_box={"cx": 100, "cy": 100, "w": 40, "h": 20, "angle": 30},
    )
    ann = BBoxAnnotation.model_validate(payload)
    assert ann.type == "bbox"
    assert ann.oriented_box == OrientedBoxGeometry(cx=100, cy=100, w=40, h=20, angle=30)
    assert ann.bounding_box == BoundingBox(x=77.7, y=81.3, w=44.6, h=37.4)
    # Round-trips: the oriented_box survives a dump/reparse.
    assert BBoxAnnotation.model_validate(ann.model_dump(mode="json")) == ann


def test_bbox_annotation_rejects_a_stray_polygon_key() -> None:
    """The canonical rotated-box shape carries NO polygon. `extra="forbid"` means a
    stray `polygon` on a bbox is a hard failure (the old obb type is gone)."""
    payload = _bbox_payload(
        oriented_box={"cx": 100, "cy": 100, "w": 40, "h": 20, "angle": 30},
        polygon={"paths": [[{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 0, "y": 1}]]},
    )
    with pytest.raises(ValidationError):
        BBoxAnnotation.model_validate(payload)


def test_obb_type_literal_is_rejected() -> None:
    """`type:"obb"` no longer exists - the discriminated union rejects it."""
    with pytest.raises(ValidationError):
        ANNOTATION_ADAPTER.validate_python(
            {
                "id": "a",
                "name": "ship",
                "type": "obb",
                "oriented_box": {"cx": 1, "cy": 1, "w": 2, "h": 2, "angle": 10},
            }
        )


def test_attributes_accepts_ontology_dict() -> None:
    # Per-annotation ontology attributes as a {name: value} map (CVAT/COCO/Datumaro
    # interop) - accepted alongside the legacy opaque-list form, field order kept.
    payload = _bbox_payload(attributes={"occluded": "true", "pose": "standing"})
    ann = BBoxAnnotation.model_validate(payload)
    assert ann.attributes == {"occluded": "true", "pose": "standing"}
    dumped = ann.model_dump(mode="json")
    assert dumped["attributes"] == {"occluded": "true", "pose": "standing"}
    assert list(dumped.keys())[-1] == "attributes"  # still serializes last
    assert BBoxAnnotation.model_validate(dumped) == ann


def test_attributes_still_accepts_legacy_list() -> None:
    # Back-compat: the legacy opaque-list shape (and the [] default) still validate.
    assert BBoxAnnotation.model_validate(_bbox_payload()).attributes == []
    assert BBoxAnnotation.model_validate(_bbox_payload(attributes=[{"k": "v"}])).attributes == [
        {"k": "v"}
    ]


def test_bbox_annotation_requires_bounding_box() -> None:
    payload = _bbox_payload()
    payload.pop("bounding_box")
    with pytest.raises(ValidationError) as exc:
        BBoxAnnotation.model_validate(payload)
    assert exc.value.errors()[0]["loc"] == ("bounding_box",)
    assert exc.value.errors()[0]["type"] == "missing"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", ""),
        ("id", "   "),
        ("name", ""),
        ("name", "\t"),
    ],
)
def test_bbox_annotation_rejects_blank_id_or_name(field: str, value: str) -> None:
    payload = _bbox_payload(**{field: value})
    with pytest.raises(ValidationError) as exc:
        BBoxAnnotation.model_validate(payload)
    err = exc.value.errors()[0]
    assert err["loc"] == (field,)


@pytest.mark.parametrize("conf", [-0.001, 1.001, -10, 100])
def test_bbox_annotation_rejects_out_of_range_confidence(conf: float) -> None:
    with pytest.raises(ValidationError) as exc:
        BBoxAnnotation.model_validate(_bbox_payload(confidence=conf))
    err = exc.value.errors()[0]
    assert err["loc"] == ("confidence",)


@pytest.mark.parametrize("conf", [0.0, 0.5, 1.0])
def test_bbox_annotation_accepts_in_range_confidence(conf: float) -> None:
    ann = BBoxAnnotation.model_validate(_bbox_payload(confidence=conf))
    assert ann.confidence == conf


def test_bbox_annotation_extra_field_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        BBoxAnnotation.model_validate(_bbox_payload(rotation=45))
    assert exc.value.errors()[0]["type"] == "extra_forbidden"


# ───────────── PolygonAnnotation ─────────────


def _polygon_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "ann-2",
        "name": "car",
        "type": "polygon",
        "polygon": {
            "paths": [
                [{"x": 100, "y": 200}, {"x": 150, "y": 200}, {"x": 125, "y": 250}],
            ],
        },
    }
    base.update(overrides)
    return base


def test_polygon_annotation_round_trip_canonical_with_holes() -> None:
    payload = _polygon_payload(
        bounding_box={"x": 0, "y": 0, "w": 100, "h": 100},
        polygon={
            "paths": [
                # outer ring
                [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 100, "y": 100}, {"x": 0, "y": 100}],
                # hole
                [{"x": 20, "y": 20}, {"x": 40, "y": 20}, {"x": 40, "y": 40}, {"x": 20, "y": 40}],
            ],
        },
    )
    ann = PolygonAnnotation.model_validate(payload)
    assert len(ann.polygon.paths) == 2
    dumped = ann.model_dump(mode="json")
    assert PolygonAnnotation.model_validate(dumped) == ann


def test_polygon_annotation_bounding_box_optional_when_omitted() -> None:
    ann = PolygonAnnotation.model_validate(_polygon_payload())
    # Server computes bounding_box on save; client can omit.
    assert ann.bounding_box is None


def test_polygon_annotation_bounding_box_excluded_from_dump_when_none() -> None:
    ann = PolygonAnnotation.model_validate(_polygon_payload())
    dumped = ann.model_dump(mode="json", exclude_none=True)
    assert "bounding_box" not in dumped


def test_polygon_annotation_canonical_field_order() -> None:
    payload = _polygon_payload(
        bounding_box={"x": 100, "y": 200, "w": 50, "h": 50},
        confidence=0.9,
        created_by="user-1",
        attributes=[],
    )
    dumped = PolygonAnnotation.model_validate(payload).model_dump(mode="json")
    # Per audit: id, name, type, bounding_box, polygon, confidence, created_by, attributes
    assert list(dumped.keys()) == [
        "id",
        "name",
        "type",
        "bounding_box",
        "polygon",
        "confidence",
        "created_by",
        "attributes",
    ]


def test_polygon_annotation_invalid_ring_propagates_validation_error() -> None:
    with pytest.raises(ValidationError) as exc:
        PolygonAnnotation.model_validate(
            _polygon_payload(polygon={"paths": [[{"x": 0, "y": 0}, {"x": 1, "y": 1}]]})
        )
    msg = str(exc.value)
    assert "polygon" in msg
    assert "at least 3 points" in msg


# ───────────── PolylineAnnotation ─────────────


def _polyline_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "ann-3",
        "name": "road",
        "type": "polyline",
        "polyline": {"path": [{"x": 0, "y": 100}, {"x": 50, "y": 100}, {"x": 100, "y": 100}]},
    }
    base.update(overrides)
    return base


def test_polyline_annotation_construction() -> None:
    ann = PolylineAnnotation.model_validate(_polyline_payload())
    assert len(ann.polyline.path) == 3
    assert ann.bounding_box is None


def test_polyline_annotation_canonical_field_order() -> None:
    payload = _polyline_payload(bounding_box={"x": 0, "y": 100, "w": 100, "h": 1})
    dumped = PolylineAnnotation.model_validate(payload).model_dump(mode="json")
    assert list(dumped.keys()) == [
        "id",
        "name",
        "type",
        "bounding_box",
        "polyline",
        "confidence",
        "created_by",
        "attributes",
    ]


# ───────────── KeypointAnnotation ─────────────


def _keypoint_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "ann-4",
        "name": "landmark",
        "type": "keypoint",
        "keypoint": {"x": 150, "y": 200},
    }
    base.update(overrides)
    return base


def test_keypoint_annotation_construction() -> None:
    ann = KeypointAnnotation.model_validate(_keypoint_payload())
    assert ann.keypoint == Point(x=150, y=200)


def test_keypoint_annotation_canonical_field_order_has_no_bounding_box() -> None:
    dumped = KeypointAnnotation.model_validate(_keypoint_payload()).model_dump(mode="json")
    # Keypoints have zero extent - there is no bounding_box field.
    assert "bounding_box" not in dumped
    # `instance_id` goes LAST, after `attributes`: the wire order is shared with the
    # backend and the editor, so a new optional field must not shift anything before it.
    assert list(dumped.keys()) == [
        "id",
        "name",
        "type",
        "keypoint",
        "confidence",
        "created_by",
        "attributes",
        "instance_id",
    ]


# ───────────── Discriminated Annotation union ─────────────


@pytest.mark.parametrize(
    ("payload_factory", "expected_cls"),
    [
        (_bbox_payload, BBoxAnnotation),
        (_polygon_payload, PolygonAnnotation),
        (_polyline_payload, PolylineAnnotation),
        (_keypoint_payload, KeypointAnnotation),
    ],
)
def test_annotation_union_dispatches_on_type_field(
    payload_factory: Any,
    expected_cls: type,
) -> None:
    ann = ANNOTATION_ADAPTER.validate_python(payload_factory())
    assert isinstance(ann, expected_cls)


def test_annotation_union_missing_type_raises_discriminator_error() -> None:
    payload = _bbox_payload()
    payload.pop("type")
    with pytest.raises(ValidationError) as exc:
        ANNOTATION_ADAPTER.validate_python(payload)
    err = exc.value.errors()[0]
    # Pydantic v2 emits "union_tag_not_found" or "missing" depending on version.
    assert err["type"] in {"union_tag_not_found", "missing"}


def test_annotation_union_unknown_type_raises_discriminator_error() -> None:
    payload = _bbox_payload(type="not-a-real-type")
    with pytest.raises(ValidationError) as exc:
        ANNOTATION_ADAPTER.validate_python(payload)
    err = exc.value.errors()[0]
    assert err["type"] == "union_tag_invalid"


def test_annotation_union_round_trip_preserves_canonical_format() -> None:
    payloads = [
        _bbox_payload(),
        _polygon_payload(),
        _polyline_payload(),
        _keypoint_payload(),
    ]
    parsed = [ANNOTATION_ADAPTER.validate_python(p) for p in payloads]
    dumped = [a.model_dump(mode="json", exclude_none=True) for a in parsed]
    re_parsed = [ANNOTATION_ADAPTER.validate_python(d) for d in dumped]
    assert parsed == re_parsed


def test_annotation_type_alias_covers_all_annotation_types() -> None:
    # If a new annotation type is added without updating AnnotationType,
    # this assertion will fail and the test serves as an exhaustiveness check.
    expected: set[AnnotationType] = {"bbox", "polygon", "polyline", "keypoint"}
    declared = {
        BBoxAnnotation.model_fields["type"].default,
        PolygonAnnotation.model_fields["type"].default,
        PolylineAnnotation.model_fields["type"].default,
        KeypointAnnotation.model_fields["type"].default,
    }
    assert declared == expected


# ───────────── List[Annotation] round-trip ─────────────


def test_list_of_annotations_round_trips_through_typeadapter() -> None:
    list_adapter: TypeAdapter[list[Annotation]] = TypeAdapter(list[Annotation])
    payloads = [_bbox_payload(), _polygon_payload(), _polyline_payload(), _keypoint_payload()]
    annotations = list_adapter.validate_python(payloads)
    assert [type(a) for a in annotations] == [
        BBoxAnnotation,
        PolygonAnnotation,
        PolylineAnnotation,
        KeypointAnnotation,
    ]
    re_dumped = list_adapter.dump_python(annotations, mode="json", exclude_none=True)
    re_parsed = list_adapter.validate_python(re_dumped)
    assert annotations == re_parsed


# ───────────── JSON Schema generation ─────────────


def test_annotation_json_schema_uses_type_as_discriminator() -> None:
    schema = ANNOTATION_ADAPTER.json_schema()
    # The schema should advertise `type` as the discriminator so agent frameworks
    # can generate correctly-typed tool inputs.
    assert "discriminator" in schema or "oneOf" in schema or "anyOf" in schema
    # All four type literals must appear somewhere in the schema.
    schema_str = repr(schema)
    for literal in ("bbox", "polygon", "polyline", "keypoint"):
        assert literal in schema_str
