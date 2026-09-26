"""Tests for ``pictograph.resources.annotations.Annotations``.

Coverage targets:
- ``get`` parses a heterogeneous list of canonical annotations into the
  correct discriminated-union subclasses (BBox / Polygon / Polyline /
  Keypoint) - pinning the wire format the backend is expected to emit.
- ``save`` serialises ``Annotation`` Pydantic models to the canonical wire
  format (``bounding_box`` object, polygon ``paths`` of ``{x,y}`` objects,
  ``polyline.path``, ``keypoint`` object) - the test reads the actual HTTP
  body and asserts the structure key by key. This is the regression guard
  against future drift back to shorthand.
- ``delete`` returns the count, propagates 403/404 typed errors.
- ``SaveResult`` and ``DeleteResult`` are frozen dataclasses (immutable).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import ForbiddenError, NotFoundError, ValidationError
from pictograph.models.annotation import (
    BBoxAnnotation,
    KeypointAnnotation,
    PolygonAnnotation,
    PolygonGeometry,
    PolylineAnnotation,
    PolylineGeometry,
)
from pictograph.models.common import BoundingBox, Point
from pictograph.resources.annotations import (
    Annotations,
    BulkSaveResult,
    DeleteResult,
    SaveResult,
)

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"


@pytest.fixture
def transport() -> Transport:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def annotations(transport: Transport) -> Annotations:
    return Annotations(transport)


# ───────────── get ─────────────


def test_get_parses_canonical_annotations_into_typed_models(
    httpx_mock: HTTPXMock, annotations: Annotations
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        json={
            "success": True,
            "image_id": "99999999-8888-7777-6666-555555555555",
            "filename": "img.jpg",
            "annotations": [
                {
                    "id": "a1",
                    "name": "person",
                    "type": "bbox",
                    "bounding_box": {"x": 10.0, "y": 20.0, "w": 50.0, "h": 80.0},
                },
                {
                    "id": "a2",
                    "name": "car",
                    "type": "polygon",
                    "polygon": {
                        "paths": [
                            [
                                {"x": 0.0, "y": 0.0},
                                {"x": 10.0, "y": 0.0},
                                {"x": 5.0, "y": 8.0},
                            ],
                        ],
                    },
                },
                {
                    "id": "a3",
                    "name": "road",
                    "type": "polyline",
                    "polyline": {"path": [{"x": 0.0, "y": 0.0}, {"x": 100.0, "y": 100.0}]},
                },
                {
                    "id": "a4",
                    "name": "landmark",
                    "type": "keypoint",
                    "keypoint": {"x": 50.0, "y": 60.0},
                },
            ],
            "annotation_count": 4,
        },
    )
    result = annotations.get("road-signs", "img.jpg")
    assert len(result) == 4
    types = [type(a) for a in result]
    assert types == [BBoxAnnotation, PolygonAnnotation, PolylineAnnotation, KeypointAnnotation]


def test_get_empty_annotations_list(httpx_mock: HTTPXMock, annotations: Annotations) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        json={
            "success": True,
            "image_id": "99999999-8888-7777-6666-555555555555",
            "annotations": [],
            "annotation_count": 0,
        },
    )
    assert annotations.get("road-signs", "img.jpg") == []


def test_get_missing_annotations_key_returns_empty(
    httpx_mock: HTTPXMock, annotations: Annotations
) -> None:
    # Defensive: if backend regression drops the key, return empty rather than crash.
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        json={"success": True, "image_id": "99999999-8888-7777-6666-555555555555"},
    )
    assert annotations.get("road-signs", "img.jpg") == []


def test_get_404_raises_not_found(httpx_mock: HTTPXMock, annotations: Annotations) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        status_code=404,
        json={"detail": "Image not found"},
    )
    with pytest.raises(NotFoundError):
        annotations.get("road-signs", "img.jpg")


def test_get_rejects_shorthand_response_from_backend(
    httpx_mock: HTTPXMock, annotations: Annotations
) -> None:
    # If the backend regresses and emits shorthand `bbox: [x,y,w,h]` instead of
    # `bounding_box: {...}`, the discriminated-union validator must fail -
    # surfacing the contract break loudly rather than silently corrupting data.
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        json={
            "annotations": [
                {"id": "a1", "name": "x", "type": "bbox", "bbox": [1, 2, 3, 4]},
            ],
        },
    )
    with pytest.raises(Exception) as exc:
        annotations.get("road-signs", "img.jpg")
    # Pydantic ValidationError surfaces; we don't wrap it here because malformed
    # backend data is exceptional and the original error path is more useful.
    assert (
        "validation" in str(exc.value).lower()
        or "99999999-8888-7777-6666-555555555555" in str(exc.value).lower()
    )


# ───────────── save - wire format ─────────────


def _captured_save_body(httpx_mock: HTTPXMock) -> dict[str, Any]:
    """Pull the body of the most recent request and parse JSON."""
    sent = httpx_mock.get_request()
    assert sent is not None
    return json.loads(sent.read())


def test_save_serialises_bbox_annotation_in_canonical_format(
    httpx_mock: HTTPXMock, annotations: Annotations
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        json={
            "success": True,
            "image_id": "99999999-8888-7777-6666-555555555555",
            "previous_count": 0,
            "new_count": 1,
            "status": "in_progress",
        },
    )
    ann = BBoxAnnotation(
        id="a1",
        name="person",
        bounding_box=BoundingBox(x=100.0, y=200.0, w=50.0, h=80.0),
    )
    annotations.save("road-signs", "img.jpg", [ann])
    body = _captured_save_body(httpx_mock)
    assert len(body["annotations"]) == 1
    sent_ann = body["annotations"][0]
    # Canonical wire format: dict, not array, for bounding_box.
    assert sent_ann["bounding_box"] == {"x": 100.0, "y": 200.0, "w": 50.0, "h": 80.0}
    assert "bbox" not in sent_ann
    assert sent_ann["type"] == "bbox"


def test_bulk_save_round_trip_and_wire_shape(
    httpx_mock: HTTPXMock, annotations: Annotations
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/annotations/bulk",
        json={
            "success": True,
            "saved": [
                {
                    "image_id": "99999999-8888-7777-6666-555555555555",
                    "previous_count": 0,
                    "new_count": 1,
                    "status": "in_progress",
                },
            ],
            "failed": [{"image_id": "img-2", "error": "not found"}],
        },
    )
    ann = BBoxAnnotation(
        id="a1", name="person", bounding_box=BoundingBox(x=1.0, y=2.0, w=3.0, h=4.0)
    )
    result = annotations.bulk_save({"99999999-8888-7777-6666-555555555555": [ann], "img-2": []})

    assert isinstance(result, BulkSaveResult)
    assert result.saved_count == 1
    assert result.saved[0].image_id == "99999999-8888-7777-6666-555555555555"
    assert result.saved[0].new_count == 1
    assert result.failed[0].image_id == "img-2"
    assert result.failed[0].error == "not found"

    # Wire shape: a `saves` array, one entry per image, canonical annotation JSON.
    body = _captured_save_body(httpx_mock)
    assert [s["image_id"] for s in body["saves"]] == [
        "99999999-8888-7777-6666-555555555555",
        "img-2",
    ]
    assert body["saves"][0]["annotations"][0]["bounding_box"] == {
        "x": 1.0,
        "y": 2.0,
        "w": 3.0,
        "h": 4.0,
    }
    assert body["saves"][1]["annotations"] == []


def test_save_serialises_polygon_with_holes_in_canonical_format(
    httpx_mock: HTTPXMock, annotations: Annotations
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        json={
            "image_id": "99999999-8888-7777-6666-555555555555",
            "previous_count": 0,
            "new_count": 1,
            "status": "in_progress",
        },
    )
    outer = [Point(x=0, y=0), Point(x=100, y=0), Point(x=100, y=100), Point(x=0, y=100)]
    hole = [Point(x=20, y=20), Point(x=40, y=20), Point(x=40, y=40), Point(x=20, y=40)]
    ann = PolygonAnnotation(
        id="a2",
        name="building",
        polygon=PolygonGeometry(paths=[outer, hole]),
    )
    annotations.save("road-signs", "img.jpg", [ann])
    body = _captured_save_body(httpx_mock)
    sent = body["annotations"][0]
    # Multi-path polygon preserves both rings; each point is a {x,y} object.
    assert len(sent["polygon"]["paths"]) == 2
    assert sent["polygon"]["paths"][0][0] == {"x": 0.0, "y": 0.0}
    assert sent["polygon"]["paths"][1][0] == {"x": 20.0, "y": 20.0}
    # bounding_box is omitted on save (server computes it).
    assert "bounding_box" not in sent


def test_save_serialises_polyline_in_canonical_format(
    httpx_mock: HTTPXMock, annotations: Annotations
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        json={
            "image_id": "99999999-8888-7777-6666-555555555555",
            "previous_count": 0,
            "new_count": 1,
            "status": "in_progress",
        },
    )
    ann = PolylineAnnotation(
        id="a3",
        name="road",
        polyline=PolylineGeometry(
            path=[Point(x=0, y=100), Point(x=50, y=100), Point(x=100, y=100)]
        ),
    )
    annotations.save("road-signs", "img.jpg", [ann])
    body = _captured_save_body(httpx_mock)
    sent = body["annotations"][0]
    # polyline.path (singular) is a list of {x,y} objects, not nested.
    assert sent["polyline"] == {
        "path": [{"x": 0.0, "y": 100.0}, {"x": 50.0, "y": 100.0}, {"x": 100.0, "y": 100.0}]
    }


def test_save_serialises_keypoint_in_canonical_format(
    httpx_mock: HTTPXMock, annotations: Annotations
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        json={
            "image_id": "99999999-8888-7777-6666-555555555555",
            "previous_count": 0,
            "new_count": 1,
            "status": "in_progress",
        },
    )
    ann = KeypointAnnotation(id="a4", name="landmark", keypoint=Point(x=150, y=200))
    annotations.save("road-signs", "img.jpg", [ann])
    body = _captured_save_body(httpx_mock)
    sent = body["annotations"][0]
    assert sent["keypoint"] == {"x": 150.0, "y": 200.0}


def test_save_omits_none_optional_fields(httpx_mock: HTTPXMock, annotations: Annotations) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        json={
            "image_id": "99999999-8888-7777-6666-555555555555",
            "previous_count": 0,
            "new_count": 1,
            "status": "in_progress",
        },
    )
    ann = BBoxAnnotation(
        id="a1",
        name="person",
        bounding_box=BoundingBox(x=0, y=0, w=10, h=10),
        # No created_by, no overrides
    )
    annotations.save("road-signs", "img.jpg", [ann])
    body = _captured_save_body(httpx_mock)
    sent = body["annotations"][0]
    assert "created_by" not in sent  # exclude_none kicked in


def test_save_serialises_multiple_annotations_in_one_call(
    httpx_mock: HTTPXMock, annotations: Annotations
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        json={
            "image_id": "99999999-8888-7777-6666-555555555555",
            "previous_count": 0,
            "new_count": 4,
            "status": "complete",
        },
    )
    anns = [
        BBoxAnnotation(id="a", name="person", bounding_box=BoundingBox(x=0, y=0, w=10, h=10)),
        PolygonAnnotation(
            id="b",
            name="car",
            polygon=PolygonGeometry(paths=[[Point(x=0, y=0), Point(x=1, y=0), Point(x=0, y=1)]]),
        ),
        PolylineAnnotation(
            id="c",
            name="road",
            polyline=PolylineGeometry(path=[Point(x=0, y=0), Point(x=1, y=1)]),
        ),
        KeypointAnnotation(id="d", name="pt", keypoint=Point(x=0, y=0)),
    ]
    annotations.save("road-signs", "img.jpg", anns)
    body = _captured_save_body(httpx_mock)
    assert len(body["annotations"]) == 4
    types = [a["type"] for a in body["annotations"]]
    assert types == ["bbox", "polygon", "polyline", "keypoint"]


def test_save_returns_typed_save_result(httpx_mock: HTTPXMock, annotations: Annotations) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        json={
            "success": True,
            "image_id": "99999999-8888-7777-6666-555555555555",
            "previous_count": 5,
            "new_count": 7,
            "status": "in_progress",
        },
    )
    ann = BBoxAnnotation(id="a", name="x", bounding_box=BoundingBox(x=0, y=0, w=1, h=1))
    result = annotations.save("road-signs", "img.jpg", [ann])
    assert isinstance(result, SaveResult)
    assert result.image_id == "99999999-8888-7777-6666-555555555555"
    assert result.previous_count == 5
    assert result.new_count == 7
    assert result.status == "in_progress"


def test_save_empty_list_clears_annotations(
    httpx_mock: HTTPXMock, annotations: Annotations
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        json={
            "image_id": "99999999-8888-7777-6666-555555555555",
            "previous_count": 3,
            "new_count": 0,
            "status": "new",
        },
    )
    result = annotations.save("road-signs", "img.jpg", [])
    body = _captured_save_body(httpx_mock)
    assert body["annotations"] == []
    assert result.new_count == 0


def test_save_404_propagates(httpx_mock: HTTPXMock, annotations: Annotations) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        status_code=404,
        json={"detail": "Image not found"},
    )
    ann = BBoxAnnotation(id="a", name="x", bounding_box=BoundingBox(x=0, y=0, w=1, h=1))
    with pytest.raises(NotFoundError):
        annotations.save("road-signs", "img.jpg", [ann])


def test_save_403_propagates(httpx_mock: HTTPXMock, annotations: Annotations) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        status_code=403,
        json={"detail": "Insufficient permissions"},
    )
    ann = BBoxAnnotation(id="a", name="x", bounding_box=BoundingBox(x=0, y=0, w=1, h=1))
    with pytest.raises(ForbiddenError):
        annotations.save("road-signs", "img.jpg", [ann])


def test_save_400_propagates_as_validation_error(
    httpx_mock: HTTPXMock, annotations: Annotations
) -> None:
    # Backend rejected the payload (e.g., class name not in project).
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        status_code=400,
        json={"detail": "Class 'unknown_class' not in project"},
    )
    ann = BBoxAnnotation(id="a", name="unknown_class", bounding_box=BoundingBox(x=0, y=0, w=1, h=1))
    with pytest.raises(ValidationError):
        annotations.save("road-signs", "img.jpg", [ann])


# ───────────── delete ─────────────


def test_delete_returns_typed_delete_result(
    httpx_mock: HTTPXMock, annotations: Annotations
) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        json={
            "success": True,
            "image_id": "99999999-8888-7777-6666-555555555555",
            "deleted_count": 5,
            "message": "Annotations deleted",
        },
    )
    result = annotations.delete("road-signs", "img.jpg")
    assert isinstance(result, DeleteResult)
    assert result.image_id == "99999999-8888-7777-6666-555555555555"
    assert result.deleted_count == 5


def test_delete_when_no_annotations_returns_zero_count(
    httpx_mock: HTTPXMock, annotations: Annotations
) -> None:
    # Backend returns "No annotations to delete" with no deleted_count field.
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        json={
            "success": True,
            "image_id": "99999999-8888-7777-6666-555555555555",
            "message": "No annotations to delete",
        },
    )
    result = annotations.delete("road-signs", "img.jpg")
    assert result.deleted_count == 0


def test_delete_403_propagates(httpx_mock: HTTPXMock, annotations: Annotations) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        status_code=403,
        json={"detail": "Insufficient permissions"},
    )
    with pytest.raises(ForbiddenError):
        annotations.delete("road-signs", "img.jpg")


def test_delete_404_propagates(httpx_mock: HTTPXMock, annotations: Annotations) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/annotations/road-signs/img.jpg",
        status_code=404,
        json={"detail": "Image not found"},
    )
    with pytest.raises(NotFoundError):
        annotations.delete("road-signs", "img.jpg")


# ───────────── result dataclasses ─────────────


def test_save_result_is_frozen() -> None:
    r = SaveResult(image_id="x", previous_count=0, new_count=1, status="in_progress")
    with pytest.raises(Exception):
        r.image_id = "y"  # type: ignore[misc]


def test_delete_result_is_frozen() -> None:
    r = DeleteResult(image_id="x", deleted_count=0)
    with pytest.raises(Exception):
        r.deleted_count = 1  # type: ignore[misc]


# ───────────── rename_class (parity) ─────────────


def test_rename_class_posts_and_parses_envelope(
    httpx_mock: HTTPXMock, annotations: Annotations
) -> None:
    import json

    from pictograph.resources.annotations import RenameClassResult

    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/annotations/rename-class",
        json={
            "data": {
                "dataset_id": "11111111-2222-3333-4444-555555555555",
                "old_name": "car",
                "new_name": "vehicle",
                "images_updated": 4,
                "annotations_updated": 9,
                "config_updated": True,
            }
        },
    )
    result = annotations.rename_class("road-signs", "car", "vehicle")
    assert isinstance(result, RenameClassResult)
    assert result.annotations_updated == 9
    assert result.images_updated == 4
    assert result.config_updated is True
    sent = httpx_mock.get_request()
    assert sent is not None
    body = json.loads(sent.read())
    assert body == {
        "dataset": "road-signs",
        "old_name": "car",
        "new_name": "vehicle",
    }


def test_rename_class_conflict_propagates(httpx_mock: HTTPXMock, annotations: Annotations) -> None:
    from pictograph.exceptions import ConflictError

    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/annotations/rename-class",
        status_code=409,
        json={"detail": "A class named 'vehicle' already exists for that annotation type"},
    )
    with pytest.raises(ConflictError):
        annotations.rename_class("road-signs", "car", "vehicle")


# ───────────── merge_class / delete_class (class management) ─────────────


def test_merge_class_posts_and_parses_envelope(
    httpx_mock: HTTPXMock, annotations: Annotations
) -> None:
    import json

    from pictograph.resources.annotations import MergeClassResult

    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/annotations/merge-class",
        json={
            "data": {
                "dataset_id": "11111111-2222-3333-4444-555555555555",
                "source_name": "auto",
                "target_name": "vehicle",
                "images_updated": 3,
                "annotations_updated": 7,
                "config_updated": True,
            }
        },
    )
    result = annotations.merge_class("road-signs", "auto", "vehicle")
    assert isinstance(result, MergeClassResult)
    assert result.annotations_updated == 7
    assert result.target_name == "vehicle"
    assert result.config_updated is True
    sent = httpx_mock.get_request()
    assert sent is not None
    body = json.loads(sent.read())
    assert body == {
        "dataset": "road-signs",
        "source_name": "auto",
        "target_name": "vehicle",
    }


def test_delete_class_posts_and_parses_envelope(
    httpx_mock: HTTPXMock, annotations: Annotations
) -> None:
    import json

    from pictograph.resources.annotations import DeleteClassResult

    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/annotations/delete-class",
        json={
            "data": {
                "dataset_id": "11111111-2222-3333-4444-555555555555",
                "name": "obsolete",
                "config_updated": True,
                "images_updated": 2,
                "annotations_removed": 5,
            }
        },
    )
    result = annotations.delete_class(
        "road-signs",
        "obsolete",
        class_type="bbox",
        delete_annotations=True,
    )
    assert isinstance(result, DeleteClassResult)
    assert result.annotations_removed == 5
    assert result.config_updated is True
    sent = httpx_mock.get_request()
    assert sent is not None
    body = json.loads(sent.read())
    assert body == {
        "dataset": "road-signs",
        "name": "obsolete",
        "class_type": "bbox",
        "delete_annotations": True,
    }


def test_delete_class_config_only_defaults(httpx_mock: HTTPXMock, annotations: Annotations) -> None:
    import json

    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/annotations/delete-class",
        json={
            "data": {
                "dataset_id": "11111111-2222-3333-4444-555555555555",
                "name": "obsolete",
                "config_updated": True,
                "images_updated": 0,
                "annotations_removed": 0,
            }
        },
    )
    result = annotations.delete_class("road-signs", "obsolete")
    assert result.annotations_removed == 0
    sent = httpx_mock.get_request()
    assert sent is not None
    body = json.loads(sent.read())
    # default: config-only, no type narrowing.
    assert body == {
        "dataset": "road-signs",
        "name": "obsolete",
        "class_type": None,
        "delete_annotations": False,
    }
