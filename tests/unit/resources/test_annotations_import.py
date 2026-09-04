"""Tests for the ``Annotations.import_*`` methods.

Orchestrator tests: we mock the Client's resource methods (the underlying HTTP +
the COCO/YOLO parsing are covered by resources/ and test_formats.py) and focus on
the import recipe - filename→id matching, missing-class creation, chunked
bulk_save, YOLO dimension resolution, and per-image / unmatched reporting.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from pictograph.exceptions import ApiError, NotFoundError
from pictograph.models.dataset import Dataset, DatasetClass
from pictograph.models.image import Image
from pictograph.resources.annotations import (
    AnnotationImportReport,
    Annotations,
    BulkSaveFailure,
    BulkSaveResult,
    SaveResult,
)
from tests.unit.resources._orchestration import build, sibling_resources


def _invoke(method: str, client: MagicMock, *args: object, **kwargs: object) -> object:
    """Invoke the real importer on a real resource with its siblings stubbed."""
    with sibling_resources(client):
        resource = build(Annotations, client, own="annotations", delegate=["bulk_save"])
        return getattr(resource, method)(*args, **kwargs)


def _image(
    image_id: str, filename: str, *, width: int | None = 100, height: int | None = 100
) -> Image:
    return Image(
        id=image_id,
        filename=filename,
        image_url="https://cdn/x",
        width=width,
        height=height,
        created_at=datetime.now(timezone.utc),
    )


def _project(classes: list[DatasetClass] | None = None) -> Dataset:
    return Dataset(
        id="proj-1",
        name="road-signs",
        organization_id="org-1",
        classes=classes or [],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _client(
    images: list[Image],
    *,
    project: Dataset | None = None,
    bulk_save: MagicMock | None = None,
) -> MagicMock:
    client = MagicMock()
    client.datasets.get.return_value = project or _project()
    client.images.iter.return_value = images
    client.annotations.bulk_save = bulk_save or MagicMock(
        return_value=BulkSaveResult(
            saved=[
                SaveResult(image_id=img.id, previous_count=0, new_count=1, status="in_progress")
                for img in images
            ],
            failed=[],
        )
    )
    return client


def _coco(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
        "categories": [{"id": 1, "name": "car"}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]},
            {"id": 11, "image_id": 2, "category_id": 1, "bbox": [1, 1, 5, 5]},
        ],
    }
    base.update(over)
    return base


# ───────────── COCO ─────────────


def test_import_coco_happy_path() -> None:
    client = _client([_image("img-a", "a.jpg"), _image("img-b", "b.jpg")])
    report = _invoke("import_coco", client, "road-signs", _coco())
    assert isinstance(report, AnnotationImportReport)
    assert report.images_matched == 2
    assert report.images_saved == 2
    assert report.annotations_saved == 2
    assert report.unmatched_files == []
    assert report.success
    # bulk_save called once with both image ids.
    saved_map = client.annotations.bulk_save.call_args[0][0]
    assert set(saved_map) == {"img-a", "img-b"}


def test_import_coco_unmatched_file() -> None:
    # COCO references a.jpg + b.jpg but the dataset only has a.jpg.
    client = _client([_image("img-a", "a.jpg")])
    report = _invoke("import_coco", client, "road-signs", _coco())
    assert report.images_matched == 1
    assert report.unmatched_files == ["b.jpg"]
    assert not report.success  # unmatched => not fully successful


def test_import_coco_creates_missing_classes() -> None:
    client = _client([_image("img-a", "a.jpg")])
    _invoke("import_coco", client, "road-signs", _coco(), create_missing_classes=True)
    client.datasets.update.assert_called_once()
    new_classes = client.datasets.update.call_args.kwargs["classes"]
    assert any(c.name == "car" and c.type == "bbox" for c in new_classes)


def test_import_coco_skips_class_creation_when_disabled() -> None:
    client = _client([_image("img-a", "a.jpg")])
    _invoke("import_coco", client, "road-signs", _coco(), create_missing_classes=False)
    client.datasets.update.assert_not_called()


def test_import_coco_existing_class_not_recreated() -> None:
    project = _project(classes=[DatasetClass(name="car", type="bbox")])
    client = _client([_image("img-a", "a.jpg")], project=project)
    _invoke("import_coco", client, "road-signs", _coco())
    client.datasets.update.assert_not_called()  # car already exists


def test_import_coco_bulk_save_partial_failure() -> None:
    bulk = MagicMock(
        return_value=BulkSaveResult(
            saved=[
                SaveResult(image_id="img-a", previous_count=0, new_count=1, status="in_progress")
            ],
            failed=[BulkSaveFailure(image_id="img-b", error="nope")],
        )
    )
    client = _client([_image("img-a", "a.jpg"), _image("img-b", "b.jpg")], bulk_save=bulk)
    report = _invoke("import_coco", client, "road-signs", _coco())
    assert report.images_saved == 1
    assert len(report.failures) == 1 and report.failures[0].reason == "nope"


def test_import_coco_bulk_save_raises_is_reported() -> None:
    bulk = MagicMock(side_effect=ApiError("boom", status_code=500))
    client = _client([_image("img-a", "a.jpg")], bulk_save=bulk)
    report = _invoke("import_coco", client, "road-signs", _coco())
    assert report.images_saved == 0
    assert len(report.failures) == 1 and "boom" in report.failures[0].reason


def test_import_coco_chunks_bulk_save() -> None:
    images = [_image(f"img-{i}", f"{i}.jpg") for i in range(5)]
    coco = {
        "images": [{"id": i, "file_name": f"{i}.jpg"} for i in range(5)],
        "categories": [{"id": 1, "name": "car"}],
        "annotations": [{"image_id": i, "category_id": 1, "bbox": [1, 1, 2, 2]} for i in range(5)],
    }
    client = _client(images)
    _invoke("import_coco", client, "road-signs", coco, save_chunk=2, create_missing_classes=False)
    # 5 images / chunk 2 => 3 bulk_save calls.
    assert client.annotations.bulk_save.call_count == 3


def test_import_coco_dataset_not_found_propagates() -> None:
    client = MagicMock()
    client.datasets.get.side_effect = NotFoundError("no such dataset", status_code=404)
    with pytest.raises(NotFoundError):
        _invoke("import_coco", client, "ghost", _coco())


# ───────────── YOLO ─────────────


def test_import_yolo_happy_path() -> None:
    client = _client([_image("img-a", "a.jpg", width=100, height=100)])
    labels = {"a.jpg": "0 0.5 0.5 0.2 0.2"}
    report = _invoke("import_yolo", client, "road-signs", labels, ["car"])
    assert report.images_matched == 1
    assert report.images_saved == 1
    saved_map = client.annotations.bulk_save.call_args[0][0]
    ann = saved_map["img-a"][0]
    assert ann.type == "bbox" and ann.name == "car"


def test_import_yolo_missing_dims_is_failure() -> None:
    client = _client([_image("img-a", "a.jpg", width=None, height=None)])
    report = _invoke("import_yolo", client, "road-signs", {"a.jpg": "0 0.5 0.5 0.2 0.2"}, ["car"])
    assert report.images_saved == 0
    assert len(report.failures) == 1 and "width/height" in report.failures[0].reason


def test_import_yolo_unmatched_label() -> None:
    client = _client([_image("img-a", "a.jpg")])
    report = _invoke("import_yolo", client, "road-signs", {"z.jpg": "0 0.5 0.5 0.2 0.2"}, ["car"])
    assert report.unmatched_files == ["z.jpg"]
    assert report.images_matched == 0


# ───────────── Pascal VOC ─────────────

_VOC = (
    "<annotation><filename>{f}</filename>"
    "<object><name>car</name><bndbox>"
    "<xmin>10</xmin><ymin>20</ymin><xmax>40</xmax><ymax>60</ymax></bndbox></object></annotation>"
)


def test_import_pascal_voc_happy_path() -> None:

    client = _client([_image("img-a", "a.jpg")])
    report = _invoke("import_pascal_voc", client, "road-signs", {"a.jpg": _VOC.format(f="a.jpg")})
    assert report.images_matched == 1 and report.images_saved == 1
    saved_map = client.annotations.bulk_save.call_args[0][0]
    assert saved_map["img-a"][0].name == "car" and saved_map["img-a"][0].type == "bbox"


def test_import_pascal_voc_malformed_is_failure_not_crash() -> None:

    # bulk_save echoes exactly the chunk it received (only a.jpg saves; b.jpg is a
    # parse failure recorded before matching, so it never reaches a chunk).
    bulk = MagicMock(
        side_effect=lambda chunk: BulkSaveResult(
            saved=[
                SaveResult(image_id=i, previous_count=0, new_count=1, status="in_progress")
                for i in chunk
            ],
            failed=[],
        )
    )
    client = _client([_image("img-a", "a.jpg"), _image("img-b", "b.jpg")], bulk_save=bulk)
    report = _invoke(
        "import_pascal_voc",
        client,
        "road-signs",
        {"a.jpg": _VOC.format(f="a.jpg"), "b.jpg": "<annotation><object>"},  # b is malformed
    )
    assert report.images_saved == 1  # a saved
    assert any(f.image_filename == "b.jpg" for f in report.failures)  # b recorded, no crash


def test_import_pascal_voc_unmatched() -> None:

    client = _client([_image("img-a", "a.jpg")])
    report = _invoke("import_pascal_voc", client, "road-signs", {"z.jpg": _VOC.format(f="z.jpg")})
    assert report.unmatched_files == ["z.jpg"]
