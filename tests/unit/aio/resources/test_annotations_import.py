"""Tests for the ``AsyncAnnotations.import_*`` methods.

Mirrors the sync ``tests/unit/pipelines/test_import_annotations.py`` - mocks the
AsyncClient resources (coroutine methods via ``AsyncMock``, ``images.iter`` as an
async iterator) and verifies the same recipe plus the concurrent chunked save.
Inherits the ``anyio_backend`` fixture from ``tests/unit/aio/conftest.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from pictograph.aio.resources.annotations import AsyncAnnotations
from pictograph.exceptions import ApiError, NotFoundError
from pictograph.models.dataset import Dataset, DatasetClass
from pictograph.models.image import Image
from pictograph.resources.annotations import (
    AnnotationImportReport,
    BulkSaveFailure,
    BulkSaveResult,
    SaveResult,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

from tests.unit.resources._orchestration import build, sibling_resources


async def _invoke(method: str, client: MagicMock, *args: object, **kwargs: object) -> object:
    """Invoke the real importer on a real async resource with its siblings stubbed."""
    with sibling_resources(client, is_async=True):
        resource = build(AsyncAnnotations, client, own="annotations", delegate=["bulk_save"])
        return await getattr(resource, method)(*args, **kwargs)


pytestmark = pytest.mark.anyio


class _AsyncIter:
    """Minimal async iterator over a fixed list (mirrors AsyncOffsetPager)."""

    def __init__(self, items: Iterable[Image]) -> None:
        self._items = list(items)

    async def __aiter__(self) -> AsyncIterator[Image]:
        for item in self._items:
            yield item


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


def _async_client(
    images: list[Image],
    *,
    project: Dataset | None = None,
    bulk_save: AsyncMock | None = None,
) -> MagicMock:
    client = MagicMock()
    client.datasets.get = AsyncMock(return_value=project or _project())
    client.datasets.update = AsyncMock()
    client.images.iter = MagicMock(return_value=_AsyncIter(images))
    client.annotations.bulk_save = bulk_save or AsyncMock(
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


async def test_async_import_coco_happy_path() -> None:
    client = _async_client([_image("img-a", "a.jpg"), _image("img-b", "b.jpg")])
    report = await _invoke("import_coco", client, "road-signs", _coco())
    assert isinstance(report, AnnotationImportReport)
    assert report.images_matched == 2
    assert report.images_saved == 2
    assert report.annotations_saved == 2
    assert report.success


async def test_async_import_coco_unmatched() -> None:
    client = _async_client([_image("img-a", "a.jpg")])
    report = await _invoke("import_coco", client, "road-signs", _coco())
    assert report.unmatched_files == ["b.jpg"]
    assert not report.success


async def test_async_import_coco_creates_classes() -> None:
    client = _async_client([_image("img-a", "a.jpg")])
    await _invoke("import_coco", client, "road-signs", _coco(), create_missing_classes=True)
    client.datasets.update.assert_awaited_once()
    new_classes = client.datasets.update.await_args.kwargs["classes"]
    assert any(c.name == "car" and c.type == "bbox" for c in new_classes)


async def test_async_import_coco_skips_class_creation() -> None:
    client = _async_client([_image("img-a", "a.jpg")])
    await _invoke("import_coco", client, "road-signs", _coco(), create_missing_classes=False)
    client.datasets.update.assert_not_called()


async def test_async_import_coco_concurrent_chunks() -> None:
    images = [_image(f"img-{i}", f"{i}.jpg") for i in range(5)]
    coco = {
        "images": [{"id": i, "file_name": f"{i}.jpg"} for i in range(5)],
        "categories": [{"id": 1, "name": "car"}],
        "annotations": [{"image_id": i, "category_id": 1, "bbox": [1, 1, 2, 2]} for i in range(5)],
    }
    client = _async_client(images)
    await _invoke(
        "import_coco", client, "road-signs", coco, save_chunk=2, create_missing_classes=False
    )
    # 5 images / chunk 2 => 3 concurrent bulk_save calls.
    assert client.annotations.bulk_save.await_count == 3


async def test_async_import_coco_partial_failure() -> None:
    bulk = AsyncMock(
        return_value=BulkSaveResult(
            saved=[
                SaveResult(image_id="img-a", previous_count=0, new_count=1, status="in_progress")
            ],
            failed=[BulkSaveFailure(image_id="img-b", error="nope")],
        )
    )
    client = _async_client([_image("img-a", "a.jpg"), _image("img-b", "b.jpg")], bulk_save=bulk)
    report = await _invoke("import_coco", client, "road-signs", _coco())
    assert report.images_saved == 1
    assert any(f.reason == "nope" for f in report.failures)


async def test_async_import_coco_bulk_save_raises_is_reported() -> None:
    bulk = AsyncMock(side_effect=ApiError("boom", status_code=500))
    client = _async_client([_image("img-a", "a.jpg")], bulk_save=bulk)
    report = await _invoke("import_coco", client, "road-signs", _coco())
    assert report.images_saved == 0
    assert any("boom" in f.reason for f in report.failures)


async def test_async_import_coco_not_found_propagates() -> None:
    client = MagicMock()
    client.datasets.get = AsyncMock(side_effect=NotFoundError("no such dataset", status_code=404))
    with pytest.raises(NotFoundError):
        await _invoke("import_coco", client, "ghost", _coco())


# ───────────── YOLO ─────────────


async def test_async_import_yolo_happy_path() -> None:
    client = _async_client([_image("img-a", "a.jpg", width=100, height=100)])
    report = await _invoke(
        "import_yolo", client, "road-signs", {"a.jpg": "0 0.5 0.5 0.2 0.2"}, ["car"]
    )
    assert report.images_saved == 1
    saved_map = client.annotations.bulk_save.await_args[0][0]
    assert saved_map["img-a"][0].name == "car"


async def test_async_import_yolo_missing_dims() -> None:
    client = _async_client([_image("img-a", "a.jpg", width=None, height=None)])
    report = await _invoke(
        "import_yolo", client, "road-signs", {"a.jpg": "0 0.5 0.5 0.2 0.2"}, ["car"]
    )
    assert report.images_saved == 0
    assert any("width/height" in f.reason for f in report.failures)


async def test_async_import_yolo_unmatched() -> None:
    client = _async_client([_image("img-a", "a.jpg")])
    report = await _invoke(
        "import_yolo", client, "road-signs", {"z.jpg": "0 0.5 0.5 0.2 0.2"}, ["car"]
    )
    assert report.unmatched_files == ["z.jpg"]


# ───────────── Pascal VOC ─────────────

_VOC = (
    "<annotation><filename>a.jpg</filename>"
    "<object><name>car</name><bndbox>"
    "<xmin>10</xmin><ymin>20</ymin><xmax>40</xmax><ymax>60</ymax></bndbox></object></annotation>"
)


async def test_async_import_pascal_voc_happy_path() -> None:

    client = _async_client([_image("img-a", "a.jpg")])
    report = await _invoke("import_pascal_voc", client, "road-signs", {"a.jpg": _VOC})
    assert report.images_saved == 1
    saved_map = client.annotations.bulk_save.await_args[0][0]
    assert saved_map["img-a"][0].name == "car"


async def test_async_import_pascal_voc_malformed_is_failure() -> None:

    client = _async_client([_image("img-a", "a.jpg")])
    report = await _invoke(
        "import_pascal_voc", client, "road-signs", {"a.jpg": "<annotation><object>"}
    )
    assert report.images_saved == 0
    assert any(f.image_filename == "a.jpg" for f in report.failures)
