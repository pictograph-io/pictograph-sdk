"""``AsyncAutoAnnotate.dataset`` - the async twin of the whole-dataset draft (B603, SDK 1.69.176).

Driven over the wire with ``AsyncClient``: the images are read page by page to the end, the
already-annotated ones are held back, the batch is submitted and awaited, and text mode saves
every class's drafts in one save per image - the same plan the sync twin runs (both share
``_normalize_classes`` / ``_plan_dataset_run``).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import httpx
import pytest

from pictograph import AsyncClient

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

pytestmark = pytest.mark.anyio

BASE = "https://api.test.local"
DEV = f"{BASE}/api/v1/developer"
_NOW = "2026-04-01T00:00:00Z"


def _images(rows: list[dict[str, object]], httpx_mock: HTTPXMock) -> None:
    def page(request: httpx.Request) -> httpx.Response:
        # One page: the dataset's rows, and no next cursor.
        assert "cursor" not in request.url.params
        return httpx.Response(200, json={"data": rows, "pagination": {"next_cursor": None}})

    httpx_mock.add_callback(
        page, url=re.compile(rf"{re.escape(DEV)}/images/\?.*"), is_reusable=True
    )


def _row(i: str, count: int = 0) -> dict[str, object]:
    return {"id": i, "filename": f"{i}.jpg", "annotation_count": count, "created_at": _NOW}


async def test_batch_mode_submits_the_unannotated_images_and_awaits_the_job(
    httpx_mock: HTTPXMock,
) -> None:
    _images([_row("a"), _row("b"), _row("done", count=2)], httpx_mock)
    httpx_mock.add_response(
        method="POST", url=f"{DEV}/auto-annotate/batch", json={"job_id": "j1", "status": "pending"}
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/auto-annotate/batch/j1",
        json={
            "job_id": "j1",
            "status": "completed",
            "progress": 100,
            "total_images": 2,
            "processed_images": 2,
            "total_annotations_added": 5,
        },
    )
    async with AsyncClient(api_key="pk_live_test", base_url=BASE, max_retries=0) as client:
        report = await client.auto_annotate.dataset(
            "road-signs", classes=[("stop sign", "bbox")], poll_interval=0.01
        )
    sent = json.loads(next(r for r in httpx_mock.get_requests() if r.method == "POST").content)
    assert sent["images"] == ["a", "b"]
    assert report.images_attempted == 2 and report.images_skipped == 1
    assert report.job_id == "j1" and report.job_status == "completed"
    assert report.images_processed == 2 and report.annotations_added == 5 and report.success


async def test_a_failed_batch_is_a_report_failure_not_a_raise(httpx_mock: HTTPXMock) -> None:
    _images([_row("a")], httpx_mock)
    httpx_mock.add_response(
        method="POST", url=f"{DEV}/auto-annotate/batch", status_code=500, json={"detail": "boom"}
    )
    async with AsyncClient(api_key="pk_live_test", base_url=BASE, max_retries=0) as client:
        report = await client.auto_annotate.dataset("road-signs", classes=[("car", "bbox")])
    assert not report.success
    assert report.failures[0].image_filename == "<batch>"


async def test_text_mode_saves_every_class_once_per_image(httpx_mock: HTTPXMock) -> None:
    _images([_row("a")], httpx_mock)
    box = {"name": "car", "type": "bbox", "bounding_box": {"x": 1, "y": 1, "w": 2, "h": 2}}
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/auto-annotate/sam3/text",
        json={"status": "success", "annotations": [box]},
        is_reusable=True,
    )
    httpx_mock.add_response(
        method="POST",
        url=re.compile(rf"{re.escape(DEV)}/annotations/.*"),
        json={
            "success": True,
            "image_id": "a",
            "previous_count": 0,
            "new_count": 2,
            "status": "annotate",
        },
    )
    async with AsyncClient(api_key="pk_live_test", base_url=BASE, max_retries=0) as client:
        report = await client.auto_annotate.dataset(
            "road-signs", classes=[("car", "bbox"), ("bus", "bbox")], mode="text"
        )
    saves = [r for r in httpx_mock.get_requests() if "/annotations/" in str(r.url)]
    assert len(saves) == 1, "one save per image, carrying every class"
    assert len(json.loads(saves[0].content)["annotations"]) == 2
    assert report.images_processed == 1 and report.annotations_added == 2


async def test_nothing_to_draft_returns_at_once(httpx_mock: HTTPXMock) -> None:
    _images([_row("done", count=1)], httpx_mock)
    async with AsyncClient(api_key="pk_live_test", base_url=BASE, max_retries=0) as client:
        report = await client.auto_annotate.dataset("road-signs", classes=[("car", "bbox")])
    assert report.images_attempted == 0 and report.images_skipped == 1
    assert not [r for r in httpx_mock.get_requests() if r.method == "POST"]


async def test_model_with_text_mode_is_refused() -> None:
    async with AsyncClient(api_key="pk_live_test", base_url=BASE, max_retries=0) as client:
        with pytest.raises(ValueError, match="batch"):
            await client.auto_annotate.dataset(
                "road-signs", classes=[("car", "bbox")], mode="text", model="m"
            )
