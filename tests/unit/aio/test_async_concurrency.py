"""Regression guard: the AsyncClient genuinely issues requests concurrently.

The whole point of :class:`pictograph.AsyncClient` (and the async pipelines built
on it) is that ``asyncio.gather`` over its methods runs the underlying HTTP calls
concurrently on one connection pool, rather than serially. This proves that
property **deterministically** - no wall-clock timing assertion (flaky in CI).
Instead a mock endpoint tracks the number of in-flight requests: if the client
serialized calls, the peak would be 1; concurrency drives it well above 1.

This is the gated companion to the operator-run ``benchmarks/async_bench.py``
(which measures the actual wall-clock speed-up against a live endpoint).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
import pytest

from pictograph import AsyncClient
from pictograph.aio.resources.annotations import AsyncAnnotations
from tests.unit.resources._orchestration import build, sibling_resources

from .conftest import API_KEY, BASE_URL

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

pytestmark = pytest.mark.anyio


class _ConcurrencyProbe:
    """Counts concurrent in-flight requests through a mock endpoint."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak = 0

    async def respond(self, request: httpx.Request) -> httpx.Response:  # noqa: ARG002
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        # Yield control so other queued requests start before this one finishes;
        # a serial client would never let a second request begin here.
        await asyncio.sleep(0.02)
        self.in_flight -= 1
        return httpx.Response(200, json={"datasets": []})


async def test_async_client_gather_is_concurrent(httpx_mock: HTTPXMock) -> None:
    probe = _ConcurrencyProbe()
    httpx_mock.add_callback(probe.respond, is_reusable=True)
    async with AsyncClient(api_key=API_KEY, base_url=BASE_URL, max_retries=0) as client:
        results = await asyncio.gather(*(client.datasets.list() for _ in range(8)))
    assert len(results) == 8
    # If the client serialized the 8 requests, peak would be 1. Concurrency drives
    # it much higher (capped only by the connection pool). >= 5 is a wide margin.
    assert probe.peak >= 5, f"expected concurrent requests, peak in-flight was {probe.peak}"


async def test_async_import_bulk_saves_concurrently() -> None:
    """The async import pipeline fans its chunked bulk_saves out concurrently."""
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock, MagicMock

    from pictograph.models.dataset import Dataset
    from pictograph.models.image import Image
    from pictograph.resources.annotations import BulkSaveResult, SaveResult

    probe = _ConcurrencyProbe()

    async def bulk_save(chunk: dict[str, object]) -> BulkSaveResult:
        probe.in_flight += 1
        probe.peak = max(probe.peak, probe.in_flight)
        await asyncio.sleep(0.02)
        probe.in_flight -= 1
        return BulkSaveResult(
            saved=[
                SaveResult(image_id=i, previous_count=0, new_count=1, status="in_progress")
                for i in chunk
            ],
            failed=[],
        )

    images = [
        Image(
            id=f"img-{i}",
            filename=f"{i}.jpg",
            image_url="https://cdn/x",
            created_at=datetime.now(timezone.utc),
        )
        for i in range(6)
    ]

    class _AsyncIter:
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            for img in images:
                yield img

    client = MagicMock()
    client.datasets.get = AsyncMock(
        return_value=Dataset(
            id="p1",
            name="d",
            organization_id="o",
            classes=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    )
    client.datasets.update = AsyncMock()
    client.images.iter = MagicMock(return_value=_AsyncIter())
    client.annotations.bulk_save = AsyncMock(side_effect=bulk_save)

    coco = {
        "images": [{"id": i, "file_name": f"{i}.jpg"} for i in range(6)],
        "categories": [{"id": 1, "name": "car"}],
        "annotations": [{"image_id": i, "category_id": 1, "bbox": [1, 1, 2, 2]} for i in range(6)],
    }
    # 6 images / chunk 2 => 3 chunks; they must run concurrently, not serially.
    with sibling_resources(client, is_async=True):
        annotations = build(AsyncAnnotations, client, own="annotations", delegate=["bulk_save"])
        await annotations.import_coco("d", coco, save_chunk=2, create_missing_classes=False)
    assert probe.peak >= 2, f"expected concurrent bulk_saves, peak was {probe.peak}"
