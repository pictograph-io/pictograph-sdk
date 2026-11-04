"""``AsyncExplore`` and ``AsyncDatasets.import_public`` - the async twins (B533 #8)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from pictograph._http.async_transport import AsyncTransport
from pictograph.aio.resources.datasets import AsyncDatasets
from pictograph.aio.resources.explore import AsyncExplore
from pictograph.exceptions import ApiError
from tests.unit.resources.test_explore import IMPORT_ID, dataset_row, import_row, model_row, page

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

pytestmark = pytest.mark.anyio

PUBLIC = "https://api.test.local/api/v1/developer/public"


async def test_datasets_dataset_and_models(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    explore = AsyncExplore(transport)
    httpx_mock.add_response(json=page([dataset_row()], total=1))
    httpx_mock.add_response(json={"data": dataset_row(readme="# card")})
    httpx_mock.add_response(json=page([model_row()], total=1))
    [ds] = await explore.datasets(sort="popular")
    detail = await explore.dataset("pictograph-research/coco-5k")
    [model] = await explore.models(model_type="instance_segmentation")
    urls = [str(r.url).split("?")[0] for r in httpx_mock.get_requests()]
    assert urls == [
        f"{PUBLIC}/datasets",
        f"{PUBLIC}/datasets/pictograph-research/coco-5k",
        f"{PUBLIC}/models",
    ]
    assert ds.ref == detail.ref == "pictograph-research/coco-5k" and detail.readme == "# card"
    assert model.ref == "pictograph-research/pictograph-sceneparse150-m"


async def test_iter_datasets(httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
    httpx_mock.add_response(json=page([dataset_row(id="a")], total=1))
    rows = [r async for r in AsyncExplore(transport).iter_datasets()]
    assert [r.id for r in rows] == ["a"]


async def test_import_waits_and_raises_like_the_sync_twin(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    datasets = AsyncDatasets(transport)
    httpx_mock.add_response(status_code=202, json={"data": import_row()})
    httpx_mock.add_response(json={"data": import_row(status="completed", image_count=5000)})
    job = await datasets.import_public(
        "pictograph-research/coco-5k", name="mine", poll_interval=0.01
    )
    assert job.status == "completed"
    assert json.loads(httpx_mock.get_requests()[0].content) == {"name": "mine"}

    httpx_mock.add_response(json={"data": import_row(status="failed", error="boom")})

    async def _no_sleep(_s: float) -> None:
        return None

    with pytest.raises(ApiError) as err:
        await datasets.wait_for_public_import(IMPORT_ID, sleep=_no_sleep)
    assert err.value.code == "job_failed"
