"""``AsyncTraining.quote`` - the async twin of ``Training.quote`` (B533 #10)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from pictograph._http.async_transport import AsyncTransport
from pictograph.aio.resources.training import AsyncTraining
from tests.unit.resources.test_training import _quote_payload

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

pytestmark = pytest.mark.anyio

BASE_URL = "https://api.test.local"


async def test_async_quote_posts_the_plan_and_parses_the_quote(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/api/v1/developer/training/quote",
        json={"data": _quote_payload()},
    )
    quote = await AsyncTraining(transport).quote(
        "road-signs", "v1", pipeline_type="rfdetr_detection", gpu_type="auto"
    )
    body = json.loads(httpx_mock.get_requests()[0].content)
    assert body == {
        "dataset_name": "road-signs",
        "export_name": "v1",
        "pipeline_type": "rfdetr_detection",
        "gpu_type": "auto",
        "config": {},
    }
    assert quote.export_name == "v1" and quote.deleted_images == 2
    assert quote.recommended_gpu_type == "a10g"
