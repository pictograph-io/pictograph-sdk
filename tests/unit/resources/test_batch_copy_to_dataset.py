"""``client.batch.copy_to_dataset`` - copy images into another dataset (B533 P1 #6).

The body is exactly the developer route's ``BatchCopyToDatasetRequest`` (``extra="forbid"``
on the server, so a stray key is a 422 there): the source dataset, the ids, the target
dataset and the folder in it. The answer says what was copied, renamed, refused and the
classes added to the target. Sync and async twins share one body builder; both are
exercised against the same responses.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from pictograph import AsyncClient
from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import NotFoundError, ValidationError
from pictograph.models.batch import BatchFailure, CopyToDatasetResult
from pictograph.resources.batch import Batch

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
URL = f"{BASE}/api/v1/developer/batch/images/copy-to-dataset"
A, B, C = (
    "aaaaaaaa-0000-4000-8000-000000000001",
    "aaaaaaaa-0000-4000-8000-000000000002",
    "aaaaaaaa-0000-4000-8000-000000000003",
)

_ANSWER: dict[str, Any] = {
    "success": True,
    "processed": 2,
    "failed": [{"id": C, "reason": "Not an image of dataset 'shelf'"}],
    "affected_directories": ["/val"],
    "renamed": 1,
    "operation": "copied",
    "target_dataset": "curated",
    "classes_added": ["car", "night"],
}


@pytest.fixture
def batch() -> Iterator[Batch]:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    transport = Transport(config, api_key=KEY)
    yield Batch(transport)
    transport.close()


@pytest.fixture
async def aclient() -> AsyncIterator[AsyncClient]:
    c = AsyncClient(api_key=KEY, base_url=BASE, max_retries=0)
    yield c
    await c.aclose()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _sent(httpx_mock: HTTPXMock) -> dict[str, Any]:
    (request,) = httpx_mock.get_requests()
    return json.loads(request.read())  # type: ignore[no-any-return]


def test_sends_exactly_the_route_fields_and_parses_the_answer(
    httpx_mock: HTTPXMock, batch: Batch
) -> None:
    httpx_mock.add_response(method="POST", url=URL, json=_ANSWER)
    result = batch.copy_to_dataset("shelf", (A, B, C), "curated", target_directory_path="/val")
    assert _sent(httpx_mock) == {
        "dataset_name": "shelf",
        "image_ids": [A, B, C],
        "target_dataset_name": "curated",
        "target_directory_path": "/val",
    }
    assert isinstance(result, CopyToDatasetResult)
    assert (result.processed, result.renamed, result.target_dataset, result.classes_added) == (
        2,
        1,
        "curated",
        ["car", "night"],
    )
    assert result.failed == [BatchFailure(id=C, reason="Not an image of dataset 'shelf'")]
    assert result.affected_directories == ["/val"] and result.operation == "copied"


def test_the_root_is_the_default_folder(httpx_mock: HTTPXMock, batch: Batch) -> None:
    httpx_mock.add_response(
        method="POST",
        url=URL,
        json={**_ANSWER, "failed": [], "classes_added": [], "affected_directories": ["/"]},
    )
    result = batch.copy_to_dataset("shelf", [A], "curated")
    assert _sent(httpx_mock)["target_directory_path"] == "/"
    assert result.classes_added == [] and result.failed == []


def test_an_unknown_dataset_is_not_found(httpx_mock: HTTPXMock, batch: Batch) -> None:
    httpx_mock.add_response(
        method="POST", url=URL, status_code=404, json={"detail": "Dataset 'nope' not found"}
    )
    with pytest.raises(NotFoundError, match="'nope' not found"):
        batch.copy_to_dataset("shelf", [A], "nope")


def test_an_invalid_folder_is_a_validation_error(httpx_mock: HTTPXMock, batch: Batch) -> None:
    httpx_mock.add_response(
        method="POST", url=URL, status_code=400, json={"detail": "Invalid target directory path"}
    )
    with pytest.raises(ValidationError, match="Invalid target directory path"):
        batch.copy_to_dataset("shelf", [A], "curated", target_directory_path="../up")


@pytest.mark.anyio
async def test_the_async_twin_sends_the_same_body(
    httpx_mock: HTTPXMock, aclient: AsyncClient
) -> None:
    httpx_mock.add_response(method="POST", url=URL, json=_ANSWER)
    result = await aclient.batch.copy_to_dataset(
        "shelf", [A, B, C], "curated", target_directory_path="/val"
    )
    assert _sent(httpx_mock) == {
        "dataset_name": "shelf",
        "image_ids": [A, B, C],
        "target_dataset_name": "curated",
        "target_directory_path": "/val",
    }
    assert isinstance(result, CopyToDatasetResult) and result.classes_added == ["car", "night"]
