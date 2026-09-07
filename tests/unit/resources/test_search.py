"""Tests for ``pictograph.resources.search.Search``.

Coverage targets:
- ``by_similarity``: param forwarding, typed results, 404 on missing image,
  directory_path override.
- ``by_tag``: validates at least one category required (client-side), AND
  semantics, dataset_name forwarding, 404 on missing dataset.
- Response models surface ``image_auto_tags`` as nested category dict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import NotFoundError
from pictograph.models.search import SimilarImage, TaggedImage
from pictograph.resources.search import Search

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
def search(transport: Transport) -> Search:
    return Search(transport)


def _similar_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "img-uuid-1",
        "filename": "stop-sign-front.jpg",
        "virtual_directory_path": "/",
        "status": "complete",
        "annotation_count": 3,
        "image_auto_tags": {"objects": ["stop_sign", "road"], "scenes": ["outdoor"]},
        "similarity": 0.91,
    }
    base.update(overrides)
    return base


def _tagged_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "img-uuid-2",
        "project_id": "proj-uuid",
        "filename": "outdoor-car.jpg",
        "virtual_directory_path": "/cars",
        "status": "new",
        "annotation_count": 0,
        "image_auto_tags": {"objects": ["car"], "scenes": ["outdoor"]},
    }
    base.update(overrides)
    return base


# ───────────── by_similarity ─────────────


def test_by_similarity_returns_typed_rows(httpx_mock: HTTPXMock, search: Search) -> None:
    httpx_mock.add_response(
        method="GET",
        url=(
            f"{BASE}/api/v1/developer/search/similar?image_id=11aa22bb-33cc-44dd-55ee-66ff77008899&threshold=0.6&limit=50"
        ),
        json={
            "results": [
                _similar_payload(),
                _similar_payload(id="i2", similarity=0.84),
            ],
            "count": 2,
            "threshold": 0.6,
            "query_image_id": "11aa22bb-33cc-44dd-55ee-66ff77008899",
        },
    )
    results = search.by_similarity(
        "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "11aa22bb-33cc-44dd-55ee-66ff77008899"
    )
    assert len(results) == 2
    assert all(isinstance(r, SimilarImage) for r in results)
    assert results[0].similarity == 0.91
    assert results[0].image_auto_tags == {
        "objects": ["stop_sign", "road"],
        "scenes": ["outdoor"],
    }


def test_by_similarity_threshold_and_limit(httpx_mock: HTTPXMock, search: Search) -> None:
    httpx_mock.add_response(
        method="GET",
        url=(
            f"{BASE}/api/v1/developer/search/similar?image_id=11aa22bb-33cc-44dd-55ee-66ff77008899&threshold=0.85&limit=10"
        ),
        json={"results": [], "count": 0, "threshold": 0.85, "query_image_id": "src"},
    )
    search.by_similarity(
        "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "11aa22bb-33cc-44dd-55ee-66ff77008899",
        threshold=0.85,
        limit=10,
    )
    # FORWARDING is the claim, so read the query string back. Registering the URL on the
    # mock does assert it - via pytest-httpx's `assert_all_responses_were_requested` - but
    # only implicitly, and one fixture option would silently disarm every test in this file.
    params = httpx_mock.get_requests()[0].url.params
    assert params["threshold"] == "0.85"
    assert params["limit"] == "10"


def test_by_similarity_directory_override_forwarded(httpx_mock: HTTPXMock, search: Search) -> None:
    httpx_mock.add_response(
        method="GET",
        url=(
            f"{BASE}/api/v1/developer/search/similar"
            "?image_id=11aa22bb-33cc-44dd-55ee-66ff77008899&threshold=0.6&limit=50&directory_path=%2Ftraffic"
        ),
        json={"results": [], "count": 0, "threshold": 0.6, "query_image_id": "src"},
    )
    search.by_similarity(
        "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "11aa22bb-33cc-44dd-55ee-66ff77008899",
        directory_path="/traffic",
    )
    # The override must reach the wire UNENCODED-as-a-path (httpx decodes on read).
    assert httpx_mock.get_requests()[0].url.params["directory_path"] == "/traffic"


def test_by_similarity_404_missing_source(httpx_mock: HTTPXMock, search: Search) -> None:
    httpx_mock.add_response(
        method="GET",
        url=(
            f"{BASE}/api/v1/developer/search/similar?image_id=11aa22bb-33cc-44dd-55ee-66ff77008899&threshold=0.6&limit=50"
        ),
        status_code=404,
        json={"detail": "Source image not found"},
    )
    with pytest.raises(NotFoundError):
        search.by_similarity(
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "11aa22bb-33cc-44dd-55ee-66ff77008899"
        )


# ───────────── by_tag ─────────────


def test_by_tag_requires_at_least_one_category(search: Search) -> None:
    with pytest.raises(ValueError, match="At least one of"):
        search.by_tag()


def test_by_tag_objects_filter_forwarded(httpx_mock: HTTPXMock, search: Search) -> None:
    httpx_mock.add_response(
        method="GET",
        url=(f"{BASE}/api/v1/developer/search/by-tags?limit=50&offset=0&objects=car"),
        json={
            "results": [_tagged_payload()],
            "count": 1,
            "filters": {
                "objects": ["car"],
                "scenes": None,
                "attributes": None,
                "dataset_name": None,
            },
            "pagination": {"limit": 50, "offset": 0},
        },
    )
    results = search.by_tag(objects=["car"])
    assert len(results) == 1
    assert isinstance(results[0], TaggedImage)
    assert results[0].image_auto_tags["objects"] == ["car"]


def test_by_tag_multiple_categories_anded(httpx_mock: HTTPXMock, search: Search) -> None:
    httpx_mock.add_response(
        method="GET",
        url=(
            f"{BASE}/api/v1/developer/search/by-tags"
            "?limit=50&offset=0&objects=car&scenes=outdoor&attributes=sunny"
        ),
        json={
            "results": [_tagged_payload()],
            "count": 1,
            "filters": {
                "objects": ["car"],
                "scenes": ["outdoor"],
                "attributes": ["sunny"],
                "dataset_name": None,
            },
            "pagination": {"limit": 50, "offset": 0},
        },
    )
    search.by_tag(objects=["car"], scenes=["outdoor"], attributes=["sunny"])
    # All three categories must ride together - dropping one silently WIDENS the search.
    params = httpx_mock.get_requests()[0].url.params
    assert params["objects"] == "car"
    assert params["scenes"] == "outdoor"
    assert params["attributes"] == "sunny"


def test_by_tag_dataset_filter_forwarded(httpx_mock: HTTPXMock, search: Search) -> None:
    httpx_mock.add_response(
        method="GET",
        url=(
            f"{BASE}/api/v1/developer/search/by-tags"
            "?limit=50&offset=0&objects=stop_sign&dataset_name=road-signs"
        ),
        json={
            "results": [],
            "count": 0,
            "filters": {
                "objects": ["stop_sign"],
                "scenes": None,
                "attributes": None,
                "dataset_name": "road-signs",
            },
            "pagination": {"limit": 50, "offset": 0},
        },
    )
    search.by_tag(objects=["stop_sign"], dataset_name="road-signs")
    # A dropped dataset filter searches the WHOLE org - the failure mode is silent and wide.
    params = httpx_mock.get_requests()[0].url.params
    assert params["dataset_name"] == "road-signs"
    assert params["objects"] == "stop_sign"


def test_by_tag_pagination_params(httpx_mock: HTTPXMock, search: Search) -> None:
    httpx_mock.add_response(
        method="GET",
        url=(f"{BASE}/api/v1/developer/search/by-tags?limit=10&offset=20&objects=car"),
        json={
            "results": [],
            "count": 0,
            "filters": {
                "objects": ["car"],
                "scenes": None,
                "attributes": None,
                "dataset_name": None,
            },
            "pagination": {"limit": 10, "offset": 20},
        },
    )
    search.by_tag(objects=["car"], limit=10, offset=20)
    params = httpx_mock.get_requests()[0].url.params
    assert params["limit"] == "10"
    assert params["offset"] == "20"


def test_by_tag_404_missing_dataset(httpx_mock: HTTPXMock, search: Search) -> None:
    httpx_mock.add_response(
        method="GET",
        url=(
            f"{BASE}/api/v1/developer/search/by-tags"
            "?limit=50&offset=0&objects=car&dataset_name=missing"
        ),
        status_code=404,
        json={"detail": "Dataset 'missing' not found"},
    )
    with pytest.raises(NotFoundError):
        search.by_tag(objects=["car"], dataset_name="missing")
