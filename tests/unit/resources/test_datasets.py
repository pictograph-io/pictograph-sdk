"""Tests for ``pictograph.resources.datasets.Datasets``.

Coverage targets:
- ``list`` and ``iter`` against the developer datasets endpoint, including
  pagination across multiple pages and the ``max_total`` cap.
- ``get`` (by name) and ``get_by_id`` (by UUID) - happy path and 404.
- ``download`` - the most complex method: fetches a batch of URLs, then
  downloads images (signed GCS URLs, no auth) and annotations (authenticated
  API endpoint) in parallel via a worker pool.
  - Mode permutations (``full`` / ``images_only`` / ``annotations_only``)
  - Partial-failure reporting (failures collected, never raised)
  - Progress callback semantics (one fire per file, monotonic ``completed``)
  - Empty dataset short-circuits cleanly
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import NotFoundError, ServerError
from pictograph.models.dataset import Dataset
from pictograph.resources.datasets import (
    Datasets,
    DownloadFailure,
    DownloadReport,
)

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"

# ───────────── fixtures ─────────────


@pytest.fixture
def transport() -> Transport:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def datasets(transport: Transport) -> Datasets:
    return Datasets(transport)


def _dataset_payload(**overrides: object) -> dict[str, object]:
    """Minimal valid Dataset payload matching the backend's list/get response."""
    base: dict[str, object] = {
        "id": "ds-uuid-1",
        "name": "road-signs",
        "description": "Road signs dataset",
        "image_count": 100,
        "completed_image_count": 80,
        "total_size": 12345,
        "archived_image_count": 2,
        "organization_id": "org-1",
        "classes": [{"name": "stop_sign", "type": "object", "color": "#FF0000"}],
        "created_at": "2026-01-01T00:00:00Z",
    }
    base.update(overrides)
    return base


# ───────────── list ─────────────


def test_list_returns_typed_datasets(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/datasets/?limit=100&offset=0",
        json={
            "data": [_dataset_payload(), _dataset_payload(id="ds-2", name="cars")],
            "pagination": {"limit": 100, "offset": 0, "total": 2, "has_more": False},
        },
    )
    result = datasets.list()
    assert len(result) == 2
    assert all(isinstance(d, Dataset) for d in result)
    assert {d.name for d in result} == {"road-signs", "cars"}


def test_list_empty_result(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/datasets/?limit=100&offset=0",
        json={"data": [], "pagination": {"limit": 100, "offset": 0, "total": 0, "has_more": False}},
    )
    assert datasets.list() == []


def test_list_passes_limit_param(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/datasets/?limit=50&offset=0",
        json={"data": []},
    )
    datasets.list(limit=50)
    sent = httpx_mock.get_request()
    assert sent is not None
    assert "limit=50" in str(sent.url)


def test_list_invalid_dataset_payload_raises_server_error(
    httpx_mock: HTTPXMock, datasets: Datasets
) -> None:
    # Backend returns malformed item (missing required `name`).
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/datasets/?limit=100&offset=0",
        json={"data": [{"id": "x", "image_count": 1, "created_at": "2026-01-01T00:00:00Z"}]},
    )
    with pytest.raises(ServerError):
        datasets.list()


def test_list_missing_data_key_returns_empty(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    # Defensive: backend regression that drops the key shouldn't raise on the
    # client side - empty result is the safe interpretation.
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/datasets/?limit=100&offset=0",
        json={"pagination": {"limit": 100, "offset": 0, "total": 0, "has_more": False}},
    )
    assert datasets.list() == []


# ───────────── iter ─────────────


def test_iter_paginates_across_pages(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    page1 = [_dataset_payload(id="d1", name="n1"), _dataset_payload(id="d2", name="n2")]
    page2 = [_dataset_payload(id="d3", name="n3"), _dataset_payload(id="d4", name="n4")]
    page3: list[dict[str, object]] = [_dataset_payload(id="d5", name="n5")]
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/?offset=0&limit=2",
        json={"data": page1},
    )
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/?offset=2&limit=2",
        json={"data": page2},
    )
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/?offset=4&limit=2",
        json={"data": page3},
    )
    all_items = list(datasets.iter(page_size=2))
    assert [d.name for d in all_items] == ["n1", "n2", "n3", "n4", "n5"]


def test_iter_max_total_caps_results(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/?offset=0&limit=2",
        json={"data": [_dataset_payload(id="d1", name="n1"), _dataset_payload(id="d2", name="n2")]},
    )
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/?offset=2&limit=2",
        json={"data": [_dataset_payload(id="d3", name="n3"), _dataset_payload(id="d4", name="n4")]},
    )
    items = list(datasets.iter(page_size=2, max_total=3))
    assert [d.name for d in items] == ["n1", "n2", "n3"]


def test_iter_first_short_page_terminates(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/?offset=0&limit=10",
        json={"data": [_dataset_payload(id="x", name="x")]},
    )
    items = list(datasets.iter(page_size=10))
    assert len(items) == 1
    assert len(httpx_mock.get_requests()) == 1


def test_iter_empty_first_page(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/?offset=0&limit=10",
        json={"data": []},
    )
    assert list(datasets.iter(page_size=10)) == []


# ───────────── get / get_by_id ─────────────


def test_get_by_name_happy_path(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/road-signs",
        json={"data": _dataset_payload(organization_id="org-1")},
    )
    ds = datasets.get("road-signs")
    assert ds.name == "road-signs"
    assert ds.organization_id == "org-1"


def test_get_by_name_404(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/missing",
        status_code=404,
        json={"detail": "Dataset 'missing' not found"},
    )
    with pytest.raises(NotFoundError):
        datasets.get("missing")


def _insights_payload() -> dict[str, object]:
    return {
        "total_images": 162,
        "total_annotations": 1992,
        "annotated_images": 162,
        "unannotated_images": 0,
        "avg_annotations_per_image": 12.3,
        "total_bytes": 3394727,
        "status_counts": {"new": 0, "annotate": 0, "review": 0, "complete": 162},
        "class_annotation_counts": {"player": 1756, "ball": 96, "ref": 140},
        "class_image_counts": {"player": 162, "ball": 96, "ref": 140},
        "type_counts": {"bbox": 1992},
        "annotation_density": {
            "0": 0,
            "1-2": 0,
            "3-5": 7,
            "6-10": 47,
            "11-20": 108,
            "21-50": 0,
            "51+": 0,
        },
        "dimensions": {
            "min_width": 398,
            "max_width": 1280,
            "avg_width": 643,
            "min_height": 224,
            "max_height": 720,
            "avg_height": 362,
            "orientation": {"landscape": 162, "portrait": 0, "square": 0},
            "sizes": [{"w": 398, "h": 224, "count": 117}, {"w": 1280, "h": 720, "count": 45}],
            "distinct_size_count": 2,
            "images_with_dimensions": 162,
            "images_missing_dimensions": 0,
        },
        "model_confidence": {
            "flagged": 12,
            "lowest": 0.31,
            "avg_flagged": 0.62,
            "buckets": {"lt50": 3, "b50_70": 4, "b70_90": 5, "b90_100": 0},
        },
    }


def test_insights_happy_path(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/futbol/insights",
        json={"data": _insights_payload()},
    )
    ins = datasets.insights("futbol")
    assert ins.total_images == 162
    assert ins.total_annotations == 1992
    assert ins.avg_annotations_per_image == 12.3
    assert ins.status_counts.complete == 162
    assert ins.class_annotation_counts["player"] == 1756
    assert ins.class_image_counts["ball"] == 96
    assert ins.type_counts["bbox"] == 1992
    assert ins.dimensions.orientation.landscape == 162
    assert ins.dimensions.distinct_size_count == 2
    assert ins.dimensions.sizes[0].w == 398
    # Active learning: the model-uncertainty rollup is exposed.
    assert ins.model_confidence is not None
    assert ins.model_confidence.flagged == 12
    assert ins.model_confidence.lowest == 0.31
    assert ins.model_confidence.avg_flagged == 0.62
    assert ins.model_confidence.buckets.lt50 == 3
    assert ins.model_confidence.buckets.b70_90 == 5


def test_insights_model_confidence_optional(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    # A dataset with no model predictions (or an older backend) omits it → None.
    payload = _insights_payload()
    del payload["model_confidence"]
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/futbol/insights", json={"data": payload}
    )
    assert datasets.insights("futbol").model_confidence is None


def test_insights_url_encodes_name(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/road%20signs/insights",
        json={"data": _insights_payload()},
    )
    datasets.insights("road signs")
    assert httpx_mock.get_request() is not None


def test_insights_404(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/missing/insights",
        status_code=404,
        json={"detail": "Dataset 'missing' not found"},
    )
    with pytest.raises(NotFoundError):
        datasets.insights("missing")


def test_insights_empty_dataset_defaults(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    """An empty dataset (null dims, empty maps) parses to safe defaults."""
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/empty/insights",
        json={
            "data": {
                "total_images": 0,
                "total_annotations": 0,
                "annotated_images": 0,
                "unannotated_images": 0,
                "avg_annotations_per_image": 0,
                "total_bytes": 0,
                "status_counts": {"new": 0, "annotate": 0, "review": 0, "complete": 0},
                "class_annotation_counts": {},
                "class_image_counts": {},
                "type_counts": {},
                "annotation_density": {},
                "dimensions": {
                    "min_width": None,
                    "max_width": None,
                    "avg_width": None,
                    "min_height": None,
                    "max_height": None,
                    "avg_height": None,
                    "orientation": {"landscape": 0, "portrait": 0, "square": 0},
                    "sizes": [],
                    "distinct_size_count": 0,
                    "images_with_dimensions": 0,
                    "images_missing_dimensions": 0,
                },
            }
        },
    )
    ins = datasets.insights("empty")
    assert ins.total_images == 0
    assert ins.dimensions.min_width is None
    assert ins.class_annotation_counts == {}
    assert ins.dimensions.sizes == []


# ───────────── near_duplicates (data curation) ─────────────

_DUP_ID = "dedede00-1111-2222-3333-444455556666"


def _dup_payload() -> dict[str, object]:
    return {
        "groups": [
            {
                "members": [
                    {
                        "id": "a",
                        "filename": "a.jpg",
                        "virtual_directory_path": "/",
                        "status": "new",
                        "annotation_count": 3,
                    },
                    {
                        "id": "b",
                        "filename": "b.jpg",
                        "virtual_directory_path": "/",
                        "status": "new",
                        "annotation_count": 0,
                    },
                    {
                        "id": "c",
                        "filename": "c.jpg",
                        "virtual_directory_path": "/",
                        "status": "new",
                        "annotation_count": 0,
                    },
                ],
                "size": 3,
                "max_similarity": 0.981,
            },
            {
                "members": [
                    {
                        "id": "x",
                        "filename": "x.jpg",
                        "virtual_directory_path": "/",
                        "status": "new",
                        "annotation_count": 0,
                    },
                    {
                        "id": "y",
                        "filename": "y.jpg",
                        "virtual_directory_path": "/",
                        "status": "new",
                        "annotation_count": 0,
                    },
                ],
                "size": 2,
                "max_similarity": 0.94,
            },
        ],
        "group_count": 2,
        "duplicate_image_count": 5,
        "redundant_count": 3,
        "analyzed": 120,
        "total_images": 120,
        "sample_limit": 1000,
        "sample_capped": False,
        "pairs_capped": False,
        "threshold": 0.92,
    }


def test_near_duplicates_happy_path(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/{_DUP_ID}/duplicates",
        json={"data": _dup_payload()},
    )
    dup = datasets.near_duplicates(dataset_id=_DUP_ID)
    assert dup.group_count == 2
    assert dup.redundant_count == 3
    assert dup.groups[0].size == 3
    assert dup.groups[0].max_similarity == 0.981
    assert dup.groups[0].members[0].filename == "a.jpg"
    assert dup.groups[0].members[0].annotation_count == 3
    # a caller keeps the first of each cluster and archives the rest
    redundant = [m.id for g in dup.groups for m in g.members[1:]]
    assert redundant == ["b", "c", "y"]


def test_near_duplicates_passes_query_params(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/{_DUP_ID}/duplicates?threshold=0.96&sample=500",
        json={"data": _dup_payload()},
    )
    datasets.near_duplicates(dataset_id=_DUP_ID, threshold=0.96, sample=500)
    req = httpx_mock.get_request()
    assert req is not None
    assert b"threshold=0.96" in req.url.query
    assert b"sample=500" in req.url.query


def test_near_duplicates_passes_directory_path(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    # Directory scope: the directory rides into the query; omitted when None.
    httpx_mock.add_response(json={"data": _dup_payload()})
    datasets.near_duplicates(dataset_id=_DUP_ID, directory_path="/train")
    req = httpx_mock.get_request()
    assert req is not None
    assert (b"directory_path=%2Ftrain" in req.url.query) or (
        b"directory_path=/train" in req.url.query
    )


def test_near_duplicates_404(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/{_DUP_ID}/duplicates",
        status_code=404,
        json={"detail": "Dataset not found"},
    )
    with pytest.raises(NotFoundError):
        datasets.near_duplicates(dataset_id=_DUP_ID)


def test_get_with_include_images_passes_query_params(
    httpx_mock: HTTPXMock, datasets: Datasets
) -> None:
    httpx_mock.add_response(
        url=(
            f"{BASE}/api/v1/developer/datasets/road-signs"
            "?include_images=true&images_limit=500&images_offset=0"
        ),
        json={"data": _dataset_payload()},
    )
    datasets.get("road-signs", include_images=True, images_limit=500)
    sent = httpx_mock.get_request()
    assert sent is not None
    url = str(sent.url)
    assert "include_images=true" in url
    assert "images_limit=500" in url


def test_get_without_include_images_omits_image_params(
    httpx_mock: HTTPXMock, datasets: Datasets
) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/road-signs",
        json={"data": _dataset_payload()},
    )
    datasets.get("road-signs")
    sent = httpx_mock.get_request()
    assert sent is not None
    assert "include_images" not in str(sent.url)


def test_get_by_uuid_keyword(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/ds-uuid-1",
        json={"data": _dataset_payload()},
    )
    ds = datasets.get(dataset_id="ds-uuid-1")
    assert ds.id == "ds-uuid-1"


def test_get_requires_exactly_one_of_name_or_id(datasets: Datasets) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        datasets.get()
    with pytest.raises(ValueError, match="exactly one"):
        datasets.get("road-signs", dataset_id="ds-uuid-1")


# ───────────── download ─────────────


@pytest.fixture
def small_image_url() -> str:
    return "https://storage.test/img1.signed?token=abc"


@pytest.fixture
def small_annotation_url() -> str:
    # Relative path → the SDK transport prepends BASE and adds X-API-Key.
    return "/api/v1/developer/annotations/img-1/file"


def _download_listing(image_url: str, annotation_url: str | None) -> dict[str, object]:
    item: dict[str, object] = {
        "id": "img-1",
        "filename": "img_001.jpg",
        "file_size": 100,
        "annotation_count": 1 if annotation_url else 0,
    }
    if image_url:
        item["image_url"] = image_url
    if annotation_url:
        item["annotation_url"] = annotation_url
    return {
        "data": {
            "id": "ds-1",
            "items": [item],
            "total_items": 1,
        }
    }


def test_download_full_mode_writes_image_and_annotation_files(
    httpx_mock: HTTPXMock,
    datasets: Datasets,
    tmp_path: Path,
    small_image_url: str,
    small_annotation_url: str,
) -> None:
    httpx_mock.add_response(
        url=(f"{BASE}/api/v1/developer/datasets/road-signs/download?mode=full&limit=10000"),
        json=_download_listing(small_image_url, small_annotation_url),
    )
    httpx_mock.add_response(
        method="GET",
        url=small_image_url,
        content=b"\xff\xd8\xff\xe0FAKEJPEG",  # JPEG magic + filler
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}{small_annotation_url}",
        json={"image": "img_001.jpg", "annotations": []},
    )
    report = datasets.download("road-signs", tmp_path)
    assert report.success
    assert report.images_downloaded == 1
    assert report.annotations_downloaded == 1
    assert report.dataset_id == "ds-1"
    assert (tmp_path / "img_001.jpg").read_bytes() == b"\xff\xd8\xff\xe0FAKEJPEG"
    ann = json.loads((tmp_path / "img_001.jpg.json").read_text())
    assert ann["image"] == "img_001.jpg"


def test_download_images_only_mode_skips_annotations(
    httpx_mock: HTTPXMock,
    datasets: Datasets,
    tmp_path: Path,
    small_image_url: str,
    small_annotation_url: str,
) -> None:
    httpx_mock.add_response(
        url=(f"{BASE}/api/v1/developer/datasets/road-signs/download?mode=images_only&limit=10000"),
        json=_download_listing(small_image_url, small_annotation_url),
    )
    httpx_mock.add_response(method="GET", url=small_image_url, content=b"x")
    report = datasets.download("road-signs", tmp_path, mode="images_only")
    assert report.images_downloaded == 1
    assert report.annotations_downloaded == 0
    assert (tmp_path / "img_001.jpg").exists()
    assert not (tmp_path / "img_001.jpg.json").exists()


def test_download_annotations_only_mode_skips_images(
    httpx_mock: HTTPXMock,
    datasets: Datasets,
    tmp_path: Path,
    small_image_url: str,
    small_annotation_url: str,
) -> None:
    httpx_mock.add_response(
        url=(
            f"{BASE}/api/v1/developer/datasets/road-signs/download"
            "?mode=annotations_only&limit=10000"
        ),
        json=_download_listing(small_image_url, small_annotation_url),
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}{small_annotation_url}",
        json={"image": "img_001.jpg", "annotations": []},
    )
    report = datasets.download("road-signs", tmp_path, mode="annotations_only")
    assert report.images_downloaded == 0
    assert report.annotations_downloaded == 1
    assert not (tmp_path / "img_001.jpg").exists()
    assert (tmp_path / "img_001.jpg.json").exists()


def test_download_creates_output_directory_when_missing(
    httpx_mock: HTTPXMock, datasets: Datasets, tmp_path: Path
) -> None:
    out = tmp_path / "nested" / "fresh"
    httpx_mock.add_response(
        url=(f"{BASE}/api/v1/developer/datasets/road-signs/download?mode=full&limit=10000"),
        json={"data": {"items": [], "id": "ds-1"}},
    )
    report = datasets.download("road-signs", out)
    assert out.exists()
    assert report.success
    assert report.images_downloaded == 0


def test_download_status_filter_propagates(
    httpx_mock: HTTPXMock, datasets: Datasets, tmp_path: Path
) -> None:
    httpx_mock.add_response(
        url=(
            f"{BASE}/api/v1/developer/datasets/road-signs/download"
            "?mode=full&limit=10000&status_filter=complete"
        ),
        json={"data": {"items": [], "id": "ds-1"}},
    )
    datasets.download("road-signs", tmp_path, status_filter="complete")
    sent = [r for r in httpx_mock.get_requests() if "datasets/road-signs" in str(r.url)]
    assert len(sent) == 1
    assert "status_filter=complete" in str(sent[0].url)


def test_download_failed_image_recorded_in_report_not_raised(
    httpx_mock: HTTPXMock,
    datasets: Datasets,
    tmp_path: Path,
    small_image_url: str,
) -> None:
    httpx_mock.add_response(
        url=(f"{BASE}/api/v1/developer/datasets/road-signs/download?mode=images_only&limit=10000"),
        json=_download_listing(small_image_url, None),
    )
    httpx_mock.add_response(
        method="GET",
        url=small_image_url,
        status_code=403,
        content=b"<Error>denied</Error>",
    )
    report = datasets.download("road-signs", tmp_path, mode="images_only")
    assert not report.success
    assert report.images_downloaded == 0
    assert len(report.failures) == 1
    assert report.failures[0].kind == "image"
    assert report.failures[0].filename == "img_001.jpg"


def test_download_progress_callback_fires_once_per_file(
    httpx_mock: HTTPXMock,
    datasets: Datasets,
    tmp_path: Path,
    small_image_url: str,
    small_annotation_url: str,
) -> None:
    httpx_mock.add_response(
        url=(f"{BASE}/api/v1/developer/datasets/road-signs/download?mode=full&limit=10000"),
        json=_download_listing(small_image_url, small_annotation_url),
    )
    httpx_mock.add_response(method="GET", url=small_image_url, content=b"x")
    httpx_mock.add_response(
        method="GET", url=f"{BASE}{small_annotation_url}", json={"annotations": []}
    )

    events: list[tuple[int, int, str | None]] = []

    def cb(completed: int, total: int, fname: str | None) -> None:
        events.append((completed, total, fname))

    datasets.download("road-signs", tmp_path, progress=cb)
    # Two files (1 image + 1 annotation), so two callbacks.
    assert len(events) == 2
    # Total is constant; completed monotonically increases 1, 2.
    completed_seq = [e[0] for e in events]
    totals = {e[1] for e in events}
    assert completed_seq in ([1, 2],)
    assert totals == {2}


def test_download_empty_dataset_short_circuits_without_workers(
    httpx_mock: HTTPXMock, datasets: Datasets, tmp_path: Path
) -> None:
    httpx_mock.add_response(
        url=(f"{BASE}/api/v1/developer/datasets/road-signs/download?mode=full&limit=10000"),
        json={"data": {"items": [], "id": "ds-1"}},
    )
    report = datasets.download("road-signs", tmp_path)
    assert report.success
    assert report.images_downloaded == 0
    assert report.annotations_downloaded == 0


def test_download_initial_listing_404_propagates(
    httpx_mock: HTTPXMock, datasets: Datasets, tmp_path: Path
) -> None:
    httpx_mock.add_response(
        url=(f"{BASE}/api/v1/developer/datasets/missing/download?mode=full&limit=10000"),
        status_code=404,
        json={"detail": "Dataset 'missing' not found"},
    )
    with pytest.raises(NotFoundError):
        datasets.download("missing", tmp_path)


def test_download_partial_failure_reports_per_file(
    httpx_mock: HTTPXMock, datasets: Datasets, tmp_path: Path
) -> None:
    # Two items: one downloads, one 404s.
    httpx_mock.add_response(
        url=(f"{BASE}/api/v1/developer/datasets/road-signs/download?mode=images_only&limit=10000"),
        json={
            "data": {
                "id": "ds-1",
                "items": [
                    {"id": "i1", "filename": "ok.jpg", "image_url": "https://gcs/ok"},
                    {"id": "i2", "filename": "bad.jpg", "image_url": "https://gcs/bad"},
                ],
                "total_items": 2,
            }
        },
    )
    httpx_mock.add_response(method="GET", url="https://gcs/ok", content=b"OK")
    httpx_mock.add_response(method="GET", url="https://gcs/bad", status_code=500)
    report = datasets.download("road-signs", tmp_path, mode="images_only", max_workers=2)
    assert report.images_downloaded == 1
    assert len(report.failures) == 1
    assert report.failures[0].filename == "bad.jpg"
    assert report.total_attempted == 2
    assert not report.success


# ───────────── DownloadReport ─────────────


def test_download_report_success_property() -> None:
    assert DownloadReport(dataset_id="d").success is True
    failed = DownloadReport(
        dataset_id="d",
        failures=[DownloadFailure(filename="x", kind="image", reason="boom")],
    )
    assert failed.success is False


def test_download_report_total_attempted_sums_components() -> None:
    r = DownloadReport(
        dataset_id="d",
        images_downloaded=3,
        annotations_downloaded=2,
        failures=[
            DownloadFailure(filename="a", kind="image", reason="x"),
            DownloadFailure(filename="b", kind="annotation", reason="y"),
        ],
    )
    assert r.total_attempted == 7


def test_download_failure_is_frozen() -> None:
    f = DownloadFailure(filename="x", kind="image", reason="r")
    with pytest.raises(Exception):
        f.filename = "y"  # type: ignore[misc]


# ───────────── cold storage ─────────────


def _storage_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "storage_class": "coldline",
        "storage_state": "idle",
        "cold_since": "2026-06-12T00:30:30+00:00",
        "cold_bytes": 1507079,
        "cold_image_count": 20,
        "storage_job_id": "job-1",
        "restore_estimate": {
            "operation": "dataset_restore",
            "cold_bytes": 1507079,
            "cold_image_count": 20,
            "days_in_cold": 0.0,
            "min_storage_days": 90.0,
            "monthly_savings_micro_usd": 29,
            "retrieval_micro_usd": 36,
            "early_delete_micro_usd": 22,
            "operations_micro_usd": 125,
            "total_micro_usd": 183,
        },
    }
    base.update(overrides)
    return base


def test_storage_status_cold_with_estimate(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/ds-uuid-1/storage",
        json={"data": _storage_payload()},
    )
    status = datasets.storage_status(dataset_id="ds-uuid-1")
    assert status.storage_class == "coldline"
    assert status.restore_estimate is not None
    # Components sum exactly to the total - the UI renders the breakdown.
    est = status.restore_estimate
    assert (
        est.retrieval_micro_usd + est.early_delete_micro_usd + est.operations_micro_usd
        == est.total_micro_usd
    )


def test_storage_status_warm_has_no_estimate(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/ds-uuid-1/storage",
        json={"data": {"storage_class": "standard", "storage_state": "idle"}},
    )
    status = datasets.storage_status(dataset_id="ds-uuid-1")
    assert status.storage_class == "standard"
    assert status.restore_estimate is None


def test_freeze_returns_job(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/datasets/ds-uuid-1/storage/freeze",
        json={"data": {"job_id": "job-9", "storage_state": "freezing"}},
    )
    job = datasets.freeze(dataset_id="ds-uuid-1")
    assert job.job_id == "job-9"
    assert job.storage_state == "freezing"
    assert job.quoted_micro_usd is None


def test_restore_carries_quote(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    # The restore fee is billed on the transition's success, so the ack
    # carries the QUOTED amount that will be charged, not a completed charge.
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/datasets/ds-uuid-1/storage/restore",
        json={
            "data": {
                "job_id": "job-10",
                "storage_state": "restoring",
                "quoted_micro_usd": 183,
            }
        },
    )
    job = datasets.restore(dataset_id="ds-uuid-1")
    assert job.quoted_micro_usd == 183


def test_wait_for_storage_polls_until_idle(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    url = f"{BASE}/api/v1/developer/datasets/ds-uuid-1/storage"
    httpx_mock.add_response(
        url=url, json={"data": _storage_payload(storage_state="freezing", restore_estimate=None)}
    )
    httpx_mock.add_response(url=url, json={"data": _storage_payload(restore_estimate=None)})
    status = datasets.wait_for_storage(dataset_id="ds-uuid-1", timeout=10.0, poll_interval=0.01)
    assert status.storage_state == "idle"


def test_wait_for_storage_times_out(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    from pictograph.exceptions import PollTimeoutError

    url = f"{BASE}/api/v1/developer/datasets/ds-uuid-1/storage"
    httpx_mock.add_response(
        url=url,
        json={"data": _storage_payload(storage_state="restoring", restore_estimate=None)},
        is_reusable=True,
    )
    with pytest.raises(PollTimeoutError):
        datasets.wait_for_storage(dataset_id="ds-uuid-1", timeout=0.05, poll_interval=0.01)


def test_wait_for_storage_rejects_nonpositive_poll_interval(datasets: Datasets) -> None:
    # Parity with every sibling wait_for_* - guard a busy-spin (poll_interval<=0
    # would hammer storage_status with no delay) BEFORE any request is made.
    with pytest.raises(ValueError, match="poll_interval must be > 0"):
        datasets.wait_for_storage(dataset_id="ds-uuid-1", poll_interval=0)
    with pytest.raises(ValueError, match="poll_interval must be > 0"):
        datasets.wait_for_storage(dataset_id="ds-uuid-1", poll_interval=-1.0)


def test_wait_for_storage_rejects_nonpositive_timeout(datasets: Datasets) -> None:
    with pytest.raises(ValueError, match="timeout must be > 0"):
        datasets.wait_for_storage(dataset_id="ds-uuid-1", timeout=0)
    with pytest.raises(ValueError, match="timeout must be > 0"):
        datasets.wait_for_storage(dataset_id="ds-uuid-1", timeout=-5.0)


def test_wait_for_storage_uses_injected_sleep(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    # The `sleep` hook (present on every sibling) lets a poll loop be unit-tested
    # without real wall-clock time and proves it's called between polls.
    url = f"{BASE}/api/v1/developer/datasets/ds-uuid-1/storage"
    httpx_mock.add_response(
        url=url, json={"data": _storage_payload(storage_state="freezing", restore_estimate=None)}
    )
    httpx_mock.add_response(url=url, json={"data": _storage_payload(restore_estimate=None)})
    calls: list[float] = []
    status = datasets.wait_for_storage(
        dataset_id="ds-uuid-1", timeout=10.0, poll_interval=2.5, sleep=calls.append
    )
    assert status.storage_state == "idle"
    assert calls == [2.5]  # slept once, between the two polls, the injected fn


# ───────────── download helpers: atomic .part writes ─────────────


class _FakeResponse:
    def __init__(self, chunks: list[bytes], *, raise_after: int | None = None) -> None:
        self.status_code = 200
        self.request = None
        self._chunks = chunks
        self._raise_after = raise_after

    def read(self) -> bytes:
        return b""

    def iter_bytes(self, **_kwargs: object):  # chunk_size passed by keyword, ignored
        for i, chunk in enumerate(self._chunks):
            if self._raise_after is not None and i == self._raise_after:
                raise RuntimeError("network blip mid-stream")
            yield chunk


class _FakeStreamCtx:
    def __init__(self, response: _FakeResponse) -> None:
        self._r = response

    def __enter__(self) -> _FakeResponse:
        return self._r

    def __exit__(self, *exc: object) -> bool:
        return False


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def stream(self, *_args: object, **_kwargs: object) -> _FakeStreamCtx:
        return _FakeStreamCtx(self._response)


def test_stream_to_file_success_leaves_no_part(tmp_path: Path) -> None:
    from pictograph.resources.datasets import _stream_to_file

    dest = tmp_path / "img.jpg"
    _stream_to_file(_FakeClient(_FakeResponse([b"abc", b"def"])), "http://x", dest)  # type: ignore[arg-type]
    assert dest.read_bytes() == b"abcdef"
    assert not dest.with_name(dest.name + ".part").exists()


def test_stream_to_file_midstream_error_leaves_no_partial(tmp_path: Path) -> None:
    from pictograph.resources.datasets import _stream_to_file

    dest = tmp_path / "img.jpg"
    client = _FakeClient(_FakeResponse([b"abc", b"def"], raise_after=1))
    with pytest.raises(RuntimeError, match="network blip"):
        _stream_to_file(client, "http://x", dest)  # type: ignore[arg-type]
    # Regression: no truncated file at dest, and the .part is cleaned up.
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_fetch_annotation_to_file_success_leaves_no_part(tmp_path: Path) -> None:
    from pictograph.resources.datasets import _fetch_annotation_to_file

    class _T:
        def request(self, *_a: object, **_k: object) -> dict[str, object]:
            return {"annotations": [{"name": "stop"}]}

    dest = tmp_path / "ann.json"
    _fetch_annotation_to_file(_T(), "/url", dest)
    assert json.loads(dest.read_text())["annotations"][0]["name"] == "stop"
    assert not dest.with_name(dest.name + ".part").exists()


def test_fetch_annotation_to_file_serialize_error_leaves_no_partial(tmp_path: Path) -> None:
    from pictograph.resources.datasets import _fetch_annotation_to_file

    class _T:
        def request(self, *_a: object, **_k: object) -> dict[str, object]:
            return {"bad": object()}  # not JSON-serializable -> json.dump raises mid-write

    dest = tmp_path / "ann.json"
    with pytest.raises(TypeError):
        _fetch_annotation_to_file(_T(), "/url", dest)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_uses_http1_to_stay_thread_safe(
    httpx_mock: HTTPXMock,
    datasets: Datasets,
    tmp_path: Path,
    small_image_url: str,
    small_annotation_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GCS download client is shared across a ThreadPoolExecutor, so it MUST
    be HTTP/1.1 - httpx+h2 on a shared client races ('dictionary changed size
    during iteration'). Regression for the http2=True bug."""
    import httpx as _httpx

    import pictograph.resources.datasets as ds_mod

    captured: dict[str, object] = {}
    real_client = _httpx.Client

    def _spy(*args: object, **kwargs: object) -> object:
        if "http2" in kwargs:  # only the GCS client passes http2 explicitly
            captured["http2"] = kwargs["http2"]
        return real_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ds_mod.httpx, "Client", _spy)
    httpx_mock.add_response(
        url=(f"{BASE}/api/v1/developer/datasets/road-signs/download?mode=images_only&limit=10000"),
        json=_download_listing(small_image_url, small_annotation_url),
    )
    httpx_mock.add_response(method="GET", url=small_image_url, content=b"x")
    datasets.download("road-signs", tmp_path, mode="images_only")
    assert captured.get("http2") is False


def test_download_warns_and_flags_truncated_at_the_image_cap(
    httpx_mock: HTTPXMock,
    datasets: Datasets,
    tmp_path: Path,
) -> None:
    """A listing at the backend's 10000-image cap means the dataset is (very
    likely) larger; download flags report.truncated + warns, instead of silently
    returning a partial dataset with success=True."""
    from pictograph.resources.datasets import _DEFAULT_DOWNLOAD_LIMIT

    # url-less items → the cap is hit but zero tasks are built (no GCS mocks
    # needed); the truncation signal fires before the no-tasks short-circuit.
    items = [
        {"id": f"i-{n}", "filename": f"f{n}.jpg", "file_size": 1, "annotation_count": 0}
        for n in range(_DEFAULT_DOWNLOAD_LIMIT)
    ]
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/big/download?mode=full&limit=10000",
        json={
            "data": {
                "id": "ds-big",
                "items": items,
                "total_items": len(items),
            }
        },
    )
    with pytest.warns(UserWarning, match="truncated"):
        report = datasets.download("big", tmp_path)
    assert report.truncated is True
    assert report.images_downloaded == 0


# ───────────── merged CRUD (the former Projects surface) ─────────────


def test_create_posts_body_and_parses_dataset(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/datasets/",
        status_code=201,
        json={"data": _dataset_payload()},
    )
    ds = datasets.create(
        "road-signs",
        description="Road signs dataset",
        annotation_types=["bbox", "polygon"],
        classes=[{"name": "stop_sign", "type": "bbox", "color": "#FF0000"}],
    )
    assert isinstance(ds, Dataset)
    req = httpx_mock.get_request()
    assert req is not None
    body = json.loads(req.content)
    assert body == {
        "name": "road-signs",
        "description": "Road signs dataset",
        "annotation_types": ["bbox", "polygon"],
        "classes": [{"name": "stop_sign", "type": "bbox", "color": "#FF0000"}],
    }


def test_create_accepts_model_classes(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    from pictograph.models.dataset import DatasetClass

    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/datasets/",
        status_code=201,
        json={"data": _dataset_payload()},
    )
    datasets.create("x", classes=[DatasetClass(name="car", type="bbox")])
    req = httpx_mock.get_request()
    assert req is not None
    body = json.loads(req.content)
    # model_dump(exclude_none=True) - no null color/attributes on the wire.
    assert body["classes"] == [{"name": "car", "type": "bbox"}]


def test_update_renames_via_new_name(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE}/api/v1/developer/datasets/old-name",
        json={"data": _dataset_payload(name="new-name")},
    )
    ds = datasets.update("old-name", new_name="new-name")
    assert ds.name == "new-name"
    req = httpx_mock.get_request()
    assert req is not None
    assert json.loads(req.content) == {"new_name": "new-name"}


def test_update_by_uuid_keyword(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE}/api/v1/developer/datasets/ds-uuid-1",
        json={"data": _dataset_payload()},
    )
    datasets.update(dataset_id="ds-uuid-1", description="d")
    req = httpx_mock.get_request()
    assert req is not None
    assert json.loads(req.content) == {"description": "d"}


def test_update_with_no_fields_raises_before_any_request(datasets: Datasets) -> None:
    with pytest.raises(ValueError, match="Nothing to update"):
        datasets.update("x")


def test_delete_returns_summary(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/datasets/road-signs",
        json={
            "data": {
                "id": "ds-uuid-1",
                "name": "road-signs",
                "deleted": True,
                "images_deleted": 100,
                "directories_deleted": 3,
                "gcs_blobs_deleted": 98,
                "gcs_blobs_retained_for_forks": 2,
            }
        },
    )
    summary = datasets.delete("road-signs")
    assert summary["deleted"] is True
    assert summary["images_deleted"] == 100
    assert summary["gcs_blobs_retained_for_forks"] == 2


def test_archive_and_unarchive_return_dataset(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/datasets/road-signs/archive",
        json={"data": _dataset_payload(is_archived=True)},
    )
    archived = datasets.archive("road-signs")
    assert archived.is_archived is True
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/datasets/road-signs/unarchive",
        json={"data": _dataset_payload(is_archived=False)},
    )
    assert datasets.unarchive("road-signs").is_archived is False


def test_list_archived_flag_rides_query(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/datasets/?limit=100&offset=0&archived=true",
        json={"data": []},
    )
    assert datasets.list(archived=True) == []


def test_iter_stops_on_server_has_more_false(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    # A FULL page whose pagination says has_more=False must terminate without
    # fetching the (empty) next page - the envelope saves that round-trip.
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/?offset=0&limit=2",
        json={
            "data": [_dataset_payload(id="d1", name="n1"), _dataset_payload(id="d2", name="n2")],
            "pagination": {"limit": 2, "offset": 0, "total": 2, "has_more": False},
        },
    )
    items = list(datasets.iter(page_size=2))
    assert [d.name for d in items] == ["n1", "n2"]
    assert len(httpx_mock.get_requests()) == 1


# ── path traversal via a server-supplied filename ────────────────────────────
# `filename` in the download listing is DATA from the API. Joined raw, an
# absolute value discards output_dir entirely and "../" walks out of it, so a
# malicious or compromised server could steer a write anywhere the process can
# reach. Reduced to a single component by _path_safety.safe_download_name.


@pytest.mark.parametrize(
    ("hostile", "written_as"),
    [
        ("../escaped.jpg", "escaped.jpg"),
        ("../../deep_escape.jpg", "deep_escape.jpg"),
        ("/abs_escape.jpg", "abs_escape.jpg"),
        ("..\\..\\win_escape.jpg", "win_escape.jpg"),
        # A legitimate name with spaces and parentheses must survive untouched -
        # this is why the guard strips directories rather than rewriting chars.
        ("my photo (1).jpg", "my photo (1).jpg"),
    ],
)
def test_download_filename_cannot_escape_output_dir(
    httpx_mock: HTTPXMock,
    datasets: Datasets,
    tmp_path: Path,
    small_image_url: str,
    hostile: str,
    written_as: str,
) -> None:
    out = tmp_path / "out"
    listing = _download_listing(small_image_url, None)
    listing["data"]["items"][0]["filename"] = hostile  # type: ignore[index]
    httpx_mock.add_response(
        url=(f"{BASE}/api/v1/developer/datasets/road-signs/download?mode=images_only&limit=10000"),
        json=listing,
    )
    httpx_mock.add_response(method="GET", url=small_image_url, content=b"\xff\xd8\xff\xe0PAYLOAD")

    report = datasets.download("road-signs", out, mode="images_only")

    assert report.success, report.failures
    landed = out / written_as
    assert landed.is_file(), f"expected the write inside {out}, got {list(out.iterdir())}"
    assert landed.read_bytes() == b"\xff\xd8\xff\xe0PAYLOAD"
    # Nothing may exist outside the output directory.
    assert not (tmp_path / written_as).exists()
    assert not (tmp_path.parent / written_as).exists()
    for produced in out.rglob("*"):
        assert out.resolve() in produced.resolve().parents
