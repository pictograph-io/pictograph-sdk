"""``client.explore`` (the public catalog) and ``client.datasets.import_public`` (B533 #8).

Driven through the real transport against canned responses shaped as the developer API
answers (``routes/developer/public.py``; the backend's contract replay holds the real
routes to these models).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import ApiError, NotFoundError, PaymentRequiredError, PollTimeoutError
from pictograph.models.explore import PublicDataset, PublicModel
from pictograph.resources.datasets import Datasets
from pictograph.resources.explore import Explore, split_ref

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
PUBLIC = f"{BASE}/api/v1/developer/public"
IMPORT_ID = "f1f1f1f1-0000-4000-8000-000000000001"


@pytest.fixture
def transport() -> Transport:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def explore(transport: Transport) -> Explore:
    return Explore(transport)


@pytest.fixture
def datasets(transport: Transport) -> Datasets:
    return Datasets(transport)


ORG = {
    "id": "org-b",
    "name": "Pictograph Research",
    "slug": "pictograph-research",
    "avatar_url": None,
    "orb_palette": None,
    "is_public": True,
}


def dataset_row(**over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "ds-1",
        "name": "COCO-5k",
        "slug": "coco-5k",
        "description": "5,000 images",
        "organization_id": "org-b",
        "organization": ORG,
        "image_count": 5000,
        "annotation_count": 36335,
        "annotation_types": ["polygon", "bbox"],
        "classes": [{"name": "person", "type": "polygon", "color": "#e6194b"}],
        "class_count": 80,
        "non_tag_class_count": 80,
        "top_auto_tags": ["street"],
        "license_id": "cc-by-4.0",
        "star_count": 3,
        "comment_count": 0,
        "is_fork": False,
        "hf_repo": None,
        "preview_image_ids": ["img-1"],
        "starred": False,
        "published_at": "2026-07-03T14:27:57+00:00",
        "created_at": "2026-07-01T00:00:00Z",
        "updated_at": "2026-07-01T00:00:00Z",
    }
    row.update(over)
    return row


def model_row(**over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "m-1",
        "organization_id": "org-b",
        "organization": ORG,
        "name": "Pictograph-SceneParse150-M",
        "slug": "pictograph-sceneparse150-m",
        "model_type": "instance_segmentation",
        "architecture": "RF-DETR Seg Medium",
        "visibility": "public",
        "status": "ready",
        "version": "1.0.0",
        "precision": "fp32",
        "metrics": {"mAP": 0.28},
        "class_mapping": {"classes": ["wall", "floor"]},
        "training_config": {"epochs": 40},
        "license_id": "bsd-3-clause",
        "star_count": 2,
        "weights_size_bytes": 1234,
        "is_fork": False,
        "hf_repo": None,
        "created_at": "2026-10-01T00:00:00Z",
        "updated_at": "2026-10-01T00:00:00Z",
    }
    row.update(over)
    return row


def page(rows: list[dict[str, Any]], *, total: int, offset: int = 0) -> dict[str, Any]:
    return {
        "data": rows,
        "pagination": {
            "limit": len(rows),
            "offset": offset,
            "total": total,
            "has_more": offset + len(rows) < total,
        },
    }


def _query(request: Any) -> dict[str, list[str]]:
    return parse_qs(urlsplit(str(request.url)).query)


# ───────────── addressing ─────────────


@pytest.mark.parametrize(
    ("ref", "organization", "expected"),
    [
        ("pictograph-research/coco-5k", None, ("pictograph-research", "coco-5k")),
        (" acme / Parking Lots ", None, ("acme", "Parking Lots")),
        ("coco-5k", "pictograph-research", ("pictograph-research", "coco-5k")),
    ],
)
def test_split_ref(ref: str, organization: str | None, expected: tuple[str, str]) -> None:
    assert split_ref(ref, organization) == expected


@pytest.mark.parametrize("ref", ["coco-5k", "/coco-5k", "acme/", "  /  "])
def test_split_ref_refuses_what_names_no_organization(ref: str) -> None:
    with pytest.raises(ValueError, match="organization"):
        split_ref(ref)


# ───────────── datasets ─────────────


def test_datasets_sends_the_filters_and_parses_the_rows(
    httpx_mock: HTTPXMock, explore: Explore
) -> None:
    httpx_mock.add_response(json=page([dataset_row()], total=1))
    rows = explore.datasets(search="coco", class_name="person", sort="popular", limit=10)
    request = httpx_mock.get_requests()[0]
    assert str(request.url).startswith(f"{PUBLIC}/datasets?")
    assert _query(request) == {
        "search": ["coco"],
        "class": ["person"],
        "sort": ["popular"],
        "offset": ["0"],
        "limit": ["10"],
    }
    [row] = rows
    assert isinstance(row, PublicDataset)
    assert row.ref == "pictograph-research/coco-5k"
    assert row.license_id == "cc-by-4.0" and row.classes[0].name == "person"


def test_iter_datasets_pages_until_the_total(httpx_mock: HTTPXMock, explore: Explore) -> None:
    httpx_mock.add_response(json=page([dataset_row(id="a"), dataset_row(id="b")], total=3))
    httpx_mock.add_response(json=page([dataset_row(id="c")], total=3, offset=2))
    rows = list(explore.iter_datasets(page_size=2))
    assert [r.id for r in rows] == ["a", "b", "c"]
    assert [_query(r)["offset"] for r in httpx_mock.get_requests()] == [["0"], ["2"]]


def test_dataset_quotes_both_segments(httpx_mock: HTTPXMock, explore: Explore) -> None:
    httpx_mock.add_response(json={"data": dataset_row(readme="# card")})
    ds = explore.dataset("acme/Parking Lots")
    assert str(httpx_mock.get_requests()[0].url) == f"{PUBLIC}/datasets/acme/Parking%20Lots"
    assert ds.readme == "# card"


def test_dataset_404(httpx_mock: HTTPXMock, explore: Explore) -> None:
    httpx_mock.add_response(status_code=404, json={"detail": "Public dataset 'acme/x' not found"})
    with pytest.raises(NotFoundError):
        explore.dataset("acme/x")


# ───────────── models ─────────────


def test_models_sends_the_filters_and_reads_the_classes(
    httpx_mock: HTTPXMock, explore: Explore
) -> None:
    httpx_mock.add_response(json=page([model_row()], total=1))
    [model] = explore.models(search="scene", model_type="instance_segmentation", sort="popular")
    request = httpx_mock.get_requests()[0]
    assert str(request.url).startswith(f"{PUBLIC}/models?")
    assert _query(request)["model_type"] == ["instance_segmentation"]
    assert isinstance(model, PublicModel)
    assert model.ref == "pictograph-research/pictograph-sceneparse150-m"
    assert model.classes == ["wall", "floor"]


def test_iter_models_stops_at_max_total(httpx_mock: HTTPXMock, explore: Explore) -> None:
    httpx_mock.add_response(json=page([model_row(id="a"), model_row(id="b")], total=5))
    assert [m.id for m in explore.iter_models(page_size=2, max_total=2)] == ["a", "b"]
    assert len(httpx_mock.get_requests()) == 1


def test_a_model_with_dict_classes_reads_their_names() -> None:
    model = PublicModel.model_validate(model_row(class_mapping={"classes": [{"name": "cat"}]}))
    assert model.classes == ["cat"]


# ───────────── import ─────────────


def import_row(**over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "import_id": IMPORT_ID,
        "status": "pending",
        "dataset_id": "new-1",
        "dataset_name": "COCO-5k",
        "source": {
            "organization": "pictograph-research",
            "dataset": "COCO-5k",
            "dataset_id": "ds-1",
        },
        "total_images": 5000,
        "image_count": 0,
        "progress": 0.0,
        "error": None,
        "created_at": None,
        "completed_at": None,
    }
    row.update(over)
    return row


def test_import_without_waiting(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(status_code=202, json={"data": import_row()})
    job = datasets.import_public("pictograph-research/coco-5k", name="my coco", wait=False)
    request = httpx_mock.get_requests()[0]
    assert request.method == "POST"
    assert str(request.url) == f"{PUBLIC}/datasets/pictograph-research/coco-5k/import"
    assert json.loads(request.content) == {"name": "my coco"}
    assert job.status == "pending" and not job.done


def test_import_with_no_name_sends_no_body(httpx_mock: HTTPXMock, datasets: Datasets) -> None:
    httpx_mock.add_response(status_code=202, json={"data": import_row()})
    datasets.import_public("coco-5k", organization="pictograph-research", wait=False)
    assert httpx_mock.get_requests()[0].content == b""


def test_import_waits_until_completed(
    httpx_mock: HTTPXMock, datasets: Datasets, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pictograph.resources.datasets.time.sleep", lambda _s: None)
    httpx_mock.add_response(status_code=202, json={"data": import_row()})
    httpx_mock.add_response(
        json={"data": import_row(status="running", image_count=2500, progress=60.0)}
    )
    httpx_mock.add_response(
        json={"data": import_row(status="completed", image_count=5000, progress=100.0)}
    )
    job = datasets.import_public("pictograph-research/coco-5k", poll_interval=0.01)
    assert job.status == "completed" and job.done and job.image_count == 5000
    polls = [str(r.url) for r in httpx_mock.get_requests()[1:]]
    assert polls == [f"{PUBLIC}/imports/{IMPORT_ID}"] * 2


@pytest.mark.parametrize(
    ("status", "code"), [("failed", "job_failed"), ("cancelled", "job_cancelled")]
)
def test_a_wait_on_an_import_that_ended_badly_raises(
    httpx_mock: HTTPXMock, datasets: Datasets, status: str, code: str
) -> None:
    httpx_mock.add_response(json={"data": import_row(status=status, error="copy failed")})
    with pytest.raises(ApiError) as err:
        datasets.wait_for_public_import(IMPORT_ID, sleep=lambda _s: None)
    assert err.value.code == code
    assert err.value.response["import_id"] == IMPORT_ID


def test_a_wait_that_runs_out_says_it_keeps_running(
    httpx_mock: HTTPXMock, datasets: Datasets, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = iter([0.0, 0.0, 5.0, 10.0])
    monkeypatch.setattr("pictograph.resources.datasets.time.monotonic", lambda: next(clock))
    httpx_mock.add_response(
        json={"data": import_row(status="running", image_count=10)}, is_reusable=True
    )
    with pytest.raises(PollTimeoutError, match="still running"):
        datasets.wait_for_public_import(IMPORT_ID, timeout=4.0, sleep=lambda _s: None)


def test_the_image_limit_refusal_is_a_payment_error(
    httpx_mock: HTTPXMock, datasets: Datasets
) -> None:
    httpx_mock.add_response(
        status_code=402,
        json={
            "detail": {
                "error": "image_limit_exceeded",
                "message": "Importing this dataset (5000 images) would exceed",
                "current_images": 4000,
                "additional_images": 5000,
                "max_images": 5000,
            }
        },
    )
    with pytest.raises(PaymentRequiredError):
        datasets.import_public("pictograph-research/coco-5k")
