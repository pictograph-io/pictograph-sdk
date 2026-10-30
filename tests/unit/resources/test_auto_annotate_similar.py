"""``client.auto_annotate.similar`` / ``quote_similar`` - the app's "Annotate similar" (B533).

The body each sends is the developer route's ``SimilarRequest`` (``extra="forbid"`` on the
server, so a stray key is a 422 there): examples named ONE way (ids or a class), the
confidence omitted unless given (the server's default is the app modal's, one number),
the job followed through the batch status route. Sync and async twins share one body
builder; both are exercised here against the same responses.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from pictograph import AsyncClient
from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import ApiError, ValidationError
from pictograph.models.auto_annotate import BatchJob, SimilarQuote
from pictograph.resources.auto_annotate import AutoAnnotate

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
START = f"{BASE}/api/v1/developer/auto-annotate/similar"
QUOTE = f"{BASE}/api/v1/developer/auto-annotate/similar/quote"
JOB = "44444444-3333-2222-1111-000000000576"

_KICKOFF = {
    "job_id": JOB,
    "job": JOB,
    "status": "pending",
    "total_images": 11,
    "estimated_credits": 52_000,
    "total_in_scope": 11,
    "capped": False,
    "max_images": 2000,
    "examples": 8,
    "target_class": "car",
    "output_type": "polygon",
    "scope": "directory",
    "confidence": 0.45,
}
_QUOTE = {
    "total_images": 11,
    "total_in_scope": 11,
    "capped": False,
    "max_images": 2000,
    "estimated_credits": 52_000,
    "containers": 2,
    "remaining_credits": 4_000_000,
    "sufficient": True,
    "examples": 8,
    "target_class": "car",
    "output_type": "polygon",
    "scope": "directory",
    "directory": "/exemplar-test",
}


@pytest.fixture
def auto_annotate() -> Iterator[AutoAnnotate]:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    transport = Transport(config, api_key=KEY)
    yield AutoAnnotate(transport)
    transport.close()


@pytest.fixture
async def aclient() -> AsyncIterator[AsyncClient]:
    c = AsyncClient(api_key=KEY, base_url=BASE, max_retries=0)
    yield c
    await c.aclose()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _sent(httpx_mock: HTTPXMock, url: str) -> dict[str, Any]:
    (request,) = [r for r in httpx_mock.get_requests() if str(r.url) == url]
    return json.loads(request.read())  # type: ignore[no-any-return]


# ── the body ───────────────────────────────────────────────────────────────────


def test_by_class_with_defaults_sends_exactly_the_route_fields(
    httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate
) -> None:
    httpx_mock.add_response(method="POST", url=START, json=_KICKOFF)
    job = auto_annotate.similar("COCO-5k", "exemplar-test/a.jpg", reference_class="car", wait=False)
    assert isinstance(job, BatchJob)
    assert (job.job_id, job.status, job.total_images, job.estimated_credits) == (
        JOB,
        "pending",
        11,
        52_000,
    )
    assert _sent(httpx_mock, START) == {
        "dataset_name": "COCO-5k",
        "image_filename": "exemplar-test/a.jpg",
        "reference_class": "car",
        "scope": "directory",
        "output_type": "polygon",
    }  # no confidence: the server's default applies; no target_class: the examples' class


def test_by_ids_with_every_option(httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate) -> None:
    httpx_mock.add_response(method="POST", url=START, json=_KICKOFF)
    auto_annotate.similar(
        "COCO-5k",
        "a.jpg",
        annotations=["x", "y", "x"],
        scope="dataset",
        target_class="vehicle",
        output_type="obb",
        confidence=0.6,
        wait=False,
    )
    assert _sent(httpx_mock, START) == {
        "dataset_name": "COCO-5k",
        "image_filename": "a.jpg",
        "annotation_ids": ["x", "y"],
        "scope": "dataset",
        "output_type": "obb",
        "target_class": "vehicle",
        "confidence": 0.6,
    }


@pytest.mark.parametrize(
    "kwargs, words",
    [({}, "Name the example"), ({"annotations": ["x"], "reference_class": "car"}, "not both")],
)
def test_examples_are_named_one_way_before_any_request(
    httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate, kwargs: dict[str, Any], words: str
) -> None:
    with pytest.raises(ValueError, match=words):
        auto_annotate.similar("d", "a.jpg", wait=False, **kwargs)
    with pytest.raises(ValueError, match=words):
        auto_annotate.quote_similar("d", "a.jpg", **kwargs)
    assert httpx_mock.get_requests() == []


# ── the job ────────────────────────────────────────────────────────────────────


def test_wait_follows_the_job_through_the_batch_status_route(
    httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pictograph.resources.auto_annotate.time.sleep", lambda _s: None)
    httpx_mock.add_response(method="POST", url=START, json=_KICKOFF)
    status = f"{BASE}/api/v1/developer/auto-annotate/batch/{JOB}"
    httpx_mock.add_response(
        method="GET", url=status, json={"job_id": JOB, "status": "running", "progress": 40}
    )
    httpx_mock.add_response(
        method="GET",
        url=status,
        json={
            "job_id": JOB,
            "status": "completed",
            "progress": 100,
            "total_images": 11,
            "processed_images": 11,
            "total_annotations_added": 23,
            "failed_images": 0,
        },
    )
    job = auto_annotate.similar("COCO-5k", "a.jpg", reference_class="car", poll_interval=0.01)
    assert (job.status, job.total_annotations_added) == ("completed", 23)


def test_a_job_that_failed_raises_with_the_servers_reason(
    httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B576: a reference SAM3 cannot read ends the job "failed" with the sentence that says why."""
    monkeypatch.setattr("pictograph.resources.auto_annotate.time.sleep", lambda _s: None)
    sentence = "'a.jpg' can't be read as an image (its JPEG data is damaged or cut short)."
    httpx_mock.add_response(method="POST", url=START, json=_KICKOFF)
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/auto-annotate/batch/{JOB}",
        json={"job_id": JOB, "status": "failed", "error_message": sentence},
    )
    with pytest.raises(ApiError, match="can't be read as an image") as exc:
        auto_annotate.similar("COCO-5k", "a.jpg", reference_class="car", poll_interval=0.01)
    assert exc.value.code == "job_failed"


def test_an_unreadable_reference_refused_at_start_is_a_validation_error(
    httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=START,
        status_code=422,
        json={
            "detail": {
                "error": "image_unreadable",
                "image_name": "a.svg",
                "message": "'a.svg' can't be read as an image (it is an SVG (vector) file, and a dataset "
                "image must be a raster). Upload a JPEG or PNG copy to annotate it with SAM3.",
            }
        },
    )
    with pytest.raises(ValidationError, match="can't be read as an image"):
        auto_annotate.similar("d", "a.svg", reference_class="car")


# ── the quote ──────────────────────────────────────────────────────────────────


def test_quote_sends_the_same_body_and_parses(
    httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate
) -> None:
    httpx_mock.add_response(method="POST", url=QUOTE, json=_QUOTE)
    quote = auto_annotate.quote_similar(
        "COCO-5k", "a.jpg", reference_class="car", scope="directory"
    )
    assert isinstance(quote, SimilarQuote)
    assert (quote.total_images, quote.estimated_credits, quote.sufficient, quote.directory) == (
        11,
        52_000,
        True,
        "/exemplar-test",
    )
    assert _sent(httpx_mock, QUOTE) == {
        "dataset_name": "COCO-5k",
        "image_filename": "a.jpg",
        "reference_class": "car",
        "scope": "directory",
        "output_type": "polygon",
    }


# ── the async twins ────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_the_async_twins_send_the_same_bodies(
    httpx_mock: HTTPXMock, aclient: AsyncClient
) -> None:
    httpx_mock.add_response(method="POST", url=START, json=_KICKOFF)
    httpx_mock.add_response(method="POST", url=QUOTE, json=_QUOTE)
    job = await aclient.auto_annotate.similar("COCO-5k", "a.jpg", annotations=["x"], wait=False)
    quote = await aclient.auto_annotate.quote_similar("COCO-5k", "a.jpg", annotations=["x"])
    assert job.job_id == JOB and quote.total_images == 11
    expected = {
        "dataset_name": "COCO-5k",
        "image_filename": "a.jpg",
        "annotation_ids": ["x"],
        "scope": "directory",
        "output_type": "polygon",
    }
    assert _sent(httpx_mock, START) == expected == _sent(httpx_mock, QUOTE)


@pytest.mark.anyio
async def test_the_async_twin_refuses_like_the_sync_one(
    httpx_mock: HTTPXMock, aclient: AsyncClient
) -> None:
    with pytest.raises(ValueError, match="not both"):
        await aclient.auto_annotate.similar("d", "a.jpg", annotations=["x"], reference_class="car")
    assert httpx_mock.get_requests() == []
