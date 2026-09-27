"""Tests for ``pictograph.resources.auto_annotate.AutoAnnotate``.

Coverage targets:
- ``point``: body shape, success path returns single annotation, no_detection
  yields empty annotations list.
- ``box``: body shape with negative_boxes, status filtering, multi-detection.
- ``text``: body shape, output_type forwarding.
- ``batch``: kicker body shape, BatchClass instances vs raw dicts;
  wait=False returns immediately; wait=True polls + raises on failed.
- ``get_batch`` / ``cancel_batch`` happy paths.
- 402 PaymentRequired propagates through credit context.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import (
    ApiError,
    NotFoundError,
    PaymentRequiredError,
    PollTimeoutError,
)
from pictograph.models.auto_annotate import (
    BatchClass,
    BatchJob,
    ProjectedImages,
    PromptResult,
)
from pictograph.resources.auto_annotate import AutoAnnotate

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
def auto_annotate(transport: Transport) -> AutoAnnotate:
    return AutoAnnotate(transport)


def _polygon_annotation(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "ann-1",
        "name": "stop_sign",
        "type": "polygon",
        "polygon": {"paths": [[{"x": 100, "y": 100}, {"x": 200, "y": 100}, {"x": 150, "y": 200}]]},
        "bounding_box": {"x": 100, "y": 100, "w": 100, "h": 100},
        "confidence": 0.95,
        "created_by": "user-uuid",
    }
    base.update(overrides)
    return base


def _bbox_annotation(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "ann-2",
        "name": "car",
        "type": "bbox",
        "bounding_box": {"x": 200, "y": 200, "w": 50, "h": 80},
        "confidence": 0.88,
        "created_by": "user-uuid",
    }
    base.update(overrides)
    return base


def _job_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "job_id": "job-uuid-1",
        "status": "pending",
        "progress": 0,
        "total_images": 10,
        "processed_images": 0,
        "total_annotations_added": 0,
        "failed_images": 0,
        "estimated_credits": 5,
        "error_message": None,
        "completed_at": None,
    }
    base.update(overrides)
    return base


# ───────────── point ─────────────


def test_point_body_shape(httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/sam3/point",
        json={
            "status": "success",
            "annotation": _polygon_annotation(),
            "score": 0.92,
            "inference_time": 0.45,
        },
    )
    result = auto_annotate.point(
        "road-signs",
        "frame_001.jpg",
        x=150,
        y=150,
        name="stop_sign",
        score_threshold=0.8,
    )
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {
        "dataset_name": "road-signs",
        "image_filename": "frame_001.jpg",
        "x": 150,
        "y": 150,
        "name": "stop_sign",
        "score_threshold": 0.8,
    }
    assert isinstance(result, PromptResult)
    assert result.status == "success"
    assert len(result.annotations) == 1
    assert result.score == 0.92


def test_point_with_negative_points(httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/sam3/point",
        json={"status": "success", "annotation": _polygon_annotation()},
    )
    auto_annotate.point(
        "ds",
        "img.jpg",
        x=10,
        y=10,
        positive_points=[(20, 20), (30, 30)],
        negative_points=[(40, 40)],
    )
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body["positive_points"] == [[20, 20], [30, 30]]
    assert body["negative_points"] == [[40, 40]]


def test_point_no_detection_yields_empty_annotations(
    httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/sam3/point",
        json={"status": "no_detection"},
    )
    result = auto_annotate.point("ds", "img.jpg", x=0, y=0)
    assert result.status == "no_detection"
    assert result.annotations == []


def test_point_402_insufficient_credits(httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/sam3/point",
        status_code=402,
        json={
            "detail": {
                "error": "insufficient_credits",
                "required": 2_500,  # µUSD
                "remaining": 1_000,  # µUSD
                "unit": "micro_usd",
                "upgrade_url": "/settings?tab=billing",
            }
        },
    )
    with pytest.raises(PaymentRequiredError) as exc:
        auto_annotate.point("ds", "img.jpg", x=0, y=0)
    assert exc.value.credit_cost == 2_500
    assert exc.value.credits_remaining == 1_000
    assert exc.value.unit == "micro_usd"


def test_point_404_image_missing(httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/sam3/point",
        status_code=404,
        json={"detail": "Image 'missing.jpg' not found in dataset 'ds'"},
    )
    with pytest.raises(NotFoundError):
        auto_annotate.point("ds", "missing.jpg", x=0, y=0)


# ───────────── box ─────────────


def test_box_body_shape(httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/sam3/box",
        json={
            "status": "success",
            "annotations": [_polygon_annotation(), _bbox_annotation()],
            "score": 0.87,
        },
    )
    result = auto_annotate.box(
        "road-signs",
        "frame_001.jpg",
        box={"x": 100, "y": 100, "w": 200, "h": 200},
        name="stop_sign",
        confidence_threshold=0.6,
        return_polygon=True,
        negative_boxes=[{"x": 50, "y": 50, "w": 30, "h": 30}],
    )
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body["box"] == {"x": 100, "y": 100, "w": 200, "h": 200}
    assert body["name"] == "stop_sign"
    assert body["confidence_threshold"] == 0.6
    assert body["return_polygon"] is True
    assert body["negative_boxes"] == [{"x": 50, "y": 50, "w": 30, "h": 30}]
    assert len(result.annotations) == 2


def test_box_below_threshold(httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/sam3/box",
        json={"status": "below_threshold"},
    )
    result = auto_annotate.box(
        "ds",
        "img.jpg",
        box={"x": 0, "y": 0, "w": 100, "h": 100},
        name="x",
    )
    assert result.status == "below_threshold"
    assert result.annotations == []


# ───────────── text ─────────────


def test_text_body_shape(httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/sam3/text",
        json={
            "status": "success",
            "annotations": [_bbox_annotation(), _bbox_annotation(id="ann-3")],
            "detection_count": 2,
        },
    )
    result = auto_annotate.text(
        "ds",
        "img.jpg",
        text_prompt="red car",
        output_type="bbox",
        confidence_threshold=0.4,
        max_detections=20,
    )
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {
        "dataset_name": "ds",
        "image_filename": "img.jpg",
        "text_prompt": "red car",
        "output_type": "bbox",
        "confidence_threshold": 0.4,
        "max_detections": 20,
    }
    assert len(result.annotations) == 2


# ───────────── batch ─────────────


def test_batch_kicker_body_shape(httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/batch",
        json=_job_payload(),
    )
    classes = [
        BatchClass(name="stop_sign", output_type="polygon"),
        BatchClass(name="yield", output_type="bbox"),
    ]
    job = auto_annotate.batch(
        "road-signs",
        ["a.jpg", "b.jpg"],
        classes,
        confidence_threshold=0.4,
        wait=False,
    )
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {
        "dataset_name": "road-signs",
        "image_filenames": ["a.jpg", "b.jpg"],
        "classes": [
            {"name": "stop_sign", "output_type": "polygon"},
            {"name": "yield", "output_type": "bbox"},
        ],
        "confidence_threshold": 0.4,
        "top_k": 1,
    }
    assert isinstance(job, BatchJob)
    assert job.status == "pending"


def test_batch_accepts_dict_classes(httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/batch",
        json=_job_payload(),
    )
    auto_annotate.batch(
        "ds",
        ["a.jpg"],
        [{"name": "x", "output_type": "polygon"}],
        wait=False,
    )
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body["classes"] == [{"name": "x", "output_type": "polygon"}]


def test_batch_with_model_forwarded(httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/batch",
        json=_job_payload(),
    )
    auto_annotate.batch(
        "ds", ["a.jpg"], [], model="abcdef01-2345-6789-abcd-ef0123456789", wait=False
    )
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body["model_id"] == "abcdef01-2345-6789-abcd-ef0123456789"


def test_batch_sahi_fields_forwarded(httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/batch",
        json=_job_payload(),
    )
    auto_annotate.batch(
        "ds",
        ["a.jpg"],
        [{"name": "x", "output_type": "polygon"}],
        sahi=True,
        sahi_slice_size=512,
        wait=False,
    )
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body["sahi_enabled"] is True
    assert body["sahi_slice_size"] == 512


def test_batch_sahi_off_omits_fields(httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate) -> None:
    """SAHI keys are only sent when opted in - the backend request model is
    extra="forbid", so omission keeps requests valid against pre-SAHI
    backends."""
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/batch",
        json=_job_payload(),
    )
    auto_annotate.batch(
        "ds",
        ["a.jpg"],
        [{"name": "x", "output_type": "polygon"}],
        wait=False,
    )
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert "sahi_enabled" not in body
    assert "sahi_slice_size" not in body


def test_batch_wait_polls_until_completed(
    httpx_mock: HTTPXMock,
    auto_annotate: AutoAnnotate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/batch",
        json=_job_payload(status="pending"),
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/auto-annotate/batch/job-uuid-1",
        json=_job_payload(status="running", progress=40, processed_images=4),
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/auto-annotate/batch/job-uuid-1",
        json=_job_payload(
            status="completed",
            progress=100,
            processed_images=10,
            total_annotations_added=42,
        ),
    )
    sleeps: list[float] = []
    monkeypatch.setattr(
        "pictograph.resources.auto_annotate.time.sleep",
        lambda d: sleeps.append(d),
    )
    job = auto_annotate.batch(
        "ds",
        ["a.jpg"],
        [BatchClass(name="x")],
        wait=True,
        poll_interval=2.0,
        timeout=120.0,
    )
    assert job.status == "completed"
    assert job.total_annotations_added == 42
    assert sleeps == [2.0]


def test_batch_wait_raises_on_failed_status(
    httpx_mock: HTTPXMock,
    auto_annotate: AutoAnnotate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/batch",
        json=_job_payload(status="pending"),
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/auto-annotate/batch/job-uuid-1",
        json=_job_payload(
            status="failed",
            error_message="GPU service timeout after 30 minutes",
        ),
    )
    monkeypatch.setattr("pictograph.resources.auto_annotate.time.sleep", lambda _: None)
    with pytest.raises(ApiError, match="GPU service timeout"):
        auto_annotate.batch(
            "ds",
            ["a.jpg"],
            [BatchClass(name="x")],
            wait=True,
            poll_interval=0.1,
            timeout=10.0,
        )


def test_batch_wait_polltimeout(
    httpx_mock: HTTPXMock,
    auto_annotate: AutoAnnotate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/batch",
        json=_job_payload(status="pending"),
    )
    for _ in range(3):
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE}/api/v1/developer/auto-annotate/batch/job-uuid-1",
            json=_job_payload(status="running", progress=50, processed_images=5),
        )
    times = iter([100.0, 100.0, 105.0, 110.0])  # deadline 110
    monkeypatch.setattr(
        "pictograph.resources.auto_annotate.time.monotonic",
        lambda: next(times),
    )
    monkeypatch.setattr("pictograph.resources.auto_annotate.time.sleep", lambda _: None)
    with pytest.raises(PollTimeoutError, match="did not complete"):
        auto_annotate.batch(
            "ds",
            ["a.jpg"],
            [BatchClass(name="x")],
            wait=True,
            poll_interval=0.1,
            timeout=10.0,
        )


def test_batch_400_no_classes_for_sam3(httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/batch",
        status_code=400,
        json={"detail": "At least one class must be selected for SAM3 batch jobs."},
    )
    from pictograph.exceptions import ValidationError

    with pytest.raises(ValidationError, match="class"):
        auto_annotate.batch("ds", ["a.jpg"], [], wait=False)


# ───────────── get_batch / cancel_batch ─────────────


def test_get_batch_returns_typed_job(httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/auto-annotate/batch/job-uuid-1",
        json=_job_payload(status="running", progress=50),
    )
    job = auto_annotate.get_batch("job-uuid-1")
    assert job.status == "running"
    assert job.progress == 50


def test_cancel_batch(httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/batch/job-uuid-1/cancel",
        json=_job_payload(status="cancelled"),
    )
    job = auto_annotate.cancel_batch("job-uuid-1")
    assert job.status == "cancelled"


def test_wait_for_batch_argument_validation(auto_annotate: AutoAnnotate) -> None:
    with pytest.raises(ValueError, match="poll_interval"):
        auto_annotate.wait_for_batch("job", poll_interval=0.0)
    with pytest.raises(ValueError, match="timeout"):
        auto_annotate.wait_for_batch("job", timeout=0.0)


# ── quote - price a job WITHOUT running it ───────────────────────────────────


def _quote_payload(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "total_images": 2,
        "estimated_credits": 28_273,
        "sahi_tiles": 0,
        "containers": 1,
        "remaining_credits": 10_000_000,
        "sufficient": True,
        "max_images": 5000,
        "exceeds_max_images": False,
    }
    base.update(over)
    return base


def test_quote_prices_existing_images(httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/batch/quote",
        json=_quote_payload(),
    )
    quote = auto_annotate.quote(
        dataset_name="road-signs",
        image_filenames=["a.jpg", "b.jpg"],
        classes=[BatchClass(name="stop_sign", output_type="bbox")],
    )
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {
        "dataset_name": "road-signs",
        "image_filenames": ["a.jpg", "b.jpg"],
        "projected": [],
        "classes": [{"name": "stop_sign", "output_type": "bbox"}],
    }
    assert quote.estimated_credits == 28_273
    assert quote.sufficient is True


def test_quote_prices_a_videos_frames_before_it_exists(
    httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate
) -> None:
    """The whole point of `projected`.

    A video is ONE file and hundreds of frames, and the frames are what you pay for. This
    is how a caller finds out what labelling them costs BEFORE uploading anything - the
    dimensions come from `client.video.probe`, so SAHI tiles price off the real size.
    """
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/batch/quote",
        json=_quote_payload(total_images=600, estimated_credits=213_057),
    )
    quote = auto_annotate.quote(
        projected=[ProjectedImages(count=600, width=1920, height=1080)],
        classes=[{"name": "car", "output_type": "bbox"}],
    )
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body["projected"] == [{"count": 600, "width": 1920, "height": 1080}]
    assert "dataset_name" not in body  # not needed to price images that don't exist yet
    assert quote.total_images == 600


def test_quote_reports_an_over_cap_job_rather_than_truncating(
    httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/batch/quote",
        json=_quote_payload(total_images=6200, exceeds_max_images=True),
    )
    quote = auto_annotate.quote(projected=[{"count": 6200}])
    assert quote.exceeds_max_images is True
    assert quote.max_images == 5000


def test_quote_forwards_sahi(httpx_mock: HTTPXMock, auto_annotate: AutoAnnotate) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/auto-annotate/batch/quote",
        json=_quote_payload(sahi_tiles=16),
    )
    quote = auto_annotate.quote(
        dataset_name="ds", image_filenames=["a.jpg"], sahi=True, sahi_slice_size=512
    )
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body["sahi_enabled"] is True
    assert body["sahi_slice_size"] == 512
    assert quote.sahi_tiles == 16
