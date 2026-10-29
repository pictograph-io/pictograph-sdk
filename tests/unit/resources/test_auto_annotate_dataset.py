"""Tests for ``AutoAnnotate.dataset``.

The workflow is an orchestrator over Client resources. We mock the
Client's resource methods directly (via ``unittest.mock``) since the
underlying HTTP behavior is exhaustively covered in the resource tests
- we don't want to re-test that here. The workflow tests focus on
orchestration: class normalization, batch vs text mode, the overwrite
flag, the max_images cap, and per-image failure handling.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import httpx
import pytest

from pictograph import Client
from pictograph.exceptions import (
    ApiError,
    NetworkError,
    NotFoundError,
    PaymentRequiredError,
    PollTimeoutError,
    RequestTimeoutError,
)
from pictograph.models.auto_annotate import BatchClass, BatchJob
from pictograph.models.image import Image
from pictograph.resources.auto_annotate import (
    AnnotateReport,
    AnnotationFailure,
    AutoAnnotate,
)
from tests.unit.resources._orchestration import build, sibling_resources

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

_NOW = "2026-04-01T00:00:00Z"


def _invoke(client: MagicMock, *args: object, **kwargs: object) -> object:
    """Invoke the real method on a real resource with its siblings stubbed.

    ``dataset`` submits with ``batch(wait=False)`` and then waits with
    ``wait_for_batch``; unless a test configures the wait, it ends with the job the
    test's ``batch`` stub returns (the terminal snapshot those tests describe).
    """
    aa = client.auto_annotate
    if aa.wait_for_batch.side_effect is None and isinstance(
        aa.wait_for_batch.return_value, MagicMock
    ):
        aa.wait_for_batch.side_effect = lambda *_a, **_k: aa.batch.return_value
    with sibling_resources(client):
        resource = build(
            AutoAnnotate,
            client,
            own="auto_annotate",
            delegate=["batch", "text", "wait_for_batch"],
        )
        return resource.dataset(*args, **kwargs)


def _make_image(
    image_id: str,
    filename: str,
    annotation_count: int = 0,
    directory_path: str = "/",
) -> Image:
    return Image(
        id=image_id,
        filename=filename,
        status="new",
        annotation_count=annotation_count,
        file_size=12345,
        width=640,
        height=480,
        directory_path=directory_path,
        created_at=datetime.now(timezone.utc),
    )


def _ok_batch_job(processed: int = 3, added: int = 6, failed: int = 0) -> BatchJob:
    return BatchJob(
        job_id="job-uuid-1",
        status="completed",
        progress=100,
        total_images=processed + failed,
        processed_images=processed,
        total_annotations_added=added,
        failed_images=failed,
        error_message=None,
        estimated_credits=None,
        completed_at=datetime.now(timezone.utc),
    )


# ───────────── class normalization ─────────────


def test_annotate_normalizes_batchclass_tuple_dict() -> None:
    """All three class-spec forms (BatchClass / tuple / dict) work."""
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-1", "a.jpg")]
    client.auto_annotate.batch.return_value = _ok_batch_job(processed=1, added=3)

    report = _invoke(
        client,
        "test-dataset",
        classes=[
            BatchClass(name="car", output_type="polygon"),
            ("person", "bbox"),
            {"name": "tree", "output_type": "polygon"},
        ],
    )
    assert report.success
    # Confirm canonical BatchClass list reached the resource.
    sent_classes = client.auto_annotate.batch.call_args.kwargs["classes"]
    assert all(isinstance(c, BatchClass) for c in sent_classes)
    assert {c.name for c in sent_classes} == {"car", "person", "tree"}


def test_annotate_rejects_unknown_class_spec() -> None:
    client = MagicMock()
    client.images.iter.return_value = []
    with pytest.raises(ValueError, match="Unsupported class spec"):
        _invoke(client, "test-dataset", classes=[12345])  # type: ignore[list-item]


def test_annotate_one_tuple_defaults_to_polygon() -> None:
    """``(name,)`` is a valid shorthand → defaults to polygon."""
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-1", "a.jpg")]
    client.auto_annotate.batch.return_value = _ok_batch_job(processed=1, added=3)

    report = _invoke(client, "test-dataset", classes=[("person",)])
    assert report.success
    sent = client.auto_annotate.batch.call_args.kwargs["classes"]
    assert len(sent) == 1
    assert sent[0].name == "person"
    assert sent[0].output_type == "polygon"


def test_annotate_three_tuple_is_rejected_not_silently_truncated() -> None:
    """A malformed 3-tuple must raise - never silently drop the caller's
    output_type and annotate as polygon (the pre-fix behavior)."""
    client = MagicMock()
    client.images.iter.return_value = []
    with pytest.raises(ValueError, match="Unsupported class spec"):
        _invoke(
            client,
            "test-dataset",
            classes=[("car", "bbox", "extra")],  # type: ignore[list-item]
        )


def test_annotate_empty_tuple_raises_value_error_not_index_error() -> None:
    """A 0-tuple must raise the clean ValueError, not crash on ``c[0]``."""
    client = MagicMock()
    client.images.iter.return_value = []
    with pytest.raises(ValueError, match="Unsupported class spec"):
        _invoke(client, "test-dataset", classes=[()])  # type: ignore[list-item]


# ───────────── batch mode (default) ─────────────


def test_annotate_batch_mode_aggregates_job_outcome() -> None:
    """processed_images / total_annotations_added / job_id surface on the report."""
    client = MagicMock()
    client.images.iter.return_value = [
        _make_image("img-1", "a.jpg"),
        _make_image("img-2", "b.jpg"),
    ]
    client.auto_annotate.batch.return_value = _ok_batch_job(processed=2, added=5)

    report = _invoke(client, "test-dataset", classes=[("car", "polygon")])
    assert isinstance(report, AnnotateReport)
    assert report.images_attempted == 2
    assert report.images_processed == 2
    assert report.annotations_added == 5
    assert report.job_id == "job-uuid-1"
    assert report.success
    assert report.failures == []


def test_annotate_batch_mode_reports_partial_failures() -> None:
    """failed_images > 0 → batch failure recorded (workflow doesn't raise)."""
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-1", "a.jpg")]
    client.auto_annotate.batch.return_value = _ok_batch_job(processed=4, added=4, failed=2)

    report = _invoke(client, "test-dataset", classes=[("car", "polygon")])
    assert len(report.failures) == 1
    failure = report.failures[0]
    assert isinstance(failure, AnnotationFailure)
    assert failure.image_filename == "<batch>"
    assert "2 images failed" in failure.reason


def test_annotate_batch_mode_recovers_from_apierror() -> None:
    """ApiError on batch kicker → recorded as failure, not raised."""
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-1", "a.jpg")]
    client.auto_annotate.batch.side_effect = ApiError("batch endpoint exploded")

    report = _invoke(client, "test-dataset", classes=[("car", "polygon")])
    assert report.images_processed == 0
    assert len(report.failures) == 1
    assert "batch endpoint exploded" in report.failures[0].reason


def test_annotate_batch_mode_recovers_from_poll_timeout() -> None:
    """A batch poll timeout is captured in the report (NOT raised). The function
    documents 'Returns AnnotateReport; inspect failures' (it already swallows
    ApiError the same way), so a caller staging upload -> annotate -> train keeps
    the already-completed upload result rather than crashing."""
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-1", "a.jpg")]
    client.auto_annotate.batch.side_effect = PollTimeoutError(
        "Batch job job-9 did not complete within 1800s; fetch later via get_batch(job-9)."
    )

    report = _invoke(client, "test-dataset", classes=[("car", "polygon")])
    assert report.images_processed == 0
    assert report.success is False
    assert len(report.failures) == 1
    assert report.failures[0].image_filename == "<batch>"
    assert "job-9" in report.failures[0].reason


def test_annotate_batch_mode_recovers_from_network_error() -> None:
    """A transient NetworkError mid-batch is captured in the report, NOT raised -
    it's a sibling of ApiError under PictographError, so the old
    `except (ApiError, PollTimeoutError)` missed it and crashed the whole call
    (losing the already-completed upload phase)."""
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-1", "a.jpg")]
    client.auto_annotate.batch.side_effect = NetworkError("connection reset mid-poll")

    report = _invoke(client, "test-dataset", classes=[("car", "polygon")])
    assert report.images_processed == 0
    assert report.success is False
    assert len(report.failures) == 1
    assert "connection reset" in report.failures[0].reason


def test_annotate_text_mode_recovers_from_network_error() -> None:
    """A transient NetworkError on one image's text prompt is recorded per-image;
    the loop continues to the next image instead of aborting the dataset run."""
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-1", "a.jpg"), _make_image("img-2", "b.jpg")]
    text_ok = MagicMock(status="success", annotations=[{"name": "car", "type": "polygon"}])
    client.auto_annotate.text.side_effect = [
        RequestTimeoutError("read timed out"),  # img-1
        text_ok,  # img-2
    ]
    client.annotations.save.return_value = MagicMock(new_count=1, previous_count=0)

    report = _invoke(client, "test-dataset", classes=[("car", "polygon")], mode="text")
    assert report.images_processed == 1  # img-2 still processed
    assert len(report.failures) == 1
    assert report.failures[0].image_filename == "a.jpg"
    assert "timed out" in report.failures[0].reason


def test_annotate_batch_passes_poll_settings() -> None:
    """The job is submitted without waiting, then waited on with poll_interval and
    timeout - so the report names the job even when the wait runs out."""
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-1", "a.jpg")]
    client.auto_annotate.batch.return_value = _ok_batch_job(processed=1, added=1)

    report = _invoke(
        client,
        "test-dataset",
        classes=[("car", "polygon")],
        poll_interval=2.0,
        timeout=600.0,
    )
    assert client.auto_annotate.batch.call_args.kwargs["wait"] is False
    client.auto_annotate.wait_for_batch.assert_called_once_with(
        "job-uuid-1", poll_interval=2.0, timeout=600.0
    )
    assert report.job_status == "completed"


def test_a_wait_that_runs_out_still_names_the_job() -> None:
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-1", "a.jpg")]
    client.auto_annotate.batch.return_value = BatchJob(job_id="job-9", status="running")
    client.auto_annotate.wait_for_batch.side_effect = PollTimeoutError("still running")

    report = _invoke(client, "ds", classes=[("car", "polygon")])

    assert report.job_id == "job-9" and report.job_status == "running"
    (failure,) = report.failures
    assert failure.error_type == "PollTimeoutError"


def test_a_job_that_ended_failed_reports_its_id_and_status() -> None:
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-1", "a.jpg")]
    client.auto_annotate.batch.return_value = BatchJob(job_id="job-9", status="pending")
    client.auto_annotate.wait_for_batch.side_effect = ApiError(
        "Batch job job-9 ended with status 'failed': boom",
        response={"job_id": "job-9", "status": "failed"},
        code="job_failed",
    )

    report = _invoke(client, "ds", classes=[("car", "polygon")])

    assert report.job_id == "job-9" and report.job_status == "failed"
    assert report.failures[0].code == "job_failed"


def test_no_wait_never_polls() -> None:
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-1", "a.jpg")]
    client.auto_annotate.batch.return_value = BatchJob(job_id="job-9", status="pending")

    report = _invoke(client, "ds", classes=[("car", "polygon")], wait=False)

    client.auto_annotate.wait_for_batch.assert_not_called()
    assert report.job_id == "job-9" and report.job_status == "pending"


# ───────────── text mode ─────────────


def test_annotate_text_mode_calls_text_per_image_per_class() -> None:
    """One client.auto_annotate.text() call per (image × class) pair, but exactly
    ONE save per image (all classes accumulated into a single full-replacement)."""
    client = MagicMock()
    client.images.iter.return_value = [
        _make_image("img-1", "a.jpg"),
        _make_image("img-2", "b.jpg"),
    ]
    text_result = MagicMock(status="success", annotations=[MagicMock()])
    client.auto_annotate.text.return_value = text_result
    client.annotations.save.return_value = MagicMock(previous_count=0, new_count=2)

    report = _invoke(
        client,
        "test-dataset",
        classes=[("car", "polygon"), ("truck", "bbox")],
        mode="text",
    )
    # 2 images x 2 classes = 4 text calls.
    assert client.auto_annotate.text.call_count == 4
    # ONE save per image (not per class) - each carries BOTH classes' annotations.
    assert client.annotations.save.call_count == 2
    assert report.images_processed == 2
    assert report.annotations_added == 4


def test_annotate_text_mode_multiclass_saves_all_classes_once() -> None:
    """Regression: a multi-class text run must NOT overwrite earlier classes.

    ``annotations.save`` is a full replacement, so the buggy per-class save left
    only the LAST class on each image. The fix accumulates every class's
    annotations and writes them in a single save per image.
    """
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-1", "a.jpg")]
    car_ann = {"name": "car", "type": "polygon"}
    person_ann = {"name": "person", "type": "polygon"}
    client.auto_annotate.text.side_effect = [
        MagicMock(status="success", annotations=[car_ann]),  # class "car"
        MagicMock(status="success", annotations=[person_ann]),  # class "person"
    ]
    client.annotations.save.return_value = MagicMock(previous_count=0, new_count=2)

    report = _invoke(
        client,
        "test-dataset",
        classes=[("car", "polygon"), ("person", "polygon")],
        mode="text",
    )

    # Exactly one save, carrying BOTH classes (pre-fix: two saves, last wins).
    client.annotations.save.assert_called_once()
    saved = client.annotations.save.call_args.kwargs["annotations"]
    assert saved == [car_ann, person_ann]
    assert report.images_processed == 1
    assert report.annotations_added == 2


def test_annotate_text_mode_overwrite_counts_written_not_net_delta() -> None:
    """Regression: with overwrite=True, re-annotating an already-annotated image
    must report the annotations WRITTEN (new_count), not the net delta
    new_count - previous_count (which under-counted, and went NEGATIVE when the
    image previously had more annotations than this run produced)."""
    client = MagicMock()
    # An image that already has 5 annotations; overwrite=True makes it eligible.
    client.images.iter.return_value = [_make_image("img-1", "a.jpg", annotation_count=5)]
    client.auto_annotate.text.return_value = MagicMock(
        status="success", annotations=[{"name": "car", "type": "polygon"}]
    )
    # Full replacement: image had 5, now has 2 → old code reported 2 - 5 = -3.
    client.annotations.save.return_value = MagicMock(previous_count=5, new_count=2)

    report = _invoke(
        client,
        "test-dataset",
        classes=[("car", "polygon")],
        mode="text",
        overwrite=True,
    )

    client.annotations.save.assert_called_once()
    # Reports what this run wrote (2), never the negative net delta.
    assert report.annotations_added == 2
    assert report.images_processed == 1


def test_annotate_text_mode_records_save_failure() -> None:
    """A failure on the single per-image save is recorded against the image."""
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-1", "a.jpg")]
    client.auto_annotate.text.return_value = MagicMock(
        status="success", annotations=[{"name": "car", "type": "polygon"}]
    )
    client.annotations.save.side_effect = ApiError("save rejected")

    report = _invoke(client, "test-dataset", classes=[("car", "polygon")], mode="text")
    assert report.images_processed == 0
    assert len(report.failures) == 1
    assert report.failures[0].image_filename == "a.jpg"
    assert "save: save rejected" in report.failures[0].reason


def test_annotate_text_mode_records_per_class_failures() -> None:
    """A per-class failure surfaces as an AnnotationFailure for that image."""
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-1", "a.jpg")]
    client.auto_annotate.text.side_effect = ApiError("class not understood")

    report = _invoke(
        client,
        "test-dataset",
        classes=[("car", "polygon")],
        mode="text",
    )
    assert report.images_processed == 0
    assert len(report.failures) == 1
    failure = report.failures[0]
    assert failure.image_filename == "a.jpg"
    assert "car: class not understood" in failure.reason


def test_annotate_text_tag_output_coerced_to_polygon() -> None:
    """``output_type='tag'`` is downgraded to ``polygon`` for SAM3."""
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-1", "a.jpg")]
    client.auto_annotate.text.return_value = MagicMock(status="no_detection", annotations=[])

    _invoke(
        client,
        "test-dataset",
        classes=[BatchClass(name="cat", output_type="tag")],
        mode="text",
    )
    kwargs = client.auto_annotate.text.call_args.kwargs
    assert kwargs["output_type"] == "polygon"


# ───────────── overwrite + max_images ─────────────


def test_annotate_skips_already_annotated_by_default() -> None:
    """``overwrite=False`` (default) skips images with annotation_count > 0."""
    client = MagicMock()
    client.images.iter.return_value = [
        _make_image("img-1", "a.jpg", annotation_count=3),
        _make_image("img-2", "b.jpg", annotation_count=0),
    ]
    client.auto_annotate.batch.return_value = _ok_batch_job(processed=1, added=2)

    report = _invoke(client, "test-dataset", classes=[("car", "polygon")])
    assert report.images_attempted == 1
    assert report.images_skipped == 1
    sent = client.auto_annotate.batch.call_args.kwargs["image_filenames"]
    # The image's ID, not its name (B412): "b.jpg" could also name an annotated b.jpg in
    # another folder, which the batch would then annotate (and charge for) again.
    assert sent == ["img-2"]


def test_annotate_overwrite_processes_all() -> None:
    """``overwrite=True`` processes every image regardless of count."""
    client = MagicMock()
    client.images.iter.return_value = [
        _make_image("img-1", "a.jpg", annotation_count=3),
        _make_image("img-2", "b.jpg", annotation_count=0),
    ]
    client.auto_annotate.batch.return_value = _ok_batch_job(processed=2, added=4)

    report = _invoke(
        client,
        "test-dataset",
        classes=[("car", "polygon")],
        overwrite=True,
    )
    assert report.images_attempted == 2
    assert report.images_skipped == 0


def test_annotate_max_images_caps_payload() -> None:
    client = MagicMock()
    client.images.iter.return_value = [_make_image(f"img-{i}", f"img-{i}.jpg") for i in range(10)]
    client.auto_annotate.batch.return_value = _ok_batch_job(processed=3, added=3)

    report = _invoke(
        client,
        "test-dataset",
        classes=[("car", "polygon")],
        max_images=3,
    )
    assert report.images_attempted == 3
    sent_filenames = client.auto_annotate.batch.call_args.kwargs["image_filenames"]
    assert len(sent_filenames) == 3


def test_annotate_max_images_does_not_inflate_skipped() -> None:
    """The cap remainder is reported as images_capped, NOT images_skipped.

    All 10 images are unannotated; max_images=3 caps the run. The 7 held-back
    images were not skipped-as-already-done (pre-fix images_skipped reported 7).
    """
    client = MagicMock()
    client.images.iter.return_value = [
        _make_image(f"img-{i}", f"img-{i}.jpg", annotation_count=0) for i in range(10)
    ]
    client.auto_annotate.batch.return_value = _ok_batch_job(processed=3, added=3)

    report = _invoke(client, "test-dataset", classes=[("car", "polygon")], max_images=3)
    assert report.images_attempted == 3
    assert report.images_skipped == 0
    assert report.images_capped == 7


def test_annotate_skipped_and_capped_are_distinct() -> None:
    """3 already-annotated (skipped) + 7 eligible, capped at 2 → skipped=3, capped=5."""
    client = MagicMock()
    imgs = [_make_image(f"done-{i}", f"done-{i}.jpg", annotation_count=2) for i in range(3)]
    imgs += [_make_image(f"new-{i}", f"new-{i}.jpg", annotation_count=0) for i in range(7)]
    client.images.iter.return_value = imgs
    client.auto_annotate.batch.return_value = _ok_batch_job(processed=2, added=2)

    report = _invoke(client, "test-dataset", classes=[("car", "polygon")], max_images=2)
    assert report.images_attempted == 2
    assert report.images_skipped == 3  # already annotated
    assert report.images_capped == 5  # eligible but over the cap


# ───────────── empty / missing ─────────────


def test_annotate_empty_dataset_returns_empty_report() -> None:
    client = MagicMock()
    client.images.iter.return_value = []
    report = _invoke(client, "test-dataset", classes=[("car", "polygon")])
    assert report.images_attempted == 0
    assert report.images_processed == 0
    assert report.failures == []
    assert not report.success
    client.auto_annotate.batch.assert_not_called()


def test_annotate_propagates_dataset_not_found() -> None:
    client = MagicMock()
    client.images.iter.side_effect = NotFoundError("missing dataset")
    with pytest.raises(NotFoundError):
        _invoke(client, "missing", classes=[("car", "polygon")])


# ───────────── full enumeration + per-batch cap (no silent truncation) ─────────────


def test_annotate_reads_every_page_of_the_dataset(httpx_mock: HTTPXMock) -> None:
    """Regression: the dataset was read with ONE ``Datasets.get(include_images=True)``
    capped at 10,000 images, so a bigger dataset was silently annotated in part - and
    one whose first 10,000 were already annotated was reported as "nothing to do".

    Drives the REAL method over the wire: the images list is paged to the end, and
    the batch submits every eligible image from every page."""
    base = "https://api.test.local"
    first = [
        {"id": f"img-{i}", "filename": f"f{i}.jpg", "annotation_count": 0, "created_at": _NOW}
        for i in range(1000)
    ]
    second = [
        {"id": "img-late", "filename": "late.jpg", "annotation_count": 0, "created_at": _NOW},
        {"id": "img-done", "filename": "done.jpg", "annotation_count": 3, "created_at": _NOW},
    ]

    def page(request: httpx.Request) -> httpx.Response:
        assert request.url.params["directory_path"] == "/val"
        offset = int(request.url.params["offset"])
        assert request.url.params["limit"] == "1000"
        rows = first if offset == 0 else second
        return httpx.Response(200, json={"data": rows})

    httpx_mock.add_callback(
        page, url=re.compile(rf"{re.escape(base)}/api/v1/developer/images/\\?.*"), is_reusable=True
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{base}/api/v1/developer/auto-annotate/batch",
        json={"job_id": "job-1", "status": "pending", "progress": 0},
    )
    with Client(api_key="pk_live_test", base_url=base, max_retries=0) as client:
        report = client.auto_annotate.dataset(
            "big", [("car", "bbox")], directory_path="/val", wait=False
        )

    batch = json.loads(next(r for r in httpx_mock.get_requests() if r.method == "POST").content)
    assert len(batch["image_filenames"]) == 1001, "page 2's unannotated image is submitted"
    assert "img-late" in batch["image_filenames"]
    assert "img-done" not in batch["image_filenames"]
    assert report.images_attempted == 1001 and report.images_skipped == 1
    assert report.job_id == "job-1" and report.job_status == "pending"


def test_annotate_reads_one_folder_at_the_list_page_cap() -> None:
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-1", "a.jpg", directory_path="/val")]
    client.auto_annotate.batch.return_value = _ok_batch_job(processed=1, added=2)

    _invoke(client, "ds", classes=[("car", "polygon")], directory_path="/val")

    kwargs = client.images.iter.call_args.kwargs
    assert kwargs["directory_path"] == "/val"
    assert kwargs["page_size"] == 1000


def test_annotate_forwards_a_model_sahi_and_no_wait_to_the_batch() -> None:
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-1", "a.jpg")]
    client.auto_annotate.batch.return_value = BatchJob(
        job_id="job-7", status="running", progress=0, total_images=1
    )

    report = _invoke(
        client,
        "ds",
        classes=[("car", "bbox")],
        model="Shelf Detector",
        sahi=True,
        sahi_slice_size=512,
        wait=False,
    )

    kwargs = client.auto_annotate.batch.call_args.kwargs
    assert kwargs["model"] == "Shelf Detector"
    assert kwargs["sahi"] is True and kwargs["sahi_slice_size"] == 512
    assert kwargs["wait"] is False
    assert report.job_id == "job-7" and report.job_status == "running"
    assert report.images_processed == 0


def test_annotate_with_a_model_needs_no_classes_but_sam3_does() -> None:
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-1", "a.jpg")]
    client.auto_annotate.batch.return_value = _ok_batch_job(processed=1, added=1)

    _invoke(client, "ds", classes=[], model="Shelf Classifier")
    assert client.auto_annotate.batch.call_args.kwargs["classes"] == []

    with pytest.raises(ValueError, match="at least one class"):
        _invoke(client, "ds", classes=[])


def test_annotate_text_mode_refuses_a_model_or_sahi() -> None:
    client = MagicMock()
    with pytest.raises(ValueError, match="mode='batch' only"):
        _invoke(client, "ds", classes=[("car", "bbox")], mode="text", model="m")
    with pytest.raises(ValueError, match="mode='batch' only"):
        _invoke(client, "ds", classes=[("car", "bbox")], mode="text", sahi=True)
    client.images.iter.assert_not_called()


def test_a_failed_batch_keeps_its_errors_class_and_code() -> None:
    """A caller (the CLI's exit code) branches on these, not on the message."""
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-1", "a.jpg")]
    client.auto_annotate.batch.side_effect = PaymentRequiredError(
        "Not enough compute credit.", code="insufficient_credits"
    )

    report = _invoke(client, "ds", classes=[("car", "bbox")])

    (failure,) = report.failures
    assert failure.image_filename == "<batch>"
    assert failure.error_type == "PaymentRequiredError"
    assert failure.code == "insufficient_credits"
    assert "Not enough compute credit." in failure.reason


def test_annotate_batch_cap_overflow_is_explicit_not_silent() -> None:
    """>5000 eligible images: submit exactly 5000 and record the remainder as a
    failure (success=False) - never silently drop it or 422 the whole batch."""
    client = MagicMock()
    client.images.iter.return_value = [_make_image(f"img-{i}", f"f{i}.jpg") for i in range(5003)]
    client.auto_annotate.batch.return_value = _ok_batch_job(processed=5000, added=50)

    report = _invoke(client, "huge", classes=[("car", "polygon")])

    sent = client.auto_annotate.batch.call_args.kwargs["image_filenames"]
    assert len(sent) == 5000  # capped to the batch limit
    assert report.images_attempted == 5000
    assert report.images_capped == 3  # the overflow is accounted for
    assert any(f.image_filename == "<batch-cap>" for f in report.failures)
    assert report.success is False  # incomplete → not a success


def test_annotate_text_mode_has_no_batch_cap() -> None:
    """text mode is per-image (no 5000 batch limit), so it must NOT cap."""
    client = MagicMock()
    imgs = [_make_image(f"img-{i}", f"f{i}.jpg") for i in range(5003)]
    client.images.iter.return_value = imgs
    # text mode calls a per-image SAM3 text endpoint; stub it to a no-op result.
    client.auto_annotate.text.return_value = MagicMock(annotations=[])

    report = _invoke(client, "huge", classes=[("car", "polygon")], mode="text")

    assert report.images_attempted == 5003  # nothing capped in text mode
    assert not any(f.image_filename == "<batch-cap>" for f in report.failures)


def test_text_mode_prompts_the_image_it_saves_to_by_id() -> None:
    """B412: text mode prompts SAM3 on ``img.id`` - the same image ``annotations.save``
    then writes to. A bare filename could resolve to a same-named image in another folder
    and save THAT image's masks onto this one."""
    client = MagicMock()
    client.images.iter.return_value = [_make_image("img-2", "b.jpg", annotation_count=0)]
    client.auto_annotate.text.return_value = MagicMock(status="no_detection", annotations=[])

    _invoke(client, "test-dataset", classes=[("car", "polygon")], mode="text")
    assert client.auto_annotate.text.call_args.kwargs["image_filename"] == "img-2"
