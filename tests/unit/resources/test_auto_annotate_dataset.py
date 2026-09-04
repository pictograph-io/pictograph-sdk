"""Tests for ``AutoAnnotate.dataset``.

The workflow is an orchestrator over Client resources. We mock the
Client's resource methods directly (via ``unittest.mock``) since the
underlying HTTP behavior is exhaustively covered in the resource tests
- we don't want to re-test that here. The workflow tests focus on
orchestration: class normalization, batch vs text mode, the overwrite
flag, the max_images cap, and per-image failure handling.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from pictograph.exceptions import (
    ApiError,
    NetworkError,
    NotFoundError,
    PollTimeoutError,
    RequestTimeoutError,
)
from pictograph.models.auto_annotate import BatchClass, BatchJob
from pictograph.models.dataset import Dataset, DatasetImage
from pictograph.resources.auto_annotate import (
    AnnotateReport,
    AnnotationFailure,
    AutoAnnotate,
)
from tests.unit.resources._orchestration import build, sibling_resources


def _invoke(client: MagicMock, *args: object, **kwargs: object) -> object:
    """Invoke the real method on a real resource with its siblings stubbed."""
    with sibling_resources(client):
        resource = build(AutoAnnotate, client, own="auto_annotate", delegate=["batch", "text"])
        return resource.dataset(*args, **kwargs)


def _make_image(
    image_id: str,
    filename: str,
    annotation_count: int = 0,
) -> DatasetImage:
    return DatasetImage(
        id=image_id,
        filename=filename,
        status="new",  # type: ignore[arg-type]
        annotation_count=annotation_count,
        file_size=12345,
        width=640,
        height=480,
        image_url=None,
        thumbnail_url=None,
        annotation_url=None,
        created_at=datetime.now(timezone.utc),
    )


def _make_dataset(images: list[DatasetImage] | None = None) -> Dataset:
    return Dataset(
        id="proj-uuid-1",
        name="test-dataset",
        description=None,
        image_count=len(images or []),
        completed_image_count=0,
        total_size=0,
        archived_images=0,
        classes=[],
        images=images,
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
    client.datasets.get.return_value = _make_dataset(images=[_make_image("img-1", "a.jpg")])
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
    client.datasets.get.return_value = _make_dataset()
    with pytest.raises(ValueError, match="Unsupported class spec"):
        _invoke(client, "test-dataset", classes=[12345])  # type: ignore[list-item]


def test_annotate_one_tuple_defaults_to_polygon() -> None:
    """``(name,)`` is a valid shorthand → defaults to polygon."""
    client = MagicMock()
    client.datasets.get.return_value = _make_dataset(images=[_make_image("img-1", "a.jpg")])
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
    client.datasets.get.return_value = _make_dataset()
    with pytest.raises(ValueError, match="Unsupported class spec"):
        _invoke(
            client,
            "test-dataset",
            classes=[("car", "bbox", "extra")],  # type: ignore[list-item]
        )


def test_annotate_empty_tuple_raises_value_error_not_index_error() -> None:
    """A 0-tuple must raise the clean ValueError, not crash on ``c[0]``."""
    client = MagicMock()
    client.datasets.get.return_value = _make_dataset()
    with pytest.raises(ValueError, match="Unsupported class spec"):
        _invoke(client, "test-dataset", classes=[()])  # type: ignore[list-item]


# ───────────── batch mode (default) ─────────────


def test_annotate_batch_mode_aggregates_job_outcome() -> None:
    """processed_images / total_annotations_added / job_id surface on the report."""
    client = MagicMock()
    client.datasets.get.return_value = _make_dataset(
        images=[
            _make_image("img-1", "a.jpg"),
            _make_image("img-2", "b.jpg"),
        ]
    )
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
    client.datasets.get.return_value = _make_dataset(images=[_make_image("img-1", "a.jpg")])
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
    client.datasets.get.return_value = _make_dataset(images=[_make_image("img-1", "a.jpg")])
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
    client.datasets.get.return_value = _make_dataset(images=[_make_image("img-1", "a.jpg")])
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
    client.datasets.get.return_value = _make_dataset(images=[_make_image("img-1", "a.jpg")])
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
    client.datasets.get.return_value = _make_dataset(
        images=[_make_image("img-1", "a.jpg"), _make_image("img-2", "b.jpg")]
    )
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
    """poll_interval and timeout flow through to the batch resource."""
    client = MagicMock()
    client.datasets.get.return_value = _make_dataset(images=[_make_image("img-1", "a.jpg")])
    client.auto_annotate.batch.return_value = _ok_batch_job(processed=1, added=1)

    _invoke(
        client,
        "test-dataset",
        classes=[("car", "polygon")],
        poll_interval=2.0,
        timeout=600.0,
    )
    kwargs = client.auto_annotate.batch.call_args.kwargs
    assert kwargs["poll_interval"] == 2.0
    assert kwargs["timeout"] == 600.0
    assert kwargs["wait"] is True


# ───────────── text mode ─────────────


def test_annotate_text_mode_calls_text_per_image_per_class() -> None:
    """One client.auto_annotate.text() call per (image × class) pair, but exactly
    ONE save per image (all classes accumulated into a single full-replacement)."""
    client = MagicMock()
    client.datasets.get.return_value = _make_dataset(
        images=[
            _make_image("img-1", "a.jpg"),
            _make_image("img-2", "b.jpg"),
        ]
    )
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
    client.datasets.get.return_value = _make_dataset(images=[_make_image("img-1", "a.jpg")])
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
    client.datasets.get.return_value = _make_dataset(
        images=[_make_image("img-1", "a.jpg", annotation_count=5)]
    )
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
    client.datasets.get.return_value = _make_dataset(images=[_make_image("img-1", "a.jpg")])
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
    client.datasets.get.return_value = _make_dataset(images=[_make_image("img-1", "a.jpg")])
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
    client.datasets.get.return_value = _make_dataset(images=[_make_image("img-1", "a.jpg")])
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
    client.datasets.get.return_value = _make_dataset(
        images=[
            _make_image("img-1", "a.jpg", annotation_count=3),
            _make_image("img-2", "b.jpg", annotation_count=0),
        ]
    )
    client.auto_annotate.batch.return_value = _ok_batch_job(processed=1, added=2)

    report = _invoke(client, "test-dataset", classes=[("car", "polygon")])
    assert report.images_attempted == 1
    assert report.images_skipped == 1
    sent_filenames = client.auto_annotate.batch.call_args.kwargs["image_filenames"]
    assert sent_filenames == ["b.jpg"]


def test_annotate_overwrite_processes_all() -> None:
    """``overwrite=True`` processes every image regardless of count."""
    client = MagicMock()
    client.datasets.get.return_value = _make_dataset(
        images=[
            _make_image("img-1", "a.jpg", annotation_count=3),
            _make_image("img-2", "b.jpg", annotation_count=0),
        ]
    )
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
    client.datasets.get.return_value = _make_dataset(
        images=[_make_image(f"img-{i}", f"img-{i}.jpg") for i in range(10)]
    )
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
    client.datasets.get.return_value = _make_dataset(
        images=[_make_image(f"img-{i}", f"img-{i}.jpg", annotation_count=0) for i in range(10)]
    )
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
    client.datasets.get.return_value = _make_dataset(images=imgs)
    client.auto_annotate.batch.return_value = _ok_batch_job(processed=2, added=2)

    report = _invoke(client, "test-dataset", classes=[("car", "polygon")], max_images=2)
    assert report.images_attempted == 2
    assert report.images_skipped == 3  # already annotated
    assert report.images_capped == 5  # eligible but over the cap


# ───────────── empty / missing ─────────────


def test_annotate_empty_dataset_returns_empty_report() -> None:
    client = MagicMock()
    client.datasets.get.return_value = _make_dataset(images=[])
    report = _invoke(client, "test-dataset", classes=[("car", "polygon")])
    assert report.images_attempted == 0
    assert report.images_processed == 0
    assert report.failures == []
    assert not report.success
    client.auto_annotate.batch.assert_not_called()


def test_annotate_propagates_dataset_not_found() -> None:
    client = MagicMock()
    client.datasets.get.side_effect = NotFoundError("missing dataset")
    with pytest.raises(NotFoundError):
        _invoke(client, "missing", classes=[("car", "polygon")])


# ───────────── full enumeration + per-batch cap (no silent truncation) ─────────────


def test_annotate_fetches_full_image_list_not_default_1000_page() -> None:
    """Regression: the workflow must enumerate the dataset at the backend's
    per-call maximum, not let ``Datasets.get`` default to images_limit=1000 and
    silently annotate only the first page of a larger dataset."""
    client = MagicMock()
    client.datasets.get.return_value = _make_dataset(
        images=[_make_image(f"img-{i}", f"f{i}.jpg") for i in range(1500)]
    )
    client.auto_annotate.batch.return_value = _ok_batch_job(processed=1500, added=10)

    report = _invoke(client, "big", classes=[("car", "polygon")])

    # The fetch asks for the full per-call window (10000), not the silent 1000 default.
    assert client.datasets.get.call_args.kwargs.get("images_limit") == 10000
    # All 1500 (≤ the 5000 batch cap) are submitted - nothing silently dropped.
    assert report.images_attempted == 1500
    assert report.images_capped == 0
    assert len(client.auto_annotate.batch.call_args.kwargs["image_filenames"]) == 1500


def test_annotate_batch_cap_overflow_is_explicit_not_silent() -> None:
    """>5000 eligible images: submit exactly 5000 and record the remainder as a
    failure (success=False) - never silently drop it or 422 the whole batch."""
    client = MagicMock()
    client.datasets.get.return_value = _make_dataset(
        images=[_make_image(f"img-{i}", f"f{i}.jpg") for i in range(5003)]
    )
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
    client.datasets.get.return_value = _make_dataset(images=imgs)
    # text mode calls a per-image SAM3 text endpoint; stub it to a no-op result.
    client.auto_annotate.text.return_value = MagicMock(annotations=[])

    report = _invoke(client, "huge", classes=[("car", "polygon")], mode="text")

    assert report.images_attempted == 5003  # nothing capped in text mode
    assert not any(f.image_filename == "<batch-cap>" for f in report.failures)
