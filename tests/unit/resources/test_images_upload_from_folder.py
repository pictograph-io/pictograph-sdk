"""Tests for ``Images.upload_from_directory``.

The workflow is an orchestrator over Client resources. We mock the
Client's resource methods directly (via ``unittest.mock``) since the
underlying HTTP behavior is exhaustively covered in the resource tests
- we don't want to re-test that here. The workflow tests focus on
orchestration: directory walking, virtual-directory mapping, parallel
upload behavior, conflict handling, and the failure report shape.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pictograph.exceptions import (
    ApiError,
    ConflictError,
    NetworkError,
    NotFoundError,
    RequestTimeoutError,
)
from pictograph.models.dataset import Dataset, DatasetClass
from pictograph.resources.images import (
    Images,
    UploadFailure,
    UploadReport,
    virtual_directory_for,
)
from tests.unit.resources._orchestration import build, sibling_resources


def _invoke(client: MagicMock, *args: object, **kwargs: object) -> object:
    """Invoke the real method on a real resource with its siblings stubbed."""
    with sibling_resources(client):
        resource = build(Images, client, own="images", delegate=["upload"])
        return resource.upload_from_directory(*args, **kwargs)


def _make_project(name: str = "test-dataset") -> Dataset:
    """Minimal valid Dataset for mock returns."""
    return Dataset(
        id="proj-uuid-1",
        organization_id="org-uuid",
        name=name,
        description=None,
        annotation_types=["bbox"],
        classes=[],
        image_count=0,
        completed_image_count=0,
        total_size=0,
        archived_image_count=0,
        created_at="2026-04-19T00:00:00Z",  # type: ignore[arg-type]
    )


@pytest.fixture
def populated_directory(tmp_path: Path) -> Path:
    """Directory with two class subdirectories + a root-level image."""
    (tmp_path / "cars").mkdir()
    (tmp_path / "cars" / "car1.jpg").write_bytes(b"\xff\xd8\xff" + b"x" * 100)
    (tmp_path / "cars" / "car2.png").write_bytes(b"\x89PNG" + b"y" * 100)
    (tmp_path / "trucks").mkdir()
    (tmp_path / "trucks" / "truck1.jpg").write_bytes(b"\xff\xd8\xff" + b"z" * 100)
    (tmp_path / "root_image.jpg").write_bytes(b"\xff\xd8\xff" + b"a" * 100)
    # Non-image file should be ignored.
    (tmp_path / "notes.txt").write_text("not an image")
    return tmp_path


def _ok_client(project: Dataset | None = None) -> MagicMock:
    """Client mock where projects.get returns the dataset and uploads succeed."""
    client = MagicMock()
    client.datasets.get.return_value = project or _make_project()
    client.images.upload.return_value = MagicMock(id="img-uuid")
    return client


# ───────────── happy paths ─────────────


def test_upload_walks_directory_and_maps_class_directories(
    populated_directory: Path,
) -> None:
    """First-level subdirectory becomes the virtual_directory_path; root files land at /."""
    client = _ok_client()
    report = _invoke(
        client,
        "test-dataset",
        populated_directory,
        organize_by_class=True,
        parallel=False,
    )
    assert isinstance(report, UploadReport)
    assert report.images_attempted == 4
    assert report.images_uploaded == 4
    assert report.success

    # Check directory mapping in upload calls.
    call_kwargs = [c.kwargs for c in client.images.upload.call_args_list]
    directories_by_filename = {
        Path(kw["file_path"]).name: kw["directory_path"] for kw in call_kwargs
    }
    assert directories_by_filename["car1.jpg"] == "/cars"
    assert directories_by_filename["car2.png"] == "/cars"
    assert directories_by_filename["truck1.jpg"] == "/trucks"
    assert directories_by_filename["root_image.jpg"] == "/"


def test_upload_organize_by_class_false_flattens(populated_directory: Path) -> None:
    """All files land at root when organize_by_class=False."""
    client = _ok_client()
    _invoke(
        client,
        "test-dataset",
        populated_directory,
        organize_by_class=False,
        parallel=False,
    )
    directories = {c.kwargs["directory_path"] for c in client.images.upload.call_args_list}
    assert directories == {"/"}


# ───────────── directory layout ─────────────


@pytest.fixture
def nested_directory(tmp_path: Path) -> Path:
    """A tree deeper than one level, with the SAME basename in two subdirectories."""
    (tmp_path / "cars" / "red").mkdir(parents=True)
    (tmp_path / "cars" / "red" / "0001.jpg").write_bytes(b"\xff\xd8\xff" + b"a" * 10)
    (tmp_path / "trucks").mkdir()
    (tmp_path / "trucks" / "0001.jpg").write_bytes(b"\xff\xd8\xff" + b"b" * 10)
    return tmp_path


def test_preserve_structure_recreates_the_full_tree(nested_directory: Path) -> None:
    """`preserve_structure` keeps every level, matching the web app's directory upload.

    `organize_by_class` deliberately collapses deeper nesting onto the first level, which
    is right for ImageFolder-style class datasets but silently loses the tree for anyone
    who just wants their directories back.
    """
    client = _ok_client()
    _invoke(
        client,
        "test-dataset",
        nested_directory,
        preserve_structure=True,
        parallel=False,
    )
    call_kwargs = [c.kwargs for c in client.images.upload.call_args_list]
    directories = {Path(kw["file_path"]).parent.name: kw["directory_path"] for kw in call_kwargs}
    assert directories["red"] == "/cars/red"
    assert directories["trucks"] == "/trucks"


def test_preserve_structure_keeps_same_named_files_apart(nested_directory: Path) -> None:
    """Both 0001.jpg files upload, into different directories - neither is dropped."""
    client = _ok_client()
    report = _invoke(
        client,
        "test-dataset",
        nested_directory,
        preserve_structure=True,
        parallel=False,
    )
    assert report.images_uploaded == 2
    directories = sorted(c.kwargs["directory_path"] for c in client.images.upload.call_args_list)
    assert directories == ["/cars/red", "/trucks"]


def test_preserve_structure_takes_precedence_over_organize_by_class(
    nested_directory: Path,
) -> None:
    client = _ok_client()
    _invoke(
        client,
        "test-dataset",
        nested_directory,
        organize_by_class=True,
        preserve_structure=True,
        parallel=False,
    )
    directories = {c.kwargs["directory_path"] for c in client.images.upload.call_args_list}
    assert directories == {"/cars/red", "/trucks"}


def test_organize_by_class_still_collapses_deep_nesting(nested_directory: Path) -> None:
    """Regression guard for the existing default: cars/red/x.jpg -> /cars, not /cars/red."""
    client = _ok_client()
    _invoke(
        client,
        "test-dataset",
        nested_directory,
        organize_by_class=True,
        parallel=False,
    )
    directories = {c.kwargs["directory_path"] for c in client.images.upload.call_args_list}
    assert directories == {"/cars", "/trucks"}


@pytest.mark.parametrize(
    ("relative", "organize", "preserve", "expected"),
    [
        (Path("x.jpg"), True, False, "/"),
        (Path("x.jpg"), False, True, "/"),
        (Path("cars/x.jpg"), True, False, "/cars"),
        (Path("cars/red/x.jpg"), True, False, "/cars"),
        (Path("cars/red/x.jpg"), False, True, "/cars/red"),
        (Path("cars/red/night/x.jpg"), False, True, "/cars/red/night"),
        (Path("cars/red/x.jpg"), False, False, "/"),
    ],
)
def test_virtual_directory_for(
    relative: Path, organize: bool, preserve: bool, expected: str
) -> None:
    """The one helper both pipelines share - sync and async can't drift apart."""
    assert (
        virtual_directory_for(relative, organize_by_class=organize, preserve_structure=preserve)
        == expected
    )


def test_upload_skips_non_image_files(populated_directory: Path) -> None:
    """Files outside the supported extension list are not uploaded."""
    client = _ok_client()
    report = _invoke(client, "test-dataset", populated_directory, parallel=False)
    # 4 images, the .txt is skipped.
    assert report.images_attempted == 4
    assert client.images.upload.call_count == 4


# ───────────── parallelism ─────────────


def test_upload_parallel_pool_size(populated_directory: Path) -> None:
    """parallel=True with explicit max_workers fans out via thread pool."""
    client = _ok_client()
    report = _invoke(
        client,
        "test-dataset",
        populated_directory,
        parallel=True,
        max_workers=4,
    )
    assert report.images_uploaded == 4
    assert client.images.upload.call_count == 4


# ───────────── dataset creation ─────────────


def test_upload_creates_dataset_when_missing(tmp_path: Path) -> None:
    """If projects.get() raises NotFoundError, projects.create() is called."""
    (tmp_path / "img.jpg").write_bytes(b"\xff\xd8\xff" + b"x" * 100)
    client = MagicMock()
    client.datasets.get.side_effect = NotFoundError("Dataset 'new' not found")
    client.datasets.create.return_value = _make_project("new")
    client.images.upload.return_value = MagicMock(id="img-uuid")

    _invoke(client, "new", tmp_path, parallel=False)
    client.datasets.create.assert_called_once_with("new")


def test_upload_raises_when_missing_and_create_disabled(tmp_path: Path) -> None:
    """create_if_missing=False propagates the NotFoundError."""
    (tmp_path / "img.jpg").write_bytes(b"\xff\xd8\xff" + b"x" * 100)
    client = MagicMock()
    client.datasets.get.side_effect = NotFoundError("missing")

    with pytest.raises(NotFoundError):
        _invoke(client, "missing", tmp_path, create_if_missing=False)


# ───────────── failure handling ─────────────


def test_upload_conflict_skipped_by_default(tmp_path: Path) -> None:
    """ConflictError is reported as 'skipped' when skip_existing=True (default)."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff" + b"x" * 100)
    (tmp_path / "b.jpg").write_bytes(b"\xff\xd8\xff" + b"y" * 100)
    client = _ok_client()
    client.images.upload.side_effect = [
        ConflictError("a.jpg already exists"),
        MagicMock(id="img-b"),
    ]
    report = _invoke(client, "test-dataset", tmp_path, parallel=False)
    assert report.images_uploaded == 1
    assert report.images_skipped == 1
    assert report.failures == []
    assert report.success  # at least one succeeded, no hard failures


def test_upload_conflict_recorded_as_failure_when_skip_disabled(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff" + b"x" * 100)
    client = _ok_client()
    client.images.upload.side_effect = ConflictError("a.jpg already exists")

    report = _invoke(
        client,
        "test-dataset",
        tmp_path,
        parallel=False,
        skip_existing=False,
    )
    assert report.images_uploaded == 0
    assert len(report.failures) == 1
    assert isinstance(report.failures[0], UploadFailure)
    assert "conflict" in report.failures[0].reason


def test_upload_api_error_recorded_as_failure(tmp_path: Path) -> None:
    """Generic ApiError on a single image lands in failures, not raised."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff" + b"x" * 100)
    (tmp_path / "b.jpg").write_bytes(b"\xff\xd8\xff" + b"y" * 100)
    client = _ok_client()
    client.images.upload.side_effect = [
        ApiError("upload exploded"),
        MagicMock(id="img-b"),
    ]
    report = _invoke(client, "test-dataset", tmp_path, parallel=False)
    assert report.images_uploaded == 1
    assert len(report.failures) == 1
    assert "upload exploded" in report.failures[0].reason


def test_upload_network_error_recorded_not_raised(tmp_path: Path) -> None:
    """A transient NetworkError on ONE image (e.g. from the GCS PUT phase) is
    recorded as a per-file failure and the rest of the batch continues - it must
    NOT propagate and abort the whole upload (the report contract is "partial
    success doesn't raise"). NetworkError/RequestTimeoutError are siblings of
    ApiError under PictographError, so the old `except ApiError` missed them."""
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff" + b"x" * 100)
    (tmp_path / "b.jpg").write_bytes(b"\xff\xd8\xff" + b"y" * 100)
    (tmp_path / "c.jpg").write_bytes(b"\xff\xd8\xff" + b"z" * 100)
    client = _ok_client()
    client.images.upload.side_effect = [
        NetworkError("connection reset during GCS upload"),
        RequestTimeoutError("read timed out"),
        MagicMock(id="img-c"),
    ]
    report = _invoke(client, "test-dataset", tmp_path, parallel=False)
    assert report.images_uploaded == 1  # the third image still lands
    assert len(report.failures) == 2
    reasons = " ".join(f.reason for f in report.failures)
    assert "connection reset" in reasons
    assert "timed out" in reasons


# ───────────── input validation ─────────────


def test_upload_raises_on_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _invoke(MagicMock(), "x", tmp_path / "does-not-exist")


def test_upload_empty_directory_returns_empty_report(tmp_path: Path) -> None:
    client = _ok_client()
    report = _invoke(client, "test-dataset", tmp_path)
    assert report.images_attempted == 0
    assert report.images_uploaded == 0
    assert report.failures == []
    # Don't claim success on empty input - there's nothing to succeed at.
    assert not report.success


# Suppress unused-import warning for the auxiliary class - used for type
# checking in the test fixtures only.
_ = DatasetClass
