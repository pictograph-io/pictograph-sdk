"""Tests for ``pictograph.resources.batch.Batch``.

Coverage targets:
- ``move``: body shape, NotFound, Forbidden, partial-success report.
- ``copy``: body shape (duplicate handling, copy_annotations), failures
  surface in BatchResult.failed.
- ``delete``: soft (default) vs permanent body shape; 403 on permanent
  without admin role.
- ``update``: body shape with partial fields; client-side ValueError
  when no fields supplied; 400 invalid status.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import (
    ForbiddenError,
    NotFoundError,
    ValidationError,
)
from pictograph.models.batch import BatchFailure, BatchResult
from pictograph.resources.batch import Batch

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
def batch(transport: Transport) -> Batch:
    return Batch(transport)


def _ok_response(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "success": True,
        "processed": 0,
        "failed": [],
        "affected_directories": [],
    }
    base.update(overrides)
    return base


# ───────────── move ─────────────


def test_move_body_shape(httpx_mock: HTTPXMock, batch: Batch) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/batch/images/move",
        json=_ok_response(
            processed=3,
            affected_directories=["/old", "/new"],
        ),
    )
    result = batch.move(
        "road-signs",
        ["img-1", "img-2", "img-3"],
        target_directory_path="/new",
    )
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {
        "dataset_name": "road-signs",
        "image_ids": ["img-1", "img-2", "img-3"],
        "target_directory_path": "/new",
    }
    assert isinstance(result, BatchResult)
    assert result.processed == 3
    assert result.affected_directories == ["/old", "/new"]
    # Absent in this response → default 0 (a no-collision move).
    assert result.renamed == 0


def test_move_reports_renamed_collisions(httpx_mock: HTTPXMock, batch: Batch) -> None:
    """A move into a directory that already holds same-named images auto-renames the
    collisions "-{n}" server-side; the count surfaces on BatchResult.renamed."""
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/batch/images/move",
        json=_ok_response(processed=5, renamed=2, affected_directories=["/src", "/"]),
    )
    result = batch.move("road-signs", ["a", "b", "c", "d", "e"])
    assert result.processed == 5
    assert result.renamed == 2


def test_move_default_root_directory(httpx_mock: HTTPXMock, batch: Batch) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/batch/images/move",
        json=_ok_response(processed=1),
    )
    batch.move("road-signs", ["img-1"])
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body["target_directory_path"] == "/"


def test_move_404_dataset_not_found(httpx_mock: HTTPXMock, batch: Batch) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/batch/images/move",
        status_code=404,
        json={"detail": "Dataset 'missing' not found"},
    )
    with pytest.raises(NotFoundError):
        batch.move("missing", ["img-1"])


def test_move_403_no_write_role(httpx_mock: HTTPXMock, batch: Batch) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/batch/images/move",
        status_code=403,
        json={"detail": "Insufficient permissions for batch image operations."},
    )
    with pytest.raises(ForbiddenError):
        batch.move("road-signs", ["img-1"])


# ───────────── copy ─────────────


def test_copy_body_shape_with_all_options(httpx_mock: HTTPXMock, batch: Batch) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/batch/images/copy",
        json=_ok_response(processed=2, affected_directories=["/copies"]),
    )
    batch.copy(
        "road-signs",
        ["img-1", "img-2"],
        target_directory_path="/copies",
        duplicate_handling="overwrite",
        copy_annotations=True,
    )
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {
        "dataset_name": "road-signs",
        "image_ids": ["img-1", "img-2"],
        "target_directory_path": "/copies",
        "duplicate_handling": "overwrite",
        "copy_annotations": True,
    }


def test_copy_defaults(httpx_mock: HTTPXMock, batch: Batch) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/batch/images/copy",
        json=_ok_response(),
    )
    batch.copy("road-signs", ["img-1"])
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body["duplicate_handling"] == "rename"
    assert body["copy_annotations"] is False


def test_copy_partial_failure_surfaces_in_result(httpx_mock: HTTPXMock, batch: Batch) -> None:
    """One image failed, one succeeded - call doesn't raise; details in .failed."""
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/batch/images/copy",
        json=_ok_response(
            processed=1,
            failed=[{"id": "img-bad", "reason": "Filename 'bad.png' already exists (skipped)"}],
        ),
    )
    result = batch.copy(
        "road-signs",
        ["img-good", "img-bad"],
        duplicate_handling="skip",
    )
    assert result.processed == 1
    assert len(result.failed) == 1
    assert isinstance(result.failed[0], BatchFailure)
    assert result.failed[0].id == "img-bad"


# ───────────── delete ─────────────


def test_delete_default_is_soft_archive(httpx_mock: HTTPXMock, batch: Batch) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/batch/images/delete",
        json=_ok_response(processed=2),
    )
    batch.delete("road-signs", ["img-1", "img-2"])
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body["permanent"] is False


def test_delete_permanent_body_shape(httpx_mock: HTTPXMock, batch: Batch) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/batch/images/delete",
        json=_ok_response(processed=1),
    )
    batch.delete("road-signs", ["img-1"], permanent=True)
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body["permanent"] is True


def test_delete_403_permanent_without_admin(httpx_mock: HTTPXMock, batch: Batch) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/batch/images/delete",
        status_code=403,
        json={"detail": "Only admin or owner roles can permanently delete images."},
    )
    with pytest.raises(ForbiddenError, match="permanently delete"):
        batch.delete("road-signs", ["img-1"], permanent=True)


# ───────────── update ─────────────


def test_update_status(httpx_mock: HTTPXMock, batch: Batch) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE}/api/v1/developer/batch/images/update",
        json=_ok_response(processed=2),
    )
    batch.update("road-signs", ["img-1", "img-2"], status="complete")
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {
        "dataset_name": "road-signs",
        "image_ids": ["img-1", "img-2"],
        "updates": {"status": "complete"},
    }


def test_update_multiple_fields(httpx_mock: HTTPXMock, batch: Batch) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE}/api/v1/developer/batch/images/update",
        json=_ok_response(processed=1),
    )
    batch.update(
        "road-signs",
        ["img-1"],
        status="review",
        display_name="Stop Sign Verified",
        is_archived=False,
    )
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body["updates"] == {
        "status": "review",
        "display_name": "Stop Sign Verified",
        "is_archived": False,
    }


def test_update_omits_none_fields(httpx_mock: HTTPXMock, batch: Batch) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE}/api/v1/developer/batch/images/update",
        json=_ok_response(processed=1),
    )
    batch.update("road-signs", ["img-1"], is_archived=True)
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body["updates"] == {"is_archived": True}


def test_update_requires_at_least_one_field(batch: Batch) -> None:
    """Empty update is a programmer error - fail fast client-side."""
    with pytest.raises(ValueError, match="At least one of"):
        batch.update("road-signs", ["img-1"])


def test_update_400_invalid_status(httpx_mock: HTTPXMock, batch: Batch) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE}/api/v1/developer/batch/images/update",
        status_code=400,
        json={
            "detail": ("Invalid status. Must be one of: ['new', 'annotate', 'review', 'complete']")
        },
    )
    with pytest.raises(ValidationError, match="Invalid status"):
        batch.update("road-signs", ["img-1"], status="bogus")
