"""Tests for ``AsyncImages.upload_from_directory``.

Mocks the AsyncClient resources (``projects``/``images`` as AsyncMocks) and
exercises the orchestration: directory discovery, virtual-directory mapping, concurrent
upload, skip-existing conflict handling, create-if-missing, and failure reporting.
Inherits ``anyio_backend`` from ``tests/unit/aio/conftest.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from pictograph.aio.resources.images import AsyncImages
from pictograph.exceptions import ApiError, ConflictError, NotFoundError
from pictograph.models.dataset import Dataset

if TYPE_CHECKING:
    from pathlib import Path

from tests.unit.resources._orchestration import build, sibling_resources


async def _invoke(client: MagicMock, *args: object, **kwargs: object) -> object:
    """Invoke the real method on a real async resource with its siblings stubbed."""
    with sibling_resources(client, is_async=True):
        resource = build(AsyncImages, client, own="images", delegate=["upload"])
        return await resource.upload_from_directory(*args, **kwargs)


pytestmark = pytest.mark.anyio


def _project() -> Dataset:
    return Dataset(
        id="proj-1",
        name="road-signs",
        organization_id="org-1",
        classes=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _client(*, upload: AsyncMock | None = None, get_side_effect: object = None) -> MagicMock:
    client = MagicMock()
    if get_side_effect is not None:
        client.datasets.get = AsyncMock(side_effect=get_side_effect)
    else:
        client.datasets.get = AsyncMock(return_value=_project())
    client.datasets.create = AsyncMock(return_value=_project())
    client.images.upload = upload or AsyncMock()
    return client


def _seed(root: Path, names: list[str]) -> None:
    for name in names:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\xff\xd8\xff")


async def test_upload_happy_path_concurrent(tmp_path: Path) -> None:
    _seed(tmp_path, ["a.jpg", "b.png", "notes.txt"])  # .txt ignored
    client = _client()
    report = await _invoke(client, "road-signs", tmp_path)
    assert report.images_attempted == 2
    assert report.images_uploaded == 2
    assert report.success
    assert client.images.upload.await_count == 2


async def test_upload_organizes_by_class(tmp_path: Path) -> None:
    _seed(tmp_path, ["cars/a.jpg", "signs/b.jpg", "root.jpg"])
    client = _client()
    await _invoke(client, "road-signs", tmp_path, organize_by_class=True)
    directories = {c.kwargs["directory_path"] for c in client.images.upload.await_args_list}
    assert directories == {"/cars", "/signs", "/"}


async def test_upload_flat_when_organize_disabled(tmp_path: Path) -> None:
    _seed(tmp_path, ["cars/a.jpg"])
    client = _client()
    await _invoke(client, "road-signs", tmp_path, organize_by_class=False)
    assert client.images.upload.await_args.kwargs["directory_path"] == "/"


async def test_upload_skips_existing_conflict(tmp_path: Path) -> None:
    _seed(tmp_path, ["a.jpg"])
    client = _client(upload=AsyncMock(side_effect=ConflictError("dup", status_code=409)))
    report = await _invoke(client, "road-signs", tmp_path, skip_existing=True)
    assert report.images_skipped == 1
    assert report.failures == []


async def test_upload_conflict_as_failure_when_not_skipping(tmp_path: Path) -> None:
    _seed(tmp_path, ["a.jpg"])
    client = _client(upload=AsyncMock(side_effect=ConflictError("dup", status_code=409)))
    report = await _invoke(client, "road-signs", tmp_path, skip_existing=False)
    assert report.images_uploaded == 0
    assert len(report.failures) == 1 and "conflict" in report.failures[0].reason


async def test_upload_400_already_exists_treated_as_skip(tmp_path: Path) -> None:
    _seed(tmp_path, ["a.jpg"])
    client = _client(upload=AsyncMock(side_effect=ApiError("already exists", status_code=400)))
    report = await _invoke(client, "road-signs", tmp_path, skip_existing=True)
    assert report.images_skipped == 1


async def test_upload_creates_dataset_if_missing(tmp_path: Path) -> None:
    _seed(tmp_path, ["a.jpg"])
    client = _client(get_side_effect=NotFoundError("no dataset", status_code=404))
    await _invoke(client, "new-set", tmp_path, create_if_missing=True)
    client.datasets.create.assert_awaited_once()


async def test_upload_raises_when_missing_and_no_create(tmp_path: Path) -> None:
    _seed(tmp_path, ["a.jpg"])
    client = _client(get_side_effect=NotFoundError("no dataset", status_code=404))
    with pytest.raises(NotFoundError):
        await _invoke(client, "ghost", tmp_path, create_if_missing=False)


async def test_upload_empty_directory(tmp_path: Path) -> None:
    client = _client()
    report = await _invoke(client, "road-signs", tmp_path)
    assert report.images_attempted == 0 and not report.success


async def test_upload_missing_directory_raises() -> None:
    client = _client()
    with pytest.raises(FileNotFoundError):
        await _invoke(client, "road-signs", "/nonexistent/path/xyz")


async def test_upload_per_file_failure_reported(tmp_path: Path) -> None:
    _seed(tmp_path, ["a.jpg"])
    client = _client(upload=AsyncMock(side_effect=ApiError("server boom", status_code=500)))
    report = await _invoke(client, "road-signs", tmp_path)
    assert report.images_uploaded == 0
    assert len(report.failures) == 1 and "boom" in report.failures[0].reason


async def test_upload_progress_callback(tmp_path: Path) -> None:
    _seed(tmp_path, ["a.jpg", "b.jpg"])
    client = _client()
    seen: list[tuple[int, int, str | None]] = []
    await _invoke(client, "road-signs", tmp_path, progress=lambda c, t, f: seen.append((c, t, f)))
    assert len(seen) == 2 and seen[-1][0] == 2 and seen[-1][1] == 2
