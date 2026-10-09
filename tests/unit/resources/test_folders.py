"""Tests for the read-only Directories resource (list / tree / stats)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.models.directory import Directory, DirectoryStats, DirectoryTreeNode
from pictograph.resources.directories import Directories

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
_PATH = f"{BASE}/api/v1/developer/directories"
PROJECT = "road-signs"


@pytest.fixture
def transport() -> Transport:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def directories(transport: Transport) -> Directories:
    return Directories(transport)


def _directory(**o: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "f1",
        "project_id": PROJECT,
        "organization_id": "org-1",
        "name": "train",
        "parent_directory_id": None,
        "full_path": "/train",
        "image_count": 3,
    }
    base.update(o)
    return base


def test_list_returns_typed(httpx_mock: HTTPXMock, directories: Directories) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}/{PROJECT}",
        json=[_directory(), _directory(id="f2", full_path="/val", name="val")],
    )
    result = directories.list(PROJECT)
    assert len(result) == 2
    assert all(isinstance(f, Directory) for f in result)
    assert result[0].full_path == "/train"


def test_list_passes_parent_path(httpx_mock: HTTPXMock, directories: Directories) -> None:
    httpx_mock.add_response(method="GET", url=f"{_PATH}/{PROJECT}?parent_path=%2Ftrain", json=[])
    assert directories.list(PROJECT, parent_path="/train") == []


def test_tree_nests_children(httpx_mock: HTTPXMock, directories: Directories) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}/{PROJECT}/tree",
        json=[
            {
                "id": "root",
                "name": "data",
                "full_path": "/data",
                "image_count": 0,
                "children": [
                    {
                        "id": "c",
                        "name": "train",
                        "full_path": "/data/train",
                        "image_count": 5,
                        "children": [],
                    }
                ],
            }
        ],
    )
    tree = directories.tree(PROJECT)
    assert len(tree) == 1 and isinstance(tree[0], DirectoryTreeNode)
    assert tree[0].children[0].full_path == "/data/train"


def test_stats_typed(httpx_mock: HTTPXMock, directories: Directories) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}/road-signs/train/stats?include_subdirectories=true",
        json={
            "total_directories": 2,
            "total_images": 10,
            "total_size_bytes": 2048,
            "directories_by_status": {"complete": 7, "new": 3},
        },
    )
    stats = directories.stats("road-signs", "/train")
    assert isinstance(stats, DirectoryStats)
    assert stats.total_images == 10
    assert stats.directories_by_status["complete"] == 7


def test_stats_exclude_subdirectories(httpx_mock: HTTPXMock, directories: Directories) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{_PATH}/road-signs/train/stats?include_subdirectories=false",
        json={
            "total_directories": 1,
            "total_images": 4,
            "total_size_bytes": 100,
            "directories_by_status": {},
        },
    )
    stats = directories.stats(
        "road-signs",
        "/train",
        include_subdirectories=False,
    )
    assert stats.total_directories == 1


def test_delete_sends_cascade_true(httpx_mock: HTTPXMock, directories: Directories) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{_PATH}/road-signs/train?cascade=true",
        json={"success": True, "directory_id": "f-1", "images_moved": 2},
    )
    assert (
        directories.delete(
            "road-signs",
            "/train",
            cascade=True,
        )
        is None
    )


def test_delete_defaults_no_cascade(httpx_mock: HTTPXMock, directories: Directories) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{_PATH}/road-signs/train?cascade=false",
        json={"success": True, "directory_id": "f-1", "images_moved": 0},
    )
    directories.delete("road-signs", "/train")
    # The DEFAULT is the whole point of this test, and it was expressed only by the
    # `?cascade=false` in the registered URL above. Say it outright: a delete that
    # silently cascades destroys the images in the directory.
    req = httpx_mock.get_requests()[0]
    assert req.method == "DELETE"
    assert req.url.params["cascade"] == "false"


def test_create_parses_canonical_envelope(httpx_mock: HTTPXMock, directories: Directories) -> None:
    import json

    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/directories/",
        json={
            "data": {
                "id": "fold-1",
                "dataset_id": "road-signs",
                "name": "positive",
                "directory_path": "/train/positive",
                "parent_directory_id": "fold-0",
                "image_count": 0,
                "created_at": "2026-07-10T00:00:00Z",
            }
        },
    )
    directory = directories.create("road-signs", "/train/positive")
    # Canonical wire names parse onto the model's attrs via aliases.
    assert directory.full_path == "/train/positive"
    assert directory.dataset_id == "road-signs"
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {
        "dataset": "road-signs",
        "directory_path": "/train/positive",
    }


def test_rename_patches_and_parses(httpx_mock: HTTPXMock, directories: Directories) -> None:
    import json

    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE}/api/v1/developer/directories/road-signs/train/rename",
        json={
            "data": {
                "id": "fold-1",
                "dataset_id": "road-signs",
                "name": "negatives",
                "directory_path": "/train/negatives",
                "parent_directory_id": "fold-0",
                "image_count": 7,
                "created_at": "2026-07-10T00:00:00Z",
            }
        },
    )
    directory = directories.rename(
        "road-signs",
        "/train",
        "negatives",
    )
    assert directory.name == "negatives" and directory.full_path == "/train/negatives"
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {"new_name": "negatives"}


def test_rename_conflict_propagates(httpx_mock: HTTPXMock, directories: Directories) -> None:
    from pictograph.exceptions import ConflictError

    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE}/api/v1/developer/directories/road-signs/train/rename",
        status_code=409,
        json={"detail": "Directory '/train/negatives' already exists"},
    )
    with pytest.raises(ConflictError):
        directories.rename(
            "road-signs",
            "/train",
            "negatives",
        )
