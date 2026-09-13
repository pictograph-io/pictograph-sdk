"""Live: datasets list / get (by name and by id) / download."""

from __future__ import annotations

from pathlib import Path

import pytest

from pictograph import Client
from pictograph.exceptions import NotFoundError
from pictograph.models.dataset import Dataset
from pictograph.resources.datasets import DownloadReport

pytestmark = pytest.mark.live


def test_list_returns_datasets(client: Client) -> None:
    ds_list = client.datasets.list(limit=5)
    assert isinstance(ds_list, list)
    for d in ds_list:
        assert isinstance(d, Dataset)


def test_iter_respects_max_total(client: Client) -> None:
    pager = client.datasets.iter(page_size=2, max_total=3)
    items = pager.all()
    assert len(items) <= 3


def test_get_by_name_round_trip(client: Client, scratch_project) -> None:
    fetched = client.datasets.get(scratch_project.name)
    assert fetched.id == scratch_project.id
    assert fetched.name == scratch_project.name


def test_get_by_id(client: Client, scratch_project) -> None:
    """The id escape hatch. `get_by_id` was folded into `get(dataset_id=...)`
    when datasets moved to name-addressing; this test still called the removed
    method, so the live suite had been red on it."""
    fetched = client.datasets.get(dataset_id=scratch_project.id)
    assert fetched.name == scratch_project.name


def test_get_missing_name_raises(client: Client) -> None:
    with pytest.raises(NotFoundError):
        client.datasets.get("definitely-missing-dataset-xyz-2026")


def test_get_with_images(client: Client, scratch_dataset_with_images) -> None:
    project, images = scratch_dataset_with_images
    ds = client.datasets.get(project.name, include_images=True, images_limit=100)
    assert ds.images is not None
    assert len(ds.images) == len(images)


def test_download_annotations_only(
    client: Client, scratch_dataset_with_images, tmp_path: Path
) -> None:
    project, _images = scratch_dataset_with_images
    report = client.datasets.download(project.name, tmp_path / "dl", mode="annotations_only")
    assert isinstance(report, DownloadReport)
    # No annotations on fresh uploads, but the call should succeed.
    assert report.dataset_id == project.id
