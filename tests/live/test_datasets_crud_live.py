"""Live: dataset CRUD + class config replacement (the merged dataset surface)."""

from __future__ import annotations

import pytest

from pictograph import Client
from pictograph.exceptions import ConflictError, NotFoundError
from pictograph.models.dataset import Dataset, DatasetClass

pytestmark = pytest.mark.live


def test_create_get_delete(client: Client, unique_name: str) -> None:
    proj = client.datasets.create(
        unique_name,
        description="sdk test",
        annotation_types=["bbox"],
        classes=[{"name": "widget", "type": "bbox", "color": "#e6194b"}],
    )
    try:
        assert isinstance(proj, Dataset)
        assert proj.name == unique_name
        assert proj.description == "sdk test"

        fetched = client.datasets.get(unique_name)
        assert fetched.id == proj.id
        assert fetched.classes and fetched.classes[0].name == "widget"
    finally:
        client.datasets.delete(unique_name)


def test_get_missing_raises_not_found(client: Client) -> None:
    with pytest.raises(NotFoundError):
        client.datasets.get("this-project-definitely-does-not-exist-xyz-123")


def test_create_duplicate_raises_conflict(client: Client, scratch_project: Dataset) -> None:
    with pytest.raises(ConflictError):
        client.datasets.create(scratch_project.name)


def test_update_description_and_classes(client: Client, scratch_project: Dataset) -> None:
    updated = client.datasets.update(
        scratch_project.name,
        description="updated description",
        classes=[
            DatasetClass(name="cat", type="bbox", color="#ff0000"),
            DatasetClass(name="dog", type="polygon", color="#00ff00"),
        ],
    )
    assert updated.description == "updated description"
    names = {c.name for c in (updated.classes or [])}
    assert names == {"cat", "dog"}


def test_update_rename(client: Client, unique_name: str) -> None:
    proj = client.datasets.create(unique_name)
    new_name = unique_name + "-renamed"
    try:
        updated = client.datasets.update(unique_name, new_name=new_name)
        assert updated.name == new_name
        client.datasets.get(new_name)  # confirms server-side rename
    finally:
        try:
            client.datasets.delete(new_name)
        except NotFoundError:
            client.datasets.delete(unique_name)


def test_update_empty_body_raises_value_error(client: Client, scratch_project: Dataset) -> None:
    with pytest.raises(ValueError):
        client.datasets.update(scratch_project.name)


def test_list_contains_created_project(client: Client, scratch_project: Dataset) -> None:
    names = {p.name for p in client.datasets.list(limit=200)}
    assert scratch_project.name in names


def test_iter_yields_projects(client: Client, scratch_project: Dataset) -> None:
    pager = client.datasets.iter(page_size=25, max_total=50)
    hits = [p for p in pager if p.name == scratch_project.name]
    assert len(hits) == 1
