"""Live: exports - create / list / get / download / delete."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from pictograph import Client
from pictograph.models.annotation import BBoxAnnotation
from pictograph.models.common import BoundingBox
from pictograph.models.export import Export

pytestmark = pytest.mark.live


@pytest.fixture
def dataset_with_annotation(client: Client, scratch_dataset_with_images):
    """Annotate the first image so exports have something to include."""
    project, images = scratch_dataset_with_images
    client.annotations.save(
        project.name,
        images[0].id,
        [BBoxAnnotation(name="thing", bounding_box=BoundingBox(x=0, y=0, w=50, h=50))],
    )
    return project, images


@pytest.mark.parametrize("fmt", ["pictograph", "coco", "yolo"])
def test_create_export(client: Client, dataset_with_annotation, unique_name: str, fmt: str) -> None:
    project, _ = dataset_with_annotation
    export_name = f"{unique_name}-{fmt}"
    export = client.exports.create(
        project.name,
        export_name,
        format=fmt,
        wait=True,
        poll_interval=2,
        timeout=180,
    )
    try:
        assert isinstance(export, Export)
        assert export.status == "completed"
        assert export.format == fmt
    finally:
        try:
            client.exports.delete(project.name, export_name)
        except Exception:
            pass


def test_export_list(client: Client, dataset_with_annotation, unique_name: str) -> None:
    project, _ = dataset_with_annotation
    client.exports.create(project.name, unique_name, format="pictograph", wait=True, timeout=180)
    try:
        exports = client.exports.list(dataset_name=project.name)
        names = {e.name for e in exports}
        assert unique_name in names
    finally:
        client.exports.delete(project.name, unique_name)


def test_export_get_and_download(
    client: Client, dataset_with_annotation, unique_name: str, tmp_path: Path
) -> None:
    project, _ = dataset_with_annotation
    export = client.exports.create(
        project.name, unique_name, format="pictograph", wait=True, timeout=180
    )
    try:
        fetched = client.exports.get(project.name, unique_name)
        assert fetched.name == unique_name
        assert fetched.status == "completed"

        out = tmp_path / "export.zip"
        client.exports.download(project.name, unique_name, out)
        assert out.is_file() and out.stat().st_size > 0
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            assert any(n.endswith(".json") for n in names)
    finally:
        client.exports.delete(project.name, unique_name)


def test_export_pending_when_not_wait(
    client: Client, dataset_with_annotation, unique_name: str
) -> None:
    project, _ = dataset_with_annotation
    export = client.exports.create(project.name, unique_name, format="pictograph", wait=False)
    assert export.status in {"pending", "processing", "completed"}
    try:
        client.exports.wait_for_completion(project.name, unique_name, poll_interval=2, timeout=180)
    finally:
        client.exports.delete(project.name, unique_name)
