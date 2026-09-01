"""Live: annotations get / save / delete with canonical Pictograph JSON."""

from __future__ import annotations

import pytest

from pictograph import Client
from pictograph.models.annotation import (
    BBoxAnnotation,
    PolygonAnnotation,
    PolygonGeometry,
)
from pictograph.models.common import BoundingBox, Point

pytestmark = pytest.mark.live


def test_get_empty_returns_empty_list(client: Client, scratch_dataset_with_images) -> None:
    scratch_ds, images = scratch_dataset_with_images
    anns = client.annotations.get(scratch_ds.name, images[0].id)
    assert anns == []


def test_save_bbox_round_trip(client: Client, scratch_dataset_with_images) -> None:
    scratch_ds, images = scratch_dataset_with_images
    img = images[0]
    bbox = BBoxAnnotation(
        name="thing",
        bounding_box=BoundingBox(x=10, y=20, w=40, h=60),
    )
    result = client.annotations.save(scratch_ds.name, img.id, [bbox])
    assert result.image_id == img.id
    assert result.new_count == 1
    assert result.previous_count == 0

    fetched = client.annotations.get(scratch_ds.name, img.id)
    assert len(fetched) == 1
    ann = fetched[0]
    assert isinstance(ann, BBoxAnnotation)
    assert ann.name == "thing"
    assert ann.bounding_box.x == 10
    assert ann.bounding_box.w == 40


def test_save_polygon_round_trip(client: Client, scratch_dataset_with_images) -> None:
    scratch_ds, images = scratch_dataset_with_images
    img = images[0]
    poly = PolygonAnnotation(
        name="shape",
        polygon=PolygonGeometry(paths=[[Point(x=0, y=0), Point(x=50, y=0), Point(x=25, y=50)]]),
    )
    client.annotations.save(scratch_ds.name, img.id, [poly])
    anns = client.annotations.get(scratch_ds.name, img.id)
    assert len(anns) == 1
    assert isinstance(anns[0], PolygonAnnotation)
    assert anns[0].name == "shape"
    assert len(anns[0].polygon.paths[0]) == 3


def test_save_replaces_not_merges(client: Client, scratch_dataset_with_images) -> None:
    scratch_ds, images = scratch_dataset_with_images
    img = images[0]
    first = BBoxAnnotation(name="thing", bounding_box=BoundingBox(x=0, y=0, w=10, h=10))
    second = BBoxAnnotation(name="thing", bounding_box=BoundingBox(x=20, y=20, w=10, h=10))
    client.annotations.save(scratch_ds.name, img.id, [first])
    result = client.annotations.save(scratch_ds.name, img.id, [second])
    assert result.previous_count == 1
    assert result.new_count == 1
    anns = client.annotations.get(scratch_ds.name, img.id)
    assert len(anns) == 1
    assert anns[0].bounding_box.x == 20


def test_save_empty_list_clears(client: Client, scratch_dataset_with_images) -> None:
    scratch_ds, images = scratch_dataset_with_images
    img = images[0]
    client.annotations.save(
        scratch_ds.name,
        img.id,
        [BBoxAnnotation(name="thing", bounding_box=BoundingBox(x=0, y=0, w=5, h=5))],
    )
    cleared = client.annotations.save(scratch_ds.name, img.id, [])
    assert cleared.new_count == 0
    assert client.annotations.get(scratch_ds.name, img.id) == []


def test_delete_round_trip(client: Client, scratch_dataset_with_images) -> None:
    scratch_ds, images = scratch_dataset_with_images
    img = images[0]
    client.annotations.save(
        scratch_ds.name,
        img.id,
        [BBoxAnnotation(name="thing", bounding_box=BoundingBox(x=0, y=0, w=5, h=5))],
    )
    result = client.annotations.delete(scratch_ds.name, img.id)
    assert result.image_id == img.id
    assert client.annotations.get(scratch_ds.name, img.id) == []
