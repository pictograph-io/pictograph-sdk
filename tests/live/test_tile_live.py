"""Live: dataset tiling (``client.images.tile``).

Exercises the full round-trip against the real API - annotate a scratch dataset,
tile it into a new dataset (and into the source's directory), and assert the
produced tiles + remapped annotations. The target dataset is cleaned up. Gated
by ``PICTOGRAPH_TEST_KEY`` like every live test.
"""

from __future__ import annotations

import contextlib

import pytest

from pictograph import Client
from pictograph.models.annotation import BBoxAnnotation
from pictograph.models.common import BoundingBox

pytestmark = pytest.mark.live


def _annotate(client: Client, dataset_name: str, images) -> None:
    # A box in the top-left region of each image (survives into at least one tile).
    for img in images:
        client.annotations.save(
            dataset_name,
            img.id,
            [BBoxAnnotation(name="obj", bounding_box=BoundingBox(x=5, y=5, w=20, h=20))],
        )


def test_tile_into_new_dataset(client: Client, scratch_dataset_with_images, unique_name) -> None:
    source, images = scratch_dataset_with_images
    assert images, "fixture provided no images"
    _annotate(client, source.name, images)
    target = f"{unique_name}-tiled"
    try:
        report = client.images.tile(source.name, rows=2, cols=2, into=target)
        assert report.success, report.failures
        assert report.source_images == len(images)
        assert report.tiles_created == len(images) * 4  # 2x2 grid

        out = client.datasets.get(target)
        out_imgs = list(client.images.iter(out.id))
        assert len(out_imgs) == len(images) * 4

        # every tile filename encodes its grid cell
        assert any("_tile_r0_c0_" in i.filename for i in out_imgs)
        # the annotated tile carries the remapped box
        tile = next(i for i in out_imgs if "_tile_r0_c0_" in i.filename)
        anns = client.annotations.get(target, tile.id)
        assert len(anns) >= 1
        assert anns[0].name == "obj"
    finally:
        with contextlib.suppress(Exception):
            client.datasets.delete(target)


def test_tile_in_place_appends_to_directory(client: Client, scratch_dataset_with_images) -> None:
    source, images = scratch_dataset_with_images
    _annotate(client, source.name, images)
    report = client.images.tile(source.name, rows=1, cols=2, into=None)  # → /tiles directory
    assert report.success, report.failures
    assert report.tiles_created == len(images) * 2

    tiled = list(client.images.iter(source.id, directory_path="/tiles"))
    assert len(tiled) == len(images) * 2
