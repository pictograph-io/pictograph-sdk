"""Live: dataset augmentation (``client.images.augment``).

Exercises the full round-trip against the real API - annotate a scratch dataset,
generate augmented variants into a new dataset (and into the source's directory),
and assert the produced images + remapped annotations. Both target datasets are
cleaned up. Gated by ``PICTOGRAPH_TEST_KEY`` like every live test.
"""

from __future__ import annotations

import contextlib

import pytest

from pictograph import Client
from pictograph.augment import HorizontalFlip, Rotate
from pictograph.models.annotation import BBoxAnnotation
from pictograph.models.common import BoundingBox

pytestmark = pytest.mark.live


def _annotate(client: Client, dataset_name: str, images) -> None:
    for img in images:
        client.annotations.save(
            dataset_name,
            img.id,
            [BBoxAnnotation(name="obj", bounding_box=BoundingBox(x=5, y=5, w=20, h=20))],
        )


def test_augment_into_new_dataset(client: Client, scratch_dataset_with_images, unique_name) -> None:
    source, images = scratch_dataset_with_images
    assert images, "fixture provided no images"
    _annotate(client, source.name, images)
    target = f"{unique_name}-aug"
    try:
        report = client.images.augment(
            source.name,
            [HorizontalFlip(p=1.0), Rotate(15.0)],
            multiplier=2,
            into=target,
            seed=1,
        )
        assert report.success, report.failures
        assert report.source_images == len(images)
        assert report.variants_created == len(images) * 2
        assert report.originals_copied == len(images)

        out = client.datasets.get(target)
        out_imgs = list(client.images.iter(out.id))
        # originals + 2 variants each
        assert len(out_imgs) == len(images) * 3

        variant = next(i for i in out_imgs if "_aug" in i.filename)
        anns = client.annotations.get(target, variant.id)
        assert len(anns) >= 1
        assert anns[0].name == "obj"
    finally:
        with contextlib.suppress(Exception):
            client.datasets.delete(target)


def test_augment_in_place_appends_to_directory(client: Client, scratch_dataset_with_images) -> None:
    source, images = scratch_dataset_with_images
    _annotate(client, source.name, images)
    report = client.images.augment(
        source.name,
        [HorizontalFlip(p=1.0)],
        multiplier=1,
        into=None,  # append into the source's /augmented directory
        seed=2,
    )
    assert report.success, report.failures
    assert report.originals_copied == 0  # in-place never re-copies originals
    assert report.variants_created == len(images)

    augmented = list(client.images.iter(source.id, directory_path="/augmented"))
    assert len(augmented) == len(images)
