"""Tests for ``Images.augment``.

Mocks the Client resource methods (HTTP behavior is covered in the resource
tests) and focuses on orchestration: target resolution + class copy, the
original-vs-variant upload counts, geometry actually being augmented on the
saved annotations, and the failure-report shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage

from pictograph.augment import HorizontalFlip, Rotate
from pictograph.exceptions import NetworkError, NotFoundError
from pictograph.models.annotation import BBoxAnnotation
from pictograph.models.common import BoundingBox
from pictograph.models.dataset import Dataset, DatasetClass
from pictograph.resources.images import AugmentReport, Images
from tests.unit.resources._orchestration import build, sibling_resources


def _invoke(client: MagicMock, *args: object, **kwargs: object) -> object:
    """Invoke the real method on a real resource with its siblings stubbed."""
    with sibling_resources(client):
        resource = build(Images, client, own="images", delegate=["iter", "download", "upload"])
        return resource.augment(*args, **kwargs)


def _ann(name: str) -> BBoxAnnotation:
    return BBoxAnnotation(name=name, bounding_box=BoundingBox(x=10, y=20, w=30, h=40))


if TYPE_CHECKING:
    from pathlib import Path


class _Img:
    """Lightweight stand-in for the Image model (the pipeline reads only id + filename)."""

    def __init__(self, image_id: str, filename: str) -> None:
        self.id = image_id
        self.filename = filename


def _dataset(name: str = "src-ds") -> Dataset:
    return Dataset(
        id="src-id",
        name=name,
        classes=[DatasetClass(name="car", type="bbox", color="#ff0000")],
        image_count=2,
        created_at="2026-07-06T00:00:00Z",  # type: ignore[arg-type]
    )


def _bbox_dict(x=10, y=20, w=30, h=40):
    return {
        "id": "a",
        "name": "car",
        "type": "bbox",
        "bounding_box": {"x": x, "y": y, "w": w, "h": h},
    }


def _make_client(images: list[_Img], *, project_exists: bool = False) -> MagicMock:
    client = MagicMock()
    source = _dataset()
    target = Dataset(
        id="tgt-id",
        organization_id="org",
        name="tgt",
        annotation_types=["bbox"],
        classes=[],
        image_count=0,
        completed_image_count=0,
        total_size=0,
        archived_image_count=0,
        created_at="2026-07-06T00:00:00Z",  # type: ignore[arg-type]
    )

    # Source and target both resolve through client.datasets.get - key the
    # mock on the requested name (the old split projects/datasets mocks are gone).
    if project_exists:

        def _get(name: str | None = None, **_kw: object) -> Dataset:
            return source if name == source.name else target

        client.datasets.get.side_effect = _get
    else:

        def _get(name: str | None = None, **_kw: object) -> Dataset:
            if name == source.name:
                return source
            raise NotFoundError("no dataset", status_code=404)

        client.datasets.get.side_effect = _get
        client.datasets.create.return_value = target

    client.images.iter.return_value = iter(images)

    def _download(dataset_name, image_id, path, **_kw):
        PILImage.new("RGB", (60, 40), "red").save(path, format="PNG")
        return path

    client.images.download.side_effect = _download
    client.annotations.get.return_value = [_bbox_dict()]

    counter = {"n": 0}

    def _upload(dataset_id, file_path, **_kw):
        counter["n"] += 1
        return _Img(f"up-{counter['n']}", str(file_path))

    client.images.upload.side_effect = _upload
    client.annotations.save.return_value = MagicMock(new_count=1)
    return client


def test_augment_into_new_dataset_creates_and_copies_classes():
    imgs = [_Img("i1", "a.jpg"), _Img("i2", "b.jpg")]
    client = _make_client(imgs)
    report = _invoke(
        client, "src-ds", ops=[HorizontalFlip(p=1.0)], multiplier=2, into="tgt", seed=1
    )
    assert isinstance(report, AugmentReport)
    assert report.success
    assert report.source_images == 2
    assert report.variants_created == 4  # 2 images x 2 variants
    assert report.originals_copied == 2  # include_original default, new dataset
    # class config copied into the created project
    _args, kwargs = client.datasets.create.call_args
    assert kwargs["classes"] == [{"name": "car", "type": "bbox", "color": "#ff0000"}]
    assert kwargs["annotation_types"] == ["bbox"]
    # six uploads total: two originals plus four variants
    assert client.images.upload.call_count == 6


def test_augment_in_place_skips_originals_and_project_create():
    imgs = [_Img("i1", "a.jpg")]
    client = _make_client(imgs)
    report = _invoke(client, "src-ds", ops=[HorizontalFlip(p=1.0)], multiplier=3, into=None, seed=1)
    assert report.originals_copied == 0
    assert report.variants_created == 3
    client.datasets.create.assert_not_called()
    # every upload lands in the augmented directory
    for call in client.images.upload.call_args_list:
        assert call.kwargs["directory_path"] == "/augmented"


def test_variant_geometry_is_actually_flipped():
    imgs = [_Img("i1", "a.jpg")]
    client = _make_client(imgs)
    _invoke(client, "src-ds", ops=[HorizontalFlip(p=1.0)], multiplier=1, into=None, seed=1)
    # the single variant's saved annotation must be horizontally flipped:
    # source box x=10,w=30 on a 60px-wide image -> x' = 60 - (10+30) = 20
    save_calls = client.annotations.save.call_args_list
    assert len(save_calls) == 1
    _dataset, _img_id, saved_anns = save_calls[0].args
    assert saved_anns[0].bounding_box.x == pytest.approx(20.0)


def test_multiplier_zero_raises():
    client = _make_client([_Img("i1", "a.jpg")])
    with pytest.raises(ValueError):
        _invoke(client, "src-ds", ops=[HorizontalFlip()], multiplier=0)


def test_per_image_failure_is_collected_not_raised():
    imgs = [_Img("i1", "a.jpg"), _Img("i2", "b.jpg")]
    client = _make_client(imgs)

    def _download(dataset_name, image_id, path, **_kw):
        if image_id == "i1":
            raise NetworkError("boom")
        PILImage.new("RGB", (60, 40), "red").save(path, format="PNG")
        return path

    client.images.download.side_effect = _download
    report = _invoke(client, "src-ds", ops=[Rotate(10.0)], multiplier=1, into=None, seed=1)
    assert len(report.failures) == 1
    assert report.failures[0].image_id == "i1"
    assert report.source_images == 1  # the good one still processed


def test_progress_callback_fires_per_image(tmp_path: Path):
    imgs = [_Img("i1", "a.jpg"), _Img("i2", "b.jpg")]
    client = _make_client(imgs)
    seen: list[tuple[int, int]] = []
    _invoke(
        client,
        "src-ds",
        ops=[HorizontalFlip(p=1.0)],
        multiplier=1,
        into=None,
        on_progress=lambda done, total: seen.append((done, total)),
        seed=1,
    )
    assert seen == [(1, 2), (2, 2)]


# ── preprocessing: drop_classes + skip_empty ────────────────────────────


def test_drop_classes_removes_annotations():
    client = _make_client([_Img("i1", "a.jpg")])
    client.annotations.get.return_value = [_ann("car"), _ann("person")]
    _invoke(
        client,
        "src-ds",
        ops=[HorizontalFlip(p=1.0)],
        multiplier=1,
        into=None,
        drop_classes=["person"],
        seed=1,
    )
    # the single variant's saved annotations must exclude the dropped 'person'
    saved = client.annotations.save.call_args_list
    assert len(saved) == 1
    _dataset, _img_id, anns = saved[0].args
    assert [a.name for a in anns] == ["car"]


def test_drop_classes_filters_new_target_class_config():
    client = _make_client([_Img("i1", "a.jpg")])  # source class config = [{car}]
    client.annotations.get.return_value = [_ann("car")]
    _invoke(
        client,
        "src-ds",
        ops=[HorizontalFlip(p=1.0)],
        multiplier=1,
        into="tgt",
        drop_classes=["car"],
        seed=1,
    )
    _args, kwargs = client.datasets.create.call_args
    # 'car' was dropped → the created target carries no classes
    assert kwargs["classes"] is None


def test_skip_empty_skips_images_with_no_annotations_after_drop():
    client = _make_client([_Img("i1", "a.jpg"), _Img("i2", "b.jpg")])
    client.annotations.get.return_value = [_ann("person")]  # both images only 'person'
    report = _invoke(
        client,
        "src-ds",
        ops=[HorizontalFlip(p=1.0)],
        multiplier=2,
        into=None,
        drop_classes=["person"],
        skip_empty=True,
        seed=1,
    )
    assert report.skipped_empty == 2
    assert report.variants_created == 0
    assert report.source_images == 0
    client.images.upload.assert_not_called()  # nothing materialised
