"""Tests for ``Images.tile``.

Mocks the Client resource methods (HTTP behavior is covered in the resource
tests) and focuses on orchestration: target resolution + class copy, the tile
counts, tile geometry actually being clipped/translated on the saved
annotations, and the failure-report shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage

from pictograph.exceptions import NetworkError, NotFoundError
from pictograph.models.dataset import Dataset, DatasetClass
from pictograph.resources.images import Images, TileReport

if TYPE_CHECKING:
    from pathlib import Path
from tests.unit.resources._orchestration import build, sibling_resources


def _invoke(client: MagicMock, *args: object, **kwargs: object) -> object:
    """Invoke the real method on a real resource with its siblings stubbed."""
    with sibling_resources(client):
        resource = build(Images, client, own="images", delegate=["iter", "download", "upload"])
        return resource.tile(*args, **kwargs)


class _Img:
    """Lightweight stand-in for the Image model (pipeline reads only id + filename)."""

    def __init__(self, image_id: str, filename: str) -> None:
        self.id = image_id
        self.filename = filename


def _dataset(name: str = "src-ds") -> Dataset:
    return Dataset(
        id="src-id",
        name=name,
        classes=[DatasetClass(name="car", type="bbox", color="#ff0000")],
        image_count=1,
        created_at="2026-07-07T00:00:00Z",  # type: ignore[arg-type]
    )


def _project() -> Dataset:
    return Dataset(
        id="tgt-id",
        organization_id="org",
        name="tgt",
        annotation_types=["bbox"],
        classes=[],
        image_count=0,
        completed_image_count=0,
        total_size=0,
        archived_image_count=0,
        created_at="2026-07-07T00:00:00Z",  # type: ignore[arg-type]
    )


def _bbox_dict(x, y, w, h, name="car"):
    return {
        "id": f"a-{x}-{y}",
        "name": name,
        "type": "bbox",
        "bounding_box": {"x": x, "y": y, "w": w, "h": h},
    }


def _make_client(
    images: list[_Img],
    annotations: list[dict] | None = None,
    *,
    project_exists: bool = False,
    img_size: tuple[int, int] = (100, 100),
) -> MagicMock:
    client = MagicMock()
    source = _dataset()
    target = _project()

    # Source and target both resolve through client.datasets.get - key the
    # mock on the requested name (the old split projects/datasets mocks are gone).
    if project_exists:

        def _get(name: str | None = None, **_kw: object) -> object:
            return source if name == source.name else target

        client.datasets.get.side_effect = _get
    else:

        def _get(name: str | None = None, **_kw: object) -> object:
            if name == source.name:
                return source
            raise NotFoundError("no dataset", status_code=404)

        client.datasets.get.side_effect = _get
        client.datasets.create.return_value = target

    client.images.iter.return_value = iter(images)

    def _download(dataset_name, image_id, path, **_kw):
        PILImage.new("RGB", img_size, "red").save(path, format="PNG")
        return path

    client.images.download.side_effect = _download
    client.annotations.get.return_value = annotations if annotations is not None else []

    counter = {"n": 0}

    def _upload(dataset_id, file_path, **_kw):
        counter["n"] += 1
        return _Img(f"up-{counter['n']}", str(file_path))

    client.images.upload.side_effect = _upload
    client.annotations.save.return_value = MagicMock(new_count=1)
    return client


def test_tile_into_new_dataset_creates_and_copies_classes():
    imgs = [_Img("i1", "a.jpg")]
    client = _make_client(imgs)
    report = _invoke(client, "src-ds", rows=2, cols=2, into="tgt")
    assert isinstance(report, TileReport)
    assert report.source_images == 1
    assert report.tiles_created == 4  # 2x2 grid
    # class config copied into the created project
    _args, kwargs = client.datasets.create.call_args
    assert kwargs["classes"] == [{"name": "car", "type": "bbox", "color": "#ff0000"}]
    # every tile uploaded into the /tiles directory
    for call in client.images.upload.call_args_list:
        assert call.kwargs["directory_path"] == "/tiles"


def test_tile_in_place_skips_project_create():
    imgs = [_Img("i1", "a.jpg")]
    client = _make_client(imgs)
    report = _invoke(client, "src-ds", rows=2, cols=2, into=None)
    assert report.tiles_created == 4
    client.datasets.create.assert_not_called()


def test_tile_geometry_is_translated_and_clipped_on_saved_annotations():
    # A box wholly inside the top-right quadrant of a 100x100 image, 2x2 grid.
    anns = [_bbox_dict(60, 10, 20, 20)]
    client = _make_client([_Img("i1", "a.jpg")], anns)
    report = _invoke(client, "src-ds", rows=2, cols=2, into=None)
    # exactly one tile carries the (translated) annotation
    save_calls = client.annotations.save.call_args_list
    assert len(save_calls) == 1
    _dataset, _img_id, saved = save_calls[0].args
    b = saved[0].bounding_box
    # translated into the top-right tile's local frame (origin 50,0)
    assert (b.x, b.y, b.w, b.h) == pytest.approx((10.0, 10.0, 20.0, 20.0))
    assert report.annotations_written == 1
    # the other 3 tiles are empty (annotationless) → counted, but no save
    assert report.empty_tiles == 3


def test_exclude_empty_only_uploads_annotated_tiles():
    anns = [_bbox_dict(10, 10, 20, 20)]  # only the top-left tile
    client = _make_client([_Img("i1", "a.jpg")], anns)
    report = _invoke(client, "src-ds", rows=2, cols=2, into=None, include_empty=False)
    assert report.tiles_created == 1
    assert report.empty_tiles == 0
    assert client.images.upload.call_count == 1


def test_rows_cols_zero_raises():
    client = _make_client([_Img("i1", "a.jpg")])
    with pytest.raises(ValueError):
        _invoke(client, "src-ds", rows=0, cols=2)


def test_per_image_failure_is_collected_not_raised():
    imgs = [_Img("i1", "a.jpg"), _Img("i2", "b.jpg")]
    client = _make_client(imgs)

    def _download(dataset_name, image_id, path, **_kw):
        if image_id == "i1":
            raise NetworkError("boom")
        PILImage.new("RGB", (100, 100), "red").save(path, format="PNG")
        return path

    client.images.download.side_effect = _download
    report = _invoke(client, "src-ds", rows=2, cols=2, into=None)
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
        rows=1,
        cols=1,
        into=None,
        on_progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 2), (2, 2)]


def test_existing_target_is_reused():
    imgs = [_Img("i1", "a.jpg")]
    client = _make_client(imgs, project_exists=True)
    report = _invoke(client, "src-ds", rows=2, cols=2, into="tgt")
    assert report.tiles_created == 4
    client.datasets.create.assert_not_called()  # reused the existing target
