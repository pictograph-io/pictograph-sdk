"""Tests for the optional PyTorch dataset adapter (B32b).

Gated on the ``torch`` extra - skipped cleanly when torch isn't installed
(mirrors the live-test skip convention), so the default CI/dev gate is unaffected.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage
from pydantic import TypeAdapter

torch = pytest.importorskip("torch")

from pictograph._torch_dataset import (  # noqa: E402 - after importorskip
    PictographTorchDataset,
    build_detection_target,
)
from pictograph.models.annotation import Annotation  # noqa: E402
from pictograph.models.dataset import DatasetImage  # noqa: E402

_ANN = TypeAdapter(list[Annotation])


def _png(path: Path, size: tuple[int, int] = (32, 24)) -> None:
    buf = io.BytesIO()
    PILImage.new("RGB", size, (10, 20, 30)).save(buf, format="PNG")
    path.write_bytes(buf.getvalue())


def _img_meta(image_id: str, filename: str) -> DatasetImage:
    return DatasetImage(
        id=image_id,
        filename=filename,
        status="complete",
        annotation_count=1,
        created_at="2026-01-01T00:00:00Z",  # type: ignore[arg-type]
    )


def _annotations() -> list[Annotation]:
    return _ANN.validate_python(
        [
            {
                "id": "a",
                "name": "car",
                "type": "bbox",
                "bounding_box": {"x": 5, "y": 6, "w": 10, "h": 8},
            },
            {
                "id": "b",
                "name": "road",
                "type": "polygon",
                "polygon": {"paths": [[{"x": 0, "y": 0}, {"x": 20, "y": 0}, {"x": 10, "y": 15}]]},
            },
            {"id": "c", "name": "unknown-class", "type": "keypoint", "keypoint": {"x": 1, "y": 2}},
        ]
    )


def test_build_detection_target_shapes_and_skips_unknown() -> None:
    target = build_detection_target(_annotations(), {"car": 0, "road": 1}, "img-1")
    # 'unknown-class' is skipped (not in class_to_idx) → 2 boxes, not 3.
    assert target["boxes"].shape == (2, 4)
    assert target["labels"].tolist() == [0, 1]
    # bbox xyxy = (5,6,15,14); area = 10*8 = 80.
    assert target["boxes"][0].tolist() == [5.0, 6.0, 15.0, 14.0]
    assert target["area"][0].item() == pytest.approx(80.0)
    assert target["iscrowd"].tolist() == [0, 0]
    assert target["image_id"] == "img-1"
    # raw annotations preserved in full (all 3, nothing silently dropped).
    assert len(target["annotations"]) == 3


def test_build_detection_target_empty() -> None:
    target = build_detection_target([], {"car": 0}, "img-0")
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].numel() == 0


def test_dataset_len_getitem_and_classes(tmp_path: Path) -> None:
    meta = _img_meta("img-1", "photo.png")
    _png(tmp_path / "img-1.png")  # pre-seed so download=False reads it

    ann_res = MagicMock()
    ann_res.get.return_value = _annotations()
    ds = PictographTorchDataset(
        dataset_name="road-signs",
        images=[meta],
        image_resource=MagicMock(),
        annotation_resource=ann_res,
        class_to_idx={"car": 0, "road": 1},
        root=tmp_path,
        download=False,
    )
    assert len(ds) == 1
    assert ds.classes == ["car", "road"]

    image, target = ds[0]
    assert isinstance(image, PILImage.Image)
    assert image.mode == "RGB"
    assert target["boxes"].shape == (2, 4)
    ann_res.get.assert_called_once_with("road-signs", "img-1")


def test_dataset_downloads_when_missing(tmp_path: Path) -> None:
    meta = _img_meta("img-9", "x.png")

    def fake_download(dataset_name: str, image: str, output_path: Path) -> Path:
        _png(Path(output_path))
        return Path(output_path)

    img_res = MagicMock()
    img_res.download.side_effect = fake_download
    ann_res = MagicMock()
    ann_res.get.return_value = []

    ds = PictographTorchDataset(
        dataset_name="road-signs",
        images=[meta],
        image_resource=img_res,
        annotation_resource=ann_res,
        class_to_idx={},
        root=tmp_path,
        download=True,
    )
    image, _ = ds[0]
    assert isinstance(image, PILImage.Image)
    img_res.download.assert_called_once()


def test_transforms_applied(tmp_path: Path) -> None:
    meta = _img_meta("img-1", "photo.png")
    _png(tmp_path / "img-1.png")
    ann_res = MagicMock()
    ann_res.get.return_value = []

    ds = PictographTorchDataset(
        dataset_name="road-signs",
        images=[meta],
        image_resource=MagicMock(),
        annotation_resource=ann_res,
        class_to_idx={},
        root=tmp_path,
        download=False,
        transform=lambda _im: "TRANSFORMED",
        target_transform=lambda t: {"wrapped": t},
    )
    image, target = ds[0]
    assert image == "TRANSFORMED"
    assert "wrapped" in target


def test_augment_remaps_target_boxes(tmp_path: Path) -> None:
    from pictograph.augment import Augmenter, HorizontalFlip

    meta = _img_meta("img-1", "photo.png")
    _png(tmp_path / "img-1.png", size=(32, 24))
    ann_res = MagicMock()
    ann_res.get.return_value = _annotations()  # 'car' box xyxy = (5,6,15,14)

    ds = PictographTorchDataset(
        dataset_name="road-signs",
        images=[meta],
        image_resource=MagicMock(),
        annotation_resource=ann_res,
        class_to_idx={"car": 0, "road": 1},
        root=tmp_path,
        download=False,
        augment=Augmenter([HorizontalFlip(p=1.0)], seed=1),
    )
    image, target = ds[0]
    assert isinstance(image, PILImage.Image)
    assert image.size == (32, 24)
    # HorizontalFlip on a 32px-wide image: box x=5,w=10 -> x'=32-(5+10)=17,
    # so xyxy becomes (17,6,27,14). The detection target reflects the flip.
    assert target["boxes"][0].tolist() == [17.0, 6.0, 27.0, 14.0]


def test_no_augment_leaves_boxes_unchanged(tmp_path: Path) -> None:
    meta = _img_meta("img-1", "photo.png")
    _png(tmp_path / "img-1.png", size=(32, 24))
    ann_res = MagicMock()
    ann_res.get.return_value = _annotations()
    ds = PictographTorchDataset(
        dataset_name="road-signs",
        images=[meta],
        image_resource=MagicMock(),
        annotation_resource=ann_res,
        class_to_idx={"car": 0, "road": 1},
        root=tmp_path,
        download=False,
    )
    _image, target = ds[0]
    assert target["boxes"][0].tolist() == [5.0, 6.0, 15.0, 14.0]


def test_works_with_dataloader(tmp_path: Path) -> None:
    meta = _img_meta("img-1", "photo.png")
    _png(tmp_path / "img-1.png")
    ann_res = MagicMock()
    ann_res.get.return_value = []
    ds = PictographTorchDataset(
        dataset_name="road-signs",
        images=[meta],
        image_resource=MagicMock(),
        annotation_resource=ann_res,
        class_to_idx={},
        root=tmp_path,
        download=False,
        transform=lambda _im: torch.zeros(3, 4, 4),
        target_transform=lambda t: t["image_id"],
    )
    # The map-style protocol is accepted by DataLoader despite no subclassing.
    loader = torch.utils.data.DataLoader(ds, batch_size=1)
    batch = next(iter(loader))
    images, ids = batch
    assert images.shape == (1, 3, 4, 4)
    assert list(ids) == ["img-1"]  # default collate returns a tuple of strings


def test_as_pytorch_derives_class_to_idx(tmp_path: Path) -> None:
    from pictograph.models.dataset import Dataset, DatasetClass
    from pictograph.resources.datasets import Datasets

    datasets = Datasets(MagicMock())
    fake = Dataset(
        id="ds-1",
        name="road-signs",
        classes=[DatasetClass(name="stop"), DatasetClass(name="yield")],
        images=[_img_meta("img-1", "a.png")],
        created_at="2026-01-01T00:00:00Z",
    )
    datasets.get = MagicMock(return_value=fake)  # type: ignore[method-assign]
    ds = datasets.as_pytorch("road-signs", root=tmp_path, download=False)
    # alphabetical: stop=0, yield=1
    assert ds.class_to_idx == {"stop": 0, "yield": 1}
    assert len(ds) == 1
