"""PyTorch map-style dataset adapter (optional - requires the ``torch`` extra).

``client.datasets.as_pytorch(name)`` returns a **map-style** dataset
(``__len__`` + ``__getitem__``) that plugs straight into
``torch.utils.data.DataLoader``. Each item is ``(image, target)`` where ``image``
is a ``PIL.Image`` (or whatever ``transform`` returns) and ``target`` follows the
torchvision detection convention (``boxes`` xyxy + integer ``labels`` + ``area``
+ ``iscrowd`` + ``image_id`` + the raw Pictograph ``annotations``).

We implement the map-style protocol rather than subclassing
``torch.utils.data.Dataset`` on purpose: it keeps the SDK's ``mypy --strict``
gate torch-free (the gate doesn't install the ``torch`` extra), and ``DataLoader``
accepts any object with ``__len__`` + ``__getitem__`` all the same.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image as _PILImage

from pictograph._path_safety import safe_path_component
from pictograph.models.annotation import (
    KeypointAnnotation,
    PolygonAnnotation,
    PolylineAnnotation,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pictograph.augment import Augmenter
    from pictograph.models.annotation import Annotation
    from pictograph.models.dataset import DatasetImage
    from pictograph.resources.annotations import Annotations
    from pictograph.resources.images import Images

__all__ = ["PictographTorchDataset", "build_detection_target"]


def _annotation_box(ann: Annotation) -> tuple[float, float, float, float] | None:
    """Axis-aligned xyxy box for an annotation, or ``None`` if not derivable."""
    bbox = getattr(ann, "bounding_box", None)
    if bbox is not None:
        return (bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h)
    if isinstance(ann, PolygonAnnotation):
        points = [p for ring in ann.polygon.paths for p in ring]
    elif isinstance(ann, PolylineAnnotation):
        points = list(ann.polyline.path)
    elif isinstance(ann, KeypointAnnotation):
        kp = ann.keypoint
        return (kp.x, kp.y, kp.x, kp.y)
    else:
        return None
    if not points:
        return None
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def build_detection_target(
    annotations: list[Annotation],
    class_to_idx: dict[str, int],
    image_id: str,
) -> dict[str, Any]:
    """torchvision-style detection target from Pictograph annotations.

    Annotations whose class is absent from ``class_to_idx`` are skipped (they
    can't be assigned a label); the raw ``annotations`` list is always returned
    in full so nothing is silently lost.
    """
    boxes: list[tuple[float, float, float, float]] = []
    labels: list[int] = []
    for ann in annotations:
        if ann.name not in class_to_idx:
            continue
        box = _annotation_box(ann)
        if box is None:
            continue
        boxes.append(box)
        labels.append(class_to_idx[ann.name])

    boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    labels_t = torch.as_tensor(labels, dtype=torch.int64)
    if boxes:
        area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
    else:
        area = torch.zeros((0,), dtype=torch.float32)
    return {
        "boxes": boxes_t,
        "labels": labels_t,
        "area": area,
        "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        "image_id": image_id,
        "annotations": [a.model_dump(mode="json", exclude_none=True) for a in annotations],
    }


class PictographTorchDataset:
    """Map-style dataset over a Pictograph dataset's images + annotations.

    Use via :meth:`pictograph.resources.datasets.Datasets.as_pytorch`. Images are
    downloaded lazily on first access into ``root`` (a temp dir by default) and
    cached there, mirroring torchvision's download-once behavior.

    Pass ``augment`` (a :class:`pictograph.augment.Augmenter`) for on-the-fly
    augmentation: each ``__getitem__`` produces a freshly-augmented variant with
    the detection target's boxes remapped to match. Note that with a multi-worker
    ``DataLoader`` the augmenter is forked into each worker, so for maximum
    variant diversity across workers construct it without a ``seed`` (the default)
    or re-seed per worker via ``DataLoader(worker_init_fn=...)``.
    """

    def __init__(
        self,
        *,
        dataset_name: str,
        images: list[DatasetImage],
        image_resource: Images,
        annotation_resource: Annotations,
        class_to_idx: dict[str, int],
        root: str | Path | None = None,
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[dict[str, Any]], Any] | None = None,
        download: bool = True,
        augment: Augmenter | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.images = list(images)
        self._images = image_resource
        self._annotations = annotation_resource
        self.class_to_idx = dict(class_to_idx)
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.augment = augment
        self.root = (
            Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="pictograph-torch-"))
        )
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def classes(self) -> list[str]:
        """Class names ordered by their integer label index."""
        return [name for name, _ in sorted(self.class_to_idx.items(), key=lambda kv: kv[1])]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        meta = self.images[index]
        # meta.id and meta.filename both come from the API. Neither may steer
        # where this writes, so both are reduced to safe components.
        # A suffix legitimately STARTS with a dot, which safe_path_component
        # strips (a leading/trailing dot is a Windows hazard on a whole
        # component). Sanitise the extension body, then re-attach the dot.
        raw_suffix = Path(meta.filename).suffix.lstrip(".")
        suffix = f".{safe_path_component(raw_suffix, fallback='img')}" if raw_suffix else ".img"
        local = self.root / f"{safe_path_component(meta.id, fallback='image')}{suffix}"
        if self.download and not local.exists():
            self._images.download(self.dataset_name, meta.id, local)
        with _PILImage.open(local) as img:
            image: Any = img.convert("RGB")

        annotations = self._annotations.get(self.dataset_name, meta.id)
        if self.augment is not None:
            # Augment the image and remap the annotation geometry together, so the
            # detection target below reflects the augmented boxes.
            image, annotations = self.augment.augment(image, annotations)
        target: Any = build_detection_target(annotations, self.class_to_idx, meta.id)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
