"""Client-side dataset augmentation (``pictograph.augment``).

A native, batteries-included way to expand a dataset with augmented copies -
the SDK equivalent of "generate a version" in other CV tools - built on the
base **Pillow** dependency alone (no numpy, no OpenCV, no third-party SDK). Each
op transforms the image *and* remaps the annotation geometry consistently, so a
flip / rotate / crop keeps every bounding box, polygon, polyline, and keypoint
correct.

Compose ops into an :class:`Augmenter` and produce reproducible variants::

    from pictograph.augment import Augmenter, HorizontalFlip, Rotate, Brightness

    aug = Augmenter([HorizontalFlip(), Rotate((-15, 15)), Brightness((0.8, 1.2))], seed=42)
    image, annotations = aug("photo.jpg", annotations)  # one variant
    variants = aug.generate("photo.jpg", annotations, n=3)  # three distinct variants

To augment a whole Pictograph dataset in place - pulling images + annotations,
generating variants, and uploading them back through the standard ingest
pipeline (embeddings, auto-tags, thumbnails) - use the images resource::

    report = client.images.augment("road-signs", aug.ops, multiplier=3, into="road-signs-aug")

**Available ops.** Geometric (remap geometry): :class:`HorizontalFlip`,
:class:`VerticalFlip`, :class:`Rotate90`, :class:`Rotate`, :class:`Resize`,
:class:`Crop`. Photometric (pixels only, geometry unchanged):
:class:`Brightness`, :class:`Contrast`, :class:`Saturation`, :class:`Grayscale`,
:class:`Blur`, :class:`Noise`.
"""

from __future__ import annotations

from pictograph.augment._engine import Augmenter
from pictograph.augment._ops import (
    Augmentation,
    Blur,
    Brightness,
    Contrast,
    Crop,
    CutOut,
    Grayscale,
    HorizontalFlip,
    HueShift,
    Noise,
    Resize,
    Rotate,
    Rotate90,
    Saturation,
    Shear,
    VerticalFlip,
)
from pictograph.augment._specs import OP_NAMES, OP_SPECS, build_ops

__all__ = [
    "OP_NAMES",
    "OP_SPECS",
    "Augmentation",
    "Augmenter",
    "Blur",
    "Brightness",
    "Contrast",
    "Crop",
    "CutOut",
    "Grayscale",
    "HorizontalFlip",
    "HueShift",
    "Noise",
    "Resize",
    "Rotate",
    "Rotate90",
    "Saturation",
    "Shear",
    "VerticalFlip",
    "build_ops",
]
