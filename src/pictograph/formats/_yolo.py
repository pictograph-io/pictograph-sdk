"""YOLO ⇄ Pictograph annotation conversion (client-side, offline).

YOLO stores one ``.txt`` per image with normalized coordinates, so both
converters take the image's pixel ``(width, height)`` and the ordered
``class_names`` (YOLO lines reference a class by integer index).

- **Detection** line: ``<cls> <cx> <cy> <w> <h>`` - box center + size, each
  normalized to ``[0, 1]``. Maps to :class:`BBoxAnnotation`.
- **Segmentation** line: ``<cls> <x1> <y1> <x2> <y2> …`` - a normalized polygon
  (Ultralytics YOLO-seg). Maps to a single-ring :class:`PolygonAnnotation`.

Polylines and keypoints have no YOLO equivalent and are skipped by
:func:`to_yolo`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pictograph.models.annotation import (
    Annotation,
    BBoxAnnotation,
    PolygonAnnotation,
    PolygonGeometry,
)
from pictograph.models.common import BoundingBox, Point

from ._shared import annotation_bbox

if TYPE_CHECKING:
    from collections.abc import Sequence

_COORD = "{:.6f}"


def from_yolo(
    text: str,
    class_names: Sequence[str],
    image_width: int,
    image_height: int,
) -> list[Annotation]:
    """Parse one image's YOLO label text into Pictograph annotations.

    Args:
        text: The contents of the image's ``.txt`` label file.
        class_names: Ordered class names - YOLO's integer class index maps into
            this list.
        image_width: Image width in pixels (to denormalize coordinates).
        image_height: Image height in pixels.

    Returns:
        The annotations for that image (bbox lines → boxes, polygon lines →
        polygons). Blank lines and lines whose class index is out of range are
        skipped.

    Raises:
        ValueError: ``image_width`` or ``image_height`` is not positive.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive.")
    w, h = float(image_width), float(image_height)

    out: list[Annotation] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        try:
            cls = int(tokens[0])
            coords = [float(t) for t in tokens[1:]]
        except (ValueError, IndexError):
            continue
        if cls < 0 or cls >= len(class_names):
            continue
        name = class_names[cls]

        if len(coords) == 4:
            cx, cy, bw, bh = coords
            px_w, px_h = bw * w, bh * h
            if px_w <= 0 or px_h <= 0:
                continue
            x = cx * w - px_w / 2
            y = cy * h - px_h / 2
            out.append(
                BBoxAnnotation(name=name, bounding_box=BoundingBox(x=x, y=y, w=px_w, h=px_h))
            )
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            pts = [Point(x=coords[i] * w, y=coords[i + 1] * h) for i in range(0, len(coords), 2)]
            if len(pts) < 3:
                continue
            poly = PolygonAnnotation(name=name, polygon=PolygonGeometry(paths=[pts]))
            box = annotation_bbox(poly)
            out.append(
                PolygonAnnotation(name=name, polygon=PolygonGeometry(paths=[pts]), bounding_box=box)
            )
    return out


def to_yolo(
    annotations: Sequence[Annotation],
    class_names: Sequence[str],
    image_width: int,
    image_height: int,
    *,
    segmentation: bool = False,
) -> str:
    """Serialize one image's annotations to YOLO label text.

    Args:
        annotations: The image's annotations.
        class_names: Ordered class names - a name's position becomes its YOLO
            class index. Every annotation's ``name`` MUST appear here.
        image_width: Image width in pixels (to normalize coordinates).
        image_height: Image height in pixels.
        segmentation: When ``True``, polygons are written as normalized
            polygon lines (YOLO-seg); when ``False`` (default) a polygon is
            written as its enclosing box, matching a detection dataset.

    Returns:
        The ``.txt`` contents (one line per convertible annotation, no trailing
        newline). Polylines and keypoints are skipped.

    Raises:
        ValueError: A non-positive image dimension, or an annotation whose
            ``name`` is not in ``class_names``.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive.")
    w, h = float(image_width), float(image_height)
    index_by_name = {name: i for i, name in enumerate(class_names)}

    lines: list[str] = []
    for ann in annotations:
        if ann.type in ("polyline", "keypoint"):
            continue
        if ann.name not in index_by_name:
            raise ValueError(
                f"Annotation class {ann.name!r} is not in class_names "
                f"{list(class_names)!r}; provide the full class list."
            )
        idx = index_by_name[ann.name]

        if segmentation and ann.type == "polygon":
            coords: list[str] = []
            for p in ann.polygon.paths[0]:
                coords.append(_COORD.format(p.x / w))
                coords.append(_COORD.format(p.y / h))
            lines.append(f"{idx} " + " ".join(coords))
            continue

        box = annotation_bbox(ann)
        if box is None:
            continue
        cx = (box.x + box.w / 2) / w
        cy = (box.y + box.h / 2) / h
        lines.append(f"{idx} " + " ".join(_COORD.format(v) for v in (cx, cy, box.w / w, box.h / h)))

    return "\n".join(lines)
