"""Geometry helpers shared by the COCO and YOLO converters.

Pure functions over :class:`~pictograph.models.common.Point` /
:class:`~pictograph.models.common.BoundingBox` - no I/O, no third-party deps
(numpy et al. are deliberately avoided so ``pictograph.formats`` works with the
base install).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pictograph.models.common import BoundingBox, Point

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pictograph.models.annotation import Annotation


def bbox_from_points(points: Sequence[Point]) -> BoundingBox | None:
    """Axis-aligned enclosing box of ``points``, or ``None`` if degenerate.

    Returns ``None`` when the points span zero width or height (a box needs
    strictly-positive ``w``/``h`` per :class:`BoundingBox`).
    """
    if not points:
        return None
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    return BoundingBox(x=x0, y=y0, w=w, h=h)


def polygon_area(ring: Sequence[Point]) -> float:
    """Absolute area of a simple polygon ring via the shoelace formula."""
    n = len(ring)
    if n < 3:
        return 0.0
    acc = 0.0
    for i in range(n):
        a = ring[i]
        b = ring[(i + 1) % n]
        acc += a.x * b.y - b.x * a.y
    return abs(acc) / 2.0


def annotation_bbox(ann: Annotation) -> BoundingBox | None:
    """Best-effort ``BoundingBox`` for any annotation type.

    Uses the stored ``bounding_box`` when present, else derives the enclosing
    rectangle from the geometry. Returns ``None`` only for a keypoint (a single
    point has no positive-area box).
    """
    box = getattr(ann, "bounding_box", None)
    if isinstance(box, BoundingBox):
        return box
    if ann.type == "polygon":
        return bbox_from_points(ann.polygon.paths[0])
    if ann.type == "polyline":
        return bbox_from_points(ann.polyline.path)
    return None


def flatten_ring(ring: Sequence[Point]) -> list[float]:
    """``[x0, y0, x1, y1, ...]`` - the flat coord list COCO segmentation uses."""
    out: list[float] = []
    for p in ring:
        out.append(p.x)
        out.append(p.y)
    return out


def points_from_flat(coords: Sequence[float]) -> list[Point]:
    """Inverse of :func:`flatten_ring` - pair a flat coord list into points.

    A trailing odd coordinate (malformed input) is dropped rather than raising.
    """
    return [Point(x=coords[i], y=coords[i + 1]) for i in range(0, len(coords) - 1, 2)]
