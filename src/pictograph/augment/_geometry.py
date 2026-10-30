"""Geometry remapping for augmentation - pure functions, no I/O, no numpy.

The augmentation ops in :mod:`pictograph.augment._ops` transform an image with
Pillow and hand this module a *point map* ``fn: (x, y) -> (x', y')`` plus the
output image size. This module applies that map to every annotation's geometry
(bbox / polygon / polyline / keypoint), re-deriving enclosing boxes and - for
cropping ops - clipping geometry to the new frame and dropping annotations that
fall (mostly) outside it.

Everything here operates on the canonical typed models
(:data:`~pictograph.models.annotation.Annotation`,
:class:`~pictograph.models.common.Point` /
:class:`~pictograph.models.common.BoundingBox`), which are ``frozen=True`` - so
every transform builds *fresh* instances (never mutates in place). Reuses the
existing shoelace/enclosing-box helpers in :mod:`pictograph.formats._shared`
rather than re-deriving them (DRY).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pictograph._obb import obb_corners, obb_from_corners
from pictograph.formats._shared import bbox_from_points, polygon_area
from pictograph.models.annotation import (
    Annotation,
    BBoxAnnotation,
    KeypointAnnotation,
    OrientedBoxGeometry,
    PolygonAnnotation,
    PolylineAnnotation,
)
from pictograph.models.common import BoundingBox, Point

if TYPE_CHECKING:
    from collections.abc import Sequence

#: A point map: source pixel ``(x, y)`` → destination pixel ``(x', y')``.
PointFn = Callable[[float, float], "tuple[float, float]"]


def _map_ring(ring: Sequence[Point], fn: PointFn) -> list[Point]:
    """Apply ``fn`` to every point of a ring/path."""
    out: list[Point] = []
    for p in ring:
        nx, ny = fn(p.x, p.y)
        out.append(Point(x=nx, y=ny))
    return out


def _bbox_corners(box: BoundingBox) -> list[Point]:
    """The four corners of a box (so a rotation maps them individually)."""
    return [
        Point(x=box.x, y=box.y),
        Point(x=box.x + box.w, y=box.y),
        Point(x=box.x + box.w, y=box.y + box.h),
        Point(x=box.x, y=box.y + box.h),
    ]


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _clip_box(box: BoundingBox, out_w: float, out_h: float) -> BoundingBox | None:
    """Intersect a box with the ``[0, out_w] x [0, out_h]`` frame (exact)."""
    x0 = _clamp(box.x, 0.0, out_w)
    y0 = _clamp(box.y, 0.0, out_h)
    x1 = _clamp(box.x + box.w, 0.0, out_w)
    y1 = _clamp(box.y + box.h, 0.0, out_h)
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return BoundingBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)


def _clip_ring(ring: Sequence[Point], out_w: float, out_h: float) -> list[Point]:
    """Sutherland-Hodgman clip of a polygon ring against the axis-aligned frame.

    Clips against the four half-planes ``x>=0``, ``x<=out_w``, ``y>=0``,
    ``y<=out_h`` in turn. Returns the (possibly empty) clipped ring.
    """

    # Each edge: (inside test, intersection with the boundary line).
    def clip_edge(
        poly: list[Point],
        inside: Callable[[Point], bool],
        intersect: Callable[[Point, Point], Point],
    ) -> list[Point]:
        if not poly:
            return []
        out: list[Point] = []
        prev = poly[-1]
        prev_in = inside(prev)
        for cur in poly:
            cur_in = inside(cur)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif prev_in:
                out.append(intersect(prev, cur))
            prev, prev_in = cur, cur_in
        return out

    def lerp(a: Point, b: Point, t: float) -> Point:
        return Point(x=a.x + (b.x - a.x) * t, y=a.y + (b.y - a.y) * t)

    poly = list(ring)
    # Clip against the left boundary (keep points with x within the frame).
    poly = clip_edge(
        poly,
        lambda p: p.x >= 0.0,
        lambda a, b: lerp(a, b, (0.0 - a.x) / (b.x - a.x)) if b.x != a.x else a,
    )
    # Clip against the right boundary.
    poly = clip_edge(
        poly,
        lambda p: p.x <= out_w,
        lambda a, b: lerp(a, b, (out_w - a.x) / (b.x - a.x)) if b.x != a.x else a,
    )
    # Clip against the top boundary.
    poly = clip_edge(
        poly,
        lambda p: p.y >= 0.0,
        lambda a, b: lerp(a, b, (0.0 - a.y) / (b.y - a.y)) if b.y != a.y else a,
    )
    # Clip against the bottom boundary.
    return clip_edge(
        poly,
        lambda p: p.y <= out_h,
        lambda a, b: lerp(a, b, (out_h - a.y) / (b.y - a.y)) if b.y != a.y else a,
    )


def _remap_bbox(
    ann: BBoxAnnotation,
    fn: PointFn,
    out_w: float,
    out_h: float,
    *,
    clip: bool,
    min_visibility: float,
) -> BBoxAnnotation | None:
    if ann.oriented_box is not None:
        return _remap_oriented_bbox(
            ann, ann.oriented_box, fn, out_w, out_h, clip=clip, min_visibility=min_visibility
        )
    corners = _map_ring(_bbox_corners(ann.bounding_box), fn)
    box = bbox_from_points(corners)
    if box is None:
        return None
    full_area = box.w * box.h
    if clip:
        clipped = _clip_box(box, out_w, out_h)
        if clipped is None:
            return None
        if full_area > 0 and (clipped.w * clipped.h) / full_area < min_visibility:
            return None
        box = clipped
    return ann.model_copy(update={"bounding_box": box})


def _remap_oriented_bbox(
    ann: BBoxAnnotation,
    oriented_box: OrientedBoxGeometry,
    fn: PointFn,
    out_w: float,
    out_h: float,
    *,
    clip: bool,
    min_visibility: float,
) -> BBoxAnnotation | None:
    """Remap a ROTATED box (a bbox carrying ``oriented_box``) under a point transform.

    The ANGLE has to move too, and that is the whole difficulty: a horizontal flip
    MIRRORS it, a rotation ADDS to it, and a non-uniform resize changes it by an amount
    that depends on the angle itself. Case-splitting on the op would get all three
    subtly wrong the moment they compose.

    So don't. Map the four CORNERS through the same point function every other geometry
    here uses, and re-fit the oriented box from them. That is exact for any affine map,
    correct under composition, and needs no knowledge of which op is running.
    ``bounding_box`` is re-derived as the axis-aligned enclosure of the moved corners,
    exactly as the AABB was maintained before.

    The box is never geometrically CLIPPED: a clipped rectangle is not a rectangle, and
    clamping its corners would silently shear it into a shape the format cannot express.
    An out-of-frame box is instead kept whole (the OBB exporters clamp normalized coords)
    or DROPPED when too little of it is visible - the same accept/reject `_remap_polygon`
    makes. If the transform destroys the box's rectangularity so badly that no oriented
    box can be recovered, it degrades to a plain axis-aligned box (``oriented_box=None``)
    rather than emitting an inconsistent one.
    """
    corners = obb_corners(oriented_box)
    mapped = [Point(x=nx, y=ny) for nx, ny in (fn(p.x, p.y) for p in corners)]

    if clip:
        full = polygon_area(mapped)
        visible = polygon_area(_clip_ring(mapped, out_w, out_h))
        if full > 0 and visible / full < min_visibility:
            return None

    box = bbox_from_points(mapped)
    if box is None:
        return None

    fitted = obb_from_corners(mapped)
    # If the oriented box can't be recovered (degenerate transform), degrade to a plain
    # axis-aligned box rather than carry a stale/inconsistent oriented_box.
    return ann.model_copy(update={"oriented_box": fitted, "bounding_box": box})


def _remap_polygon(
    ann: PolygonAnnotation,
    fn: PointFn,
    out_w: float,
    out_h: float,
    *,
    clip: bool,
    min_visibility: float,
) -> PolygonAnnotation | None:
    new_rings: list[list[Point]] = []
    outer_full_area = 0.0
    outer_clipped_area = 0.0
    for idx, ring in enumerate(ann.polygon.paths):
        mapped = _map_ring(ring, fn)
        if clip:
            mapped = _clip_ring(mapped, out_w, out_h)
        if len(mapped) < 3:
            if idx == 0:
                return None  # outer ring gone → annotation gone
            continue  # a hole disappeared - fine
        if idx == 0:
            outer_full_area = polygon_area(_map_ring(ring, fn))
            outer_clipped_area = polygon_area(mapped)
        new_rings.append(mapped)
    if not new_rings:
        return None
    if clip and outer_full_area > 0 and outer_clipped_area / outer_full_area < min_visibility:
        return None
    box = bbox_from_points(new_rings[0])
    return ann.model_copy(
        update={
            "polygon": ann.polygon.model_copy(update={"paths": new_rings}),
            "bounding_box": box,
        }
    )


def _remap_polyline(
    ann: PolylineAnnotation,
    fn: PointFn,
    out_w: float,
    out_h: float,
    *,
    clip: bool,
) -> PolylineAnnotation | None:
    mapped = _map_ring(ann.polyline.path, fn)
    if clip:
        # Polylines are open paths - Sutherland-Hodgman (which closes the ring)
        # does not apply. Clamp points to the frame and drop the annotation only
        # if it collapses below the 2-point minimum after de-duplicating.
        clamped = [Point(x=_clamp(p.x, 0.0, out_w), y=_clamp(p.y, 0.0, out_h)) for p in mapped]
        deduped: list[Point] = []
        for p in clamped:
            if not deduped or deduped[-1] != p:
                deduped.append(p)
        if len(deduped) < 2:
            return None
        mapped = deduped
    box = bbox_from_points(mapped)
    return ann.model_copy(
        update={
            "polyline": ann.polyline.model_copy(update={"path": mapped}),
            "bounding_box": box,
        }
    )


def _remap_keypoint(
    ann: KeypointAnnotation,
    fn: PointFn,
    out_w: float,
    out_h: float,
    *,
    clip: bool,
) -> KeypointAnnotation | None:
    """Remap one joint. A multi-joint POSE is several of these sharing an ``instance_id``.

    The point goes through ``fn``, which is the ONLY correct way: a flip mirrors it, a
    rotation turns it, a resize scales it, and case-splitting per operation is wrong the
    moment they compose.

    Two things this must NOT do, both of which corrupt a pose invisibly:

    * **It must not swap left/right joint NAMES on a horizontal flip.** It is tempting -
      a mirrored person's left wrist really is on the right of the frame - but the name is
      the joint's CLASS, and it is what the exporter slots into the class template by.
      Permuting it here would silently re-align every exported ``[x, y, v]`` triplet
      against the wrong joint. Mirroring is the training pipeline's job, via YOLO-pose's
      ``flip_idx`` (which the exporter derives), where it applies to the model's output
      space rather than to ground truth.
    * **It must not touch ``instance_id``.** That is METADATA, not geometry: it says which
      OBJECT the point belongs to, and a transform moves objects without renumbering,
      merging or splitting them. ``model_copy`` carries it through untouched, which is the
      point of updating only the geometry key.

    A joint carried out of frame under ``clip`` is dropped, and the joints that stayed are
    unaffected - the template alignment is by NAME at export time, not by list position,
    so nothing shifts behind the gap (which is what made the old template-complete node
    list so delicate).
    """
    nx, ny = fn(ann.keypoint.x, ann.keypoint.y)
    if clip and (nx < 0.0 or nx > out_w or ny < 0.0 or ny > out_h):
        return None
    return ann.model_copy(update={"keypoint": Point(x=nx, y=ny)})


def remap_annotations(
    annotations: Sequence[Annotation],
    fn: PointFn,
    out_w: float,
    out_h: float,
    *,
    clip: bool = False,
    min_visibility: float = 0.1,
) -> list[Annotation]:
    """Apply a point map to every annotation, returning fresh models.

    Args:
        annotations: Typed annotations in the source image's pixel space.
        fn: Maps a source pixel ``(x, y)`` to the destination pixel.
        out_w: Destination image width (pixels).
        out_h: Destination image height (pixels).
        clip: When ``True`` (cropping ops), clip geometry to the destination
            frame and drop annotations that fall (mostly) outside it. When
            ``False`` (flips / rotations with an expanding canvas / resize /
            photometric ops - everything stays in frame), geometry is remapped
            without clipping and no annotation is dropped.
        min_visibility: For ``clip=True`` only - drop an annotation whose
            visible area is less than this fraction of its transformed area
            (``0.0``-``1.0``). Ignored when ``clip=False``.

    Returns:
        A new list of annotations. May be shorter than the input when
        ``clip=True`` drops out-of-frame annotations.
    """
    out: list[Annotation] = []
    for ann in annotations:
        result: Annotation | None
        if isinstance(ann, BBoxAnnotation):
            result = _remap_bbox(ann, fn, out_w, out_h, clip=clip, min_visibility=min_visibility)
        elif isinstance(ann, PolygonAnnotation):
            result = _remap_polygon(ann, fn, out_w, out_h, clip=clip, min_visibility=min_visibility)
        elif isinstance(ann, PolylineAnnotation):
            result = _remap_polyline(ann, fn, out_w, out_h, clip=clip)
        else:  # KeypointAnnotation - exhaustive over the discriminated union
            result = _remap_keypoint(ann, fn, out_w, out_h, clip=clip)
        if result is not None:
            out.append(result)
    return out
