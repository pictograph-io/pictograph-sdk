"""Oriented (rotated) bounding-box geometry.

The SDK's copy of the geometry that the annotation editor (the editor) and
the server's oriented-box geometry (the server) share. All three are one shared definition: the
corner ORDER and the angle SIGN are a wire contract, and if any of them drifts, a box
drawn in the editor lands mirrored or transposed once it is exported - with nothing
else in the system noticing.

Angle convention: DEGREES, clockwise-positive, in image space (x → right, y → DOWN),
normalized to [0, 360). Matches CVAT's ``rotation``.

It is deliberately dependency-free (no numpy): the base SDK installs with nothing but
httpx + pydantic, and four corners of a rectangle are not worth a hard numeric dep.
"""

from __future__ import annotations

import math

from .models.annotation import OrientedBoxGeometry
from .models.common import Point

__all__ = ["normalize_angle", "obb_aabb", "obb_corners", "obb_from_corners"]

MIN_OBB_SIDE = 1.0


def normalize_angle(deg: float) -> float:
    """Normalize any angle to [0, 360)."""
    m = math.fmod(deg, 360.0)
    return m + 360.0 if m < 0 else m


def obb_corners(obb: OrientedBoxGeometry) -> list[Point]:
    """The box's 4 corners, in the canonical order
    ``[top-left, top-right, bottom-right, bottom-left]`` **of the box's own frame** -
    i.e. walked clockwise from the corner that is top-left at ``angle == 0``.

    This order is the contract: YOLO-OBB and DOTA both write four ``x y`` pairs walked
    around the box in exactly this order, and :func:`obb_from_corners` is its inverse.
    """
    rad = math.radians(obb.angle)
    cos = math.cos(rad)
    sin = math.sin(rad)
    hw = obb.w / 2.0
    hh = obb.h / 2.0
    # With y pointing DOWN, [[cos, -sin], [sin, cos]] turns the box clockwise on screen.
    local = ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
    return [
        Point(x=obb.cx + lx * cos - ly * sin, y=obb.cy + lx * sin + ly * cos) for lx, ly in local
    ]


def obb_from_corners(corners: list[Point]) -> OrientedBoxGeometry | None:
    """Inverse of :func:`obb_corners` - recover the parametric box from 4 corners.

    Best-fit rather than exact, on purpose: an augmentation maps the corners through a
    transform, and float round-trips (or a non-uniform resize, which shears a rotated
    rectangle into a parallelogram) leave a quad that is only ALMOST a rectangle. Taking
    the mean centre, the mean of each opposing side pair, and the top edge's bearing
    gives the nearest box instead of raising.

    Returns ``None`` for a ring that is not 4 usable points.
    """
    if len(corners) != 4:
        return None
    p0, p1, p2, p3 = corners
    if not all(math.isfinite(v) for p in corners for v in (p.x, p.y)):
        return None

    cx = (p0.x + p1.x + p2.x + p3.x) / 4.0
    cy = (p0.y + p1.y + p2.y + p3.y) / 4.0

    def dist(a: Point, b: Point) -> float:
        return math.hypot(b.x - a.x, b.y - a.y)

    # Opposing sides: top (p0→p1) & bottom (p3→p2); right (p1→p2) & left (p0→p3).
    w = (dist(p0, p1) + dist(p3, p2)) / 2.0
    h = (dist(p1, p2) + dist(p0, p3)) / 2.0

    # The top edge defines the box's +x axis. A horizontal flip reverses that edge, so
    # this is exactly what mirrors the angle - which is the correct result, and the
    # reason the corners are mapped rather than the angle being patched per-op.
    angle = math.degrees(math.atan2(p1.y - p0.y, p1.x - p0.x))

    return OrientedBoxGeometry(
        cx=cx,
        cy=cy,
        w=max(MIN_OBB_SIDE, w),
        h=max(MIN_OBB_SIDE, h),
        angle=normalize_angle(angle),
    )


def obb_aabb(obb: OrientedBoxGeometry) -> tuple[float, float, float, float]:
    """The axis-aligned enclosure as ``(x0, y0, x1, y1)``.

    Note this is derived from the ROTATED corners, not from ``w``/``h``: those are the
    box's own extents, so using them would under-report a rotated box's real footprint
    by up to sqrt(2).
    """
    pts = obb_corners(obb)
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))
