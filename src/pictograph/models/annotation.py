"""Pictograph annotation models - the canonical wire format.

Every annotation in the platform is one of these types:

============   ===================================================  =================================
``type``       Geometry                                             Discriminator field
============   ===================================================  =================================
``bbox``       :class:`BoundingBox`                                 ``bounding_box``
``polygon``    :class:`PolygonGeometry` (multi-ring)                ``polygon``
``polyline``   :class:`PolylineGeometry` (open path)                ``polyline``
``keypoint``   :class:`Point`                                        ``keypoint``
============   ===================================================  =================================

A multi-joint POSE is not its own type. It is several ``keypoint`` annotations -
one per joint, each named for the joint's own class - that share an
:attr:`~KeypointAnnotation.instance_id`. See that field for why the former
``skeleton`` primitive was removed.

An ORIENTED (rotated) box is NOT its own type: it is a ``bbox`` that additionally
carries an optional :class:`OrientedBoxGeometry` under ``oriented_box``. ``bounding_box``
stays the axis-aligned enclosure (what training and every OBB-unaware consumer reads);
a plain, non-rotated box has ``oriented_box=None``.

The :data:`Annotation` type alias is a discriminated union over these classes.
Pydantic dispatches on the ``type`` literal::

    >>> from pydantic import TypeAdapter
    >>> ta = TypeAdapter(Annotation)
    >>> ann = ta.validate_python({
    ...     "id": "ann-1",
    ...     "name": "person",
    ...     "type": "bbox",
    ...     "bounding_box": {"x": 100, "y": 200, "w": 50, "h": 80},
    ... })
    >>> isinstance(ann, BBoxAnnotation)
    True

Polygon and polyline annotations may omit ``bounding_box`` on save - the
backend computes the enclosing rectangle server-side.

The wire format uses snake_case field names (``bounding_box``, ``created_by``)
and matches what the annotation editor emits. There is no shorthand: any
``model_dump(mode="json", exclude_none=True)`` of these models is bit-for-bit
what gets stored in an image's ``annotations_json``.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import BoundingBox, NonBlankStr, Point

AnnotationType = Literal["bbox", "polygon", "polyline", "keypoint"]


# ───────────── geometry containers ─────────────


class PolygonGeometry(BaseModel):
    """Polygon geometry with multi-ring (hole) support.

    ``paths`` is a list of rings. The first ring is the outer boundary; each
    subsequent ring carves out a hole, rendered via the even-odd fill rule.
    A polygon with disconnected outer regions is represented by emitting two
    separate annotations, not by stuffing both into ``paths``.

    Each ring requires at least three points (a triangle is the minimum
    polygonal area).
    """

    model_config = ConfigDict(extra="forbid")

    paths: list[list[Point]] = Field(
        min_length=1,
        description=(
            "List of rings. First is the outer boundary; subsequent rings are "
            "holes (even-odd fill rule). Each ring needs >= 3 points."
        ),
    )

    @field_validator("paths")
    @classmethod
    def _check_ring_sizes(cls, value: list[list[Point]]) -> list[list[Point]]:
        for index, ring in enumerate(value):
            if len(ring) < 3:
                raise ValueError(
                    f"paths[{index}] has {len(ring)} point(s); "
                    f"a polygon ring requires at least 3 points."
                )
        return value


class PolylineGeometry(BaseModel):
    """Open path: an ordered list of points connected by line segments.

    A polyline has at least two points (a single segment is the minimum). It
    does not close - the last point is not implicitly connected to the first.
    """

    model_config = ConfigDict(extra="forbid")

    path: list[Point] = Field(
        min_length=2,
        description="Ordered list of points (>= 2). Does not close.",
    )


class OrientedBoxGeometry(BaseModel):
    """A rotated rectangle: a centre, extents along the box's OWN axes, and an angle.

    ``angle`` is in DEGREES, clockwise-positive in image space (x to the right, y
    DOWN), normalized to [0, 360). This is the same convention CVAT's ``rotation``
    attribute uses, so a rotated CVAT box round-trips without a sign flip.

    ``w`` and ``h`` are measured along the box's own axes, so they do NOT change when
    it is turned - which is the whole point of an oriented box. The axis-aligned
    enclosure of a rotated box is up to sqrt(2) larger and is mostly background.
    """

    model_config = ConfigDict(extra="forbid")

    cx: float = Field(description="Centre x, absolute image pixels.")
    cy: float = Field(description="Centre y, absolute image pixels.")
    w: float = Field(gt=0, description="Extent along the box's own +x axis.")
    h: float = Field(gt=0, description="Extent along the box's own +y axis.")
    angle: float = Field(
        default=0.0,
        ge=0.0,
        lt=360.0,
        description="Rotation in degrees, clockwise, [0, 360).",
    )


# ───────────── base annotation ─────────────


class _AnnotationBase(BaseModel):
    """Fields shared by every annotation type.

    Field declaration order matches the canonical wire format:
    ``id, name, type, <geometry>, confidence, created_by, attributes``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: NonBlankStr = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description=(
            "Unique annotation ID within the image. Auto-generated as a UUID4 "
            "when omitted - agents can construct annotations without tracking IDs."
        ),
    )
    name: NonBlankStr = Field(
        description=(
            "Class label. Must match a class defined on the dataset's "
            "class ontology - case-sensitive."
        ),
    )


# ───────────── concrete annotations ─────────────


class BBoxAnnotation(_AnnotationBase):
    """Bounding-box annotation - axis-aligned by default, optionally ROTATED.

    ``bounding_box`` is ALWAYS the axis-aligned enclosure and is what every OBB-unaware
    consumer reads - the export converters, training data-prep, the grid's class filter.
    Pictograph does not train on oriented boxes: a box trains on its orthogonal
    ``bounding_box`` whether or not it is rotated.

    ``oriented_box`` is present ONLY on a ROTATED box (aerial, document, shelf and pose
    imagery, where an axis-aligned box would be mostly background). When set, it is the
    rotated rectangle (centre, own-axis extents, angle) that ``bounding_box`` encloses;
    a plain, non-rotated box leaves it ``None`` so the common case stays minimal. There
    is no derived ``polygon`` key - the four rotated corners are reconstructed from
    ``oriented_box`` on demand.

    Only consumers with an OBB export pipeline read ``oriented_box``: export it with
    ``format="yolo_obb"`` (Ultralytics YOLO-OBB) or ``format="dota"`` (the aerial
    standard); ``cvat`` also carries the rotation natively.
    """

    type: Literal["bbox"] = "bbox"
    bounding_box: BoundingBox
    oriented_box: OrientedBoxGeometry | None = Field(
        default=None,
        description=(
            "Present only on a ROTATED box. When set, bounding_box is its "
            "axis-aligned enclosure; None means a plain axis-aligned box. "
            "Pictograph does not train OBB - a box trains on its orthogonal "
            "bounding_box whether rotated or not."
        ),
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_by: str | None = None
    attributes: dict[str, object] | list[object] = Field(default_factory=list)


class PolygonAnnotation(_AnnotationBase):
    """Polygon annotation, optionally with holes (multi-path).

    ``bounding_box`` may be omitted on save - the backend computes the
    enclosing rectangle from ``polygon.paths`` server-side. When emitted by
    the backend, ``bounding_box`` is always populated.
    """

    type: Literal["polygon"] = "polygon"
    bounding_box: BoundingBox | None = Field(
        default=None,
        description=(
            "Enclosing rectangle. Optional on save - the backend computes it "
            "from polygon.paths when omitted."
        ),
    )
    polygon: PolygonGeometry
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_by: str | None = None
    attributes: dict[str, object] | list[object] = Field(default_factory=list)


class PolylineAnnotation(_AnnotationBase):
    """Open polyline annotation (e.g., road centerlines, lanes)."""

    type: Literal["polyline"] = "polyline"
    bounding_box: BoundingBox | None = Field(
        default=None,
        description=(
            "Enclosing rectangle. Optional on save - the backend computes it "
            "from polyline.path when omitted."
        ),
    )
    polyline: PolylineGeometry
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_by: str | None = None
    attributes: dict[str, object] | list[object] = Field(default_factory=list)


class KeypointAnnotation(_AnnotationBase):
    """Single-point annotation (e.g., facial landmarks, joint locations).

    **It carries NO ``bounding_box``, deliberately** - a point has no extent, so there
      is nothing to enclose. The base is ``extra="forbid"``, so this is not a soft
      convention: an annotation that arrives with a ``bounding_box`` **raises** rather
      than being ignored, and any producer that adds one makes every such annotation
      unreadable through the typed clients. A consumer needing a point's extent must
      DERIVE one (see the exporters' ``MIN_KEYPOINT_SIDE`` box), never read a field
      that does not exist.

    **A multi-joint pose is several of these sharing an ``instance_id``** - one
      annotation per joint, each ``name``d for the joint's own class (``nose``,
      ``left_eye``, …). There is no ``skeleton`` annotation type; see
      :attr:`instance_id`.
    """

    type: Literal["keypoint"] = "keypoint"
    keypoint: Point
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_by: str | None = None
    attributes: dict[str, object] | list[object] = Field(default_factory=list)
    instance_id: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Which OBJECT this point belongs to, 1-based, scoped to the image. "
            "Points sharing an instance_id are joints of one object; the joint each "
            "one denotes is its `name`. None means 'unassociated' - a lone landmark "
            "with no multi-joint object around it."
        ),
    )
    """Which OBJECT on this image the point belongs to - 1-based, image-scoped.

    This is the ONLY field in the schema that carries instance identity, and it is
    what replaced the former ``skeleton`` annotation type (removed in SDK 1.68.1).
    A skeleton's edge list was redundant - a per-class template identical for every
    instance - while what it uniquely carried was the grouping. Three people at
    seventeen joints each is fifty-one points; without a grouping key they are
    fifty-one *unassociated* points, and multi-instance pose cannot be trained,
    because a query-based/top-down keypoint head (RF-DETR's) takes ground-truth
    instance grouping AS the supervision signal.

    The connectivity that the old primitive stored per annotation now lives ONCE per
    class, on ``project_config``: a keypoint class may declare a template
    (``{"nodes": [{"name": ...}], "edges": [[i, j]]}``) whose node names are joint
    class names. Skeletons therefore become postprocessing - group points by
    ``instance_id``, connect them via the class template.

    Auto-assigned by the editor per CLASS: the first ``nose`` on an image is
    instance 1, the second ``nose`` is instance 2. So placing every joint of one
    object needs no bookkeeping - each is the first of its own class and lands on
    instance 1 - while a second object's joints land on instance 2.
    """


# ───────────── discriminated union ─────────────


Annotation = Annotated[
    BBoxAnnotation | PolygonAnnotation | PolylineAnnotation | KeypointAnnotation,
    Field(discriminator="type"),
]
"""Discriminated union over all annotation types.

Use with ``pydantic.TypeAdapter`` for parsing arbitrary annotation dicts::

    from pydantic import TypeAdapter
    ann_adapter = TypeAdapter(Annotation)
    ann = ann_adapter.validate_python(some_dict)  # returns the right subclass

Or as a list type::

    annotations: list[Annotation] = ...
"""
