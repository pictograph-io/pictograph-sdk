"""Client-side annotation visualization (``pictograph.viz``) - THE annotation look.

``draw_annotations`` draws Pictograph annotations onto an image the way Pictograph's live views,
annotated videos and the app's image test draw them - THE annotation look ("Color frost") - built
only on the base **Pillow** dependency (the fonts ship with the SDK):

    >>> from pictograph import draw_annotations
    >>> annotated = draw_annotations("photo.jpg", annotations)
    >>> annotated.save("photo.annotated.png")

Every annotation type renders: ``bbox`` (a sharp, pixel-snapped box filled 20% with its class
color; a rotated box when it carries an ``oriented_box``), ``polygon`` (an instance mask, holes
included: filled 28% under a crisp rim), ``polyline`` (an open path with its end points) and
``keypoint`` (a solid point). Every line sits on a thin ink keyline, so it holds on any ground.
A label is the class color as matte glass at the top middle of its annotation, its words in
ink or inverse ink by contrast (Open Sans); a classifier's classes are badges in the top-left
corner. Sizes follow the image's longer side: label text is 20 px at 1080p, 15.7 at 720p.

Class colors are THE Pictograph palette by the name hash the app and the engine use (a class is
the same color everywhere), unless ``colors`` (a dataset's stored colors) names one.

Every attribute is configurable through ``look`` - the same config the engine draws with - and
everything you leave out keeps THE look::

    >>> draw_annotations(img, anns, look={"box": {"fill": 0, "width": 3}, "label": {"position": "tl"}})
    >>> draw_annotations(img, anns, look="smoke")  # one of LOOK_PRESETS

A multi-joint POSE is not a fifth type - it is several ``keypoint`` annotations sharing an
``instance_id``. Pass ``keypoint_templates`` (the per-class ``{"nodes": [...], "edges": [[i, j]]}``
from ``project_config``) and the joints of each instance are connected by its class's edges, one
palette color per person; without one the joints still render.

A classifier has no geometry: pass its result as ``classification`` and each of its classes at or
over ``min_confidence`` is a badge in the image's top-left corner::

    >>> result = model.predict("photo.jpg")  # a classifier: ClassificationResult
    >>> draw_annotations("photo.jpg", classification=result)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from PIL import Image as _PILImage
from pydantic import TypeAdapter

from pictograph._keypoint import (
    group_instances,
    match_template,
    slot_instance,
    template_edge_pairs,
    template_node_names,
)
from pictograph._look import LOOK_PRESETS, PALETTE, Badge, Bone, Look, Shape, class_color, render
from pictograph.inference.results import ClassificationResult, ClassScore
from pictograph.models.annotation import (
    Annotation,
    BBoxAnnotation,
    PolygonAnnotation,
    PolylineAnnotation,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from PIL.Image import Image as PILImage

__all__ = ["DEFAULT_PALETTE", "LOOK_PRESETS", "Look", "class_color", "draw_annotations"]

#: THE Pictograph palette - the colors a class is drawn in, by a stable hash of its name (the app's and
#: the engine's hash, so a class is the same color in the SDK, the app and every live view).
DEFAULT_PALETTE: tuple[str, ...] = PALETTE

# Re-validates a dict OR an existing annotation model into the right subclass.
_ANN_ADAPTER: TypeAdapter[Annotation] = TypeAdapter(Annotation)

AnnotationInput = Union["Annotation", "Mapping[str, Any]"]

ClassificationInput = Union["ClassificationResult", "ClassScore", "Mapping[str, Any]"]
"""A classifier's output for one image: its result (each of its classes at or over ``min_confidence``
is drawn), one class score, or a ``{"name", "confidence"}`` mapping."""

LookInput = Union[Look, "Mapping[str, Any]", str]
"""THE look's config (``Look``), a mapping of only what changes (``{"box": {"fill": 0}}``), or the name
of one of ``LOOK_PRESETS``."""

#: The confidence a classifier's class must reach to be a badge unless ``min_confidence`` says
#: otherwise: the engine's and a deployment's default threshold (a classifier always reports its best
#: class, so a weak model's guess under it is drawn by no one).
DEFAULT_MIN_CONFIDENCE = 0.5


def _is_mapping(value: object) -> bool:
    return hasattr(value, "keys") and hasattr(value, "__getitem__")


def _corner_labels(
    classification: ClassificationInput | Iterable[ClassificationInput] | None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> list[ClassScore]:
    """Each classifier's classes at or over ``min_confidence``, best first, in the order given, each
    class once (the rule the live view and the app draw by)."""
    if classification is None:
        return []
    single = isinstance(classification, (ClassificationResult, ClassScore)) or _is_mapping(
        classification
    )
    items: list[object] = [classification] if single else list(classification)
    out: list[ClassScore] = []
    for item in items:
        if isinstance(item, ClassificationResult):
            scores = list(item.classes)  # ranked, best first
        elif isinstance(item, ClassScore):
            scores = [item]
        elif _is_mapping(item):
            scores = [ClassScore.model_validate(dict(item))]  # type: ignore[call-overload]
        else:
            raise TypeError(
                "classification takes a ClassificationResult, a ClassScore or a {name, confidence} mapping "
                f"(or several), not {type(item).__name__}"
            )
        for score in scores:
            if score.confidence >= min_confidence and all(seen.name != score.name for seen in out):
                out.append(score)
    return out


def _resolve_look(look: LookInput | None) -> Look:
    if look is None:
        return LOOK_PRESETS["color"]
    if isinstance(look, Look):
        return look
    if isinstance(look, str):
        try:
            return LOOK_PRESETS[look]
        except KeyError:
            names = ", ".join(sorted(LOOK_PRESETS))
            raise ValueError(f"no look named {look!r}: one of {names}") from None
    return Look.model_validate(dict(look))


def _track_id(ann: Annotation) -> int | None:
    """A tracked detection's id (the engine writes it as ``attributes.track_id``)."""
    attrs = ann.attributes
    if not isinstance(attrs, dict):
        return None
    raw = attrs.get("track_id")
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    return None


def _obb_corners(
    cx: float, cy: float, w: float, h: float, angle: float
) -> list[tuple[float, float]]:
    """A rotated box's corners, clockwise from its own top-left (``angle``: degrees, clockwise in image space)."""
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    out = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        out.append((cx + dx * c - dy * s, cy + dx * s + dy * c))
    return out


def _shape(ann: Annotation, confidence: float | None) -> Shape | None:
    track = _track_id(ann)
    if isinstance(ann, BBoxAnnotation):
        box = ann.bounding_box
        enclosure = (box.x, box.y, box.w, box.h)
        if ann.oriented_box is not None:
            o = ann.oriented_box
            return Shape(
                "obb",
                ann.name,
                confidence,
                track,
                box=enclosure,
                corners=_obb_corners(o.cx, o.cy, o.w, o.h, o.angle),
            )
        return Shape("box", ann.name, confidence, track, box=enclosure)
    if isinstance(ann, PolygonAnnotation):
        rings = [[(p.x, p.y) for p in ring] for ring in ann.polygon.paths if len(ring) >= 2]
        if not rings:
            return None
        pts = [p for r in rings for p in r]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return Shape(
            "polygon",
            ann.name,
            confidence,
            track,
            box=(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)),
            rings=rings,
        )
    if isinstance(ann, PolylineAnnotation):
        path = [(p.x, p.y) for p in ann.polyline.path]
        return Shape("polyline", ann.name, confidence, track, path=path) if len(path) >= 2 else None
    kp = ann.keypoint  # the union's last member: a keypoint
    if not (math.isfinite(kp.x) and math.isfinite(kp.y)):
        return None
    return Shape("keypoint", ann.name, confidence, track, point=(kp.x, kp.y))


def _poses(
    parsed: list[Annotation], templates: Mapping[str, Mapping[str, Any]]
) -> tuple[list[Bone], dict[int, int]]:
    """Each keypoint INSTANCE's limbs (its class template's edges), and the person number of every joint that is
    part of one (``id(annotation) -> n``: one palette color per person).

    Two rules the grouping enforces: an edge only joins two joints of the SAME ``instance_id`` (two people's heads
    are not connected), and only when BOTH ends are placed (a limb never streaks to a joint the instance lacks).
    """
    bones: list[Bone] = []
    person: dict[int, int] = {}
    for class_name, template in ((n, templates[n]) for n in sorted(templates)):
        names = template_node_names(template)
        edges = template_edge_pairs(template)
        if not names:
            continue
        for instance in group_instances(parsed, names):
            if match_template(instance.points, templates) != class_name:
                continue
            n = len(set(person.values())) + 1
            for p in instance.points:
                person[id(p)] = n
            slots, _ = slot_instance(instance.points, names)
            for i, j in edges:
                a, b = slots[i], slots[j]
                if a is None or b is None:
                    continue
                bones.append(
                    Bone((a.keypoint.x, a.keypoint.y), (b.keypoint.x, b.keypoint.y), class_name, n)
                )
    return bones, person


def draw_annotations(
    image: str | Path | PILImage,
    annotations: Iterable[AnnotationInput] = (),
    *,
    classification: ClassificationInput | Iterable[ClassificationInput] | None = None,
    colors: Mapping[str, str] | None = None,
    width: int | None = None,
    show_labels: bool = True,
    show_confidence: bool | None = None,
    show_track_ids: bool | None = None,
    font_size: float | None = None,
    keypoint_templates: Mapping[str, Mapping[str, Any]] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    look: LookInput | None = None,
) -> PILImage:
    """Render annotations onto a copy of ``image`` in THE annotation look and return the new image.

    The input image is never mutated - a path/str is opened, a Pillow image is copied. Coordinates
    are absolute pixels in the image's own space (the space Pictograph stores), so do not pre-rotate
    via EXIF.

    Args:
        image: A file path, or an already-open Pillow ``Image``.
        annotations: An iterable of annotation models or raw annotation dicts (each is validated into
            the right type via the discriminated union).
        classification: A classifier's output - its :class:`ClassificationResult`, one :class:`ClassScore`
            or a ``{"name", "confidence"}`` mapping - or several (one per classifier). Each of its classes
            at or over ``min_confidence`` is a badge in the image's top-left corner, stacked best first in
            the order given, each class once: the way Pictograph's live views draw a classifier.
        colors: A dataset's stored class colors (``{"car": "#ff0000"}``); classes not listed take THE
            palette color their name hashes to.
        width: Every line's width in pixels, unscaled. By default THE look's widths (2 px at 1080p for a
            box, following the image's longer side, never under 2 px).
        show_labels: Draw each annotation's label (and a classifier's badges).
        show_confidence: Add each annotation's score to its label (``"car 73%"``). By default a score
            shows when the annotation has one below 1 - a model's prediction, not a person's label - and
            the look shows scores (``look={"label": {"parts": {"confidence": False}}}`` hides them).
        show_track_ids: Add a tracked detection's id (its ``attributes.track_id``) to its label
            (``"car 7 73%"``). By default the look's (``label.parts.track``: shown). Independent of the score.
        font_size: The label text's size in pixels. By default THE look's rule: 20 px at 1080p on a
            gentle curve of the image's longer side (15.7 at 720p, 30.3 at 4K, never under 12).
        keypoint_templates: Per-class keypoint templates (class name -> ``{"nodes": [{"name": ...}],
            "edges": [[i, j]]}``, the shape ``project_config`` stores). Joints sharing an
            ``instance_id`` are connected by their class's edges, one palette color per person.
        min_confidence: The confidence a classifier's class must reach to be a badge (default 0.5,
            Pictograph's default threshold; ``0`` draws every class it reports).
        look: What to change about THE look - a :class:`Look`, a mapping of only the fields that change
            (``{"box": {"fill": 0}, "label": {"position": "tl"}}``) or a preset's name (``LOOK_PRESETS``).
            Every attribute of every primitive is there: widths (px at 1080p), opacities, colors,
            keylines, fills, label glass / words / type / size / place, and per-class overrides.

    Returns:
        A new ``PIL.Image.Image`` (RGB) with the annotations drawn on it.

    Raises:
        TypeError: ``classification`` holds something that is not a classifier's output.
        ValueError: ``look`` names no preset, or a field of it is not THE look's.
    """
    if isinstance(image, (str, Path)):
        base = _PILImage.open(image).convert("RGB")
    else:
        base = image.convert("RGB")
    cfg = _resolve_look(look)
    label_update: dict[str, Any] = {}
    if not show_labels:
        label_update["show"] = False
    parts_update = {
        key: value
        for key, value in (("confidence", show_confidence), ("track", show_track_ids))
        if value is not None
    }
    if parts_update:
        label_update["parts"] = cfg.label.parts.model_copy(update=parts_update)
    if label_update:
        cfg = cfg.model_copy(update={"label": cfg.label.model_copy(update=label_update)})

    parsed = [_ANN_ADAPTER.validate_python(raw) for raw in annotations]
    bones, person = _poses(parsed, keypoint_templates) if keypoint_templates else ([], {})

    def shown(confidence: float) -> float | None:
        # By default only a prediction's score shows (a person's label is 1.0); show_confidence=True shows every one.
        return confidence if show_confidence or confidence < 1 else None

    shapes: list[Shape] = []
    for ann in parsed:
        sh = _shape(ann, shown(ann.confidence))
        if sh is None:
            continue
        if sh.kind == "keypoint" and id(ann) in person:
            sh.instance = person[id(ann)]
        shapes.append(sh)
    badges = [
        Badge(score.name, shown(score.confidence))
        for score in _corner_labels(classification, min_confidence)
    ]
    return render(
        base,
        shapes,
        cfg=cfg,
        bones=bones,
        badges=badges,
        stored_colors=dict(colors) if colors else None,
        font_size=font_size,
        stroke=width,
    )
