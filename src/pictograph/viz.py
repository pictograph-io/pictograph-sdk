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
``keypoint`` (a solid point). A label is the class color as matte glass at the top middle of its
annotation, its words - the class name and a tracked detection's id - in ink or inverse ink by
contrast (Open Sans); a classifier's classes are badges in the top-left corner. Scores are off
by default (``show_confidence=True`` adds them). Sizes follow the image's longer side: label
text is 20 px at 1080p, 15.7 at 720p.

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

A model's result draws as it is - every model type::

    >>> draw_annotations("photo.jpg", model.predict("photo.jpg"))  # boxes, masks, points or badges
    >>> draw_annotations("photo.jpg", client.models.predict(model_id, "photo.jpg"))  # a hosted run's

A semantic segmentation model's polygons are class REGIONS, drawn in THE look's region style (filled 34%
under a thin edge, labelled by name); pass ``regions=True`` to draw a list of polygons that way. A
classifier has no geometry: each of its classes at or over ``min_confidence`` is a badge in the image's
top-left corner (its result as ``annotations`` or as ``classification``)::

    >>> result = model.predict("photo.jpg")  # a classifier: ClassificationResult
    >>> draw_annotations("photo.jpg", classification=result)

An image given as a path or as bytes is drawn with its EXIF orientation applied - the pixels
``model.predict`` reads, and the space Pictograph stores annotations in - so a phone photo's predictions
land where they were found. A Pillow image is drawn as it is; a numpy array is read as BGR (the SDK's
convention for a decoded frame, as ``cv2`` gives it).
"""

from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Union

from PIL import Image as _PILImage, ImageOps as _PILImageOps
from pydantic import TypeAdapter

from pictograph._keypoint import (
    group_instances,
    match_template,
    slot_instance,
    template_edge_pairs,
    template_node_names,
)
from pictograph._look import LOOK_PRESETS, PALETTE, Badge, Bone, Look, Shape, class_color, render
from pictograph.inference.results import (
    ClassificationResult,
    ClassScore,
    InferenceResult,
    SemanticSegmentationResult,
)
from pictograph.models.annotation import (
    Annotation,
    BBoxAnnotation,
    PolygonAnnotation,
    PolylineAnnotation,
)
from pictograph.models.model import ModelPredictResult

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from PIL.Image import Image as PILImage

__all__ = [
    "DEFAULT_PALETTE",
    "LOOK_PRESETS",
    "AnnotationsInput",
    "ImageInput",
    "Look",
    "class_color",
    "draw_annotations",
]

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


class ArrayImage(Protocol):
    """A decoded image as an array - numpy's ``ndarray``, BGR (the SDK's convention for a raw frame)."""

    @property
    def shape(self) -> tuple[int, ...]: ...


ImageInput = Union[str, Path, bytes, bytearray, "PILImage", ArrayImage]
"""What ``draw_annotations`` draws on: a file path, the image's bytes, a Pillow image, or a BGR array."""

AnnotationsInput = Union["Iterable[AnnotationInput]", InferenceResult, ModelPredictResult]
"""What it draws: annotations (models or dicts), a model's result (``model.predict``), or a hosted run's
(``client.models.predict``)."""

#: The confidence a classifier's class must reach to be a badge unless ``min_confidence`` says
#: otherwise: the engine's and a deployment's default threshold (a classifier always reports its best
#: class, so a weak model's guess under it is drawn by no one).
DEFAULT_MIN_CONFIDENCE = 0.5


def _is_mapping(value: object) -> bool:
    return hasattr(value, "keys") and hasattr(value, "__getitem__")


def _classifier_outputs(
    classification: ClassificationInput | Iterable[ClassificationInput] | None,
) -> list[object]:
    """One classifier's output, or several, as a list."""
    if classification is None:
        return []
    single = isinstance(classification, (ClassificationResult, ClassScore)) or _is_mapping(
        classification
    )
    return [classification] if single else list(classification)


def _corner_labels(
    items: list[object], min_confidence: float = DEFAULT_MIN_CONFIDENCE
) -> list[ClassScore]:
    """Each classifier's classes at or over ``min_confidence``, best first, in the order given, each
    class once (the rule the live view and the app draw by)."""
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


def _shape(ann: Annotation, confidence: float | None, region: bool = False) -> Shape | None:
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
        # A semantic model's region is an area, not an object: its label names it, no score (THE look's region).
        return Shape(
            "region" if region else "polygon",
            ann.name,
            None if region else confidence,
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


def _open_image(image: ImageInput) -> PILImage:
    """The pixels to draw on: a path's or bytes' EXIF orientation applied (what ``model.predict`` decodes - OpenCV
    applies it - and the space Pictograph stores annotations in), a Pillow image as it is, a BGR array as RGB."""
    if isinstance(image, (str, Path, bytes, bytearray)):
        source = BytesIO(bytes(image)) if isinstance(image, (bytes, bytearray)) else image
        with _PILImage.open(source) as opened:
            return _PILImageOps.exif_transpose(opened).convert("RGB")
    if isinstance(image, _PILImage.Image):
        return image.convert("RGB")
    shape = getattr(image, "shape", None)
    if isinstance(shape, tuple) and len(shape) in (2, 3):
        import numpy as np

        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, 2::-1]  # BGR(A) -> RGB
        return _PILImage.fromarray(np.ascontiguousarray(arr, dtype=np.uint8)).convert("RGB")
    raise TypeError(
        "image takes a file path, the image's bytes, a Pillow image or a BGR numpy array, "
        f"not {type(image).__name__}"
    )


def _read_annotations(
    annotations: AnnotationsInput,
) -> tuple[list[AnnotationInput], list[object], bool]:
    """``(annotations, classifier outputs, regions)`` from what was given: a model's result is read by its type (a
    classifier's classes are badges; a semantic model's polygons are regions), a hosted run's by its ``model_type``."""
    if isinstance(annotations, ClassificationResult):
        return [], [annotations], False
    if isinstance(annotations, InferenceResult):
        predictions: list[AnnotationInput] = list(getattr(annotations, "predictions", []))
        return predictions, [], isinstance(annotations, SemanticSegmentationResult)
    if isinstance(annotations, ModelPredictResult):
        scores = annotations.tag_scores
        # A class with no score (a backend before B209) keeps its badge: 1.0 passes any threshold and shows no score.
        badges: list[object] = [
            ClassScore(name=tag, confidence=scores[i] if i < len(scores) else 1.0)
            for i, tag in enumerate(annotations.tags)
        ]
        return (
            list(annotations.annotations),
            badges,
            annotations.model_type == "semantic_segmentation",
        )
    return list(annotations), [], False


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
    image: ImageInput,
    annotations: AnnotationsInput = (),
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
    regions: bool | None = None,
    look: LookInput | None = None,
) -> PILImage:
    """Render annotations onto a copy of ``image`` in THE annotation look and return the new image.

    The input image is never mutated - a path or bytes are opened, a Pillow image is copied. Coordinates
    are absolute pixels in the image's displayed space: a path's or bytes' EXIF orientation is applied
    first (what ``model.predict`` reads, the space Pictograph stores), so do not pre-rotate it yourself.

    Args:
        image: A file path, the image's bytes, an already-open Pillow ``Image`` (drawn as it is), or a
            decoded BGR numpy array (a video frame as ``cv2`` gives it).
        annotations: An iterable of annotation models or raw annotation dicts (each is validated into
            the right type via the discriminated union) - or a model's result as it is: ``model.predict``'s
            (any task) or a hosted run's (``client.models.predict``).
        classification: A classifier's output - its :class:`ClassificationResult`, one :class:`ClassScore`
            or a ``{"name", "confidence"}`` mapping - or several (one per classifier). Each of its classes
            at or over ``min_confidence`` is a badge in the image's top-left corner, stacked best first in
            the order given, each class once: the way Pictograph's live views draw a classifier.
        colors: A dataset's stored class colors (``{"car": "#ff0000"}``); classes not listed take THE
            palette color their name hashes to.
        width: Every line's width in pixels, unscaled. By default THE look's widths (2 px at 1080p for a
            box, following the image's longer side, never under 2 px).
        show_labels: Draw each annotation's label (and a classifier's badges).
        show_confidence: Add each annotation's score to its label (``"car 73%"``) - a person's label
            (1.0) too - and to a classifier's badges. Off by default: THE look's label is its name and track
            id. ``look={"label": {"parts": {"confidence": True}}}`` instead shows a score only where it is
            below 1 (a model's prediction).
        show_track_ids: Add a tracked detection's id (its ``attributes.track_id``) to its label
            (``"car 7 73%"``). By default the look's (``label.parts.track``: shown). Independent of the score.
        font_size: The label text's size in pixels. By default THE look's rule: 20 px at 1080p on a
            gentle curve of the image's longer side (15.7 at 720p, 30.3 at 4K, never under 12).
        keypoint_templates: Per-class keypoint templates (class name -> ``{"nodes": [{"name": ...}],
            "edges": [[i, j]]}``, the shape ``project_config`` stores). Joints sharing an
            ``instance_id`` are connected by their class's edges, one palette color per person.
        min_confidence: The confidence a classifier's class must reach to be a badge (default 0.5,
            Pictograph's default threshold; ``0`` draws every class it reports).
        regions: Draw the polygons as a semantic model's class regions (filled 34% under a thin edge,
            labelled by name alone). By default a semantic segmentation result's are, and nothing else is.
        look: What to change about THE look - a :class:`Look`, a mapping of only the fields that change
            (``{"box": {"fill": 0}, "label": {"position": "tl"}}``) or a preset's name (``LOOK_PRESETS``).
            Every attribute of every primitive is there: widths (px at 1080p), opacities, colors,
            keylines (none by default), fills, label glass / words / type / size / place, and per-class
            overrides.

    Returns:
        A new ``PIL.Image.Image`` (RGB) with the annotations drawn on it.

    Raises:
        TypeError: ``classification`` holds something that is not a classifier's output, or ``image`` is
            none of the forms above.
        ValueError: ``look`` names no preset, or a field of it is not THE look's.
    """
    base = _open_image(image)
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

    given, outputs, result_regions = _read_annotations(annotations)
    as_regions = result_regions if regions is None else regions
    parsed = [_ANN_ADAPTER.validate_python(raw) for raw in given]
    bones, person = _poses(parsed, keypoint_templates) if keypoint_templates else ([], {})

    def shown(confidence: float) -> float | None:
        # A score shows only where the label shows scores (show_confidence, or the look's label.parts): with the
        # look's parts alone only a prediction's (a person's label is 1.0); show_confidence=True every one.
        return confidence if show_confidence or confidence < 1 else None

    shapes: list[Shape] = []
    for ann in parsed:
        sh = _shape(ann, shown(ann.confidence), region=as_regions)
        if sh is None:
            continue
        if sh.kind == "keypoint" and id(ann) in person:
            sh.instance = person[id(ann)]
        shapes.append(sh)
    badges = [
        Badge(score.name, shown(score.confidence))
        for score in _corner_labels(
            [*outputs, *_classifier_outputs(classification)], min_confidence
        )
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
