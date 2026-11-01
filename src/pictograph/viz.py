"""Client-side annotation visualization (``pictograph.viz``).

A lightweight, SDK-native way to draw Pictograph annotations onto an image -
the batteries-included alternative to wiring up a third-party renderer. Built
only on the base **Pillow** dependency, so it works out of the box:

    >>> from pictograph import Client, draw_annotations
    >>> client = Client()
    >>> img = client.images.get(image_id)
    >>> # ... fetch the annotations for that image ...
    >>> annotated = draw_annotations("photo.jpg", annotations)
    >>> annotated.save("photo.annotated.png")

All four annotation types render: ``bbox`` (rectangle), ``polygon`` (closed
multi-ring outline, holes included), ``polyline`` (open path), and
``keypoint`` (marker). Each class gets a stable, distinct color; class labels
are drawn with a filled backing chip for legibility and can be turned off. A
label sits at the top centre of its annotation and reads on its color (dark on
a light one), as Pictograph's live views and annotated videos draw it.

A multi-joint POSE is not a fifth type - it is several ``keypoint`` annotations
sharing an ``instance_id``. Pass ``keypoint_templates`` (the per-class
``{"nodes": [...], "edges": [[i, j]]}`` from ``project_config``) and the joints of
each instance are connected by its class's edges; without one the joints still
render, and only the cosmetic connectivity is missing.

A classifier has no geometry: pass its result as ``classification`` and its top
class is a badge in the image's top-left corner, the way Pictograph's live views
and annotated videos draw it::

    >>> result = model.predict("photo.jpg")  # a classifier: ClassificationResult
    >>> draw_annotations("photo.jpg", classification=result, show_confidence=True)
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from PIL import Image as _PILImage, ImageDraw, ImageFont
from pydantic import TypeAdapter

from pictograph._keypoint import (
    group_instances,
    match_template,
    slot_instance,
    template_edge_pairs,
    template_node_names,
)
from pictograph.inference.results import ClassificationResult, ClassScore
from pictograph.models.annotation import (
    Annotation,
    BBoxAnnotation,
    KeypointAnnotation,
    PolygonAnnotation,
    PolylineAnnotation,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from PIL.Image import Image as PILImage

__all__ = ["DEFAULT_PALETTE", "draw_annotations"]

#: A fixed, visually-distinct palette. Classes are assigned a color by a stable
#: hash of the class name, so the same class is always the same color across
#: images and runs (deterministic - no per-call randomness).
DEFAULT_PALETTE: tuple[str, ...] = (
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#469990",
    "#9a6324",
    "#800000",
    "#808000",
    "#000075",
    "#a9a9a9",
    "#e6beff",
)

# Re-validates a dict OR an existing annotation model into the right subclass.
_ANN_ADAPTER: TypeAdapter[Annotation] = TypeAdapter(Annotation)

AnnotationInput = Union["Annotation", "Mapping[str, Any]"]

ClassificationInput = Union["ClassificationResult", "ClassScore", "Mapping[str, Any]"]
"""A classifier's output for one image: its result (each of its classes at or over ``min_confidence``
is drawn), one class score, or a ``{"name", "confidence"}`` mapping."""

#: A classifier's corner badge: its inset from the image's top and left edges, a fraction of the
#: shorter side (at least 8 px - the engine's ``kCornerLabelInset``), and its font size, a fraction of
#: the image's LONGER side (at least 15 px): its words ~2% of that side tall, as Pictograph's live
#: views (pictograph-prt D-089) and the app's image test draw them - read at a glance at any size.
CORNER_LABEL_INSET = 0.015
CORNER_LABEL_FONT_SHARE = 0.0275
CORNER_LABEL_MIN_FONT = 15
#: An annotation's label size unless ``font_size`` says otherwise: a quarter larger than it was (14), as Pictograph's
#: live views and annotated videos draw their labels a quarter larger since 2026-10-31 (pictograph-prt D-095).
DEFAULT_LABEL_FONT_SIZE = 17.5
#: The confidence a classifier's class must reach to be a badge unless ``min_confidence`` says
#: otherwise: the engine's and a deployment's default threshold (a classifier always reports its best
#: class, so a weak model's guess under it is drawn by no one).
DEFAULT_MIN_CONFIDENCE = 0.5


def _color_for(name: str, overrides: Mapping[str, str] | None) -> str:
    if overrides and name in overrides:
        return overrides[name]
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()  # noqa: S324 - non-crypto, stable color hash
    return DEFAULT_PALETTE[int(digest, 16) % len(DEFAULT_PALETTE)]


def _load_font(size: float) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    # Pillow >= 10.1 sizes the bundled font (a fraction of a pixel too); older
    # releases ignore it. Fall back so the helper never hard-fails on font.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - only on very old Pillow
        return ImageFont.load_default()


def _label_ink(color: str) -> str:
    """The ink a label is written in on its class color: dark on a light color, white on the rest - the live view's
    rule (pictograph-prt ``contrast_text``: Rec.601 luminance over 150; the app's ``labelInk``), so a yellow class's
    label stays readable. A color it cannot read (a named one) is written in white."""
    m = re.fullmatch(r"#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})", color.strip())
    if m is None:
        return "#ffffff"
    hexv = m.group(1) if len(m.group(1)) == 6 else "".join(c * 2 for c in m.group(1))
    r, g, b = (int(hexv[i : i + 2], 16) for i in (0, 2, 4))
    return "#1e1e1e" if 0.299 * r + 0.587 * g + 0.114 * b > 150 else "#ffffff"


def _top_center(ann: Annotation, width: int) -> tuple[float, float]:
    """The middle of an annotation's top edge, where its label hangs: a box's own; a polygon's and a polyline's, their
    points' box; a keypoint's, its marker's top (the marker ``draw_annotations`` draws, radius ``max(3, width + 2)``)."""
    if isinstance(ann, BBoxAnnotation):
        box = ann.bounding_box
        return box.x + box.w / 2, box.y
    if isinstance(ann, KeypointAnnotation):
        return ann.keypoint.x, ann.keypoint.y - max(3, width + 2)
    if isinstance(ann, PolygonAnnotation):
        pts = [p for ring in ann.polygon.paths for p in ring]
    else:
        pts = list(ann.polyline.path)
    xs = [p.x for p in pts]
    return (min(xs) + max(xs)) / 2, min(p.y for p in pts)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    text: str,
    top_center: tuple[float, float],
    color: str,
    size: tuple[int, int],
    gap: int,
) -> None:
    """Draw ``text`` on a filled chip at the TOP CENTRE of an annotation (``top_center``: the middle of its top edge) -
    centred over it, ``gap`` px above it, inside under its top edge when the image's top leaves no room, kept inside the
    image at its sides - where Pictograph's live views hang an instance's label (pictograph-prt D-095; Noah, 2026-10-31:
    "place labels on the center top of the detected items instead of upper left")."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    pad = 2
    w, h = (right - left) + 2 * pad, (bottom - top) + 2 * pad
    cx, edge = top_center
    x = min(max(0.0, cx - w / 2), max(0.0, size[0] - 1 - w))
    y = edge - gap - h
    if y < 0:
        y = edge + gap
    y = min(y, max(0.0, size[1] - 1 - h))
    draw.rectangle([x, y, x + w, y + h], fill=color)
    draw.text((x + pad - left, y + pad - top), text, fill=_label_ink(color), font=font)


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


def _draw_corner_label(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    text: str,
    top_left: tuple[int, int],
    color: str,
    size: tuple[int, int],
) -> int:
    """Draw ``text`` on a filled chip whose top-left corner is ``top_left``, kept inside the image.
    Returns the chip's height, for the next one stacked below it."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    pad = 3
    w, h = round(right - left) + 2 * pad, round(bottom - top) + 2 * pad
    x = min(max(0, top_left[0]), max(0, size[0] - w - 1))
    y = min(max(0, top_left[1]), max(0, size[1] - h - 1))
    draw.rectangle([x, y, x + w, y + h], fill=color)
    draw.text((x + pad - left, y + pad - top), text, fill=_label_ink(color), font=font)
    return h + 1


def _draw_instance_limbs(
    draw: ImageDraw.ImageDraw,
    parsed: list[Annotation],
    templates: Mapping[str, Mapping[str, Any]],
    colors: Mapping[str, str] | None,
    width: int,
) -> None:
    """Connect each keypoint INSTANCE's joints via its class template's edges.

    Limbs are drawn in a pre-pass so every edge passes BEHIND the joint markers the main
    loop puts down afterwards.

    Two rules the grouping is here to enforce:

    * an edge only ever joins two joints of the SAME ``instance_id`` - two people's heads
      are not connected, which is the whole reason instance identity is on the wire;
    * an edge is drawn only when BOTH ends are actually placed. A limb running to a joint
      the instance never placed would streak a line to wherever the missing slot's
      placeholder sat.
    """
    for class_name, template in ((n, templates[n]) for n in sorted(templates)):
        names = template_node_names(template)
        edges = template_edge_pairs(template)
        if not names or not edges:
            continue
        color = _color_for(class_name, colors)
        for instance in group_instances(parsed, names):
            if match_template(instance.points, templates) != class_name:
                continue
            slots, _ = slot_instance(instance.points, names)
            for i, j in edges:
                a, b = slots[i], slots[j]
                if a is None or b is None:
                    continue
                draw.line(
                    [(a.keypoint.x, a.keypoint.y), (b.keypoint.x, b.keypoint.y)],
                    fill=color,
                    width=width,
                )


def draw_annotations(
    image: str | Path | PILImage,
    annotations: Iterable[AnnotationInput] = (),
    *,
    classification: ClassificationInput | Iterable[ClassificationInput] | None = None,
    colors: Mapping[str, str] | None = None,
    width: int = 3,
    show_labels: bool = True,
    show_confidence: bool = False,
    font_size: float = DEFAULT_LABEL_FONT_SIZE,
    keypoint_templates: Mapping[str, Mapping[str, Any]] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> PILImage:
    """Render annotations onto a copy of ``image`` and return the new image.

    The input image is never mutated - a path/str is opened, a Pillow image is
    copied. Coordinates are absolute pixels in the image's own space (the same
    space Pictograph stores), so do not pre-rotate via EXIF.

    Args:
        image: A file path, or an already-open Pillow ``Image``.
        annotations: An iterable of annotation models or raw annotation dicts
            (each is validated into the right type via the discriminated union).
        classification: A classifier's output - its :class:`ClassificationResult`, one
            :class:`ClassScore` or a ``{"name", "confidence"}`` mapping - or several (one per
            classifier). Each of its classes at or over ``min_confidence`` is a badge in the
            image's top-left corner, inset 1.5% of its shorter side (at least 8 px), its words ~2%
            of its longer side tall, stacked best first in the order given, each class once: the
            way Pictograph's live views and annotated videos draw a classifier. Follows
            ``show_labels`` and ``show_confidence`` like every label.
        colors: Optional per-class color overrides (``{"car": "#ff0000"}``);
            classes not listed fall back to the stable palette color.
        width: Outline stroke width in pixels.
        show_labels: Draw the class-name chip on each annotation (and a classifier's badge): at the
            annotation's top centre - centred over its box, just above it (inside it when the image's
            top leaves no room) - written dark on a light class color, as Pictograph's live views draw it.
        show_confidence: Append each annotation's confidence to its label (e.g.
            ``"car 73%"``) - handy for eyeballing which predictions are
            low-confidence. Requires ``show_labels`` (the default).
        font_size: Label font size in pixels (default 17.5: a quarter larger than it was, as
            Pictograph's live views draw their labels since 2026-10-31).
        keypoint_templates: Optional per-class keypoint templates (class name →
            ``{"nodes": [{"name": ...}], "edges": [[i, j]]}``, exactly the shape
            ``project_config`` stores). Joints sharing an ``instance_id`` are connected
            by their class's edges. Omit it and each joint still renders as its own
            marker - only the connectivity is missing.
        min_confidence: The confidence a classifier's class must reach to be a badge (default
            0.5, Pictograph's default threshold; ``0`` draws every class it reports).

    Returns:
        A new ``PIL.Image.Image`` (RGB) with the annotations drawn on it.

    Raises:
        TypeError: ``classification`` holds something that is not a classifier's output.
    """
    if isinstance(image, (str, Path)):
        base = _PILImage.open(image).convert("RGB")
    else:
        base = image.convert("RGB")
    draw = ImageDraw.Draw(base)
    font = _load_font(font_size) if show_labels else None

    parsed = [_ANN_ADAPTER.validate_python(raw) for raw in annotations]
    labels = _corner_labels(classification, min_confidence)
    if keypoint_templates:
        _draw_instance_limbs(draw, parsed, keypoint_templates, colors, width)

    for ann in parsed:
        color = _color_for(ann.name, colors)

        if isinstance(ann, BBoxAnnotation):
            box = ann.bounding_box
            draw.rectangle(
                [box.x, box.y, box.x + box.w, box.y + box.h],
                outline=color,
                width=width,
            )
        elif isinstance(ann, PolygonAnnotation):
            for ring in ann.polygon.paths:
                pts = [(p.x, p.y) for p in ring]
                if len(pts) >= 2:
                    draw.line([*pts, pts[0]], fill=color, width=width)  # close the ring
        elif isinstance(ann, PolylineAnnotation):
            pts = [(p.x, p.y) for p in ann.polyline.path]
            draw.line(pts, fill=color, width=width)  # open path - no closure
        elif isinstance(ann, KeypointAnnotation):
            kp = ann.keypoint
            r = max(3, width + 2)
            draw.ellipse([kp.x - r, kp.y - r, kp.x + r, kp.y + r], outline=color, width=width)
            draw.line([kp.x - r, kp.y, kp.x + r, kp.y], fill=color, width=max(1, width - 1))
            draw.line([kp.x, kp.y - r, kp.x, kp.y + r], fill=color, width=max(1, width - 1))

        if show_labels and font is not None:
            label = f"{ann.name} {ann.confidence:.0%}" if show_confidence else ann.name
            # Its gap above the annotation: the outline's width, at least 2 px (the live view's).
            _draw_label(draw, font, label, _top_center(ann, width), color, base.size, max(2, width))

    # Last, over everything: it labels the whole image.
    if show_labels and labels:
        corner_font = _load_font(
            max(CORNER_LABEL_MIN_FONT, round(CORNER_LABEL_FONT_SHARE * max(base.size)))
        )
        inset = max(8, round(CORNER_LABEL_INSET * min(base.size)))
        y = inset
        for score in labels:
            text = f"{score.name} {score.confidence:.0%}" if show_confidence else score.name
            y += _draw_corner_label(
                draw, corner_font, text, (inset, y), _color_for(score.name, colors), base.size
            )
            y += max(2, width)

    return base
