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
are drawn with a filled backing chip for legibility and can be turned off.

A multi-joint POSE is not a fifth type - it is several ``keypoint`` annotations
sharing an ``instance_id``. Pass ``keypoint_templates`` (the per-class
``{"nodes": [...], "edges": [[i, j]]}`` from ``project_config``) and the joints of
each instance are connected by its class's edges; without one the joints still
render, and only the cosmetic connectivity is missing.
"""

from __future__ import annotations

import hashlib
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


def _color_for(name: str, overrides: Mapping[str, str] | None) -> str:
    if overrides and name in overrides:
        return overrides[name]
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()  # noqa: S324 - non-crypto, stable color hash
    return DEFAULT_PALETTE[int(digest, 16) % len(DEFAULT_PALETTE)]


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    # Pillow >= 10 supports a size arg on the bundled bitmap font; older
    # releases ignore it. Fall back so the helper never hard-fails on font.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - only on very old Pillow
        return ImageFont.load_default()


def _draw_label(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    text: str,
    anchor: tuple[float, float],
    color: str,
) -> None:
    """Draw ``text`` with a filled backing chip just above ``anchor``."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    tw, th = right - left, bottom - top
    pad = 2
    x, y = anchor
    # Sit the chip above the anchor; clamp so it never goes off the top edge.
    chip_top = max(0.0, y - th - 2 * pad)
    draw.rectangle(
        [x, chip_top, x + tw + 2 * pad, chip_top + th + 2 * pad],
        fill=color,
    )
    draw.text((x + pad - left, chip_top + pad - top), text, fill="#ffffff", font=font)


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
    annotations: Iterable[AnnotationInput],
    *,
    colors: Mapping[str, str] | None = None,
    width: int = 3,
    show_labels: bool = True,
    show_confidence: bool = False,
    font_size: int = 14,
    keypoint_templates: Mapping[str, Mapping[str, Any]] | None = None,
) -> PILImage:
    """Render annotations onto a copy of ``image`` and return the new image.

    The input image is never mutated - a path/str is opened, a Pillow image is
    copied. Coordinates are absolute pixels in the image's own space (the same
    space Pictograph stores), so do not pre-rotate via EXIF.

    Args:
        image: A file path, or an already-open Pillow ``Image``.
        annotations: An iterable of annotation models or raw annotation dicts
            (each is validated into the right type via the discriminated union).
        colors: Optional per-class color overrides (``{"car": "#ff0000"}``);
            classes not listed fall back to the stable palette color.
        width: Outline stroke width in pixels.
        show_labels: Draw the class-name chip on each annotation.
        show_confidence: Append each annotation's confidence to its label (e.g.
            ``"car 73%"``) - handy for eyeballing which predictions are
            low-confidence. Requires ``show_labels`` (the default).
        font_size: Label font size.
        keypoint_templates: Optional per-class keypoint templates (class name →
            ``{"nodes": [{"name": ...}], "edges": [[i, j]]}``, exactly the shape
            ``project_config`` stores). Joints sharing an ``instance_id`` are connected
            by their class's edges. Omit it and each joint still renders as its own
            marker - only the connectivity is missing.

    Returns:
        A new ``PIL.Image.Image`` (RGB) with the annotations drawn on it.
    """
    if isinstance(image, (str, Path)):
        base = _PILImage.open(image).convert("RGB")
    else:
        base = image.convert("RGB")
    draw = ImageDraw.Draw(base)
    font = _load_font(font_size) if show_labels else None

    parsed = [_ANN_ADAPTER.validate_python(raw) for raw in annotations]
    if keypoint_templates:
        _draw_instance_limbs(draw, parsed, keypoint_templates, colors, width)

    for ann in parsed:
        color = _color_for(ann.name, colors)
        label_anchor: tuple[float, float] | None = None

        if isinstance(ann, BBoxAnnotation):
            box = ann.bounding_box
            draw.rectangle(
                [box.x, box.y, box.x + box.w, box.y + box.h],
                outline=color,
                width=width,
            )
            label_anchor = (box.x, box.y)
        elif isinstance(ann, PolygonAnnotation):
            for ring in ann.polygon.paths:
                pts = [(p.x, p.y) for p in ring]
                if len(pts) >= 2:
                    draw.line([*pts, pts[0]], fill=color, width=width)  # close the ring
            outer = ann.polygon.paths[0]
            label_anchor = (outer[0].x, outer[0].y)
        elif isinstance(ann, PolylineAnnotation):
            pts = [(p.x, p.y) for p in ann.polyline.path]
            draw.line(pts, fill=color, width=width)  # open path - no closure
            label_anchor = pts[0]
        elif isinstance(ann, KeypointAnnotation):
            kp = ann.keypoint
            r = max(3, width + 2)
            draw.ellipse([kp.x - r, kp.y - r, kp.x + r, kp.y + r], outline=color, width=width)
            draw.line([kp.x - r, kp.y, kp.x + r, kp.y], fill=color, width=max(1, width - 1))
            draw.line([kp.x, kp.y - r, kp.x, kp.y + r], fill=color, width=max(1, width - 1))
            label_anchor = (kp.x - r, kp.y - r)

        if show_labels and font is not None and label_anchor is not None:
            label = f"{ann.name} {ann.confidence:.0%}" if show_confidence else ann.name
            _draw_label(draw, font, label, label_anchor, color)

    return base
