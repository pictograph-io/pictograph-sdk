"""Pascal VOC ⇄ Pictograph annotation conversion (client-side, offline).

Pascal VOC stores one XML file per image with axis-aligned bounding boxes in
absolute pixel corners (``xmin, ymin, xmax, ymax``). :func:`from_pascal_voc`
parses that into Pictograph :data:`~pictograph.models.annotation.Annotation`
objects; :func:`to_pascal_voc` emits the XML for one image. Bounding boxes
round-trip exactly. Pascal VOC has no polygon / polyline / keypoint form, so
:func:`to_pascal_voc` writes a polygon as its enclosing box and skips the other
types (matching how the YOLO detection converter degrades geometry to boxes).

Pure stdlib (``xml.etree.ElementTree``) - no third-party dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

from pictograph.models.annotation import Annotation, BBoxAnnotation
from pictograph.models.common import BoundingBox

from ._shared import annotation_bbox

if TYPE_CHECKING:
    from collections.abc import Sequence


def from_pascal_voc(xml: str) -> list[Annotation]:
    """Parse one Pascal VOC XML annotation file into Pictograph annotations.

    Args:
        xml: The contents of a Pascal VOC ``.xml`` file (an ``<annotation>``
            document with ``<object>`` children).

    Returns:
        One :class:`BBoxAnnotation` per ``<object>`` with a valid ``<bndbox>``.
        Objects without a name or a positive-area box are skipped.

    Raises:
        ValueError: ``xml`` is not well-formed XML.
    """
    try:
        root = ET.fromstring(xml)  # noqa: S314 - user's own local annotation file
    except ET.ParseError as e:
        raise ValueError(f"Invalid Pascal VOC XML: {e}") from e

    out: list[Annotation] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        box_el = obj.find("bndbox")
        if name_el is None or not (name_el.text or "").strip() or box_el is None:
            continue
        xmin = _float(box_el, "xmin")
        ymin = _float(box_el, "ymin")
        xmax = _float(box_el, "xmax")
        ymax = _float(box_el, "ymax")
        if xmin is None or ymin is None or xmax is None or ymax is None:
            continue
        w, h = xmax - xmin, ymax - ymin
        if w <= 0 or h <= 0:
            continue
        out.append(
            BBoxAnnotation(
                name=(name_el.text or "").strip(),
                bounding_box=BoundingBox(x=xmin, y=ymin, w=w, h=h),
            )
        )
    return out


def to_pascal_voc(
    annotations: Sequence[Annotation],
    *,
    filename: str,
    image_width: int,
    image_height: int,
    directory: str = "",
    depth: int = 3,
) -> str:
    """Serialize one image's annotations to a Pascal VOC XML document.

    Args:
        annotations: The image's annotations. Boxes and polygons (as their
            enclosing box) are written; polylines and keypoints are skipped.
        filename: The image file name recorded in ``<filename>``.
        image_width: Image width in pixels (``<size><width>``).
        image_height: Image height in pixels (``<size><height>``).
        directory: Optional ``<directory>`` value.
        depth: Channel count for ``<size><depth>`` (default 3 = RGB).

    Returns:
        A Pascal VOC XML string (UTF-8, no XML declaration).
    """
    root = ET.Element("annotation")
    ET.SubElement(root, "directory").text = directory
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(int(image_width))
    ET.SubElement(size, "height").text = str(int(image_height))
    ET.SubElement(size, "depth").text = str(int(depth))

    for ann in annotations:
        if ann.type in ("polyline", "keypoint"):
            continue
        box = annotation_bbox(ann)
        if box is None:
            continue
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = ann.name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = _fmt(box.x)
        ET.SubElement(bnd, "ymin").text = _fmt(box.y)
        ET.SubElement(bnd, "xmax").text = _fmt(box.x + box.w)
        ET.SubElement(bnd, "ymax").text = _fmt(box.y + box.h)

    return ET.tostring(root, encoding="unicode")


def _float(parent: ET.Element, tag: str) -> float | None:
    """Parse ``<tag>`` as a float, or ``None`` if missing / non-numeric."""
    el = parent.find(tag)
    if el is None or el.text is None:
        return None
    try:
        return float(el.text)
    except ValueError:
        return None


def _fmt(value: float) -> str:
    """Pascal VOC boxes are integer pixels; round to the nearest int."""
    return str(round(value))
