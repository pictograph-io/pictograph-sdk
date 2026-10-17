"""Client-side annotation format converters - COCO, YOLO, and Pascal VOC, offline.

Pure, dependency-free converters between external annotation formats and
Pictograph's typed :data:`~pictograph.models.annotation.Annotation` models. They
run entirely on your machine (no API call, no third-party SDK), so you can bring
an existing COCO / YOLO / Pascal VOC dataset into Pictograph's models - then save
it with ``client.annotations.save`` / ``bulk_save`` - or emit those formats from
annotations you already hold::

    from pictograph.formats import from_coco, to_yolo

    imp = from_coco("instances_val.json")  # → {file_name: [Annotation, ...]}
    yolo_txt = to_yolo(imp.annotations["a.jpg"], imp.class_names, 640, 480)

For hole-accurate COCO (RLE) or a full dataset ZIP in any of the 8 formats,
use the server-side export instead: ``client.exports.create(..., format="coco")``.
"""

from __future__ import annotations

from pictograph.formats._coco import CocoImport, from_coco, to_coco
from pictograph.formats._pascal_voc import from_pascal_voc, to_pascal_voc
from pictograph.formats._yolo import from_yolo, to_yolo

__all__ = [
    "CocoImport",
    "from_coco",
    "from_pascal_voc",
    "from_yolo",
    "to_coco",
    "to_pascal_voc",
    "to_yolo",
]
