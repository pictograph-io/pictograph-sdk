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

:func:`trace_mask` turns a ground-truth instance mask into the rings of one
polygon (pieces and holes) at the platform's tolerance, and :func:`decode_rle`
reads a COCO RLE mask; both need NumPy + OpenCV (``pictograph[inference]``). For a
full dataset ZIP in any export format, use ``client.exports.create(...)``.
"""

from __future__ import annotations

from pictograph.formats._coco import CocoImport, from_coco, to_coco
from pictograph.formats._mask import MaskTrace, decode_rle, trace_mask
from pictograph.formats._pascal_voc import from_pascal_voc, to_pascal_voc
from pictograph.formats._yolo import from_yolo, to_yolo

__all__ = [
    "CocoImport",
    "MaskTrace",
    "decode_rle",
    "from_coco",
    "from_pascal_voc",
    "from_yolo",
    "to_coco",
    "to_pascal_voc",
    "to_yolo",
    "trace_mask",
]
