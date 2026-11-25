"""THE annotation look, for the SDK: its config (``Look``) and its Pillow renderer.

Public through ``pictograph.viz`` (``draw_annotations(..., look=...)``, ``Look``, ``LOOK_PRESETS``).
"""

from pictograph._look.config import LOOK_PRESETS, Look
from pictograph._look.render import PALETTE, Badge, Bone, Shape, class_color, render

__all__ = ["LOOK_PRESETS", "PALETTE", "Badge", "Bone", "Look", "Shape", "class_color", "render"]
