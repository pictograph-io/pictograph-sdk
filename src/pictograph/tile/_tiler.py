"""The tiling engine - slice one image into an NxM grid of tiles.

Tiling is a dataset *preprocessing* step (the counterpart to Roboflow's "Tile"):
each source image is cut into a grid of smaller tiles so that small objects
(aerial / satellite / microscopy imagery) occupy a larger fraction of each tile
and train better. Every tile carries the annotations that fall inside it, with
their geometry translated into tile-local pixel coordinates and clipped to the
tile frame - reusing the same battle-tested clip/remap machinery the crop
augmentation uses (:func:`pictograph.augment._geometry.remap_annotations`), so a
box that straddles a tile boundary is clipped correctly in every tile and the
geometry logic stays DRY.

Built on **Pillow** alone (the SDK's base dependency) - no numpy, no OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from PIL import Image as _PILImage
from pydantic import TypeAdapter

from pictograph._keypoint import instance_bbox
from pictograph._obb import obb_aabb
from pictograph.augment._geometry import PointFn, remap_annotations
from pictograph.formats._shared import bbox_from_points
from pictograph.models.annotation import (
    Annotation,
    BBoxAnnotation,
    KeypointAnnotation,
    PolygonAnnotation,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from PIL.Image import Image as PILImage

# Validates a list of raw dicts OR existing annotation models into typed models.
_ANN_LIST_ADAPTER: TypeAdapter[list[Annotation]] = TypeAdapter(list[Annotation])

ImageInput = Union[str, Path, "PILImage"]
AnnotationInput = Union["Annotation", "dict[str, Any]"]


def _coerce_image(image: ImageInput) -> PILImage:
    if isinstance(image, (str, Path)):
        return _PILImage.open(image).convert("RGB")
    return image.convert("RGB")


def _translate(dx: float, dy: float) -> PointFn:
    """A point map that shifts source pixels into a tile's local frame."""
    return lambda x, y: (x - dx, y - dy)


def _source_rect(ann: Annotation) -> tuple[float, float, float, float] | None:
    """The annotation's axis-aligned bounds in *source* pixels (``x0,y0,x1,y1``).

    Returns ``None`` when the bounds can't be derived (a degenerate geometry) -
    the caller keeps such an annotation and lets the clip decide.
    """
    if isinstance(ann, BBoxAnnotation):
        if ann.oriented_box is not None:
            # A ROTATED box: enclose the ROTATED corners - NOT w x h. Those are the
            # box's own extents, so a turned box would under-report its real footprint
            # by up to sqrt(2), and a tile it actually overlaps would decide it doesn't.
            return obb_aabb(ann.oriented_box)
        b = ann.bounding_box
        return (b.x, b.y, b.x + b.w, b.y + b.h)
    if isinstance(ann, KeypointAnnotation):
        # A point's extent is DERIVED (the MIN_KEYPOINT_SIDE box), never read from a field
        # it deliberately does not have. Read as a zero-area rect it overlaps NOTHING
        # under the strict `_touches` test below - so a joint landing exactly on a tile
        # seam belonged to neither side and vanished from the dataset.
        #
        # Each joint is tiled independently: a pose's grouping travels on `instance_id`,
        # which `remap_annotations` carries through untouched, so the joints of one object
        # that share a tile arrive still sharing their id.
        point_box = instance_bbox([ann])
        return (
            point_box.x,
            point_box.y,
            point_box.x + point_box.w,
            point_box.y + point_box.h,
        )
    if isinstance(ann, PolygonAnnotation):
        box = bbox_from_points(ann.polygon.paths[0])
    else:  # PolylineAnnotation - exhaustive over the discriminated union
        box = bbox_from_points(ann.polyline.path)
    if box is None:
        return None
    return (box.x, box.y, box.x + box.w, box.y + box.h)


def _touches(ann: Annotation, x0: float, y0: float, x1: float, y1: float) -> bool:
    """True if the annotation overlaps the tile rect at all (edge-touch excluded)."""
    rect = _source_rect(ann)
    if rect is None:
        return True
    ax0, ay0, ax1, ay1 = rect
    return ax0 < x1 and ax1 > x0 and ay0 < y1 and ay1 > y0


@dataclass(frozen=True)
class Tile:
    """One tile cut from a source image, with its geometry-remapped annotations.

    Attributes:
        image: The cropped tile as a fresh RGB Pillow image (never a view into
            the source).
        annotations: The source annotations that fall inside this tile, clipped
            and translated into the tile's local pixel space.
        row: 0-based grid row, top → bottom.
        col: 0-based grid column, left → right.
        origin: ``(x0, y0)`` - the tile's top-left corner in source pixels.
    """

    image: PILImage
    annotations: list[Annotation]
    row: int
    col: int
    origin: tuple[int, int]

    @property
    def size(self) -> tuple[int, int]:
        """The tile's ``(width, height)`` in pixels."""
        return self.image.size


def tile_image(
    image: ImageInput,
    annotations: Iterable[AnnotationInput] | None = None,
    *,
    rows: int = 2,
    cols: int = 2,
    overlap: float = 0.0,
    min_visibility: float = 0.1,
    include_empty: bool = True,
) -> list[Tile]:
    """Slice ``image`` into a ``rows`` x ``cols`` grid of tiles.

    Tiles are returned in row-major order (row 0 left→right, then row 1, …). The
    grid partitions the image exactly; with ``overlap`` each tile additionally
    extends outward by that fraction of its size (clamped to the image edges), so
    an object on a boundary appears whole in at least one tile.

    Each tile's annotations are the source annotations that intersect it,
    translated into tile-local coordinates and clipped to the tile frame - a box
    or polygon straddling a boundary is split correctly across the adjacent
    tiles; an annotation whose visible area drops below ``min_visibility`` of its
    original area in a tile is dropped from that tile.

    Args:
        image: A file path/str or an already-open Pillow image. Never mutated -
            every tile is a fresh crop.
        annotations: Annotation models or raw annotation dicts (validated into
            the typed discriminated union). Defaults to none.
        rows: Number of grid rows (>= 1).
        cols: Number of grid columns (>= 1).
        overlap: Fractional overlap added to each tile edge, ``[0.0, 0.9)``. ``0``
            gives a clean, non-overlapping partition.
        min_visibility: Drop an annotation from a tile when less than this
            fraction of its area survives the clip (``0.0``-``1.0``).
        include_empty: When ``False``, tiles left with no annotations are omitted
            from the result (Roboflow's "exclude tiles without annotations").

    Returns:
        The tiles, row-major. At most ``rows * cols`` entries (fewer when
        ``include_empty=False`` or a tile is degenerately small).

    Raises:
        ValueError: ``rows``/``cols`` < 1, ``overlap`` outside ``[0, 0.9)``, or
            ``min_visibility`` outside ``[0, 1]``.
    """
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must both be >= 1")
    if not 0.0 <= overlap < 0.9:
        raise ValueError("overlap must be in [0.0, 0.9)")
    if not 0.0 <= min_visibility <= 1.0:
        raise ValueError("min_visibility must be in [0.0, 1.0]")

    img = _coerce_image(image)
    width, height = img.size
    anns: list[Annotation] = (
        _ANN_LIST_ADAPTER.validate_python(list(annotations)) if annotations else []
    )

    # Exact integer boundaries partition the image with no gaps or double-cover.
    xs = [round(c * width / cols) for c in range(cols + 1)]
    ys = [round(r * height / rows) for r in range(rows + 1)]

    tiles: list[Tile] = []
    for r in range(rows):
        for c in range(cols):
            bx0, by0, bx1, by1 = xs[c], ys[r], xs[c + 1], ys[r + 1]
            pad_x = round(overlap * (bx1 - bx0))
            pad_y = round(overlap * (by1 - by0))
            ox0 = max(0, bx0 - pad_x)
            oy0 = max(0, by0 - pad_y)
            ox1 = min(width, bx1 + pad_x)
            oy1 = min(height, by1 + pad_y)
            tw, th = ox1 - ox0, oy1 - oy0
            if tw <= 0 or th <= 0:
                continue  # degenerate tile (tiny image / very fine grid) - skip
            crop = img.crop((ox0, oy0, ox1, oy1))
            # Pre-filter annotations that don't touch this tile at all - both a
            # speed-up and a correctness win (a fully-outside open polyline would
            # otherwise be clamped onto the tile edge by the clip).
            candidates = [a for a in anns if _touches(a, ox0, oy0, ox1, oy1)]
            tile_anns = remap_annotations(
                candidates,
                _translate(ox0, oy0),
                float(tw),
                float(th),
                clip=True,
                min_visibility=min_visibility,
            )
            if not tile_anns and not include_empty:
                continue
            tiles.append(Tile(image=crop, annotations=tile_anns, row=r, col=c, origin=(ox0, oy0)))
    return tiles
