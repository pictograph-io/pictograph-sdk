"""Ground-truth masks to Pictograph polygons: COCO RLE decoding and the mask tracer.

A mask that IS the label (a dataset's ground truth, not a model's prediction)
becomes ONE polygon annotation per instance: its separate pieces and its holes are
rings of that one polygon, filled even-odd. Each ring is traced by OpenCV at the
platform's mask tolerance, ``min(0.0036 x perimeter, 0.00216 x image diagonal)``
(the same constants as the platform's model-mask tracer), but unlike a model's mask
nothing is cleared at the frame border and nothing is dropped for its size: a
contour too small for the tolerance keeps its exact traced outline, and only a
contour no polygon can hold (fewer than 3 points, or no area) is left out, counted
in :attr:`MaskTrace.dropped`.

Rings come out largest outer boundary first, then the other pieces, then the holes,
so a reader that takes ``paths[0]`` as "the outline" gets the main body.

Needs NumPy and OpenCV: ``pip install "pictograph[inference]"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pictograph.models.common import Point

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The platform's mask tolerance (LOCKSTEP the platform's model-mask tracer).
EPS_PERIMETER = 0.0036
EPS_DIAGONAL = 0.00216

_NEEDS = 'needs NumPy and OpenCV: pip install "pictograph[inference]"'


def _np_cv2() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(f"Tracing a mask {_NEEDS}") from exc
    return np, cv2


@dataclass(frozen=True)
class MaskTrace:
    """One instance mask, traced.

    Attributes:
        rings: The polygon's rings, in frame pixels: the largest outer boundary,
            then the other pieces, then the holes. Empty when no polygon can hold
            any part of the mask.
        dropped: Contours left out because no polygon can hold them (under 3
            points or zero area; at most a pixel or two wide).
    """

    rings: list[list[Point]] = field(default_factory=list)
    dropped: int = 0


def _counts_from_string(s: str) -> list[int]:
    """COCO's compressed RLE string to run lengths (pycocotools ``rleFrString``)."""
    counts: list[int] = []
    p = 0
    while p < len(s):
        x = 0
        k = 0
        more = True
        while more:
            if p >= len(s):
                raise ValueError("RLE counts string ends inside a run length")
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and c & 0x10:
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def decode_rle(segmentation: Mapping[str, Any]) -> Any:
    """A COCO RLE ``segmentation`` (``{"size": [h, w], "counts": ...}``) to a boolean
    ``(h, w)`` NumPy mask.

    ``counts`` is either COCO's compressed string or the uncompressed list of run
    lengths; runs alternate background / foreground, column-major, starting with
    background (pycocotools' layout).

    Raises:
        ValueError: The runs do not cover exactly ``h x w`` pixels.
        ImportError: NumPy is not installed.
    """
    np, _ = _np_cv2()
    h, w = (int(v) for v in segmentation["size"])
    raw = segmentation["counts"]
    if isinstance(raw, bytes):
        raw = raw.decode("ascii")
    counts = _counts_from_string(raw) if isinstance(raw, str) else [int(c) for c in raw]
    if any(c < 0 for c in counts) or sum(counts) != h * w:
        raise ValueError(f"RLE runs cover {sum(counts)} pixels, the mask is {h}x{w} = {h * w}")
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    for i, run in enumerate(counts):
        if i % 2:
            flat[pos : pos + run] = True
        pos += run
    return flat.reshape((h, w), order="F")


def _shoelace(ring: Any) -> float:
    np, _ = _np_cv2()
    x = ring[:, 0].astype(np.float64)
    y = ring[:, 1].astype(np.float64)
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def trace_mask(
    mask: Any, *, diagonal: float | None = None, offset: tuple[int, int] = (0, 0)
) -> MaskTrace:
    """Trace one instance's mask into the rings of ONE polygon.

    Args:
        mask: A 2-D array; non-zero pixels are the instance.
        diagonal: The image diagonal the tolerance is relative to. Defaults to the
            mask's own; pass the full frame's when ``mask`` is a crop of it.
        offset: ``(x, y)`` added to every vertex - the crop's origin in the frame.

    Returns:
        A :class:`MaskTrace`. Traced in a crop of the mask's bounding box with a
        1 px margin: the same contours a full-frame trace finds, since outside the
        mask reads as background either way.
    """
    np, cv2 = _np_cv2()
    binary = (np.asarray(mask) != 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"mask has shape {binary.shape}, expected 2-D")
    h, w = binary.shape
    if diagonal is None:
        diagonal = float((h * h + w * w) ** 0.5)
    ys, xs = np.nonzero(binary)
    if ys.size == 0:
        return MaskTrace()
    y0, y1 = max(int(ys.min()) - 1, 0), min(int(ys.max()) + 2, h)
    x0, x1 = max(int(xs.min()) - 1, 0), min(int(xs.max()) + 2, w)
    contours, hierarchy = cv2.findContours(
        np.ascontiguousarray(binary[y0:y1, x0:x1]),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
        offset=(x0 + int(offset[0]), y0 + int(offset[1])),
    )
    outers: list[tuple[float, Any]] = []
    holes: list[tuple[float, Any]] = []
    dropped = 0
    for i, contour in enumerate(contours):
        ring = _ring_for(contour, diagonal, cv2)
        if ring is None:
            dropped += 1
            continue
        depth, parent = 0, int(hierarchy[0][i][3])
        while parent != -1:
            depth += 1
            parent = int(hierarchy[0][parent][3])
        (outers if depth % 2 == 0 else holes).append((_shoelace(ring), ring))
    if not outers:
        return MaskTrace(rings=[], dropped=dropped + len(holes))
    outers.sort(key=lambda t: -t[0])
    holes.sort(key=lambda t: -t[0])
    rings = [[Point(x=float(px), y=float(py)) for px, py in r.tolist()] for _, r in outers + holes]
    return MaskTrace(rings=rings, dropped=dropped)


def _ring_for(contour: Any, diagonal: float, cv2: Any) -> Any:
    """A contour to a ring at the platform tolerance, or None when no polygon can
    hold it. A contour too small for the tolerance keeps its exact traced outline."""
    eps = min(EPS_PERIMETER * cv2.arcLength(contour, True), EPS_DIAGONAL * diagonal)
    for candidate in (cv2.approxPolyDP(contour, eps, True), contour):
        ring = candidate.reshape(-1, 2)
        if len(ring) >= 3 and _shoelace(ring) > 0:
            return ring
    return None
