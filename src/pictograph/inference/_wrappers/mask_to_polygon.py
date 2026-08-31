"""Binary mask → Pictograph polygon path conversion.

Mirrors ``_mask_to_polygon`` / ``_mask_to_all_polygons`` from
``pictograph_sam3_service.py`` (same image-diagonal-tuned RDP simplification:
``epsilon = min(0.002 * perimeter, 0.00216 * image_diagonal)``) so auto-annotate
polygons produced by user-trained models land in the exact same shape as those
produced by SAM3.
"""

from __future__ import annotations

import cv2
import numpy as np


def _as_binary_u8(mask: np.ndarray) -> np.ndarray:
    """A mask -> a fresh, writable ``uint8`` 0/1 array, in ONE full-resolution pass.

    ``(mask > 0.5).astype(np.uint8)`` allocates the full-resolution image TWICE -
    once for the bool result, once for the uint8 copy - and both are hot on the
    semantic-seg path, which runs this per class per component.

    A numpy bool array is one byte per element holding exactly 0 or 1, and the
    comparison already returned a fresh C-contiguous buffer, so reinterpreting it
    as ``uint8`` is free and produces byte-identical values. It is a VIEW of an
    array nobody else holds, which is what makes the in-place ``_clear_edges``
    below still safe.
    """
    return (mask > 0.5).view(np.uint8)


def _clear_edges(binary_mask: np.ndarray) -> None:
    """Zero out border pixels in-place so contours don't stick to the edge."""
    binary_mask[0, :] = 0
    binary_mask[-1, :] = 0
    binary_mask[:, 0] = 0
    binary_mask[:, -1] = 0


def mask_to_polygon(mask: np.ndarray) -> list[dict[str, float]]:
    """Convert a binary mask to a single polygon path (outer contour only).

    Returns the vertices of the largest contour as ``[{"x", "y"}, ...]``,
    simplified with Douglas-Peucker at the canonical Pictograph tolerance.
    Returns an empty list if no contour found.
    """
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    binary_mask = (mask > 0.5).astype(np.uint8)
    _clear_edges(binary_mask)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    largest = max(contours, key=cv2.contourArea)
    h, w = binary_mask.shape[:2]
    image_diagonal = (h * h + w * w) ** 0.5
    epsilon = min(0.0036 * cv2.arcLength(largest, True), 0.00216 * image_diagonal)
    simplified = cv2.approxPolyDP(largest, epsilon, True)
    if len(simplified) < 3:
        # A degenerate mask (a thin line / tiny blob) can simplify to 1-2 points,
        # which is NOT a valid polygon (>=3 vertices). Emit no polygon rather than
        # an invalid one - mirrors mask_to_all_polygons' >=3 guard, the SDK
        # auto-annotate guard, and the editor's sam3Rings check, and the caller
        # already treats [] as "no contour".
        return []
    return [{"x": float(pt[0][0]), "y": float(pt[0][1])} for pt in simplified]


def mask_to_all_polygons(
    mask: np.ndarray,
    min_area_ratio: float = 0.002,
) -> dict | None:
    """Convert a binary mask to a Pictograph polygon annotation body with holes.

    Uses ``RETR_TREE`` so outer contours *and* inner holes are all captured as
    separate rings in a single annotation; the frontend renders them with the
    even-odd fill rule. Returns ``None`` if no qualifying contour exists, or a
    dict ``{"bounding_box": {...}, "polygon": {"paths": [[{x,y}, ...], ...]}}``
    on success. Caller supplies ``id`` / ``name`` / ``type`` wrapper fields.
    """
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    binary_mask = _as_binary_u8(mask)
    _clear_edges(binary_mask)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_h, img_w = binary_mask.shape[:2]
    image_diagonal = (img_h * img_h + img_w * img_w) ** 0.5
    diagonal_eps_cap = 0.00216 * image_diagonal
    area_threshold = img_h * img_w * min_area_ratio

    all_paths: list[list[dict[str, float]]] = []
    global_min_x = float("inf")
    global_min_y = float("inf")
    global_max_x = 0
    global_max_y = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < area_threshold and all_paths:
            continue

        epsilon = min(0.0036 * cv2.arcLength(contour, True), diagonal_eps_cap)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        if len(simplified) < 3:
            continue

        points = [{"x": float(pt[0][0]), "y": float(pt[0][1])} for pt in simplified]
        all_paths.append(points)

        x, y, w, h = cv2.boundingRect(simplified)
        global_min_x = min(global_min_x, int(x))
        global_min_y = min(global_min_y, int(y))
        global_max_x = max(global_max_x, int(x) + int(w))
        global_max_y = max(global_max_y, int(y) + int(h))

    # Fallback: keep the largest contour if nothing passed the threshold
    if not all_paths and contours:
        contour = max(contours, key=cv2.contourArea)
        epsilon = min(0.0036 * cv2.arcLength(contour, True), diagonal_eps_cap)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        if len(simplified) >= 3:
            points = [{"x": float(pt[0][0]), "y": float(pt[0][1])} for pt in simplified]
            all_paths.append(points)
            x, y, w, h = cv2.boundingRect(simplified)
            global_min_x = int(x)
            global_min_y = int(y)
            global_max_x = int(x) + int(w)
            global_max_y = int(y) + int(h)

    if not all_paths:
        return None

    return {
        "bounding_box": {
            "x": global_min_x,
            "y": global_min_y,
            "w": global_max_x - global_min_x,
            "h": global_max_y - global_min_y,
        },
        "polygon": {"paths": all_paths},
    }


# A noisy / low-quality semantic-seg model fragments a class mask into hundreds
# or thousands of tiny connected components. Without a real area floor + a hard
# cap this emits thousands of junk polygons per image (measured LIVE: 4289 on
# one 640×513 image from an 11%-mIoU model, warm latency 8.5s) that pollute
# annotations_json and hammer both the deployment /predict path and the batch
# auto-annotate path (both reach this through the shared single-image dispatch).
# Keep only components above the area floor, and at most this many (largest
# first); a per-image, per-class instance count in the hundreds is already
# extreme (a dense crowd), the thousands are always noise.
MAX_INSTANCES_PER_CLASS = 300


def mask_to_instance_polygons(
    mask: np.ndarray,
    min_area_ratio: float = 0.002,
    max_instances: int = MAX_INSTANCES_PER_CLASS,
) -> list[dict]:
    """Split a class mask into per-instance polygon annotation bodies.

    Each connected component in the binary mask becomes its own annotation:
    holes inside a component are preserved as additional rings within that
    annotation's ``polygon.paths``, but spatially-distinct components are
    returned as separate entries in the list.

    Components below ``min_area_ratio`` of the image are dropped as noise (if
    NONE qualify, the single largest is kept so a legitimately-small lone object
    still yields a polygon), and at most ``max_instances`` are returned (largest
    first). ``connectedComponentsWithStats`` gives per-component area in one pass,
    so the area floor is applied HERE - previously it was silently defeated,
    because ``mask_to_all_polygons`` called on a single-component mask hits its
    "keep the largest contour" fallback and never actually filtered by area.

    Used by semantic segmentation where a single class mask may contain
    multiple disjoint objects that should each be their own annotation.
    """
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    binary_mask = _as_binary_u8(mask)
    _clear_edges(binary_mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return []

    img_h, img_w = binary_mask.shape[:2]
    area_threshold = img_h * img_w * min_area_ratio
    # (label_id, pixel_area) for every real component (label 0 is background).
    comps = [(i, int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, num_labels)]
    qualifying = [c for c in comps if c[1] >= area_threshold]
    if not qualifying:
        # Nothing clears the floor - keep the single largest so a small lone
        # object (or a whole-image-below-threshold mask) still yields a polygon.
        qualifying = [max(comps, key=lambda t: t[1])]
    # Largest first, then cap - thousands of instances per class are never useful.
    qualifying.sort(key=lambda t: t[1], reverse=True)
    if len(qualifying) > max_instances:
        qualifying = qualifying[:max_instances]

    results: list[dict] = []
    for label_id, _area in qualifying:
        component = (labels == label_id).view(np.uint8)
        poly_body = mask_to_all_polygons(component, min_area_ratio=min_area_ratio)
        if poly_body is not None:
            results.append(poly_body)
    return results
