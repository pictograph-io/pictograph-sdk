"""Unit tests for ``pictograph.tile`` - the dataset tiling engine.

Tiling's load-bearing, hard-to-eyeball parts are (1) the grid partitions the
image exactly and (2) each annotation is translated + clipped into the right
tile(s). These are pinned exhaustively per annotation type, plus the overlap,
``min_visibility``, and ``include_empty`` behaviours and the input contracts.
"""

from __future__ import annotations

import pytest
from PIL import Image

from pictograph.models.annotation import (
    BBoxAnnotation,
    KeypointAnnotation,
    PolygonAnnotation,
    PolylineAnnotation,
)
from pictograph.models.common import BoundingBox, Point
from pictograph.tile import Tile, tile_image

# ── helpers ──────────────────────────────────────────────────────────────


def _img(w: int = 100, h: int = 100) -> Image.Image:
    """A deterministic, high-frequency RGB image (so a crop is verifiable)."""
    im = Image.new("RGB", (w, h))
    px = im.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = ((x * 7) % 256, (y * 13) % 256, (x * 17 + y * 31) % 256)
    return im


def _bbox(x=10.0, y=20.0, w=30.0, h=40.0, name="car") -> BBoxAnnotation:
    return BBoxAnnotation(id="b1", name=name, bounding_box=BoundingBox(x=x, y=y, w=w, h=h))


def _poly(corners, name="road") -> PolygonAnnotation:
    ring = [Point(x=cx, y=cy) for cx, cy in corners]
    return PolygonAnnotation(id="p1", name=name, polygon={"paths": [ring]})


def _kp(x, y, name="joint") -> KeypointAnnotation:
    return KeypointAnnotation(id="k1", name=name, keypoint=Point(x=x, y=y))


# ── grid geometry ─────────────────────────────────────────────────────────


def test_two_by_two_produces_four_equal_tiles():
    tiles = tile_image(_img(100, 100), rows=2, cols=2)
    assert len(tiles) == 4
    assert all(t.size == (50, 50) for t in tiles)
    # row-major order, origins at the four quadrants
    assert [(t.row, t.col, t.origin) for t in tiles] == [
        (0, 0, (0, 0)),
        (0, 1, (50, 0)),
        (1, 0, (0, 50)),
        (1, 1, (50, 50)),
    ]


def test_grid_partitions_image_exactly_and_reassembles():
    src = _img(101, 100)  # odd width → uneven tile widths
    tiles = tile_image(src, rows=2, cols=3)
    assert len(tiles) == 6
    # areas sum to the whole image (no gaps, no double-cover)
    assert sum(t.size[0] * t.size[1] for t in tiles) == 101 * 100
    # pasting the tiles back at their origins reconstructs the source byte-exactly
    canvas = Image.new("RGB", src.size)
    for t in tiles:
        canvas.paste(t.image, t.origin)
    assert canvas.tobytes() == src.convert("RGB").tobytes()


def test_one_by_one_returns_whole_image():
    src = _img(60, 40)
    tiles = tile_image(src, rows=1, cols=1)
    assert len(tiles) == 1
    assert tiles[0].size == (60, 40)
    assert tiles[0].image.tobytes() == src.convert("RGB").tobytes()


def test_non_square_grid():
    tiles = tile_image(_img(100, 100), rows=1, cols=4)
    assert len(tiles) == 4
    assert all(t.size == (25, 100) for t in tiles)


# ── annotation translation + routing ──────────────────────────────────────


def test_bbox_translated_into_its_tile_only():
    # A box wholly inside the top-right quadrant of a 100x100 2x2 grid.
    box = _bbox(x=60, y=10, w=20, h=20, name="sign")
    tiles = tile_image(_img(100, 100), [box], rows=2, cols=2)
    by_cell = {(t.row, t.col): t for t in tiles}
    # present in the top-right tile, translated by its origin (50, 0)
    tr = by_cell[(0, 1)].annotations
    assert len(tr) == 1
    b = tr[0].bounding_box
    assert (b.x, b.y, b.w, b.h) == pytest.approx((10.0, 10.0, 20.0, 20.0))
    # absent from every other tile
    assert by_cell[(0, 0)].annotations == []
    assert by_cell[(1, 0)].annotations == []
    assert by_cell[(1, 1)].annotations == []


def test_boundary_spanning_box_clipped_into_all_tiles():
    # A 20x20 box centered on the grid crossing → a 10x10 corner in each tile.
    box = _bbox(x=40, y=40, w=20, h=20, name="ctr")
    tiles = tile_image(_img(100, 100), [box], rows=2, cols=2, min_visibility=0.1)
    by_cell = {(t.row, t.col): t.annotations for t in tiles}
    assert all(len(v) == 1 for v in by_cell.values())
    # top-left tile: box local (40,40)-(50,50) → x=40,y=40,w=10,h=10
    tl = by_cell[(0, 0)][0].bounding_box
    assert (tl.x, tl.y, tl.w, tl.h) == pytest.approx((40.0, 40.0, 10.0, 10.0))
    # bottom-right tile (origin 50,50): local (0,0)-(10,10)
    br = by_cell[(1, 1)][0].bounding_box
    assert (br.x, br.y, br.w, br.h) == pytest.approx((0.0, 0.0, 10.0, 10.0))


def test_min_visibility_drops_slivers():
    # A box (48..68) straddling the vertical midline (x=50): a 2px sliver in the
    # left tile [0,50], the 18px majority in the right tile [50,100].
    box = _bbox(x=48, y=10, w=20, h=20, name="edge")
    tiles = tile_image(_img(100, 100), [box], rows=1, cols=2, min_visibility=0.5)
    left, right = tiles[0].annotations, tiles[1].annotations
    # left sliver (2/20 wide → 0.1 < 0.5) is dropped
    assert left == []
    # right keeps the majority (18/20 wide → 0.9 >= 0.5)
    assert len(right) == 1


def test_keypoint_routed_to_one_tile():
    kp = _kp(75, 25, name="pt")  # top-right quadrant
    tiles = tile_image(_img(100, 100), [kp], rows=2, cols=2)
    by_cell = {(t.row, t.col): t.annotations for t in tiles}
    assert len(by_cell[(0, 1)]) == 1
    p = by_cell[(0, 1)][0].keypoint
    assert (p.x, p.y) == pytest.approx((25.0, 25.0))  # 75-50, 25-0
    assert sum(len(v) for v in by_cell.values()) == 1  # exactly one tile


def test_polygon_clipped_per_tile():
    # A 40x40 square [30..70]^2 straddling the center → clipped into all 4 tiles.
    poly = _poly([(30, 30), (70, 30), (70, 70), (30, 70)])
    tiles = tile_image(_img(100, 100), [poly], rows=2, cols=2, min_visibility=0.0)
    by_cell = {(t.row, t.col): t.annotations for t in tiles}
    assert all(len(v) == 1 for v in by_cell.values())
    # top-left tile: the clipped ring stays within [0,50]^2 and includes the corner
    ring = by_cell[(0, 0)][0].polygon.paths[0]
    assert max(p.x for p in ring) <= 50.0 + 1e-6
    assert max(p.y for p in ring) <= 50.0 + 1e-6


def test_fully_outside_polyline_is_dropped_not_smeared():
    # A polyline living entirely in the right half must NOT appear (clamped) in the
    # left tile - the source-rect pre-filter drops it there.
    line = PolylineAnnotation(
        id="l1", name="lane", polyline={"path": [Point(x=70, y=10), Point(x=90, y=40)]}
    )
    tiles = tile_image(_img(100, 100), [line], rows=1, cols=2)
    left, right = tiles[0].annotations, tiles[1].annotations
    assert left == []  # not smeared onto the boundary
    assert len(right) == 1


# ── overlap ────────────────────────────────────────────────────────────────


def test_overlap_expands_tiles_and_covers_boundary_objects():
    # Without overlap a boundary box is split; with overlap it appears whole in
    # a neighbouring tile.
    box = _bbox(x=45, y=40, w=10, h=10, name="obj")  # straddles the vertical midline
    tiles = tile_image(_img(100, 100), [box], rows=1, cols=2, overlap=0.2)
    # each tile is base 50 wide + 20% (10px) on the inner edge → 60 wide, clamped
    assert tiles[0].size[0] == 60  # left: [0, 60]
    assert tiles[1].size[0] == 60  # right: [40, 100]
    # the whole box (45..55) is inside the left tile [0,60] → present, unclipped
    lb = tiles[0].annotations[0].bounding_box
    assert (lb.x, lb.w) == pytest.approx((45.0, 10.0))


def test_zero_overlap_is_clean_partition():
    tiles = tile_image(_img(80, 80), rows=2, cols=2, overlap=0.0)
    assert [t.size for t in tiles] == [(40, 40)] * 4


# ── include_empty ──────────────────────────────────────────────────────────


def test_include_empty_false_drops_annotationless_tiles():
    box = _bbox(x=10, y=10, w=20, h=20, name="tl")  # only top-left tile
    kept = tile_image(_img(100, 100), [box], rows=2, cols=2, include_empty=False)
    assert len(kept) == 1
    assert (kept[0].row, kept[0].col) == (0, 0)
    assert len(kept[0].annotations) == 1


def test_include_empty_true_keeps_all_tiles():
    box = _bbox(x=10, y=10, w=20, h=20, name="tl")
    all_tiles = tile_image(_img(100, 100), [box], rows=2, cols=2, include_empty=True)
    assert len(all_tiles) == 4


# ── input contracts ────────────────────────────────────────────────────────


def test_accepts_dict_annotations():
    raw = {
        "id": "b1",
        "name": "car",
        "type": "bbox",
        "bounding_box": {"x": 60, "y": 10, "w": 20, "h": 20},
    }
    tiles = tile_image(_img(100, 100), [raw], rows=2, cols=2)
    tr = {(t.row, t.col): t.annotations for t in tiles}[(0, 1)]
    assert isinstance(tr[0], BBoxAnnotation)
    assert tr[0].bounding_box.x == pytest.approx(10.0)


def test_accepts_path(tmp_path):
    p = tmp_path / "src.png"
    _img(60, 40).save(p)
    tiles = tile_image(p, rows=2, cols=2)
    assert len(tiles) == 4
    assert all(t.size == (30, 20) for t in tiles)


def test_does_not_mutate_source_image():
    src = _img(40, 40)
    before = src.tobytes()
    tile_image(src, rows=2, cols=2)
    assert src.tobytes() == before


def test_no_annotations_yields_empty_annotation_lists():
    tiles = tile_image(_img(40, 40), None, rows=2, cols=2)
    assert all(t.annotations == [] for t in tiles)


def test_tile_is_frozen():
    from dataclasses import FrozenInstanceError

    t = tile_image(_img(20, 20), rows=1, cols=1)[0]
    assert isinstance(t, Tile)
    with pytest.raises(FrozenInstanceError):
        t.row = 5  # type: ignore[misc]


# ── validation ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rows": 0, "cols": 2}, "rows and cols"),
        ({"rows": 2, "cols": 0}, "rows and cols"),
        ({"rows": 2, "cols": 2, "overlap": 0.9}, "overlap"),
        ({"rows": 2, "cols": 2, "overlap": -0.1}, "overlap"),
        ({"rows": 2, "cols": 2, "min_visibility": 1.5}, "min_visibility"),
    ],
)
def test_invalid_params_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        tile_image(_img(40, 40), **kwargs)
