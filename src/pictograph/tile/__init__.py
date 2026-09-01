"""Dataset tiling (``pictograph.tile``) - slice images into an NxM grid.

A native, batteries-included **preprocessing** step (the SDK counterpart of
Roboflow's "Tile"): cut every image into a grid of smaller tiles so small
objects occupy more pixels per tile and train better - the standard fix for
aerial, satellite, and microscopy detection. Each tile keeps the annotations
that fall inside it, with geometry translated into tile-local coordinates and
clipped to the tile frame (a box straddling a boundary is split correctly across
the adjacent tiles). Built on the base **Pillow** dependency alone.

Tile a single image::

    from pictograph.tile import tile_image

    tiles = tile_image("aerial.jpg", annotations, rows=2, cols=2, overlap=0.1)
    for t in tiles:
        t.image.save(f"tile_r{t.row}_c{t.col}.jpg")
        print(len(t.annotations), "annotations in this tile")

To tile a whole Pictograph dataset - pulling images + annotations, slicing each,
and uploading the tiles back through the standard ingest pipeline (embeddings,
auto-tags, thumbnails) - use the images resource::

    report = client.images.tile("aerial", rows=2, cols=2, into="aerial-tiled")
    print(report.tiles_created, "tiles generated")
"""

from __future__ import annotations

from pictograph.tile._tiler import Tile, tile_image

__all__ = ["Tile", "tile_image"]
