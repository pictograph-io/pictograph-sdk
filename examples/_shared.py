"""Support helpers for the runnable examples - not an example itself.

The examples generate their own demo images with Pillow (a base dependency of
the SDK) so every script runs with nothing but ``PICTOGRAPH_API_KEY`` set - no
image files to download first.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

from pictograph.exceptions import ConflictError

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def reusing_existing(what: str) -> Iterator[None]:
    """Treat "already exists" as success, so every example re-runs cleanly.

    The API answers a repeated create with 409 ``ConflictError``. For a demo
    script that is not a failure - it means the previous run already did this
    step - so the examples reuse what is there instead of erroring out. Applies
    to datasets, image uploads and exports alike, which is why it lives here
    rather than being rewritten in each script.
    """
    try:
        yield
    except ConflictError:
        print(f"{what} already exists - reusing it.")


# A small, fixed palette so the generated images are deterministic run to run.
_COLORS = [(219, 68, 55), (66, 133, 244), (52, 168, 83), (251, 188, 5)]


def demo_image(
    path: str | Path,
    *,
    size: tuple[int, int] = (640, 480),
    shapes: int = 3,
    seed: int = 0,
) -> Path:
    """Write a deterministic synthetic image to ``path`` and return it.

    The image has a dark background with a few solid rectangles - enough for a
    real upload, embedding pass and auto-tag to run against, with no external
    asset needed.
    """
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    canvas = Image.new("RGB", size, (24, 27, 36))
    draw = ImageDraw.Draw(canvas)
    width, height = size
    for i in range(shapes):
        rng = (seed * 31 + i * 97) % 100
        x = int((rng / 100) * (width * 0.5)) + 20
        y = int(((rng * 3) % 100) / 100 * (height * 0.5)) + 20
        w = width // 4
        h = height // 4
        draw.rectangle([x, y, x + w, y + h], fill=_COLORS[i % len(_COLORS)])

    canvas.save(out, format="JPEG", quality=90)
    return out
