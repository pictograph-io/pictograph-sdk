"""Matching an annotation file to the dataset image it describes, by name.

A label set keys each file by a name: an image file name (``a.jpg`` - COCO's
``file_name``, a VOC ``<filename>``), or the label file's own (``a.txt`` for YOLO,
``a.xml`` for Pascal VOC), which shares only the STEM with the image. An exact
image name wins; otherwise the stem names the image when exactly one image has it.

An import REPLACES each image's annotations, so a name that fits more than one image
is never guessed: ``a.jpg`` and ``a.png`` in one dataset make ``a.txt`` ambiguous,
and two folders each holding an ``a.jpg`` make ``a.jpg`` ambiguous. Each is
reported with the reason and left untouched. So is a second file for an image
another file already matched (``a.jpg`` in a COCO file and ``a.txt`` beside it).
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

T = TypeVar("T")


class ImageIndex:
    """The dataset's images by file name, to resolve an annotation file's key."""

    def __init__(self, images: Iterable[tuple[str, str]]) -> None:
        """``images``: ``(file name, image id)`` of every image of the dataset."""
        self._ids: dict[str, list[str]] = {}
        for filename, image_id in images:
            self._ids.setdefault(filename, []).append(image_id)
        self._stems: dict[str, list[str]] = {}
        for filename in self._ids:
            self._stems.setdefault(PurePosixPath(filename).stem, []).append(filename)

    def resolve(self, key: str) -> tuple[str | None, str | None]:
        """``(image id, None)``; ``(None, reason)`` when ambiguous; ``(None, None)`` = no image."""
        name: str | None = key if key in self._ids else None
        if name is None:
            stem = PurePosixPath(key).stem
            hits = self._stems.get(stem, [])
            if len(hits) > 1:
                return None, (
                    f"{len(hits)} images match {key!r} by name ({', '.join(sorted(hits))}); "
                    "name the file after one of them"
                )
            if not hits:
                return None, None
            name = hits[0]
        ids = self._ids[name]
        if len(ids) > 1:
            return None, (
                f"{len(ids)} images of the dataset are named {name!r} (in different folders); "
                "give each a unique name to import onto it"
            )
        return ids[0], None


def match_sources(
    sources: Mapping[str, T], index: ImageIndex
) -> tuple[dict[str, tuple[str, T]], list[str], list[tuple[str, str]]]:
    """Resolve every source file to one image.

    Returns:
        ``matched`` (image id -> (source key, value)), ``unmatched`` source keys (no
        image has that name), and ``refused`` (source key, reason) pairs: ambiguous,
        or a second file for an image already matched.
    """
    matched: dict[str, tuple[str, T]] = {}
    unmatched: list[str] = []
    refused: list[tuple[str, str]] = []
    for key, value in sources.items():
        image_id, why = index.resolve(key)
        if why is not None:
            refused.append((key, why))
        elif image_id is None:
            unmatched.append(key)
        elif image_id in matched:
            refused.append((key, f"the same image as {matched[image_id][0]!r}; one file per image"))
        else:
            matched[image_id] = (key, value)
    return matched, unmatched, refused
