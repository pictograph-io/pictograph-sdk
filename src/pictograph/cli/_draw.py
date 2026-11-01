"""``--draw`` for the ``predict`` commands: each prediction drawn on its image in THE annotation look.

The same look Pictograph's live views, annotated videos and the app's image test draw
(:func:`pictograph.viz.draw_annotations`), saved where ``--draw`` says:

- **one image** - ``--draw out.jpg`` writes that file (its format from its suffix); ``--draw
  drawn/`` (any path without an image suffix) writes ``drawn/<the image's name>``;
- **several images, or a dataset** - ``--draw`` is a folder, each picture named after its input
  (a dataset image's address keeps its folders: ``val/x.jpg``); a name met twice gets ``-2``,
  ``-3``;
- **a video** - its sampled frames, drawn, become one video at the sampling rate (so it plays
  in real time): ``--draw out.mp4`` for one video, else ``<folder>/<the video's name>.mp4``.

``--look`` changes the look: a preset's name (``frost``, ``smoke``), the fields that change as
JSON (``'{"box": {"fill": 0}}'``), or ``@look.json``. Each JSON line of a batch names the file
its drawing went to (``drawn``); one image's goes to stderr, so its JSON or table stays clean.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import typer

from pictograph._image_input import is_url, read_image
from pictograph._look.config import LOOK_PRESETS, Look
from pictograph.inference.results import TASK_RESULT_TYPES, build_result
from pictograph.viz import draw_annotations

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from pictograph.cli._inputs import Item, Source

DRAW_HELP = (
    "Draw each prediction on its image (Pictograph's annotation look) and save it: a file for one "
    "image (out.jpg), else a folder each picture is written into, named after its input; a video's "
    "frames become one drawn video (out.mp4, or <folder>/<video>.mp4)."
)
LOOK_HELP = (
    "With --draw: change the look - a preset (frost, smoke), the fields that change as JSON "
    '(\'{"box": {"fill": 0}}\'), or @path to a JSON file. Every field: pictograph.viz.Look.'
)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv"})
# The quality a drawn JPEG is saved at: the drawing's thin lines and small text survive it.
JPEG_QUALITY = 95


def parse_look(value: str | None) -> Look | None:
    """``--look``: a preset's name, JSON, or ``@file`` - checked here, before anything runs."""
    if value is None:
        return None
    text = value.strip()
    if text in LOOK_PRESETS:
        return LOOK_PRESETS[text]
    if text.startswith("@"):
        path = Path(text[1:]).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise typer.BadParameter(
                f"Could not read {path}: {exc.strerror}", param_hint="--look"
            ) from None
    if not text.startswith("{"):
        names = ", ".join(sorted(LOOK_PRESETS))
        raise typer.BadParameter(
            f"{value!r} is not a look: give a preset ({names}), JSON, or @path to a JSON file.",
            param_hint="--look",
        )
    try:
        fields = json.loads(text)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"Not JSON: {exc.msg} (line {exc.lineno}).", param_hint="--look"
        ) from None
    try:
        return Look.model_validate(fields)
    except ValueError as exc:
        first = str(exc).splitlines()
        detail = " ".join(line.strip() for line in first[1:3]) if len(first) > 1 else str(exc)
        raise typer.BadParameter(
            f"Not a field of the look: {detail}", param_hint="--look"
        ) from None


def _input_name(value: str) -> str:
    """The file name an input gives its drawing: a path's name, a URL's last segment, ``image`` for stdin."""
    if value == "-":
        return "image.png"
    if is_url(value):
        name = PurePosixPath(unquote(urlparse(value).path)).name
        return name or "image.png"
    return Path(value).name or "image.png"


def _with_image_suffix(name: str) -> str:
    """``name`` with a suffix Pillow writes; a name without one (or with another) becomes ``.png``."""
    path = PurePosixPath(name)
    return name if path.suffix.lower() in IMAGE_SUFFIXES else f"{path.stem or 'image'}.png"


@dataclass
class Drawer:
    """Draws each result on its image and saves it where ``--draw`` says (one per command run)."""

    target: Path
    source: Source
    look: Look | None
    #: How many times each output name was given out this run (a name met twice gets ``-2``).
    _names: dict[str, int] = field(default_factory=dict)
    _video: Any = None
    _video_input: str | None = None
    _video_path: Path | None = None

    @classmethod
    def open(cls, target: str | None, source: Source, look: str | None) -> Drawer | None:
        """The drawer ``--draw`` asks for, its target checked against what is run (or None)."""
        if target is None:
            if look is not None:
                raise typer.BadParameter("--look needs --draw.", param_hint="--look")
            return None
        path = Path(target).expanduser()
        suffix = path.suffix.lower()
        if source.kind == "video":
            if suffix in IMAGE_SUFFIXES:
                raise typer.BadParameter(
                    f"A video's frames are drawn into a video: give out.mp4 or a folder, not {target}.",
                    param_hint="--draw",
                )
            if suffix in VIDEO_SUFFIXES and len(source.values) > 1:
                raise typer.BadParameter(
                    "Several videos are drawn into a folder, one video each: give a folder.",
                    param_hint="--draw",
                )
        elif not source.single_image and suffix in IMAGE_SUFFIXES:
            raise typer.BadParameter(
                "Several images are drawn into a folder, one picture each: give a folder, not "
                f"{target}.",
                param_hint="--draw",
            )
        elif source.single_image and suffix in VIDEO_SUFFIXES:
            raise typer.BadParameter(
                f"An image is drawn into an image, not {target}.", param_hint="--draw"
            )
        if path.exists() and not path.is_dir() and suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            raise typer.BadParameter(f"{target} is a file, not a folder.", param_hint="--draw")
        return cls(target=path, source=source, look=parse_look(look))

    # ── one item ──────────────────────────────────────────────────────────────────────────────

    def draw(self, item: Item, image: Any, result: Any) -> str:
        """Draw ``result`` on ``image`` (what the model was handed) and save it; the path it went to."""
        if isinstance(image, str) and is_url(image):
            image = read_image(image)[0]  # the model fetched it itself; drawing needs its bytes too
        picture = draw_annotations(image, self._drawable(result), look=self.look)
        if self.source.kind == "video":
            return str(self._write_frame(item, picture))
        path = self._image_path(item)
        path.parent.mkdir(parents=True, exist_ok=True)
        options: dict[str, Any] = {}
        if path.suffix.lower() in (".jpg", ".jpeg"):
            options = {"quality": JPEG_QUALITY}
        picture.save(path, **options)
        return str(path)

    @staticmethod
    def _drawable(result: Any) -> Any:
        """A raw endpoint answer (a dict) as its typed result; anything else as it is."""
        if isinstance(result, dict):
            task = result.get("model_type")
            if task not in TASK_RESULT_TYPES:
                raise ValueError(f"The answer names no model type it can be drawn by ({task!r}).")
            return build_result(result, task=task, source="The deployment")
        return result

    def _image_path(self, item: Item) -> Path:
        if self.source.single_image and self.target.suffix.lower() in IMAGE_SUFFIXES:
            return self.target
        address = str(item.fields.get("input", "image"))
        # A dataset image keeps its folders (its address, ``val/x.jpg``); anything else, its own name.
        relative = (
            PurePosixPath(address)
            if self.source.kind == "dataset"
            else PurePosixPath(_input_name(address))
        )
        folders = [part for part in relative.parent.parts if part not in ("", ".", "..", "/")]
        name = "/".join([*folders, _with_image_suffix(relative.name)])
        seen = self._names.get(name, 0) + 1
        self._names[name] = seen
        if seen > 1:
            stem = PurePosixPath(name)
            name = str(stem.with_name(f"{stem.stem}-{seen}{stem.suffix}"))
        return self.target / name

    # ── a video ───────────────────────────────────────────────────────────────────────────────

    def _write_frame(self, item: Item, picture: PILImage) -> Path:
        import cv2
        import numpy as np

        video = str(item.fields.get("input", "video"))
        if video != self._video_input:
            self.close()
            self._video_input = video
            if self.target.suffix.lower() in VIDEO_SUFFIXES:
                path = self.target
            else:
                stem = PurePosixPath(_input_name(video)).stem or "video"
                path = self.target / f"{stem}.mp4"
                seen = self._names.get(str(path), 0) + 1
                self._names[str(path)] = seen
                if seen > 1:
                    path = self.target / f"{stem}-{seen}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            fps = max(self.source.sample_fps, 0.1)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, picture.size)
            if not writer.isOpened():
                raise ValueError(f"OpenCV could not write a video to {path}.")
            self._video, self._video_path = writer, path
        frame = cv2.cvtColor(np.asarray(picture), cv2.COLOR_RGB2BGR)
        self._video.write(frame)
        assert self._video_path is not None  # noqa: S101 - set with the writer above
        return self._video_path

    def close(self) -> None:
        """Finish the video being written (each video's last frame is flushed here)."""
        if self._video is not None:
            self._video.release()
            self._video = None


__all__ = ["DRAW_HELP", "LOOK_HELP", "Drawer", "parse_look"]
