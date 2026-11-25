"""SAM3 resource - run SAM3 on ANY image: a file, a URL or bytes. Nothing is saved.

SAM3 is Pictograph's built-in segmentation model: no training, any class. Prompt it with
a phrase, a box or points and it returns the matching objects as polygons (or boxes)::

    result = client.sam3.predict("shelf.jpg", text="price tag")
    for a in result.annotations:
        print(a.name, a.type, a.confidence)

    client.sam3.predict("https://example.com/street.jpg", box=(120, 80, 200, 160), name="car")
    client.sam3.predict(frame_bytes, points=[(320, 240)], negative_points=[(300, 400)])

The image is SENT - it does not have to be in a dataset, and nothing is written: no
dataset, no image, no annotation. For an image already in a dataset the
:mod:`~pictograph.resources.auto_annotate` prompts run on it in place (nothing to
upload), and :meth:`AutoAnnotate.batch` saves drafts for a whole dataset.

**Cost:** the SAM3 session fee (``pictograph tools describe sam3`` shows it), charged
once per IMAGE - by the SHA-256 of its bytes - when a prompt on it completes. More
prompts on the same image are free, and so is a prompt that fails. The first call
after a quiet spell starts a GPU and can take a minute or two; prompts on the same
image after that are fast.

**Coordinates** - the prompt's and the results' - are absolute pixels of the image as
it is displayed (EXIF orientation applied), ``result.width`` by ``result.height``.
"""

from __future__ import annotations

import mimetypes
from collections.abc import Mapping, Sequence
from typing import Any

from pictograph._image_input import ImageInput, read_image
from pictograph.models.sam3 import Sam3Output, Sam3Prediction
from pictograph.resources._base import Resource

_API_PATH = "/api/v1/developer/sam3"

# A first prompt can start a GPU and embed the image - longer than the default timeout
# (a cold start was measured at 84 s, 2026-10-17).
PREDICT_TIMEOUT_SECONDS = 300.0

BoxInput = Mapping[str, float] | Sequence[float]
"""``{"x", "y", "w", "h"}`` or ``(x, y, w, h)``: absolute pixels, (x, y) the top-left."""

PointInput = Sequence[float]
"""``(x, y)``: absolute pixels."""


def _num(value: float) -> str:
    return f"{float(value):g}"


def _box_field(box: BoxInput, what: str) -> str:
    if isinstance(box, Mapping):
        try:
            values = [box["x"], box["y"], box["w"], box["h"]]
        except KeyError as exc:
            raise ValueError(f"{what} needs x, y, w and h (missing {exc.args[0]!r}).") from None
    else:
        values = list(box)
        if len(values) != 4:
            raise ValueError(f"{what} is (x, y, w, h): four numbers, got {len(values)}.")
    return ",".join(_num(v) for v in values)


def _point_field(point: PointInput, *, negative: bool) -> str:
    values = list(point)
    if len(values) != 2:
        raise ValueError(f"A point is (x, y): two numbers, got {len(values)}.")
    field = ",".join(str(round(float(v))) for v in values)
    return f"{field},neg" if negative else field


def prompt_form(
    *,
    text: str | None,
    box: BoxInput | None,
    points: Sequence[PointInput] | None,
    negative_points: Sequence[PointInput] | None,
    negative_boxes: Sequence[BoxInput] | None,
    name: str | None,
    output: Sam3Output,
    confidence: float | None,
    max_detections: int,
) -> dict[str, Any]:
    """The multipart form fields for ONE prompt (shared by the sync and async resources).

    Raises:
        ValueError: No prompt or more than one; ``negative_points`` without
            ``points``; ``negative_boxes`` without ``box``; a box or point of the
            wrong shape; an ``output`` other than ``polygon`` / ``bbox``.
    """
    given = [
        kind
        for kind, value in (("text", text), ("box", box), ("points", points))
        if value is not None and (kind != "points" or len(value) > 0)
    ]
    if len(given) != 1:
        raise ValueError(
            "Give exactly one prompt: text= (a phrase), box= ((x, y, w, h)) or points= "
            f"([(x, y), ...]). Got {' and '.join(given) if given else 'none'}."
        )
    if negative_points and points is None:
        raise ValueError(
            "negative_points go with points=: they carve regions out of a point prompt."
        )
    if negative_boxes and box is None:
        raise ValueError("negative_boxes go with box=: they carve regions out of a box prompt.")
    if output not in ("polygon", "bbox"):
        raise ValueError(f"output is 'polygon' or 'bbox', got {output!r}.")

    form: dict[str, Any] = {"output": output}
    if text is not None:
        if not text.strip():
            raise ValueError("text must not be blank.")
        form["text"] = text.strip()
        form["max_detections"] = str(max_detections)
    elif box is not None:
        form["box"] = _box_field(box, "box")
        if negative_boxes:
            form["negative_box"] = [_box_field(b, "Each negative box") for b in negative_boxes]
    else:
        form["point"] = [_point_field(p, negative=False) for p in points or ()] + [
            _point_field(p, negative=True) for p in negative_points or ()
        ]
    if name is not None:
        form["name"] = name
    if confidence is not None:
        form["confidence_threshold"] = _num(confidence)
    return form


def image_file(data: bytes, filename: str) -> dict[str, tuple[str, bytes, str]]:
    """The multipart ``file`` part for the image."""
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return {"file": (filename, data, content_type)}


class Sam3(Resource):
    """SAM3 on any image you send - see the module docstring."""

    def predict(
        self,
        image: ImageInput,
        *,
        text: str | None = None,
        box: BoxInput | None = None,
        points: Sequence[PointInput] | None = None,
        negative_points: Sequence[PointInput] | None = None,
        negative_boxes: Sequence[BoxInput] | None = None,
        name: str | None = None,
        output: Sam3Output = "polygon",
        confidence: float | None = None,
        max_detections: int = 50,
    ) -> Sam3Prediction:
        """Run SAM3 on one image with ONE prompt and return the objects it finds.

        Give exactly one of ``text``, ``box`` or ``points``:

        * ``text`` - a short noun phrase (``"red car"``, ``"forklift"``): every match
          comes back, named after the phrase.
        * ``box`` - a rough box around one object: it and every object like it come
          back. ``negative_boxes`` carve regions out.
        * ``points`` - points on ONE object: its mask comes back as one annotation.
          ``negative_points`` exclude regions.

        Args:
            image: A path to an image file, an ``http(s)://`` URL (fetched on this
                machine, then uploaded), or the image's bytes. JPEG, PNG, WebP, BMP,
                GIF or TIFF, up to 25 MB.
            text: Text prompt.
            box: Box prompt, ``{"x", "y", "w", "h"}`` or ``(x, y, w, h)`` in absolute
                pixels.
            points: Point prompt, ``[(x, y), ...]`` on the object.
            negative_points: With ``points``: regions to exclude.
            negative_boxes: With ``box``: regions to exclude.
            name: Class name for the results. Default: the phrase, or ``"object"``.
            output: ``"polygon"`` (default) for mask outlines, ``"bbox"`` for their
                bounding boxes.
            confidence: Minimum score (0-1). Default by prompt: 0.3 for text, 0.5 for
                a box, 0.75 for points (the mask's quality).
            max_detections: Text prompts: the most matches to return, best first
                (1-100).

        Returns:
            :class:`~pictograph.models.sam3.Sam3Prediction` - ``annotations`` plus
            the upright image's ``width`` / ``height`` (the space every coordinate
            is in). Nothing is saved.

        Raises:
            ValueError: No prompt, several, or a malformed box / point.
            FileNotFoundError: ``image`` is a path that does not exist.
            ValidationError: The server refused the image or the prompt (not an
                image it reads, a prompt outside the image, ...).
            PaymentRequiredError: The organization's credits do not cover the
                session fee.
            ForbiddenError: The API key's role is below member.
        """
        form = prompt_form(
            text=text,
            box=box,
            points=points,
            negative_points=negative_points,
            negative_boxes=negative_boxes,
            name=name,
            output=output,
            confidence=confidence,
            max_detections=max_detections,
        )
        data, filename = read_image(image)
        return self._predict_form(data, filename, form)

    def _predict_form(self, data: bytes, filename: str, form: dict[str, Any]) -> Sam3Prediction:
        """One call with a prompt already built: the CLI builds it once for many images."""
        response = self._transport.request(
            "POST",
            f"{_API_PATH}/predict",
            data=form,
            files=image_file(data, filename),
            timeout=PREDICT_TIMEOUT_SECONDS,
        )
        return self._parse(Sam3Prediction, response["data"])


__all__ = [
    "PREDICT_TIMEOUT_SECONDS",
    "BoxInput",
    "PointInput",
    "Sam3",
    "image_file",
    "prompt_form",
]
