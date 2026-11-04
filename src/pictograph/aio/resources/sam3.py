"""Async SAM3 resource - run SAM3 on ANY image: a file, a URL or bytes. Nothing is saved.

Async twin of :class:`pictograph.resources.sam3.Sam3` (see that module for the full
description): a URL is fetched without blocking the event loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pictograph._image_input import ImageInput, read_image_async
from pictograph.models.sam3 import Sam3Output, Sam3Prediction
from pictograph.resources._base import AsyncResource
from pictograph.resources.sam3 import (
    _API_PATH,
    PREDICT_TIMEOUT_SECONDS,
    BoxInput,
    PointInput,
    image_file,
    prompt_form,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class AsyncSam3(AsyncResource):
    """SAM3 on any image you send (async)."""

    async def predict(
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
        """Run SAM3 on one image with ONE prompt (``text``, ``box`` or ``points``).

        See :meth:`pictograph.resources.sam3.Sam3.predict` for every argument.
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
        data, filename = await read_image_async(image)
        return await self._predict_form(data, filename, form)

    async def _predict_form(
        self, data: bytes, filename: str, form: dict[str, Any]
    ) -> Sam3Prediction:
        response = await self._transport.request(
            "POST",
            f"{_API_PATH}/predict",
            data=form,
            files=image_file(data, filename),
            timeout=PREDICT_TIMEOUT_SECONDS,
        )
        return self._parse(Sam3Prediction, response["data"])


__all__ = ["AsyncSam3"]
