"""SAM3 on any image - what :meth:`pictograph.resources.sam3.Sam3.predict` returns."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from pictograph.models.auto_annotate import PromptResult

Sam3PromptKind = Literal["text", "box", "point"]
"""The prompt a :class:`Sam3Prediction` answers: a phrase, a box, or points."""

Sam3Output = Literal["polygon", "bbox"]
"""What each object comes back as: its mask outline, or its bounding box."""


class Sam3Prediction(PromptResult):
    """SAM3's answer to one prompt on an image you sent. Nothing was saved.

    ``status`` is ``"success"`` when ``annotations`` holds at least one object,
    ``"no_detection"`` when SAM3 found nothing for the prompt, and
    ``"below_threshold"`` when it found objects but none scored above
    ``confidence_threshold``.

    Every coordinate - in the annotations and in the prompt you sent - is an
    absolute pixel of the image as it is DISPLAYED: SAM3 applies the EXIF
    orientation first, so a phone photo taken sideways is the upright picture,
    ``width`` by ``height``.
    """

    model_config = ConfigDict(extra="ignore")

    prompt: Sam3PromptKind
    width: int = Field(ge=1, description="The upright image's width in pixels.")
    height: int = Field(ge=1, description="The upright image's height in pixels.")
    image_sha256: str | None = Field(
        default=None,
        description=(
            "SHA-256 of the image bytes - the identity the SAM3 session fee is charged "
            "against once, so more prompts on the same image are free."
        ),
    )
    confidence_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="The minimum score that applied (the per-prompt default when none was given).",
    )
    degraded: bool = Field(
        default=False,
        description=(
            "True when the image held more matches than fit in GPU memory, so only the "
            "most confident came back; ``confidence_threshold`` says where the cut fell."
        ),
    )


__all__ = ["Sam3Output", "Sam3Prediction", "Sam3PromptKind"]
