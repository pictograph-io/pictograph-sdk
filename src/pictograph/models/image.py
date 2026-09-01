"""Image Pydantic model - represents a single image inside a dataset.

THE image primitive: every endpoint that returns an image - the images
list/metadata, upload/register, and the dataset ``include_images=True`` embed -
serializes through one backend serializer, so ONE model validates them all.
Canonical field names: ``width``/``height``, ``content_type`` (MIME),
``directory_path``. Response-side schema (``extra="ignore"``) so backend column
additions don't break callers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ImageStatus = Literal["new", "annotate", "review", "complete"]
"""Annotation lifecycle stages stored on an image's ``status``.

Pinned by the database CHECK constraint - backend rejects any other value.
Stage transitions are caller-controlled (the editor's "Stage" selector,
or an explicit SDK status-set call). The annotations save endpoint never
mutates this field.
"""

ImageSplit = Literal["train", "val", "test"]
"""Dataset split (an image's ``split``): the train/val/test partition an
image belongs to, or ``None`` when unassigned. User-controlled (the grid "Split"
selector or :meth:`~pictograph.resources.images.Images.set_split`); filter with
the ``split`` argument to :meth:`~pictograph.resources.images.Images.list`."""


class Image(BaseModel):
    """An image within a Pictograph dataset."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    dataset_id: str | None = Field(
        default=None,
        description="Parent dataset UUID. Always present on API responses; "
        "optional here so the model stays cheap to construct in fixtures.",
    )
    filename: str
    status: ImageStatus = Field(
        default="new",
        description="Annotation lifecycle stage. See :data:`ImageStatus`.",
    )
    split: ImageSplit | None = Field(
        default=None,
        description="Dataset split (train/val/test), or None if unassigned. See :data:`ImageSplit`.",
    )
    annotation_count: int = Field(default=0)
    min_confidence: float | None = Field(
        default=None,
        description=(
            "Active-learning: the MINIMUM per-annotation model confidence over "
            "this image's annotations, in [0, 1] (1.0 = fully certain / human-drawn). "
            "Lower = more uncertain - sort or filter by it (see the ``min_confidence_lt`` "
            "argument to :meth:`~pictograph.resources.images.Images.list`) to build a "
            "human-review queue. ``None`` on older backends / images predating the field."
        ),
    )
    file_size: int = Field(default=0, description="File size in bytes.")
    width: int | None = Field(
        default=None,
        description="Pixel width (None if unknown).",
    )
    height: int | None = Field(
        default=None,
        description="Pixel height (None if unknown).",
    )
    content_type: str | None = Field(
        default=None, description="Stored MIME type, e.g. 'image/png'."
    )
    directory_path: str | None = Field(
        default=None,
        description="Virtual directory path within the dataset (``/`` for root).",
    )
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "USER image tags (the ones the app's tag filters use and "
            ":meth:`~pictograph.resources.images.Images.bulk_tag` writes) - "
            "not the SigLIP2/Gemini auto-tags."
        ),
    )
    is_archived: bool = Field(
        default=False,
        description="True when soft-deleted (archived). Use `permanent=True` to hard delete.",
    )
    image_url: str | None = Field(
        default=None,
        description=(
            "Authenticated URL for fetching the full image bytes. Always "
            "present on API responses (optional here so the model stays cheap "
            "to construct in fixtures). Use client.images.download() to stream "
            "the bytes to disk; do not GET this URL directly without an "
            "X-API-Key header."
        ),
    )
    thumbnail_url: str | None = Field(
        default=None,
        description="CDN URL for a sized thumbnail (typically 200px).",
    )
    annotation_url: str | None = Field(
        default=None,
        description=(
            "API endpoint that returns the annotation JSON file for this image "
            "(an empty set when the image has no annotations)."
        ),
    )
    created_at: datetime
