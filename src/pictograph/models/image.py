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


def image_path(directory_path: str | None, filename: str) -> str:
    """An image's ADDRESS in its dataset: its filename, preceded by its directory when it
    is not at the root (``val/x.jpg``; ``x.jpg`` at the root). The string every per-image
    call takes as ``image`` (REFERENCE_LANGUAGE R2b)."""
    directory = (directory_path or "").strip("/")
    return f"{directory}/{filename}" if directory else filename


class Image(BaseModel):
    """An image within a Pictograph dataset."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    dataset_id: str | None = Field(
        default=None,
        description="Its dataset's id.",
    )
    filename: str
    status: ImageStatus = Field(
        default="new",
        description="Its labeling stage.",
    )
    split: ImageSplit | None = Field(
        default=None,
        description="train, val or test; None if unassigned.",
    )
    annotation_count: int = Field(default=0)
    min_confidence: float | None = Field(
        default=None,
        description=(
            "The lowest confidence among its annotations, 0 to 1 (1.0 for hand-drawn); "
            "``images.list(min_confidence_lt=...)`` filters by it for a review queue. "
            "None for images saved before it was recorded."
        ),
    )
    file_size: int = Field(default=0, description="File size in bytes.")
    width: int | None = Field(
        default=None,
        description="Width in pixels; None if unknown.",
    )
    height: int | None = Field(
        default=None,
        description="Height in pixels; None if unknown.",
    )
    content_type: str | None = Field(
        default=None, description="Stored MIME type, e.g. 'image/png'."
    )
    directory_path: str | None = Field(
        default=None,
        description="Its directory in the dataset (``/`` for the root).",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags set by people (``images.bulk_tag`` writes them), not auto-tags.",
    )
    is_archived: bool = Field(
        default=False,
        description="True when archived; delete with ``permanent=True`` to remove it for good.",
    )
    image_url: str | None = Field(
        default=None,
        description=(
            "URL of the full image; fetching it needs your X-API-Key "
            "(``images.download`` saves it to disk)."
        ),
    )
    thumbnail_url: str | None = Field(
        default=None,
        description="CDN URL of a thumbnail, about 200 px.",
    )
    annotation_url: str | None = Field(
        default=None,
        description="API URL of its annotations, as JSON.",
    )
    created_at: datetime

    @property
    def path(self) -> str:
        """The image's address in its dataset - pass it as ``image`` to any per-image call."""
        return image_path(self.directory_path, self.filename)
