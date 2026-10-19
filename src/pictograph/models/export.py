"""Export Pydantic model - represents a generated dataset export.

Returned by every method on the :class:`pictograph.resources.exports.Exports`
resource. Response-side schema (``extra="ignore"``) so backend column
additions don't break callers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

ExportFormat = Literal[
    "pictograph",
    "darwin",
    "coco",
    "yolo",
    "yolo_obb",
    "yolo_pose",
    "dota",
    "pascal_voc",
    "cvat",
    "datumaro",
    "labelme",
    "csv",
]
"""Supported export formats. Pin the literal so SDK callers get IDE
autocomplete and the discriminator catches typos at construction time."""

ExportStatus = Literal["pending", "processing", "completed", "failed"]
"""Export lifecycle states reported by the backend."""


class Export(BaseModel):
    """A dataset export - produced asynchronously, downloaded as a ZIP."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    dataset_id: str = Field(validation_alias=AliasChoices("dataset_id", "project_id"))
    dataset_name: str
    name: str
    format: ExportFormat
    include_images: bool = False
    class_filter: list[str] | None = Field(
        default=None,
        description="Subset of class names included in this export, or None for all.",
    )
    status_filter: str | None = Field(
        default=None,
        description="Image status restriction (e.g., 'complete'), or None for all.",
    )
    status: ExportStatus
    error_message: str | None = Field(
        default=None,
        description="Populated when status='failed'.",
    )
    file_size: int | None = Field(default=None, description="ZIP size in bytes.")
    image_count: int | None = Field(default=None)
    annotation_count: int | None = Field(default=None)
    created_at: datetime
    expires_at: datetime | None = Field(
        default=None,
        description="Soft TTL after which the ZIP may be garbage-collected.",
    )
    download_url: str | None = Field(
        default=None,
        description=(
            "Pre-signed download URL. Set on the get/list responses; absent "
            "on the create response (call .get() or .download() to obtain)."
        ),
    )
    organization_id: str | None = None
