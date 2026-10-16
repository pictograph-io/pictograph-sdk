"""Dataset Pydantic models - THE typed shape of a Pictograph dataset.

One model per concept: :class:`Dataset` is the single source for the
dataset resource - list rows and detail responses validate against the same
class because the backend serializes both through one serializer. The former
``Project``/``ProjectClass`` models (the same table under a leaked internal
codename) were folded in here for SDK 2.0.

Response models use ``extra="ignore"`` so the SDK is forward-compatible with
backend additions - a new column doesn't break older SDK versions, it just
isn't surfaced as a typed attribute. Internal storage details (``gcs_path``)
are deliberately not modeled.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pictograph.models.image import Image

DatasetAnnotationType = Literal["bbox", "box", "polygon", "polyline", "keypoint"]
"""Annotation types supported on a dataset. ``bbox``/``box`` are aliases - the
backend accepts both for legacy frontend compatibility."""


AttributeInputType = Literal["text", "number", "select", "checkbox"]
"""The input control a class-level ontology attribute uses in the app."""


class ClassAttribute(BaseModel):
    """A class-level ONTOLOGY attribute definition.

    Declares an attribute that annotations of the owning class may carry.
    Annotations store the values as a ``{name: value}`` map (exported to
    COCO/CVAT/Datumaro); this model is the *schema* half.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1, max_length=120)
    type: AttributeInputType = Field(
        default="text",
        description="Input control: text | number | select | checkbox.",
    )
    values: list[str] | None = Field(
        default=None,
        description="Allowed values for a 'select' attribute.",
    )


class DatasetClass(BaseModel):
    """A class definition stored on the dataset's config.

    On reads, ``type`` and ``color`` may be missing on legacy datasets -
    both default to ``None``. On writes, supply ``name`` and ``type`` at
    minimum; the backend assigns default colors if omitted.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1, max_length=120)
    type: str | None = Field(
        default=None,
        description="Annotation type the class is intended for (bbox/polygon/...).",
    )
    color: str | None = Field(
        default=None,
        description="Hex color for UI rendering (e.g. '#e6194b').",
    )
    attributes: list[ClassAttribute] | None = Field(
        default=None,
        description="Class-level ontology: attribute definitions annotations of "
        "this class may carry. Preserved on read so round-trips don't drop it.",
    )


# The dataset's `include_images=True` embed and GET /developer/images/ run
# through ONE backend serializer, so ONE model validates both - the
# former separate DatasetImage summary class is now an alias of the full
# :class:`pictograph.models.image.Image` primitive.
DatasetImage = Image


class Dataset(BaseModel):
    """A Pictograph dataset - a group of images sharing an annotation config.

    Combines the dataset row + its class config into one surface. Counters
    (``image_count``, ``completed_image_count``, ``archived_image_count``,
    ``total_size``) are denormalized server-side. List rows and detail
    responses share this exact shape (detail may additionally embed
    ``images`` when requested).
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    organization_id: str | None = Field(
        default=None,
        description="Owning organization. Always present on API responses; "
        "optional here so the model stays cheap to construct in fixtures.",
    )
    name: str
    description: str | None = None
    annotation_types: list[str] = Field(default_factory=lambda: ["bbox"])
    classes: list[DatasetClass] = Field(
        default_factory=list,
        description=(
            "Class definitions from the dataset config. Empty when the dataset "
            "has not been configured yet."
        ),
    )
    image_count: int = Field(default=0, ge=0, description="Total non-archived images.")
    completed_image_count: int = Field(
        default=0,
        ge=0,
        description="Images with status='complete' (annotation finalised).",
    )
    archived_image_count: int = Field(
        default=0,
        ge=0,
        description="Soft-deleted images (recoverable from the Archive tab).",
    )
    total_size: int = Field(
        default=0,
        ge=0,
        description="Sum of file sizes in bytes across all non-archived images.",
    )
    is_public: bool = Field(default=False, description="Published to Explore (world-readable).")
    #: Dataset-level archive state (hidden from the default list; reversible).
    is_archived: bool = Field(default=False)
    archived_at: datetime | None = None
    storage_class: str = Field(
        default="standard",
        description="Storage class of the image bytes: standard | coldline.",
    )
    images: list[DatasetImage] | None = Field(
        default=None,
        description=(
            "Populated only when fetched with ``include_images=True``. "
            "``None`` when not requested. Use :meth:`pictograph.resources.datasets.Datasets.get` "
            "with ``include_images=True`` to populate."
        ),
    )
    created_at: datetime
    updated_at: datetime | None = None


class DatasetRestoreEstimate(BaseModel):
    """Price quote (integer micro-USD) for restoring a cold dataset.

    Components are individually marked-up server-side and sum exactly to
    ``total_micro_usd``. ``early_delete_micro_usd`` covers the remainder of
    cold storage's 90-day minimum storage duration when restoring early.
    """

    model_config = ConfigDict(extra="ignore")

    operation: str = "dataset_restore"
    cold_bytes: int = 0
    cold_image_count: int = 0
    days_in_cold: float = 0.0
    min_storage_days: float = 90.0
    monthly_savings_micro_usd: int = 0
    retrieval_micro_usd: int = 0
    early_delete_micro_usd: int = 0
    operations_micro_usd: int = 0
    total_micro_usd: int = 0


class DatasetStorageStatus(BaseModel):
    """Cold-storage state of a dataset (``GET /developer/datasets/{id}/storage``).

    ``storage_class='coldline'`` means the dataset's owned image objects sit
    in cold storage: browsing/annotations keep working, byte-heavy operations
    (uploads, exports, auto-annotation) are paused, and images count at a
    discounted rate toward the org quota. ``restore_estimate`` is present
    only while cold and idle.
    """

    model_config = ConfigDict(extra="ignore")

    storage_class: str = "standard"
    storage_state: str = "idle"
    cold_since: datetime | None = None
    cold_bytes: int = 0
    cold_image_count: int = 0
    storage_job_id: str | None = None
    restore_estimate: DatasetRestoreEstimate | None = None


class DatasetStorageTransition(BaseModel):
    """Acknowledgement that a freeze/restore background job started."""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    storage_state: str
    #: For a restore, the µUSD that WILL be charged when the transition
    #: succeeds - the restore fee is billed on success, not up front, so a
    #: failed or cancelled restore costs nothing. ``None`` for a freeze (free).
    quoted_micro_usd: int | None = None
