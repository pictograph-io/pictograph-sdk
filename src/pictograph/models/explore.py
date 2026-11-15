"""The public catalog - datasets and models any organization published to Explore -
and the import of a public dataset into yours.

Returned by :class:`pictograph.resources.explore.Explore` and by
:meth:`pictograph.resources.datasets.Datasets.import_public`. Response-side schemas
(``extra="ignore"``), so a field the API adds later never breaks a caller.

A public dataset or model lives in ANOTHER organization, where a bare name is not
unique, so it is addressed by its qualified public name: ``ref`` -
``"<organization slug>/<slug>"`` - is what ``datasets.import_public`` and
``models.fork`` take.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pictograph.models.dataset import DatasetClass

PublicModelType = Literal[
    "object_detection",
    "semantic_segmentation",
    "instance_segmentation",
    "classification",
    "keypoint_detection",
]
"""The task a public model performs - the ``model_type`` filter of
:meth:`~pictograph.resources.explore.Explore.models`."""

DatasetSort = Literal["recent", "popular", "trending", "alphabetical"]
ModelSort = Literal["recent", "popular", "alphabetical"]

ImportStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


class PublicOrganization(BaseModel):
    """The organization that published a dataset or a model."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    name: str
    slug: str


class PublicDataset(BaseModel):
    """A dataset in the public catalog, as Explore lists it."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    slug: str | None = None
    description: str | None = None
    organization: PublicOrganization | None = None
    image_count: int = Field(default=0, description="Images in the dataset.")
    annotation_count: int = 0
    annotation_types: list[str] = Field(default_factory=list)
    classes: list[DatasetClass] = Field(
        default_factory=list,
        description="Its classes. A list row carries the first few; the detail has them all.",
    )
    class_count: int = 0
    top_auto_tags: list[str] = Field(
        default_factory=list, description="What its images most often show."
    )
    license_id: str | None = Field(
        default=None, description="The license it was published under (e.g. 'cc-by-4.0')."
    )
    star_count: int = 0
    is_fork: bool = False
    hf_repo: str | None = Field(default=None, description="Its Hugging Face mirror, if any.")
    readme: str | None = Field(default=None, description="Its dataset card (detail only).")
    published_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def ref(self) -> str:
        """The qualified public name ``"<organization>/<dataset>"`` an import takes."""
        org = self.organization.slug if self.organization else ""
        return f"{org}/{self.slug or self.name}"


class PublicModel(BaseModel):
    """A ready model in the public catalog, as Explore lists it."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    slug: str | None = None
    description: str | None = None
    organization: PublicOrganization | None = None
    model_type: str
    architecture: str | None = None
    precision: str | None = None
    class_mapping: dict[str, Any] | None = Field(
        default=None, description="Its classes, in index order (``{'classes': [...]}``)."
    )
    metrics: dict[str, Any] = Field(default_factory=dict, description="Its evaluation scores.")
    training_config: dict[str, Any] | None = None
    license_id: str | None = None
    star_count: int = 0
    weights_size_bytes: int | None = None
    is_fork: bool = False
    hf_repo: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def ref(self) -> str:
        """The qualified public name ``"<organization>/<model>"`` ``models.fork`` takes."""
        org = self.organization.slug if self.organization else ""
        return f"{org}/{self.slug or self.name}"

    @property
    def classes(self) -> list[str]:
        """Its class names, in index order."""
        raw = (self.class_mapping or {}).get("classes") or []
        return [str(c.get("name") if isinstance(c, dict) else c) for c in raw]


class PublicModelDetail(PublicModel):
    """One public model, as its own Explore page shows it (``explore.model``)."""

    readme: str | None = Field(default=None, description="Its model card, in Markdown.")
    license_custom_text: str | None = Field(
        default=None, description="The license's text, when license_id is 'custom'."
    )


class PublicImportSource(BaseModel):
    """What an import copied from."""

    model_config = ConfigDict(extra="ignore")

    organization: str | None = None
    dataset: str | None = None
    dataset_id: str | None = None


class PublicDatasetImport(BaseModel):
    """An import of a public dataset into your organization.

    ``status`` is ``pending`` when it was just started, ``running`` while its image rows
    copy, then ``completed`` (``dataset_name`` is ready to use), ``failed`` (with
    ``error``) or ``cancelled``.
    """

    model_config = ConfigDict(extra="ignore")

    import_id: str
    status: ImportStatus | str
    dataset_id: str | None = Field(default=None, description="The copy in your organization.")
    dataset_name: str | None = Field(
        default=None,
        description="The copy's name (a name your organization already used is suffixed "
        "'(imported)').",
    )
    source: PublicImportSource | None = None
    total_images: int = 0
    image_count: int = Field(default=0, description="Images copied so far.")
    progress: float = Field(default=0.0, description="0-100.")
    error: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def done(self) -> bool:
        """It finished, one way or another."""
        return self.status in ("completed", "failed", "cancelled")
