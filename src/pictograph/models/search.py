"""Search Pydantic models - typed results from similarity + tag queries.

Backend strips ``gcs_image_path`` (internal storage detail) before this
SDK ever sees the row, so neither :class:`SimilarImage` nor :class:`TaggedImage`
exposes it. Use :class:`pictograph.resources.images.Images` to download
actual bytes.
"""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from pictograph.models.image import ImageStatus


class SimilarImage(BaseModel):
    """One result from :meth:`pictograph.resources.search.Search.by_similarity`.

    ``similarity`` is cosine similarity in ``[0, 1]`` - 1.0 means identical
    embeddings (the query image is excluded from results).
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    filename: str
    directory_path: str = Field(
        default="/",
        description="Its directory in the dataset.",
        validation_alias=AliasChoices("directory_path", "virtual_directory_path"),
    )
    status: ImageStatus
    annotation_count: int = Field(ge=0)
    auto_tags: dict[str, Any] = Field(
        default_factory=dict,
        description="Its auto-tags, by kind: objects, scenes and attributes.",
        validation_alias=AliasChoices("auto_tags", "image_auto_tags"),
    )
    similarity: float = Field(ge=0.0, le=1.0)


class TaggedImage(BaseModel):
    """One result from :meth:`pictograph.resources.search.Search.by_tag`.

    Includes the source ``dataset_id``: a tag search without ``dataset`` spans every
    dataset.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    dataset_id: str = Field(validation_alias=AliasChoices("dataset_id", "project_id"))
    filename: str
    directory_path: str = Field(
        default="/",
        description="Its directory in the dataset.",
        validation_alias=AliasChoices("directory_path", "virtual_directory_path"),
    )
    status: ImageStatus
    annotation_count: int = Field(ge=0)
    auto_tags: dict[str, Any] = Field(
        default_factory=dict,
        description="Its auto-tags, by kind: objects, scenes and attributes.",
        validation_alias=AliasChoices("auto_tags", "image_auto_tags"),
    )
