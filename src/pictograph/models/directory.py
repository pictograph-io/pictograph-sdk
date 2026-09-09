"""Directory Pydantic models - a dataset's virtual directory tree.

Directories are virtual - a stored object's path is immutable, so moving an image
between directories never moves bytes; ``full_path`` is the addressable key. The developer API exposes the READ surface - list / tree / stats - via
:class:`pictograph.resources.directories.Directories`.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class Directory(BaseModel):
    """A single virtual directory in a dataset.

    Wire-shape transition note: the read endpoints (list/tree) still
    emit the legacy ``project_id``/``full_path``; the create/rename mutations emit
    the canonical ``dataset_id``/``directory_path``. The canonical attribute is
    ``dataset_id``; the aliases below let ONE model validate either wire
    shape until the directories reads land on the canonical keys.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    dataset_id: str = Field(validation_alias=AliasChoices("dataset_id", "project_id"))
    organization_id: str | None = None
    name: str
    parent_directory_id: str | None = None
    full_path: str = Field(validation_alias=AliasChoices("full_path", "directory_path"))
    image_count: int = 0
    created_by: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DirectoryTreeNode(BaseModel):
    """A node in the hierarchical directory tree (children nested recursively)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    full_path: str
    image_count: int = 0
    children: list[DirectoryTreeNode] = Field(default_factory=list)


class DirectoryStats(BaseModel):
    """Aggregate image statistics for a directory (and, by default, its subdirectories)."""

    model_config = ConfigDict(extra="ignore")

    total_directories: int
    total_images: int
    total_size_bytes: int
    directories_by_status: dict[str, int] = Field(default_factory=dict)
