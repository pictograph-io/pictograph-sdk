"""Near-duplicate / data-curation detection models.

Describe what ``GET /developer/datasets/{dataset_id}/duplicates`` returns: the
server-side near-duplicate scan (a SigLIP2-embedding k-NN self-join) grouped
into connected-component clusters. Use these to find visually redundant images
and keep one per cluster + archive the rest - cutting annotation volume +
dataset bloat before labeling.

Response models use ``extra="ignore"`` so a new backend field doesn't break
older SDK versions.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DuplicateMember(BaseModel):
    """One image in a near-duplicate cluster (the metadata a review UI needs)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    filename: str = ""
    virtual_directory_path: str = ""
    status: str = ""
    annotation_count: int = 0


class DuplicateGroup(BaseModel):
    """A cluster of near-duplicate images - keep one, the rest are redundant."""

    model_config = ConfigDict(extra="ignore")

    members: list[DuplicateMember] = Field(default_factory=list)
    size: int = 0
    max_similarity: float = 0.0  # max pairwise cosine similarity in the cluster [0,1]


class NearDuplicatesResult(BaseModel):
    """Near-duplicate clusters for a dataset + headline data-curation counts.

    ``redundant_count`` is how many images you could archive by keeping one per
    cluster (= sum of ``size - 1``). ``sample_capped`` / ``pairs_capped`` flag
    when the scan hit a bound (no silent caps - raise ``sample`` to scan more).
    """

    model_config = ConfigDict(extra="ignore")

    groups: list[DuplicateGroup] = Field(default_factory=list)
    group_count: int = 0
    duplicate_image_count: int = 0  # total images that belong to some cluster
    redundant_count: int = 0  # keep-one-per-cluster savings = sum(size - 1)
    analyzed: int = 0  # # source images actually scanned
    total_images: int = 0  # non-archived images in the dataset
    sample_limit: int = 0
    sample_capped: bool = False
    pairs_capped: bool = False
    threshold: float = 0.0
    # Directory scope: the virtual directory the scan was restricted to, or None
    # when the whole dataset was scanned.
    directory_path: str | None = None
