"""Batch operation Pydantic models - request status + per-failure details.

Returned by every method on :class:`pictograph.resources.batch.Batch`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DuplicateHandling = Literal["rename", "skip", "overwrite"]
"""How copies handle pre-existing filenames in the target directory.

- ``"rename"`` (default): append ``-2``, ``-3``, … to the new filename.
- ``"skip"``: leave the existing image alone, record the source as failed.
- ``"overwrite"``: delete the existing image before inserting the copy."""


class BatchFailure(BaseModel):
    """One image that failed within a batch operation."""

    model_config = ConfigDict(extra="ignore")

    id: str
    reason: str


class BatchResult(BaseModel):
    """Outcome of a batch operation.

    ``processed`` and ``failed`` partition the input - ``processed`` is the
    count that landed, ``failed`` is the per-image failure list. Inspect
    ``failed`` to retry the subset (or surface to the user).

    On :meth:`~pictograph.resources.batch.Batch.move`, ``renamed`` is how many
    of the moved images had their filename auto-suffixed ``-{n}`` because a
    same-named image already lived in the target directory - the move renames the
    collision instead of failing. ``0`` for every other operation.

    On :meth:`~pictograph.resources.batch.Batch.delete`, ``operation`` states
    what actually happened: ``"archived"`` (the default - images move to the
    Archive tab, recoverable) or ``"deleted"`` (``permanent=True`` - the stored
    blob is gone, irreversibly). If you meant to hard-delete and see
    ``"archived"``, you forgot ``permanent=True``.
    """

    model_config = ConfigDict(extra="ignore")

    success: bool
    processed: int = Field(ge=0)
    failed: list[BatchFailure] = Field(default_factory=list)
    affected_directories: list[str] = Field(default_factory=list)
    renamed: int = Field(default=0, ge=0)
    operation: str | None = None
