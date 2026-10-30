"""Connector Pydantic models - V7 / Roboflow validation + import progress.

Returned by :class:`pictograph.resources.connectors.Connectors`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from pictograph.exceptions import ServerError

ConnectorProvider = Literal["v7", "roboflow"]
"""Source providers the connector resource imports from."""

ImportStatus = Literal["processing", "completed", "error", "cancelled"]
"""Lifecycle of an import operation."""

DatasetImportStatus = Literal["pending", "processing", "completed", "error", "cancelled"]
"""Per-dataset status within an import."""

LimitType = Literal["images", "storage", "both"]
"""Which tier limit was exceeded in a check-limits preflight."""


class RemoteDataset(BaseModel):
    """One dataset as listed by the source provider's validate call.

    Field shape mirrors what V7 / Roboflow return - names, slugs, optional
    image counts, optional version (Roboflow only).
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    slug: str
    image_count: int = Field(default=0, ge=0)
    version: int | None = None


class ValidationResult(BaseModel):
    """Outcome of :meth:`Connectors.validate`.

    On ``valid=False``, ``error`` carries the source provider's message
    (e.g. ``"Invalid API key"``) and ``datasets`` is empty.
    """

    model_config = ConfigDict(extra="ignore")

    valid: bool
    workspace: str = ""
    datasets: list[RemoteDataset] = Field(default_factory=list)
    error: str | None = None


class LimitCheckResult(BaseModel):
    """Outcome of :meth:`Connectors.check_limits`.

    ``allowed=True`` means the import can proceed under the current tier.
    Otherwise inspect ``exceeded`` to see which limit blocked it.
    """

    model_config = ConfigDict(extra="ignore")

    allowed: bool
    current_images: int = Field(ge=0)
    image_limit: int = Field(ge=0)
    images_after_import: int = Field(ge=0)
    current_storage_bytes: int = Field(ge=0)
    storage_limit_bytes: int = Field(ge=0)
    storage_after_import_bytes: int = Field(ge=0)
    exceeded: LimitType | None = None


class DatasetImportProgress(BaseModel):
    """Per-dataset progress row inside an :class:`ImportJob`."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # The import-status payload labels this ``project_name`` (the worker's
    # dataset dicts carry no ``name``) and it is null until the row resolves -
    # accept either key and tolerate null so a status poll never fails to parse.
    name: str | None = Field(default=None, validation_alias=AliasChoices("name", "project_name"))
    dataset_id: str | None = Field(
        default=None, validation_alias=AliasChoices("dataset_id", "project_id")
    )
    status: DatasetImportStatus | None = None
    imported: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)


def kicker_import_id(response: dict[str, Any]) -> str:
    """The import's id off the ``/import/start`` kicker - ``import_id``, or ``job``
    from the 2026-08-01..10-02 builds that sent only that key."""
    ref = response.get("import_id") or response.get("job")
    if not ref:
        raise ServerError(
            "Backend import kicker carried no import id (expected 'import_id')",
            response=response,
        )
    return str(ref)


class ImportJob(BaseModel):
    """Snapshot of an import operation - totals + per-dataset breakdown."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # ``import_id``, or ``job`` from builds of 2026-08-01..10-02 (see BatchJob.job_id).
    import_id: str = Field(validation_alias=AliasChoices("import_id", "job"))
    status: ImportStatus
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    total_images: int = Field(default=0, ge=0)
    imported_images: int = Field(default=0, ge=0)
    failed_images: int = Field(default=0, ge=0)
    current_dataset: str = ""
    datasets: list[DatasetImportProgress] = Field(default_factory=list)
