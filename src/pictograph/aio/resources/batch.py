"""Async Batch resource - bulk move / copy / delete / update on images.

Async twin of :class:`pictograph.resources.batch.Batch`. Failures land in
:attr:`BatchResult.failed` per-image; the call doesn't raise on partial failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pictograph.models.batch import BatchResult, DuplicateHandling
from pictograph.resources._base import AsyncResource

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

_API_PATH = "/api/v1/developer/batch/images"


class AsyncBatch(AsyncResource):
    """Bulk image operations on a dataset's virtual filesystem (async)."""

    async def move(
        self,
        dataset_name: str,
        image_ids: Sequence[str],
        *,
        target_directory_path: str = "/",
    ) -> BatchResult:
        """Move images to a different virtual directory (database-only).

        Args:
            dataset_name: Project name within your org.
            image_ids: Image UUIDs to move (1-10000).
            target_directory_path: Destination virtual directory; ``"/"`` for root.

        Raises:
            NotFoundError: ``dataset_name`` doesn't exist or no IDs match.
            ForbiddenError: API key lacks member+/admin/owner role.
        """
        body: dict[str, Any] = {
            "dataset_name": dataset_name,
            "image_ids": list(image_ids),
            "target_directory_path": target_directory_path,
        }
        response = await self._transport.request("POST", f"{_API_PATH}/move", json=body)
        return self._parse(BatchResult, response)

    async def copy(
        self,
        dataset_name: str,
        image_ids: Sequence[str],
        *,
        target_directory_path: str = "/",
        duplicate_handling: DuplicateHandling = "rename",
        copy_annotations: bool = False,
    ) -> BatchResult:
        """Copy images to a different virtual directory via a server-side object copy.

        Args:
            dataset_name: Project name.
            image_ids: Image UUIDs to copy.
            target_directory_path: Destination directory.
            duplicate_handling: Collision policy - ``"rename"`` (default),
                ``"skip"``, or ``"overwrite"``.
            copy_annotations: When ``True``, also copy ``annotations_json`` and
                ``status``. Defaults to ``False`` - copies start clean.

        Raises:
            NotFoundError, ForbiddenError as for :meth:`move`.
        """
        body: dict[str, Any] = {
            "dataset_name": dataset_name,
            "image_ids": list(image_ids),
            "target_directory_path": target_directory_path,
            "duplicate_handling": duplicate_handling,
            "copy_annotations": copy_annotations,
        }
        response = await self._transport.request("POST", f"{_API_PATH}/copy", json=body)
        return self._parse(BatchResult, response)

    async def delete(
        self,
        dataset_name: str,
        image_ids: Sequence[str],
        *,
        permanent: bool = False,
    ) -> BatchResult:
        """Soft-archive (default) or permanently delete images.

        Args:
            dataset_name: Project name.
            image_ids: Image UUIDs to delete.
            permanent: When ``True``, hard delete (requires admin/owner role).
                Defaults to ``False`` (soft archive).

        Raises:
            NotFoundError: ``dataset_name`` or no matching images.
            ForbiddenError: ``permanent=True`` without admin/owner role.
        """
        body: dict[str, Any] = {
            "dataset_name": dataset_name,
            "image_ids": list(image_ids),
            "permanent": permanent,
        }
        response = await self._transport.request("POST", f"{_API_PATH}/delete", json=body)
        return self._parse(BatchResult, response)

    async def update(
        self,
        dataset_name: str,
        image_ids: Sequence[str],
        *,
        status: str | None = None,
        display_name: str | None = None,
        is_archived: bool | None = None,
    ) -> BatchResult:
        """Update metadata fields on a batch of images.

        Pass exactly the fields you want to change - ``None`` is omitted.

        Args:
            dataset_name: Project name.
            image_ids: Image UUIDs to update.
            status: New status - ``"new"``, ``"annotate"``, ``"review"``, ``"complete"``.
            display_name: Display name override.
            is_archived: ``True`` archives, ``False`` restores from Archive.

        Raises:
            NotFoundError: ``dataset_name`` or no matching images.
            ValidationError: No update fields supplied or invalid status.
        """
        updates: dict[str, Any] = {}
        if status is not None:
            updates["status"] = status
        if display_name is not None:
            updates["display_name"] = display_name
        if is_archived is not None:
            updates["is_archived"] = is_archived
        if not updates:
            raise ValueError(
                "At least one of status / display_name / is_archived must be provided."
            )

        body: dict[str, Any] = {
            "dataset_name": dataset_name,
            "image_ids": list(image_ids),
            "updates": updates,
        }
        response = await self._transport.request("PATCH", f"{_API_PATH}/update", json=body)
        return self._parse(BatchResult, response)
