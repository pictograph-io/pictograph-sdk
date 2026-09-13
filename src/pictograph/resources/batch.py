"""Batch resource - bulk move / copy / delete / update on images.

Every method here works on the virtual filesystem layer. Move and update
are database-only - a stored object's path never changes. Copy is a
server-side object copy (no data transfer, ~zero cost - pixels are
duplicated under a new path). Delete defaults to soft-archive;
``permanent=True`` cleans up the actual blob.

The batch surface accepts ``dataset_name`` (a name, not a UUID) so agents
pass strings users gave them. Image IDs *are* UUIDs - they're opaque
handles with no human equivalent.

For very large operations, prefer multiple smaller calls over one giant
one - the backend caps each request at 10,000 images. Failures land in
:attr:`BatchResult.failed` per-image; the call doesn't raise on partial
failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pictograph.models.batch import BatchResult, DuplicateHandling
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

_API_PATH = "/api/v1/developer/batch/images"


class Batch(Resource):
    """Bulk image operations on a dataset's virtual filesystem."""

    def move(
        self,
        dataset_name: str,
        image_ids: Sequence[str],
        *,
        target_directory_path: str = "/",
    ) -> BatchResult:
        """Move images to a different virtual directory.

        Database-only - the stored object paths don't move. The directory hierarchy
        is virtual, organized via :attr:`Image.virtual_directory_path`.

        Filename collisions with the target directory are resolved automatically:
        a moved image whose name already exists there is suffixed ``-{n}``
        (``photo.png`` → ``photo-1.png``) rather than failing the batch. The
        count is reported on :attr:`BatchResult.renamed`; pre-existing images in
        the target directory are never touched.

        Args:
            dataset_name: Dataset name within your org.
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
        response = self._transport.request("POST", f"{_API_PATH}/move", json=body)
        return self._parse(BatchResult, response)

    def copy(
        self,
        dataset_name: str,
        image_ids: Sequence[str],
        *,
        target_directory_path: str = "/",
        duplicate_handling: DuplicateHandling = "rename",
        copy_annotations: bool = False,
    ) -> BatchResult:
        """Copy images to a different virtual directory via a server-side object copy.

        Server-side copy is instant (no data transfer, no cost) but creates
        new image rows pointing to the copied blobs.

        Args:
            dataset_name: Dataset name.
            image_ids: Image UUIDs to copy.
            target_directory_path: Destination directory.
            duplicate_handling: How to handle filename collisions in target.
                ``"rename"`` (default), ``"skip"``, or ``"overwrite"``.
            copy_annotations: When ``True``, also copy ``annotations_json``
                and ``status``. Defaults to ``False`` - copies start clean.

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
        response = self._transport.request("POST", f"{_API_PATH}/copy", json=body)
        return self._parse(BatchResult, response)

    def delete(
        self,
        dataset_name: str,
        image_ids: Sequence[str],
        *,
        permanent: bool = False,
    ) -> BatchResult:
        """Soft-archive (default) or permanently delete images.

        Soft-archive sets ``is_archived=true`` - images move to the Archive
        tab but remain in storage, recoverable via :meth:`update`. Permanent
        delete cleans up the blob plus all cached thumbnails and **cannot
        be undone**.

        Args:
            dataset_name: Dataset name.
            image_ids: Image UUIDs to delete.
            permanent: When ``True``, hard delete. Requires admin/owner
                role. Defaults to ``False`` (soft archive).

        Raises:
            NotFoundError: ``dataset_name`` or no matching images.
            ForbiddenError: ``permanent=True`` without admin/owner role.
        """
        body: dict[str, Any] = {
            "dataset_name": dataset_name,
            "image_ids": list(image_ids),
            "permanent": permanent,
        }
        response = self._transport.request("POST", f"{_API_PATH}/delete", json=body)
        return self._parse(BatchResult, response)

    def update(
        self,
        dataset_name: str,
        image_ids: Sequence[str],
        *,
        status: str | None = None,
        display_name: str | None = None,
        is_archived: bool | None = None,
    ) -> BatchResult:
        """Update metadata fields on a batch of images.

        Pass exactly the fields you want to change - ``None`` is omitted
        from the request entirely.

        Args:
            dataset_name: Dataset name.
            image_ids: Image UUIDs to update.
            status: New status. Must be one of: ``"new"``, ``"annotate"``,
                ``"review"``, ``"complete"``.
            display_name: Display name override (raw filename remains
                ``filename`` in DB).
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
        response = self._transport.request("PATCH", f"{_API_PATH}/update", json=body)
        return self._parse(BatchResult, response)
