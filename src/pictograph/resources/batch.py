"""Batch resource - bulk move / copy / delete / update on images.

Every method here works on the virtual filesystem layer. Move and update
are database-only - a stored object's path never changes. Copy is a
server-side object copy (no data transfer, ~zero cost - pixels are
duplicated under a new path). Delete defaults to soft-archive;
``permanent=True`` cleans up the actual blob.

The batch surface takes ``dataset`` (a name, not a UUID) so agents
pass strings users gave them. Image IDs *are* UUIDs - they're opaque
handles with no human equivalent.

For very large operations, prefer multiple smaller calls over one giant
one - the backend caps each request at 10,000 images. Failures land in
:attr:`BatchResult.failed` per-image; the call doesn't raise on partial
failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pictograph.models.batch import BatchResult, CopyToDatasetResult, DuplicateHandling
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

_API_PATH = "/api/v1/developer/batch/images"


def _copy_to_dataset_body(
    dataset: str,
    images: Sequence[str],
    into: str,
    directory: str,
) -> dict[str, Any]:
    """The ``/copy-to-dataset`` request (the route refuses any other key), shared by the
    sync and async twins."""
    return {
        "dataset": dataset,
        "images": list(images),
        "into": into,
        "directory": directory,
    }


class Batch(Resource):
    """Bulk image operations on a dataset's virtual filesystem."""

    def move(
        self,
        dataset: str,
        images: Sequence[str],
        *,
        directory: str = "/",
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
            dataset: Dataset name within your org.
            images: Image UUIDs to move (1-10000).
            directory: Destination directory; ``"/"`` for the root.

        Raises:
            NotFoundError: ``dataset`` doesn't exist or no IDs match.
            ForbiddenError: API key lacks member+/admin/owner role.
        """
        body: dict[str, Any] = {
            "dataset": dataset,
            "images": list(images),
            "directory": directory,
        }
        response = self._transport.request("POST", f"{_API_PATH}/move", json=body)
        return self._parse(BatchResult, response)

    def copy(
        self,
        dataset: str,
        images: Sequence[str],
        *,
        directory: str = "/",
        duplicate_handling: DuplicateHandling = "rename",
        copy_annotations: bool = False,
    ) -> BatchResult:
        """Copy images to a different virtual directory via a server-side object copy.

        Server-side copy is instant (no data transfer, no cost) but creates
        new image rows pointing to the copied blobs.

        Args:
            dataset: Dataset name.
            images: Image UUIDs to copy.
            directory: Destination directory.
            duplicate_handling: How to handle filename collisions in target.
                ``"rename"`` (default), ``"skip"``, or ``"overwrite"``.
            copy_annotations: When ``True``, also copy ``annotations_json``
                and ``status``. Defaults to ``False`` - copies start clean.

        Raises:
            NotFoundError, ForbiddenError as for :meth:`move`.
        """
        body: dict[str, Any] = {
            "dataset": dataset,
            "images": list(images),
            "directory": directory,
            "duplicate_handling": duplicate_handling,
            "copy_annotations": copy_annotations,
        }
        response = self._transport.request("POST", f"{_API_PATH}/copy", json=body)
        return self._parse(BatchResult, response)

    def copy_to_dataset(
        self,
        dataset: str,
        images: Sequence[str],
        *,
        into: str,
        directory: str = "/",
    ) -> CopyToDatasetResult:
        """Copy images into another dataset of your organization - the app's grid Copy/Paste.

        The copies SHARE the stored files: nothing is uploaded or duplicated, and they do not
        count toward your image quota a second time. Each copy carries its annotations, tags,
        training split and review stage. A name already in the target folder is renamed ``-{n}``
        (:attr:`~CopyToDatasetResult.renamed`), and the classes the copied labels use that the
        target dataset lacks are added to it as the source defines them
        (:attr:`~CopyToDatasetResult.classes_added`). The source dataset is not changed.

        Unlike :meth:`copy` (a copy of the files into a folder of the SAME dataset, labels
        optional), this always carries the labels and never copies a file. The target dataset
        must exist (:meth:`~pictograph.resources.datasets.Datasets.create`); naming the source
        dataset again copies into one of its own folders.

        Args:
            dataset: The dataset the images are in.
            images: Image UUIDs to copy (1-10000).
            into: The dataset to copy into.
            directory: The folder in the target dataset, created if missing;
                ``"/"`` (the default) is its root.

        Returns:
            :class:`CopyToDatasetResult` - ``processed``, ``failed`` (each id that made no
            copy, with the reason), ``renamed``, ``target_dataset`` and ``classes_added``.

        Raises:
            NotFoundError: either dataset is not in your organization, or none of the ids is
                an image of ``dataset``.
            ValidationError: ``directory`` is not a valid folder path.
            ForbiddenError: API key lacks member+/admin/owner role.
        """
        body = _copy_to_dataset_body(dataset, images, into, directory)
        response = self._transport.request("POST", f"{_API_PATH}/copy-to-dataset", json=body)
        return self._parse(CopyToDatasetResult, response)

    def delete(
        self,
        dataset: str,
        images: Sequence[str],
        *,
        permanent: bool = False,
    ) -> BatchResult:
        """Soft-archive (default) or permanently delete images.

        Soft-archive sets ``is_archived=true`` - images move to the Archive
        tab but remain in storage, recoverable via :meth:`update`. Permanent
        delete cleans up the blob plus all cached thumbnails and **cannot
        be undone**.

        Args:
            dataset: Dataset name.
            images: Image UUIDs to delete.
            permanent: When ``True``, hard delete. Requires admin/owner
                role. Defaults to ``False`` (soft archive).

        Raises:
            NotFoundError: ``dataset`` or no matching images.
            ForbiddenError: ``permanent=True`` without admin/owner role.
        """
        body: dict[str, Any] = {
            "dataset": dataset,
            "images": list(images),
            "permanent": permanent,
        }
        response = self._transport.request("POST", f"{_API_PATH}/delete", json=body)
        return self._parse(BatchResult, response)

    def update(
        self,
        dataset: str,
        images: Sequence[str],
        *,
        status: str | None = None,
        display_name: str | None = None,
        is_archived: bool | None = None,
    ) -> BatchResult:
        """Update metadata fields on a batch of images.

        Pass exactly the fields you want to change - ``None`` is omitted
        from the request entirely.

        Args:
            dataset: Dataset name.
            images: Image UUIDs to update.
            status: New status. Must be one of: ``"new"``, ``"annotate"``,
                ``"review"``, ``"complete"``.
            display_name: Display name override (raw filename remains
                ``filename`` in DB).
            is_archived: ``True`` archives, ``False`` restores from Archive.

        Raises:
            NotFoundError: ``dataset`` or no matching images.
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
            "dataset": dataset,
            "images": list(images),
            "updates": updates,
        }
        response = self._transport.request("PATCH", f"{_API_PATH}/update", json=body)
        return self._parse(BatchResult, response)
