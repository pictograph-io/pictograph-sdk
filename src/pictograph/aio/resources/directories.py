"""Async Directories resource - inspect a project's virtual directory tree.

Async twin of :class:`pictograph.resources.directories.Directories`, including its
addressing: a dataset is named, and a directory is identified by its PATH
(``/train/cars``) - the pair a user can read off the grid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pictograph.models.directory import Directory, DirectoryStats, DirectoryTreeNode

# The path builder is IMPORTED from the sync twin, not restated here. It is pure
# and I/O-free, exactly like the Pydantic decode helpers `AsyncResource` already
# shares, and a second copy is precisely what broke this module: while the sync
# side moved to name-addressing at P1, this one went on resolving UUIDs and
# calling `/directories/list/{id}`, `/directories/tree/{id}` and
# `/directories/{id}` - none of which exist. Every async directory call 404d or
# hit the wrong verb in the published wheel.
from pictograph.resources._base import AsyncResource
from pictograph.resources.directories import _API_PATH, _seg

if TYPE_CHECKING:
    from collections.abc import Sequence


class AsyncDirectories(AsyncResource):
    """Read a project's virtual directory structure (async)."""

    async def list(
        self, dataset_name: str, *, parent_path: str | None = None
    ) -> Sequence[Directory]:
        """List a dataset's directories.

        Args:
            dataset_name: The dataset's name (a UUID is also accepted).
            parent_path: If given, return only the direct children of this
                directory path (``""`` for root-level directories); otherwise every
                directory in the dataset.
        """
        params = {} if parent_path is None else {"parent_path": parent_path}
        response = await self._transport.request("GET", _seg(dataset_name), params=params)
        # The endpoint returns a bare JSON array.
        return self._parse_list(Directory, response if isinstance(response, list) else [])

    async def tree(self, dataset_name: str) -> Sequence[DirectoryTreeNode]:
        """Fetch the dataset's hierarchical directory tree (children nested)."""
        response = await self._transport.request("GET", _seg(dataset_name, suffix="/tree"))
        return self._parse_list(DirectoryTreeNode, response if isinstance(response, list) else [])

    async def stats(
        self, dataset_name: str, directory_path: str, *, include_subdirectories: bool = True
    ) -> DirectoryStats:
        """Image statistics for a directory (and, by default, its subdirectories)."""
        response = await self._transport.request(
            "GET",
            _seg(dataset_name, directory_path, "/stats"),
            params={"include_subdirectories": include_subdirectories},
        )
        return self._parse(DirectoryStats, response)

    async def delete(
        self, dataset_name: str, directory_path: str, *, cascade: bool = False
    ) -> None:
        """Delete a directory. Empty-only by default; ``cascade=True`` moves its
        images to the parent directory (or root) first."""
        await self._transport.request(
            "DELETE", _seg(dataset_name, directory_path), params={"cascade": cascade}
        )

    async def create(self, dataset_name: str, directory_path: str) -> Directory:
        """Create a virtual directory (idempotent, auto-parents). Member+ role.

        Async twin of :meth:`pictograph.resources.directories.Directories.create`.
        """
        response = await self._transport.request(
            "POST",
            f"{_API_PATH}/",
            json={"dataset": dataset_name, "directory_path": directory_path},
        )
        return self._parse(Directory, response.get("data", response))

    async def rename(self, dataset_name: str, directory_path: str, new_name: str) -> Directory:
        """Rename a directory (row + descendants + contained images). Member+ role.

        Async twin of :meth:`pictograph.resources.directories.Directories.rename`.

        Args:
            dataset_name: The dataset's name (a UUID is also accepted).
            directory_path: Full virtual path of the directory, e.g. ``"/train"``.
            new_name: The new name (a single path segment, not a path).
        """
        response = await self._transport.request(
            "PATCH",
            _seg(dataset_name, directory_path, "/rename"),
            json={"new_name": new_name},
        )
        return self._parse(Directory, response.get("data", response))
