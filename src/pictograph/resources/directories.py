"""Directories resource - inspect and clean up a dataset's virtual directory tree.

The developer API exposes directory reads (list / tree / stats) plus delete.
Directories are created implicitly when you upload images into a path; create /
rename stay on the web app for now. Cross-org access returns 404 (consistent
with the other developer resources).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import quote

from pictograph.models.directory import Directory, DirectoryStats, DirectoryTreeNode
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Sequence

_API_PATH = "/api/v1/developer/directories"


def _seg(dataset_name: str, directory_path: str = "", suffix: str = "") -> str:
    """`/directories/{dataset}[/{directory}][/suffix]`.

    Every route in this family addresses a directory by its DATASET plus its
    PATH. This resource used to resolve both to UUIDs client-side and call
    `/directories/list/{id}`, `/directories/tree/{id}` and `/directories/{id}` -
    none of which exist. Every method here 404d or hit the wrong verb.

    The directory segment is a `:path` converter, so inner slashes are kept and
    only the leading one is dropped.
    """
    base = f"{_API_PATH}/{quote(dataset_name, safe='')}"
    if directory_path:
        base += "/" + quote(directory_path.lstrip("/"), safe="/")
    return base + suffix


class Directories(Resource):
    """Read a dataset's virtual directory structure."""

    def list(self, dataset_name: str, *, parent_path: str | None = None) -> Sequence[Directory]:
        """List a dataset's directories.

        Args:
            dataset_name: The dataset's name (a UUID also works).
            parent_path: If given, return only the direct children of this
                directory path (``""`` for root-level directories); otherwise every
                directory in the dataset.
        """
        params = {} if parent_path is None else {"parent_path": parent_path}
        response = self._transport.request("GET", _seg(dataset_name), params=params)
        # The endpoint returns a bare JSON array.
        return self._parse_list(Directory, response if isinstance(response, list) else [])

    def tree(self, dataset_name: str) -> Sequence[DirectoryTreeNode]:
        """Fetch the dataset's hierarchical directory tree (children nested)."""
        response = self._transport.request("GET", _seg(dataset_name, suffix="/tree"))
        return self._parse_list(DirectoryTreeNode, response if isinstance(response, list) else [])

    def stats(
        self, dataset_name: str, directory_path: str, *, include_subdirectories: bool = True
    ) -> DirectoryStats:
        """Image statistics for a directory (and, by default, its subdirectories)."""
        response = self._transport.request(
            "GET",
            _seg(dataset_name, directory_path, "/stats"),
            params={"include_subdirectories": include_subdirectories},
        )
        return self._parse(DirectoryStats, response)

    def delete(self, dataset_name: str, directory_path: str, *, cascade: bool = False) -> None:
        """Delete a directory.

        Empty-only by default; ``cascade=True`` first moves the directory's images
        to its parent directory (or root). Raises ``NotFoundError`` for an unknown
        directory, or ``ApiError`` (400) if it has subdirectories - or, without
        ``cascade``, images.
        """
        self._transport.request(
            "DELETE", _seg(dataset_name, directory_path), params={"cascade": cascade}
        )

    def create(self, dataset_name: str, directory_path: str) -> Directory:
        """Create a virtual directory (idempotent - an existing path is returned).

        Parents are auto-created, the same way an upload into a directory path
        does - use this to pre-stage an EMPTY directory structure ahead of
        uploads. Member role or higher.

        Args:
            dataset_name: The dataset's name (a UUID also works).
            directory_path: Full virtual path to create, e.g. ``"/train/positive"``.
        """
        response = self._transport.request(
            "POST",
            f"{_API_PATH}/",
            json={"dataset": dataset_name, "directory_path": directory_path},
        )
        return self._parse(Directory, response.get("data", response))

    def rename(self, dataset_name: str, directory_path: str, new_name: str) -> Directory:
        """Rename a directory - the directory row, every descendant directory's path,
        and every contained image's directory path move together in one call.

        A sibling directory with the target name raises ``ConflictError``.
        Member role or higher.

        Args:
            directory_path: Full virtual path of the directory to rename.
            new_name: The new name (a single path segment, not a path).
        """
        response = self._transport.request(
            "PATCH",
            _seg(dataset_name, directory_path, "/rename"),
            json={"new_name": new_name},
        )
        return self._parse(Directory, response.get("data", response))
