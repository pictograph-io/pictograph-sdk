"""Async Search resource - SigLIP visual similarity + auto-tag filtering.

Async twin of :class:`pictograph.resources.search.Search`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pictograph.aio.resources import _resolve
from pictograph.models.search import SimilarImage, TaggedImage
from pictograph.resources._base import AsyncResource

if TYPE_CHECKING:
    from collections.abc import Sequence

_API_PATH = "/api/v1/developer/search"


class AsyncSearch(AsyncResource):
    """Visual + tag-based image search across the org (async)."""

    async def by_similarity(
        self,
        dataset_name: str,
        image: str,
        *,
        threshold: float = 0.6,
        limit: int = 50,
        directory_path: str | None = None,
    ) -> list[SimilarImage]:
        """Find images visually similar to one of a dataset's images.

        SigLIP-1152 embeddings + a pgvector HNSW index.

        Args:
            dataset_name: The dataset's name (a UUID also works).
            image: The source image's FILENAME (an id also works).
            threshold: Minimum cosine similarity in ``[0, 1]`` (default 0.6).
            limit: Max results (backend cap: 500).
            directory_path: Override the source image's directory. ``"/"`` = dataset root.

        Returns:
            List of :class:`SimilarImage`, descending similarity; source excluded.

        Raises:
            NotFoundError: no such dataset, or no such image in it (or it belongs to another org).
        """
        params: dict[str, Any] = {
            "image_id": await _resolve.image_id(self._transport, dataset_name, image),
            "threshold": threshold,
            "limit": limit,
        }
        if directory_path is not None:
            params["directory_path"] = directory_path
        response = await self._transport.request("GET", f"{_API_PATH}/similar", params=params)
        return self._parse_list(SimilarImage, response.get("results", []))

    async def by_tag(
        self,
        *,
        objects: Sequence[str] | None = None,
        scenes: Sequence[str] | None = None,
        attributes: Sequence[str] | None = None,
        dataset_name: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TaggedImage]:
        """Search images by their SigLIP auto-tags.

        Each category filter ANDs its tags; cross-category filters AND each
        other. At least one tag in any category is required.

        Args:
            objects: Tags to require under ``image_auto_tags.objects``.
            scenes: Tags under ``image_auto_tags.scenes``.
            attributes: Tags under ``image_auto_tags.attributes``.
            dataset_name: Restrict to one dataset. ``None`` = whole org.
            limit: Max results (backend cap: 500).
            offset: Pagination offset.

        Raises:
            ValidationError: No tags supplied in any category.
            NotFoundError: ``dataset_name`` doesn't exist.
        """
        if not (objects or scenes or attributes):
            raise ValueError("At least one of objects, scenes, or attributes must be provided.")
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if objects:
            params["objects"] = list(objects)
        if scenes:
            params["scenes"] = list(scenes)
        if attributes:
            params["attributes"] = list(attributes)
        if dataset_name is not None:
            params["dataset_name"] = dataset_name
        response = await self._transport.request("GET", f"{_API_PATH}/by-tags", params=params)
        return self._parse_list(TaggedImage, response.get("results", []))
