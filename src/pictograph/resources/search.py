"""Search resource - SigLIP-powered visual similarity + auto-tag filtering.

Two query modes correspond 1:1 to the backend's
``/api/v1/developer/search/{similar,by-tags}`` endpoints:

- :meth:`Search.by_similarity` - pgvector cosine similarity against a
  source image's SigLIP-1152 embedding. Always scoped to the source
  image's dataset + virtual directory (matches the editor's "find similar"
  feature).
- :meth:`Search.by_tag` - JSONB containment search against
  an image's ``image_auto_tags``. Filter by any subset of the three
  SigLIP categories (``objects`` / ``scenes`` / ``attributes``), with
  optional dataset restriction.

The backend strips ``gcs_image_path`` before responses reach the SDK -
fetch actual bytes via :class:`pictograph.resources.images.Images`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pictograph.models.search import SimilarImage, TaggedImage
from pictograph.resources import _resolve
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Sequence

_API_PATH = "/api/v1/developer/search"


class Search(Resource):
    """Visual + tag-based image search across the org."""

    def by_similarity(
        self,
        dataset_name: str,
        image: str,
        *,
        threshold: float = 0.6,
        limit: int = 50,
        directory_path: str | None = None,
    ) -> list[SimilarImage]:
        """Find images visually similar to one of a dataset's images.

        Uses SigLIP-1152 embeddings + pgvector HNSW index. Scope is always
        ``(image's dataset, image's directory)`` unless ``directory_path`` is
        explicitly overridden.

        Args:
            dataset_name: The dataset's name (a UUID also works).
            image: The source image's FILENAME (an id also works).
            threshold: Minimum cosine similarity in ``[0, 1]``. Defaults
                to 0.6 - a sane "visually related" cutoff.
            limit: Max results (backend cap: 500).
            directory_path: Override the source image's directory. Pass ``"/"``
                to search the dataset root.

        Returns:
            List of :class:`SimilarImage`, sorted by descending similarity.
            The source image is excluded.

        Raises:
            NotFoundError: no such dataset, or no such image in it (or it belongs to
                another org).
        """
        params: dict[str, Any] = {
            "image_id": _resolve.image_id(self._transport, dataset_name, image),
            "threshold": threshold,
            "limit": limit,
        }
        if directory_path is not None:
            params["directory_path"] = directory_path
        response = self._transport.request("GET", f"{_API_PATH}/similar", params=params)
        return self._parse_list(SimilarImage, response.get("results", []))

    def by_tag(
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

        Each category filter ANDs its tags. Cross-category filters AND
        each other. At least one tag in any category is required.

        Examples:

            # All cars in the org.
            client.search.by_tag(objects=["car"])

            # Outdoor cars in the road-signs dataset.
            client.search.by_tag(
                objects=["car"], scenes=["outdoor"], dataset_name="road-signs"
            )

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
        response = self._transport.request("GET", f"{_API_PATH}/by-tags", params=params)
        return self._parse_list(TaggedImage, response.get("results", []))
