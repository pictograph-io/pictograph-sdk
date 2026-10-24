"""Async Explore resource - the public catalog (datasets and models published to Explore).

Async twin of :class:`pictograph.resources.explore.Explore`; see that module for the
full description and how a public dataset or model is addressed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pictograph._http.pagination import AsyncOffsetPager
from pictograph.models.explore import (
    DatasetSort,
    ModelSort,
    PublicDataset,
    PublicModel,
    PublicModelType,
)
from pictograph.resources._base import AsyncResource
from pictograph.resources.explore import (
    _API_PATH,
    dataset_params,
    dataset_path,
    model_params,
    split_ref,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


class AsyncExplore(AsyncResource):
    """The public catalog: datasets and models any organization published (async)."""

    async def datasets(
        self,
        *,
        search: str | None = None,
        class_name: str | None = None,
        sort: DatasetSort = "recent",
        limit: int = 50,
        offset: int = 0,
    ) -> list[PublicDataset]:
        """One page of public datasets (see the sync twin)."""
        response = await self._transport.request(
            "GET",
            f"{_API_PATH}/datasets",
            params=dataset_params(search, class_name, sort, offset, limit),
        )
        return self._parse_list(PublicDataset, response.get("data", []))

    def iter_datasets(
        self,
        *,
        search: str | None = None,
        class_name: str | None = None,
        sort: DatasetSort = "recent",
        page_size: int = 50,
        max_total: int | None = None,
    ) -> AsyncOffsetPager[PublicDataset]:
        """Every public dataset that matches, page by page (``async for``)."""

        async def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            return cast(
                "Mapping[str, Any]",
                await self._transport.request(
                    "GET",
                    f"{_API_PATH}/datasets",
                    params=dataset_params(search, class_name, sort, offset, limit),
                ),
            )

        return AsyncOffsetPager(
            fetch,
            items_key="data",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(PublicDataset, raw),
        )

    async def dataset(self, ref: str, *, organization: str | None = None) -> PublicDataset:
        """One public dataset by its qualified public name (see the sync twin)."""
        org, name = split_ref(ref, organization)
        response = await self._transport.request("GET", dataset_path(org, name))
        return self._parse(PublicDataset, response["data"])

    async def models(
        self,
        *,
        search: str | None = None,
        model_type: PublicModelType | None = None,
        sort: ModelSort = "recent",
        limit: int = 50,
        offset: int = 0,
    ) -> list[PublicModel]:
        """One page of public, ready models (see the sync twin)."""
        response = await self._transport.request(
            "GET",
            f"{_API_PATH}/models",
            params=model_params(search, model_type, sort, offset, limit),
        )
        return self._parse_list(PublicModel, response.get("data", []))

    def iter_models(
        self,
        *,
        search: str | None = None,
        model_type: PublicModelType | None = None,
        sort: ModelSort = "recent",
        page_size: int = 50,
        max_total: int | None = None,
    ) -> AsyncOffsetPager[PublicModel]:
        """Every public model that matches, page by page (``async for``)."""

        async def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            return cast(
                "Mapping[str, Any]",
                await self._transport.request(
                    "GET",
                    f"{_API_PATH}/models",
                    params=model_params(search, model_type, sort, offset, limit),
                ),
            )

        return AsyncOffsetPager(
            fetch,
            items_key="data",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(PublicModel, raw),
        )
