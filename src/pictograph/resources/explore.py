"""Explore resource - the public catalog: datasets and models published to Explore.

Every organization can publish datasets and models to Explore. This resource reads
that catalog - what the app's Explore pages show - so an agent can find a dataset to
start from or a model to build on::

    for ds in client.explore.datasets(search="coco", sort="popular"):
        print(ds.ref, ds.image_count, ds.license_id)

    ds = client.explore.dataset("pictograph-research/coco-5k")
    job = client.datasets.import_public("pictograph-research/coco-5k")  # a private copy

A public dataset or model lives in ANOTHER organization, where a bare name is not
unique, so it is addressed by its qualified public name - the organization's slug and
the dataset's (or model's) slug or display name: ``"pictograph-research/coco-5k"``.
Every row's ``ref`` is exactly that. :meth:`Datasets.import_public
<pictograph.resources.datasets.Datasets.import_public>` copies a public dataset into
your organization; :meth:`Models.fork <pictograph.resources.models.Models.fork>` does
the same for a model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote

from pictograph._http.pagination import OffsetPager
from pictograph.models.explore import (
    DatasetSort,
    ModelSort,
    PublicDataset,
    PublicModel,
    PublicModelType,
)
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Mapping

_API_PATH = "/api/v1/developer/public"


def split_ref(
    ref: str, organization: str | None = None, *, kind: str = "dataset"
) -> tuple[str, str]:
    """``(organization, name)`` from a qualified public name.

    Takes ``"org/name"`` as ``ref`` alone, or ``ref`` as the bare name with
    ``organization`` given separately. Shared by the sync and async resources and by
    :meth:`Datasets.import_public`, so all read an address the same way.
    """
    if organization is not None:
        org, name = organization.strip(), ref.strip()
    else:
        org, sep, name = ref.strip().partition("/")
        org, name = org.strip(), name.strip()
        if not sep:
            raise ValueError(
                f"Not a public {kind}: {ref!r}. Name it as '<organization>/<{kind}>' - the "
                f"organization's slug and the {kind}'s slug or name, e.g. "
                "'pictograph-research/coco-5k' (the `ref` of a row from client.explore)."
            )
    if not org or not name:
        raise ValueError(f"Not a public {kind}: {ref!r} - the organization or the name is empty.")
    return org, name


def dataset_path(organization: str, name: str) -> str:
    return f"{_API_PATH}/datasets/{quote(organization, safe='')}/{quote(name, safe='')}"


def dataset_params(
    search: str | None, class_name: str | None, sort: DatasetSort, offset: int, limit: int
) -> dict[str, Any]:
    params: dict[str, Any] = {"sort": sort, "offset": offset, "limit": limit}
    if search:
        params["search"] = search
    if class_name:
        params["class"] = class_name
    return params


def model_params(
    search: str | None, model_type: PublicModelType | None, sort: ModelSort, offset: int, limit: int
) -> dict[str, Any]:
    params: dict[str, Any] = {"sort": sort, "offset": offset, "limit": limit}
    if search:
        params["search"] = search
    if model_type:
        params["model_type"] = model_type
    return params


class Explore(Resource):
    """The public catalog: datasets and models any organization published."""

    def datasets(
        self,
        *,
        search: str | None = None,
        class_name: str | None = None,
        sort: DatasetSort = "recent",
        limit: int = 50,
        offset: int = 0,
    ) -> list[PublicDataset]:
        """One page of public datasets.

        Args:
            search: Match a dataset's name or what its images show (their tags).
            class_name: Only datasets that have this class.
            sort: ``"recent"`` (default), ``"popular"`` (most starred),
                ``"trending"`` (most starred this week) or ``"alphabetical"``.
            limit: Up to 100 per page; :meth:`iter_datasets` pages for you.
            offset: Rows to skip.

        Returns:
            :class:`~pictograph.models.explore.PublicDataset` rows - each one's
            ``ref`` is the name :meth:`Datasets.import_public` takes.
        """
        response = self._transport.request(
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
    ) -> OffsetPager[PublicDataset]:
        """Every public dataset that matches, page by page.

        Args:
            search: Match a dataset's name or what its images show (their tags).
            class_name: Only datasets that have this class.
            sort: ``"recent"`` (default), ``"popular"``, ``"trending"`` or
                ``"alphabetical"``.
            page_size: Rows fetched per request (up to 100).
            max_total: Stop after this many datasets. ``None`` walks them all.

        Returns:
            An iterator of :class:`~pictograph.models.explore.PublicDataset` rows.
        """

        def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            return cast(
                "Mapping[str, Any]",
                self._transport.request(
                    "GET",
                    f"{_API_PATH}/datasets",
                    params=dataset_params(search, class_name, sort, offset, limit),
                ),
            )

        return OffsetPager(
            fetch,
            items_key="data",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(PublicDataset, raw),
        )

    def dataset(self, ref: str, *, organization: str | None = None) -> PublicDataset:
        """One public dataset: its card (``readme``), license, every class and counts.

        Args:
            ref: ``"<organization>/<dataset>"`` - the organization's slug and the
                dataset's slug or display name. Or pass the bare name and
                ``organization=``.
            organization: The owning organization's slug, when ``ref`` is a bare name.

        Raises:
            ValueError: ``ref`` names no organization.
            NotFoundError: No such public dataset (a private one reads the same).
        """
        org, name = split_ref(ref, organization)
        response = self._transport.request("GET", dataset_path(org, name))
        return self._parse(PublicDataset, response["data"])

    def models(
        self,
        *,
        search: str | None = None,
        model_type: PublicModelType | None = None,
        sort: ModelSort = "recent",
        limit: int = 50,
        offset: int = 0,
    ) -> list[PublicModel]:
        """One page of public, ready models.

        Args:
            search: Match a model's name.
            model_type: Only models of this task (``"object_detection"``,
                ``"instance_segmentation"``, ...).
            sort: ``"recent"`` (default), ``"popular"`` or ``"alphabetical"``.
            limit: Up to 100 per page; :meth:`iter_models` pages for you.
            offset: Rows to skip.

        Returns:
            :class:`~pictograph.models.explore.PublicModel` rows - each one's ``ref``
            is the pair :meth:`Models.fork` takes (``models.fork(*model.ref.split("/"))``).
        """
        response = self._transport.request(
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
    ) -> OffsetPager[PublicModel]:
        """Every public, ready model that matches, page by page.

        Args:
            search: Match a model's name.
            model_type: Only models of this task (``"object_detection"``,
                ``"instance_segmentation"``, ...).
            sort: ``"recent"`` (default), ``"popular"`` or ``"alphabetical"``.
            page_size: Rows fetched per request (up to 100).
            max_total: Stop after this many models. ``None`` walks them all.

        Returns:
            An iterator of :class:`~pictograph.models.explore.PublicModel` rows.
        """

        def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            return cast(
                "Mapping[str, Any]",
                self._transport.request(
                    "GET",
                    f"{_API_PATH}/models",
                    params=model_params(search, model_type, sort, offset, limit),
                ),
            )

        return OffsetPager(
            fetch,
            items_key="data",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(PublicModel, raw),
        )
