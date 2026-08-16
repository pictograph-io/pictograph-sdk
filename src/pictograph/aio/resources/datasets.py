"""Async Datasets resource - the full dataset lifecycle (async twin).

Async twin of :class:`pictograph.resources.datasets.Datasets`: same
name-or-``dataset_id=`` addressing, same merged CRUD + archive/unarchive +
insights/near-duplicates/download/cold-storage surface. The parallel
:meth:`AsyncDatasets.download` uses ``asyncio.gather`` + a concurrency
semaphore instead of a thread pool. ``as_pytorch`` is intentionally absent -
a map-style ``torch.utils.data.Dataset`` is consumed synchronously by
``DataLoader`` workers, so use the sync :meth:`Client.datasets.as_pytorch`
for that.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import httpx

from pictograph._http.pagination import AsyncOffsetPager
from pictograph._http.streaming import DEFAULT_CHUNK_SIZE
from pictograph._path_safety import safe_download_name
from pictograph.exceptions import ApiError, PollTimeoutError
from pictograph.models.dataset import (
    Dataset,
    DatasetClass,
    DatasetStorageStatus,
    DatasetStorageTransition,
)
from pictograph.models.insights import DatasetInsights
from pictograph.models.near_duplicates import NearDuplicatesResult
from pictograph.resources._base import AsyncResource
from pictograph.resources.datasets import (
    _API_PATH,
    _DEFAULT_DOWNLOAD_LIMIT,
    _DEFAULT_DOWNLOAD_WORKERS,
    DownloadFailure,
    DownloadMode,
    DownloadReport,
    _class_payload,
    _single_path,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from pictograph._http.async_transport import AsyncTransport


class AsyncDatasets(AsyncResource):
    """The full dataset lifecycle in the authenticated organization (async)."""

    # ───────────── list / iter ─────────────

    async def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        archived: bool = False,
    ) -> list[Dataset]:
        """Single-page list of datasets (prefer :meth:`iter` for full enumeration)."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if archived:
            params["archived"] = True
        response = await self._transport.request("GET", _API_PATH, params=params)
        return self._parse_list(Dataset, response.get("data", []))

    def iter(
        self,
        *,
        page_size: int = 100,
        max_total: int | None = None,
        archived: bool = False,
    ) -> AsyncOffsetPager[Dataset]:
        """Auto-paging async iterator over every dataset in the organization."""

        async def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            params: dict[str, Any] = {"offset": offset, "limit": limit}
            if archived:
                params["archived"] = True
            return cast(
                "Mapping[str, Any]",
                await self._transport.request("GET", _API_PATH, params=params),
            )

        return AsyncOffsetPager(
            fetch,
            items_key="data",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(Dataset, raw),
        )

    # ───────────── get (by name / by id) ─────────────

    async def get(
        self,
        name: str | None = None,
        *,
        dataset_id: str | None = None,
        include_images: bool = False,
        images_limit: int = 1000,
        images_offset: int = 0,
    ) -> Dataset:
        """Fetch a dataset by name (or ``dataset_id=`` UUID).

        Args:
            name: Dataset name. Case-sensitive, unique within the org.
            dataset_id: Dataset UUID - the keyword alternative to ``name``.
            include_images: Include the first ``images_limit`` images.
            images_limit: How many images to include (backend cap: 10000).
            images_offset: Paginate the embedded image list.
        """
        params: dict[str, Any] = {}
        if include_images:
            params["include_images"] = "true"
            params["images_limit"] = images_limit
            params["images_offset"] = images_offset
        response = await self._transport.request(
            "GET", _single_path(name, dataset_id), params=params or None
        )
        return self._parse(Dataset, response["data"])

    # ───────────── create / update / delete ─────────────

    async def create(
        self,
        name: str,
        *,
        readme: str | None = None,
        description: str | None = None,
        annotation_types: Sequence[str] | None = None,
        classes: Sequence[DatasetClass | dict[str, Any]] | None = None,
    ) -> Dataset:
        """Create a new dataset + initial class config. Member+ API key.

        Raises:
            ConflictError: A dataset with this name already exists (409).
            PaymentRequiredError: The org hit its tier's dataset cap (402).
            ForbiddenError: API key lacks member/admin/owner role.
        """
        body: dict[str, Any] = {"name": name}
        if readme is not None:
            body["readme"] = readme
        if description is not None:
            body["description"] = description
        if annotation_types is not None:
            body["annotation_types"] = list(annotation_types)
        if classes is not None:
            body["classes"] = _class_payload(classes)
        response = await self._transport.request("POST", _API_PATH, json=body)
        return self._parse(Dataset, response["data"])

    async def update(
        self,
        name: str | None = None,
        *,
        dataset_id: str | None = None,
        new_name: str | None = None,
        readme: str | None = None,
        description: str | None = None,
        annotation_types: Sequence[str] | None = None,
        classes: Sequence[DatasetClass | dict[str, Any]] | None = None,
    ) -> Dataset:
        """Update a dataset's metadata, types, or classes (partial). Member+.

        ``classes`` REPLACES the full class list (fetch → mutate → pass back).

        Raises:
            ValueError: No update field was provided.
            NotFoundError: The dataset doesn't exist in your org.
            ConflictError: ``new_name`` collides with an existing dataset.
        """
        body: dict[str, Any] = {}
        if new_name is not None:
            body["new_name"] = new_name
        if readme is not None:
            body["readme"] = readme
        if description is not None:
            body["description"] = description
        if annotation_types is not None:
            body["annotation_types"] = list(annotation_types)
        if classes is not None:
            body["classes"] = _class_payload(classes)
        if not body:
            raise ValueError(
                "Nothing to update - pass at least one of new_name / description / "
                "annotation_types / classes."
            )
        response = await self._transport.request("PATCH", _single_path(name, dataset_id), json=body)
        return self._parse(Dataset, response["data"])

    async def delete(
        self, name: str | None = None, *, dataset_id: str | None = None
    ) -> dict[str, Any]:
        """Permanently delete a dataset + its images + storage. Admin+ API key.

        Returns the deletion summary: ``{id, name, deleted, images_deleted,
        directories_deleted, gcs_blobs_deleted, gcs_blobs_retained_for_forks}``.
        """
        response = await self._transport.request("DELETE", _single_path(name, dataset_id))
        return cast("dict[str, Any]", response["data"])

    # ───────────── archive / unarchive ─────────────

    async def archive(self, name: str | None = None, *, dataset_id: str | None = None) -> Dataset:
        """Archive a dataset (hide from the default list; reversible). Admin+;
        idempotent. A PUBLIC dataset must be unpublished first (400)."""
        response = await self._transport.request("POST", _single_path(name, dataset_id, "/archive"))
        return self._parse(Dataset, response["data"])

    async def unarchive(self, name: str | None = None, *, dataset_id: str | None = None) -> Dataset:
        """Bring an archived dataset back into the default list. Admin+; idempotent."""
        response = await self._transport.request(
            "POST", _single_path(name, dataset_id, "/unarchive")
        )
        return self._parse(Dataset, response["data"])

    # ───────────── insights / near-duplicates ─────────────

    async def insights(
        self, name: str | None = None, *, dataset_id: str | None = None
    ) -> DatasetInsights:
        """Dataset Health / Insights (by name or ``dataset_id=``).

        Headline totals, labeling-stage counts, per-class instance + image
        counts, per-annotation-type totals, an annotations-per-image density
        histogram, and image-dimension insights - aggregated server-side over
        denormalized columns (a single fast call even for 100k+ image datasets).

        Raises:
            NotFoundError: No dataset with this name/id in your org.
        """
        response = await self._transport.request("GET", _single_path(name, dataset_id, "/insights"))
        return self._parse(DatasetInsights, response["data"])

    async def near_duplicates(
        self,
        name: str | None = None,
        *,
        dataset_id: str | None = None,
        threshold: float | None = None,
        sample: int | None = None,
        neighbors: int | None = None,
        max_pairs: int | None = None,
        directory_path: str | None = None,
    ) -> NearDuplicatesResult:
        """Find near-duplicate images in a dataset - data curation.

        Groups visually near-duplicate images (SigLIP2 embedding cosine
        similarity >= ``threshold``) into clusters so you can keep one per
        cluster and archive the redundant rest. Expensive + on-demand; every
        bound is clamped server-side and the result reports the analyzed sample
        + cap flags (no silent caps). Non-archived images only. ``directory_path``
        scopes the scan to one virtual directory (``None`` = whole dataset).

        Raises:
            NotFoundError: No dataset with this name/id in your org.
        """
        params: dict[str, Any] = {}
        if threshold is not None:
            params["threshold"] = threshold
        if sample is not None:
            params["sample"] = sample
        if neighbors is not None:
            params["neighbors"] = neighbors
        if max_pairs is not None:
            params["max_pairs"] = max_pairs
        if directory_path is not None:
            params["directory_path"] = directory_path
        response = await self._transport.request(
            "GET", _single_path(name, dataset_id, "/duplicates"), params=params or None
        )
        return self._parse(NearDuplicatesResult, response["data"])

    # ───────────── batch download ─────────────

    async def download(
        self,
        name: str | None = None,
        output_dir: str | Path | None = None,
        *,
        dataset_id: str | None = None,
        mode: DownloadMode = "full",
        status_filter: str | None = None,
        max_workers: int = _DEFAULT_DOWNLOAD_WORKERS,
        progress: Callable[[int, int, str | None], None] | None = None,
    ) -> DownloadReport:
        """Download a dataset's images and/or annotations to a local directory.

        Fetches a batch of signed storage URLs in one backend call, then downloads
        concurrently via ``asyncio.gather`` bounded by a ``max_workers`` semaphore.
        Individual file failures are collected in :attr:`DownloadReport.failures`,
        never raised.

        Args:
            name: Dataset name (or pass ``dataset_id=``).
            output_dir: Local directory to write into (created if missing).
            dataset_id: Dataset UUID - the keyword alternative to ``name``.
            mode: ``"full"`` / ``"images_only"`` / ``"annotations_only"``.
            status_filter: Restrict to images with this status.
            max_workers: Max concurrent downloads (default 10).
            progress: Optional ``(completed, total, filename)`` callback.

        Raises:
            ValidationError / NotFoundError: From the initial batch-URL fetch.
        """
        if output_dir is None:
            raise ValueError("output_dir is required")
        out = Path(output_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)

        params: dict[str, Any] = {"mode": mode, "limit": _DEFAULT_DOWNLOAD_LIMIT}
        if status_filter is not None:
            params["status_filter"] = status_filter
        listing = (
            await self._transport.request(
                "GET", _single_path(name, dataset_id, "/download"), params=params
            )
        )["data"]

        items: list[dict[str, Any]] = listing.get("items", [])
        report = DownloadReport(dataset_id=listing.get("id", ""))
        if len(items) >= _DEFAULT_DOWNLOAD_LIMIT:
            report.truncated = True
            warnings.warn(
                f"Dataset '{name or dataset_id}' has at least "
                f"{_DEFAULT_DOWNLOAD_LIMIT} images; download covers only the "
                f"first {_DEFAULT_DOWNLOAD_LIMIT}. Check report.truncated; the "
                "remaining images were not fetched.",
                stacklevel=2,
            )
        if not items:
            return report

        @dataclass(frozen=True)
        class _Task:
            kind: Literal["image", "annotation"]
            url: str
            dest: Path
            filename: str

        tasks: list[_Task] = []
        for item in items:
            # Server DATA, not a path component - see the sync twin in
            # pictograph/resources/datasets.py for why this must be reduced.
            filename = safe_download_name(item["filename"], fallback="image")
            if mode in ("full", "images_only") and item.get("image_url"):
                tasks.append(
                    _Task(
                        kind="image", url=item["image_url"], dest=out / filename, filename=filename
                    )
                )
            if mode in ("full", "annotations_only") and item.get("annotation_url"):
                tasks.append(
                    _Task(
                        kind="annotation",
                        url=item["annotation_url"],
                        dest=out / f"{filename}.json",
                        filename=filename,
                    )
                )

        total = len(tasks)
        if total == 0:
            return report

        semaphore = asyncio.Semaphore(max_workers)
        completed = 0

        # One shared AsyncClient for the signed-URL image streams (no auth). http2 is
        # safe here (single event loop, no thread contention on the h2 stream
        # dict). Annotation JSON routes through the SDK transport (adds X-API-Key).
        async with httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(self._transport._config.timeout, read=300.0),
            limits=httpx.Limits(
                max_connections=max_workers,
                max_keepalive_connections=max_workers,
            ),
        ) as gcs_client:

            async def run_task(task: _Task) -> tuple[_Task, str | None]:
                async with semaphore:
                    try:
                        if task.kind == "image":
                            await _astream_to_file(gcs_client, task.url, task.dest)
                        else:
                            await _afetch_annotation_to_file(self._transport, task.url, task.dest)
                    except (httpx.HTTPError, ApiError, OSError) as exc:
                        return task, str(exc)
                    return task, None

            for coro in asyncio.as_completed([run_task(t) for t in tasks]):
                task, err = await coro
                completed += 1
                if err is None:
                    if task.kind == "image":
                        report.images_downloaded += 1
                    else:
                        report.annotations_downloaded += 1
                else:
                    report.failures.append(
                        DownloadFailure(filename=task.filename, kind=task.kind, reason=err)
                    )
                if progress is not None:
                    progress(completed, total, task.filename)

        return report

    # ───────────── cold storage ─────────────

    async def storage_status(
        self, name: str | None = None, *, dataset_id: str | None = None
    ) -> DatasetStorageStatus:
        """Cold-storage state (+ restore price quote while cold) for a dataset."""
        response = await self._transport.request("GET", _single_path(name, dataset_id, "/storage"))
        return self._parse(DatasetStorageStatus, response["data"])

    async def freeze(
        self, name: str | None = None, *, dataset_id: str | None = None
    ) -> DatasetStorageTransition:
        """Move a dataset to cold storage. Free; background job.

        Requires admin/owner; public datasets and datasets with forks are 409.
        """
        response = await self._transport.request(
            "POST", _single_path(name, dataset_id, "/storage/freeze")
        )
        return self._parse(DatasetStorageTransition, response["data"])

    async def restore(
        self, name: str | None = None, *, dataset_id: str | None = None
    ) -> DatasetStorageTransition:
        """Restore a cold dataset to standard storage (charges compute credits).

        The restore fee is billed on the transition's SUCCESS - a restore that
        fails or is abandoned costs nothing; the returned ``quoted_micro_usd``
        is that pending amount. The charge is idempotent per frozen generation,
        so a retried restore never double-charges.
        """
        response = await self._transport.request(
            "POST", _single_path(name, dataset_id, "/storage/restore")
        )
        return self._parse(DatasetStorageTransition, response["data"])

    async def wait_for_storage(
        self,
        name: str | None = None,
        *,
        dataset_id: str | None = None,
        timeout: float = 600.0,
        poll_interval: float = 3.0,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> DatasetStorageStatus:
        """Block until a freeze/restore transition finishes (state ``idle``).

        Raises:
            PollTimeoutError: ``timeout`` elapsed before the transition finished.
        """
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else asyncio.sleep
        deadline = time.monotonic() + timeout
        while True:
            status = await self.storage_status(name, dataset_id=dataset_id)
            if status.storage_state == "idle":
                return status
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Dataset {name or dataset_id} storage transition did not "
                    f"finish within {timeout:.0f}s (state={status.storage_state!r})"
                )
            await sleep_fn(poll_interval)


async def _astream_to_file(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Stream a signed storage URL to disk in chunks (atomic ``.part`` rename)."""
    tmp = dest.with_name(dest.name + ".part")
    try:
        async with client.stream("GET", url) as response:
            if response.status_code >= 300:
                await response.aread()
                raise httpx.HTTPStatusError(
                    f"Image download failed: HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
            fh = await asyncio.to_thread(tmp.open, "wb")
            try:
                async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                    await asyncio.to_thread(fh.write, chunk)
            finally:
                await asyncio.to_thread(fh.close)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    await asyncio.to_thread(tmp.replace, dest)


async def _afetch_annotation_to_file(
    transport: AsyncTransport,
    url: str,
    dest: Path,
) -> None:
    """Fetch an annotation JSON file via the authenticated SDK transport (atomic write)."""
    payload = await transport.request("GET", url)
    tmp = dest.with_name(dest.name + ".part")

    def _write() -> None:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    try:
        await asyncio.to_thread(_write)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    await asyncio.to_thread(tmp.replace, dest)
