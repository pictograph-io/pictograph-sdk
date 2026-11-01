"""Async Exports resource - create, list, get, download, delete dataset exports.

Async twin of :class:`pictograph.resources.exports.Exports`. Exports are
asynchronous server-side; ``create`` polls until terminal by default.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast

from pictograph._http.pagination import AsyncOffsetPager
from pictograph._http.streaming import DEFAULT_CHUNK_SIZE
from pictograph.aio._download import stream_url_to_file
from pictograph.exceptions import ApiError, PollTimeoutError, job_end_code
from pictograph.models.common import BulkDeleteResult
from pictograph.models.export import Export, ExportFormat
from pictograph.resources._base import AsyncResource
from pictograph.resources.exports import _export_path

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from pathlib import Path

_API_PATH = "/api/v1/developer/exports/"
_DEFAULT_POLL_INTERVAL = 2.0
_DEFAULT_TIMEOUT = 300.0


class AsyncExports(AsyncResource):
    """Operations on dataset exports (async)."""

    # ───────────── create ─────────────

    async def create(
        self,
        dataset: str,
        *,
        name: str,
        format: ExportFormat = "pictograph",
        include_images: bool = False,
        class_filter: list[str] | None = None,
        status_filter: str | None = None,
        organize_by_split: bool = False,
        wait: bool = True,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> Export:
        """Create a new export and (by default) wait for it to finish.

        Args:
            dataset: Name of the dataset to export.
            name: Human-readable name (≤100 chars).
            format: Output format (default ``"pictograph"``).
            include_images: Bundle the original image bytes alongside annotations.
            class_filter: Restrict to these class names. ``None`` exports all.
            status_filter: Restrict to images with this status.
            organize_by_split: When ``True``, organize the ZIP into top-level
                ``train/`` / ``valid/`` / ``test/`` directories by each image's assigned
                split; images with no split go to ``train``. Yields a directly-
                trainable YOLO/COCO layout. Default ``False`` keeps the flat layout.
            wait: Poll until the export completes (default ``True``).
            poll_interval: Seconds between checks when ``wait=True``.
            timeout: Max seconds to wait for completion.

        Raises:
            PollTimeoutError: ``timeout`` elapsed while still processing.
            ApiError: The export reached ``failed`` status.
        """
        body: dict[str, Any] = {
            "dataset": dataset,
            "name": name,
            "format": format,
            "include_images": include_images,
        }
        if class_filter is not None:
            body["class_filter"] = class_filter
        if status_filter is not None:
            body["status_filter"] = status_filter
        if organize_by_split:
            body["organize_by_split"] = True

        response = await self._transport.request("POST", _API_PATH, json=body)
        export = self._parse(Export, response["data"])
        if not wait:
            return export
        return await self.wait_for_completion(
            dataset,
            name,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    # ───────────── list / iter ─────────────

    async def list(
        self,
        *,
        dataset: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Export]:
        """Single-page list of exports for the organization."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if dataset is not None:
            params["dataset"] = dataset
        if status is not None:
            params["status"] = status
        response = await self._transport.request("GET", _API_PATH, params=params)
        return self._parse_list(Export, response.get("data", []))

    def iter(
        self,
        *,
        dataset: str | None = None,
        status: str | None = None,
        page_size: int = 100,
        max_total: int | None = None,
    ) -> AsyncOffsetPager[Export]:
        """Auto-paging async iterator across every export in the organization."""
        base_params: dict[str, Any] = {}
        if dataset is not None:
            base_params["dataset"] = dataset
        if status is not None:
            base_params["status"] = status

        async def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            params = {**base_params, "offset": offset, "limit": limit}
            return cast(
                "Mapping[str, Any]",
                await self._transport.request("GET", _API_PATH, params=params),
            )

        return AsyncOffsetPager(
            fetch,
            items_key="data",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(Export, raw),
        )

    # ───────────── get / wait ─────────────

    async def get(self, dataset: str, export: str) -> Export:
        """Fetch a single export by ``(dataset, export)``."""
        path, params = _export_path(dataset, export)
        response = await self._transport.request("GET", path, params=params)
        return self._parse(Export, response["data"])

    async def wait_for_completion(
        self,
        dataset: str,
        export: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> Export:
        """Poll until the export reaches a terminal status.

        Raises:
            ApiError: The export reached ``failed`` status.
            PollTimeoutError: ``timeout`` elapsed before completion.
        """
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else asyncio.sleep
        deadline = time.monotonic() + timeout
        while True:
            current = await self.get(dataset, export)
            if current.status == "completed":
                return current
            if current.status == "failed":
                raise ApiError(
                    f"Export '{export}' failed: "
                    f"{current.error_message or 'no error message provided'}",
                    response=current.model_dump(mode="json"),
                    code=job_end_code(current.status),
                )
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Export '{export}' did not complete within {timeout:.1f}s "
                    f"(last status: {current.status}). The server-side job is still "
                    f"running - fetch its status later via client.exports.get(...)."
                )
            await sleep_fn(poll_interval)

    # ───────────── download / delete ─────────────

    async def download(
        self,
        dataset: str,
        export: str,
        *,
        output_path: str | Path,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Download an export ZIP to disk (streamed, atomic ``.part`` rename)."""
        path, params = _export_path(dataset, export, "/download")
        url_response = await self._transport.request("GET", path, params=params)
        return await stream_url_to_file(
            url_response["data"]["download_url"],
            output_path,
            timeout=self._transport._config.timeout,
            chunk_size=chunk_size,
            progress=progress,
            error_prefix="Export download",
        )

    async def delete(self, dataset: str, export: str) -> None:
        """Delete an export (database row + stored file). Requires admin/owner."""
        path, params = _export_path(dataset, export)
        await self._transport.request("DELETE", path, params=params)

    async def bulk_delete(self, exports: Sequence[str]) -> BulkDeleteResult:
        """Delete many exports by id in one atomic, org-scoped server-side call.

        A single chunked, org-scoped delete (rows + stored ZIPs). Requires
        ``admin``/``owner``. Ids that don't resolve in your org land in
        :attr:`~pictograph.models.common.BulkDeleteResult.not_found`.

        Raises:
            ForbiddenError: Your API key role cannot delete exports.
            ValidationError: ``exports`` is empty.
        """
        response = await self._transport.request(
            "POST", f"{_API_PATH}bulk-delete", json={"exports": list(exports)}
        )
        return self._parse(BulkDeleteResult, response.get("data", response))
