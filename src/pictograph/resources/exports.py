"""Exports resource - create, list, get, download, delete dataset exports.

Exports are asynchronous: ``create`` returns immediately with a ``pending``
:class:`Export`, and the backend processes the ZIP in a background task.
By default ``create`` polls until the export reaches a terminal state
(``completed`` or ``failed``); pass ``wait=False`` to fire-and-forget and
poll later via :meth:`Exports.wait_for_completion`.

Polling raises :class:`PollTimeoutError` (a server-side job is still
running, distinct from an HTTP timeout) or :class:`ApiError` (the export
itself failed). The export's ``error_message`` is included in the failure
message.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote

import httpx

from pictograph._http.pagination import OffsetPager
from pictograph._http.streaming import DEFAULT_CHUNK_SIZE
from pictograph.exceptions import ApiError, PollTimeoutError
from pictograph.models.common import BulkDeleteResult
from pictograph.models.export import Export, ExportFormat
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

_API_PATH = "/api/v1/developer/exports/"
_DEFAULT_POLL_INTERVAL = 2.0
_DEFAULT_TIMEOUT = 300.0


def _export_path(
    dataset_name: str, export_name: str, suffix: str = ""
) -> tuple[str, dict[str, str]]:
    """``(path, params)`` for one export: ``/exports/{export}?dataset={dataset}``.

    Export names are unique within a dataset, not globally, so the dataset has
    to travel too - it now rides as a QUERY PARAM rather than a second path
    segment. The old compound `/exports/by-name/{dataset}/{export}` form was
    removed from the API and 404s.

    Returns a tuple, deliberately: making the shape change visible at every call
    site is what stops one of the six from quietly keeping the old form.
    """
    return f"{_API_PATH}{quote(export_name, safe='')}{suffix}", {"dataset": dataset_name}


class Exports(Resource):
    """Operations on dataset exports."""

    # ───────────── create ─────────────

    def create(
        self,
        dataset_name: str,
        name: str,
        *,
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
            dataset_name: Name of the dataset to export.
            name: Human-readable name for this export (≤100 chars).
            format: Output format. Defaults to ``"pictograph"`` (canonical JSON).
            include_images: Bundle the original image bytes alongside annotations.
                Roughly doubles the ZIP size; useful for full reproducibility.
            class_filter: Restrict to these class names. ``None`` exports all.
            status_filter: Restrict to images with this status (e.g., ``"complete"``).
            organize_by_split: When ``True``, organize the ZIP into top-level
                ``train/`` / ``valid/`` / ``test/`` directories by each image's assigned
                split (see :meth:`Images.set_split`); images with no split go to
                ``train``. Yields a directly-trainable YOLO/COCO layout. Default
                ``False`` keeps the flat layout.
            wait: When ``True`` (default), poll until the export completes.
                When ``False``, return immediately with status ``"pending"``.
            poll_interval: Seconds between status checks when ``wait=True``.
            timeout: Maximum seconds to wait for completion.

        Returns:
            The :class:`Export` - completed (with ``download_url`` populated)
            when ``wait=True``, pending when ``wait=False``.

        Raises:
            PollTimeoutError: ``timeout`` elapsed and the export was still
                processing. The export continues running on the server; call
                :meth:`get` or :meth:`wait_for_completion` later.
            ApiError: The export reached ``failed`` status. The error
                message includes ``error_message`` from the backend.
        """
        body: dict[str, Any] = {
            "dataset_name": dataset_name,
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

        response = self._transport.request("POST", _API_PATH, json=body)
        export = self._parse(Export, response["data"])
        if not wait:
            return export
        return self.wait_for_completion(
            dataset_name,
            name,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    # ───────────── list / iter ─────────────

    def list(
        self,
        *,
        dataset_name: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Export]:
        """Single-page list of exports for the organization."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if dataset_name is not None:
            params["dataset_name"] = dataset_name
        if status is not None:
            params["status"] = status
        response = self._transport.request("GET", _API_PATH, params=params)
        return self._parse_list(Export, response.get("data", []))

    def iter(
        self,
        *,
        dataset_name: str | None = None,
        status: str | None = None,
        page_size: int = 100,
        max_total: int | None = None,
    ) -> OffsetPager[Export]:
        """Auto-paging iterator across every export in the organization."""
        base_params: dict[str, Any] = {}
        if dataset_name is not None:
            base_params["dataset_name"] = dataset_name
        if status is not None:
            base_params["status"] = status

        def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            params = {**base_params, "offset": offset, "limit": limit}
            return cast(
                "Mapping[str, Any]",
                self._transport.request("GET", _API_PATH, params=params),
            )

        return OffsetPager(
            fetch,
            items_key="data",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(Export, raw),
        )

    # ───────────── get / wait ─────────────

    def get(self, dataset_name: str, export_name: str) -> Export:
        """Fetch a single export by ``(dataset_name, export_name)``."""
        path, params = _export_path(dataset_name, export_name)
        response = self._transport.request("GET", path, params=params)
        return self._parse(Export, response["data"])

    def wait_for_completion(
        self,
        dataset_name: str,
        export_name: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        sleep: Callable[[float], None] | None = None,
    ) -> Export:
        """Poll until the export reaches a terminal status.

        Args:
            dataset_name: Dataset the export belongs to.
            export_name: Export name.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait.
            sleep: Override the sleep function (testing hook).

        Returns:
            The :class:`Export` in ``completed`` status.

        Raises:
            ApiError: The export reached ``failed`` status.
            PollTimeoutError: ``timeout`` elapsed before completion.
        """
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else time.sleep
        deadline = time.monotonic() + timeout
        while True:
            export = self.get(dataset_name, export_name)
            if export.status == "completed":
                return export
            if export.status == "failed":
                raise ApiError(
                    f"Export '{export_name}' failed: "
                    f"{export.error_message or 'no error message provided'}",
                    response=export.model_dump(mode="json"),
                )
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Export '{export_name}' did not complete within {timeout:.1f}s "
                    f"(last status: {export.status}). The server-side job is still "
                    f"running - fetch its status later via client.exports.get(...)."
                )
            sleep_fn(poll_interval)

    # ───────────── download / delete ─────────────

    def download(
        self,
        dataset_name: str,
        export_name: str,
        output_path: str | Path,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Download an export ZIP to disk.

        Fetches a signed URL from the backend, then streams the bytes over a
        SECOND request to a different host - the storage service - through a
        separate, scoped ``httpx.Client`` that sends NO SDK credentials (the
        authorization is the signature inside the URL). Bytes land in a
        sibling ``.part`` file and are renamed atomically on success.

        Args:
            dataset_name: Dataset the export belongs to.
            export_name: Export name.
            output_path: Local destination. Parent dirs created if missing.
            chunk_size: Streaming chunk size; 8 MB by default, tuned for the
                storage transfer.
            progress: Optional ``(bytes_so_far, total_bytes)`` callback.
                ``total_bytes`` is ``0`` if the server did not provide
                ``Content-Length``.

        Returns:
            The output path (same as ``output_path``).
        """
        path, params = _export_path(dataset_name, export_name, "/download")
        url_response = self._transport.request("GET", path, params=params)
        return self._stream_to_file(
            url_response["data"]["download_url"],
            output_path,
            chunk_size=chunk_size,
            progress=progress,
        )

    def get_by_id(self, export_id: str) -> Export:
        """Fetch a single export by its UUID (e.g. from :meth:`list` / ``Export.id``).

        The by-id complement to :meth:`get` - convenient when you already hold an
        export id and don't want to thread ``(dataset_name, export_name)``. A
        cross-org id resolves to nothing → :class:`NotFoundError` (existence is
        never leaked across tenants).
        """
        response = self._transport.request("GET", f"{_API_PATH}{export_id}")
        return self._parse(Export, response["data"])

    def download_by_id(
        self,
        export_id: str,
        output_path: str | Path,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Download an export ZIP to disk by its UUID. The by-id complement to
        :meth:`download` - same streaming/atomic-rename behavior."""
        url_response = self._transport.request("GET", f"{_API_PATH}{export_id}/download")
        return self._stream_to_file(
            url_response["data"]["download_url"],
            output_path,
            chunk_size=chunk_size,
            progress=progress,
        )

    def _stream_to_file(
        self,
        download_url: str,
        output_path: str | Path,
        *,
        chunk_size: int,
        progress: Callable[[int, int], None] | None,
    ) -> Path:
        """Stream a signed storage URL to disk via a separate, scoped
        ``httpx.Client`` that sends no SDK credentials; bytes land in a sibling
        ``.part`` file renamed atomically on success. Shared by
        :meth:`download` / :meth:`download_by_id`."""
        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")

        try:
            with (
                httpx.Client(
                    http2=True,
                    timeout=httpx.Timeout(self._transport._config.timeout, read=600.0),
                ) as gcs,
                gcs.stream("GET", download_url) as response,
            ):
                if response.status_code >= 300:
                    response.read()
                    raise ApiError(
                        f"Export download failed: HTTP {response.status_code}",
                        status_code=response.status_code,
                        response=response.text,
                    )
                total = int(response.headers.get("Content-Length", 0))
                sent = 0
                with tmp.open("wb") as fh:
                    for chunk in response.iter_bytes(chunk_size=chunk_size):
                        fh.write(chunk)
                        sent += len(chunk)
                        if progress is not None:
                            progress(sent, total)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise

        tmp.replace(out)
        return out

    def delete(self, dataset_name: str, export_name: str) -> None:
        """Delete an export (database row + stored file). Requires admin/owner."""
        path, params = _export_path(dataset_name, export_name)
        self._transport.request("DELETE", path, params=params)

    def bulk_delete(self, export_ids: Sequence[str]) -> BulkDeleteResult:
        """Delete many exports by id in one atomic, org-scoped server-side call.

        Unlike calling :meth:`delete` per export, this issues a single request
        the backend resolves with chunked, organization-scoped deletes (rows +
        their stored ZIPs), so it never fans out N calls. Requires the ``admin`` or
        ``owner`` role.

        Args:
            export_ids: UUIDs of the exports to delete (from
                :meth:`list` / :attr:`Export.id`). Duplicates are ignored; ids
                that don't resolve in your organization are reported in
                :attr:`~pictograph.models.common.BulkDeleteResult.not_found`
                rather than raising, so a re-run still succeeds.

        Returns:
            A :class:`~pictograph.models.common.BulkDeleteResult`.

        Raises:
            ForbiddenError: Your API key role cannot delete exports.
            ValidationError: ``export_ids`` is empty.
        """
        response = self._transport.request(
            "POST", f"{_API_PATH}bulk-delete", json={"export_ids": list(export_ids)}
        )
        return self._parse(BulkDeleteResult, response.get("data", response))
