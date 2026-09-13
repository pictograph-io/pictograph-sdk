"""Async Images resource: metadata, streaming download, chunked upload, delete.

Async twin of :class:`pictograph.resources.images.Images`. The MIME/dimension
helpers, query-param builder, and result dataclasses are reused verbatim from
the sync module (pure, no transport).
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pictograph._http.pagination import AsyncOffsetPager
from pictograph._http.streaming import DEFAULT_CHUNK_SIZE
from pictograph.aio.resources import _resolve
from pictograph.exceptions import (
    ApiError,
    ConflictError,
    NotFoundError,
    PictographError,
)
from pictograph.models.image import Image, ImageSplit, ImageStatus
from pictograph.resources._base import AsyncResource
from pictograph.resources.images import (
    _DEFAULT_FOLDER_WORKERS,
    _IMAGE_EXTS,
    BulkUploadFailure,
    BulkUploadResult,
    UploadFailure,
    UploadReport,
    _image_route,
    _infer_content_type,
    _list_params,
    _safe_image_dimensions,
    virtual_directory_for,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

_API_PATH = "/api/v1/developer/images"

__all__ = [
    "AsyncImages",
    "BulkUploadFailure",
    "BulkUploadResult",
    "UploadFailure",
    "UploadReport",
]


@dataclass(frozen=True)
class _DirectoryTask:
    local_path: Path
    virtual_directory_path: str


class AsyncImages(AsyncResource):
    """Operations on individual images within a dataset (async)."""

    # ───────────── list / iter ─────────────

    async def list(
        self,
        dataset_name: str,
        directory_path: str | None = None,
        filename: str | None = None,
        status: ImageStatus | None = None,
        split: ImageSplit | None = None,
        include_archived: bool = False,
        min_confidence_lt: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Image]:
        """A single page of a dataset's images, newest first.

        Args:
            dataset_name: Dataset (project) NAME whose images to list. An id is
                accepted too.
            directory_path: Restrict to one virtual directory; ``None`` lists all.
            filename: Exact filename match - the indexed lookup that turns a
                filename into an image, without paging the dataset.
            status: Restrict to one annotation stage; ``None`` lists all stages.
            split: Filter to one dataset split (train/val/test); ``None`` all.
            include_archived: Include soft-deleted images (default ``False``).
            min_confidence_lt: Active-learning filter - keep only images whose
                ``min_confidence`` is below this (0..1); ``None`` applies no filter.
            limit: Page size (backend cap: 1000).
            offset: Pagination offset.
        """
        response = await self._transport.request(
            "GET",
            f"{_API_PATH}/",
            params=_list_params(
                dataset_name,
                directory_path,
                status,
                include_archived,
                limit,
                offset,
                min_confidence_lt,
                split,
                filename,
            ),
        )
        return self._parse_list(Image, response.get("data", []))

    def iter(
        self,
        dataset_name: str,
        directory_path: str | None = None,
        filename: str | None = None,
        status: ImageStatus | None = None,
        split: ImageSplit | None = None,
        include_archived: bool = False,
        min_confidence_lt: float | None = None,
        page_size: int = 100,
        max_total: int | None = None,
    ) -> AsyncOffsetPager[Image]:
        """Auto-paging async iterator over every image in a dataset, newest first.

        Filters mirror :meth:`list`. Use as ``async for img in
        client.images.iter(dataset_name, directory_path="/train"): ...``.

        The name goes to the route as a name - it resolves server-side,
        so no page pays a separate dataset lookup.
        """

        async def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            return cast(
                "Mapping[str, Any]",
                await self._transport.request(
                    "GET",
                    f"{_API_PATH}/",
                    params=_list_params(
                        dataset_name,
                        directory_path,
                        status,
                        include_archived,
                        limit,
                        offset,
                        min_confidence_lt,
                        split,
                        filename,
                    ),
                ),
            )

        return AsyncOffsetPager(
            fetch,
            items_key="data",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(Image, raw),
        )

    # ───────────── metadata ─────────────

    async def get(self, dataset_name: str, image: str) -> Image:
        """Fetch metadata for a single image (annotations live on ``client.annotations``)."""
        # A FILENAME is answered by the list route in ONE request - it already
        # returns the whole row, so resolving the id and then re-fetching that
        # same row by id (which is what this used to do) was a wasted
        # round-trip. An id still goes straight to the metadata route.
        if not _resolve.looks_like_id(image):
            matches = await self.list(dataset_name, filename=image, limit=2)
            if not matches:
                raise NotFoundError(f"No image named {image!r} in dataset {dataset_name!r}.")
            if len(matches) > 1:
                directories = sorted({m.directory_path or "/" for m in matches})
                raise ValueError(
                    f"{image!r} exists in more than one directory of {dataset_name!r} "
                    f"({', '.join(directories)}). Fetch it by id, or narrow with list()."
                )
            return matches[0]
        response = await self._transport.request("GET", f"{_API_PATH}/{image}/metadata")
        return self._parse(Image, response["data"])

    # ───────────── download ─────────────

    async def download(
        self,
        dataset_name: str,
        image: str,
        output_path: str | Path,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> Path:
        """Stream image bytes to ``output_path`` (atomic ``.part`` rename); return the path.

        Parent dirs are created if missing. A mid-download failure leaves no
        partial file at the destination, so retries are safe.
        """
        image_id = await _resolve.image_id(self._transport, dataset_name, image)
        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")
        fh = await asyncio.to_thread(tmp.open, "wb")
        try:
            async for chunk in self._transport.stream_bytes("GET", f"{_API_PATH}/{image_id}"):
                await asyncio.to_thread(fh.write, chunk)
                _ = chunk_size  # forward-compat hint; httpx default chunking
        except BaseException:
            await asyncio.to_thread(fh.close)
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        await asyncio.to_thread(fh.close)
        await asyncio.to_thread(tmp.replace, out)
        return out

    async def download_bundle(
        self,
        dataset_name: str,
        image: str,
        output_path: str | Path,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> Path:
        """Stream one image's DATA BUNDLE to ``output_path``; return the path.

        The async twin of :meth:`pictograph.resources.images.Images.download_bundle`.
        A zip carrying the original image bytes, its depth map (when one exists),
        its annotations as Pictograph JSON, and a ``manifest.json`` naming what is
        and is not inside - the same archive the annotation editor's "Image data"
        button produces, because one server-side builder assembles both.

        A missing depth map is not an error; the manifest records the omission.
        Atomic ``.part`` rename, so a failed transfer leaves no partial zip.
        """
        # Path-addressed, like the sync twin: the bundle has no id-addressed
        # route, so this resolves a SEGMENT rather than an image id.
        segment = await _resolve.image_segment(self._transport, image)
        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")
        fh = await asyncio.to_thread(tmp.open, "wb")
        try:
            async for chunk in self._transport.stream_bytes(
                "GET", _image_route(dataset_name, segment, "/data-bundle")
            ):
                await asyncio.to_thread(fh.write, chunk)
                _ = chunk_size  # forward-compat hint; httpx default chunking
        except BaseException:
            await asyncio.to_thread(fh.close)
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        await asyncio.to_thread(fh.close)
        await asyncio.to_thread(tmp.replace, out)
        return out

    # ───────────── upload ─────────────

    async def upload(
        self,
        dataset_name: str,
        file_path: str | Path,
        *,
        directory_path: str = "/",
        filename: str | None = None,
        content_type: str | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> Image:
        """Upload an image to a dataset via the three-step signed-URL flow.

        Args:
            dataset_name: Target dataset NAME (a UUID is also accepted).
            file_path: Local file (read in chunks, never fully buffered).
            directory_path: Virtual directory (created if missing). Default root.
            filename: Override the destination filename (default basename).
            content_type: Override the inferred MIME type.
            progress: ``(bytes_sent, total_bytes)`` callback during the upload PUT.

        Raises:
            FileNotFoundError: ``file_path`` does not exist.
            ValueError: MIME type can't be inferred and wasn't given.
            ApiError / NetworkError: Per-step backend/transport failures.
        """
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        resolved_filename = filename or path.name
        resolved_content_type = content_type or _infer_content_type(resolved_filename)
        if resolved_content_type == "application/octet-stream":
            raise ValueError(
                f"Could not infer image MIME type from filename '{resolved_filename}'. "
                f"Pass content_type=... explicitly (e.g. 'image/jpeg')."
            )

        file_size = path.stat().st_size
        width, height = _safe_image_dimensions(path)

        url_response = await self._transport.request(
            "POST",
            f"{_API_PATH}/upload-url",
            json={
                "dataset": dataset_name,
                "filename": resolved_filename,
                "directory_path": directory_path,
                "content_type": resolved_content_type,
            },
        )

        await self._transport.upload_external(
            url_response["data"]["upload_url"],
            path,
            content_type=resolved_content_type,
            progress=progress,
        )

        # Register (JSON body, the same field names step 1 used; the server
        # re-derives storage paths). Returns the full canonical image.
        register_response = await self._transport.request(
            "POST",
            f"{_API_PATH}/register",
            json={
                "dataset": dataset_name,
                "filename": resolved_filename,
                "directory_path": directory_path,
                "file_size": file_size,
                "content_type": resolved_content_type,
                "width": width,
                "height": height,
            },
        )

        return self._parse(Image, register_response["data"])

    async def bulk_upload(
        self,
        dataset_name: str,
        file_paths: Sequence[str | Path],
        *,
        directory_path: str = "/",
        max_concurrency: int = _DEFAULT_FOLDER_WORKERS,
        progress: Callable[[int, int], None] | None = None,
    ) -> BulkUploadResult:
        """Upload many images to one directory with two server round-trips, not N.

        One ``bulk-upload-url`` call, a per-file PUT to storage (index-matched), then
        one ``bulk-register`` call. Up to 500 files per call. The result is
        returned WITHOUT re-fetching each image.

        Raises:
            FileNotFoundError: A path doesn't exist.
            ValueError: ``file_paths`` is empty, or a MIME type can't be inferred.
            ApiError / NetworkError: A backend or upload-PUT failure aborts the batch.
        """
        paths = [Path(p).expanduser() for p in file_paths]
        if not paths:
            raise ValueError("file_paths must contain at least one file.")

        metas: list[dict[str, Any]] = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            content_type = _infer_content_type(path.name)
            if content_type == "application/octet-stream":
                raise ValueError(
                    f"Could not infer image MIME type from filename '{path.name}'. "
                    f"Rename it with an image extension (e.g. .jpg/.png)."
                )
            width, height = _safe_image_dimensions(path)
            metas.append(
                {
                    "path": path,
                    "filename": path.name,
                    "content_type": content_type,
                    "file_size": path.stat().st_size,
                    "width": width,
                    "height": height,
                }
            )

        url_response = await self._transport.request(
            "POST",
            f"{_API_PATH}/bulk-upload-url",
            json={
                "dataset": dataset_name,
                "images": [
                    {
                        "filename": m["filename"],
                        "directory_path": directory_path,
                        "content_type": m["content_type"],
                    }
                    for m in metas
                ],
            },
        )
        upload_urls = url_response["data"].get("upload_urls", [])

        # Overlapped for the same reason as the sync twin: two API calls plus N
        # SEQUENTIAL storage round-trips measured at 709ms/image, barely 2x
        # single upload. `metas` order is untouched - step 3 registers by
        # index against the URLs minted in step 1.
        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def _put(meta: dict[str, Any], url_entry: dict[str, Any]) -> None:
            async with semaphore:
                await self._transport.upload_external(
                    url_entry["upload_url"],
                    meta["path"],
                    content_type=meta["content_type"],
                    progress=progress,
                )

        # return_exceptions=False: the first failure propagates and aborts the
        # batch before anything is registered, matching the sync twin.
        await asyncio.gather(
            *(_put(meta, url_entry) for meta, url_entry in zip(metas, upload_urls, strict=True))
        )

        register_response = await self._transport.request(
            "POST",
            f"{_API_PATH}/bulk-register",
            json={
                "dataset": dataset_name,
                "images": [
                    {
                        "filename": m["filename"],
                        "directory_path": directory_path,
                        "file_size": m["file_size"],
                        "content_type": m["content_type"],
                        "width": m["width"],
                        "height": m["height"],
                    }
                    for m in metas
                ],
            },
        )
        result = register_response.get("data", register_response)
        succeeded = self._parse_list(Image, result.get("succeeded", []))
        failed = [
            BulkUploadFailure(
                filename=f["filename"],
                error=str(f.get("error", "")),
                directory_path=f.get("directory_path", directory_path),
            )
            for f in result.get("failed", [])
        ]
        return BulkUploadResult(succeeded=succeeded, failed=failed)

    # ───────────── delete ─────────────

    async def delete(self, dataset_name: str, image: str, *, permanent: bool = False) -> None:
        """Archive (default) or permanently delete an image (permanent needs admin/owner)."""
        segment = await _resolve.image_segment(self._transport, image)
        params: dict[str, Any] | None = {"permanent": "true"} if permanent else None
        await self._transport.request(
            "DELETE",
            _image_route(dataset_name, segment),
            params=cast("Any", params),
        )

    # ───────────── bulk tagging ─────────────

    async def bulk_tag(
        self,
        dataset_name: str,
        image_ids: Sequence[str],
        tags: Sequence[str],
        *,
        add: bool = True,
    ) -> int:
        """Add or remove user ``image_tags`` across many images in ONE server call.

        Org-scoped, chunked, idempotent (``add=True`` unions, ``add=False``
        removes). Returns the number of images touched. Member role or higher.
        """
        body = {
            "dataset": dataset_name,
            "image_ids": list(image_ids),
            "tags": list(tags),
            "add": add,
        }
        response = await self._transport.request("POST", f"{_API_PATH}/bulk-tag", json=body)
        data = response.get("data", response)
        return int(data.get("processed", 0))

    async def review(
        self,
        dataset_name: str,
        image: str,
        action: Literal["approve", "request_changes"],
        *,
        note: str | None = None,
    ) -> ImageStatus:
        """Approve or request changes on an image (the annotation review workflow).

        ``approve`` marks the image ``complete`` and clears any pending note;
        ``request_changes`` sends it back to ``annotate`` with ``note`` (the
        message the annotator sees). Enables programmatic QA. Member role or
        higher. Returns the image's new status.
        """
        body: dict[str, Any] = {"action": action}
        if note is not None:
            body["note"] = note
        segment = await _resolve.image_segment(self._transport, image)
        response = await self._transport.request(
            "POST", _image_route(dataset_name, segment, "/review"), json=body
        )
        data = response.get("data", response)
        default: ImageStatus = "complete" if action == "approve" else "annotate"
        return cast("ImageStatus", data.get("status", default))

    async def set_split(
        self, dataset_name: str, image: str, split: ImageSplit | None
    ) -> ImageSplit | None:
        """Assign (or clear, ``split=None``) an image's train/val/test dataset split.

        Enables programmatic dataset organization (e.g. a deterministic 80/10/10
        split). Member role or higher. Returns the split the image now carries.
        """
        segment = await _resolve.image_segment(self._transport, image)
        response = await self._transport.request(
            "POST", _image_route(dataset_name, segment, "/split"), json={"split": split}
        )
        data = response.get("data", response)
        return cast("ImageSplit | None", data.get("split", split))

    async def assign_splits(
        self,
        dataset_name: str,
        train: int = 70,
        val: int = 20,
        test: int = 10,
        seed: int = 42,
        mode: Literal["random", "embedding"] = "random",
    ) -> dict[str, int]:
        """One-click "Rebalance": assign a train/val/test split across the WHOLE
        (non-archived) dataset by ratio, in one atomic step.

        Async twin of :meth:`pictograph.resources.images.Images.assign_splits` -
        weights are integer percentages; ``val``/``test`` take their floor and
        ``train`` the remainder, deterministic under ``seed``. ``mode`` selects
        a global shuffle (``"random"``) or a per-cluster one (``"embedding"``),
        exactly as on the sync twin. Member role or higher.

        Returns:
            The assigned counts, e.g.
            ``{"processed": 100, "train": 70, "val": 20, "test": 10}``, plus
            ``clusters``/``unclustered`` in ``"embedding"`` mode.
        """
        response = await self._transport.request(
            "POST",
            f"{_API_PATH}/assign-splits",
            json={
                "dataset": dataset_name,
                "train": train,
                "val": val,
                "test": test,
                "seed": seed,
                "split_mode": mode,
            },
        )
        data = response.get("data", response)
        out = {k: int(data.get(k, 0)) for k in ("processed", "train", "val", "test")}
        # Only present in embedding mode; omitted rather than defaulted to 0, so a
        # caller can tell "no clusters reported" from "zero clusters".
        for extra in ("clusters", "unclustered"):
            if extra in data:
                out[extra] = int(data[extra])
        return out

    # ───────────── directory upload ─────────────

    async def upload_from_directory(
        self,
        dataset_name: str,
        directory: str | Path,
        *,
        organize_by_class: bool = True,
        preserve_structure: bool = False,
        max_workers: int = _DEFAULT_FOLDER_WORKERS,
        skip_existing: bool = True,
        create_if_missing: bool = True,
        progress: Callable[[int, int, str | None], None] | None = None,
    ) -> UploadReport:
        """Walk ``directory`` and upload every supported image to ``dataset_name`` (async).

        Async twin of :meth:`pictograph.resources.images.Images.upload_from_directory`,
        uploading **concurrently** via ``asyncio.gather`` (bounded by ``max_workers``)
        rather than a thread pool - the natural fit inside an async application.
        Directory layout is decided by the shared
        :func:`~pictograph.resources.images.virtual_directory_for`, so sync and async
        cannot drift.

        Args:
            dataset_name: Destination dataset (project) within your org.
            directory: Local directory to walk (recursive).
            organize_by_class: When ``True`` (default), each first-level subdirectory
                becomes a virtual directory on the destination dataset. Deeper nesting
                COLLAPSES onto that first level - pass ``preserve_structure=True`` to
                recreate the whole tree instead.
            preserve_structure: Recreate the directory's FULL subdirectory tree
                (``cars/red/x.jpg`` -> ``/cars/red``), matching the web app's directory
                upload. Takes precedence over ``organize_by_class``.
            max_workers: Max concurrent uploads in flight (default 8).
            skip_existing: Record a same-filename conflict as ``skipped`` rather than
                a failure (default True).
            create_if_missing: Create the dataset if it doesn't exist (default True).
            progress: Optional ``(completed, total, filename)`` callback, fired after
                each file finishes (success or failure).

        Returns:
            An :class:`~pictograph.resources.images.UploadReport` with per-file
            failure context.

        Raises:
            FileNotFoundError: ``directory`` doesn't exist or isn't a directory.
            NotFoundError: ``dataset_name`` missing and ``create_if_missing=False``.
        """
        from pictograph.aio.resources.datasets import AsyncDatasets

        root = Path(directory).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found or not a directory: {root}")

        datasets = AsyncDatasets(self._transport)
        try:
            project = await datasets.get(dataset_name)
        except NotFoundError:
            if not create_if_missing:
                raise
            project = await datasets.create(dataset_name)
        dataset_id = project.id

        tasks: list[_DirectoryTask] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTS:
                continue
            virtual = virtual_directory_for(
                path.relative_to(root),
                organize_by_class=organize_by_class,
                preserve_structure=preserve_structure,
            )
            tasks.append(_DirectoryTask(local_path=path, virtual_directory_path=virtual))

        report = UploadReport(dataset_name=dataset_name, images_attempted=len(tasks))
        if not tasks:
            return report

        semaphore = asyncio.Semaphore(max(1, max_workers))
        completed = 0

        async def upload_one(task: _DirectoryTask) -> None:
            nonlocal completed
            status, error = await self._upload_directory_file(
                dataset_id, task, skip_existing=skip_existing
            )
            if status == "uploaded":
                report.images_uploaded += 1
            elif status == "skipped":
                report.images_skipped += 1
            else:
                report.failures.append(
                    UploadFailure(path=task.local_path, reason=error or "unknown")
                )
            completed += 1
            if progress is not None:
                progress(completed, len(tasks), task.local_path.name)

        async def guarded(task: _DirectoryTask) -> None:
            async with semaphore:
                await upload_one(task)

        await asyncio.gather(*(guarded(t) for t in tasks))
        return report

    async def _upload_directory_file(
        self,
        dataset_id: str,
        task: _DirectoryTask,
        *,
        skip_existing: bool,
    ) -> tuple[str | None, str | None]:
        """Upload one file; return ``(status, error)`` - status is uploaded/skipped/None."""
        try:
            await self.upload(
                dataset_name=dataset_id,  # a uuid - _resolve passes it straight through
                file_path=task.local_path,
                directory_path=task.virtual_directory_path,
            )
            return "uploaded", None
        except ConflictError as e:
            if skip_existing:
                return "skipped", str(e)
            return None, f"conflict: {e}"
        except ApiError as e:
            # Backend currently raises 400 (not 409) on duplicate filename in a directory;
            # treat "already exists" as a conflict so re-runs stay idempotent.
            if skip_existing and "already exists" in str(e).lower():
                return "skipped", str(e)
            return None, str(e)
        except PictographError as e:
            return None, str(e)
        except OSError as e:
            return None, str(e)
