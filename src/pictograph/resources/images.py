"""Images resource: metadata, streaming download, chunked upload, delete.

Bulk shapes live here too, on the noun they act on: :meth:`Images.upload_from_directory`
(walk a local tree), :meth:`Images.augment` (generate variants of a dataset) and
:meth:`Images.tile` (slice a dataset into a grid). Each returns a report dataclass
rather than raising on a single bad file, so a partial run is inspectable and
retryable.

Upload uses the backend's three-step signed-URL flow (``upload-url`` →
chunked ``PUT`` to storage → ``register``) rather than the convenience
``POST /upload`` endpoint. The three-step path:

- Sends bytes directly to object storage on a separate host, never relayed
  through the Pictograph API (cheaper, faster, and bypasses the 10 MB
  request-body limit).
- Lets us stream chunks with a progress callback.
- Yields the same final result (a registered :class:`Image`) as the
  one-step path - ``register`` returns the full canonical image, so no
  follow-up metadata fetch is needed.

Image dimensions are extracted client-side via Pillow when possible and
sent to ``register`` so the backend doesn't have to re-decode the bytes.
A missing or undecodable file falls through with ``width=None, height=None``
(the backend tolerates this).
"""

from __future__ import annotations

import contextlib
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

try:
    from PIL import (
        Image as _PILImage,
        ImageOps as _PIL_ImageOps,
        UnidentifiedImageError as _UnidentifiedImageError,
    )

    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - Pillow is a base dep, but degrade if absent
    _PIL_AVAILABLE = False

from pictograph._http.pagination import OffsetPager
from pictograph._http.streaming import DEFAULT_CHUNK_SIZE
from pictograph.exceptions import (
    ApiError,
    ConflictError,
    NotFoundError,
    PictographError,
)
from pictograph.models.image import Image, ImageSplit, ImageStatus
from pictograph.resources import _resolve
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from pictograph.augment import Augmentation
    from pictograph.resources.datasets import Datasets


_API_PATH = "/api/v1/developer/images"


def _image_route(dataset_name: str, segment: str, suffix: str = "") -> str:
    """`/images/{dataset}/{directory}/{filename}[/suffix]`.

    The per-image routes fold the directory into a trailing `:path` segment.
    These call sites used to resolve the filename to a UUID and request
    `/images/{uuid}`, which serves GET only - so delete, review and set_split
    were hitting a route that does not accept their verb at all.
    """
    return f"{_API_PATH}/{quote(dataset_name, safe='')}/{quote(segment, safe='/')}{suffix}"


# Filename suffix → MIME type. Backend accepts anything image/*; an unknown
# suffix falls back to a generic binary type and the backend will reject it,
# which is the right behaviour (don't silently pretend a .txt is an image).
_CONTENT_TYPE_BY_SUFFIX: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".heic": "image/heic",
    ".heif": "image/heif",
}


class BulkUploadFailure(BaseModel):
    """One file that couldn't be registered in a :meth:`Images.bulk_upload` call
    (e.g. a filename that already exists in the target directory)."""

    model_config = ConfigDict(frozen=True)

    filename: str
    error: str
    directory_path: str = "/"


class BulkUploadResult(BaseModel):
    """Outcome of :meth:`Images.bulk_upload`.

    Attributes:
        succeeded: One full :class:`Image` per file that was uploaded AND
            registered (the backend returns the canonical image shape on
            register - no re-fetch needed).
        failed: One :class:`BulkUploadFailure` per file the backend declined to
            register (the bytes may still have uploaded to storage).
    """

    model_config = ConfigDict(frozen=True)

    succeeded: list[Image]
    failed: list[BulkUploadFailure]

    @property
    def count(self) -> int:
        """Number of files successfully uploaded + registered."""
        return len(self.succeeded)


# ───────────── directory upload ─────────────

#: Default concurrency for :meth:`Images.upload_from_directory`. Higher values risk
#: tripping the per-org rate limit.
_DEFAULT_FOLDER_WORKERS = 8

#: Suffixes :meth:`Images.upload_from_directory` treats as images. Anything else in
#: the walked tree is ignored rather than uploaded and rejected.
_IMAGE_EXTS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".heic"}
)


class UploadFailure(BaseModel):
    """One file that failed to upload in :meth:`Images.upload_from_directory`."""

    model_config = ConfigDict(frozen=True)

    path: Path
    reason: str


class UploadReport(BaseModel):
    """Outcome of an :meth:`Images.upload_from_directory` call.

    ``success`` is ``True`` only when every supported file landed. Inspect
    ``failures`` to retry the subset - a partial upload does not raise.
    """

    dataset_name: str
    images_attempted: int = 0
    images_uploaded: int = 0
    images_skipped: int = 0
    failures: list[UploadFailure] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.failures and self.images_uploaded > 0


def virtual_directory_for(
    relative: Path,
    *,
    organize_by_class: bool,
    preserve_structure: bool,
) -> str:
    """Destination ``virtual_directory_path`` for a file at ``relative`` within the root.

    The single source of truth for directory layout, shared by
    :meth:`Images.upload_from_directory` and its async twin so the two can't drift.

    * ``preserve_structure`` - recreate the FULL directory chain
      (``cars/red/x.jpg`` -> ``/cars/red``). Wins over ``organize_by_class``.
    * ``organize_by_class`` - first level only (``cars/red/x.jpg`` -> ``/cars``), for
      ImageFolder-style datasets where the top directory *is* the class.
    * neither - everything at the root.

    Returns the canonical form the API stores: ``"/"`` or ``"/a/b"``.
    """
    parts = relative.parts[:-1]  # drop the filename
    if not parts:
        return "/"
    if preserve_structure:
        return "/" + "/".join(parts)
    if organize_by_class:
        return "/" + parts[0]
    return "/"


# ───────────── augment / tile ─────────────

_DEFAULT_AUGMENT_FOLDER = "/augmented"
_DEFAULT_TILE_FOLDER = "/tiles"


class AugmentFailure(BaseModel):
    """One source image that failed to augment (kind in ``reason``)."""

    model_config = ConfigDict(frozen=True)

    image_id: str
    filename: str
    reason: str


class AugmentReport(BaseModel):
    """Outcome of an :meth:`Images.augment` run.

    ``success`` is ``True`` when at least one variant landed and nothing failed.
    Inspect ``failures`` to retry the affected source images.
    """

    source: str
    target: str
    source_images: int = 0
    originals_copied: int = 0
    variants_created: int = 0
    annotations_written: int = 0
    skipped_empty: int = 0
    failures: list[AugmentFailure] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.failures and self.variants_created > 0


class TileFailure(BaseModel):
    """One source image that failed to tile (kind in ``reason``)."""

    model_config = ConfigDict(frozen=True)

    image_id: str
    filename: str
    reason: str


class TileReport(BaseModel):
    """Outcome of an :meth:`Images.tile` run.

    ``success`` is ``True`` when at least one tile landed and nothing failed.
    Inspect ``failures`` to retry the affected source images.
    """

    source: str
    target: str
    source_images: int = 0
    tiles_created: int = 0
    empty_tiles: int = 0
    annotations_written: int = 0
    failures: list[TileFailure] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.failures and self.tiles_created > 0


class Images(Resource):
    """Operations on individual images within a dataset."""

    # ───────────── list / iter ─────────────

    def list(
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

        For full enumeration prefer :meth:`iter`, which auto-pages.

        Args:
            dataset_name: Dataset NAME. An id is accepted too.
            directory_path: Restrict to one virtual directory (e.g. ``"/train"``).
                ``None`` (default) lists images across every directory.
            status: Restrict to one annotation stage (``"new"`` / ``"annotate"``
                / ``"review"`` / ``"complete"``). ``None`` lists all stages.
            split: Filter to one dataset split (``"train"`` / ``"val"`` /
                ``"test"``). ``None`` (default) lists all splits. Pull a training
                set with ``list(dataset_id, split="train")``.
            include_archived: Include soft-deleted (archived) images. Defaults
                to ``False`` - the same set the app's Data tab shows.
            min_confidence_lt: Active-learning filter - keep only images whose
                ``min_confidence`` is below this (0..1). ``None`` (default) applies
                no confidence filter. Use it to page the model-uncertainty review
                queue, e.g. ``list(dataset_id, min_confidence_lt=0.9)``.
            limit: Page size (backend cap: 1000).
            offset: Pagination offset.

        Returns:
            Up to ``limit`` :class:`Image` objects. Fewer than ``limit`` means
            the last page.
        """
        response = self._transport.request(
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
                filename=filename,
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
    ) -> OffsetPager[Image]:
        """Auto-paging iterator over every image in a dataset, newest first.

        Returns a :class:`OffsetPager` - iterate it directly, materialise with
        ``.all()``, or peek the first match with ``.first()``. This is the
        method that closes the long-standing gap of "list a dataset's images"::

            for img in client.images.iter(dataset_id, directory_path="/train"):
                print(img.filename, img.annotation_count)

        Filters mirror :meth:`list`.

        Args:
            dataset_name: Dataset NAME. An id is accepted too.
            directory_path: Restrict to one virtual directory; ``None`` iterates all.
            status: Restrict to one annotation stage; ``None`` iterates all.
            include_archived: Include archived images (default ``False``).
            min_confidence_lt: Active-learning filter - iterate only images whose
                ``min_confidence`` is below this (0..1); ``None`` iterates all.
            page_size: Items fetched per backend round-trip (backend cap: 1000).
            max_total: Stop after this many items, even mid-page. ``None``
                yields every match.
        """

        def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            return cast(
                "Mapping[str, Any]",
                self._transport.request(
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
                        filename=filename,
                    ),
                ),
            )

        return OffsetPager(
            fetch,
            items_key="data",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(Image, raw),
        )

    # ───────────── metadata ─────────────

    def get(self, dataset_name: str, image: str) -> Image:
        """Fetch metadata for a single image.

        For the actual annotations on the image, use
        :meth:`Annotations.get` - the SDK exposes annotations as a separate
        resource so the data ownership boundary stays clean.
        """
        # A FILENAME is answered by the list route in ONE request - it already
        # returns the whole row, so resolving the id and then re-fetching that
        # same row by id (which is what this used to do) was a wasted
        # round-trip. An id still goes straight to the metadata route.
        if not _resolve.looks_like_id(image):
            matches = self.list(dataset_name, filename=image, limit=2)
            if not matches:
                raise NotFoundError(f"No image named {image!r} in dataset {dataset_name!r}.")
            if len(matches) > 1:
                directories = sorted({m.directory_path or "/" for m in matches})
                raise ValueError(
                    f"{image!r} exists in more than one directory of {dataset_name!r} "
                    f"({', '.join(directories)}). Fetch it by id, or narrow with list()."
                )
            return matches[0]
        # An id goes to the by-id route, which exists and returns the whole row.
        # `/images/{id}/metadata` does NOT exist - the metadata route is
        # `/images/{dataset}/{image_path}/metadata`, so the id form 404d.
        response = self._transport.request("GET", f"{_API_PATH}/{image}")
        return self._parse(Image, response["data"])

    # ───────────── download ─────────────

    def download(
        self,
        dataset_name: str,
        image: str,
        output_path: str | Path,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> Path:
        """Stream image bytes to ``output_path``; return the path.

        Parent directories are created if missing. Bytes land in a sibling
        ``<name>.part`` file first and are renamed atomically onto
        ``output_path`` only after the transfer completes - a 404 / 5xx /
        network failure mid-download leaves no partial file at the
        destination, so retries are safe with the same call.
        """
        # Direct per-image route, like delete / review / set_split - a filename
        # needs no lookup at all, which is the point of server-side resolution.
        segment = _resolve.image_segment(self._transport, image)
        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")
        try:
            with tmp.open("wb") as fh:
                for chunk in self._transport.stream_bytes(
                    "GET",
                    _image_route(dataset_name, segment),
                ):
                    fh.write(chunk)
                    _ = chunk_size  # forward-compat hint; httpx default chunking
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        tmp.replace(out)
        return out

    def download_bundle(
        self,
        dataset_name: str,
        image: str,
        output_path: str | Path,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> Path:
        """Stream one image's DATA BUNDLE to ``output_path``; return the path.

        The bundle is a zip carrying the original image bytes, its depth map
        (when one has been generated), its annotations as Pictograph JSON, and a
        ``manifest.json`` naming exactly what is and is not inside - the same
        archive the annotation editor's "Image data" button hands over, byte for
        byte, because both are assembled by one server-side builder.

        A missing depth map is not an error: the zip still arrives and the
        manifest records the omission and why.

        Same atomic-write discipline as :meth:`download` - bytes land in a
        sibling ``<name>.part`` and are renamed onto ``output_path`` only once
        the transfer completes, so a failure mid-download leaves no partial zip
        and retrying with the same call is safe.
        """
        segment = _resolve.image_segment(self._transport, image)
        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")
        try:
            with tmp.open("wb") as fh:
                for chunk in self._transport.stream_bytes(
                    "GET",
                    _image_route(dataset_name, segment, "/data-bundle"),
                ):
                    fh.write(chunk)
                    _ = chunk_size  # forward-compat hint; httpx default chunking
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        tmp.replace(out)
        return out

    # ───────────── upload ─────────────

    def upload(
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
            dataset_name: Dataset NAME. An id is accepted too.
            file_path: Local file to upload. Read in chunks; never loaded
                fully into memory.
            directory_path: Virtual directory within the dataset (created if
                missing). Defaults to root (``"/"``).
            filename: Override the destination filename. Defaults to the
                local file's basename.
            content_type: Override the inferred MIME type. Defaults to a
                lookup by file extension (``.jpg`` → ``image/jpeg``, etc).
            progress: Optional ``(bytes_sent, total_bytes)`` callback fired
                during the upload phase.

        Returns:
            The newly-registered :class:`Image`.

        Raises:
            FileNotFoundError: ``file_path`` does not exist.
            ValueError: ``content_type`` cannot be inferred and was not given.
            ApiError / NetworkError: Per-step backend or transport failures.
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

        # Step 1 - request a signed upload URL.
        url_response = self._transport.request(
            "POST",
            f"{_API_PATH}/upload-url",
            json={
                "dataset": dataset_name,
                "filename": resolved_filename,
                "directory_path": directory_path,
                "content_type": resolved_content_type,
            },
        )

        # Step 2 - PUT bytes to object storage in chunks (no SDK auth on this URL).
        self._transport.upload_external(
            url_response["data"]["upload_url"],
            path,
            content_type=resolved_content_type,
            progress=progress,
        )

        # Step 3 - register the uploaded blob in the database (JSON body, the
        # same field names step 1 used; the server re-derives storage paths).
        register_response = self._transport.request(
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

        # Register returns the full canonical image - no re-fetch round-trip.
        return self._parse(Image, register_response["data"])

    def bulk_upload(
        self,
        dataset_name: str,
        file_paths: Sequence[str | Path],
        *,
        directory_path: str = "/",
        max_workers: int = _DEFAULT_FOLDER_WORKERS,
        progress: Callable[[int, int], None] | None = None,
    ) -> BulkUploadResult:
        """Upload many images to one directory with two server round-trips, not N.

        Orchestrates the bulk three-step flow: one ``bulk-upload-url`` call for
        all files, a per-file PUT to storage, then one ``bulk-register`` call. Far
        fewer API round-trips than N :meth:`upload` calls (and it spawns the
        SigLIP2 embedding/auto-tag pass once for the whole batch). Up to 500
        files per call.

        Read ``succeeded`` for the registered images (each a full
        :class:`Image` - the backend returns the canonical shape on register)
        and ``failed`` for any the backend declined (e.g. a filename collision
        in the target directory).

        Args:
            dataset_name: Dataset NAME. An id is accepted too.
            file_paths: Local image files (all land in ``directory_path``). Each is
                read in chunks; never fully loaded into memory.
            directory_path: Virtual directory within the dataset (created if missing).
            progress: Optional ``(bytes_sent, total_bytes)`` callback, fired
                per file during its upload PUT.

        Returns:
            :class:`BulkUploadResult` (``succeeded`` / ``failed`` / ``count``).

        Raises:
            FileNotFoundError: A path doesn't exist.
            ValueError: ``file_paths`` is empty, or a file's MIME type can't be
                inferred from its name.
            ApiError / NetworkError: A backend or upload-PUT failure. A PUT failure
                aborts the batch (already-PUT blobs are left unregistered).
        """
        paths = [Path(p).expanduser() for p in file_paths]
        if not paths:
            raise ValueError("file_paths must contain at least one file.")

        # Gather metadata up front (also validates every file before any upload).
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

        # Step 1 - one signed-URL request for the whole batch. The backend
        # preserves request order, so we match by index (robust to server-side
        # filename sanitisation, which the matching-by-name approach is not).
        url_response = self._transport.request(
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

        # Step 2 - PUT each file's bytes to its signed URL (no SDK auth there).
        #
        # These ran one after another, which made "bulk" upload two API calls
        # plus N SEQUENTIAL round-trips to storage - measured at 709ms/image, a
        # bare 2x over uploading them one at a time. Each PUT is
        # independent and network-bound, so they overlap in a bounded pool, the
        # same way upload_from_directory already does.
        #
        # `metas` is NOT reordered: step 3 registers by index against the URLs
        # minted in step 1, so the pool parallelises the side effects only.
        # A failure still aborts the whole batch before anything is registered -
        # the first exception propagates out of the `with` block.
        if max_workers > 1 and len(metas) > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(metas))) as pool:
                futures = [
                    pool.submit(
                        self._transport.upload_external,
                        url_entry["upload_url"],
                        meta["path"],
                        content_type=meta["content_type"],
                        progress=progress,
                    )
                    for meta, url_entry in zip(metas, upload_urls, strict=True)
                ]
                for future in futures:
                    future.result()
        else:
            for meta, url_entry in zip(metas, upload_urls, strict=True):
                self._transport.upload_external(
                    url_entry["upload_url"],
                    meta["path"],
                    content_type=meta["content_type"],
                    progress=progress,
                )

        # Step 3 - one register call. We resend the same filename we sent in
        # step 1, so the backend's identical sanitisation reproduces the same
        # gcs_path it just minted a URL for.
        register_response = self._transport.request(
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

    def delete(self, dataset_name: str, image: str, *, permanent: bool = False) -> None:
        """Archive (default) or permanently delete an image.

        Archived images survive in the Archive tab and can be restored. A
        ``permanent=True`` delete removes the stored object and the database
        row irreversibly - there is no undo. Requires admin or owner role.
        """
        segment = _resolve.image_segment(self._transport, image)
        params: dict[str, Any] | None = {"permanent": "true"} if permanent else None
        self._transport.request(
            "DELETE",
            _image_route(dataset_name, segment),
            params=cast("Any", params),
        )

    # ───────────── bulk tagging ─────────────

    def bulk_tag(
        self,
        dataset_name: str,
        image_ids: Sequence[str],
        tags: Sequence[str],
        *,
        add: bool = True,
    ) -> int:
        """Add or remove user tags across many images in ONE server-side call.

        These are the user ``image_tags`` (the field the app's tag filters use)
        - NOT the SigLIP2/Gemini auto-tags. The op is org-scoped, chunked, and
        idempotent: ``add=True`` unions the tags in (re-applying a tag is a
        no-op), ``add=False`` removes them. No client-side fan-out - one bulk
        request regardless of how many images. Member role or higher.

        Args:
            dataset_name: Dataset NAME. An id is accepted too.
            image_ids: Image UUIDs to tag/untag.
            tags: User tag names to add or remove.
            add: ``True`` to add the tags (default), ``False`` to remove them.

        Returns:
            The number of images the operation touched.
        """
        body = {
            "dataset": dataset_name,
            "image_ids": list(image_ids),
            "tags": list(tags),
            "add": add,
        }
        response = self._transport.request("POST", f"{_API_PATH}/bulk-tag", json=body)
        data = response.get("data", response)
        return int(data.get("processed", 0))

    def review(
        self,
        dataset_name: str,
        image: str,
        action: Literal["approve", "request_changes"],
        *,
        note: str | None = None,
    ) -> ImageStatus:
        """Approve or request changes on an image (the annotation review workflow).

        ``approve`` marks the image ``complete`` and clears any pending review
        note; ``request_changes`` sends it back to ``annotate`` with ``note`` -
        the message the annotator sees in the editor. Enables programmatic QA
        (e.g. auto-approve high-confidence predictions, bounce low-confidence
        ones for a human). Member role or higher.

        Args:
            image_id: The image UUID.
            action: ``"approve"`` or ``"request_changes"``.
            note: Optional note surfaced to the annotator on ``request_changes``.

        Returns:
            The image's new status - ``"complete"`` for approve, ``"annotate"``
            for request_changes.
        """
        body: dict[str, Any] = {"action": action}
        if note is not None:
            body["note"] = note
        segment = _resolve.image_segment(self._transport, image)
        response = self._transport.request(
            "POST", _image_route(dataset_name, segment, "/review"), json=body
        )
        data = response.get("data", response)
        default: ImageStatus = "complete" if action == "approve" else "annotate"
        return cast("ImageStatus", data.get("status", default))

    def set_split(
        self, dataset_name: str, image: str, split: ImageSplit | None
    ) -> ImageSplit | None:
        """Assign (or clear, ``split=None``) an image's train/val/test dataset split.

        Enables programmatic dataset organization - e.g. deterministically split a
        dataset 80/10/10 for training. Member role or higher. Returns the split
        the image now carries.

        Args:
            image_id: The image UUID.
            split: ``"train"`` / ``"val"`` / ``"test"``, or ``None`` to unassign.
        """
        segment = _resolve.image_segment(self._transport, image)
        response = self._transport.request(
            "POST", _image_route(dataset_name, segment, "/split"), json={"split": split}
        )
        data = response.get("data", response)
        return cast("ImageSplit | None", data.get("split", split))

    def assign_splits(
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

        The far faster way to organize a dataset than per-image :meth:`set_split`.
        Weights are integer percentages; ``val``/``test`` take their floor and
        ``train`` the remainder, so the counts sum to the total exactly (a ``0``
        weight yields ``0`` images for that bucket, e.g. ``80/20/0``). The
        assignment is a deterministic pseudo-random shuffle under ``seed``. Member
        role or higher. Pairs with :meth:`Exports.create` (``organize_by_split=True``).

        Args:
            dataset_id: The dataset UUID to rebalance.
            train: Train weight (%). Default 70.
            val: Validation weight (%). Default 20.
            test: Test weight (%). Default 10.
            seed: Deterministic shuffle seed. Default 42.
            mode: ``"random"`` (default) shuffles the whole dataset once.
                ``"embedding"`` takes the same ratio out of EACH cluster the
                embedding map found, so no visual mode of the data can land
                entirely in one split - a global random split can hand a whole
                region of the distribution to train and leave the evaluation
                blind to it. Requires an embedding map to exist for the dataset;
                raises :class:`~pictograph.exceptions.ConflictError` if none
                does, rather than silently splitting at random.

        Returns:
            The assigned counts, e.g.
            ``{"processed": 100, "train": 70, "val": 20, "test": 10}``. In
            ``"embedding"`` mode also ``clusters`` (how many groups the ratio was
            applied within, the unclustered group included) and ``unclustered``
            (images the map had never seen, split as their own group).

        Example:
            >>> client.images.assign_splits(ds.id, mode="embedding")
            {'processed': 412, 'train': 288, 'val': 82, 'test': 42, 'clusters': 9, 'unclustered': 3}
        """
        response = self._transport.request(
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

    def upload_from_directory(
        self,
        dataset_name: str,
        directory: str | Path,
        *,
        organize_by_class: bool = True,
        preserve_structure: bool = False,
        parallel: bool = True,
        max_workers: int = _DEFAULT_FOLDER_WORKERS,
        skip_existing: bool = True,
        create_if_missing: bool = True,
        progress: Callable[[int, int, str | None], None] | None = None,
    ) -> UploadReport:
        """Walk ``directory`` and upload every supported image to ``dataset_name``.

        The many-files form of :meth:`upload`: it walks a local directory,
        ensures the destination dataset exists, and maps subdirectory names onto
        virtual directories. Partial success does not raise - inspect the report's
        ``failures`` to retry a subset.

        Args:
            dataset_name: Destination dataset within your org.
            directory: Local directory to walk. Recursive by default.
            organize_by_class: When ``True`` (default), each first-level
                subdirectory becomes a virtual directory on the destination
                dataset (e.g. ``./images/cars/x.jpg`` lands under ``/cars``).
              **Deeper nesting collapses** - ``./images/cars/red/x.jpg`` also
                lands under ``/cars``. That is intentional for ImageFolder-style
                datasets where the top directory *is* the class; pass
                ``preserve_structure=True`` when you want the whole tree.
                When ``False`` (and ``preserve_structure=False``), all files land
                at root.
            preserve_structure: Recreate the directory's FULL subdirectory tree on the
                dataset (``./images/cars/red/x.jpg`` -> ``/cars/red``), matching what
                the web app does when you pick or drag a directory. Takes precedence
                over ``organize_by_class``.
            parallel: Use a thread pool for concurrent uploads.
            max_workers: Pool size when ``parallel=True``. Defaults to 8 -
                higher values risk hitting the per-org rate limit.
            skip_existing: When ``True`` (default), conflict errors (image
                with the same filename already exists in the same directory)
                are recorded as ``skipped`` rather than failures.
            create_if_missing: When ``True`` (default), create the dataset
                if it doesn't exist. Pass ``False`` to require the dataset
                to be pre-created (and raise otherwise).
            progress: Optional ``(completed, total, filename)`` callback,
                fired after each file finishes (success or failure).

        Returns:
            :class:`UploadReport` with per-file failure context.

        Raises:
            FileNotFoundError: ``directory`` doesn't exist or isn't a directory.
            NotFoundError: ``dataset_name`` missing and ``create_if_missing=False``.

        Example:
            >>> report = client.images.upload_from_directory("road-signs", "./images")
            >>> report.images_uploaded
            412
        """
        from pictograph.resources.datasets import Datasets

        root = Path(directory).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found or not a directory: {root}")

        # Ensure dataset exists; capture the UUID for the per-file upload().
        datasets = Datasets(self._transport)
        try:
            project = datasets.get(dataset_name)
        except NotFoundError:
            if not create_if_missing:
                raise
            project = datasets.create(dataset_name)
        # Keep the id: this pre-resolves so the per-file upload() below does not
        # repeat the lookup once per image.
        dataset_id = project.id

        # Discover image files.
        @dataclass(frozen=True)
        class _Task:
            local_path: Path
            virtual_directory_path: str

        tasks: list[_Task] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _IMAGE_EXTS:
                continue
            virtual = virtual_directory_for(
                path.relative_to(root),
                organize_by_class=organize_by_class,
                preserve_structure=preserve_structure,
            )
            tasks.append(_Task(local_path=path, virtual_directory_path=virtual))

        report = UploadReport(dataset_name=dataset_name, images_attempted=len(tasks))
        if not tasks:
            return report

        def _upload_one(task: _Task) -> tuple[_Task, str | None, str | None]:
            """Returns (task, status, error). status is "uploaded" / "skipped" / None."""
            try:
                self.upload(
                    dataset_name=dataset_id,  # a uuid - _resolve passes it straight through
                    file_path=task.local_path,
                    directory_path=task.virtual_directory_path,
                )
                return task, "uploaded", None
            except ConflictError as e:
                if skip_existing:
                    return task, "skipped", str(e)
                return task, None, f"conflict: {e}"
            except ApiError as e:
                # Backend currently raises 400 (not 409) on duplicate filename in a directory.
                # Treat "already exists" as a conflict for skip_existing purposes so the
                # call stays idempotent across re-runs even before the backend fixes
                # the status code.
                if skip_existing and "already exists" in str(e).lower():
                    return task, "skipped", str(e)
                return task, None, str(e)
            except PictographError as e:
                # Any other SDK-domain error on a single file (NetworkError /
                # RequestTimeoutError from the storage PUT phase, a transient
                # ServerError / RateLimitError, PaymentRequiredError, ...) is recorded
                # as a per-file failure, never an uncaught raise - the report contract
                # is "partial success doesn't raise" (ConflictError / ApiError are
                # handled above). Real Python bugs (not PictographError) still propagate.
                return task, None, str(e)
            except OSError as e:
                return task, None, str(e)

        completed = 0
        if parallel and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_upload_one, t) for t in tasks]
                for future in as_completed(futures):
                    task, status, error = future.result()
                    completed += 1
                    if status == "uploaded":
                        report.images_uploaded += 1
                    elif status == "skipped":
                        report.images_skipped += 1
                    else:
                        report.failures.append(
                            UploadFailure(path=task.local_path, reason=error or "unknown")
                        )
                    if progress is not None:
                        progress(completed, len(tasks), task.local_path.name)
        else:
            for task in tasks:
                t, status, error = _upload_one(task)
                completed += 1
                if status == "uploaded":
                    report.images_uploaded += 1
                elif status == "skipped":
                    report.images_skipped += 1
                else:
                    report.failures.append(
                        UploadFailure(path=t.local_path, reason=error or "unknown")
                    )
                if progress is not None:
                    progress(completed, len(tasks), t.local_path.name)

        return report

    # ───────────── augment / tile ─────────────

    def augment(
        self,
        source: str,
        ops: Sequence[Augmentation],
        *,
        multiplier: int = 3,
        into: str | None = None,
        include_original: bool = True,
        directory_path: str = _DEFAULT_AUGMENT_FOLDER,
        seed: int | None = None,
        max_source_images: int | None = None,
        jpeg_quality: int = 95,
        drop_classes: Iterable[str] | None = None,
        skip_empty: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> AugmentReport:
        """Generate ``multiplier`` augmented variants of every image in ``source``.

        Synchronous-only: there is no ``AsyncClient`` twin. The per-image cost is
        CPU-bound Pillow work, run sequentially for deterministic output, which
        ``asyncio`` would not overlap.

        For each source image this downloads the image + its annotations, produces
        ``multiplier`` augmented variants (image transformed, geometry remapped by
        :mod:`pictograph.augment`), and uploads each variant into the target dataset
        under ``directory_path`` - where the standard ingest pipeline embeds +
        auto-tags + thumbnails it, exactly as for a normal upload. Runs sequentially
        so a fixed ``seed`` yields byte-identical output across re-runs.

        Every generated image counts toward your organization's image quota.

        Args:
            source: Source dataset name.
            ops: Augmentation ops (from :mod:`pictograph.augment`) applied in order.
            multiplier: Variants generated per source image (>= 1).
            into: Target dataset name. ``None`` (or equal to ``source``) appends the
                variants into the source dataset itself; any other name is created
                if missing, copying the source's class config.
            include_original: When writing to a *new* dataset, also copy each
                original image + annotations (so the new dataset is a superset).
                Ignored when appending to the source (the originals are already there).
            directory_path: Virtual directory the generated images land in.
            seed: RNG seed for reproducible variants.
            max_source_images: Cap the number of source images processed (handy for
                a quick trial). ``None`` processes all.
            jpeg_quality: Quality for the generated JPEG images (1-100).
            drop_classes: Preprocessing - annotation class names to remove before
                augmenting. Dropped classes are also removed from a newly-created
                target's class config.
            skip_empty: Preprocessing - when ``True``, a source image left with **no**
                annotations (originally, or after ``drop_classes``) is skipped entirely
                and counted in ``report.skipped_empty``.
            on_progress: Optional ``(done, total)`` callback fired per source image.

        Returns:
            An :class:`AugmentReport`. Per-image failures are collected rather than
            raised, so a single bad image never aborts the batch.

        Raises:
            ValueError: ``multiplier`` < 1.
            NotFoundError: ``source`` does not exist.

        Example:
            >>> from pictograph.augment import HorizontalFlip, Rotate
            >>> client.images.augment(
            ...     "road-signs", [HorizontalFlip(), Rotate((-15, 15))], into="road-signs-aug"
            ... ).variants_created
            1236
        """
        from pictograph.augment import Augmenter
        from pictograph.resources.annotations import Annotations
        from pictograph.resources.datasets import Datasets

        if multiplier < 1:
            raise ValueError("multiplier must be >= 1")

        datasets = Datasets(self._transport)
        annotations_resource = Annotations(self._transport)
        drop = frozenset(drop_classes or ())
        target_id, target_name, same = _ensure_target(datasets, source, into, drop)
        source_ds = datasets.get(source)
        augmenter = Augmenter(ops, seed=seed)
        report = AugmentReport(source=source, target=target_name)

        # Materialise the source image list up front so variants we upload into the
        # same dataset are never re-augmented.
        images = list(self.iter(source_ds.id, max_total=max_source_images))
        total = len(images)

        with tempfile.TemporaryDirectory(prefix="pictograph-augment-") as tmpdir:
            tmp = Path(tmpdir)
            for done, image in enumerate(images, start=1):
                try:
                    src_path = tmp / f"src_{image.id}"
                    annotations = annotations_resource.get(source, image.id)
                    if drop:
                        annotations = [a for a in annotations if a.name not in drop]
                    if skip_empty and not annotations:
                        report.skipped_empty += 1
                        continue
                    self.download(source, image.id, src_path)
                    stem = Path(image.filename).stem or "image"
                    short = image.id.replace("-", "")[:8]

                    if include_original and not same:
                        orig = self.upload(
                            target_id,
                            src_path,
                            directory_path=directory_path,
                            filename=image.filename,
                        )
                        if annotations:
                            annotations_resource.save(target_name, orig.id, annotations)
                            report.annotations_written += len(annotations)
                        report.originals_copied += 1

                    base = _PILImage.open(src_path).convert("RGB")
                    for k in range(multiplier):
                        aug_img, aug_anns = augmenter.augment(base, annotations)
                        var_path = tmp / f"{stem}_aug{k}_{short}.jpg"
                        aug_img.save(var_path, format="JPEG", quality=jpeg_quality)
                        uploaded = self.upload(
                            target_id,
                            var_path,
                            directory_path=directory_path,
                            filename=var_path.name,
                        )
                        report.variants_created += 1
                        if aug_anns:
                            annotations_resource.save(target_name, uploaded.id, aug_anns)
                            report.annotations_written += len(aug_anns)
                    report.source_images += 1
                except ConflictError as exc:
                    report.failures.append(
                        AugmentFailure(
                            image_id=image.id,
                            filename=image.filename,
                            reason=f"already exists: {exc}",
                        )
                    )
                except PictographError as exc:
                    report.failures.append(
                        AugmentFailure(image_id=image.id, filename=image.filename, reason=str(exc))
                    )
                finally:
                    if on_progress is not None:
                        on_progress(done, total)

        return report

    def tile(
        self,
        source: str,
        *,
        rows: int = 2,
        cols: int = 2,
        overlap: float = 0.0,
        min_visibility: float = 0.1,
        include_empty: bool = True,
        into: str | None = None,
        directory_path: str = _DEFAULT_TILE_FOLDER,
        max_source_images: int | None = None,
        jpeg_quality: int = 95,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> TileReport:
        """Slice every image in ``source`` into a ``rows`` x ``cols`` grid of tiles.

        Synchronous-only: there is no ``AsyncClient`` twin. Like :meth:`augment`,
        the cost is sequential, deterministic Pillow work, so ``asyncio`` buys
        nothing.

        For each source image this downloads the image + its annotations, cuts it
        into tiles (geometry clipped + translated per tile by :mod:`pictograph.tile`),
        and uploads each tile into the target dataset under ``directory_path`` - where
        the standard ingest pipeline embeds + auto-tags + thumbnails it. Runs
        sequentially and deterministically (tile geometry is a pure function of the
        grid), so a re-run yields byte-identical output.

        Every generated tile counts toward your organization's image quota.

        Args:
            source: Source dataset name.
            rows: Grid rows per image (>= 1).
            cols: Grid columns per image (>= 1).
            overlap: Fractional overlap added to each tile edge, ``[0.0, 0.9)``.
            min_visibility: Drop an annotation from a tile when less than this
                fraction of its area survives the clip.
            include_empty: When ``False``, tiles with no surviving annotations are
                not uploaded.
            into: Target dataset name. ``None`` (or equal to ``source``) appends the
                tiles into the source dataset itself; any other name is created if
                missing, copying the source's class config.
            directory_path: Virtual directory the generated tiles land in.
            max_source_images: Cap the number of source images processed. ``None``
                processes all.
            jpeg_quality: Quality for the generated JPEG tiles (1-100).
            on_progress: Optional ``(done, total)`` callback fired per source image.

        Returns:
            A :class:`TileReport`. Per-image failures are collected rather than
            raised, so a single bad image never aborts the batch.

        Raises:
            ValueError: ``rows``/``cols`` < 1.
            NotFoundError: ``source`` does not exist.

        Example:
            >>> client.images.tile("aerial", rows=2, cols=2, overlap=0.1, into="aerial-tiled")
            TileReport(source='aerial', target='aerial-tiled', ...)
        """
        from pictograph.resources.annotations import Annotations
        from pictograph.resources.datasets import Datasets
        from pictograph.tile import tile_image

        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must both be >= 1")

        datasets = Datasets(self._transport)
        annotations_resource = Annotations(self._transport)
        target_id, target_name, _same = _ensure_target(datasets, source, into, frozenset())
        source_ds = datasets.get(source)
        report = TileReport(source=source, target=target_name)

        # Materialise the source list up front so tiles uploaded into the same dataset
        # are never themselves re-tiled.
        images = list(self.iter(source_ds.id, max_total=max_source_images))
        total = len(images)

        with tempfile.TemporaryDirectory(prefix="pictograph-tile-") as tmpdir:
            tmp = Path(tmpdir)
            for done, image in enumerate(images, start=1):
                try:
                    src_path = tmp / f"src_{image.id}"
                    annotations = annotations_resource.get(source, image.id)
                    self.download(source, image.id, src_path)
                    stem = Path(image.filename).stem or "image"
                    short = image.id.replace("-", "")[:8]

                    base = _PILImage.open(src_path).convert("RGB")
                    tiles = tile_image(
                        base,
                        annotations,
                        rows=rows,
                        cols=cols,
                        overlap=overlap,
                        min_visibility=min_visibility,
                        include_empty=include_empty,
                    )
                    for t in tiles:
                        tile_path = tmp / f"{stem}_tile_r{t.row}_c{t.col}_{short}.jpg"
                        t.image.save(tile_path, format="JPEG", quality=jpeg_quality)
                        uploaded = self.upload(
                            target_id,
                            tile_path,
                            directory_path=directory_path,
                            filename=tile_path.name,
                        )
                        report.tiles_created += 1
                        if t.annotations:
                            annotations_resource.save(target_name, uploaded.id, t.annotations)
                            report.annotations_written += len(t.annotations)
                        else:
                            report.empty_tiles += 1
                    report.source_images += 1
                except ConflictError as exc:
                    report.failures.append(
                        TileFailure(
                            image_id=image.id,
                            filename=image.filename,
                            reason=f"already exists: {exc}",
                        )
                    )
                except PictographError as exc:
                    report.failures.append(
                        TileFailure(image_id=image.id, filename=image.filename, reason=str(exc))
                    )
                finally:
                    if on_progress is not None:
                        on_progress(done, total)

        return report


# ───────────── module-private helpers ─────────────


def _ensure_target(
    datasets: Datasets,
    source_name: str,
    into: str | None,
    drop_classes: frozenset[str],
) -> tuple[str, str, bool]:
    """Resolve/create the target dataset for augment/tile. Returns (id, name, same).

    ``same`` is ``True`` when the generated images land back in the source dataset.
    """
    source = datasets.get(source_name)
    if into is None or into == source_name:
        return source.id, source_name, True

    try:
        target = datasets.get(into)
        return target.id, into, False
    except NotFoundError:
        pass

    # Create the target, copying the source's class config (minus dropped classes)
    # so annotation saves (which reject class names not on the dataset) resolve.
    classes = [
        {
            "name": c.name,
            **({"type": c.type} if c.type else {}),
            **({"color": c.color} if c.color else {}),
        }
        for c in source.classes
        if c.name not in drop_classes
    ]
    ann_types = sorted({c.type for c in source.classes if c.type}) or None
    project = datasets.create(
        into,
        description=f"Augmented from '{source_name}'.",
        annotation_types=ann_types,
        classes=classes or None,
    )
    return project.id, into, False


def _list_params(
    dataset: str,
    directory_path: str | None,
    status: str | None,
    include_archived: bool,
    limit: int,
    offset: int,
    min_confidence_lt: float | None = None,
    split: str | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    """Query params for the images list endpoint, omitting unset filters.

    ``include_archived`` is sent only when ``True`` (the backend defaults it to
    ``False``), and ``directory_path`` / ``status`` / ``split`` / ``min_confidence_lt``
    are omitted when ``None`` so the backend applies no filter for them.
    """
    # Send the NAME when we have one. The route resolves it server-side,
    # which removes a whole round-trip from every name-addressed call - the SDK
    # used to fetch the dataset just to turn its name into the id it then sent.
    #
    # ONE key, `dataset`, for both forms. This used to pick between `dataset_id`
    # and `dataset_name`; the route now accepts neither, and rejected the request
    # outright with "Unknown query parameter(s)". That broke `images.list` - and
    # with it every method that resolves an image through it.
    params: dict[str, Any] = {"dataset": dataset, "limit": limit, "offset": offset}
    if directory_path is not None:
        params["directory_path"] = directory_path
    if status is not None:
        params["status"] = status
    if split is not None:
        params["split"] = split
    if include_archived:
        params["include_archived"] = "true"
    if min_confidence_lt is not None:
        params["min_confidence_lt"] = min_confidence_lt
    if filename is not None:
        params["filename"] = filename
    return params


def _infer_content_type(filename: str) -> str:
    """Map a filename suffix to a MIME type.

    Returns ``application/octet-stream`` when the suffix is unrecognised; the
    caller decides whether that's an error (it is, for image upload).
    """
    suffix = Path(filename).suffix.lower()
    return _CONTENT_TYPE_BY_SUFFIX.get(suffix, "application/octet-stream")


def _safe_image_dimensions(path: Path) -> tuple[int | None, int | None]:
    """Extract EXIF-corrected ``(width, height)`` via Pillow without raising.

    Returns ``(None, None)`` when Pillow is not installed, the file is not a
    decodable image, or the dimensions can't be read for any reason. The
    backend tolerates ``None`` and stores ``NULL`` in the column.

    EXIF orientation matters: rotated phone photos store width/height that
    don't match the displayed pixel layout. ImageOps.exif_transpose returns
    the rotated image; reading .size after that gives dims that match every
    downstream consumer (training data-prep, browser <img>, etc.).
    Without this, training pipelines that trust the DB-cached dims fail on
    a dimension mismatch mid-training on rotated images.
    """
    if not _PIL_AVAILABLE:
        return (None, None)
    try:
        with _PILImage.open(path) as img_file:
            # exif_transpose returns a new Image (or None on older PIL);
            # keep the variable typed as the abstract Image to satisfy mypy
            # (ImageFile narrows back to Image on the `or` fallback).
            oriented = _PIL_ImageOps.exif_transpose(img_file) or img_file
            width, height = oriented.size
            return (width, height)
    except (_UnidentifiedImageError, OSError):
        return (None, None)
