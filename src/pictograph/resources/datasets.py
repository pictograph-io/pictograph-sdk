"""Datasets resource - the full dataset lifecycle: list, get, create, update,
archive, delete, insights, near-duplicates, batch download, cold storage.

Datasets are unique by ``(organization, name)`` and the SDK strongly prefers
name-based addressing (agents pass strings users gave them; UUID indirection
is friction). Every single-dataset method therefore takes one ``dataset``
argument: its name, or its UUID (the API resolves either in the same path slot).

Vocabulary: :meth:`Datasets.archive` / :meth:`Datasets.unarchive` hide/show a
dataset in the default list (fully reversible, nothing deleted);
:meth:`Datasets.freeze` / :meth:`Datasets.restore` move the image bytes
between standard and cold storage. Two different verb pairs on purpose.

Class-list updates are **replace, not merge**. To add or remove a single
class, fetch the dataset, mutate the list locally, and pass the full result
to :meth:`Datasets.update`. This matches the editor's behavior and keeps the
LLM-facing surface simple ("set classes to X" beats "add Y, remove Z").

The :meth:`Datasets.download` method is the workhorse: it fetches a batch
of signed storage URLs in one call, then downloads images and annotations in
parallel via a worker pool. Failures don't halt the operation - the call
returns a structured :class:`DownloadReport` with a per-file failure list
so callers can retry the subset.
"""

from __future__ import annotations

import contextlib
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field

from pictograph._http.pagination import OffsetPager
from pictograph._http.streaming import DEFAULT_CHUNK_SIZE
from pictograph._path_safety import safe_download_name
from pictograph.exceptions import ApiError, PollTimeoutError, job_end_code
from pictograph.models.dataset import (
    Dataset,
    DatasetClass,
    DatasetDeleteResult,
    DatasetStorageStatus,
    DatasetStorageTransition,
    EmbeddingBackfill,
)
from pictograph.models.explore import PublicDatasetImport
from pictograph.models.insights import DatasetInsights
from pictograph.models.near_duplicates import NearDuplicatesResult
from pictograph.resources._base import Resource
from pictograph.resources.explore import dataset_path, split_ref

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pictograph._torch_dataset import PictographTorchDataset
    from pictograph.augment import Augmenter

_IMPORTS_PATH = "/api/v1/developer/public/imports/"
_DEFAULT_IMPORT_TIMEOUT = 1800.0  # 30 min - a large public dataset copies its rows for minutes


def public_import_ended_badly(job: PublicDatasetImport) -> ApiError:
    """The error a wait raises when a public-dataset import ends without completing.

    Shared by the sync and async resources so both raise the same thing: the job is
    the error's ``response`` and its ``code`` says how it ended (``job_failed`` /
    ``job_cancelled``).
    """
    return ApiError(
        f"Importing {job.dataset_name or 'the public dataset'} ended with status "
        f"'{job.status}': {job.error or 'no error message provided'}",
        response=job.model_dump(mode="json"),
        code=job_end_code(job.status),
    )


def import_body(name: str | None) -> dict[str, Any] | None:
    return {"name": name} if name is not None else None


DownloadMode = Literal["full", "images_only", "annotations_only"]
"""Three modes mapped 1:1 to the backend ``download`` endpoint."""

_API_PATH = "/api/v1/developer/datasets/"
_DEFAULT_DOWNLOAD_WORKERS = 10
_DEFAULT_DOWNLOAD_LIMIT = 10000  # backend caps at 10000


def _single_path(dataset: str, suffix: str = "") -> str:
    """The path of one dataset: ``/{dataset}`` takes its name or its UUID in the same
    segment (the API resolves either), so one argument addresses it (REFERENCE_LANGUAGE.md
    R1: never two ways to say one thing). Shared with the async twin."""
    return f"{_API_PATH}{quote(dataset, safe='')}{suffix}"


def _class_payload(
    classes: Sequence[DatasetClass | dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize a classes argument (models or raw dicts) to wire dicts."""
    return [
        c.model_dump(exclude_none=True) if isinstance(c, DatasetClass) else dict(c) for c in classes
    ]


class DownloadFailure(BaseModel):
    """Single failed download - collected and returned, not raised."""

    model_config = ConfigDict(frozen=True)

    filename: str
    kind: Literal["image", "annotation"]
    reason: str


class DownloadReport(BaseModel):
    """Result of a :meth:`Datasets.download` invocation.

    Inspect ``failures`` to retry a partial failure. ``success`` is ``True``
    only when every requested file landed.
    """

    dataset_id: str
    images_downloaded: int = 0
    annotations_downloaded: int = 0
    failures: list[DownloadFailure] = Field(default_factory=list)
    # True when the dataset's image listing hit the backend's per-call cap, so
    # the download covers only the first ``_DEFAULT_DOWNLOAD_LIMIT`` images and
    # the rest were never fetched (the endpoint exposes no offset/total, so this
    # is the only signal). ``success`` can still be True for what WAS fetched.
    truncated: bool = False

    @property
    def total_attempted(self) -> int:
        return self.images_downloaded + self.annotations_downloaded + len(self.failures)

    @property
    def success(self) -> bool:
        return not self.failures


class Datasets(Resource):
    """The full dataset lifecycle in the authenticated organization."""

    # ───────────── list / iter ─────────────

    def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        archived: bool = False,
    ) -> list[Dataset]:
        """Single-page list of datasets.

        For full enumeration prefer :meth:`iter`, which auto-pages.

        Args:
            limit: Maximum datasets to return in this page (backend cap: 1000).
            offset: Page offset for manual pagination.
            archived: List archived datasets instead of active ones.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if archived:
            params["archived"] = True
        response = self._transport.request("GET", _API_PATH, params=params)
        return self._parse_list(Dataset, response.get("data", []))

    def iter(
        self,
        *,
        page_size: int = 100,
        max_total: int | None = None,
        archived: bool = False,
    ) -> OffsetPager[Dataset]:
        """Auto-paging iterator over every dataset in the organization.

        Returns a :class:`OffsetPager` - iterate it, materialise via
        ``.all()``, or peek with ``.first()``. Stops on the server-computed
        ``pagination.has_more`` flag.

        Args:
            page_size: Items fetched per backend round-trip.
            max_total: Stop after this many items, even mid-page. ``None``
                yields every available item.
            archived: Iterate archived datasets instead of active ones.
        """

        def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            params: dict[str, Any] = {"offset": offset, "limit": limit}
            if archived:
                params["archived"] = True
            return cast(
                "Mapping[str, Any]",
                self._transport.request("GET", _API_PATH, params=params),
            )

        return OffsetPager(
            fetch,
            items_key="data",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(Dataset, raw),
        )

    # ───────────── get (by name / by id) ─────────────

    def get(
        self,
        dataset: str,
        *,
        include_images: bool = False,
        images_limit: int = 1000,
        images_offset: int = 0,
    ) -> Dataset:
        """Fetch a dataset by name.

        Args:
            dataset: The dataset's name (its UUID also works).
            include_images: When ``True``, the returned dataset includes its
                first ``images_limit`` images. Use :meth:`Datasets.download`
                for bulk image enumeration.
            images_limit: How many images to include (backend cap: 10000).
            images_offset: Paginate the embedded image list.
        """
        params: dict[str, Any] = {}
        if include_images:
            params["include_images"] = "true"
            params["images_limit"] = images_limit
            params["images_offset"] = images_offset
        response = self._transport.request("GET", _single_path(dataset), params=params or None)
        return self._parse(Dataset, response["data"])

    # ───────────── create / update / delete ─────────────

    def create(
        self,
        name: str,
        *,
        readme: str | None = None,
        description: str | None = None,
        annotation_types: Sequence[str] | None = None,
        classes: Sequence[DatasetClass | dict[str, Any]] | None = None,
    ) -> Dataset:
        """Create a new dataset + initial class config. Member+ API key.

        Args:
            name: Dataset name. Must be unique within the org.
            description: Optional human-readable description.
            annotation_types: Allowed types on the dataset. Defaults to
                ``["bbox"]``. Allowed values: ``bbox``/``box``, ``polygon``,
                ``polyline``, ``keypoint``.
            classes: Class definitions. Accepts :class:`DatasetClass`
                instances or raw dicts (validated server-side).

        Returns:
            The newly created :class:`Dataset` with its initial config.

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
        response = self._transport.request("POST", _API_PATH, json=body)
        return self._parse(Dataset, response["data"])

    def update(
        self,
        dataset: str,
        *,
        name: str | None = None,
        readme: str | None = None,
        description: str | None = None,
        annotation_types: Sequence[str] | None = None,
        classes: Sequence[DatasetClass | dict[str, Any]] | None = None,
        classes_base: Sequence[DatasetClass | dict[str, Any]] | None = None,
        visibility: Literal["public", "private"] | None = None,
        license: str | None = None,
        license_custom_text: str | None = None,
        hf_publish_enabled: bool | None = None,
    ) -> Dataset:
        """Update a dataset's metadata, types, classes, visibility or license (partial). Member+.

        Args:
            dataset: The dataset's name (its UUID also works).
            name: Rename the dataset (409 on collision).
            readme: Replace the dataset card (markdown). The UI renders this.
            description: DEPRECATED - superseded by ``readme``.
            annotation_types: Replace the allowed annotation types.
            classes: The FULL class list you want. Without ``classes_base`` it
                REPLACES the stored list.
            classes_base: The class list you READ and edited into ``classes``. With
                it, the edit (classes added, removed or changed) is applied to the list
                stored NOW, so a class another writer added since your read survives
                instead of being erased. Read-modify-write with both::

                    ds = client.datasets.get("road-signs")
                    new = [*ds.classes, DatasetClass(name="bus", type="box")]
                    client.datasets.update("road-signs", classes=new, classes_base=ds.classes)

            visibility: ``"public"`` lists it on Explore (and, with
                ``hf_publish_enabled``, mirrors it to Hugging Face); ``"private"``
                takes it down from both. Owner / admin.
            license: A license catalog id, e.g. ``"mit"`` or ``"cc-by-4.0"``
                (``"custom"`` with ``license_custom_text``).
            license_custom_text: The terms of a ``"custom"`` license.
            hf_publish_enabled: Mirror it to Pictograph's Hugging Face organization
                while it is public (on by default). Owner / admin.

        Raises:
            ValueError: No update field was provided, or ``classes_base`` without
                ``classes``.
            NotFoundError: The dataset doesn't exist in your org.
            ConflictError: ``name`` collides with an existing dataset.
        """
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if readme is not None:
            body["readme"] = readme
        if description is not None:
            body["description"] = description
        if annotation_types is not None:
            body["annotation_types"] = list(annotation_types)
        if classes is not None:
            body["classes"] = _class_payload(classes)
        if classes_base is not None:
            if classes is None:
                raise ValueError("classes_base is the list `classes` edits - pass classes too.")
            body["classes_base"] = _class_payload(classes_base)
        if visibility is not None:
            body["visibility"] = visibility
        if license is not None:
            body["license"] = license
        if license_custom_text is not None:
            body["license_custom_text"] = license_custom_text
        if hf_publish_enabled is not None:
            body["hf_publish_enabled"] = hf_publish_enabled
        if not body:
            raise ValueError(
                "Nothing to update - pass at least one of name / readme / description / "
                "annotation_types / classes / visibility / license / hf_publish_enabled."
            )
        response = self._transport.request("PATCH", _single_path(dataset), json=body)
        return self._parse(Dataset, response["data"])

    def delete(
        self,
        dataset: str,
    ) -> DatasetDeleteResult:
        """Delete a dataset, its images and their files, for good. Admin role.

        Files an imported copy of the dataset still uses are kept (``files_kept``).
        """
        response = self._transport.request("DELETE", _single_path(dataset))
        return self._parse(DatasetDeleteResult, response["data"])

    # ───────────── archive / unarchive ─────────────

    def archive(
        self,
        dataset: str,
    ) -> Dataset:
        """Archive a dataset - hide it from the default list without deleting
        anything. Fully reversible via :meth:`unarchive`. Admin+; idempotent.
        A PUBLIC dataset must be unpublished from Explore first (400).
        """
        response = self._transport.request("POST", _single_path(dataset, "/archive"))
        return self._parse(Dataset, response["data"])

    def unarchive(
        self,
        dataset: str,
    ) -> Dataset:
        """Bring an archived dataset back into the default list. Admin+;
        idempotent (a no-op if the dataset isn't archived)."""
        response = self._transport.request("POST", _single_path(dataset, "/unarchive"))
        return self._parse(Dataset, response["data"])

    # ───────────── insights / near-duplicates ─────────────

    def insights(
        self,
        dataset: str,
    ) -> DatasetInsights:
        """Dataset Health / Insights (by name or ``dataset``).

        Returns headline totals, labeling-stage counts, per-class instance +
        image counts (class balance), per-annotation-type totals, an
        annotations-per-image density histogram, and image-dimension insights.
        Aggregated server-side over the denormalized columns (never scans
        annotations), so it's a single fast call even for 100k+ image datasets.
        Non-archived images only.

        Example:
            >>> health = client.datasets.insights("road-signs")
            >>> health.total_annotations
            48213
            >>> sorted(health.class_annotation_counts.items(), key=lambda kv: -kv[1])[:3]
            [('car', 30112), ('sign', 9800), ('person', 4021)]
        """
        response = self._transport.request("GET", _single_path(dataset, "/insights"))
        return self._parse(DatasetInsights, response["data"])

    def near_duplicates(
        self,
        dataset: str,
        *,
        threshold: float | None = None,
        sample: int | None = None,
        neighbors: int | None = None,
        max_pairs: int | None = None,
        directory: str | None = None,
    ) -> NearDuplicatesResult:
        """Find near-duplicate images in a dataset - data curation.

        Groups visually near-duplicate images (SigLIP2 embedding cosine
        similarity >= ``threshold``) into clusters so you can keep one per
        cluster and archive the redundant rest - cutting annotation volume +
        dataset bloat before labeling. Runs one HNSW k-NN probe per source
        image over the embedding sidecar; expensive + on-demand. Every bound is
        clamped server-side and the result reports the analyzed sample + cap
        flags (no silent caps). Non-archived images only.

        Args:
            dataset: The dataset's name (its UUID also works).
            threshold: Minimum cosine similarity for a near-duplicate
                (0.5-0.9999, default 0.92). Higher = stricter (near-identical).
            sample: Max source images to scan (default 1000, cap 2000).
            neighbors: Max near-duplicate neighbours per source (default 10).
            max_pairs: Max near-duplicate edges returned (default 2000).
            directory: Scope the scan to one virtual directory (e.g. ``"/train"``);
                ``None`` (default) scans the whole dataset.

        Example:
            >>> dup = client.datasets.near_duplicates("road-signs")
            >>> dup.group_count, dup.redundant_count
            (18, 142)
            >>> # archive every image except the first of each cluster
            >>> redundant = [m.id for g in dup.groups for m in g.members[1:]]
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
        if directory is not None:
            params["directory"] = directory
        response = self._transport.request(
            "GET", _single_path(dataset, "/duplicates"), params=params or None
        )
        return self._parse(NearDuplicatesResult, response["data"])

    def as_pytorch(
        self,
        dataset: str,
        *,
        root: str | Path | None = None,
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[dict[str, Any]], Any] | None = None,
        class_to_idx: dict[str, int] | None = None,
        download: bool = True,
        images_limit: int = 10000,
        augment: Augmenter | None = None,
    ) -> PictographTorchDataset:
        """Adapt a dataset into a map-style :class:`torch.utils.data.Dataset`.

        Synchronous-only: there is no ``AsyncClient`` twin. It returns a torch
        ``Dataset`` for a synchronous training loop, and the ``DataLoader`` does
        its own worker fan-out, so an async variant would add nothing.

        Requires the ``torch`` extra (``pip install 'pictograph[torch]'``). The
        returned object plugs straight into ``torch.utils.data.DataLoader`` and
        yields ``(image, target)`` pairs - ``image`` a ``PIL.Image`` (or
        ``transform``'s output) and ``target`` a torchvision-style detection dict
        (``boxes`` xyxy, integer ``labels``, ``area``, ``iscrowd``, ``image_id``,
        raw ``annotations``). Images download lazily into ``root`` (a temp dir by
        default) on first access and are cached there.

        Args:
            dataset: Dataset dataset (case-sensitive, unique within the org).
            root: Local cache dir for downloaded images (created if missing).
            transform: Applied to the ``PIL.Image`` (e.g. a torchvision transform).
            target_transform: Applied to the target dict.
            class_to_idx: Class name -> integer label. Defaults to the dataset's
                classes in their order, which is how a model Pictograph trains numbers them.
            download: Download images on access (set ``False`` if ``root`` is
                already populated).
            images_limit: Cap on images pulled into the dataset (backend max 10000).
            augment: Optional :class:`pictograph.augment.Augmenter` for on-the-fly
                augmentation - each item is a freshly-augmented variant with the
                target boxes remapped to match.
        """
        try:
            from pictograph._torch_dataset import PictographTorchDataset as _Ds
        except ImportError as exc:  # pragma: no cover - exercised only without torch
            raise ImportError(
                "Datasets.as_pytorch requires the 'torch' extra: pip install 'pictograph[torch]'"
            ) from exc

        from pictograph.resources.annotations import Annotations
        from pictograph.resources.images import Images

        found = self.get(dataset, include_images=True, images_limit=images_limit)
        images = found.images or []
        if class_to_idx is None:
            # The dataset's class order, as every training pipeline numbers its classes
            # (an alphabetical default gave a class another index than a trained model's).
            ordered = list(dict.fromkeys(cls.name for cls in found.classes))
            class_to_idx = {cls_name: idx for idx, cls_name in enumerate(ordered)}

        return _Ds(
            dataset_name=found.name,
            images=images,
            image_resource=Images(self._transport),
            annotation_resource=Annotations(self._transport),
            class_to_idx=class_to_idx,
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
            augment=augment,
        )

    # ───────────── batch download ─────────────

    def download(
        self,
        dataset: str,
        *,
        output_path: str | Path | None = None,
        mode: DownloadMode = "full",
        status_filter: str | None = None,
        max_workers: int = _DEFAULT_DOWNLOAD_WORKERS,
        progress: Callable[[int, int, str | None], None] | None = None,
    ) -> DownloadReport:
        """Download a dataset's images and/or annotations to a local directory.

        Fetches a batch of signed storage URLs in one backend call, then
        downloads in parallel via a thread pool. Each file transfer is an
        unauthenticated request to the storage host, not to the API.

        Args:
            dataset: The dataset's name (its UUID also works).
            output_path: Local directory to write into. Created if missing.
                Image files land at ``output_path/<filename>``; annotation
                JSON files at ``output_path/<filename>.json``.
            mode: ``"full"`` (images + annotations), ``"images_only"``,
                or ``"annotations_only"``.
            status_filter: Restrict to images with this status
                (``"complete"`` / ``"in_progress"`` / ``"new"``).
            max_workers: Concurrent download threads. Defaults to 10.
            progress: Optional ``(completed, total, filename)`` callback,
                fired after each file finishes (success or failure).

        Returns:
            A :class:`DownloadReport`. Inspect ``.failures`` for partial
            failures; the call does not raise on individual file errors.

        Raises:
            ValidationError / NotFoundError: From the initial batch-URL
                fetch (e.g., dataset doesn't exist).
        """
        if output_path is None:
            raise ValueError("output_dir is required")
        out = Path(output_path).expanduser()
        out.mkdir(parents=True, exist_ok=True)

        params: dict[str, Any] = {"mode": mode, "limit": _DEFAULT_DOWNLOAD_LIMIT}
        if status_filter is not None:
            params["status_filter"] = status_filter
        listing = self._transport.request("GET", _single_path(dataset, "/download"), params=params)[
            "data"
        ]

        items: list[dict[str, Any]] = listing.get("items", [])
        report = DownloadReport(dataset_id=listing.get("id", ""))
        # The backend caps the image listing at _DEFAULT_DOWNLOAD_LIMIT with no
        # offset/total field, so a full page means the dataset is (very likely)
        # larger than what we fetched. Surface it loudly instead of silently
        # returning a partial dataset with success=True.
        if len(items) >= _DEFAULT_DOWNLOAD_LIMIT:
            report.truncated = True
            warnings.warn(
                f"Dataset '{dataset}' has at least "
                f"{_DEFAULT_DOWNLOAD_LIMIT} images; download covers only the "
                f"first {_DEFAULT_DOWNLOAD_LIMIT}. Check report.truncated; the "
                "remaining images were not fetched.",
                stacklevel=2,
            )
        if not items:
            return report

        # Build the list of work units - one per image, plus optionally one per
        # annotation. Order matters only for predictable progress reporting.
        @dataclass(frozen=True)
        class _Task:
            kind: Literal["image", "annotation"]
            url: str
            dest: Path
            filename: str

        tasks: list[_Task] = []
        for item in items:
            # The filename is server DATA, not a path component. Left raw, an
            # absolute value discards output_dir entirely (Path("/tmp/dl") /
            # "/etc/cron.d/pwn" == "/etc/cron.d/pwn") and "../" walks out of it.
            # Only the base directory is created here, so a single component is
            # all that could ever be written anyway.
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

        # Shared external httpx.Client for the signed-URL image downloads (no
        # auth, with connection pooling). Annotation downloads route through the
        # SDK's transport (which adds X-API-Key automatically).
        #
        # HTTP/1.1 on purpose: this single client is shared across a
        # ThreadPoolExecutor below, and httpx+h2 is NOT thread-safe on a shared
        # client (the h2 connection's stream dict mutates under iteration →
        # ``RuntimeError: dictionary changed size during iteration``). The signed
        # storage URLs all resolve to one host, so an http2 client would
        # multiplex the concurrent streams over one connection and trip exactly
        # that race. This is the same reason the SDK transport is http2=False
        # (see _http/transport.py::_build_client). Keep-alive still pools.
        with httpx.Client(
            http2=False,
            timeout=httpx.Timeout(self._transport._config.timeout, read=300.0),
            limits=httpx.Limits(max_connections=max_workers, max_keepalive_connections=max_workers),
        ) as gcs_client:

            def run_task(task: _Task) -> tuple[_Task, str | None]:
                try:
                    if task.kind == "image":
                        _stream_to_file(gcs_client, task.url, task.dest)
                    else:
                        _fetch_annotation_to_file(self._transport, task.url, task.dest)
                    return task, None
                except (httpx.HTTPError, ApiError, OSError) as exc:
                    # ``httpx.HTTPError`` is the union of RequestError (transport
                    # failures) and HTTPStatusError (4xx/5xx from storage). Both must
                    # land in ``failures``, never propagate from the worker pool.
                    return task, str(exc)

            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(run_task, t) for t in tasks]
                for future in as_completed(futures):
                    task, err = future.result()
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

    # ── Public datasets ───────────────────────────────────────────────

    def import_public(
        self,
        ref: str,
        *,
        organization: str | None = None,
        name: str | None = None,
        wait: bool = True,
        poll_interval: float = 3.0,
        timeout: float = _DEFAULT_IMPORT_TIMEOUT,
    ) -> PublicDatasetImport:
        """Import (fork) a public dataset into your organization as a private copy.

        The app's "Import Dataset": the copy gets the dataset's classes, its license
        and upstream attribution, and its images - which are referenced, not copied,
        so even a large dataset is ready in minutes - and it stays yours if the
        source is later unpublished. Find datasets with :meth:`Explore.datasets
        <pictograph.resources.explore.Explore.datasets>`.

        Args:
            ref: ``"<organization>/<dataset>"`` - the publishing organization's slug
                and the dataset's slug or display name (a row's ``ref`` from
                ``client.explore.datasets()``). Or the bare name with
                ``organization=``.
            organization: The publishing organization's slug, when ``ref`` is a bare
                name.
            name: The copy's name. Defaults to the source's; a name your organization
                already uses is suffixed ``"(imported)"`` - read ``dataset_name`` for
                the final one.
            wait: ``True`` (default) waits until the copy is ready; ``False`` returns
                at once with ``status="pending"`` (follow it with
                :meth:`wait_for_public_import`).
            poll_interval: Seconds between checks when waiting.
            timeout: Maximum seconds to wait (30 minutes by default).

        Returns:
            The :class:`~pictograph.models.explore.PublicDatasetImport`: ``completed``
            when waited on, ``pending`` with ``wait=False``.

        Raises:
            ValueError: ``ref`` names no organization.
            NotFoundError: No such public dataset (a private one reads the same).
            PaymentRequiredError: The copy would take your plan past its image limit
                (``image_limit_exceeded``); nothing was created.
            ForbiddenError: Your API key's role cannot create datasets (member+).
            ValidationError: ``name`` cannot be a dataset name (it has a ``/``).
            ApiError: The import ended ``failed`` or ``cancelled`` while waiting
                (``code`` is ``job_failed`` / ``job_cancelled``).
            PollTimeoutError: ``timeout`` passed; the import keeps running.
        """
        org, source = split_ref(ref, organization)
        response = self._transport.request(
            "POST", f"{dataset_path(org, source)}/import", json=import_body(name)
        )
        job = self._parse(PublicDatasetImport, response["data"])
        if not wait:
            return job
        return self.wait_for_public_import(
            job.import_id, poll_interval=poll_interval, timeout=timeout
        )

    def get_public_import(self, job: str) -> PublicDatasetImport:
        """A public-dataset import's progress and outcome (see :meth:`import_public`).

        Args:
            job: The import's id - ``job`` of what :meth:`import_public`
                returned.

        Returns:
            The :class:`~pictograph.models.explore.PublicDatasetImport`, at once.

        Raises:
            NotFoundError: No such import in your organization.
        """
        response = self._transport.request("GET", f"{_IMPORTS_PATH}{quote(job, safe='')}")
        return self._parse(PublicDatasetImport, response["data"])

    def wait_for_public_import(
        self,
        job: str,
        *,
        poll_interval: float = 3.0,
        timeout: float = _DEFAULT_IMPORT_TIMEOUT,
        sleep: Callable[[float], None] | None = None,
    ) -> PublicDatasetImport:
        """Block until a public-dataset import finishes.

        Args:
            job: The import's id - ``job`` of what
                ``import_public(..., wait=False)`` returned.
            poll_interval: Seconds between checks.
            timeout: Maximum seconds to wait (30 minutes by default).
            sleep: Override ``time.sleep`` (tests).

        Returns:
            The ``completed`` import - its ``dataset_name`` is ready to use.

        Raises:
            ApiError: It ended ``failed`` or ``cancelled`` (``code`` says which; the
                import is the error's ``response``).
            PollTimeoutError: ``timeout`` passed; the import keeps running.
        """
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else time.sleep
        deadline = time.monotonic() + timeout
        while True:
            current = self.get_public_import(job)
            if current.status == "completed":
                return current
            if current.done:
                raise public_import_ended_badly(current)
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Import {job} did not finish within {timeout:.0f}s "
                    f"({current.image_count} of {current.total_images} images so far). It is "
                    "still running - wait again with client.datasets.wait_for_public_import()."
                )
            sleep_fn(poll_interval)

    # ── Cold storage ──────────────────────────────────────────────────

    def backfill_embeddings(
        self,
        dataset: str,
        *,
        limit: int = 500,
    ) -> EmbeddingBackfill:
        """Queue SigLIP embeddings + auto-tags for images that have none. Member+.

        Images registered with ``embed=False`` (:meth:`Images.bulk_upload`) - or whose
        embedding pass never finished - have no vector, so similarity search, the
        embedding map and the auto-tags skip them. This hands up to ``limit`` of them
        (at most 500) to the embedding service in ONE call, which works through them
        a batch at a time, so a large import embeds at a pace the database absorbs.
        ``limit=0`` only counts what is missing.

        Call again once ``missing`` has dropped by about what was queued: each call
        re-reads the images still missing, so calling before the service is done
        queues the same images twice.
        """
        if not 0 <= limit <= 500:
            raise ValueError("limit must be between 0 and 500")
        response = self._transport.request(
            "POST", _single_path(dataset, "/embeddings/backfill"), json={"limit": limit}
        )
        return self._parse(EmbeddingBackfill, response["data"])

    def storage_status(
        self,
        dataset: str,
    ) -> DatasetStorageStatus:
        """Cold-storage state (+ restore price quote while cold) for a dataset."""
        response = self._transport.request("GET", _single_path(dataset, "/storage"))
        return self._parse(DatasetStorageStatus, response["data"])

    def freeze(
        self,
        dataset: str,
    ) -> DatasetStorageTransition:
        """Move a dataset to cold storage. Free; background job.

        While cold, images count at a discounted rate toward the org quota;
        uploads, exports, and auto-annotation are paused until
        :meth:`restore`. Requires an admin/owner API key; public datasets and
        datasets with forks are rejected with 409.
        """
        response = self._transport.request("POST", _single_path(dataset, "/storage/freeze"))
        return self._parse(DatasetStorageTransition, response["data"])

    def restore(
        self,
        dataset: str,
    ) -> DatasetStorageTransition:
        """Restore a cold dataset to standard storage (charges compute credits).

        The price comes from :meth:`storage_status` (``restore_estimate``) and
        is billed on the transition's SUCCESS - a restore that fails or is
        abandoned costs nothing. The returned ``quoted_micro_usd`` is that
        pending amount. The charge is idempotent per frozen generation, so
        retrying a failed restore never double-charges.
        """
        response = self._transport.request("POST", _single_path(dataset, "/storage/restore"))
        return self._parse(DatasetStorageTransition, response["data"])

    def wait_for_storage(
        self,
        dataset: str,
        *,
        timeout: float = 600.0,
        poll_interval: float = 3.0,
        sleep: Callable[[float], None] | None = None,
    ) -> DatasetStorageStatus:
        """Block until a freeze/restore transition finishes (state ``idle``).

        Args:
            dataset: The dataset's name (its UUID also works).
            timeout: Maximum seconds to wait.
            poll_interval: Seconds between status checks.
            sleep: Override the sleep function (testing hook).

        Raises:
            PollTimeoutError: ``timeout`` elapsed before the transition finished.
        """
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else time.sleep
        deadline = time.monotonic() + timeout
        while True:
            status = self.storage_status(dataset)
            if status.storage_state == "idle":
                return status
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Dataset {dataset} storage transition did not "
                    f"finish within {timeout:.0f}s (state={status.storage_state!r})"
                )
            sleep_fn(poll_interval)


def _stream_to_file(
    client: httpx.Client,
    url: str,
    dest: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Stream a signed storage URL to disk in chunks.

    Bytes land in a sibling ``.part`` file and are renamed atomically on
    success, so a mid-stream error never leaves a truncated file at ``dest``
    (mirrors images/exports/models download). The per-file failure is recorded
    by the caller; a retry must not find a half-written file masquerading as
    complete.
    """
    tmp = dest.with_name(dest.name + ".part")
    try:
        with client.stream("GET", url) as response:
            if response.status_code >= 300:
                response.read()
                raise httpx.HTTPStatusError(
                    f"Image download failed: HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
            with tmp.open("wb") as fh:
                for chunk in response.iter_bytes(chunk_size=chunk_size):
                    fh.write(chunk)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    tmp.replace(dest)


def _fetch_annotation_to_file(
    transport: Any,  # _http.transport.Transport - typed loosely to avoid runtime import cycle
    url: str,
    dest: Path,
) -> None:
    """Fetch an annotation JSON file via the authenticated SDK transport.

    The annotation download endpoint returns JSON, not bytes, so we pretty-
    print it on disk for human-readable inspection. Bit-for-bit fidelity is
    preserved (json.dumps is round-trip stable for the data we emit).
    """
    import json

    payload = transport.request("GET", url)
    # Atomic write: a fetch/serialize error must not leave a truncated JSON
    # file at ``dest`` (mirrors _stream_to_file).
    tmp = dest.with_name(dest.name + ".part")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    tmp.replace(dest)
