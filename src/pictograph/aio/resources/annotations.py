"""Async Annotations resource - read, save, delete annotations for an image.

Async twin of :class:`pictograph.resources.annotations.Annotations`. The result
dataclasses (:class:`SaveResult`, :class:`DeleteResult`, :class:`BulkSaveResult`)
and the discriminated-union adapter are reused verbatim from the sync module -
they are pure data, no transport.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from pictograph.aio.resources import _resolve
from pictograph.exceptions import PictographError
from pictograph.resources._base import AsyncResource
from pictograph.resources.annotations import (
    _ANNOTATION_LIST_ADAPTER,
    _BULK_SAVE_CAP,
    AnnotationImportFailure,
    AnnotationImportReport,
    BulkSaveFailure,
    BulkSaveResult,
    DeleteClassResult,
    DeleteResult,
    MergeClassResult,
    RenameClassResult,
    SaveResult,
    _image_route,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from typing import Any

    from pictograph.aio.resources.datasets import AsyncDatasets
    from pictograph.models.annotation import Annotation

_API_PATH = "/api/v1/developer/annotations"
_DEFAULT_CONCURRENCY = 8

__all__ = [
    "AnnotationImportFailure",
    "AnnotationImportReport",
    "AsyncAnnotations",
    "BulkSaveResult",
    "DeleteResult",
    "SaveResult",
]


class AsyncAnnotations(AsyncResource):
    """Read, save, and delete annotations for individual images (async)."""

    async def get(
        self,
        dataset_name: str,
        image: str,
        *,
        directory_path: str | None = None,
    ) -> list[Annotation]:
        """Fetch the annotations attached to an image (typed discriminated union).

        A user knows an image by its FILENAME inside a dataset, which is the pair
        the grid shows - so that is what this takes. An image id is still accepted
        in place of the filename (detected by shape), and ``directory_path``
        disambiguates when the same filename lives in two directories.

        An image with zero annotations returns an empty list - never raises
        ``NotFoundError`` for that case (only for "no such image").
        """
        segment = await _resolve.image_segment(
            self._transport, image, directory_path=directory_path
        )
        response = await self._transport.request("GET", _image_route(dataset_name, segment))
        raw = response.get("annotations", [])
        return _ANNOTATION_LIST_ADAPTER.validate_python(raw)

    async def save(
        self,
        dataset_name: str,
        image: str,
        annotations: Sequence[Annotation],
        *,
        directory_path: str | None = None,
    ) -> SaveResult:
        """Replace an image's annotations with the given list (full replace).

        Addressed by ``(dataset name, filename)`` like :meth:`get`. Pass an empty
        list to clear (equivalent to :meth:`delete`).

        Raises:
            ValidationError: Backend rejected the payload (e.g. an unknown class).
            ForbiddenError: API key role lacks write permission.
            NotFoundError: No such dataset, or no such image in it.
        """
        segment = await _resolve.image_segment(
            self._transport, image, directory_path=directory_path
        )
        # The image is identified ENTIRELY by the URL. The request model declares
        # `annotations` and nothing else (extra="forbid"), so an `image_id`
        # alongside it - which this used to send - is a 422, not a harmless echo.
        body = {
            "annotations": [ann.model_dump(mode="json", exclude_none=True) for ann in annotations],
        }
        response = await self._transport.request(
            "POST",
            _image_route(dataset_name, segment),
            json=body,
        )
        return SaveResult(
            image_id=response["image_id"],
            previous_count=response["previous_count"],
            new_count=response["new_count"],
            status=response["status"],
        )

    async def bulk_save(
        self,
        saves: Mapping[str, Sequence[Annotation]],
    ) -> BulkSaveResult:
        """Save annotations for up to 200 images in one server-side call.

        Each entry is a *full* replacement of that image's annotations. An image
        id that doesn't resolve in your org lands in
        :attr:`BulkSaveResult.failed` rather than failing the whole batch.

        Raises:
            ValidationError: Backend rejected the payload (bad shape, or >200 entries).
            ForbiddenError: API key role lacks write permission.
        """
        body = {
            "saves": [
                {
                    "image_id": image_id,
                    "annotations": [
                        ann.model_dump(mode="json", exclude_none=True) for ann in annotations
                    ],
                }
                for image_id, annotations in saves.items()
            ]
        }
        response = await self._transport.request("POST", f"{_API_PATH}/bulk", json=body)
        saved = [
            SaveResult(
                image_id=s["image_id"],
                previous_count=s["previous_count"],
                new_count=s["new_count"],
                status=s["status"],
            )
            for s in response.get("saved", [])
        ]
        failed = [
            BulkSaveFailure(image_id=f["image_id"], error=str(f.get("error", "")))
            for f in response.get("failed", [])
        ]
        return BulkSaveResult(saved=saved, failed=failed)

    async def delete(
        self,
        dataset_name: str,
        image: str,
        *,
        directory_path: str | None = None,
    ) -> DeleteResult:
        """Remove all annotations from an image (requires admin/owner role).

        Addressed by ``(dataset name, filename)`` like :meth:`get`.
        """
        segment = await _resolve.image_segment(
            self._transport, image, directory_path=directory_path
        )
        response = await self._transport.request("DELETE", _image_route(dataset_name, segment))
        return DeleteResult(
            image_id=response.get("image_id", ""),
            deleted_count=response.get("deleted_count", 0),
        )

    async def rename_class(
        self, dataset_name: str, old_name: str, new_name: str
    ) -> RenameClassResult:
        """Rename an annotation class across a WHOLE dataset in one call.

        Async twin of :meth:`pictograph.resources.annotations.Annotations.rename_class`
        - renames the ontology entry AND every stored annotation server-side.
        Member role or higher.
        """
        response = await self._transport.request(
            "POST",
            f"{_API_PATH}/rename-class",
            json={"dataset": dataset_name, "old_name": old_name, "new_name": new_name},
        )
        data = response.get("data", response)
        return RenameClassResult(
            dataset_id=data.get("dataset_id", ""),
            old_name=data.get("old_name", old_name),
            new_name=data.get("new_name", new_name),
            images_updated=int(data.get("images_updated", 0)),
            annotations_updated=int(data.get("annotations_updated", 0)),
            config_updated=bool(data.get("config_updated", False)),
        )

    async def merge_class(
        self, dataset_name: str, source_name: str, target_name: str
    ) -> MergeClassResult:
        """Merge one annotation class into another across a WHOLE dataset.

        Async twin of :meth:`pictograph.resources.annotations.Annotations.merge_class`
        - reassigns every ``source_name`` annotation to ``target_name`` and drops
        the source class from the ontology. Member role or higher.
        """
        response = await self._transport.request(
            "POST",
            f"{_API_PATH}/merge-class",
            json={
                "dataset": dataset_name,
                "source_name": source_name,
                "target_name": target_name,
            },
        )
        data = response.get("data", response)
        return MergeClassResult(
            dataset_id=data.get("dataset_id", ""),
            source_name=data.get("source_name", source_name),
            target_name=data.get("target_name", target_name),
            images_updated=int(data.get("images_updated", 0)),
            annotations_updated=int(data.get("annotations_updated", 0)),
            config_updated=bool(data.get("config_updated", False)),
        )

    async def delete_class(
        self,
        dataset_name: str,
        name: str,
        *,
        class_type: str | None = None,
        delete_annotations: bool = False,
    ) -> DeleteClassResult:
        """Delete a class from a dataset's ontology, optionally its annotations.

        Async twin of :meth:`pictograph.resources.annotations.Annotations.delete_class`.
        With ``delete_annotations=True`` every annotation of the class is stripped
        too; otherwise only the ontology entry is removed. Member role or higher.
        """
        response = await self._transport.request(
            "POST",
            f"{_API_PATH}/delete-class",
            json={
                "dataset": dataset_name,
                "name": name,
                "class_type": class_type,
                "delete_annotations": delete_annotations,
            },
        )
        data = response.get("data", response)
        return DeleteClassResult(
            dataset_id=data.get("dataset_id", ""),
            name=data.get("name", name),
            config_updated=bool(data.get("config_updated", False)),
            images_updated=int(data.get("images_updated", 0)),
            annotations_removed=int(data.get("annotations_removed", 0)),
        )

    # ───────────── import from an external format ─────────────

    async def import_coco(
        self,
        dataset_name: str,
        coco: dict[str, Any] | str | Path,
        *,
        create_missing_classes: bool = True,
        save_chunk: int = _BULK_SAVE_CAP,
        max_concurrency: int = _DEFAULT_CONCURRENCY,
    ) -> AnnotationImportReport:
        """Async twin of :meth:`pictograph.resources.annotations.Annotations.import_coco`.

        Same recipe - parse via :mod:`pictograph.formats`, create missing classes,
        match images by file name, chunked :meth:`bulk_save` - but the per-chunk
        saves run **concurrently** (bounded by ``max_concurrency``), a real speed-up
        when importing a large dataset (many 200-image chunks).

        Args:
            dataset_name: Destination project (must already exist and hold the images).
            coco: A parsed COCO dict, or a path / JSON string to one.
            create_missing_classes: Add referenced-but-undefined classes (default True).
            save_chunk: Images per :meth:`bulk_save` call (backend cap 200).
            max_concurrency: Max concurrent :meth:`bulk_save` chunks in flight.

        Raises:
            NotFoundError: ``dataset_name`` does not exist.
        """
        from pictograph.aio.resources.datasets import AsyncDatasets
        from pictograph.aio.resources.images import AsyncImages
        from pictograph.formats import from_coco

        imp = from_coco(coco)
        report = AnnotationImportReport(dataset_name=dataset_name)
        project = await AsyncDatasets(self._transport).get(dataset_name)
        id_by_filename = {
            img.filename: img.id async for img in AsyncImages(self._transport).iter(project.id)
        }
        await self._save_resolved(
            report,
            imp.annotations,
            id_by_filename=id_by_filename,
            create_missing_classes=create_missing_classes,
            save_chunk=save_chunk,
            max_concurrency=max_concurrency,
        )
        return report

    async def import_pascal_voc(
        self,
        dataset_name: str,
        xml_by_filename: Mapping[str, str],
        *,
        create_missing_classes: bool = True,
        save_chunk: int = _BULK_SAVE_CAP,
        max_concurrency: int = _DEFAULT_CONCURRENCY,
    ) -> AnnotationImportReport:
        """Async twin of
        :meth:`pictograph.resources.annotations.Annotations.import_pascal_voc`.

        ``xml_by_filename`` maps each image's file name to its Pascal VOC ``.xml``
        contents; the chunked bulk-saves run concurrently.
        """
        from pictograph.aio.resources.datasets import AsyncDatasets
        from pictograph.aio.resources.images import AsyncImages
        from pictograph.formats import from_pascal_voc

        annotations_by_filename: dict[str, list[Annotation]] = {}
        report = AnnotationImportReport(dataset_name=dataset_name)
        for filename, xml in xml_by_filename.items():
            try:
                parsed = from_pascal_voc(xml)
            except ValueError as e:
                # A single malformed XML file must not abort the whole batch import.
                report.failures.append(
                    AnnotationImportFailure(image_filename=filename, reason=str(e))
                )
                continue
            if parsed:
                annotations_by_filename[filename] = parsed
        project = await AsyncDatasets(self._transport).get(dataset_name)
        id_by_filename = {
            img.filename: img.id async for img in AsyncImages(self._transport).iter(project.id)
        }
        await self._save_resolved(
            report,
            annotations_by_filename,
            id_by_filename=id_by_filename,
            create_missing_classes=create_missing_classes,
            save_chunk=save_chunk,
            max_concurrency=max_concurrency,
        )
        return report

    async def import_yolo(
        self,
        dataset_name: str,
        labels: Mapping[str, str],
        class_names: Sequence[str],
        *,
        create_missing_classes: bool = True,
        save_chunk: int = _BULK_SAVE_CAP,
        max_concurrency: int = _DEFAULT_CONCURRENCY,
    ) -> AnnotationImportReport:
        """Async twin of :meth:`pictograph.resources.annotations.Annotations.import_yolo`."""
        from pictograph.aio.resources.datasets import AsyncDatasets
        from pictograph.aio.resources.images import AsyncImages
        from pictograph.formats import from_yolo

        project = await AsyncDatasets(self._transport).get(dataset_name)
        dims_by_filename: dict[str, tuple[str, int | None, int | None]] = {}
        async for img in AsyncImages(self._transport).iter(project.id):
            dims_by_filename[img.filename] = (img.id, img.width, img.height)

        annotations_by_filename: dict[str, list[Annotation]] = {}
        report = AnnotationImportReport(dataset_name=dataset_name)
        for filename, text in labels.items():
            entry = dims_by_filename.get(filename)
            if entry is None:
                report.unmatched_files.append(filename)
                continue
            _image_id, width, height = entry
            if not width or not height:
                report.failures.append(
                    AnnotationImportFailure(
                        image_filename=filename,
                        reason=(
                            "image has no stored width/height; cannot denormalize YOLO coordinates"
                        ),
                    )
                )
                continue
            parsed = from_yolo(text, class_names, width, height)
            if parsed:
                annotations_by_filename[filename] = parsed

        await self._save_resolved(
            report,
            annotations_by_filename,
            id_by_filename={fn: v[0] for fn, v in dims_by_filename.items()},
            create_missing_classes=create_missing_classes,
            save_chunk=save_chunk,
            max_concurrency=max_concurrency,
        )
        return report

    # ───────────── import internals ─────────────

    async def _save_resolved(
        self,
        report: AnnotationImportReport,
        annotations_by_filename: Mapping[str, Sequence[Annotation]],
        *,
        id_by_filename: Mapping[str, str],
        create_missing_classes: bool,
        save_chunk: int,
        max_concurrency: int,
    ) -> None:
        """Ensure classes exist, then bulk-save resolved images concurrently."""
        from pictograph.aio.resources.datasets import AsyncDatasets

        matched: dict[str, list[Annotation]] = {}
        for filename, anns in annotations_by_filename.items():
            image_id = id_by_filename.get(filename)
            if image_id is None:
                report.unmatched_files.append(filename)
                continue
            matched[image_id] = list(anns)
        report.images_matched = len(matched)
        if not matched:
            return

        if create_missing_classes:
            type_by_class: dict[str, str] = {}
            for anns in matched.values():
                for ann in anns:
                    type_by_class.setdefault(ann.name, ann.type)
            await _ensure_classes(
                AsyncDatasets(self._transport), report.dataset_name, type_by_class, report
            )

        items = list(matched.items())
        chunk_size = max(1, min(save_chunk, _BULK_SAVE_CAP))
        chunks = [
            dict(items[start : start + chunk_size]) for start in range(0, len(items), chunk_size)
        ]
        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def save_chunk_task(chunk: dict[str, list[Annotation]]) -> None:
            async with semaphore:
                try:
                    result = await self.bulk_save(chunk)
                except PictographError as e:
                    for image_id in chunk:
                        report.failures.append(
                            AnnotationImportFailure(image_filename=image_id, reason=str(e))
                        )
                    return
                report.images_saved += result.saved_count
                report.annotations_saved += sum(s.new_count for s in result.saved)
                for f in result.failed:
                    report.failures.append(
                        AnnotationImportFailure(image_filename=f.image_id, reason=f.error)
                    )

        # The report is mutated from each task, but asyncio is single-threaded and
        # cooperative - the mutations happen after each await, never truly in parallel.
        await asyncio.gather(*(save_chunk_task(chunk) for chunk in chunks))


async def _ensure_classes(
    datasets: AsyncDatasets,
    dataset_name: str,
    type_by_class: Mapping[str, str],
    report: AnnotationImportReport,
) -> None:
    """Add referenced-but-undefined classes (best-effort; failure → report entry)."""
    from pictograph.models.dataset import DatasetClass

    try:
        project = await datasets.get(dataset_name)
        existing = {c.name for c in project.classes}
        missing = [(n, t) for n, t in type_by_class.items() if n not in existing]
        if not missing:
            return
        new_classes: list[DatasetClass] = [*project.classes] + [
            DatasetClass(name=name, type=ann_type) for name, ann_type in missing
        ]
        await datasets.update(dataset_name, classes=new_classes)
    except PictographError as e:
        report.failures.append(
            AnnotationImportFailure(
                image_filename="<classes>",
                reason=f"could not create missing classes: {e}",
            )
        )
