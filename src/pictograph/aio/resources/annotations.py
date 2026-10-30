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
from pictograph.resources._files import ImageIndex, match_sources
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
        dataset: str,
        image: str,
        *,
        directory: str | None = None,
    ) -> list[Annotation]:
        """Fetch the annotations attached to an image (typed discriminated union).

        A user knows an image by its FILENAME inside a dataset, which is the pair
        the grid shows - so that is what this takes. The image is addressed by its
        path: a bare filename means the dataset ROOT, so an image in a folder needs
        ``directory`` (or ``image`` as ``directory/filename``). An image id is
        still accepted in place of the filename (detected by shape) and must belong
        to ``dataset``.

        An image with zero annotations returns an empty list - never raises
        ``NotFoundError`` for that case (only for "no such image").
        """
        segment = await _resolve.image_segment(
            self._transport, image, dataset=dataset, directory_path=directory
        )
        response = await self._transport.request("GET", _image_route(dataset, segment))
        raw = response.get("annotations", [])
        return _ANNOTATION_LIST_ADAPTER.validate_python(raw)

    async def save(
        self,
        dataset: str,
        image: str,
        annotations: Sequence[Annotation],
        *,
        directory: str | None = None,
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
            self._transport, image, dataset=dataset, directory_path=directory
        )
        # The image is identified ENTIRELY by the URL. The request model declares
        # `annotations` and nothing else (extra="forbid"), so an `image_id`
        # alongside it - which this used to send - is a 422, not a harmless echo.
        body = {
            "annotations": [ann.model_dump(mode="json", exclude_none=True) for ann in annotations],
        }
        response = await self._transport.request(
            "POST",
            _image_route(dataset, segment),
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
        dataset: str,
        image: str,
        *,
        directory: str | None = None,
    ) -> DeleteResult:
        """Remove all annotations from an image (requires admin/owner role).

        Addressed by ``(dataset name, filename)`` like :meth:`get`.
        """
        segment = await _resolve.image_segment(
            self._transport, image, dataset=dataset, directory_path=directory
        )
        response = await self._transport.request("DELETE", _image_route(dataset, segment))
        return DeleteResult(
            image_id=response.get("image_id", ""),
            deleted_count=response.get("deleted_count", 0),
        )

    async def rename_class(self, dataset: str, old_name: str, new_name: str) -> RenameClassResult:
        """Rename an annotation class across a WHOLE dataset in one call.

        Async twin of :meth:`pictograph.resources.annotations.Annotations.rename_class`
        - renames the ontology entry AND every stored annotation server-side.
        Member role or higher.
        """
        response = await self._transport.request(
            "POST",
            f"{_API_PATH}/rename-class",
            json={"dataset": dataset, "old_name": old_name, "new_name": new_name},
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
        self, dataset: str, source_name: str, target_name: str
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
                "dataset": dataset,
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
        dataset: str,
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
                "dataset": dataset,
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
        dataset: str,
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
            dataset: Destination project (must already exist and hold the images).
            coco: A parsed COCO dict, or a path / JSON string to one.
            create_missing_classes: Add referenced-but-undefined classes (default True).
            save_chunk: Images per :meth:`bulk_save` call (backend cap 200).
            max_concurrency: Max concurrent :meth:`bulk_save` chunks in flight.

        Raises:
            NotFoundError: ``dataset`` does not exist.
        """
        from pictograph.formats import from_coco

        imp = from_coco(coco)
        report = await self._import_by_filename(
            dataset,
            imp.annotations,
            create_missing_classes=create_missing_classes,
            save_chunk=save_chunk,
            max_concurrency=max_concurrency,
        )
        return report.model_copy(update={"unconverted": list(imp.unconverted)})

    async def import_pascal_voc(
        self,
        dataset: str,
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
        from pictograph.formats import from_pascal_voc

        annotations_by_filename: dict[str, list[Annotation]] = {}
        parse_failures: list[AnnotationImportFailure] = []
        for filename, xml in xml_by_filename.items():
            try:
                parsed = from_pascal_voc(xml)
            except ValueError as e:
                # A single malformed XML file must not abort the whole batch import.
                parse_failures.append(
                    AnnotationImportFailure(image_filename=filename, reason=str(e))
                )
                continue
            if parsed:
                annotations_by_filename[filename] = parsed
        report = await self._import_by_filename(
            dataset,
            annotations_by_filename,
            create_missing_classes=create_missing_classes,
            save_chunk=save_chunk,
            max_concurrency=max_concurrency,
        )
        report.failures.extend(parse_failures)
        return report

    async def import_yolo(
        self,
        dataset: str,
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

        project = await AsyncDatasets(self._transport).get(dataset)
        images = [img async for img in AsyncImages(self._transport).iter(project.id)]
        dims = {img.id: (img.width, img.height) for img in images}
        matched, unmatched, refused = match_sources(
            labels, ImageIndex((img.filename, img.id) for img in images)
        )
        report = _report(dataset, unmatched, refused)
        parsed_by_id: dict[str, tuple[str, list[Annotation]]] = {}
        for image_id, (filename, text) in matched.items():
            width, height = dims[image_id]
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
                parsed_by_id[image_id] = (filename, parsed)

        await self._save_matched(
            report,
            parsed_by_id,
            create_missing_classes=create_missing_classes,
            save_chunk=save_chunk,
            max_concurrency=max_concurrency,
        )
        return report

    # ───────────── import internals ─────────────

    async def _import_by_filename(
        self,
        dataset: str,
        annotations_by_filename: Mapping[str, Sequence[Annotation]],
        *,
        create_missing_classes: bool,
        save_chunk: int,
        max_concurrency: int,
    ) -> AnnotationImportReport:
        """Resolve file names to the dataset's images (see ``_files``), then save."""
        from pictograph.aio.resources.datasets import AsyncDatasets
        from pictograph.aio.resources.images import AsyncImages

        project = await AsyncDatasets(self._transport).get(dataset)
        index = ImageIndex(
            [(img.filename, img.id) async for img in AsyncImages(self._transport).iter(project.id)]
        )
        matched, unmatched, refused = match_sources(annotations_by_filename, index)
        report = _report(dataset, unmatched, refused)
        await self._save_matched(
            report,
            {image_id: (key, list(anns)) for image_id, (key, anns) in matched.items()},
            create_missing_classes=create_missing_classes,
            save_chunk=save_chunk,
            max_concurrency=max_concurrency,
        )
        return report

    async def _save_matched(
        self,
        report: AnnotationImportReport,
        matched: Mapping[str, tuple[str, list[Annotation]]],
        *,
        create_missing_classes: bool,
        save_chunk: int,
        max_concurrency: int,
    ) -> None:
        """Ensure classes exist, then bulk-save resolved images concurrently.

        A failure names the source file, never the image id.
        """
        from pictograph.aio.resources.datasets import AsyncDatasets

        report.images_matched = len(matched)
        if not matched:
            return

        if create_missing_classes:
            type_by_class: dict[str, str] = {}
            for _source, anns in matched.values():
                for ann in anns:
                    type_by_class.setdefault(ann.name, ann.type)
            await _ensure_classes(
                AsyncDatasets(self._transport), report.dataset_name, type_by_class, report
            )

        items = [(image_id, anns) for image_id, (_source, anns) in matched.items()]
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
                            AnnotationImportFailure(
                                image_filename=matched[image_id][0], reason=str(e)
                            )
                        )
                    return
                report.images_saved += result.saved_count
                report.annotations_saved += sum(s.new_count for s in result.saved)
                for f in result.failed:
                    source = matched[f.image_id][0] if f.image_id in matched else f.image_id
                    report.failures.append(
                        AnnotationImportFailure(image_filename=source, reason=f.error)
                    )

        # The report is mutated from each task, but asyncio is single-threaded and
        # cooperative - the mutations happen after each await, never truly in parallel.
        await asyncio.gather(*(save_chunk_task(chunk) for chunk in chunks))


def _report(
    dataset: str, unmatched: list[str], refused: list[tuple[str, str]]
) -> AnnotationImportReport:
    """A fresh report carrying the files no image (or no single image) matched."""
    return AnnotationImportReport(
        dataset_name=dataset,
        unmatched_files=unmatched,
        failures=[AnnotationImportFailure(image_filename=k, reason=why) for k, why in refused],
    )


async def _ensure_classes(
    datasets: AsyncDatasets,
    dataset: str,
    type_by_class: Mapping[str, str],
    report: AnnotationImportReport,
) -> None:
    """Add referenced-but-undefined classes (best-effort; failure → report entry)."""
    from pictograph.models.dataset import DatasetClass

    try:
        project = await datasets.get(dataset)
        existing = {c.name for c in project.classes}
        missing = [(n, t) for n, t in type_by_class.items() if n not in existing]
        if not missing:
            return
        new_classes: list[DatasetClass] = [*project.classes] + [
            DatasetClass(name=name, type=ann_type) for name, ann_type in missing
        ]
        # Merged onto the stored list (classes_base): a class added meanwhile is kept.
        await datasets.update(dataset, classes=new_classes, classes_base=project.classes)
    except PictographError as e:
        report.failures.append(
            AnnotationImportFailure(
                image_filename="<classes>",
                reason=f"could not create missing classes: {e}",
            )
        )
