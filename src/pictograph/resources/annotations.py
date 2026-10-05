"""Annotations resource - read, save, delete annotations for an image.

The wire format on this resource is the **canonical Pictograph JSON**
defined in :mod:`pictograph.models.annotation`. There is no shorthand:
``BoundingBox`` is always ``{"x", "y", "w", "h"}`` (object), polygon paths
are always lists of ``{"x", "y"}`` point objects, polyline path is a single
list of point objects, keypoint is a point object. The SDK serialises
``Annotation`` Pydantic models via ``model_dump(mode="json", exclude_none=True)``
and the result is bit-for-bit what lands in the image's ``annotations_json``.

The backend rejects any other shape with a structured ``ValidationError`` -
that's what catches old shorthand usage early.

The ``import_*`` methods are the capstone over :mod:`pictograph.formats`: they
parse an external label set (COCO / Pascal VOC / YOLO), create any class the
labels reference but the dataset doesn't define, resolve each image by file name,
and write the annotations back with chunked :meth:`Annotations.bulk_save` -
returning an :class:`AnnotationImportReport` rather than raising on a partial match.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from pictograph.exceptions import PictographError
from pictograph.models.annotation import Annotation
from pictograph.resources import _resolve
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from typing import Any

    from pictograph.resources.datasets import Datasets

_API_PATH = "/api/v1/developer/annotations"


def _image_route(dataset_name: str, segment: str) -> str:
    """`/annotations/{dataset}/{directory}/{filename}` - the per-image route.

    The directory is folded INTO the trailing `:path` segment. Confirmed live:
    with it 200, without it 404, and a UUID there 404 as well.
    """
    return f"{_API_PATH}/{quote(dataset_name, safe='')}/{quote(segment, safe='/')}"


# Backend cap on the bulk-save endpoint's per-call entry count.
_BULK_SAVE_CAP = 200

# Module-level adapter is built once and reused - Pydantic v2 TypeAdapter
# construction is non-trivial (compiles a discriminated-union validator).
_ANNOTATION_LIST_ADAPTER: TypeAdapter[list[Annotation]] = TypeAdapter(list[Annotation])


class SaveResult(BaseModel):
    """Outcome of :meth:`Annotations.save`.

    Attributes:
        image_id: Image whose annotations were updated.
        previous_count: Number of annotations the image had before this call.
        new_count: Number of annotations after this call.
        status: The image's annotation lifecycle status - ``"new"``,
            ``"in_progress"``, or ``"complete"``. Saving annotations does NOT
            change it (the prior status is returned unchanged); transition it
            explicitly via ``PATCH /api/v1/annotations/{id}/status``.
    """

    model_config = ConfigDict(frozen=True)

    image_id: str
    previous_count: int
    new_count: int
    status: str


class DeleteResult(BaseModel):
    """Outcome of :meth:`Annotations.delete`."""

    model_config = ConfigDict(frozen=True)

    image_id: str
    deleted_count: int


class BulkSaveFailure(BaseModel):
    """One image that could not be saved in a :meth:`Annotations.bulk_save` call."""

    model_config = ConfigDict(frozen=True)

    image_id: str
    error: str


class BulkSaveResult(BaseModel):
    """Outcome of :meth:`Annotations.bulk_save`.

    Per-image and idempotent-ish: a bad image_id lands in :attr:`failed`
    rather than failing the whole call, so the rest still save.

    Attributes:
        saved: One :class:`SaveResult` per image that was updated.
        failed: One :class:`BulkSaveFailure` per image that couldn't be saved
            (e.g. an image id that doesn't exist in your org).
    """

    model_config = ConfigDict(frozen=True)

    saved: list[SaveResult]
    failed: list[BulkSaveFailure]

    @property
    def saved_count(self) -> int:
        """Number of images successfully saved."""
        return len(self.saved)


class RenameClassResult(BaseModel):
    """Outcome of :meth:`Annotations.rename_class`.

    Attributes:
        dataset_id: Dataset the rename ran in.
        old_name: The class name that was replaced.
        new_name: The class name annotations/ontology now carry.
        images_updated: Images whose annotations were rewritten.
        annotations_updated: Individual annotations renamed.
        config_updated: True when the dataset's class ontology entry was
            renamed too (False when the class only existed on annotations).
    """

    model_config = ConfigDict(frozen=True)

    dataset_id: str
    old_name: str
    new_name: str
    images_updated: int
    annotations_updated: int
    config_updated: bool


class MergeClassResult(BaseModel):
    """Outcome of :meth:`Annotations.merge_class`.

    Attributes:
        dataset_id: Dataset the merge ran in.
        source_name: The class merged away (removed from the ontology).
        target_name: The class kept; source annotations now carry it.
        images_updated: Images whose annotations were rewritten.
        annotations_updated: Individual annotations reassigned.
        config_updated: True when the dataset's class ontology was edited.
    """

    model_config = ConfigDict(frozen=True)

    dataset_id: str
    source_name: str
    target_name: str
    images_updated: int
    annotations_updated: int
    config_updated: bool


class DeleteClassResult(BaseModel):
    """Outcome of :meth:`Annotations.delete_class`.

    Attributes:
        dataset_id: Dataset the delete ran in.
        name: The class removed from the ontology.
        config_updated: True when the ontology entry was removed.
        images_updated: Images whose annotations were rewritten (0 unless
            ``delete_annotations`` was set).
        annotations_removed: Annotations stripped (0 unless
            ``delete_annotations`` was set).
    """

    model_config = ConfigDict(frozen=True)

    dataset_id: str
    name: str
    config_updated: bool
    images_updated: int
    annotations_removed: int


class AnnotationImportFailure(BaseModel):
    """One image whose imported annotations could not be saved."""

    model_config = ConfigDict(frozen=True)

    image_filename: str
    reason: str


class AnnotationImportReport(BaseModel):
    """Outcome of an :meth:`Annotations.import_coco` / ``import_pascal_voc`` /
    ``import_yolo`` call.

    Inspect ``unmatched_files`` (file names present in the source but with no
    matching image in the dataset) and ``failures`` (images that resolved but
    whose save was rejected) to retry a subset.
    """

    dataset_name: str
    images_matched: int = 0
    images_saved: int = 0
    annotations_saved: int = 0
    unmatched_files: list[str] = Field(default_factory=list)
    failures: list[AnnotationImportFailure] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.failures and not self.unmatched_files and self.images_saved > 0


class Annotations(Resource):
    """Read, save, and delete annotations for individual images."""

    def get(
        self,
        dataset_name: str,
        image: str,
        *,
        directory_path: str | None = None,
    ) -> list[Annotation]:
        """Fetch the annotations attached to an image.

        A user knows an image by its FILENAME inside a dataset, which is the pair
        the grid shows - so that is what this takes. An image id is still accepted
        in place of the filename (detected by shape), and `directory_path`
        disambiguates when the same filename lives in two directories.


        Returns a list of typed :data:`Annotation` (discriminated union over
        bbox / polygon / polyline / keypoint subclasses). An image with zero
        annotations returns an empty list - never raises ``NotFoundError``
        for the "no annotations" case (only for "no such image").
        """
        segment = _resolve.image_segment(self._transport, image, directory_path=directory_path)
        response = self._transport.request("GET", _image_route(dataset_name, segment))
        raw = response.get("annotations", [])
        return _ANNOTATION_LIST_ADAPTER.validate_python(raw)

    def save(
        self,
        dataset_name: str,
        image: str,
        annotations: Sequence[Annotation],
        *,
        directory_path: str | None = None,
    ) -> SaveResult:
        """Replace an image's annotations with the given list.

        A user knows an image by its FILENAME inside a dataset, which is the pair
        the grid shows - so that is what this takes. An image id is still accepted
        in place of the filename (detected by shape), and `directory_path`
        disambiguates when the same filename lives in two directories.


        This is a *full* replacement: the existing annotations are
        overwritten. Pass an empty list to clear (equivalent to
        :meth:`delete`).

        Args:
            image_id: Image to update.
            annotations: Annotation objects in canonical Pictograph JSON
                format. The SDK validates each via Pydantic at construction;
                the backend re-validates the wire payload as a defence in
                depth.

        Returns:
            :class:`SaveResult` with previous/new annotation counts and the
            updated lifecycle status.

        Raises:
            ValidationError: Backend rejected the payload (e.g., a class
                name not in the dataset's class set).
            ForbiddenError: API key role lacks write permission.
            NotFoundError: ``image_id`` does not exist.
        """
        segment = _resolve.image_segment(self._transport, image, directory_path=directory_path)
        # The image is identified ENTIRELY by the URL. The request model declares
        # `annotations` and nothing else (extra="forbid"), so an `image_id`
        # alongside it - which this used to send - is a 422, not a harmless echo.
        body = {
            "annotations": [ann.model_dump(mode="json", exclude_none=True) for ann in annotations],
        }
        response = self._transport.request(
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

    def bulk_save(
        self,
        saves: Mapping[str, Sequence[Annotation]],
    ) -> BulkSaveResult:
        """Save annotations for up to 200 images in one server-side call.

        Each entry is a *full* replacement of that image's annotations (same
        semantics as :meth:`save`). One request instead of N - every image is
        org-scoped server-side, and an image id that doesn't resolve in your
        org lands in :attr:`BulkSaveResult.failed` rather than failing the whole
        batch.

        Args:
            saves: Mapping of ``image_id`` → the annotations to set on it
                (canonical Pictograph JSON ``Annotation`` objects). At most 200
                entries; a larger map raises ``ValidationError`` (the backend
                caps the batch).

        Returns:
            :class:`BulkSaveResult` - ``saved`` (per-image counts + status) and
            ``failed`` (image_id + error).

        Raises:
            ValidationError: Backend rejected the payload (bad annotation shape,
                or more than 200 entries).
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
        response = self._transport.request("POST", f"{_API_PATH}/bulk", json=body)
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

    def delete(
        self,
        dataset_name: str,
        image: str,
        *,
        directory_path: str | None = None,
    ) -> DeleteResult:
        """Remove all annotations from an image.

        A user knows an image by its FILENAME inside a dataset, which is the pair
        the grid shows - so that is what this takes. An image id is still accepted
        in place of the filename (detected by shape), and `directory_path`
        disambiguates when the same filename lives in two directories.


        Equivalent to :meth:`save` with an empty list, but uses ``DELETE``
        and requires admin/owner role server-side.
        """
        segment = _resolve.image_segment(self._transport, image, directory_path=directory_path)
        response = self._transport.request("DELETE", _image_route(dataset_name, segment))
        return DeleteResult(
            image_id=response.get("image_id", ""),
            deleted_count=response.get("deleted_count", 0),
        )

    def rename_class(self, dataset_name: str, old_name: str, new_name: str) -> RenameClassResult:
        """Rename an annotation class across a WHOLE dataset in one call.

        Renames the class ONTOLOGY entry (the dataset config the editor's
        class pills come from) AND every stored annotation carrying
        ``old_name`` - one set-based server-side statement, not a per-image
        loop. Use it for label-taxonomy cleanup (e.g. ``car`` → ``vehicle``)
        without touching the images yourself. Member role or higher.

        Args:
            dataset_name: The dataset's name (a UUID also works).
            old_name: The class name to replace.
            new_name: The replacement name. A collision with an existing
                class of the same annotation type raises ``ConflictError``.

        Returns:
            :class:`RenameClassResult` with the rewrite counts.
        """
        response = self._transport.request(
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

    def merge_class(
        self, dataset_name: str, source_name: str, target_name: str
    ) -> MergeClassResult:
        """Merge one annotation class into another across a WHOLE dataset.

        Every annotation labeled ``source_name`` is reassigned to
        ``target_name`` and the source class is dropped from the ontology (the
        target is kept). The web editor blocks renaming a class onto an existing
        one, so this is the way to combine two classes (e.g. ``car`` + ``auto``
        → ``vehicle``) - one set-based server-side statement. Member role or
        higher.

        Args:
            dataset_name: The dataset's name (a UUID also works).
            source_name: The class merged away.
            target_name: The class kept; source annotations now carry it.

        Returns:
            :class:`MergeClassResult` with the rewrite counts.
        """
        response = self._transport.request(
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

    def delete_class(
        self,
        dataset_name: str,
        name: str,
        *,
        class_type: str | None = None,
        delete_annotations: bool = False,
    ) -> DeleteClassResult:
        """Delete a class from a dataset's ontology, optionally its annotations.

        With ``delete_annotations=True`` every annotation of the class is also
        stripped in one set-based statement; otherwise only the ontology entry
        is removed and existing annotations are left in place. ``class_type``
        narrows the ontology removal to a single ``(name, type)`` entry. Member
        role or higher.

        Args:
            dataset_name: The dataset's name (a UUID also works).
            name: The class name to delete.
            class_type: Optional annotation type to narrow the ontology removal.
            delete_annotations: Also remove every annotation of the class.

        Returns:
            :class:`DeleteClassResult` with the removal counts.
        """
        response = self._transport.request(
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

    def import_coco(
        self,
        dataset_name: str,
        coco: dict[str, Any] | str | Path,
        *,
        create_missing_classes: bool = True,
        save_chunk: int = _BULK_SAVE_CAP,
    ) -> AnnotationImportReport:
        """Parse a COCO dataset and save its annotations onto a Pictograph dataset.

        COCO carries absolute pixel coordinates, so this is self-contained - the
        images just have to already be in the dataset under the file names the
        COCO ``file_name`` fields reference.

        Args:
            dataset_name: Destination dataset (must already exist and hold the images
                the COCO ``file_name`` fields reference).
            coco: A parsed COCO dict, or a path / JSON string to one.
            create_missing_classes: When ``True`` (default), add any class the COCO
                categories reference but the dataset doesn't yet define, inferring
                each class's annotation type from its first annotation.
            save_chunk: Images per :meth:`bulk_save` call (backend cap 200).

        Returns:
            An :class:`AnnotationImportReport`.

        Raises:
            NotFoundError: ``dataset_name`` does not exist.

        Example:
            >>> report = client.annotations.import_coco("road-signs", "instances_val.json")
            >>> report.images_saved, len(report.unmatched_files)
            (412, 0)
        """
        from pictograph.formats import from_coco

        imp = from_coco(coco)
        return self._import_by_filename(
            dataset_name,
            imp.annotations,
            create_missing_classes=create_missing_classes,
            save_chunk=save_chunk,
        )

    def import_pascal_voc(
        self,
        dataset_name: str,
        xml_by_filename: Mapping[str, str],
        *,
        create_missing_classes: bool = True,
        save_chunk: int = _BULK_SAVE_CAP,
    ) -> AnnotationImportReport:
        """Parse per-image Pascal VOC XML and save the annotations onto a dataset.

        Pascal VOC uses absolute pixel corners, so (unlike YOLO) no image dimensions
        are needed - build ``xml_by_filename`` by reading a directory of ``.xml``
        files keyed by the image file name they describe.

        Args:
            dataset_name: Destination dataset (must already exist and hold the images).
            xml_by_filename: ``file_name`` → that image's Pascal VOC ``.xml`` contents.
            create_missing_classes: Add referenced-but-undefined classes (default True).
            save_chunk: Images per :meth:`bulk_save` call (backend cap 200).

        Returns:
            An :class:`AnnotationImportReport`.

        Raises:
            NotFoundError: ``dataset_name`` does not exist.
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
        report = self._import_by_filename(
            dataset_name,
            annotations_by_filename,
            create_missing_classes=create_missing_classes,
            save_chunk=save_chunk,
        )
        report.failures.extend(parse_failures)
        return report

    def import_yolo(
        self,
        dataset_name: str,
        labels: Mapping[str, str],
        class_names: Sequence[str],
        *,
        create_missing_classes: bool = True,
        save_chunk: int = _BULK_SAVE_CAP,
    ) -> AnnotationImportReport:
        """Parse YOLO label text per image and save the annotations onto a dataset.

        Each image's pixel size is read from the dataset (YOLO coordinates are
        normalized), so you only pass the label text and the class list.

        Args:
            dataset_name: Destination dataset (must already exist).
            labels: ``file_name`` → that image's YOLO ``.txt`` contents.
            class_names: Ordered class names - YOLO's integer class index maps here.
            create_missing_classes: Add referenced-but-undefined classes (default True).
            save_chunk: Images per :meth:`bulk_save` call (backend cap 200).

        Returns:
            An :class:`AnnotationImportReport`. A file whose image has no stored
            width/height lands in ``failures`` (YOLO can't be denormalized without it).

        Raises:
            NotFoundError: ``dataset_name`` does not exist.
        """
        from pictograph.formats import from_yolo
        from pictograph.resources.datasets import Datasets
        from pictograph.resources.images import Images

        # datasets.get validates the dataset exists (raises NotFoundError) and gives
        # us the id to enumerate images.
        project = Datasets(self._transport).get(dataset_name)
        # Build filename → (id, width, height) from the dataset's images.
        dims_by_filename: dict[str, tuple[str, int | None, int | None]] = {}
        for img in Images(self._transport).iter(project.id):
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

        self._save_resolved(
            report,
            annotations_by_filename,
            id_by_filename={fn: v[0] for fn, v in dims_by_filename.items()},
            create_missing_classes=create_missing_classes,
            save_chunk=save_chunk,
        )
        return report

    # ───────────── import internals ─────────────

    def _import_by_filename(
        self,
        dataset_name: str,
        annotations_by_filename: Mapping[str, Sequence[Annotation]],
        *,
        create_missing_classes: bool,
        save_chunk: int,
    ) -> AnnotationImportReport:
        """Resolve file names → image ids from the dataset, then save."""
        from pictograph.resources.datasets import Datasets
        from pictograph.resources.images import Images

        report = AnnotationImportReport(dataset_name=dataset_name)
        project = Datasets(self._transport).get(dataset_name)
        id_by_filename = {img.filename: img.id for img in Images(self._transport).iter(project.id)}
        self._save_resolved(
            report,
            annotations_by_filename,
            id_by_filename=id_by_filename,
            create_missing_classes=create_missing_classes,
            save_chunk=save_chunk,
        )
        return report

    def _save_resolved(
        self,
        report: AnnotationImportReport,
        annotations_by_filename: Mapping[str, Sequence[Annotation]],
        *,
        id_by_filename: Mapping[str, str],
        create_missing_classes: bool,
        save_chunk: int,
    ) -> None:
        """Ensure classes exist, then bulk-save annotations for resolved images."""
        # Match file names to image ids; the rest are unmatched.
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
            from pictograph.resources.datasets import Datasets

            type_by_class: dict[str, str] = {}
            for anns in matched.values():
                for ann in anns:
                    type_by_class.setdefault(ann.name, ann.type)
            _ensure_classes(Datasets(self._transport), report.dataset_name, type_by_class, report)

        # Chunk the {image_id: annotations} map at the backend bulk-save cap.
        items = list(matched.items())
        chunk_size = max(1, min(save_chunk, _BULK_SAVE_CAP))
        for start in range(0, len(items), chunk_size):
            chunk = dict(items[start : start + chunk_size])
            try:
                result = self.bulk_save(chunk)
            except PictographError as e:
                for _image_id in chunk:
                    report.failures.append(
                        AnnotationImportFailure(image_filename=_image_id, reason=str(e))
                    )
                continue
            report.images_saved += result.saved_count
            report.annotations_saved += sum(s.new_count for s in result.saved)
            for f in result.failed:
                report.failures.append(
                    AnnotationImportFailure(image_filename=f.image_id, reason=f.error)
                )


def _ensure_classes(
    datasets: Datasets,
    dataset_name: str,
    type_by_class: Mapping[str, str],
    report: AnnotationImportReport,
) -> None:
    """Add classes the annotations reference but the dataset doesn't yet define.

    Best-effort: a failure to update the dataset's class list is recorded as a
    single report failure rather than aborting the import - annotations whose
    class already exists still save.
    """
    from pictograph.models.dataset import DatasetClass

    try:
        project = datasets.get(dataset_name)
        existing = {c.name for c in project.classes}
        missing = [(n, t) for n, t in type_by_class.items() if n not in existing]
        if not missing:
            return
        new_classes: list[DatasetClass] = [*project.classes] + [
            DatasetClass(name=name, type=ann_type) for name, ann_type in missing
        ]
        datasets.update(dataset_name, classes=new_classes)
    except PictographError as e:
        report.failures.append(
            AnnotationImportFailure(
                image_filename="<classes>",
                reason=f"could not create missing classes: {e}",
            )
        )
