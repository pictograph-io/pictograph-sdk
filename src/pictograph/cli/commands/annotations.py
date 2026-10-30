"""``pictograph annotations`` - read, save and delete annotations; rename, merge and delete
classes; import COCO, YOLO and Pascal VOC files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from pydantic import TypeAdapter

from pictograph.cli import _label_files as lfiles
from pictograph.cli._client import get_client
from pictograph.cli._errors import CliError, ExitCode, command_group
from pictograph.cli._format import print_json
from pictograph.models.annotation import Annotation

if TYPE_CHECKING:
    from pictograph.resources.annotations import AnnotationImportReport

app = command_group(no_args_is_help=True)


@app.command("get", help="Fetch annotations attached to an image.")
def get_annotations(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    image: Annotated[str, typer.Argument(help="Image filename (an id also works).")],
    directory: Annotated[
        str | None,
        typer.Option("--directory", help="Directory, when the filename is in more than one."),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    annotations = client.annotations.get(dataset, image, directory=directory)
    print_json([a.model_dump(mode="json", exclude_none=True) for a in annotations])


@app.command("save", help="Replace annotations on an image (full overwrite).")
def save_annotations(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    image: Annotated[str, typer.Argument(help="Image filename (an id also works).")],
    file: Annotated[
        Path,
        typer.Option(
            "--file",
            "-f",
            help="Path to a JSON file containing a list of annotations.",
        ),
    ],
    directory: Annotated[
        str | None,
        typer.Option("--directory", help="Directory, when the filename is in more than one."),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    raw = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise CliError(
            "JSON file must contain a list of annotations at the top level.",
            exit_code=ExitCode.INVALID,
            code="invalid_input",
        )
    adapter: TypeAdapter[list[Annotation]] = TypeAdapter(list[Annotation])
    parsed = adapter.validate_python(raw)
    client = get_client(api_key)
    result = client.annotations.save(dataset, image, parsed, directory=directory)
    print_json(
        {
            "image_id": result.image_id,
            "previous_count": result.previous_count,
            "new_count": result.new_count,
            "status": result.status,
        }
    )


@app.command("bulk-save", help="Replace annotations on many images in one server-side call.")
def bulk_save_annotations(
    file: Annotated[
        Path,
        typer.Option(
            "--file",
            "-f",
            help='JSON object mapping image UUID -> list of annotations, e.g. {"img-1": [...]}.',
        ),
    ],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    raw = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CliError(
            "JSON file must be an object mapping image_id -> [annotations].",
            exit_code=ExitCode.INVALID,
            code="invalid_input",
        )
    adapter: TypeAdapter[list[Annotation]] = TypeAdapter(list[Annotation])
    saves = {image_id: adapter.validate_python(anns) for image_id, anns in raw.items()}
    client = get_client(api_key)
    result = client.annotations.bulk_save(saves)
    print_json(
        {
            "saved": [
                {
                    "image_id": s.image_id,
                    "previous_count": s.previous_count,
                    "new_count": s.new_count,
                    "status": s.status,
                }
                for s in result.saved
            ],
            "failed": [{"image_id": f.image_id, "error": f.error} for f in result.failed],
        }
    )


@app.command("delete", help="Remove every annotation from an image.")
def delete_annotations(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    image: Annotated[str, typer.Argument(help="Image filename (an id also works).")],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    directory: Annotated[
        str | None,
        typer.Option("--directory", help="Directory, when the filename is in more than one."),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(f"Delete all annotations on {image!r} in {dataset!r}?"):
        raise typer.Abort()
    client = get_client(api_key)
    result = client.annotations.delete(dataset, image, directory=directory)
    print_json(
        {
            "image_id": result.image_id,
            "deleted_count": result.deleted_count,
        }
    )


@app.command(
    "rename-class",
    help="Rename a class across a whole dataset - ontology + every annotation.",
)
def rename_class(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    old_name: Annotated[str, typer.Argument(help="Current class name.")],
    new_name: Annotated[str, typer.Argument(help="New class name.")],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(
        f"Rename class {old_name!r} to {new_name!r} across dataset {dataset!r}?"
    ):
        raise typer.Abort()
    client = get_client(api_key)
    project = client.datasets.get(dataset)
    result = client.annotations.rename_class(project.id, old_name, new_name)
    print_json(
        {
            "dataset_id": result.dataset_id,
            "old_name": result.old_name,
            "new_name": result.new_name,
            "images_updated": result.images_updated,
            "annotations_updated": result.annotations_updated,
            "config_updated": result.config_updated,
        }
    )


@app.command(
    "merge-class",
    help="Merge one class into another across a dataset - reassign annotations + drop the source.",
)
def merge_class(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    source_name: Annotated[str, typer.Argument(help="Class merged away.")],
    target_name: Annotated[str, typer.Argument(help="Class kept.")],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(
        f"Merge class {source_name!r} into {target_name!r} across dataset {dataset!r}?"
    ):
        raise typer.Abort()
    client = get_client(api_key)
    project = client.datasets.get(dataset)
    result = client.annotations.merge_class(project.id, source_name, target_name)
    print_json(
        {
            "dataset_id": result.dataset_id,
            "source_name": result.source_name,
            "target_name": result.target_name,
            "images_updated": result.images_updated,
            "annotations_updated": result.annotations_updated,
            "config_updated": result.config_updated,
        }
    )


@app.command(
    "delete-class",
    help="Delete a class from a dataset's ontology, optionally its annotations too.",
)
def delete_class(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    name: Annotated[str, typer.Argument(help="Class name to delete.")],
    with_annotations: Annotated[
        bool,
        typer.Option("--with-annotations", help="Also delete every annotation of this class."),
    ] = False,
    class_type: Annotated[
        str | None,
        typer.Option("--type", help="Narrow removal to one (name, type) ontology entry."),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    scope = "and its annotations " if with_annotations else ""
    if not yes and not typer.confirm(f"Delete class {name!r} {scope}from dataset {dataset!r}?"):
        raise typer.Abort()
    client = get_client(api_key)
    project = client.datasets.get(dataset)
    result = client.annotations.delete_class(
        project.id, name, class_type=class_type, delete_annotations=with_annotations
    )
    print_json(
        {
            "dataset_id": result.dataset_id,
            "name": result.name,
            "config_updated": result.config_updated,
            "images_updated": result.images_updated,
            "annotations_removed": result.annotations_removed,
        }
    )


# ───────────── importing annotation files (B533 P1 #7) ─────────────

CreateClassesOption = Annotated[
    bool,
    typer.Option(
        "--create-classes/--no-create-classes",
        help="Add the classes the files use but the dataset lacks (added to its classes, "
        "never replacing them).",
    ),
]
_MATCHING = (
    "Each file lands on the image with its exact name, else the one image with its "
    "stem (a.txt and a.xml label a.jpg). A name two images share is reported, never "
    "guessed. Each matched image's annotations are REPLACED."
)


def finish_import(report: AnnotationImportReport, files: int, **extra: object) -> None:
    """Print the import report; exit 1 when some files failed, 4 when none matched."""
    payload: dict[str, object] = {
        "dataset": report.dataset_name,
        "files": files,
        **extra,
        "images_matched": report.images_matched,
        "images_saved": report.images_saved,
        "annotations_saved": report.annotations_saved,
        "unmatched_files": report.unmatched_files,
        "failures": [{"image": f.image_filename, "reason": f.reason} for f in report.failures],
        "success": report.success,
    }
    print_json(payload)
    if report.failures:
        raise CliError(
            f"{len(report.failures)} file(s) were not imported - each is listed under "
            "`failures` with the reason.",
            exit_code=ExitCode.FAILED,
            code="some_files_failed",
            details={"failed": len(report.failures)},
        )
    if report.images_matched == 0 and report.unmatched_files:
        raise CliError(
            f"None of the {len(report.unmatched_files)} file(s) match an image of "
            f"{report.dataset_name!r}.",
            exit_code=ExitCode.NOT_FOUND,
            code="no_images_matched",
            fix="Upload the images first (pictograph images upload), or check the names: a "
            "file matches the image with its exact name, or the one image with its stem.",
            details={"unmatched_files": report.unmatched_files[:20]},
        )
    if report.images_matched == 0:
        raise CliError(
            "The files hold no annotations.",
            exit_code=ExitCode.INVALID,
            code="nothing_to_import",
        )


@app.command(
    "import-coco",
    short_help="Import a COCO file onto the dataset's images.",
    help="Import a COCO annotation file onto a dataset that already holds its images "
    "(boxes, polygons, keypoints). " + _MATCHING,
)
def import_coco(
    dataset: Annotated[str, typer.Argument(help="Dataset name (it must hold the images).")],
    coco_file: Annotated[Path, typer.Argument(help="The COCO JSON file.")],
    create_classes: CreateClassesOption = True,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not coco_file.is_file():
        raise CliError(
            f"No such file: {coco_file}", exit_code=ExitCode.INVALID, code="file_not_found"
        )
    client = get_client(api_key)
    report = client.annotations.import_coco(
        dataset, coco_file, create_missing_classes=create_classes
    )
    finish_import(report, files=1)


@app.command(
    "import-yolo",
    short_help="Import a folder of YOLO labels onto the dataset's images.",
    help="Import a folder of YOLO .txt labels (boxes, or polygons from a segmentation "
    "export) onto a dataset that already holds the images; searched recursively. " + _MATCHING,
)
def import_yolo(
    dataset: Annotated[str, typer.Argument(help="Dataset name (it must hold the images).")],
    labels_dir: Annotated[Path, typer.Argument(help="The folder of .txt label files.")],
    classes: Annotated[
        str | None,
        typer.Option(
            "--classes",
            help="The class names in label order: a data.yaml, classes.txt or obj.names "
            "file, or 'car,truck,bus'. Default: the first of those files in the folder, "
            "its parent or its grandparent.",
        ),
    ] = None,
    create_classes: CreateClassesOption = True,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    names, class_file = lfiles.class_names(classes, labels_dir)
    files = lfiles.label_files(labels_dir, ".txt", skip={class_file} if class_file else None)
    labels = {name: lfiles.read_text(path) for name, path in files.items()}
    source = str(class_file) if class_file else "--classes"
    lfiles.check_class_numbers(labels, names, source)
    client = get_client(api_key)
    report = client.annotations.import_yolo(
        dataset, labels, names, create_missing_classes=create_classes
    )
    finish_import(report, files=len(labels), classes=names, class_list=source)


@app.command(
    "import-voc",
    short_help="Import a folder of Pascal VOC files onto the dataset's images.",
    help="Import a folder of Pascal VOC .xml files (boxes) onto a dataset that already "
    "holds the images; searched recursively. A file names its image in <filename>. " + _MATCHING,
)
def import_voc(
    dataset: Annotated[str, typer.Argument(help="Dataset name (it must hold the images).")],
    xml_dir: Annotated[Path, typer.Argument(help="The folder of .xml files.")],
    create_classes: CreateClassesOption = True,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    files = lfiles.label_files(xml_dir, ".xml")
    xml_by_image: dict[str, str] = {}
    for name, path in files.items():
        xml = lfiles.read_text(path)
        image = lfiles.voc_image_name(xml)
        # Two files naming one image keep their own names, so the import reports it.
        xml_by_image[image if image and image not in xml_by_image else name] = xml
    client = get_client(api_key)
    report = client.annotations.import_pascal_voc(
        dataset, xml_by_image, create_missing_classes=create_classes
    )
    finish_import(report, files=len(files))
