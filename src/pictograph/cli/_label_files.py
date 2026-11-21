"""Reading local label files for ``pictograph annotations import-yolo / import-voc``.

A label folder is searched recursively (a YOLO export keeps ``train/labels`` and
``valid/labels`` side by side), and each file is keyed by its own name: the
import matches ``a.txt`` / ``a.xml`` to the dataset image named ``a.jpg`` (see
:mod:`pictograph.resources._files`). Two files with one name in different folders
would describe one image twice, so they are refused before anything is sent.

YOLO labels hold class NUMBERS. The names come from ``--classes`` (a ``data.yaml``,
``classes.txt`` or ``obj.names`` file, or ``a,b,c``), else from the first of those
files found in the label folder, its parent or its grandparent - where Ultralytics
and Roboflow exports put ``data.yaml``. A label that uses a number past the end of
the list is refused: it would otherwise be dropped without a word.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import typer

from pictograph.cli._errors import CliError, ExitCode

#: Files a YOLO folder carries beside its labels (never a label themselves).
_CLASS_FILES = ("data.yaml", "dataset.yaml", "classes.txt", "obj.names")
_CLASS_FILE_SUFFIXES = {".txt", ".names", ".yaml", ".yml"}
_CLASSES_FIX = (
    "Pass the class list in the order the labels number them: --classes data.yaml "
    "(or classes.txt, obj.names, or 'car,truck,bus')."
)


def label_files(directory: Path, suffix: str, *, skip: set[Path] | None = None) -> dict[str, Path]:
    """Every ``suffix`` file under ``directory`` (recursive), keyed by its name."""
    if not directory.is_dir():
        raise CliError(
            f"No such folder: {directory}",
            exit_code=ExitCode.INVALID,
            code="folder_not_found",
        )
    skipped = {p.resolve() for p in skip or set()}
    files: dict[str, Path] = {}
    shared: dict[str, list[str]] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() != suffix or path.resolve() in skipped:
            continue
        if path.name.lower() in _CLASS_FILES or path.name.lower().startswith("readme"):
            continue
        if path.name in files:
            shared.setdefault(path.name, [str(files[path.name])]).append(str(path))
            continue
        files[path.name] = path
    if shared:
        name, paths = next(iter(shared.items()))
        raise CliError(
            f"{len(shared)} label file name(s) appear in more than one folder, e.g. "
            f"{name!r}: {', '.join(paths)}. Each names one image, so they would label it twice.",
            exit_code=ExitCode.INVALID,
            code="duplicate_label_files",
            fix="Import each folder on its own (one command per split).",
            details={"duplicates": shared},
        )
    if not files:
        raise CliError(
            f"No {suffix} files under {directory}.",
            exit_code=ExitCode.INVALID,
            code="no_label_files",
        )
    return files


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def voc_image_name(xml: str) -> str | None:
    """The image a Pascal VOC file describes (its ``<filename>``), if it names one."""
    try:
        root = ET.fromstring(xml)  # noqa: S314 - the user's own local annotation file
    except ET.ParseError:
        return None
    name = (root.findtext("filename") or "").strip()
    return name or None


def class_names(spec: str | None, labels_dir: Path) -> tuple[list[str], Path | None]:
    """The YOLO class list, and the file it came from (``None`` = given inline)."""
    if spec is not None:
        path = Path(spec)
        if path.is_file():
            return read_class_file(path), path
        if path.suffix.lower() in _CLASS_FILE_SUFFIXES or "/" in spec:
            raise typer.BadParameter(f"No such file: {spec}", param_hint="--classes")
        names = [n.strip() for n in spec.split(",") if n.strip()]
        if not names:
            raise typer.BadParameter("give the class names.", param_hint="--classes")
        return names, None
    folder = labels_dir.resolve()
    for candidate in (folder, folder.parent, folder.parent.parent):
        for name in _CLASS_FILES:
            if (candidate / name).is_file():
                return read_class_file(candidate / name), candidate / name
    raise CliError(
        f"No class list found: YOLO labels number their classes, and no data.yaml, "
        f"classes.txt or obj.names is in {labels_dir} or the two folders above it.",
        exit_code=ExitCode.INVALID,
        code="no_class_list",
        fix=_CLASSES_FIX,
    )


def read_class_file(path: Path) -> list[str]:
    text = read_text(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        names = yaml_names(text)
        if not names:
            raise CliError(
                f"{path} has no `names:` list.",
                exit_code=ExitCode.INVALID,
                code="bad_class_file",
                fix=_CLASSES_FIX,
            )
        return names
    names = [line.strip() for line in text.splitlines() if line.strip()]
    if not names:
        raise CliError(
            f"{path} is empty.", exit_code=ExitCode.INVALID, code="bad_class_file", fix=_CLASSES_FIX
        )
    return names


def yaml_names(text: str) -> list[str] | None:
    """The ``names`` of a YOLO ``data.yaml``: a list, or a ``{0: a, 1: b}`` mapping.

    PyYAML reads it when installed; otherwise the three shapes exports write are
    read by hand (``names: [a, b]``, a ``- a`` block, a ``0: a`` block).
    """
    try:
        yaml: Any = importlib.import_module("yaml")
    except ImportError:
        names: Any = _names_block(text)
    else:
        try:
            data = yaml.safe_load(text)
        except Exception:
            return None
        names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, dict):
        try:
            return [str(names[k]) for k in sorted(names, key=int)]
        except (TypeError, ValueError):
            return None
    if isinstance(names, list) and names:
        return [str(n) for n in names]
    return None


def _unquote(value: str) -> str:
    value = value.strip()
    if value[:1] in ("'", '"'):
        end = value.find(value[0], 1)
        if end > 0:
            return value[1:end]
    return value.split(" #", 1)[0].strip()


def _names_block(text: str) -> list[str] | dict[int, str] | None:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("names:"):
            continue
        rest = line[len("names:") :].strip()
        if rest.startswith("["):
            inner = rest[1 : rest.rindex("]")] if "]" in rest else rest[1:]
            return [_unquote(part) for part in inner.split(",") if part.strip()]
        if rest and not rest.startswith("#"):
            return None
        items: list[str] = []
        mapping: dict[int, str] = {}
        for nxt in lines[i + 1 :]:
            body = nxt.strip()
            if not body or body.startswith("#"):
                continue
            if body.startswith("- "):
                items.append(_unquote(body[2:]))
            elif nxt[0].isspace() and ":" in body:
                key, _, value = body.partition(":")
                if not key.strip().isdigit():
                    return None
                mapping[int(key)] = _unquote(value)
            else:
                break
        return mapping or items or None
    return None


def check_class_numbers(labels: dict[str, str], names: list[str], source: str) -> None:
    """Refuse labels that number a class the list does not have."""
    highest = -1
    where = ""
    for filename, text in labels.items():
        for line in text.splitlines():
            token = line.split(maxsplit=1)[0] if line.strip() else ""
            if token.isdigit() and int(token) > highest:
                highest, where = int(token), filename
    if highest >= len(names):
        raise CliError(
            f"{where} uses class {highest}, but the class list ({source}) has {len(names)} "
            f"name(s), numbered 0-{len(names) - 1}. Those labels would be dropped.",
            exit_code=ExitCode.INVALID,
            code="class_number_out_of_range",
            fix=_CLASSES_FIX,
            details={"highest_class_number": highest, "classes": names},
        )
