"""``pictograph classes {list,add,update,remove}`` - a dataset's classes (its ontology).

::

    pictograph classes list road-signs
    pictograph classes add road-signs "stop sign" --type box --color '#e6194b'
    pictograph classes add people person --type keypoint --skeleton person-skeleton.json
    pictograph classes update road-signs "stop sign" --attributes sign-attributes.json
    pictograph classes remove road-signs "stop sign" --yes

Every write reads the list, edits it, and sends what it read as ``classes_base``: the
platform applies the edit to the list stored NOW, so a class someone added in the app
since the read survives (a plain full replace would erase it). A class is keyed by its
name AND its type - ``car`` the box and ``car`` the polygon are two classes. Renaming
or merging a class touches its annotations too, so it lives under ``pictograph
annotations rename-class`` / ``merge-class``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from pydantic import TypeAdapter

from pictograph.cli._client import get_client
from pictograph.cli._errors import CliError, ExitCode, command_group
from pictograph.cli._format import print_json, print_table
from pictograph.models.dataset import ClassAttribute, DatasetClass, SkeletonTemplate

if TYPE_CHECKING:
    from pictograph.client import Client

app = command_group(no_args_is_help=True)

# The class-definition types (``bbox`` is accepted for ``box``: the platform stores
# ``box`` for a box class, ``bbox`` on a box annotation).
TYPES = ("box", "polygon", "polyline", "keypoint", "tag")
_ALIASES = {"bbox": "box"}
TYPE_HELP = "box, polygon, polyline, keypoint or tag (bbox is accepted for box)."
ATTRIBUTES_HELP = (
    "A JSON file: a list of attribute definitions annotations of this class may carry, "
    'e.g. [{"name": "occluded", "type": "checkbox"}, {"name": "color", "type": "select", '
    '"values": ["red", "blue"]}]. Types: text, number, select, checkbox.'
)
SKELETON_HELP = (
    "A JSON file, keypoint classes only: the joint template, "
    '{"nodes": [{"name": "nose", "x": 0.5, "y": 0.1}, ...], "edges": [[0, 1], ...]} - '
    "nodes are the joints' class names in export order (x/y a suggested position on the "
    "unit square), edges 0-indexed pairs."
)


def _type(value: str | None, *, flag: str = "--type") -> str | None:
    if value is None:
        return None
    token = _ALIASES.get(value.strip().lower(), value.strip().lower())
    if token not in TYPES:
        raise typer.BadParameter(f"{value!r}: use one of {', '.join(TYPES)}.", param_hint=flag)
    return token


def _json_file(path: Path, flag: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise typer.BadParameter(f"No such file: {path}", param_hint=flag) from None
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{path} is not JSON: {exc}", param_hint=flag) from None


def _attributes(path: Path | None) -> list[ClassAttribute] | None:
    if path is None:
        return None
    raw = _json_file(path, "--attributes")
    return TypeAdapter(list[ClassAttribute]).validate_python(raw)


def _skeleton(path: Path | None, class_type: str | None) -> SkeletonTemplate | None:
    if path is None:
        return None
    if class_type is not None and class_type != "keypoint":
        raise typer.BadParameter(
            f"only a keypoint class has a skeleton (this one is {class_type}).",
            param_hint="--skeleton",
        )
    skeleton = SkeletonTemplate.model_validate(_json_file(path, "--skeleton"))
    for i, j in skeleton.edges:
        if i == j or not (0 <= i < len(skeleton.nodes) and 0 <= j < len(skeleton.nodes)):
            raise typer.BadParameter(
                f"edge [{i}, {j}]: each edge is two different joint indexes, "
                f"0 to {len(skeleton.nodes) - 1}.",
                param_hint="--skeleton",
            )
    names = [n.name for n in skeleton.nodes]
    if len(set(names)) != len(names):
        raise typer.BadParameter("joint names must be unique.", param_hint="--skeleton")
    return skeleton


def _same(cls: DatasetClass, name: str, class_type: str | None) -> bool:
    stored = _ALIASES.get(cls.type or "", cls.type or "")
    return cls.name == name and (class_type is None or stored == class_type)


def _find(classes: list[DatasetClass], name: str, class_type: str | None, dataset: str) -> int:
    """The index of the one class ``name`` (narrowed by ``class_type``) names."""
    hits = [i for i, c in enumerate(classes) if _same(c, name, class_type)]
    if not hits:
        kind = f" of type {class_type}" if class_type else ""
        raise CliError(
            f"Dataset {dataset!r} has no class {name!r}{kind}.",
            exit_code=ExitCode.NOT_FOUND,
            code="class_not_found",
            fix=f"List them with `pictograph classes list {_quoted(dataset)}`.",
        )
    if len(hits) > 1:
        types = ", ".join(sorted({classes[i].type or "?" for i in hits}))
        raise typer.BadParameter(
            f"{name!r} names {len(hits)} classes ({types}); say which with --type.",
            param_hint="--type",
        )
    return hits[0]


def _write(
    client: Client, dataset: str, base: list[DatasetClass], desired: list[DatasetClass]
) -> list[DatasetClass]:
    """Apply ``base`` -> ``desired`` to the stored list; the list as it is now."""
    updated = client.datasets.update(dataset, classes=desired, classes_base=base)
    return list(updated.classes or [])


def _row(cls: DatasetClass) -> dict[str, Any]:
    return {
        "name": cls.name,
        "type": cls.type,
        "color": cls.color,
        "attributes": len(cls.attributes or []),
        "joints": len(cls.skeleton.nodes) if cls.skeleton else None,
    }


def _quoted(value: str) -> str:
    return f"'{value}'" if any(c.isspace() for c in value) else value


@app.command("list", help="List a dataset's classes: name, type, colour, attributes, joints.")
def list_classes(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the class definitions as JSON.")
    ] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    classes = list(client.datasets.get(dataset).classes or [])
    if json_output:
        print_json([c.model_dump(mode="json", exclude_none=True) for c in classes])
    else:
        print_table([_row(c) for c in classes], title=f"Classes of {dataset} ({len(classes)})")


@app.command(
    "add",
    help="Add a class to a dataset. A class is its name AND its type: car the box and car "
    "the polygon are two classes. Without --color the platform picks an unused one.",
)
def add_class(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    name: Annotated[str, typer.Argument(help="Class name.")],
    class_type: Annotated[str, typer.Option("--type", "-t", help=TYPE_HELP)],
    color: Annotated[
        str | None, typer.Option("--color", help="Hex colour, e.g. '#e6194b'.")
    ] = None,
    attributes: Annotated[Path | None, typer.Option("--attributes", help=ATTRIBUTES_HELP)] = None,
    skeleton: Annotated[Path | None, typer.Option("--skeleton", help=SKELETON_HELP)] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    token = _type(class_type)
    new = DatasetClass(
        name=name,
        type=token,
        color=color,
        attributes=_attributes(attributes),
        skeleton=_skeleton(skeleton, token),
    )
    client = get_client(api_key)
    base = list(client.datasets.get(dataset).classes or [])
    if any(_same(c, name, token) for c in base):
        raise CliError(
            f"Dataset {dataset!r} already has the {token} class {name!r}.",
            exit_code=ExitCode.CONFLICT,
            code="class_exists",
            fix=f"Change it with `pictograph classes update {_quoted(dataset)} {_quoted(name)}`.",
        )
    now = _write(client, dataset, base, [*base, new])
    print_json(now[_find(now, name, token, dataset)].model_dump(mode="json", exclude_none=True))


@app.command(
    "update",
    help="Change a class's colour, attributes or skeleton. To rename or merge a class "
    "(its annotations too), use pictograph annotations rename-class / merge-class.",
)
def update_class(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    name: Annotated[str, typer.Argument(help="Class name.")],
    class_type: Annotated[
        str | None,
        typer.Option("--type", "-t", help="Which class, when the name has several types."),
    ] = None,
    color: Annotated[
        str | None, typer.Option("--color", help="Hex colour, e.g. '#e6194b'.")
    ] = None,
    attributes: Annotated[
        Path | None, typer.Option("--attributes", help=f"Replace them. {ATTRIBUTES_HELP}")
    ] = None,
    clear_attributes: Annotated[
        bool, typer.Option("--clear-attributes", help="Remove every attribute definition.")
    ] = False,
    skeleton: Annotated[
        Path | None, typer.Option("--skeleton", help=f"Replace it. {SKELETON_HELP}")
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    token = _type(class_type)
    if attributes is not None and clear_attributes:
        raise typer.BadParameter(
            "give --attributes or --clear-attributes, not both.", param_hint="--attributes"
        )
    if color is None and attributes is None and not clear_attributes and skeleton is None:
        raise typer.BadParameter(
            "nothing to change: give --color, --attributes, --clear-attributes or --skeleton.",
            param_hint="--color",
        )
    new_attributes = _attributes(attributes)
    client = get_client(api_key)
    base = list(client.datasets.get(dataset).classes or [])
    at = _find(base, name, token, dataset)
    current = base[at]
    changes: dict[str, Any] = {}
    if color is not None:
        changes["color"] = color
    if new_attributes is not None:
        changes["attributes"] = new_attributes
    if clear_attributes:
        changes["attributes"] = []
    if skeleton is not None:
        changes["skeleton"] = _skeleton(skeleton, _ALIASES.get(current.type or "", current.type))
    desired = [*base[:at], current.model_copy(update=changes), *base[at + 1 :]]
    now = _write(client, dataset, base, desired)
    stored_type = _ALIASES.get(current.type or "", current.type)
    print_json(
        now[_find(now, name, stored_type, dataset)].model_dump(mode="json", exclude_none=True)
    )


@app.command(
    "remove",
    help="Remove a class from a dataset's ontology; --with-annotations also deletes every "
    "annotation of it. The same as pictograph annotations delete-class.",
)
def remove_class(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    name: Annotated[str, typer.Argument(help="Class name.")],
    class_type: Annotated[
        str | None,
        typer.Option("--type", "-t", help="Remove only the class of this type."),
    ] = None,
    with_annotations: Annotated[
        bool,
        typer.Option("--with-annotations", help="Also delete every annotation of this class."),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    token = _type(class_type)
    scope = "and its annotations " if with_annotations else ""
    if not yes and not typer.confirm(f"Remove class {name!r} {scope}from dataset {dataset!r}?"):
        raise typer.Abort()
    client = get_client(api_key)
    project = client.datasets.get(dataset)
    result = client.annotations.delete_class(
        project.id, name, class_type=token, delete_annotations=with_annotations
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
