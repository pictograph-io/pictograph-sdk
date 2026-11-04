"""An image is addressed ONE way on every surface: its path in its dataset.

REFERENCE_LANGUAGE R2b: an image's address is its filename, preceded by its directory
when it is not at the dataset root (``val/x.jpg``). Owner, 2026-11-02: "the filename is
supposed to be the address" / "the standard is NOT the uuid".

Before this was pinned, one image argument meant different things on different calls:
``images.get`` searched every directory for a bare filename and refused a path, while
``download`` / ``delete`` / ``review`` / ``set_split`` took the path and refused a bare
filename of an image in a directory; ``annotations.get`` / ``save`` / ``delete`` took the
directory a second way (``directory=``); and the CLI and agent tools described the image
in five different wordings, three of them offering a uuid. These hold every surface to
the one form, so a new method cannot drift from it without a red test.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import click
import pytest
import typer

from pictograph import Client
from pictograph.agents import _args
from pictograph.aio import AsyncClient
from pictograph.cli._app import app
from pictograph.resources._resolve import IMAGE_PATH_HELP

SRC = Path(__file__).resolve().parents[2] / "src" / "pictograph"

#: The one method whose ``directory`` is not part of the image's address: it widens or
#: narrows the SEARCH (the images compared against), the source image is still a path.
_DIRECTORY_IS_NOT_THE_ADDRESS = {"search.by_similarity"}


def _image_methods(client: Any) -> list[tuple[str, inspect.Signature]]:
    found = []
    for rname, resource in sorted(vars(client).items()):
        if not hasattr(resource, "_transport"):
            continue
        for mname, method in inspect.getmembers(resource, predicate=inspect.ismethod):
            if mname.startswith("_"):
                continue
            sig = inspect.signature(method)
            # An image named INSIDE a dataset. (`models.predict(image)` / `sam3.predict`
            # take a local image file - not an address.)
            if {"dataset", "image"} <= set(sig.parameters):
                found.append((f"{rname}.{mname}", sig))
    return found


@pytest.mark.parametrize("make", [Client, AsyncClient], ids=["sync", "async"])
def test_no_image_method_takes_the_directory_a_second_way(make: Any) -> None:
    client = make(api_key="pk_live_contract")
    methods = _image_methods(client)
    assert len(methods) >= 18, "the walk must reach the per-image methods"
    offenders = [
        name
        for name, sig in methods
        if name not in _DIRECTORY_IS_NOT_THE_ADDRESS
        and {"directory", "directory_path", "image_id"} & set(sig.parameters)
    ]
    assert offenders == [], (
        f"an image's directory is part of its path, not an argument: {offenders}"
    )


def test_every_agent_tool_describes_an_image_the_same_way() -> None:
    fields = [
        (cls.__name__, info)
        for cls in vars(_args).values()
        if isinstance(cls, type) and issubclass(cls, _args.BaseModel) and cls is not _args.BaseModel
        for name, info in getattr(cls, "model_fields", {}).items()
        if name == "image" and "dataset" in cls.model_fields
    ]
    assert len(fields) >= 9
    drifted = [name for name, info in fields if IMAGE_PATH_HELP not in (info.description or "")]
    assert drifted == []
    assert not [name for name, info in fields if "uuid" in (info.description or "").lower()]


def _cli_image_params() -> list[tuple[str, click.Parameter]]:
    root = typer.main.get_command(app)
    out: list[tuple[str, click.Parameter]] = []

    def walk(cmd: click.Command, path: str) -> None:
        names = {p.name for p in cmd.params}
        for p in cmd.params:
            # A dataset's image is the positional argument beside the dataset.
            # (`workflows run --image` is an OPTION naming a local file to run on.)
            if p.name == "image" and "dataset" in names and p.param_type_name == "argument":
                out.append((path, p))
        for name, sub in getattr(cmd, "commands", {}).items():
            walk(sub, f"{path} {name}".strip())

    walk(root, "")
    return out


def test_every_cli_image_argument_says_the_same_thing() -> None:
    params = _cli_image_params()
    assert len(params) >= 10, "the walk must reach the per-image commands"
    drifted = [path for path, p in params if IMAGE_PATH_HELP not in (getattr(p, "help", "") or "")]
    assert drifted == []


def test_no_annotations_command_takes_a_directory_option() -> None:
    root = typer.main.get_command(app)
    group = root.commands["annotations"]  # type: ignore[attr-defined]
    for name in ("get", "save", "delete"):
        opts = {o for p in group.commands[name].params for o in getattr(p, "opts", [])}
        assert "--directory" not in opts, name


def test_no_resource_shape_tests_an_image_as_a_uuid() -> None:
    """Detecting a uuid in the image argument is what made it two things at once."""
    pattern = re.compile(r"looks_like_id\(\s*image\b")
    hits = [
        str(path.relative_to(SRC))
        for folder in ("resources", "aio/resources")
        for path in sorted((SRC / folder).glob("*.py"))
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert hits == []


def test_every_image_result_hands_back_its_address() -> None:
    """A result carrying an image's filename and directory exposes the address to pass on:
    ``client.images.review(ds, img.path, ...)`` for an image from ``list()`` or a search."""
    import pydantic

    from pictograph import models

    carriers = [
        cls
        for cls in {
            obj
            for mod in vars(models).values()
            for obj in (vars(mod).values() if inspect.ismodule(mod) else [mod])
            if isinstance(obj, type) and issubclass(obj, pydantic.BaseModel)
        }
        if {"filename", "directory_path"} <= set(cls.model_fields)
    ]
    assert len(carriers) >= 5
    assert [
        c.__name__ for c in carriers if not isinstance(getattr(c, "path", None), property)
    ] == []

    from pictograph.models.image import Image

    img = Image.model_validate(
        {
            "id": "i",
            "filename": "x.jpg",
            "directory_path": "/val",
            "created_at": "2026-01-01T00:00:00Z",
        }
    )
    assert img.path == "val/x.jpg"
    assert Image.model_validate({**img.model_dump(), "directory_path": "/"}).path == "x.jpg"
    assert "path" not in img.model_dump(), "an address, not a wire field"
