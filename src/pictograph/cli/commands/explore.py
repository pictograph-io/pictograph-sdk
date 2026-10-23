"""``pictograph explore {datasets,dataset,models}`` - the public catalog.

Every organization can publish datasets and models to Explore. These read that catalog
- what the app's Explore pages show - so an agent can find a dataset to start from or
a model to build on. A public dataset or model is named by its organization and its
own slug or name: ``pictograph-research/coco-5k``. ``pictograph datasets import-public``
copies a dataset into your organization; ``pictograph models fork`` copies a model.
"""

from __future__ import annotations

from typing import Annotated

import typer

from pictograph.cli._client import get_client
from pictograph.cli._errors import command_group
from pictograph.cli._format import print_json, print_table

app = command_group(no_args_is_help=True)

_DATASET_SORTS = ("recent", "popular", "trending", "alphabetical")
_MODEL_SORTS = ("recent", "popular", "alphabetical")
_TASKS = (
    "object_detection",
    "semantic_segmentation",
    "instance_segmentation",
    "classification",
    "keypoint_detection",
)


def _choice(value: str, allowed: tuple[str, ...], flag: str) -> str:
    if value not in allowed:
        raise typer.BadParameter(f"must be one of: {', '.join(allowed)}", param_hint=flag)
    return value


@app.command(
    "datasets",
    help=(
        "List public datasets. The REF column is what `datasets import-public` takes; "
        "--json prints every field."
    ),
)
def list_datasets(
    search: Annotated[
        str | None,
        typer.Option("--search", "-s", help="Match the name or what the images show."),
    ] = None,
    class_name: Annotated[
        str | None, typer.Option("--class", help="Only datasets that have this class.")
    ] = None,
    sort: Annotated[
        str, typer.Option("--sort", help="recent, popular, trending or alphabetical.")
    ] = "recent",
    limit: Annotated[
        int,
        typer.Option(
            "--limit", "-n", min=1, help="How many to return (pages through the API as needed)."
        ),
    ] = 50,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    _choice(sort, _DATASET_SORTS, "--sort")
    client = get_client(api_key)
    rows = list(
        client.explore.iter_datasets(
            search=search,
            class_name=class_name,
            sort=sort,  # type: ignore[arg-type]
            page_size=min(limit, 100),
            max_total=limit,
        )
    )
    if json_output:
        print_json([r.model_dump(mode="json", exclude_none=True) | {"ref": r.ref} for r in rows])
        return
    print_table(
        [
            {
                "ref": r.ref,
                "images": r.image_count,
                "annotations": r.annotation_count,
                "types": ", ".join(r.annotation_types),
                "license": r.license_id or "-",
                "stars": r.star_count,
            }
            for r in rows
        ],
        title=f"Public datasets ({len(rows)}) - `pictograph datasets import-public <ref>` copies one",
        no_wrap=("ref",),
    )


@app.command(
    "dataset",
    help="Show one public dataset: its card, license, classes and counts. REF is org/dataset.",
)
def get_dataset(
    ref: Annotated[
        str,
        typer.Argument(
            help="org/dataset - the organization's slug and the dataset's slug or name."
        ),
    ],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    dataset = client.explore.dataset(ref)
    print_json(dataset.model_dump(mode="json", exclude_none=True) | {"ref": dataset.ref})


@app.command(
    "models",
    help="List public models. The REF column (org/model) is what `models fork` takes.",
)
def list_models(
    search: Annotated[
        str | None, typer.Option("--search", "-s", help="Match the model's name.")
    ] = None,
    task: Annotated[
        str | None,
        typer.Option(
            "--task",
            help=(
                "object_detection, semantic_segmentation, instance_segmentation, "
                "classification or keypoint_detection."
            ),
        ),
    ] = None,
    sort: Annotated[
        str, typer.Option("--sort", help="recent, popular or alphabetical.")
    ] = "recent",
    limit: Annotated[
        int,
        typer.Option(
            "--limit", "-n", min=1, help="How many to return (pages through the API as needed)."
        ),
    ] = 50,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    _choice(sort, _MODEL_SORTS, "--sort")
    if task is not None:
        _choice(task, _TASKS, "--task")
    client = get_client(api_key)
    rows = list(
        client.explore.iter_models(
            search=search,
            model_type=task,  # type: ignore[arg-type]
            sort=sort,  # type: ignore[arg-type]
            page_size=min(limit, 100),
            max_total=limit,
        )
    )
    if json_output:
        print_json([r.model_dump(mode="json", exclude_none=True) | {"ref": r.ref} for r in rows])
        return
    print_table(
        [
            {
                "ref": r.ref,
                "task": r.model_type.replace("_", " "),
                "architecture": r.architecture or "-",
                "classes": len(r.classes),
                "license": r.license_id or "-",
                "stars": r.star_count,
            }
            for r in rows
        ],
        title=f"Public models ({len(rows)}) - `pictograph models fork <ref>` copies one",
        no_wrap=("ref",),
    )
