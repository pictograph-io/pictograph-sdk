"""``pictograph directories {list,tree,stats,delete}`` - inspect + clean up a dataset's directory tree."""

from __future__ import annotations

from typing import Annotated

import typer

from pictograph.cli._client import get_client
from pictograph.cli._format import print_json, print_table

app = typer.Typer(no_args_is_help=True)


@app.command("list", help="List a dataset's directories (optionally direct children of a path).")
def list_directories(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    parent_path: Annotated[
        str | None, typer.Option("--parent", help="Only direct children of this path ('' = root).")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    directories = client.directories.list(dataset, parent_path=parent_path)
    if json_output:
        print_json([f.model_dump(mode="json", exclude_none=True) for f in directories])
        return
    rows = [{"full_path": f.full_path, "id": f.id, "images": f.image_count} for f in directories]
    print_table(rows, title=f"Directories ({len(rows)})")


@app.command("tree", help="Print a dataset's hierarchical directory tree.")
def directory_tree(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    print_json([n.model_dump(mode="json") for n in client.directories.tree(dataset)])


@app.command("stats", help="Image statistics for a directory (and its subdirectories).")
def directory_stats(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    directory: Annotated[str, typer.Argument(help="Directory path, e.g. /train/cars.")],
    no_subdirectories: Annotated[
        bool, typer.Option("--no-subdirectories", help="Exclude subdirectory statistics.")
    ] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    stats = client.directories.stats(
        dataset, directory, include_subdirectories=not no_subdirectories
    )
    print_json(stats.model_dump(mode="json"))


@app.command(
    "delete",
    help="Delete a directory (empty-only unless --cascade moves its images to the parent).",
)
def delete_directory(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    directory: Annotated[str, typer.Argument(help="Directory path, e.g. /train/cars.")],
    cascade: Annotated[
        bool,
        typer.Option(
            "--cascade", help="Move contained images to the parent directory, then delete."
        ),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(f"Delete directory {directory!r} in {dataset!r}?"):
        raise typer.Abort()
    client = get_client(api_key)
    client.directories.delete(dataset, directory, cascade=cascade)
    print_json({"dataset": dataset, "directory": directory, "deleted": True})


@app.command("create", help="Create a virtual directory (idempotent; parents auto-created).")
def create_directory(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    directory_path: Annotated[str, typer.Argument(help="Full path, e.g. /train/positive.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    directory = client.directories.create(dataset, directory_path)
    print_json(directory.model_dump(mode="json", exclude_none=True))


@app.command("rename", help="Rename a directory - descendants and contained images move with it.")
def rename_directory(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    directory: Annotated[str, typer.Argument(help="Directory path, e.g. /train/cars.")],
    new_name: Annotated[str, typer.Argument(help="New name (single path segment).")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    renamed = client.directories.rename(dataset, directory, new_name)
    print_json(renamed.model_dump(mode="json", exclude_none=True))
