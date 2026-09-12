"""Top-level Typer app - entry point for the ``pictograph`` CLI."""

from __future__ import annotations

import sys
from typing import Annotated

import typer

from pictograph._version import __version__
from pictograph.cli._format import print_error
from pictograph.cli.commands import (
    agents,
    annotations as annotations_cmd,
    augment,
    auto_annotate,
    connectors,
    credits,
    datasets,
    deployments,
    directories,
    exports,
    images,
    init,
    login,
    metrics,
    models,
    notifications,
    organizations,
    search,
    tasks,
    tile,
    train,
    video,
    webhooks,
    workflows,
)
from pictograph.exceptions import PictographError

app = typer.Typer(
    name="pictograph",
    help=("Pictograph CLI - agent-native CV annotation platform. Mirrors the Python SDK 1:1."),
    no_args_is_help=True,
    add_completion=True,
    pretty_exceptions_show_locals=False,
)

app.add_typer(datasets.app, name="datasets", help="Dataset CRUD + bulk download.")
app.add_typer(images.app, name="images", help="Single-image upload / download / delete.")
app.add_typer(
    annotations_cmd.app, name="annotations", help="Read / save / delete annotations on an image."
)
app.add_typer(train.app, name="train", help="Start / monitor / cancel training runs.")
app.add_typer(
    augment.app, name="augment", help="Generate an augmented version of a dataset (flip/rotate/…)."
)
app.add_typer(
    tile.app,
    name="tile",
    help="Slice a dataset into an NxM grid of tiles (small-object detection).",
)
app.add_typer(models.app, name="models", help="List + download trained models.")
app.add_typer(
    metrics.app,
    name="metrics",
    help="Offline detection metrics (P/R/F1/mAP) from JSON.",
)
app.add_typer(
    auto_annotate.app,
    name="auto-annotate",
    help="SAM3 / trained-model inference (point/box/text/batch).",
)
app.add_typer(
    deployments.app, name="deployments", help="Model inference deployments + direct predict."
)
app.add_typer(workflows.app, name="workflows", help="Node-graph workflows: run / poll / cancel.")
app.add_typer(
    organizations.app,
    name="organizations",
    help="Org info; manage members + invites.",
)
app.add_typer(connectors.app, name="connectors", help="Import datasets from V7 / Roboflow.")
app.add_typer(exports.app, name="exports", help="Create / list / download / delete exports.")
app.add_typer(
    directories.app, name="directories", help="Inspect a dataset's virtual directory tree."
)
app.add_typer(search.app, name="search", help="Similarity + tag search across a dataset.")
app.add_typer(video.app, name="video", help="Upload video + extract frames into a dataset.")
app.add_typer(webhooks.app, name="webhooks", help="Outbound webhook endpoints + deliveries.")
app.add_typer(credits.app, name="credits", help="Balance / history / cost estimates.")
app.add_typer(
    tasks.app, name="tasks", help="Annotation tasks + per-annotator contribution tracking."
)
app.add_typer(
    notifications.app, name="notifications", help="List / count / acknowledge org notifications."
)
app.add_typer(agents.app, name="agents", help="Install Skill, export tools.json.")

# Top-level commands (no subcommand group).
app.command("init", help="Drop an AGENTS.md template into the current directory.")(init.command)
app.command("login", help="Interactive API-key setup (writes ~/.pictograph/config.toml).")(
    login.command
)


def _version_callback(show: bool) -> None:
    if show:
        typer.echo(f"pictograph {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show CLI version and exit.",
        ),
    ] = False,
) -> None:
    """Root callback - Typer requires this for the eager --version flag."""


def main() -> int:
    """Module entry point - wraps :func:`app` with one-line error rendering."""
    try:
        app()
        return 0
    except PictographError as exc:
        print_error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
