"""Top-level Typer app - entry point for the ``pictograph`` CLI."""

from __future__ import annotations

import sys
from typing import Annotated

import typer

from pictograph._version import __version__
from pictograph.cli._config import set_flag_profile
from pictograph.cli._errors import PictographCli
from pictograph.cli.commands import (
    agents,
    annotation_comments,
    annotations as annotations_cmd,
    api_keys,
    augment,
    auto_annotate,
    classes,
    connectors,
    credits,
    datasets,
    deployments,
    directories,
    explore,
    exports,
    images,
    init,
    jobs,
    login,
    metrics,
    models,
    notifications,
    organizations,
    profiles,
    sam3,
    search,
    tasks,
    tile,
    tools,
    train,
    video,
    webhooks,
    workflows,
)

app = typer.Typer(
    cls=PictographCli,
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
app.add_typer(
    classes.app,
    name="classes",
    help="A dataset's classes (its ontology): list, add, update, remove.",
)
app.add_typer(
    annotation_comments.app,
    name="annotation-comments",
    help="Notes on annotations - list, add, reply, edit, resolve, delete.",
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
    sam3.app,
    name="sam3",
    help="SAM3 on any image - a file, URL, stdin, a video's frames or a dataset's images. Nothing is saved.",
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
app.add_typer(
    explore.app,
    name="explore",
    help="The public catalog: datasets and models any organization published.",
)
app.add_typer(
    jobs.app,
    name="jobs",
    help="One view of every job: list, get, wait for and cancel training, imports, exports, ...",
)
app.add_typer(
    tools.app,
    name="tools",
    help="Your vision tools (SAM3, models, deployments, workflows) and how to call each.",
)
app.add_typer(agents.app, name="agents", help="Install Skill, export tools.json.")
app.add_typer(
    api_keys.app,
    name="api-keys",
    help="List / create / revoke API keys - mint a narrowly-scoped key for an agent.",
)
app.add_typer(
    profiles.app,
    name="profiles",
    help="The profiles of ~/.pictograph/config.toml: one per API key (organization).",
)

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
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile",
            help="Run as this profile of ~/.pictograph/config.toml - one per API key "
            "(organization), saved with `pictograph login --profile <name>`. Also "
            "PICTOGRAPH_PROFILE.",
        ),
    ] = None,
) -> None:
    """Root callback: the eager --version flag, and the profile every command runs as."""
    del version  # answered (and exited on) by its eager callback, before this body runs
    try:
        set_flag_profile(profile)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--profile") from None


def main(argv: list[str] | None = None) -> int:
    """The ``pictograph`` script: run the CLI and return its exit code.

    Every failure is rendered and mapped to its exit code by :class:`PictographCli`.
    """
    code: int = app(args=argv, prog_name="pictograph", standalone_mode=False)
    return code


if __name__ == "__main__":
    sys.exit(main())
