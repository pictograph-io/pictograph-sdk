"""``pictograph agents {install-skill,export-tools}``."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated, Literal

import typer

from pictograph._path_safety import safe_path_component
from pictograph.agents import Toolkit
from pictograph.cli._format import print_json, print_table
from pictograph.skills import skill_path

app = typer.Typer(no_args_is_help=True)

InstallTarget = Literal["claude-code", "claude-ai", "both"]


@app.command(
    "install-skill",
    help="Copy the bundled pictograph-cv Skill into ~/.claude/skills/ (Claude Code) "
    "and/or zip it for upload to claude.ai.",
)
def install_skill(
    target: Annotated[
        str,
        typer.Option(
            "--target",
            help="claude-code / claude-ai / both",
        ),
    ] = "claude-code",
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="Override the destination directory (default: ~/.claude/skills/).",
        ),
    ] = None,
    skill_name: Annotated[
        str,
        typer.Option("--skill", help="Skill name (default: pictograph-cv)."),
    ] = "pictograph-cv",
) -> None:
    if target not in ("claude-code", "claude-ai", "both"):
        typer.echo(f"Invalid target: {target!r}", err=True)
        raise typer.Exit(2)
    # A skill name is a slug, and it is about to be joined to ~/.claude/skills
    # and handed to shutil.rmtree. Unvalidated, `--skill ..` resolves dest to
    # ~/.claude and deletes the user's entire Claude configuration; `--skill
    # ../../x` reaches further still. Reject anything that is not one component.
    if skill_name != safe_path_component(skill_name):
        typer.echo(
            f"Invalid skill name: {skill_name!r}. Expected a plain name such as "
            "'pictograph-cv' - no path separators or '..'.",
            err=True,
        )
        raise typer.Exit(2)
    src = skill_path(skill_name)

    results: dict[str, str] = {}

    if target in ("claude-code", "both"):
        dest_root = output_dir or (Path.home() / ".claude" / "skills")
        dest = dest_root / skill_name
        dest_root.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        results["claude_code_path"] = str(dest)

    if target in ("claude-ai", "both"):
        zip_dest = (output_dir or Path.cwd()) / f"{skill_name}.zip"
        # shutil.make_archive expects basename without extension.
        archive = shutil.make_archive(
            base_name=str(zip_dest.with_suffix("")),
            format="zip",
            root_dir=str(src.parent),
            base_dir=skill_name,
        )
        results["claude_ai_zip"] = archive

    print_json({"installed": True, "skill": skill_name, **results})


@app.command(
    "export-tools",
    help="Emit the agent tool registry as JSON Schema (drop-in for tools.json).",
)
def export_tools(
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Write to file instead of stdout.",
        ),
    ] = None,
) -> None:
    """Build a Toolkit on the fly - no API key required for schema export."""
    from unittest.mock import MagicMock

    # We don't actually call any handlers here, just emit the schemas.
    toolkit = Toolkit(MagicMock())
    schema = toolkit.as_json_schema()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        typer.echo(f"Wrote {len(schema)} tools to {output}")
    else:
        print_json(schema)


@app.command("list-tools", help="List the names + descriptions of every agent tool.")
def list_tools() -> None:
    from pictograph.agents import REGISTRY

    rows = [
        {
            "name": t.name,
            "role": t.required_role,
            "cost (USD)": f"${t.cost_micro_usd / 1_000_000:.4f}",
            "idempotent": t.idempotent,
            "description": (t.description[:80] + "…" if len(t.description) > 80 else t.description),
        }
        for t in REGISTRY
    ]
    print_table(rows, title=f"Agent tools ({len(rows)})")
