"""Tests for ``pictograph search {similar,tags}``.

Self-contained: a real CliRunner drives the standalone ``search`` Typer app,
with ``get_client`` patched at the command-module boundary and the SDK client
replaced by a MagicMock. One happy-path test per command.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pictograph.cli._config import write_config
from pictograph.cli.commands.search import app
from pictograph.models.search import SimilarImage, TaggedImage


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ~/.pictograph/* to tmp_path so tests don't touch real config."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("PICTOGRAPH_API_KEY", raising=False)
    monkeypatch.setattr(
        "pictograph.cli._config.CONFIG_DIR",
        fake_home / ".pictograph",
    )
    monkeypatch.setattr(
        "pictograph.cli._config.CONFIG_PATH",
        fake_home / ".pictograph" / "config.toml",
    )
    return fake_home


def _similar(image_id: str = "img-2", similarity: float = 0.87) -> SimilarImage:
    return SimilarImage(
        id=image_id,
        filename="cat.jpg",
        status="complete",
        annotation_count=3,
        similarity=similarity,
    )


def _tagged(image_id: str = "img-9") -> TaggedImage:
    return TaggedImage(
        id=image_id,
        project_id="proj-1",
        filename="road.jpg",
        status="annotate",
        annotation_count=1,
    )


def test_search_similar_renders_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.search.by_similarity.return_value = [_similar("a"), _similar("b")]
    with patch("pictograph.cli.commands.search.get_client", return_value=client):
        res = runner.invoke(
            app, ["similar", "road-signs", "stop.jpg", "--threshold", "0.8", "--json"]
        )
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert {row["id"] for row in payload} == {"a", "b"}
    client.search.by_similarity.assert_called_once_with(
        "road-signs", "stop.jpg", threshold=0.8, limit=50, directory_path=None
    )


def test_search_tags_renders_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.search.by_tag.return_value = [_tagged("a"), _tagged("b")]
    with patch("pictograph.cli.commands.search.get_client", return_value=client):
        res = runner.invoke(
            app,
            ["tags", "--object", "car", "--scene", "outdoor", "--dataset", "road-signs", "--json"],
        )
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert {row["id"] for row in payload} == {"a", "b"}
    client.search.by_tag.assert_called_once_with(
        objects=["car"],
        scenes=["outdoor"],
        attributes=None,
        dataset_name="road-signs",
        limit=50,
        offset=0,
    )
