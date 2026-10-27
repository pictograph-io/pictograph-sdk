"""Tests for the ``pictograph images get`` command (CLI parity).

Self-contained: a :class:`~typer.testing.CliRunner` invokes the command group
in-process, the SDK client is a :class:`~unittest.mock.MagicMock` patched at the
``images`` command-module boundary, and ``~/.pictograph`` is redirected to a
``tmp_path`` so the suite never touches real config (mirrors ``test_models_cli``).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pictograph.cli._config import write_config
from pictograph.cli.commands.images import app
from pictograph.models.image import Image
from pictograph.resources.images import Images
from tests.unit.resources._orchestration import build, sibling_resources


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("PICTOGRAPH_API_KEY", raising=False)
    monkeypatch.setattr("pictograph.cli._config.CONFIG_DIR", fake_home / ".pictograph")
    monkeypatch.setattr(
        "pictograph.cli._config.CONFIG_PATH", fake_home / ".pictograph" / "config.toml"
    )
    return fake_home


def _image() -> Image:
    return Image(
        id="img-uuid-1",
        project_id="proj-uuid",
        filename="x.jpg",
        gcs_image_path="x",
        directory_path="/",
        status="new",
        annotation_count=0,
        file_size=100,
        image_url="https://x/img",
        created_at=datetime.now(timezone.utc),
    )


def _patch_client(client: MagicMock) -> Any:
    return patch("pictograph.cli.commands.images.get_client", return_value=client)


def test_list_command_outputs_images_and_threads_filters(
    runner: CliRunner, isolated_config: Path
) -> None:
    client = MagicMock()
    _unused_datasets_get = MagicMock(id="proj-uuid")
    pager = MagicMock()
    pager.all.return_value = [_image()]
    client.images.iter.return_value = pager
    with _patch_client(client):
        result = runner.invoke(
            app,
            [
                "list",
                "my-dataset",
                "--directory",
                "/train",
                "--status",
                "complete",
                "--limit",
                "5",
                "--api-key",
                "pk_live_x",
            ],
        )
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert isinstance(out, list)
    assert out[0]["id"] == "img-uuid-1"
    assert client.datasets.get.call_args[0][0] == "my-dataset"
    _, kwargs = client.images.iter.call_args
    assert kwargs["directory_path"] == "/train"
    assert kwargs["status"] == "complete"
    assert kwargs["include_archived"] is False
    assert kwargs["max_total"] == 5


def test_get_renders_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.images.get.return_value = _image()
    with _patch_client(client):
        res = runner.invoke(app, ["get", "road-signs", "img-uuid-1"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["id"] == "img-uuid-1"
    assert payload["filename"] == "x.jpg"
    client.images.get.assert_called_once_with("road-signs", "img-uuid-1")


def test_tag_adds_tags(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    _unused_datasets_get = MagicMock(id="proj-uuid")
    client.images.bulk_tag.return_value = 2
    with _patch_client(client):
        res = runner.invoke(app, ["tag", "MyDataset", "i1", "i2", "--tag", "wet", "-t", "night"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload == {"processed": 2, "tags": ["wet", "night"], "added": True}
    client.images.bulk_tag.assert_called_once_with(
        "MyDataset", ["i1", "i2"], ["wet", "night"], add=True
    )


def test_tag_remove_sets_add_false(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    _unused_datasets_get = MagicMock(id="proj-uuid")
    client.images.bulk_tag.return_value = 1
    with _patch_client(client):
        res = runner.invoke(app, ["tag", "MyDataset", "i1", "--tag", "wet", "--remove"])
    assert res.exit_code == 0, res.stdout
    client.images.bulk_tag.assert_called_once_with("MyDataset", ["i1"], ["wet"], add=False)


def test_tag_requires_at_least_one_tag(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    with _patch_client(client):
        res = runner.invoke(app, ["tag", "MyDataset", "i1"])
    assert res.exit_code != 0


def test_delete_is_permanent_by_default(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    with _patch_client(client):
        res = runner.invoke(app, ["delete", "road-signs", "img-uuid-1", "--yes"])
    assert res.exit_code == 0, res.stdout
    assert "Deleted image" in res.stdout
    client.images.delete.assert_called_once_with("road-signs", "img-uuid-1", permanent=True)


def test_delete_archive_flag_soft_deletes(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    with _patch_client(client):
        res = runner.invoke(app, ["delete", "road-signs", "img-uuid-1", "--archive", "--yes"])
    assert res.exit_code == 0, res.stdout
    assert "Archived image" in res.stdout
    client.images.delete.assert_called_once_with("road-signs", "img-uuid-1", permanent=False)


def test_review_approve_default(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.images.review.return_value = "complete"
    with _patch_client(client):
        res = runner.invoke(app, ["review", "road-signs", "img-1"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload == {"image": "img-1", "action": "approve", "status": "complete"}
    client.images.review.assert_called_once_with("road-signs", "img-1", "approve")


def test_review_request_changes_with_note(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.images.review.return_value = "annotate"
    with _patch_client(client):
        res = runner.invoke(
            app, ["review", "road-signs", "img-2", "--request-changes", "--note", "fix bbox"]
        )
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload == {"image": "img-2", "action": "request_changes", "status": "annotate"}
    client.images.review.assert_called_once_with(
        "road-signs", "img-2", "request_changes", note="fix bbox"
    )


def test_split_assigns(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.images.set_split.return_value = "train"
    with _patch_client(client):
        res = runner.invoke(app, ["split", "road-signs", "img-1", "train"])
    assert res.exit_code == 0, res.stdout
    assert json.loads(res.stdout) == {"image": "img-1", "split": "train"}
    client.images.set_split.assert_called_once_with("road-signs", "img-1", "train")


def test_split_none_clears(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.images.set_split.return_value = None
    with _patch_client(client):
        res = runner.invoke(app, ["split", "road-signs", "img-2", "none"])
    assert res.exit_code == 0, res.stdout
    client.images.set_split.assert_called_once_with("road-signs", "img-2", None)


# ───────────── upload-directory ─────────────


@pytest.fixture
def nested_tree(tmp_path: Path) -> Path:
    """Same basename in two subdirectories - the case a flat upload silently loses."""
    root = tmp_path / "tree"
    (root / "cars" / "red").mkdir(parents=True)
    (root / "cars" / "red" / "0001.jpg").write_bytes(b"\xff\xd8\xff" + b"a" * 10)
    (root / "trucks").mkdir()
    (root / "trucks" / "0001.jpg").write_bytes(b"\xff\xd8\xff" + b"b" * 10)
    return root


def _upload_directories(client: MagicMock) -> list[str]:
    return sorted(c.kwargs["directory_path"] for c in client.images.upload.call_args_list)


def _directory_upload_client() -> MagicMock:
    """A Client whose ``images`` is the REAL resource.

    ``upload-directory`` now goes through ``client.images.upload_from_directory``, so a
    fully-mocked client would assert nothing about the directory walk. Only the
    single-file ``upload`` is stubbed - the tree walking and directory mapping are the
    real thing.
    """
    client = MagicMock()
    client.images.upload.return_value = MagicMock(id="img-1")
    client.images = build(Images, client, own="images", delegate=["upload"])
    return client


def test_upload_directory_preserves_the_tree_by_default(
    runner: CliRunner, isolated_config: Path, nested_tree: Path
) -> None:
    """The CLI twin of the web app's "Add -> Directory": full structure, no flags needed."""
    write_config(api_key="pk_live_x")
    client = _directory_upload_client()
    with _patch_client(client), sibling_resources(client):
        res = runner.invoke(app, ["upload-directory", "ds", str(nested_tree)])
    assert res.exit_code == 0, res.stdout

    # Both 0001.jpg files uploaded, into their own recreated directories.
    assert _upload_directories(client) == ["/cars/red", "/trucks"]
    assert json.loads(res.stdout)["images_uploaded"] == 2


def test_upload_directory_flat_puts_everything_at_root(
    runner: CliRunner, isolated_config: Path, nested_tree: Path
) -> None:
    write_config(api_key="pk_live_x")
    client = _directory_upload_client()
    with _patch_client(client), sibling_resources(client):
        res = runner.invoke(app, ["upload-directory", "ds", str(nested_tree), "--flat"])
    assert res.exit_code == 0, res.stdout
    assert _upload_directories(client) == ["/", "/"]


def test_upload_directory_by_class_uses_only_the_first_level(
    runner: CliRunner, isolated_config: Path, nested_tree: Path
) -> None:
    """ImageFolder mode: cars/red/x.jpg collapses onto /cars."""
    write_config(api_key="pk_live_x")
    client = _directory_upload_client()
    with _patch_client(client), sibling_resources(client):
        res = runner.invoke(app, ["upload-directory", "ds", str(nested_tree), "--by-class"])
    assert res.exit_code == 0, res.stdout
    assert _upload_directories(client) == ["/cars", "/trucks"]
