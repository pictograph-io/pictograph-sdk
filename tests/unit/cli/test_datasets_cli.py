"""Tests for ``pictograph datasets export`` - server-side export via Pictograph's
OWN converters (``client.exports``), with NO third-party dependency.

Self-contained: a ``CliRunner`` invokes the command in-process with the SDK client
patched to a ``MagicMock`` (no network), and ``isolated_config`` redirects HOME so
the suite never touches real config (mirrors ``test_exports_cli.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pictograph.cli._config import write_config
from pictograph.cli.commands.datasets import app


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


def _patch_client(client: MagicMock) -> Any:
    return patch("pictograph.cli.commands.datasets.get_client", return_value=client)


def _client_with_export(export_name: str) -> MagicMock:
    """A MagicMock client whose exports.create() returns an export with the given name."""
    client = MagicMock()
    export = MagicMock()
    export.name = export_name
    client.exports.create.return_value = export
    return client


def test_export_coco(runner: CliRunner, isolated_config: Path, tmp_path: Path) -> None:
    write_config(api_key="pk_live_x")
    client = _client_with_export("road-signs-coco")
    out = tmp_path / "coco-out"

    with _patch_client(client):
        res = runner.invoke(app, ["export", "road-signs", "--format", "coco", "-o", str(out)])

    assert res.exit_code == 0, res.stdout
    client.exports.create.assert_called_once_with(
        "road-signs", "road-signs-coco", format="coco", include_images=False
    )
    client.exports.download.assert_called_once()
    payload = json.loads(res.stdout)
    assert payload == {
        "dataset": "road-signs",
        "format": "coco",
        "export": "road-signs-coco",
        "output": str(out / "road-signs-coco.zip"),
    }


def test_export_yolo_include_images(
    runner: CliRunner, isolated_config: Path, tmp_path: Path
) -> None:
    write_config(api_key="pk_live_x")
    client = _client_with_export("d-yolo")

    with _patch_client(client):
        res = runner.invoke(
            app, ["export", "d", "--format", "yolo", "--include-images", "-o", str(tmp_path / "y")]
        )

    assert res.exit_code == 0, res.stdout
    client.exports.create.assert_called_once_with("d", "d-yolo", format="yolo", include_images=True)


def test_export_pascal_voc_format_alias(
    runner: CliRunner, isolated_config: Path, tmp_path: Path
) -> None:
    write_config(api_key="pk_live_x")
    client = _client_with_export("d-pascal_voc")

    with _patch_client(client):
        res = runner.invoke(app, ["export", "d", "-f", "pascal-voc", "-o", str(tmp_path / "p")])

    assert res.exit_code == 0, res.stdout
    # "pascal-voc" normalizes to the ExportFormat literal "pascal_voc".
    _, kwargs = client.exports.create.call_args
    assert kwargs["format"] == "pascal_voc"


def test_export_unsupported_format_exits(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()

    with _patch_client(client):
        res = runner.invoke(app, ["export", "d", "--format", "tfrecord"])

    assert res.exit_code == 1
    client.exports.create.assert_not_called()


# ── datasets insights ──────────────────────────────────────────────────────

_INSIGHTS_PAYLOAD = {
    "total_images": 162,
    "total_annotations": 1992,
    "annotated_images": 162,
    "unannotated_images": 0,
    "avg_annotations_per_image": 12.3,
    "total_bytes": 3394727,
    "status_counts": {"new": 0, "annotate": 0, "review": 0, "complete": 162},
    "class_annotation_counts": {"player": 1756, "ref": 140, "ball": 96},
    "class_image_counts": {"player": 162, "ref": 140, "ball": 96},
    "type_counts": {"bbox": 1992},
    "annotation_density": {"3-5": 7, "6-10": 47, "11-20": 108},
    "dimensions": {
        "min_width": 398,
        "max_width": 1280,
        "avg_width": 643,
        "min_height": 224,
        "max_height": 720,
        "avg_height": 362,
        "orientation": {"landscape": 162, "portrait": 0, "square": 0},
        "sizes": [{"w": 398, "h": 224, "count": 117}],
        "distinct_size_count": 2,
        "images_with_dimensions": 162,
        "images_missing_dimensions": 0,
    },
}


def _client_with_insights() -> MagicMock:
    from pictograph.models.insights import DatasetInsights

    client = MagicMock()
    client.datasets.insights.return_value = DatasetInsights.model_validate(_INSIGHTS_PAYLOAD)
    return client


def test_insights_summary(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = _client_with_insights()

    with _patch_client(client):
        res = runner.invoke(app, ["insights", "futbol"])

    assert res.exit_code == 0, res.stdout
    client.datasets.insights.assert_called_once_with("futbol")
    # headline + class-balance table both render (class names present)
    assert "162" in res.stdout
    assert "Class balance" in res.stdout
    assert "player" in res.stdout


def test_insights_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = _client_with_insights()

    with _patch_client(client):
        res = runner.invoke(app, ["insights", "futbol", "--json"])

    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["total_images"] == 162
    assert payload["class_annotation_counts"]["player"] == 1756


def test_import_coco(runner: CliRunner, isolated_config: Path, tmp_path: Path) -> None:
    """The CLI wires args to the (separately-tested) import pipeline and prints its report."""
    from pictograph.resources.annotations import AnnotationImportReport

    write_config(api_key="pk_live_x")
    coco = tmp_path / "instances.json"
    coco.write_text('{"images": [], "categories": [], "annotations": []}', encoding="utf-8")
    report = AnnotationImportReport(
        dataset_name="road-signs",
        images_matched=2,
        images_saved=2,
        annotations_saved=5,
        unmatched_files=["z.jpg"],
    )
    client = MagicMock()
    client.annotations.import_coco.return_value = report

    with _patch_client(client):
        res = runner.invoke(app, ["import-coco", "road-signs", str(coco)])

    assert res.exit_code == 0, res.stdout
    client.annotations.import_coco.assert_called_once_with(
        "road-signs", coco, create_missing_classes=True
    )
    payload = json.loads(res.stdout)
    assert payload["images_saved"] == 2
    assert payload["annotations_saved"] == 5
    assert payload["unmatched_files"] == ["z.jpg"]
    assert payload["success"] is False


def test_import_coco_no_create_classes(
    runner: CliRunner, isolated_config: Path, tmp_path: Path
) -> None:
    from pictograph.resources.annotations import AnnotationImportReport

    write_config(api_key="pk_live_x")
    coco = tmp_path / "instances.json"
    coco.write_text("{}", encoding="utf-8")
    client = MagicMock()
    client.annotations.import_coco.return_value = AnnotationImportReport(
        dataset_name="d", images_saved=1
    )
    with _patch_client(client):
        res = runner.invoke(app, ["import-coco", "d", str(coco), "--no-create-classes"])

    assert res.exit_code == 0, res.stdout
    assert client.annotations.import_coco.call_args.kwargs["create_missing_classes"] is False


def test_import_coco_missing_file(runner: CliRunner, isolated_config: Path, tmp_path: Path) -> None:
    write_config(api_key="pk_live_x")
    with _patch_client(MagicMock()):
        res = runner.invoke(app, ["import-coco", "road-signs", str(tmp_path / "nope.json")])
    assert res.exit_code == 1
    assert "not found" in res.stdout.lower() or "not found" in (res.stderr or "").lower()


# ───────────── duplicates (near-duplicate data curation) ─────────────

from pictograph.models.near_duplicates import NearDuplicatesResult  # noqa: E402

_DUP_PAYLOAD = {
    "groups": [
        {
            "members": [
                {
                    "id": "a",
                    "filename": "a.jpg",
                    "virtual_directory_path": "/",
                    "status": "new",
                    "annotation_count": 3,
                },
                {
                    "id": "b",
                    "filename": "b.jpg",
                    "virtual_directory_path": "/",
                    "status": "new",
                    "annotation_count": 0,
                },
            ],
            "size": 2,
            "max_similarity": 0.98,
        }
    ],
    "group_count": 1,
    "duplicate_image_count": 2,
    "redundant_count": 1,
    "analyzed": 50,
    "total_images": 50,
    "sample_limit": 1000,
    "sample_capped": False,
    "pairs_capped": False,
    "threshold": 0.92,
}


def _client_with_dups() -> MagicMock:
    client = MagicMock()
    client.datasets.near_duplicates.return_value = NearDuplicatesResult.model_validate(_DUP_PAYLOAD)
    return client


def test_duplicates_summary(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = _client_with_dups()
    with _patch_client(client):
        res = runner.invoke(app, ["duplicates", "ds-123", "-t", "0.9", "--sample", "500"])
    assert res.exit_code == 0, res.stdout
    client.datasets.near_duplicates.assert_called_once_with(
        "ds-123", threshold=0.9, sample=500, directory_path=None
    )
    assert "1 duplicate groups" in res.stdout
    assert "1 redundant" in res.stdout


def test_duplicates_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = _client_with_dups()
    with _patch_client(client):
        res = runner.invoke(app, ["duplicates", "ds-123", "--json"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["group_count"] == 1
    assert payload["redundant_count"] == 1
