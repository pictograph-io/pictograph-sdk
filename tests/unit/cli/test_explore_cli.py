"""``pictograph explore`` and ``pictograph datasets import-public / import-status`` (B533 #8).

The SDK client is a :class:`~unittest.mock.MagicMock` patched at each command module's
boundary; what it returns is the REAL model parsed from the payloads the resource tests
use, so what a command prints is what an agent reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pictograph.cli._app import app as root
from pictograph.cli._config import write_config
from pictograph.cli._errors import ExitCode
from pictograph.models.explore import PublicDataset, PublicDatasetImport, PublicModel
from tests.unit.resources.test_explore import IMPORT_ID, dataset_row, import_row, model_row


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
    write_config(api_key="pk_live_x")
    return fake_home


def _patch(module: str, client: MagicMock) -> Any:
    return patch(f"pictograph.cli.commands.{module}.get_client", return_value=client)


# ───────────── explore datasets / dataset ─────────────


def test_explore_datasets_prints_a_ref_per_row_and_passes_the_filters(
    runner: CliRunner, isolated_config: Path
) -> None:
    client = MagicMock()
    client.explore.iter_datasets.return_value = iter([PublicDataset.model_validate(dataset_row())])
    with _patch("explore", client):
        res = runner.invoke(
            root,
            [
                "explore",
                "datasets",
                "--search",
                "coco",
                "--class",
                "person",
                "--sort",
                "popular",
                "--limit",
                "5",
                "--json",
            ],
        )
    assert res.exit_code == 0, res.output
    [row] = json.loads(res.stdout)
    assert row["ref"] == "pictograph-research/coco-5k" and row["license_id"] == "cc-by-4.0"
    client.explore.iter_datasets.assert_called_once_with(
        search="coco", class_name="person", sort="popular", page_size=5, max_total=5
    )


def test_explore_tables_never_wrap_the_ref(runner: CliRunner, isolated_config: Path) -> None:
    """The REF is what the next command takes: on an 80-column terminal the other
    columns narrow, and the ref stays on one line, whole."""
    client = MagicMock()
    client.explore.iter_datasets.return_value = iter(
        [PublicDataset.model_validate(dataset_row(slug="a-rather-long-public-dataset-slug"))]
    )
    client.explore.iter_models.return_value = iter([PublicModel.model_validate(model_row())])
    with _patch("explore", client):
        datasets = runner.invoke(root, ["explore", "datasets"])
        models = runner.invoke(root, ["explore", "models"])
    assert datasets.exit_code == 0 and models.exit_code == 0, datasets.output + models.output
    assert "pictograph-research/a-rather-long-public-dataset-slug" in datasets.stdout
    assert "pictograph-research/pictograph-sceneparse150-m" in models.stdout
    assert "import-public" in datasets.stdout and "models fork <ref>" in models.stdout


def test_explore_refuses_an_unknown_sort(runner: CliRunner, isolated_config: Path) -> None:
    client = MagicMock()
    with _patch("explore", client):
        res = runner.invoke(root, ["explore", "datasets", "--sort", "newest", "--json"])
    assert res.exit_code == ExitCode.INVALID
    error = json.loads(res.stderr)["error"]
    assert error["code"] == "usage_error"
    assert "recent, popular, trending, alphabetical" in error["message"]
    client.explore.iter_datasets.assert_not_called()


def test_explore_dataset_prints_the_detail(runner: CliRunner, isolated_config: Path) -> None:
    client = MagicMock()
    client.explore.dataset.return_value = PublicDataset.model_validate(dataset_row(readme="# COCO"))
    with _patch("explore", client):
        res = runner.invoke(root, ["explore", "dataset", "pictograph-research/coco-5k"])
    assert res.exit_code == 0, res.output
    printed = json.loads(res.stdout)
    assert printed["readme"] == "# COCO" and printed["ref"] == "pictograph-research/coco-5k"
    client.explore.dataset.assert_called_once_with("pictograph-research/coco-5k")


def test_explore_dataset_explains_a_ref_without_an_organization(
    runner: CliRunner, isolated_config: Path
) -> None:
    """The real resource's refusal reaches the user as a usage line, not a traceback."""
    from pictograph.resources.explore import split_ref

    client = MagicMock()
    client.explore.dataset.side_effect = lambda ref: split_ref(ref)
    with _patch("explore", client):
        res = runner.invoke(root, ["explore", "dataset", "coco-5k"])
    assert res.exit_code != 0
    assert "Traceback" not in res.output
    assert "'<organization>/<dataset>'" in res.output


# ───────────── explore models ─────────────


def test_explore_models_passes_the_task(runner: CliRunner, isolated_config: Path) -> None:
    client = MagicMock()
    client.explore.iter_models.return_value = iter([PublicModel.model_validate(model_row())])
    with _patch("explore", client):
        res = runner.invoke(
            root, ["explore", "models", "--task", "instance_segmentation", "--json"]
        )
    assert res.exit_code == 0, res.output
    [row] = json.loads(res.stdout)
    assert row["ref"] == "pictograph-research/pictograph-sceneparse150-m"
    client.explore.iter_models.assert_called_once_with(
        search=None,
        model_type="instance_segmentation",
        sort="recent",
        page_size=50,
        max_total=50,
    )


def test_explore_models_refuses_an_unknown_task(runner: CliRunner, isolated_config: Path) -> None:
    client = MagicMock()
    with _patch("explore", client):
        res = runner.invoke(root, ["explore", "models", "--task", "detection"])
    assert res.exit_code == ExitCode.INVALID
    assert "object_detection" in res.output
    client.explore.iter_models.assert_not_called()


# ───────────── datasets import-public / import-status ─────────────


def test_import_public_waits_by_default(runner: CliRunner, isolated_config: Path) -> None:
    client = MagicMock()
    client.datasets.import_public.return_value = PublicDatasetImport.model_validate(
        import_row(status="completed", image_count=5000, progress=100.0)
    )
    with _patch("datasets", client):
        res = runner.invoke(
            root, ["datasets", "import-public", "pictograph-research/coco-5k", "--name", "mine"]
        )
    assert res.exit_code == 0, res.output
    printed = json.loads(res.stdout)
    assert printed["status"] == "completed" and printed["dataset_name"] == "COCO-5k"
    client.datasets.import_public.assert_called_once_with(
        "pictograph-research/coco-5k", name="mine", wait=True, timeout=1800.0
    )


def test_import_public_no_wait_prints_the_import_id(
    runner: CliRunner, isolated_config: Path
) -> None:
    client = MagicMock()
    client.datasets.import_public.return_value = PublicDatasetImport.model_validate(import_row())
    with _patch("datasets", client):
        res = runner.invoke(
            root, ["datasets", "import-public", "pictograph-research/coco-5k", "--no-wait"]
        )
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["import_id"] == IMPORT_ID
    assert client.datasets.import_public.call_args.kwargs["wait"] is False


@pytest.mark.parametrize("wait", [False, True])
def test_import_status(runner: CliRunner, isolated_config: Path, wait: bool) -> None:
    client = MagicMock()
    job = PublicDatasetImport.model_validate(import_row(status="running", image_count=10))
    client.datasets.get_public_import.return_value = job
    client.datasets.wait_for_public_import.return_value = job
    args = ["datasets", "import-status", IMPORT_ID] + (
        ["--wait", "--timeout", "60"] if wait else []
    )
    with _patch("datasets", client):
        res = runner.invoke(root, args)
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["status"] == "running"
    if wait:
        client.datasets.wait_for_public_import.assert_called_once_with(IMPORT_ID, timeout=60.0)
        client.datasets.get_public_import.assert_not_called()
    else:
        client.datasets.get_public_import.assert_called_once_with(IMPORT_ID)


def test_import_public_failure_is_a_structured_error(
    runner: CliRunner, isolated_config: Path
) -> None:
    from pictograph.exceptions import ApiError

    client = MagicMock()
    client.datasets.import_public.side_effect = ApiError(
        "Import of 'pictograph-research/coco-5k' failed: copy failed",
        status_code=200,
        code="job_failed",
        response={"import_id": IMPORT_ID, "status": "failed"},
    )
    with _patch("datasets", client):
        res = runner.invoke(
            root, ["datasets", "import-public", "pictograph-research/coco-5k", "--json"]
        )
    assert res.exit_code != 0
    assert "Traceback" not in res.output
    assert "job_failed" in res.output
