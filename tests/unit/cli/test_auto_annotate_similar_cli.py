"""``pictograph auto-annotate similar`` - the app's "Annotate similar" from the CLI (B533 #9).

What each flag reaches (the SDK call, exactly), the refusals made before any request,
the summary a person reads and the JSON an agent reads, ``--quote`` (writes nothing),
and the exit code of a job that failed.
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
from pictograph.exceptions import ApiError
from pictograph.models.auto_annotate import BatchJob, SimilarQuote

JOB = "44444444-3333-2222-1111-000000000576"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
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


def _client() -> MagicMock:
    client = MagicMock()
    client.auto_annotate.similar.return_value = BatchJob(
        job_id=JOB,
        status="completed",
        progress=100,
        total_images=11,
        processed_images=11,
        total_annotations_added=23,
        failed_images=1,
    )
    client.auto_annotate.quote_similar.return_value = SimilarQuote(
        total_images=11,
        total_in_scope=11,
        estimated_credits=52_000,
        containers=2,
        remaining_credits=4_000_000,
        sufficient=True,
        examples=8,
        target_class="car",
        output_type="polygon",
        scope="directory",
        directory="/exemplar-test",
    )
    return client


def _invoke(runner: CliRunner, client: MagicMock, *args: str) -> Any:
    with patch("pictograph.cli.commands.auto_annotate.get_client", return_value=client):
        return runner.invoke(root, ["auto-annotate", "similar", *args])


def test_by_class_waits_and_says_what_it_labelled(runner: CliRunner) -> None:
    client = _client()
    res = _invoke(runner, client, "COCO-5k", "exemplar-test/a.jpg", "--class", "car")
    assert res.exit_code == 0, res.output
    client.auto_annotate.similar.assert_called_once_with(
        "COCO-5k",
        "exemplar-test/a.jpg",
        wait=True,
        timeout=1800.0,
        annotation_ids=[],
        reference_class="car",
        scope="directory",
        target_class=None,
        output_type="polygon",
        confidence=None,
    )
    assert (
        "Labelled 10 of 11 image(s) in COCO-5k: 23 annotation(s) added, 1 image(s) failed."
        in res.stdout
    )
    assert "review them before you train" in res.stderr


def test_every_flag_reaches_the_sdk(runner: CliRunner) -> None:
    client = _client()
    res = _invoke(
        runner,
        client,
        "COCO-5k",
        "a.jpg",
        "-a",
        "x",
        "--annotation",
        "y",
        "--scope",
        "dataset",
        "--target",
        "vehicle",
        "--output-type",
        "obb",
        "--confidence",
        "0.6",
        "--no-wait",
        "--timeout",
        "60",
    )
    assert res.exit_code == 0, res.output
    client.auto_annotate.similar.assert_called_once_with(
        "COCO-5k",
        "a.jpg",
        wait=False,
        timeout=60.0,
        annotation_ids=["x", "y"],
        reference_class=None,
        scope="dataset",
        target_class="vehicle",
        output_type="obb",
        confidence=0.6,
    )
    assert f"Started job {JOB}" in res.stdout
    assert f"pictograph jobs wait {JOB}" in res.stderr


def test_json_is_the_job(runner: CliRunner) -> None:
    res = _invoke(runner, _client(), "COCO-5k", "a.jpg", "--class", "car", "--json")
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert (payload["job_id"], payload["status"], payload["total_annotations_added"]) == (
        JOB,
        "completed",
        23,
    )


def test_quote_prints_what_it_would_do_and_starts_nothing(runner: CliRunner) -> None:
    client = _client()
    res = _invoke(runner, client, "COCO-5k", "a.jpg", "--class", "car", "--quote")
    assert res.exit_code == 0, res.output
    client.auto_annotate.similar.assert_not_called()
    client.auto_annotate.quote_similar.assert_called_once_with(
        "COCO-5k",
        "a.jpg",
        annotation_ids=[],
        reference_class="car",
        scope="directory",
        target_class=None,
        output_type="polygon",
        confidence=None,
    )
    assert (
        "Would label 11 image(s) in /exemplar-test as car (polygon) from 8 example(s): about $0.05"
        in res.stdout
    )


def test_quote_json_and_an_insufficient_balance_said_on_stderr(runner: CliRunner) -> None:
    client = _client()
    client.auto_annotate.quote_similar.return_value = SimilarQuote(
        total_images=11,
        estimated_credits=52_000,
        remaining_credits=10_000,
        sufficient=False,
        target_class="car",
    )
    res = _invoke(runner, client, "COCO-5k", "a.jpg", "--class", "car", "--quote", "--json")
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["sufficient"] is False
    assert "cannot start" in res.stderr and "$0.01 remains" in res.stderr


@pytest.mark.parametrize(
    "args, words",
    [
        ((), "--class <name>"),
        (("--class", "car", "--annotation", "x"), "not both"),
        (("--class", "car", "--scope", "planet"), "directory or dataset"),
        (("--class", "car", "--output-type", "tag"), "polygon, bbox or obb"),
        (("--class", "car", "--confidence", "0.95"), "0.95"),
    ],
)
def test_refusals_before_any_request(runner: CliRunner, args: tuple[str, ...], words: str) -> None:
    client = _client()
    res = _invoke(runner, client, "COCO-5k", "a.jpg", *args)
    assert res.exit_code != 0
    assert words in res.output
    client.auto_annotate.similar.assert_not_called()
    client.auto_annotate.quote_similar.assert_not_called()


def test_a_job_that_failed_exits_with_its_reason(runner: CliRunner) -> None:
    client = _client()
    client.auto_annotate.similar.side_effect = ApiError(
        f"Batch job {JOB} ended with status 'failed': 'a.jpg' can't be read as an image.",
        code="job_failed",
    )
    res = _invoke(runner, client, "COCO-5k", "a.jpg", "--class", "car")
    assert res.exit_code != 0
    assert "can't be read as an image" in res.output
