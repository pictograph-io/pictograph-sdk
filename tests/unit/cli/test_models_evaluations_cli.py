"""Tests for ``pictograph models evaluate / evaluations / evaluation``.

The SDK client is a :class:`~unittest.mock.MagicMock`, but its ``model_evaluations``
resource is an AUTOSPEC of the real :class:`ModelEvaluations`: a command that passes a
keyword the SDK does not take (or misses a required one) fails here exactly as it
would against the API, instead of being swallowed by a permissive mock.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, create_autospec, patch

import pytest
from typer.testing import CliRunner

from pictograph.cli._config import write_config
from pictograph.cli.commands.models import app
from pictograph.exceptions import PollTimeoutError
from pictograph.models.evaluation import ModelEvaluation
from pictograph.resources.model_evaluations import ModelEvaluations

EVAL_ID = "8d1c0b7e-2f4a-4c55-9a1e-3b2d4c5e6f70"


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
    client.model_evaluations = create_autospec(ModelEvaluations, instance=True)
    return client


def _patch(client: MagicMock) -> Any:
    return patch("pictograph.cli.commands.models.get_client", return_value=client)


def _evaluation(
    status: str = "completed",
    *,
    family: str = "detection",
    iou: float = 0.5,
    evaluation_id: str = EVAL_ID,
    progress: int = 100,
) -> ModelEvaluation:
    """A row as the developer API returns it, per model family."""
    overall: dict[str, Any] = {
        "tp": 40,
        "fp": 9,
        "fn": 11,
        "precision": 0.8163,
        "recall": 0.7843,
        "f1": 0.8,
        "macro_f1": 0.79,
        "map": 0.6123,
    }
    per_class: list[dict[str, Any]] = [
        {
            "class_name": "car",
            "tp": 30,
            "fp": 5,
            "fn": 6,
            "support": 36,
            "precision": 0.8571,
            "recall": 0.8333,
            "f1": 0.8451,
            "ap": 0.7012,
        },
    ]
    if family == "classification":
        overall.update(map=None, accuracy=0.9)
    elif family == "semantic":
        overall.update(map=None, miou=0.7071, mean_dice=0.8)
        per_class[0].update(iou=0.6543, dice=0.7911)
    row: dict[str, Any] = {
        "id": evaluation_id,
        "organization_id": "org-uuid",
        "model_id": "model-uuid",
        "project_id": "dataset-uuid",
        "export_id": "export-uuid",
        "status": status,
        "progress": progress,
        "iou_threshold": iou,
        "confidence_threshold": 0.5,
        "total_images": 25,
        "evaluated_images": 25 if status == "completed" else 10,
        "failed_images": 0,
        "created_at": "2026-10-19T09:30:00Z",
    }
    if status == "completed":
        row["overall_metrics"] = overall
        row["per_class_metrics"] = per_class
    return ModelEvaluation.model_validate(row)


# ─────────────────────────── evaluate ───────────────────────────


def test_evaluate_starts_the_run_and_returns_without_waiting(runner: CliRunner) -> None:
    client = _client()
    client.model_evaluations.create.return_value = _evaluation("pending", progress=0)
    with _patch(client):
        result = runner.invoke(
            app,
            [
                "evaluate",
                "Swift Falcon",
                "--dataset",
                "road-signs",
                "--export",
                "holdout",
                "--json",
            ],
        )
    assert result.exit_code == 0, result.output
    client.model_evaluations.create.assert_called_once_with(
        "Swift Falcon", "road-signs", "holdout", iou_threshold=0.5, confidence_threshold=0.5
    )
    client.model_evaluations.wait_for_completion.assert_not_called()
    body = json.loads(result.stdout)
    assert body["id"] == EVAL_ID
    assert body["status"] == "pending"


def test_evaluate_without_json_names_the_command_that_waits(runner: CliRunner) -> None:
    client = _client()
    client.model_evaluations.create.return_value = _evaluation("pending", progress=0)
    with _patch(client):
        result = runner.invoke(
            app, ["evaluate", "Swift Falcon", "--dataset", "road-signs", "--export", "holdout"]
        )
    assert result.exit_code == 0, result.output
    assert EVAL_ID in result.output
    # The next step is printed whole, on one line, so it can be copied.
    assert f"pictograph models evaluation {EVAL_ID} --wait" in result.output


def test_evaluate_wait_passes_every_option_through_and_prints_metrics(runner: CliRunner) -> None:
    client = _client()
    client.model_evaluations.create.return_value = _evaluation("pending", iou=0.6, progress=0)
    client.model_evaluations.wait_for_completion.return_value = _evaluation(iou=0.6)
    with _patch(client):
        result = runner.invoke(
            app,
            [
                "evaluate",
                "Swift Falcon",
                "--dataset",
                "road-signs",
                "--export",
                "holdout",
                "--iou",
                "0.6",
                "-c",
                "0.3",
                "--wait",
                "--timeout",
                "60",
                "--poll-interval",
                "1",
            ],
        )
    assert result.exit_code == 0, result.output
    client.model_evaluations.create.assert_called_once_with(
        "Swift Falcon", "road-signs", "holdout", iou_threshold=0.6, confidence_threshold=0.3
    )
    client.model_evaluations.wait_for_completion.assert_called_once_with(
        EVAL_ID, poll_interval=1.0, timeout=60.0
    )
    # The headline is read at the evaluation's own IoU, and every class gets a row.
    assert "mAP@0.6" in result.output
    assert "0.6123" in result.output
    assert "car" in result.output
    assert "0.7012" in result.output


def test_evaluate_wait_json_is_the_finished_evaluation(runner: CliRunner) -> None:
    client = _client()
    client.model_evaluations.create.return_value = _evaluation("pending", progress=0)
    client.model_evaluations.wait_for_completion.return_value = _evaluation()
    with _patch(client):
        result = runner.invoke(
            app,
            [
                "evaluate",
                "Swift Falcon",
                "--dataset",
                "road-signs",
                "--export",
                "holdout",
                "--wait",
                "--json",
            ],
        )
    assert result.exit_code == 0, result.output
    # The "waiting" note is on stderr, so stdout stays one JSON document.
    body = json.loads(result.stdout)
    assert body["status"] == "completed"
    assert body["overall_metrics"]["map"] == 0.6123
    assert body["per_class_metrics"][0]["class_name"] == "car"


@pytest.mark.parametrize(
    "flags",
    [
        ["--iou", "0"],
        ["--iou", "1.5"],
        ["--iou", "-0.2"],
        ["--confidence", "-0.1"],
        ["--confidence", "2"],
    ],
)
def test_evaluate_refuses_thresholds_the_api_refuses_before_any_request(
    runner: CliRunner, flags: list[str]
) -> None:
    get_client = MagicMock()
    with patch("pictograph.cli.commands.models.get_client", get_client):
        result = runner.invoke(
            app,
            ["evaluate", "Swift Falcon", "--dataset", "road-signs", "--export", "holdout", *flags],
        )
    assert result.exit_code == 2, result.output
    get_client.assert_not_called()


def test_evaluate_accepts_the_bounds_themselves(runner: CliRunner) -> None:
    client = _client()
    client.model_evaluations.create.return_value = _evaluation("pending", progress=0)
    with _patch(client):
        result = runner.invoke(
            app,
            [
                "evaluate",
                "Swift Falcon",
                "--dataset",
                "road-signs",
                "--export",
                "holdout",
                "--iou",
                "1",
                "--confidence",
                "0",
            ],
        )
    assert result.exit_code == 0, result.output
    client.model_evaluations.create.assert_called_once_with(
        "Swift Falcon", "road-signs", "holdout", iou_threshold=1.0, confidence_threshold=0.0
    )


def test_evaluate_needs_a_dataset_and_an_export(runner: CliRunner) -> None:
    get_client = MagicMock()
    with patch("pictograph.cli.commands.models.get_client", get_client):
        result = runner.invoke(app, ["evaluate", "Swift Falcon", "--dataset", "road-signs"])
    assert result.exit_code == 2
    assert "--export" in result.output
    get_client.assert_not_called()


def test_a_wait_that_times_out_names_the_cli_next_step(runner: CliRunner) -> None:
    client = _client()
    client.model_evaluations.create.return_value = _evaluation("pending", progress=0)
    client.model_evaluations.wait_for_completion.side_effect = PollTimeoutError("did not complete")
    with _patch(client):
        result = runner.invoke(
            app,
            [
                "evaluate",
                "Swift Falcon",
                "--dataset",
                "road-signs",
                "--export",
                "holdout",
                "--wait",
                "--timeout",
                "5",
            ],
        )
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert f"pictograph models evaluation {EVAL_ID} --wait" in flat
    assert "keeps running" in flat


# ─────────────────────────── evaluations ───────────────────────────


def test_evaluations_filters_by_model_and_dataset_and_stops_at_the_limit(
    runner: CliRunner,
) -> None:
    client = _client()
    rows = [_evaluation(evaluation_id=f"eval-{i}") for i in range(3)]
    client.model_evaluations.iter.return_value = iter(rows)
    with _patch(client):
        result = runner.invoke(
            app, ["evaluations", "Swift Falcon", "--dataset", "road-signs", "-n", "2", "--json"]
        )
    assert result.exit_code == 0, result.output
    client.model_evaluations.iter.assert_called_once_with(
        model="Swift Falcon", dataset_name="road-signs", page_size=2
    )
    body = json.loads(result.stdout)
    assert [e["id"] for e in body] == ["eval-0", "eval-1"]


def test_evaluations_without_a_model_lists_the_whole_organization(runner: CliRunner) -> None:
    client = _client()
    client.model_evaluations.iter.return_value = iter([])
    with _patch(client):
        result = runner.invoke(app, ["evaluations", "--limit", "500"])
    assert result.exit_code == 0, result.output
    # One page is at most 100 rows; a larger --limit pages instead of a 422.
    client.model_evaluations.iter.assert_called_once_with(
        model=None, dataset_name=None, page_size=100
    )
    assert "No results" in result.output


def test_evaluations_table_reads_each_family_by_its_own_headline(runner: CliRunner) -> None:
    client = _client()
    client.model_evaluations.iter.return_value = iter(
        [
            _evaluation(evaluation_id="det"),
            _evaluation(evaluation_id="cls", family="classification"),
            _evaluation(evaluation_id="seg", family="semantic"),
            _evaluation("running", evaluation_id="run", progress=40),
        ]
    )
    with _patch(client):
        # A list row is several ids wide: read it on a wide terminal.
        result = runner.invoke(app, ["evaluations"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "mAP@0.5 0.6123" in flat
    assert "accuracy 0.9" in flat
    assert "mIoU 0.7071" in flat
    assert "running" in flat


# ─────────────────────────── evaluation ───────────────────────────


def test_evaluation_shows_one_run(runner: CliRunner) -> None:
    client = _client()
    client.model_evaluations.get.return_value = _evaluation(family="semantic")
    with _patch(client):
        result = runner.invoke(app, ["evaluation", EVAL_ID])
    assert result.exit_code == 0, result.output
    client.model_evaluations.get.assert_called_once_with(EVAL_ID)
    client.model_evaluations.wait_for_completion.assert_not_called()
    # Semantic segmentation reads by pixel IoU / Dice per class, not detection AP.
    assert "mIoU" in result.output
    assert "0.6543" in result.output
    assert "0.7911" in result.output
    assert "ap" not in result.output.split()


def test_evaluation_wait_polls_until_it_finishes(runner: CliRunner) -> None:
    client = _client()
    client.model_evaluations.wait_for_completion.return_value = _evaluation(family="classification")
    with _patch(client):
        result = runner.invoke(app, ["evaluation", EVAL_ID, "--wait", "--json"])
    assert result.exit_code == 0, result.output
    client.model_evaluations.wait_for_completion.assert_called_once_with(
        EVAL_ID, poll_interval=3.0, timeout=1800.0
    )
    client.model_evaluations.get.assert_not_called()
    assert json.loads(result.stdout)["overall_metrics"]["accuracy"] == 0.9


def test_evaluation_report_writes_the_json_report(runner: CliRunner, tmp_path: Path) -> None:
    client = _client()
    client.model_evaluations.get.return_value = _evaluation()
    report = {"schema_version": 1, "overall_metrics": {"map": 0.6123}, "warnings": []}
    client.model_evaluations.report.return_value = report
    out = tmp_path / "reports" / "eval.json"
    with _patch(client):
        result = runner.invoke(app, ["evaluation", EVAL_ID, "--report", str(out)])
    assert result.exit_code == 0, result.output
    client.model_evaluations.report.assert_called_once_with(EVAL_ID)
    assert json.loads(out.read_text()) == report
    summary = json.loads(result.stdout)
    assert summary["report_path"] == str(out)
    assert summary["schema_version"] == 1
    assert summary["size_bytes"] == out.stat().st_size


@pytest.mark.parametrize("name", ["report.html", "report.HTM", "report.pdf"])
def test_evaluation_report_refuses_html_or_pdf_before_any_request(
    runner: CliRunner, tmp_path: Path, name: str
) -> None:
    get_client = MagicMock()
    with patch("pictograph.cli.commands.models.get_client", get_client):
        result = runner.invoke(app, ["evaluation", EVAL_ID, "--report", str(tmp_path / name)])
    assert result.exit_code == 2
    assert "JSON report" in " ".join(result.output.split())
    get_client.assert_not_called()
    assert not (tmp_path / name).exists()


def test_evaluation_report_on_a_running_run_says_to_wait(runner: CliRunner, tmp_path: Path) -> None:
    client = _client()
    client.model_evaluations.get.return_value = _evaluation("running", progress=40)
    out = tmp_path / "eval.json"
    with _patch(client):
        result = runner.invoke(app, ["evaluation", EVAL_ID, "--report", str(out)])
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "still running (40%)" in flat
    assert "--wait" in flat
    client.model_evaluations.report.assert_not_called()
    assert not out.exists()


def test_evaluation_report_on_a_failed_run_says_there_is_none(
    runner: CliRunner, tmp_path: Path
) -> None:
    client = _client()
    client.model_evaluations.get.return_value = _evaluation("failed", progress=10)
    with _patch(client):
        result = runner.invoke(app, ["evaluation", EVAL_ID, "--report", str(tmp_path / "e.json")])
    assert result.exit_code == 1
    assert "ended failed, so it has no report" in " ".join(result.output.split())
    client.model_evaluations.report.assert_not_called()


def test_evaluation_wait_then_report(runner: CliRunner, tmp_path: Path) -> None:
    client = _client()
    client.model_evaluations.wait_for_completion.return_value = _evaluation()
    client.model_evaluations.report.return_value = {"schema_version": 1}
    out = tmp_path / "eval.json"
    with _patch(client):
        result = runner.invoke(app, ["evaluation", EVAL_ID, "--wait", "--report", str(out)])
    assert result.exit_code == 0, result.output
    client.model_evaluations.wait_for_completion.assert_called_once()
    client.model_evaluations.report.assert_called_once_with(EVAL_ID)
    assert out.is_file()


def test_evaluation_cancel(runner: CliRunner) -> None:
    client = _client()
    client.model_evaluations.cancel.return_value = _evaluation("cancelled", progress=30)
    with _patch(client):
        result = runner.invoke(app, ["evaluation", EVAL_ID, "--cancel", "--json"])
    assert result.exit_code == 0, result.output
    client.model_evaluations.cancel.assert_called_once_with(EVAL_ID)
    client.model_evaluations.get.assert_not_called()
    assert json.loads(result.stdout)["status"] == "cancelled"


@pytest.mark.parametrize("extra", [["--wait"], ["--report", "eval.json"]])
def test_evaluation_cancel_refuses_to_combine(runner: CliRunner, extra: list[str]) -> None:
    get_client = MagicMock()
    with patch("pictograph.cli.commands.models.get_client", get_client):
        result = runner.invoke(app, ["evaluation", EVAL_ID, "--cancel", *extra])
    assert result.exit_code == 2
    get_client.assert_not_called()
