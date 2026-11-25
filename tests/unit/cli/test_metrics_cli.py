"""Tests for ``pictograph metrics evaluate``.

Offline commands - no client, no network. A real CliRunner drives the standalone
``metrics`` Typer app over temp Pictograph-JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pictograph.cli.commands.metrics import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data))
    return path


def _bbox(name: str, conf: float = 1.0, x: float = 0.0) -> dict[str, Any]:
    return {
        "id": f"a{x}",
        "name": name,
        "type": "bbox",
        "bounding_box": {"x": x, "y": 0, "w": 10, "h": 10},
        "confidence": conf,
    }


# ───────────── evaluate ─────────────


def test_evaluate_perfect_match_json(runner: CliRunner, tmp_path: Path) -> None:
    gt = _write(tmp_path / "gt.json", {"img1": [_bbox("car")]})
    preds = _write(tmp_path / "preds.json", {"img1": [_bbox("car")]})
    result = runner.invoke(app, ["evaluate", str(preds), str(gt), "--json"])
    assert result.exit_code == 0, result.stdout
    out = json.loads(result.stdout)
    assert out["precision"] == 1.0
    assert out["recall"] == 1.0
    assert out["mean_average_precision"] == 1.0
    assert out["per_class"]["car"]["support"] == 1


def test_evaluate_table_mentions_class_and_map(runner: CliRunner, tmp_path: Path) -> None:
    gt = _write(tmp_path / "gt.json", {"img1": [_bbox("car")]})
    preds = _write(tmp_path / "preds.json", {"img1": [_bbox("car")]})
    result = runner.invoke(app, ["evaluate", str(preds), str(gt)])
    assert result.exit_code == 0
    assert "car" in result.stdout
    assert "mAP" in result.stdout


def test_evaluate_bad_iou_errors(runner: CliRunner, tmp_path: Path) -> None:
    gt = _write(tmp_path / "gt.json", {"img1": [_bbox("car")]})
    preds = _write(tmp_path / "preds.json", {"img1": [_bbox("car")]})
    result = runner.invoke(app, ["evaluate", str(preds), str(gt), "--iou", "1.5"])
    assert result.exit_code != 0


# ───────────── input validation ─────────────


def test_missing_file_errors(runner: CliRunner, tmp_path: Path) -> None:
    gt = _write(tmp_path / "gt.json", {"img1": [_bbox("car")]})
    result = runner.invoke(app, ["evaluate", str(tmp_path / "nope.json"), str(gt)])
    assert result.exit_code != 0


def test_non_object_json_errors(runner: CliRunner, tmp_path: Path) -> None:
    gt = _write(tmp_path / "gt.json", {"img1": [_bbox("car")]})
    bad = _write(tmp_path / "bad.json", ["not", "an", "object"])
    result = runner.invoke(app, ["evaluate", str(bad), str(gt)])
    assert result.exit_code != 0
