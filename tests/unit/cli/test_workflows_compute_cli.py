"""``pictograph workflows compute-options`` and ``--gpu`` - a workflow's compute from the terminal
(a Model block's precision is its own, set in the graph). Drives the REAL client over a mocked HTTP
transport."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from pictograph.cli._app import app
from pictograph.cli._errors import ExitCode

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
API = f"{BASE}/api/v1/developer/workflows"
WID = "fedcba98-1111-2222-3333-444455556666"
OPTIONS = {
    "gpus": [
        {
            "gpu": "t4",
            "label": "T4 GPU",
            "vram_gb": 16,
            "description": "16 GB VRAM",
            "rate_per_min_micro_usd": 19667,
        },
        {
            "gpu": "l4",
            "label": "L4 GPU",
            "vram_gb": 24,
            "description": "24 GB VRAM",
            "rate_per_min_micro_usd": 26667,
        },
    ],
    "precisions": [
        {"precision": "fp32", "label": "FP32", "description": "32-bit, the model as trained"},
        {
            "precision": "fp16",
            "label": "FP16",
            "description": "16-bit, faster, accuracy-equivalent",
        },
    ],
    "default": {"gpu": "t4"},
}


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_test")
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "200")
    runner = CliRunner()
    return lambda *args: runner.invoke(app, ["workflows", *args])


def test_compute_options_tables_each_gpu_with_its_memory_and_price(
    httpx_mock: HTTPXMock, cli: Any
) -> None:
    httpx_mock.add_response(method="GET", url=f"{API}/compute-options", json=OPTIONS)
    out = cli("compute-options")
    assert out.exit_code == 0, out.output
    assert "L4 GPU" in out.output and "$0.0267" in out.output
    assert "16 GB" in out.output and "24 GB" in out.output
    assert "16-bit, faster, accuracy-equivalent" in out.output and "Model block" in out.output


def test_compute_options_as_json(httpx_mock: HTTPXMock, cli: Any) -> None:
    httpx_mock.add_response(method="GET", url=f"{API}/compute-options", json=OPTIONS)
    out = cli("compute-options", "--json")
    assert out.exit_code == 0, out.output
    assert json.loads(out.output)["default"] == {"gpu": "t4"}


def test_run_with_its_own_gpu(httpx_mock: HTTPXMock, cli: Any) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{API}/{WID}/run", json={"run_id": "run-1", "deposit_micro_usd": 9}
    )
    out = cli("run", WID, "--gpu", "l4", "--no-wait")
    assert out.exit_code == 0, out.output
    assert json.loads(httpx_mock.get_requests()[-1].content) == {"compute": {"gpu": "l4"}}


@pytest.mark.parametrize("command", ["run", "update", "create", "invoke"])
def test_no_command_takes_a_precision_it_is_each_model_blocks_own(cli: Any, command: str) -> None:
    out = cli(command, WID, "--precision", "fp16")
    assert out.exit_code == ExitCode.INVALID and "No such option: --precision" in out.output


def test_update_sets_a_workflows_compute(httpx_mock: HTTPXMock, cli: Any) -> None:
    wf = {
        "id": WID,
        "organization_id": "o",
        "name": "Counter",
        "compute": {"gpu": "l4"},
    }
    httpx_mock.add_response(method="PATCH", url=f"{API}/{WID}", json={"workflow": wf})
    out = cli("update", WID, "--gpu", "l4")
    assert out.exit_code == 0, out.output
    assert json.loads(httpx_mock.get_requests()[-1].content) == {"compute": {"gpu": "l4"}}
    assert json.loads(out.output)["compute"] == {"gpu": "l4"}
