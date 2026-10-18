"""``pictograph deployments create / quote`` on CPU or a GPU (B533 P0-3).

``create`` never passed ``compute_type``, so the CLI could only make GPU deployments -
not the cheapest tier, the one a small model and a Free organization's credits suit -
and both commands' ``--gpu`` help still offered the retired A10G. ``quote`` had no GPU
default at all, so a bare ``quote`` was the server's 400.

These drive the REAL client over a mocked HTTP transport: the command, the SDK call and
the request on the wire.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from pictograph.cli._app import app

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
API = f"{BASE}/api/v1/developer/deployments/"
MODEL = "abcdef01-2345-6789-abcd-ef0123456789"  # an id: no name lookup
QUOTE = {
    "success": True,
    "quote": {
        "rate_per_min_micro_usd": 100,
        "cost_per_hour_micro_usd": 6000,
        "cost_per_day_micro_usd": 144000,
        "scale_to_zero": True,
        "billing_note": "Billed while serving.",
    },
}


def _created(compute_type: str, gpu_type: str | None) -> dict[str, Any]:
    return {
        "deployment": {
            "id": "dep-1",
            "organization_id": "org",
            "model_id": MODEL,
            "name": "shelf",
            "status": "provisioning",
            "compute_type": compute_type,
            "gpu_type": gpu_type,
            "min_containers": 0,
            "max_containers": 1,
            "scaledown_window": 60,
            "billing_note": "Billed while serving.",
        },
        "auth_token": "pk_deploy_once",
    }


def _plain(output: str) -> str:
    return " ".join(output.replace("│", " ").split())


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_test")
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    return lambda *args: runner.invoke(app, ["deployments", *args])


def test_create_on_cpu_sends_the_compute_type_and_no_gpu(httpx_mock: HTTPXMock, run: Any) -> None:
    httpx_mock.add_response(method="POST", url=API, json=_created("cpu", None))
    res = run("create", MODEL, "--compute", "cpu")
    assert res.exit_code == 0, res.output
    body = json.loads(httpx_mock.get_requests()[-1].content)
    assert body["compute_type"] == "cpu" and body["gpu_type"] is None


def test_create_defaults_to_a_t4_gpu(httpx_mock: HTTPXMock, run: Any) -> None:
    httpx_mock.add_response(method="POST", url=API, json=_created("gpu", "t4"))
    res = run("create", MODEL)
    assert res.exit_code == 0, res.output
    body = json.loads(httpx_mock.get_requests()[-1].content)
    assert (body["compute_type"], body["gpu_type"]) == ("gpu", "t4")


def test_create_on_an_l4(httpx_mock: HTTPXMock, run: Any) -> None:
    httpx_mock.add_response(method="POST", url=API, json=_created("gpu", "l4"))
    res = run("create", MODEL, "--compute-type", "gpu", "--gpu", "l4")
    assert res.exit_code == 0, res.output
    body = json.loads(httpx_mock.get_requests()[-1].content)
    assert (body["compute_type"], body["gpu_type"]) == ("gpu", "l4")


@pytest.mark.parametrize(
    ("args", "hint", "says"),
    [
        (("--compute", "cpu", "--gpu", "t4"), "--gpu", "A CPU deployment has no GPU"),
        (("--gpu", "a10g"), "--gpu", "use t4, l4, a100"),
        (("--compute", "tpu"), "--compute-type", "use gpu or cpu"),
    ],
    ids=["cpu-with-gpu", "retired-a10g", "unknown-compute"],
)
@pytest.mark.parametrize("command", ["create", "quote"])
def test_a_wrong_compute_choice_is_refused_before_any_request(
    httpx_mock: HTTPXMock, run: Any, command: str, args: tuple[str, ...], hint: str, says: str
) -> None:
    res = run(command, *([MODEL] if command == "create" else []), *args)
    assert res.exit_code == 2
    assert says in _plain(res.output) and hint in _plain(res.output)
    assert httpx_mock.get_requests() == []


def test_a_bare_quote_prices_the_default_gpu(httpx_mock: HTTPXMock, run: Any) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{API}quote?compute_type=gpu&min_containers=0&gpu_type=t4", json=QUOTE
    )
    res = run("quote")
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["rate_per_min_micro_usd"] == 100


def test_a_cpu_quote_sends_no_gpu(httpx_mock: HTTPXMock, run: Any) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{API}quote?compute_type=cpu&min_containers=1", json=QUOTE
    )
    res = run("quote", "--compute", "cpu", "--min-containers", "1")
    assert res.exit_code == 0, res.output


def test_a_legacy_deployment_on_a_retired_tier_still_reads(httpx_mock: HTTPXMock, run: Any) -> None:
    """The type a caller CHOOSES from is the current tiers; what a deployment REPORTS is
    read as-is, so a row made on A10G before it was retired still lists."""
    row = _created("gpu", "a10g")["deployment"]
    httpx_mock.add_response(
        method="GET",
        url=f"{API}?limit=50&offset=0",
        json={"deployments": [row], "pagination": {"total": 1, "limit": 50, "offset": 0}},
    )
    res = run("list", "--json")
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)[0]["gpu_type"] == "a10g"
