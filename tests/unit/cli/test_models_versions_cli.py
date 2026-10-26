"""``pictograph models versions / set-version / files / download-file`` (B533 P1 #11).

The SDK had ``models.versions``, ``set_current_version``, ``files`` and
``download_file``; the CLI had none of them, so an agent could not see which formats
a model publishes (ONNX, PyTorch, TensorRT, ExecuTorch, fp16) or roll a version back.
These drive the REAL app and client over a mocked HTTP transport.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import pytest
from typer.testing import CliRunner

from pictograph.cli._app import app
from pictograph.cli._errors import ExitCode

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
MODEL = f"{BASE}/api/v1/developer/models/Shelf%20Detector"
V1 = "11111111-1111-4111-8111-111111111111"
V2 = "22222222-2222-4222-8222-222222222222"
NOW = "2026-04-01T00:00:00Z"


def _version(
    vid: str, number: int, *, current: bool, latest: bool, metric: float
) -> dict[str, Any]:
    return {
        "version_id": vid,
        "version_number": number,
        "version_label": f"{number}.0.0",
        "status": "ready",
        "is_current": current,
        "is_latest": latest,
        "precision": "fp32",
        "architecture": "RF-DETR Small",
        "metrics": {"headline_metric": "mAP", "headline_value": metric},
        "created_at": NOW,
    }


VERSIONS = {
    "versions": [
        _version(V2, 2, current=True, latest=True, metric=0.61),
        _version(V1, 1, current=False, latest=False, metric=0.52),
    ],
    "current_version_id": V2,
    "pinned_version_id": None,
    "latest_version_id": V2,
}


def _file(vid: str, name: str, fmt: str, **extra: Any) -> dict[str, Any]:
    return {"version_id": vid, "name": name, "kind": "weights", "format": fmt, **extra}


MANIFEST = {
    "versions": VERSIONS["versions"],
    "files": [
        _file(V2, "model.onnx", "onnx", runtime="onnxruntime", precision="fp32", size_bytes=1024),
        _file(
            V2,
            "model-fp16.engine",
            "engine",
            runtime="tensorrt",
            precision="fp16",
            target_key="sm89",
            stale=True,
        ),
        _file(V2, "config.json", "json", kind="config"),
        _file(V1, "model.onnx", "onnx", runtime="onnxruntime", precision="fp32"),
    ],
    "pinned_version_id": None,
}


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_test")
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    monkeypatch.setenv("PICTOGRAPH_MAX_RETRIES", "0")
    monkeypatch.delenv("PICTOGRAPH_OUTPUT", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    return lambda *args: runner.invoke(app, ["models", *args])


def test_versions_lists_each_with_its_metric_and_which_is_current(
    run: Any, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=f"{MODEL}/versions", json={"data": VERSIONS})
    res = run("versions", "Shelf Detector", "--json")
    assert res.exit_code == ExitCode.OK, res.output
    payload = json.loads(res.stdout)
    assert [v["version_label"] for v in payload["versions"]] == ["2.0.0", "1.0.0"]

    httpx_mock.add_response(url=f"{MODEL}/versions", json={"data": VERSIONS})
    table = run("versions", "Shelf Detector")
    assert "follows the newest ready version" in table.stdout
    assert "0.61" in table.stdout and "0.52" in table.stdout


@pytest.mark.parametrize("ref", ["1.0.0", "1", "v1", V1])
def test_set_version_rolls_back_by_label_number_or_id(
    run: Any, httpx_mock: HTTPXMock, ref: str
) -> None:
    httpx_mock.add_response(method="GET", url=f"{MODEL}/versions", json={"data": VERSIONS})
    pinned = {
        **VERSIONS,
        "versions": [
            _version(V2, 2, current=False, latest=True, metric=0.61),
            _version(V1, 1, current=True, latest=False, metric=0.52),
        ],
        "current_version_id": V1,
        "pinned_version_id": V1,
    }
    httpx_mock.add_response(method="PATCH", url=f"{MODEL}/current-version", json={"data": pinned})

    res = run("set-version", "Shelf Detector", ref)

    assert res.exit_code == ExitCode.OK, res.output
    (patch,) = [r for r in httpx_mock.get_requests() if r.method == "PATCH"]
    assert json.loads(patch.content) == {"version_id": V1}
    assert json.loads(res.stdout) == {
        "model": "Shelf Detector",
        "current_version": "1.0.0",
        "current_version_id": V1,
        "pinned": True,
    }


def test_set_version_clear_unpins(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="PATCH", url=f"{MODEL}/current-version", json={"data": VERSIONS})
    res = run("set-version", "Shelf Detector", "--clear")
    assert res.exit_code == ExitCode.OK, res.output
    (patch,) = httpx_mock.get_requests()
    assert json.loads(patch.content) == {"version_id": None}
    assert json.loads(res.stdout)["pinned"] is False


@pytest.mark.parametrize("args", [(), ("2", "--clear")])
def test_set_version_needs_exactly_one_of_a_version_or_clear(
    run: Any, httpx_mock: HTTPXMock, args: tuple[str, ...]
) -> None:
    res = run("set-version", "Shelf Detector", *args)
    assert res.exit_code == ExitCode.INVALID
    assert httpx_mock.get_requests() == []


def test_an_unknown_version_is_not_found_and_lists_the_real_ones(
    run: Any, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PICTOGRAPH_OUTPUT", "json")
    httpx_mock.add_response(url=f"{MODEL}/versions", json={"data": VERSIONS})
    res = run("set-version", "Shelf Detector", "7")
    assert res.exit_code == ExitCode.NOT_FOUND
    error = json.loads(res.stderr.strip().splitlines()[-1])["error"]
    assert error["code"] == "version_not_found"
    assert "it has: 2.0.0, 1.0.0" in error["message"]


def test_files_shows_the_current_versions_files(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{MODEL}/files", json={"data": MANIFEST})
    res = run("files", "Shelf Detector", "--json")
    assert res.exit_code == ExitCode.OK, res.output
    names = [f["name"] for f in json.loads(res.stdout)]
    assert names == ["model.onnx", "model-fp16.engine", "config.json"], "version 2 only"


def test_files_of_another_version_or_all(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{MODEL}/files", json={"data": MANIFEST})
    one = run("files", "Shelf Detector", "--version", "1", "--json")
    assert [f["version_id"] for f in json.loads(one.stdout)] == [V1]

    httpx_mock.add_response(url=f"{MODEL}/files", json={"data": MANIFEST})
    table = run("files", "Shelf Detector", "--all-versions")
    assert table.exit_code == ExitCode.OK
    assert "stale TensorRT engine will not load" in table.stderr


def test_download_file_writes_the_named_file(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    payload = '{"_pictograph": {"model_type": "object_detection"}}'
    httpx_mock.add_response(url=f"{MODEL}/versions", json={"data": VERSIONS})
    httpx_mock.add_response(
        method="GET",
        url=f"{MODEL}/files/download?version_id={V1}&name=config.json",
        json={"data": {"download_url": "data:application/json," + quote(payload)}},
    )

    res = run(
        "download-file", "Shelf Detector", "config.json", "--version", "1", "-o", "out/c.json"
    )

    assert res.exit_code == ExitCode.OK, res.output
    assert (tmp_path / "out" / "c.json").read_text() == payload
    assert json.loads(res.stdout)["size_bytes"] == len(payload)
