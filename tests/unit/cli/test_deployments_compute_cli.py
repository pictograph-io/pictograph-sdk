"""``pictograph deployments`` on PRT: a GPU, a precision, and the wait for ``active``.

A deployment runs on a GPU only (T4, L4 or A100), at a precision (fp32 or fp16). ``create``
answers at once, ``provisioning``; ``--wait`` (and the ``wait`` command, and ``resume --wait``)
polls until the deployment is ``active``. There is no CPU tier, so there is no ``--compute``:
a CPU, a retired tier or an unknown precision is refused before any request. ``update``
changes the autoscaling after create, sending only the flags given.

These drive the REAL client over a mocked HTTP transport: the command, the SDK call and
the request on the wire.
"""

from __future__ import annotations

import json
import time as _real_time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from pictograph.cli._app import app
from pictograph.cli._errors import ExitCode

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
API = f"{BASE}/api/v1/developer/deployments/"
MODEL = "abcdef01-2345-6789-abcd-ef0123456789"  # an id: no name lookup
DEP = "d1111111-1111-1111-1111-111111111111"
QUOTE = {
    "success": True,
    "quote": {
        "rate_per_min_micro_usd": 79320,
        "container_cost_per_hour_micro_usd": 4759200,
        "cost_per_hour_micro_usd": 4759200,
        "cost_per_day_micro_usd": 114220800,
        "warm_cost_per_hour_micro_usd": 0,
        "warm_cost_per_day_micro_usd": 0,
        "max_cost_per_hour_micro_usd": 4759200,
        "scale_to_zero": True,
        "billing_note": "Charged only while a container is up; it scales to zero when idle.",
    },
}
BUILDING = {
    "state": "building",
    "models": [{"model_id": MODEL, "precision": "fp32", "state": "building", "estimate_s": 85.8}],
}
BUILT = {
    "state": "built",
    "models": [{"model_id": MODEL, "precision": "fp32", "state": "built", "build_s": 84.2}],
}


def _dep(**over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": DEP,
        "organization_id": "org",
        "model_id": MODEL,
        "version_id": "v1",
        "name": "shelf",
        "status": "provisioning",
        "compute_type": "gpu",
        "gpu_type": "t4",
        "precision": "fp32",
        "min_containers": 0,
        "max_containers": 1,
        "scaledown_window": 60,
        "endpoint_url": "https://infer.test/d1111111/predict",
        "engine_build": None,
        "error_message": None,
    }
    row.update(over)
    return row


def _created(**over: Any) -> dict[str, Any]:
    return {"deployment": _dep(**over), "auth_token": "pk_deploy_once"}


def _plain(output: str) -> str:
    return " ".join(output.replace("│", " ").split())


class _FakeTime:
    """The ``time`` module the resource polls with: real clock, recorded (instant) sleeps,
    bounded so a wait that never stops fails instead of hanging."""

    def __init__(self) -> None:
        self.slept: list[float] = []

    @staticmethod
    def monotonic() -> float:
        return _real_time.monotonic()

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        assert len(self.slept) <= 50, "polled 50 times: it never stopped"


@pytest.fixture
def fake_time(monkeypatch: pytest.MonkeyPatch) -> _FakeTime:
    fake = _FakeTime()
    monkeypatch.setattr("pictograph.resources.deployments.time", fake)
    return fake


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_test")
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    return lambda *args: runner.invoke(app, ["deployments", *args])


# ───────────── create ─────────────


def test_create_defaults_to_a_t4_at_fp32_and_sends_no_compute_type(
    httpx_mock: HTTPXMock, run: Any
) -> None:
    httpx_mock.add_response(method="POST", url=API, json=_created())
    res = run("create", MODEL)
    assert res.exit_code == 0, res.output
    body = json.loads(httpx_mock.get_requests()[-1].content)
    assert (body["gpu_type"], body["precision"]) == ("t4", "fp32")
    assert "compute_type" not in body
    out = json.loads(res.stdout)
    assert out["auth_token"] == "pk_deploy_once" and out["status"] == "provisioning"  # noqa: S105
    assert (out["gpu_type"], out["precision"]) == ("t4", "fp32")
    assert out["endpoint_url"] == "https://infer.test/d1111111/predict"
    assert len(httpx_mock.get_requests()) == 1, "no --wait: no polling"


def test_create_on_an_l4_at_fp16_with_its_scaling(httpx_mock: HTTPXMock, run: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=API,
        json=_created(gpu_type="l4", precision="fp16", engine_build=BUILDING),
    )
    res = run(
        "create", MODEL, "--gpu", "l4", "--precision", "FP16", "--min-containers", "1",
        "--max-containers", "3", "--scaledown", "120", "--name", "shelf",
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    body = json.loads(httpx_mock.get_requests()[-1].content)
    assert (body["gpu_type"], body["precision"], body["name"]) == ("l4", "fp16", "shelf")
    assert (body["min_containers"], body["max_containers"], body["scaledown_window"]) == (1, 3, 120)
    assert json.loads(res.stdout)["engine_build"]["state"] == "building"


@pytest.mark.parametrize(
    ("args", "hint", "says"),
    [
        (("--gpu", "a10g"), "--gpu", "use t4, l4, a100"),
        (("--gpu", "cpu"), "--gpu", "There is no CPU tier"),
        (("--precision", "int8"), "--precision", "use fp32 or fp16"),
    ],
    ids=["retired-a10g", "cpu", "unknown-precision"],
)
def test_a_wrong_choice_is_refused_before_any_request(
    httpx_mock: HTTPXMock, run: Any, args: tuple[str, ...], hint: str, says: str
) -> None:
    res = run("create", MODEL, *args)
    assert res.exit_code == ExitCode.INVALID
    assert says in _plain(res.output) and hint in _plain(res.output)
    assert httpx_mock.get_requests() == []


@pytest.mark.parametrize("flag", ["--compute", "--compute-type"])
@pytest.mark.parametrize("command", ["create", "quote"])
def test_there_is_no_compute_flag(httpx_mock: HTTPXMock, run: Any, command: str, flag: str) -> None:
    res = run(command, *([MODEL] if command == "create" else []), flag, "cpu")
    assert res.exit_code == ExitCode.INVALID
    assert "No such option" in _plain(res.output)
    assert httpx_mock.get_requests() == []


# ───────────── create --wait ─────────────


def test_create_wait_polls_until_active_and_prints_the_endpoint_and_token(
    httpx_mock: HTTPXMock, run: Any, fake_time: _FakeTime
) -> None:
    httpx_mock.add_response(method="POST", url=API, json=_created(engine_build=BUILDING))
    for row in (
        _dep(engine_build=BUILDING),
        _dep(engine_build=BUILT),
        _dep(status="active", engine_build=BUILT),
    ):
        httpx_mock.add_response(method="GET", url=f"{API}{DEP}", json={"deployment": row})
    res = run("create", MODEL, "--wait", "--timeout", "600")
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    assert out["status"] == "active" and out["auth_token"] == "pk_deploy_once"  # noqa: S105
    assert out["endpoint_url"] == "https://infer.test/d1111111/predict"
    assert out["engine_build"]["state"] == "built"
    # It says what it waits on: the engine build, with its estimate.
    assert "TensorRT engine for t4 at fp32 first (about 86s)" in _plain(res.stderr)
    assert fake_time.slept == [3.0, 3.0]
    assert [r.method for r in httpx_mock.get_requests()] == ["POST", "GET", "GET", "GET"]


def test_create_wait_on_a_stored_engine_just_waits(
    httpx_mock: HTTPXMock, run: Any, fake_time: _FakeTime
) -> None:
    httpx_mock.add_response(method="POST", url=API, json=_created())
    httpx_mock.add_response(
        method="GET", url=f"{API}{DEP}", json={"deployment": _dep(status="active")}
    )
    res = run("create", MODEL, "--wait")
    assert res.exit_code == 0, res.output
    assert "Waiting for shelf to turn active" in _plain(res.stderr)
    assert fake_time.slept == []


def test_create_wait_that_fails_still_prints_the_token(
    httpx_mock: HTTPXMock, run: Any, fake_time: _FakeTime
) -> None:
    """The token is shown once: a deployment that fails while provisioning must not take
    it down with it."""
    httpx_mock.add_response(method="POST", url=API, json=_created())
    httpx_mock.add_response(
        method="GET",
        url=f"{API}{DEP}",
        json={"deployment": _dep(status="failed", error_message="Provisioning failed: no image")},
    )
    res = run("create", MODEL, "--wait", "--json")
    assert res.exit_code == ExitCode.FAILED
    assert json.loads(res.stdout)["auth_token"] == "pk_deploy_once"  # noqa: S105
    error = json.loads(res.stderr.strip().splitlines()[-1])["error"]
    assert error["code"] == "job_failed" and "no image" in error["message"]
    assert error["details"] == {"id": DEP, "status": "failed"}


def test_create_wait_that_runs_out_exits_still_running(
    httpx_mock: HTTPXMock, run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Clock:
        def __init__(self) -> None:
            self._ticks = [0.0, 0.0, 30.0]

        def monotonic(self) -> float:
            return self._ticks.pop(0) if len(self._ticks) > 1 else self._ticks[0]

        def sleep(self, _seconds: float) -> None:
            self._slept = getattr(self, "_slept", 0) + 1
            assert self._slept <= 50, "polled 50 times: it never stopped"

    monkeypatch.setattr("pictograph.resources.deployments.time", _Clock())
    httpx_mock.add_response(method="POST", url=API, json=_created(engine_build=BUILDING))
    httpx_mock.add_response(
        method="GET",
        url=f"{API}{DEP}",
        json={"deployment": _dep(engine_build=BUILDING)},
        is_reusable=True,
    )
    res = run("create", MODEL, "--wait", "--timeout", "20", "--json")
    assert res.exit_code == ExitCode.STILL_RUNNING
    assert json.loads(res.stdout)["auth_token"] == "pk_deploy_once"  # noqa: S105
    error = json.loads(res.stderr.strip().splitlines()[-1])["error"]
    assert error["code"] == "wait_timeout" and "building its TensorRT engine" in error["message"]


# ───────────── wait ─────────────


def test_wait_resolves_the_name_once_and_prints_the_active_deployment(
    httpx_mock: HTTPXMock, run: Any, fake_time: _FakeTime
) -> None:
    httpx_mock.add_response(
        url=f"{API}?limit=100&offset=0&name=shelf", json={"deployments": [_dep()]}
    )
    httpx_mock.add_response(method="GET", url=f"{API}{DEP}", json={"deployment": _dep()})
    httpx_mock.add_response(
        method="GET", url=f"{API}{DEP}", json={"deployment": _dep(status="active")}
    )
    res = run("wait", "shelf", "--poll-interval", "1.5")
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    assert out["status"] == "active" and "auth_token" not in out
    assert fake_time.slept == [1.5]


@pytest.mark.parametrize(
    ("status", "code"), [("paused", "job_cancelled"), ("terminated", "job_cancelled")]
)
def test_wait_on_a_deployment_that_will_not_come_up_fails_at_once(
    httpx_mock: HTTPXMock, run: Any, fake_time: _FakeTime, status: str, code: str
) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{API}{DEP}", json={"deployment": _dep(status=status)}
    )
    res = run("wait", DEP, "--json")
    assert res.exit_code == ExitCode.FAILED
    assert json.loads(res.stderr.strip().splitlines()[-1])["error"]["code"] == code
    assert fake_time.slept == []


# ───────────── resume --wait ─────────────


def test_resume_wait_waits_for_the_deployment_to_come_back(
    httpx_mock: HTTPXMock, run: Any, fake_time: _FakeTime
) -> None:
    asleep = {
        "state": "asleep",
        "containers": 0,
        "creating": 0,
        "waiting_since": None,
        "checked_at": "2026-10-23T12:00:00Z",
    }
    httpx_mock.add_response(method="POST", url=f"{API}{DEP}/resume", json={"deployment": _dep()})
    httpx_mock.add_response(
        method="GET", url=f"{API}{DEP}", json={"deployment": _dep(status="active", runtime=asleep)}
    )
    res = run("resume", DEP, "--wait")
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    assert (out["id"], out["name"], out["status"]) == (DEP, "shelf", "active")
    assert out["runtime"]["state"] == "asleep", (
        "what it is doing now, as get / wait / create --wait print"
    )


def test_resume_without_wait_answers_provisioning(httpx_mock: HTTPXMock, run: Any) -> None:
    httpx_mock.add_response(method="POST", url=f"{API}{DEP}/resume", json={"deployment": _dep()})
    res = run("resume", DEP)
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout) == {
        "id": DEP,
        "name": "shelf",
        "status": "provisioning",
        "gpu_type": "t4",
        "precision": "fp32",
        "endpoint_url": "https://infer.test/d1111111/predict",
    }
    assert len(httpx_mock.get_requests()) == 1


def test_bulk_resume_wait_waits_for_each_one_resumed(
    httpx_mock: HTTPXMock, run: Any, fake_time: _FakeTime
) -> None:
    other = "d2222222-2222-2222-2222-222222222222"
    httpx_mock.add_response(
        method="POST",
        url=f"{API}bulk-resume",
        json={"success": True, "succeeded": [DEP], "not_found": [other], "count": 1},
    )
    httpx_mock.add_response(method="GET", url=f"{API}{DEP}", json={"deployment": _dep()})
    httpx_mock.add_response(
        method="GET", url=f"{API}{DEP}", json={"deployment": _dep(status="active")}
    )
    res = run("resume", DEP, other, "--wait")
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["succeeded"] == [DEP]
    assert [str(r.url) for r in httpx_mock.get_requests()][1:] == [f"{API}{DEP}", f"{API}{DEP}"]


# ───────────── update: the autoscaling, after create ─────────────


@pytest.mark.parametrize(
    "flags",
    [
        ("--min-containers", "1", "--max-containers", "3", "--scaledown-window", "300"),
        ("--min", "1", "--max", "3", "--scaledown", "300"),
    ],
    ids=["long", "short"],
)
def test_update_sends_the_scaling_given_and_prints_it(
    httpx_mock: HTTPXMock, run: Any, flags: tuple[str, ...]
) -> None:
    scaled = _dep(
        status="active",
        min_containers=1,
        max_containers=3,
        scaledown_window=300,
        runtime=_rt("live", containers=1),
    )
    httpx_mock.add_response(method="PATCH", url=f"{API}{DEP}", json={"deployment": scaled})
    res = run("update", DEP, *flags)
    assert res.exit_code == 0, res.output
    body = json.loads(httpx_mock.get_requests()[-1].content)
    assert body == {"min_containers": 1, "max_containers": 3, "scaledown_window": 300}
    out = json.loads(res.stdout)
    assert (out["min_containers"], out["max_containers"], out["scaledown_window"]) == (1, 3, 300)
    assert out["status"] == "active" and out["runtime"]["state"] == "live"
    assert res.stderr == "", "a live deployment took it at once: nothing more to say"


def test_update_sends_only_the_flag_given_a_zero_included(httpx_mock: HTTPXMock, run: Any) -> None:
    httpx_mock.add_response(
        url=f"{API}?limit=100&offset=0&name=shelf", json={"deployments": [_dep()]}
    )
    httpx_mock.add_response(
        method="PATCH", url=f"{API}{DEP}", json={"deployment": _dep(status="active")}
    )
    res = run("update", "shelf", "--min-containers", "0")
    assert res.exit_code == 0, res.output
    lookup, patch = httpx_mock.get_requests()
    assert (lookup.method, patch.method, str(patch.url)) == ("GET", "PATCH", f"{API}{DEP}")
    assert json.loads(patch.content) == {"min_containers": 0}


@pytest.mark.parametrize(
    ("status", "says"),
    [("paused", "takes this scaling when it is resumed"), ("provisioning", "once it is up")],
)
def test_update_says_when_a_deployment_not_running_takes_it(
    httpx_mock: HTTPXMock, run: Any, status: str, says: str
) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{API}{DEP}",
        json={"deployment": _dep(status=status, max_containers=2)},
    )
    res = run("update", DEP, "--max-containers", "2")
    assert res.exit_code == 0, res.output
    assert f"shelf is {status}" in _plain(res.stderr) and says in _plain(res.stderr)
    out = json.loads(res.stdout)
    assert out["status"] == status and out["max_containers"] == 2


def test_update_with_nothing_to_change_is_refused_before_any_request(
    httpx_mock: HTTPXMock, run: Any
) -> None:
    res = run("update", DEP, "--json")
    assert res.exit_code == ExitCode.INVALID
    error = json.loads(res.stderr.strip().splitlines()[-1])["error"]
    # The CLI's own words - its flags - not the SDK's Python keyword names.
    assert error["code"] == "usage_error"
    assert "--min-containers" in error["message"] and "--scaledown-window" in error["message"]
    assert httpx_mock.get_requests() == []


@pytest.mark.parametrize(
    "flags", [("--min-containers", "-1"), ("--max-containers", "0")], ids=["min", "max"]
)
def test_update_refuses_a_count_below_its_floor_before_any_request(
    httpx_mock: HTTPXMock, run: Any, flags: tuple[str, ...]
) -> None:
    res = run("update", DEP, *flags)
    assert res.exit_code == ExitCode.INVALID
    assert flags[0] in _plain(res.output)
    assert httpx_mock.get_requests() == []


def test_create_takes_scaledown_window_like_update(httpx_mock: HTTPXMock, run: Any) -> None:
    httpx_mock.add_response(method="POST", url=API, json=_created())
    res = run("create", MODEL, "--scaledown-window", "300")
    assert res.exit_code == 0, res.output
    assert json.loads(httpx_mock.get_requests()[-1].content)["scaledown_window"] == 300


# ───────────── quote ─────────────


def test_a_bare_quote_prices_the_default_gpu(httpx_mock: HTTPXMock, run: Any) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{API}quote?gpu_type=t4&min_containers=0&max_containers=1", json=QUOTE
    )
    res = run("quote")
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    assert out["rate_per_min_micro_usd"] == 79320 and out["max_cost_per_hour_micro_usd"] == 4759200


@pytest.mark.parametrize(
    "args",
    [
        ("--gpu", "a100", "--min-containers", "1", "--max-containers", "4"),
        ("--gpu", "a100", "--min", "1", "--max", "4"),
    ],
    ids=["long", "short"],
)
def test_a_quote_takes_the_gpu_and_both_bounds(
    httpx_mock: HTTPXMock, run: Any, args: tuple[str, ...]
) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{API}quote?gpu_type=a100&min_containers=1&max_containers=4", json=QUOTE
    )
    res = run("quote", *args)
    assert res.exit_code == 0, res.output


def test_a_quote_refuses_a_cpu_before_any_request(httpx_mock: HTTPXMock, run: Any) -> None:
    res = run("quote", "--gpu", "cpu")
    assert res.exit_code == ExitCode.INVALID and "There is no CPU tier" in _plain(res.output)
    assert httpx_mock.get_requests() == []


# ───────────── list / get / compute-options ─────────────


def test_list_shows_the_precision_and_the_engine_build(
    httpx_mock: HTTPXMock, run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    rows = [
        _dep(name="building-one", engine_build=BUILDING),
        _dep(id="d2222222-2222-2222-2222-222222222222", name="built-one", status="active",
             precision="fp16", engine_build=BUILT),
        _dep(id="d3333333-3333-3333-3333-333333333333", name="plain-one", status="active"),
    ]  # fmt: skip
    httpx_mock.add_response(
        method="GET",
        url=f"{API}?limit=50&offset=0",
        json={"deployments": rows, "pagination": {"limit": 50, "offset": 0}},
    )
    res = run("list")
    assert res.exit_code == 0, res.output
    text = _plain(res.stdout)
    assert "precision" in text.lower() and "fp16" in text and "fp32" in text
    assert "building (about 86s)" in text and "built in 84s" in text


def test_get_prints_the_whole_deployment_with_its_build(httpx_mock: HTTPXMock, run: Any) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{API}{DEP}", json={"deployment": _dep(engine_build=BUILDING)}
    )
    res = run("get", DEP)
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    assert out["precision"] == "fp32" and out["engine_build"]["models"][0]["estimate_s"] == 85.8


PRECISIONS = [
    {"precision": "fp32", "label": "FP32", "description": "32-bit, the model as trained"},
    {"precision": "fp16", "label": "FP16", "description": "16-bit, faster, accuracy-equivalent"},
]


def test_compute_options_lists_each_gpus_container(
    httpx_mock: HTTPXMock, run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    option = {
        "key": "a100",
        "label": "A100 GPU",
        "compute_type": "gpu",
        "gpu_type": "a100",
        "is_gpu": True,
        "description": "Top-end.",
        "vram_gb": 40,
        "cpu_cores": 8.0,
        "memory_gib": 32.0,
        "rate_per_min_micro_usd": 239140,
        "cost_per_hour_micro_usd": 14348400,
    }
    httpx_mock.add_response(
        method="GET",
        url=f"{API}compute-options",
        json={"success": True, "options": [option], "precisions": PRECISIONS},
        is_reusable=True,
    )
    res = run("compute-options")
    assert res.exit_code == 0, res.output
    text = _plain(res.stdout)
    assert "40 GB" in text and "32 GiB" in text and "239140" in text and "14348400" in text
    # And the precisions create takes, the default marked.
    assert "Precisions" in text and "32-bit, the model as trained" in text
    assert "16-bit, faster, accuracy-equivalent" in text
    fp32_row = next(line for line in res.stdout.splitlines() if "FP32" in line)
    assert "yes" in fp32_row
    # --json prints both lists.
    out = json.loads(run("compute-options", "--json").stdout)
    assert [o["key"] for o in out["options"]] == ["a100"]
    assert [p["precision"] for p in out["precisions"]] == ["fp32", "fp16"]


def test_compute_options_without_precisions_prints_only_the_gpus(
    httpx_mock: HTTPXMock, run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    option = {"key": "t4", "label": "T4 GPU", "gpu_type": "t4", "rate_per_min_micro_usd": 79320}
    httpx_mock.add_response(
        method="GET", url=f"{API}compute-options", json={"success": True, "options": [option]}
    )
    res = run("compute-options")
    assert res.exit_code == 0, res.output
    assert "T4 GPU" in res.stdout and "Precisions" not in res.stdout


def test_a_legacy_deployment_on_a_retired_tier_still_reads(httpx_mock: HTTPXMock, run: Any) -> None:
    """The type a caller CHOOSES from is the current tiers; what a deployment REPORTS is
    read as-is, so a row made on A10G (or on a CPU) before PRT still lists."""
    rows = [
        _dep(gpu_type="a10g", status="paused"),
        _dep(compute_type="cpu", gpu_type=None, status="paused"),
    ]
    httpx_mock.add_response(
        method="GET",
        url=f"{API}?limit=50&offset=0",
        json={"deployments": rows, "pagination": {"total": 2, "limit": 50, "offset": 0}},
    )
    res = run("list", "--json")
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    assert out[0]["gpu_type"] == "a10g" and out[1]["compute_type"] == "cpu"


# ───────────── the live state: what an active deployment is doing right now ─────────────

CHECKED = "2026-10-23T12:05:00+00:00"


def _rt(state: str, **over: Any) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "state": state,
        "containers": 0,
        "creating": 0,
        "waiting_since": None,
        "checked_at": CHECKED,
    }
    runtime.update(over)
    return runtime


@pytest.mark.parametrize(
    ("status", "runtime", "shown"),
    [
        ("active", _rt("live", containers=2), "active · live (2)"),
        ("active", _rt("live", containers=1, creating=1), "active · live (1, 1 starting)"),
        ("active", _rt("starting", creating=1), "active · starting (1)"),
        (
            "active",
            _rt("waiting", waiting_since="2026-10-23T12:03:18+00:00"),
            "active · waiting for a GPU 1m42s",
        ),
        ("active", _rt("waiting"), "active · waiting for a GPU"),
        ("active", _rt("asleep"), "active · asleep"),
        ("active", _rt("unknown"), "active · unknown"),
        ("active", _rt("hibernating"), "active · hibernating"),
        ("active", None, "active"),
        ("paused", None, "paused"),
        ("provisioning", None, "provisioning"),
        # A runtime is only ever an active deployment's: one on another status is not shown.
        ("paused", _rt("asleep"), "paused"),
    ],
    ids=[
        "live",
        "live-and-starting",
        "starting",
        "waiting",
        "waiting-no-start",
        "asleep",
        "unknown",
        "a-later-state",
        "active-no-runtime",
        "paused",
        "provisioning",
        "paused-with-a-stale-runtime",
    ],
)
def test_list_shows_the_live_state_beside_the_status(
    httpx_mock: HTTPXMock,
    run: Any,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    runtime: dict[str, Any] | None,
    shown: str,
) -> None:
    monkeypatch.setenv("COLUMNS", "250")
    httpx_mock.add_response(
        method="GET",
        url=f"{API}?limit=50&offset=0",
        json={"deployments": [_dep(status=status, runtime=runtime)]},
    )
    res = run("list")
    assert res.exit_code == 0, res.output
    row = next(line for line in res.stdout.splitlines() if "shelf" in line)
    cells = [c.strip() for c in row.split("│") if c.strip()]
    assert shown in cells, cells


def test_get_prints_the_runtime(httpx_mock: HTTPXMock, run: Any) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{API}{DEP}",
        json={"deployment": _dep(status="active", runtime=_rt("asleep"))},
    )
    res = run("get", DEP)
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    assert out["runtime"]["state"] == "asleep" and out["runtime"]["containers"] == 0


def test_wait_prints_the_runtime_of_the_deployment_it_waited_for(
    httpx_mock: HTTPXMock, run: Any, fake_time: _FakeTime
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{API}{DEP}",
        json={"deployment": _dep(status="active", runtime=_rt("live", containers=1))},
    )
    res = run("wait", DEP)
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["runtime"]["state"] == "live"


@pytest.mark.parametrize(
    ("since", "checked", "shown"),
    [
        ("2026-10-23T12:04:58+00:00", CHECKED, "2s"),
        ("2026-10-23T12:04:00+00:00", CHECKED, "1m00s"),
        ("2026-10-23T10:59:00+00:00", CHECKED, "1h06m"),
        ("2026-10-23T12:06:00+00:00", CHECKED, "0s"),  # a clock ahead never reads negative
    ],
)
def test_the_wait_is_measured_on_the_servers_clock(since: str, checked: str, shown: str) -> None:
    from pictograph.cli.commands.deployments import _waited
    from pictograph.models.deployment import DeploymentRuntime

    runtime = DeploymentRuntime.model_validate(
        _rt("waiting", waiting_since=since, checked_at=checked)
    )
    assert _waited(runtime) == shown


def test_the_wait_without_a_server_clock_is_measured_from_now() -> None:
    from datetime import datetime, timedelta, timezone

    from pictograph.cli.commands.deployments import _waited
    from pictograph.models.deployment import DeploymentRuntime

    since = (datetime.now(timezone.utc) - timedelta(seconds=125)).isoformat()
    runtime = DeploymentRuntime.model_validate({"state": "waiting", "waiting_since": since})
    assert _waited(runtime) in ("2m05s", "2m06s")
