"""How the CLI fails (B533 P0-2): one exit code per kind of failure, and a JSON error.

Until 1.69.118 every SDK error exited 1 with red text on stderr, so "out of credits",
"not found", "rate limited" and "the plan does not include this" looked identical to
a program, and an agent driving Pictograph through the CLI had to parse prose to
decide whether to retry, ask a person or fix its arguments.

These drive the REAL app and the REAL client over a mocked HTTP transport: each test
covers the response the API sends, the SDK exception it becomes, and what the process
prints and exits with - through ``main``, the ``pictograph`` script itself.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import httpx
import pytest
import typer
from typer.main import get_command
from typer.testing import CliRunner

import pictograph.cli.commands as commands_pkg
from pictograph.cli._app import app, main
from pictograph.cli._errors import (
    ExitCode,
    PictographCli,
    classify,
    json_defaults,
    json_requested,
)

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
DATASETS = f"{BASE}/api/v1/developer/datasets/"
RUN_ID = "0f6a3b1c-2d4e-4f5a-8b9c-0d1e2f3a4b5c"
ERROR_KEYS = {
    "code",
    "message",
    "exit_code",
    "type",
    "status",
    "retry_after",
    "details",
    "fix",
    "docs_url",
    "request_id",
}


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Any:
    """Run ``pictograph <args>`` through ``main`` as a signed-in user of the mocked API.

    Returns ``(exit code, stdout, stderr)``.
    """
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_test")
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    monkeypatch.setenv("PICTOGRAPH_MAX_RETRIES", "0")  # a 5xx is reported, not retried
    monkeypatch.delenv("PICTOGRAPH_OUTPUT", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    def run(*args: str) -> tuple[int, str, str]:
        code = main(list(args))
        out = capsys.readouterr()
        return code, out.out, out.err

    return run


def _json_error(stderr: str) -> dict[str, Any]:
    """The JSON error: the LAST line on stderr, one object with the whole contract."""
    last = stderr.strip().splitlines()[-1]
    payload = json.loads(last)
    assert set(payload) == {"error"}
    error: dict[str, Any] = payload["error"]
    assert set(error) == ERROR_KEYS, "every key is always present"
    return error


def _envelope(code: str, message: str, details: Any = None) -> dict[str, Any]:
    """The API's error body: the envelope plus the legacy ``detail``."""
    return {"error": {"code": code, "message": message, "details": details}, "detail": message}


# ───────────── every kind of API failure has its own exit code ─────────────

API_FAILURES = [
    # status, body, headers, exit code, code an agent branches on
    (401, _envelope("unauthorized", "Invalid API key."), {}, ExitCode.AUTH, "unauthorized"),
    (
        403,
        _envelope("feature_not_available", "Your plan does not include this."),
        {},
        ExitCode.FORBIDDEN,
        "feature_not_available",
    ),
    (404, _envelope("not_found", "No such dataset."), {}, ExitCode.NOT_FOUND, "not_found"),
    (
        409,
        _envelope("dataset_cold", "The dataset is in cold storage."),
        {},
        ExitCode.CONFLICT,
        "dataset_cold",
    ),
    (
        422,
        _envelope(
            "validation_error",
            "The request was invalid.",
            {"errors": [{"loc": ["query", "limit"], "msg": "too large"}]},
        ),
        {},
        ExitCode.INVALID,
        "validation_error",
    ),
    (
        402,
        _envelope(
            "insufficient_credits",
            "Not enough compute credit.",
            {"required_micro_usd": 5_000_000, "remaining_micro_usd": 0, "unit": "micro_usd"},
        ),
        {},
        ExitCode.PAYMENT,
        "insufficient_credits",
    ),
    (
        429,
        _envelope("rate_limited", "Too many requests."),
        {"Retry-After": "3600"},
        ExitCode.RATE_LIMITED,
        "rate_limited",
    ),
    (500, _envelope("internal_error", "Boom."), {}, ExitCode.UNAVAILABLE, "internal_error"),
    # A response from before the envelope: the code falls back to the kind's own.
    (400, {"detail": "bad input"}, {}, ExitCode.INVALID, "validation_error"),
    (503, {"detail": "down"}, {}, ExitCode.UNAVAILABLE, "server_error"),
]


@pytest.mark.parametrize(
    ("status", "body", "headers", "exit_code", "code"),
    API_FAILURES,
    ids=[str(f[0]) + "-" + f[4] for f in API_FAILURES],
)
def test_an_api_failure_exits_with_its_code_and_says_why(
    cli: Any,
    httpx_mock: HTTPXMock,
    status: int,
    body: dict[str, Any],
    headers: dict[str, str],
    exit_code: ExitCode,
    code: str,
) -> None:
    httpx_mock.add_response(status_code=status, json=body, headers=headers)
    rc, out, err = cli("datasets", "list")
    assert rc == exit_code
    assert out == ""
    assert "error:" in err and "Traceback" not in err
    message = body["error"]["message"] if "error" in body else body["detail"]
    assert message in " ".join(err.split())


@pytest.mark.parametrize(
    ("status", "body", "headers", "exit_code", "code"),
    API_FAILURES,
    ids=[str(f[0]) + "-" + f[4] for f in API_FAILURES],
)
def test_with_json_the_failure_is_one_json_line_on_stderr(
    cli: Any,
    httpx_mock: HTTPXMock,
    status: int,
    body: dict[str, Any],
    headers: dict[str, str],
    exit_code: ExitCode,
    code: str,
) -> None:
    httpx_mock.add_response(
        status_code=status, json=body, headers={"X-Request-Id": "req-123", **headers}
    )
    rc, out, err = cli("datasets", "list", "--json")
    assert rc == exit_code
    assert out == "", "stdout stays clean for a pipe into jq"
    error = _json_error(err)
    assert error["code"] == code
    assert error["exit_code"] == int(exit_code)
    assert error["status"] == status
    assert error["request_id"] == "req-123"
    assert error["message"] == (body["error"]["message"] if "error" in body else body["detail"])
    assert error["fix"], "every API failure says what to do"
    assert error["retry_after"] == (3600.0 if status == 429 else None)
    if status == 402:
        assert error["details"]["required_micro_usd"] == 5_000_000
        assert error["details"]["remaining_micro_usd"] == 0
    if status == 422:
        assert error["details"] == {"errors": [{"loc": ["query", "limit"], "msg": "too large"}]}


def test_a_network_failure_is_unavailable(cli: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))
    rc, _, err = cli("datasets", "list", "--json")
    assert rc == ExitCode.UNAVAILABLE
    error = _json_error(err)
    assert error["code"] == "network_error" and error["type"] == "NetworkError"
    assert error["status"] is None


def test_pictograph_output_json_prints_the_error_as_json_on_any_command(
    cli: Any, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`datasets get` has no --json flag; the environment still asks for a JSON error."""
    monkeypatch.setenv("PICTOGRAPH_OUTPUT", "json")
    httpx_mock.add_response(status_code=404, json=_envelope("not_found", "No such dataset."))
    rc, out, err = cli("datasets", "get", "road-signs")
    assert rc == ExitCode.NOT_FOUND
    assert out == ""
    assert _json_error(err)["code"] == "not_found"


def test_pictograph_output_json_also_turns_every_json_flag_on(
    cli: Any, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PICTOGRAPH_OUTPUT", "JSON")  # any case
    httpx_mock.add_response(
        json={
            "data": [
                {
                    "id": "p1",
                    "name": "road-signs",
                    "image_count": 4,
                    "created_at": "2026-04-01T00:00:00Z",
                }
            ],
            "pagination": {"total": 1, "limit": 100, "offset": 0, "has_more": False},
        }
    )
    rc, out, _ = cli("datasets", "list")
    assert rc == ExitCode.OK
    rows = json.loads(out)
    assert [r["name"] for r in rows] == ["road-signs"], "JSON, not the table"


def test_every_json_flag_is_turned_on_by_the_environment() -> None:
    """The default map reaches every `--json` in the tree - one added later included."""
    root = get_command(app)

    def flags(command: Any, path: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
        found = {
            (*path, p.name)
            for p in getattr(command, "params", ())
            if "--json" in (getattr(p, "opts", None) or ())
        }
        for name, sub in (getattr(command, "commands", None) or {}).items():
            found |= flags(sub, (*path, name))
        return found

    def leaves(mapping: dict[str, Any], path: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
        found: set[tuple[str, ...]] = set()
        for key, value in mapping.items():
            if isinstance(value, dict):
                found |= leaves(value, (*path, key))
            else:
                assert value is True
                found.add((*path, key))
        return found

    expected = flags(root)
    assert len(expected) >= 30, "the walk found the CLI's --json flags"
    assert leaves(json_defaults(root)) == expected


def test_every_command_group_fails_the_same_way() -> None:
    """A group run on its own (a test, ``python -m``) follows the same contract."""
    assert app.info.cls is PictographCli
    groups = 0
    for info in pkgutil.iter_modules(commands_pkg.__path__):
        module = importlib.import_module(f"{commands_pkg.__name__}.{info.name}")
        sub = getattr(module, "app", None)
        if isinstance(sub, typer.Typer):
            groups += 1
            assert sub.info.cls is PictographCli, f"{info.name}.app is a plain typer.Typer"
    assert groups >= 20


# ───────────── failures the CLI finds itself ─────────────


def test_no_api_key_is_an_auth_failure(cli: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PICTOGRAPH_API_KEY")
    rc, _, err = cli("datasets", "list", "--json")
    assert rc == ExitCode.AUTH
    error = _json_error(err)
    assert error["code"] == "no_api_key" and "pictograph login" in error["message"]


def test_a_usage_error_is_invalid_input(cli: Any) -> None:
    rc, out, err = cli("datasets", "list", "--limit", "0", "--json")
    assert rc == ExitCode.INVALID
    assert out == ""
    error = _json_error(err)
    assert error["code"] == "usage_error"
    assert "--limit" in error["details"]["param"]
    assert error["fix"] == "Run `pictograph datasets list --help` for its arguments and options."


def test_a_usage_error_for_a_person_keeps_typers_rendering(cli: Any) -> None:
    rc, _, err = cli("datasets", "get")
    assert rc == ExitCode.INVALID
    assert "Missing argument" in err and "Try 'pictograph datasets get --help'" in err
    assert not err.strip().startswith("{")


def test_an_unknown_option_is_invalid_input(cli: Any) -> None:
    rc, _, err = cli("datasets", "list", "--nope", "--json")
    assert rc == ExitCode.INVALID
    assert "No such option: --nope" in _json_error(err)["message"]


def test_a_group_with_no_command_lists_what_it_takes(
    cli: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PICTOGRAPH_OUTPUT", "json")
    rc, _, err = cli("datasets")
    assert rc == ExitCode.INVALID
    error = _json_error(err)
    assert error["code"] == "missing_command"
    assert {"list", "get", "create"} <= set(error["details"]["commands"])


def test_a_malformed_annotation_file_is_invalid_input_with_its_fields(
    cli: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PICTOGRAPH_OUTPUT", "json")  # `annotations save` has no --json
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"name": "car", "type": "bbox"}]))  # no bounding_box
    rc, _, err = cli("annotations", "save", "ds", "a.jpg", "--file", str(bad))
    assert rc == ExitCode.INVALID
    error = _json_error(err)
    assert error["code"] == "invalid_input" and error["type"] == "ValidationError"
    assert error["details"]["errors"], "pydantic's per-field errors ride along"


def test_a_file_that_is_not_json_is_invalid_input(cli: Any, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    rc, _, err = cli("annotations", "save", "ds", "a.jpg", "--file", str(bad))
    assert rc == ExitCode.INVALID
    assert "error:" in err and "Traceback" not in err


def test_a_missing_file_is_invalid_input(
    cli: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PICTOGRAPH_OUTPUT", "json")
    rc, _, err = cli("annotations", "save", "ds", "a.jpg", "--file", str(tmp_path / "nope.json"))
    assert rc == ExitCode.INVALID
    error = _json_error(err)
    assert error["code"] == "file_not_found" and "nope.json" in error["message"]


def test_a_declined_confirmation_says_how_to_confirm_in_advance() -> None:
    """No terminal to ask on (an agent): the prompt reads EOF and the command aborts."""
    result = CliRunner().invoke(
        app,
        ["train", "cancel", RUN_ID],
        input="",
        env={"PICTOGRAPH_API_KEY": "pk_x", "PICTOGRAPH_OUTPUT": "json"},
    )
    assert result.exit_code == ExitCode.FAILED
    error = _json_error(result.stderr)
    assert error["code"] == "aborted" and "--yes" in error["message"]


# ───────────── jobs: ended without succeeding, or still running ─────────────


def _run_payload(status: str, error_message: str | None = None) -> dict[str, Any]:
    return {
        "data": {
            "id": RUN_ID,
            "organization_id": "org",
            "name": "cereal-detector",
            "pipeline_type": "yolox",
            "status": status,
            "progress": 40,
            "error_message": error_message,
            "created_at": "2026-04-01T00:00:00Z",
        }
    }


@pytest.mark.parametrize(
    ("status", "code"), [("failed", "job_failed"), ("cancelled", "job_cancelled")]
)
def test_a_job_that_ended_without_succeeding_is_failed_with_its_id(
    cli: Any, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch, status: str, code: str
) -> None:
    monkeypatch.setenv("PICTOGRAPH_OUTPUT", "json")  # `train wait` prints JSON already
    httpx_mock.add_response(json=_run_payload(status, "CUDA out of memory"))
    rc, out, err = cli("train", "wait", RUN_ID, "--poll-interval", "0.01")
    assert rc == ExitCode.FAILED
    assert out == ""
    error = _json_error(err)
    assert error["code"] == code
    assert error["details"] == {"id": RUN_ID, "status": status}


def test_a_wait_that_runs_out_is_still_running(
    cli: Any, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The job continues: retrying the command that STARTED it would start a second one."""
    monkeypatch.setenv("PICTOGRAPH_OUTPUT", "json")
    httpx_mock.add_response(json=_run_payload("running"), is_reusable=True)
    rc, _, err = cli("train", "wait", RUN_ID, "--poll-interval", "0.01", "--timeout", "0.05")
    assert rc == ExitCode.STILL_RUNNING
    assert _json_error(err)["code"] == "wait_timeout"


# ───────────── the parts ─────────────


def test_json_is_requested_by_the_flag_before_a_double_dash_or_by_the_environment() -> None:
    assert json_requested(["datasets", "list", "--json"], environ={})
    assert not json_requested(["datasets", "list"], environ={})
    assert not json_requested(["workflows", "invoke", "w", "--", "--json"], environ={})
    assert json_requested(["datasets", "get", "x"], environ={"PICTOGRAPH_OUTPUT": " Json "})
    assert not json_requested(["datasets", "get", "x"], environ={"PICTOGRAPH_OUTPUT": "table"})


def test_a_bug_keeps_its_traceback() -> None:
    """Only failures the CLI owns are rendered; anything else is a bug and stays loud."""
    assert classify(KeyError("x")) is None
    assert classify(AttributeError("x")) is None
    runner = CliRunner()
    with patch("pictograph.cli.commands.datasets.get_client", side_effect=KeyError("boom")):
        result = runner.invoke(app, ["datasets", "list"], env={"PICTOGRAPH_API_KEY": "pk_x"})
    assert isinstance(result.exception, KeyError)


def test_main_returns_zero_on_success(cli: Any) -> None:
    rc, out, _ = cli("--version")
    assert rc == ExitCode.OK and out.startswith("pictograph ")


def test_the_exit_codes_are_the_documented_table() -> None:
    """The numbers are a contract: agents and scripts hard-code them."""
    assert {c.name: int(c) for c in ExitCode} == {
        "OK": 0,
        "FAILED": 1,
        "AUTH": 2,
        "FORBIDDEN": 3,
        "NOT_FOUND": 4,
        "CONFLICT": 5,
        "INVALID": 6,
        "PAYMENT": 7,
        "RATE_LIMITED": 8,
        "UNAVAILABLE": 9,
        "STILL_RUNNING": 10,
        "INTERRUPTED": 130,
    }


def test_the_skill_and_agents_md_teach_every_exit_code() -> None:
    """The agent learns the table from the skill and the AGENTS.md `init` writes; a
    code added to ExitCode without them is a code no agent knows to branch on."""
    import re

    from pictograph.cli.commands.init import _AGENTS_MD_TEMPLATE
    from pictograph.skills import skill_path

    skill = (skill_path("pictograph-cv") / "SKILL.md").read_text(encoding="utf-8")
    section = skill.split("## When calls fail", 1)[1].split("\n## ", 1)[0]
    rows = {int(m) for m in re.findall(r"^\| (\d+) \|", section, flags=re.MULTILINE)}
    taught = {int(c) for c in ExitCode if c is not ExitCode.INTERRUPTED}
    assert rows == taught

    agents = _AGENTS_MD_TEMPLATE.split("## When things fail", 1)[1]
    listed = {int(m) for m in re.findall(r"^- (\d+):", agents, flags=re.MULTILINE)}
    assert listed == taught - {int(ExitCode.OK)}
