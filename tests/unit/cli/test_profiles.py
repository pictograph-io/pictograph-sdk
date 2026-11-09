"""Profiles: one ``~/.pictograph/config.toml`` table per API key (B533 #16).

An API key belongs to one organization, so an agent working for several needs several keys
and a way to pick one per command. These drive the REAL app and the REAL client over a mocked
HTTP transport and read which key each request carried - the resolution order is the
contract: ``--api-key`` > ``--profile`` > ``PICTOGRAPH_API_KEY`` > ``PICTOGRAPH_PROFILE`` >
``[default]``.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from pictograph.cli import _config
from pictograph.cli._app import app
from pictograph.cli._config import load_config, write_config
from pictograph.cli._errors import ExitCode

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
ME = f"{BASE}/api/v1/developer/credits/balance"
DEFAULT_KEY = "pk_live_default_0000000000000000"
RESEARCH_KEY = "pk_live_research_000000000000000"
ENV_KEY = "pk_live_environment_000000000000"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    for var in (
        "PICTOGRAPH_API_KEY",
        "PICTOGRAPH_PROFILE",
        "PICTOGRAPH_BASE_URL",
        "PICTOGRAPH_OUTPUT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PICTOGRAPH_MAX_RETRIES", "0")
    monkeypatch.setattr(_config, "CONFIG_DIR", fake_home / ".pictograph")
    monkeypatch.setattr(_config, "CONFIG_PATH", fake_home / ".pictograph" / "config.toml")
    monkeypatch.setattr(_config, "_flag_profile", None)
    return fake_home


@pytest.fixture
def run() -> Any:
    runner = CliRunner()
    return lambda *args: runner.invoke(app, list(args))


def _two_profiles() -> None:
    write_config(api_key=DEFAULT_KEY, base_url=BASE)
    write_config(api_key=RESEARCH_KEY, base_url=BASE, profile="research")


def _balance(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=ME,
        json={
            "organization_id": "o",
            "included_allowance_micro_usd": 5_000_000,
            "included_remaining_micro_usd": 5_000_000,
            "period_spend_micro_usd": 0,
            "budget_micro_usd": 5_000_000,
        },
        is_optional=True,
    )


def _sent_key(httpx_mock: HTTPXMock) -> str:
    (request,) = httpx_mock.get_requests()
    return request.headers["X-API-Key"]


# ── the file ───────────────────────────────────────────────────────────────────


def test_a_second_profile_keeps_the_first_and_the_file_stays_private(home: Path) -> None:
    write_config(api_key=DEFAULT_KEY)
    path = write_config(
        api_key=RESEARCH_KEY, base_url="https://staging.example", profile="research"
    )
    assert load_config().api_key == DEFAULT_KEY
    assert load_config("research") == _config.CliConfig(RESEARCH_KEY, "https://staging.example")
    assert path.read_text().splitlines()[0] == "[default]"  # default first, the rest sorted
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    write_config(api_key=DEFAULT_KEY + "x")  # a profile written again replaces only itself
    assert load_config("research").api_key == RESEARCH_KEY


@pytest.mark.parametrize("name", ["a.b", "", "-x", "has space", "x" * 65])
def test_a_name_toml_would_misread_is_refused(home: Path, name: str) -> None:
    with pytest.raises(ValueError, match="cannot be a profile name"):
        write_config(api_key=DEFAULT_KEY, profile=name)


def test_a_value_is_escaped_not_injected(home: Path) -> None:
    write_config(api_key='pk_live_"quoted\\', profile="odd")
    assert load_config("odd").api_key == 'pk_live_"quoted\\'
    with pytest.raises(ValueError, match="control characters"):
        write_config(api_key="pk_live_x\n[default]\napi_key = 'stolen'", profile="odd")


# ── which key a command runs with ──────────────────────────────────────────────


def test_the_flag_runs_as_its_profile(home: Path, run: Any, httpx_mock: HTTPXMock) -> None:
    _two_profiles()
    _balance(httpx_mock)
    res = run("--profile", "research", "credits", "balance", "--json")
    assert res.exit_code == ExitCode.OK, res.output
    assert _sent_key(httpx_mock) == RESEARCH_KEY


def test_the_flag_beats_the_environment_key(
    home: Path, run: Any, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_profiles()
    monkeypatch.setenv("PICTOGRAPH_API_KEY", ENV_KEY)
    _balance(httpx_mock)
    assert run("--profile", "research", "credits", "balance", "--json").exit_code == ExitCode.OK
    assert _sent_key(httpx_mock) == RESEARCH_KEY


def test_the_environment_key_beats_the_environment_profile(
    home: Path, run: Any, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_profiles()
    monkeypatch.setenv("PICTOGRAPH_API_KEY", ENV_KEY)
    monkeypatch.setenv("PICTOGRAPH_PROFILE", "research")
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    _balance(httpx_mock)
    assert run("credits", "balance", "--json").exit_code == ExitCode.OK
    assert _sent_key(httpx_mock) == ENV_KEY


def test_the_environment_profile_beats_the_default(
    home: Path, run: Any, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_profiles()
    monkeypatch.setenv("PICTOGRAPH_PROFILE", "research")
    _balance(httpx_mock)
    assert run("credits", "balance", "--json").exit_code == ExitCode.OK
    assert _sent_key(httpx_mock) == RESEARCH_KEY


def test_an_explicit_key_beats_every_profile(home: Path, run: Any, httpx_mock: HTTPXMock) -> None:
    _two_profiles()
    _balance(httpx_mock)
    res = run("--profile", "research", "credits", "balance", "--json", "--api-key", ENV_KEY)
    assert res.exit_code == ExitCode.OK, res.output
    assert _sent_key(httpx_mock) == ENV_KEY


def test_the_flag_lasts_one_invocation(home: Path, run: Any, httpx_mock: HTTPXMock) -> None:
    _two_profiles()
    _balance(httpx_mock)
    _balance(httpx_mock)
    run("--profile", "research", "credits", "balance", "--json")
    run("credits", "balance", "--json")
    first, second = httpx_mock.get_requests()
    assert (first.headers["X-API-Key"], second.headers["X-API-Key"]) == (RESEARCH_KEY, DEFAULT_KEY)


def test_a_profile_without_a_base_url_never_takes_the_ambient_one(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under --profile everything comes from that profile: a PICTOGRAPH_BASE_URL in the
    environment would otherwise send this profile's key to another server."""
    write_config(api_key=RESEARCH_KEY, profile="research")
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", "https://elsewhere.example")
    _config.set_flag_profile("research")
    assert _config.resolve_base_url() is None
    assert _config.resolve_api_key() == RESEARCH_KEY


# ── refusals ───────────────────────────────────────────────────────────────────


def test_an_unknown_profile_is_refused_with_the_ones_that_exist(
    home: Path, run: Any, httpx_mock: HTTPXMock
) -> None:
    _two_profiles()
    res = run("--profile", "reserch", "credits", "balance", "--json")
    assert res.exit_code == ExitCode.AUTH
    error = json.loads(res.stderr.strip().splitlines()[-1])["error"]
    assert error["code"] == "unknown_profile"
    assert error["details"]["profiles"] == ["default", "research"]
    assert "pictograph login --profile reserch" in error["fix"]
    assert httpx_mock.get_requests() == []


def test_an_unknown_environment_profile_is_refused_too(
    home: Path, run: Any, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_profiles()
    monkeypatch.setenv("PICTOGRAPH_PROFILE", "nope")
    res = run("credits", "balance")
    assert res.exit_code == ExitCode.AUTH
    assert "No profile 'nope' (from PICTOGRAPH_PROFILE)" in res.output
    assert httpx_mock.get_requests() == []


def test_a_bad_flag_name_is_invalid_input(home: Path, run: Any) -> None:
    assert run("--profile", "a.b", "credits", "balance").exit_code == ExitCode.INVALID


# ── login + profiles list ──────────────────────────────────────────────────────


def test_login_saves_a_named_profile_beside_the_default(home: Path, run: Any) -> None:
    write_config(api_key=DEFAULT_KEY)
    res = run("login", "--profile", "research", "--api-key", RESEARCH_KEY)
    assert res.exit_code == ExitCode.OK, res.output
    assert "as profile 'research'" in res.stdout and "pictograph --profile research" in res.stdout
    assert load_config().api_key == DEFAULT_KEY and load_config("research").api_key == RESEARCH_KEY
    res = run("login", "--profile", "lab", "--api-key", RESEARCH_KEY, "--json")
    assert json.loads(res.stdout)["profile"] == "lab"


def test_login_under_the_flag_saves_that_profile(home: Path, run: Any) -> None:
    res = run("--profile", "research", "login", "--api-key", RESEARCH_KEY)
    assert res.exit_code == ExitCode.OK, res.output
    assert load_config("research").api_key == RESEARCH_KEY and load_config().api_key is None


def test_profiles_list_masks_every_key_and_marks_the_active_one(
    home: Path, run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_profiles()
    monkeypatch.setenv("PICTOGRAPH_PROFILE", "research")
    res = run("profiles", "list", "--json")
    assert res.exit_code == ExitCode.OK, res.output
    payload = json.loads(res.stdout)
    assert (payload["active"], payload["source"]) == ("research", "PICTOGRAPH_PROFILE")
    assert [(p["profile"], p["key"], p["active"]) for p in payload["profiles"]] == [
        ("default", DEFAULT_KEY[:12] + "...", False),
        ("research", RESEARCH_KEY[:12] + "...", True),
    ]
    assert DEFAULT_KEY not in res.stdout and RESEARCH_KEY not in res.stdout


def test_profiles_list_says_when_the_environment_key_wins(
    home: Path, run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_profiles()
    monkeypatch.setenv("PICTOGRAPH_API_KEY", ENV_KEY)
    res = run("profiles", "list")
    assert res.exit_code == ExitCode.OK, res.output
    assert "PICTOGRAPH_API_KEY is set" in res.stderr
    payload = json.loads(run("profiles", "list", "--json").stdout)
    assert payload["active"] is None and payload["source"] == "PICTOGRAPH_API_KEY"
