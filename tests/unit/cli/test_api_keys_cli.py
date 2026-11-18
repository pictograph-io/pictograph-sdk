"""``pictograph api-keys list / get / create / update / revoke`` - a narrowly-scoped key for an
agent (B533 #15).

Driven through the REAL app and the REAL client over a mocked HTTP transport: what each
command sends, what it prints (``create`` shows a secret exactly once - and never with
``--save-profile``, which keeps it in a 0600 CLI profile instead), and what ``update`` /
``revoke`` refuse before any request (nothing to change, a contradiction, an unknown or
ambiguous key, an unconfirmed revocation).
"""

from __future__ import annotations

import json
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
KEYS = f"{BASE}/api/v1/api-keys/"
ME = f"{BASE}/api/v1/developer/organizations/me"
NOW = "2026-10-26T00:00:00Z"
CALLER = "pk_live_callerABCD_000000000000000"
MINTED = "pk_live_freshMINTED_1234567890abcdef"
ORG = "11111111-2222-3333-4444-555555555555"


def _key(i: int, name: str, prefix: str, role: str = "member") -> dict[str, Any]:
    return {
        "id": f"aaaaaaaa-0000-4000-8000-00000000000{i}",
        "organization_id": ORG,
        "name": name,
        "key_prefix": prefix,
        "role": role,
        "rate_limit": 5000,
        "is_active": True,
        "created_at": NOW,
    }


LISTED = [
    _key(1, "ci", "pk_live_ciXX"),
    _key(2, "labeller", "pk_live_labe", "viewer"),
    _key(3, "dup", "pk_live_dup1"),
    _key(4, "dup", "pk_live_dup2"),
    _key(5, "me", CALLER[:12], "admin"),
]


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for var in ("PICTOGRAPH_PROFILE", "PICTOGRAPH_OUTPUT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PICTOGRAPH_API_KEY", CALLER)
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    monkeypatch.setenv("PICTOGRAPH_MAX_RETRIES", "0")
    monkeypatch.setattr(_config, "CONFIG_DIR", home / ".pictograph")
    monkeypatch.setattr(_config, "CONFIG_PATH", home / ".pictograph" / "config.toml")
    monkeypatch.setattr(_config, "_flag_profile", None)
    runner = CliRunner()
    return lambda *args, stdin=None: runner.invoke(app, ["api-keys", *args], input=stdin)


def _org(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=ME,
        json={
            "organization": {
                "id": ORG,
                "name": "Pictograph",
                "slug": "pictograph",
                "subscription_tier": "core",
                "credits_remaining": 0,
                "credits_monthly_allowance": 0,
                "max_users": 5,
                "max_images": 1000,
                "max_storage_bytes": 1,
                "member_count": 1,
                "pending_invite_count": 0,
                "created_at": NOW,
                "updated_at": NOW,
            }
        },
    )


def _created(httpx_mock: HTTPXMock, role: str = "viewer") -> None:
    httpx_mock.add_response(
        method="POST",
        url=KEYS,
        status_code=201,
        json={
            "api_key": MINTED,
            "key_id": "bbbbbbbb-0000-4000-8000-000000000001",
            "key_prefix": MINTED[:12],
            "name": "agent",
            "role": role,
            "rate_limit": 1000,
            "created_at": NOW,
        },
    )


def _bodies(httpx_mock: HTTPXMock, method: str) -> list[dict[str, Any]]:
    return [json.loads(r.content) for r in httpx_mock.get_requests() if r.method == method]


# ── list ───────────────────────────────────────────────────────────────────────


def test_list_shows_metadata_never_a_secret(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=KEYS, json={"api_keys": LISTED})
    res = run("list", "--json")
    assert res.exit_code == ExitCode.OK, res.output
    rows = json.loads(res.stdout)
    assert [r["name"] for r in rows] == ["ci", "labeller", "dup", "dup", "me"]
    assert all("api_key" not in r for r in rows)


def test_get_shows_one_keys_metadata_by_name(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=KEYS, json={"api_keys": LISTED})
    res = run("get", "labeller")
    assert res.exit_code == ExitCode.OK, res.output
    shown = json.loads(res.stdout)
    assert (shown["id"], shown["role"], shown["key_prefix"]) == (
        "aaaaaaaa-0000-4000-8000-000000000002",
        "viewer",
        "pk_live_labe",
    )
    assert "api_key" not in shown


# ── create ─────────────────────────────────────────────────────────────────────


def test_create_sends_the_role_cap_and_expiry_and_prints_the_secret_once(
    run: Any, httpx_mock: HTTPXMock
) -> None:
    _org(httpx_mock)
    _created(httpx_mock)
    res = run(
        "create", "agent", "--role", "viewer", "--rate-limit", "1000", "--expires", "2026-12-31"
    )
    assert res.exit_code == ExitCode.OK, res.output
    (body,) = _bodies(httpx_mock, "POST")
    assert body == {
        "organization_id": ORG,
        "name": "agent",
        "role": "viewer",
        "rate_limit": 1000,
        "expires_at": "2026-12-31",
    }
    assert json.loads(res.stdout)["api_key"] == MINTED
    assert res.stdout.count(MINTED) == 1 and MINTED not in res.stderr
    assert "printed once" in res.stderr


def test_save_profile_keeps_the_key_and_never_prints_it(run: Any, httpx_mock: HTTPXMock) -> None:
    write_config(api_key=CALLER)  # an existing default profile stays as it was
    _org(httpx_mock)
    _created(httpx_mock)
    res = run("create", "agent", "--role", "viewer", "--save-profile", "agent")
    assert res.exit_code == ExitCode.OK, res.output
    assert MINTED not in res.stdout and MINTED not in res.stderr
    assert (
        "saved it as profile 'agent'" in res.stdout and "pictograph --profile agent" in res.stdout
    )
    assert load_config("agent") == _config.CliConfig(MINTED, BASE)  # the same API as its maker
    assert load_config().api_key == CALLER


@pytest.mark.parametrize(
    ("args", "why"),
    [
        (("create", "agent", "--role", "superuser"), "viewer, member, admin or owner"),
        (("create", "agent", "--save-profile", "a.b"), "cannot be a profile name"),
        (("create", "agent", "--rate-limit", "0"), "--rate-limit"),
    ],
)
def test_create_refusals_before_any_request(
    run: Any, httpx_mock: HTTPXMock, args: tuple[str, ...], why: str
) -> None:
    res = run(*args)
    assert res.exit_code == ExitCode.INVALID
    assert why in " ".join(res.output.replace("│", " ").split())
    assert httpx_mock.get_requests() == []


def test_a_forbidden_create_says_so(run: Any, httpx_mock: HTTPXMock) -> None:
    _org(httpx_mock)
    httpx_mock.add_response(
        method="POST",
        url=KEYS,
        status_code=403,
        json={
            "detail": "Insufficient permissions: an API key cannot mint a key above its own role"
        },
    )
    res = run("create", "agent", "--role", "owner")
    assert res.exit_code == ExitCode.FORBIDDEN
    assert "above its own role" in res.output


# ── update ─────────────────────────────────────────────────────────────────────


def _patched(httpx_mock: HTTPXMock, key_id: str, **changes: Any) -> None:
    row = next(k for k in LISTED if k["id"] == key_id)
    httpx_mock.add_response(
        method="PATCH", url=f"{KEYS}{key_id}", json={"api_key": {**row, **changes}}
    )


@pytest.mark.parametrize(
    ("args", "body"),
    [
        (
            ("--name", "ci-2", "--rate-limit", "100", "--inactive"),
            {"name": "ci-2", "rate_limit": 100, "is_active": False},
        ),
        (("--clear-rate-limit",), {"rate_limit": None}),  # null clears the cap
        (("--active",), {"is_active": True}),
    ],
)
def test_update_sends_only_what_changes(
    run: Any, httpx_mock: HTTPXMock, args: tuple[str, ...], body: dict[str, Any]
) -> None:
    httpx_mock.add_response(url=KEYS, json={"api_keys": LISTED})
    _patched(httpx_mock, "aaaaaaaa-0000-4000-8000-000000000001", name="ci-2")
    res = run("update", "ci", *args)
    assert res.exit_code == ExitCode.OK, res.output
    assert _bodies(httpx_mock, "PATCH") == [body]
    assert json.loads(res.stdout)["name"] == "ci-2"


@pytest.mark.parametrize(
    ("args", "why"),
    [
        (("update", "ci"), "nothing to update"),
        (("update", "ci", "--rate-limit", "10", "--clear-rate-limit"), "not both"),
        (("update", "ci", "--rate-limit", "0"), "--rate-limit"),
    ],
)
def test_update_refusals_before_any_request(
    run: Any, httpx_mock: HTTPXMock, args: tuple[str, ...], why: str
) -> None:
    res = run(*args)
    assert res.exit_code == ExitCode.INVALID
    assert why in " ".join(res.output.replace("│", " ").split())
    assert httpx_mock.get_requests() == []


def test_turning_off_the_key_the_command_runs_with_says_so(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=KEYS, json={"api_keys": LISTED})
    _patched(httpx_mock, "aaaaaaaa-0000-4000-8000-000000000005", is_active=False)
    res = run("update", "me", "--inactive")
    assert res.exit_code == ExitCode.OK, res.output
    assert "the key this command runs with" in res.stderr
    assert "turns it back on" in " ".join(res.stderr.split())


# ── revoke ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ref",
    [
        "labeller",
        "pk_live_labe",
        "pk_live_labe_and_the_rest",
        "aaaaaaaa-0000-4000-8000-000000000002",
    ],
)
def test_revoke_finds_the_key_by_name_prefix_whole_key_or_id(
    run: Any, httpx_mock: HTTPXMock, ref: str
) -> None:
    httpx_mock.add_response(url=KEYS, json={"api_keys": LISTED})
    httpx_mock.add_response(
        method="DELETE", url=f"{KEYS}aaaaaaaa-0000-4000-8000-000000000002", json={"success": True}
    )
    res = run("revoke", ref, "--yes")
    assert res.exit_code == ExitCode.OK, res.output
    assert json.loads(res.stdout) == {
        "revoked": "aaaaaaaa-0000-4000-8000-000000000002",
        "name": "labeller",
        "key_prefix": "pk_live_labe",
        "own_key": False,
    }


@pytest.mark.parametrize(
    ("ref", "code", "exit_code"),
    [("nope", "not_found", ExitCode.NOT_FOUND), ("dup", "ambiguous_key", ExitCode.INVALID)],
)
def test_revoke_refuses_an_unknown_or_ambiguous_key_before_any_delete(
    run: Any, httpx_mock: HTTPXMock, ref: str, code: str, exit_code: int
) -> None:
    httpx_mock.add_response(url=KEYS, json={"api_keys": LISTED})
    res = run("revoke", ref, "--yes", "--json")
    assert res.exit_code == exit_code
    assert json.loads(res.stderr.strip().splitlines()[-1])["error"]["code"] == code
    assert [r.method for r in httpx_mock.get_requests()] == ["GET"]


def test_revoke_asks_first(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=KEYS, json={"api_keys": LISTED})
    res = run("revoke", "ci", stdin="n\n")
    assert res.exit_code == ExitCode.FAILED
    assert "cannot be undone" in res.output
    assert [r.method for r in httpx_mock.get_requests()] == ["GET"]


def test_revoking_the_key_the_command_runs_with_says_so(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=KEYS, json={"api_keys": LISTED})
    httpx_mock.add_response(
        method="DELETE", url=f"{KEYS}aaaaaaaa-0000-4000-8000-000000000005", json={"success": True}
    )
    res = run("revoke", "me", "--yes")
    assert res.exit_code == ExitCode.OK, res.output
    assert json.loads(res.stdout)["own_key"] is True
    assert "the key this command runs with" in res.stderr
