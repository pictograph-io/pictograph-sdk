"""Tests for the top-level Typer app + auth resolution.

The CLI is tested via ``typer.testing.CliRunner`` - a synchronous,
in-process invocation that captures stdout/stderr without spawning a
subprocess. Each command's underlying SDK client is mocked at the
import boundary.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pictograph.cli._app import app
from pictograph.cli._config import (
    CliConfig,
    load_config,
    resolve_api_key,
    resolve_base_url,
    write_config,
)
from pictograph.models.auto_annotate import PromptResult
from pictograph.models.credit import CreditBalance
from pictograph.models.dataset import Dataset
from pictograph.models.deployment import Deployment
from pictograph.models.workflow import Workflow, WorkflowRun, WorkflowRunCreated


def _patch_get_client_everywhere(client: MagicMock) -> Any:
    """Patch ``get_client`` in every command-module namespace at once.

    Because each command does ``from pictograph.cli._client import get_client``,
    patching the source module isn't enough - each importer's local binding has
    to point at the mock too.
    """
    targets = [
        "pictograph.cli.commands.datasets.get_client",
        "pictograph.cli.commands.images.get_client",
        "pictograph.cli.commands.annotations.get_client",
        "pictograph.cli.commands.train.get_client",
        "pictograph.cli.commands.models.get_client",
        "pictograph.cli.commands.auto_annotate.get_client",
        "pictograph.cli.commands.deployments.get_client",
        "pictograph.cli.commands.workflows.get_client",
        "pictograph.cli.commands.credits.get_client",
    ]
    return (
        patch.multiple(
            "pictograph.cli._client",
            **{},  # placeholder; we use stack of context managers below
        )
        if False
        else _StackPatcher(targets, client)
    )


class _StackPatcher:
    def __init__(self, targets: list[str], value: Any) -> None:
        self._patches = [patch(t, return_value=value) for t in targets]

    def __enter__(self) -> Any:
        for p in self._patches:
            p.start()
        return None

    def __exit__(self, *exc: Any) -> None:
        for p in reversed(self._patches):
            p.stop()


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ~/.pictograph/* to tmp_path so tests don't touch real config."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("PICTOGRAPH_API_KEY", raising=False)
    # The module-level CONFIG_PATH was computed at import - patch it too.
    monkeypatch.setattr(
        "pictograph.cli._config.CONFIG_DIR",
        fake_home / ".pictograph",
    )
    monkeypatch.setattr(
        "pictograph.cli._config.CONFIG_PATH",
        fake_home / ".pictograph" / "config.toml",
    )
    return fake_home


def _dataset(name: str = "ds") -> Dataset:
    return Dataset(
        id="proj-uuid",
        name=name,
        description=None,
        image_count=4,
        completed_image_count=4,
        total_size=1000,
        archived_images=0,
        classes=[],
        images=None,
        created_at=datetime.now(timezone.utc),
    )


# ───────────── version + help ─────────────


def test_version_flag_prints_version(runner: CliRunner) -> None:
    res = runner.invoke(app, ["--version"])
    assert res.exit_code == 0
    assert "pictograph " in res.stdout


def test_help_lists_subcommand_groups(runner: CliRunner) -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for group in (
        "datasets",
        "images",
        "annotations",
        "train",
        "models",
        "auto-annotate",
        "deployments",
        "workflows",
        "organizations",
        "connectors",
        "exports",
        "search",
        "video",
        "webhooks",
        "credits",
        "agents",
        "init",
        "login",
    ):
        assert group in res.stdout, f"missing {group!r} in --help"


# ───────────── auth resolution ─────────────


def test_no_api_key_exits_with_clear_error(runner: CliRunner, isolated_config: Path) -> None:
    res = runner.invoke(app, ["datasets", "list"])
    assert res.exit_code == 2
    # Error goes to stderr (mix_stderr default keeps it folded into stdout).
    assert "API key" in res.stdout or "API key" in (res.stderr or "")


def test_api_key_flag_takes_precedence(runner: CliRunner, isolated_config: Path) -> None:
    """--api-key beats env beats config."""
    os.environ["PICTOGRAPH_API_KEY"] = "pk_live_env"
    try:
        write_config(api_key="pk_live_config")
        assert resolve_api_key("pk_live_flag") == "pk_live_flag"
        assert resolve_api_key() == "pk_live_env"
    finally:
        os.environ.pop("PICTOGRAPH_API_KEY", None)


def test_env_beats_config(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_config")
    os.environ["PICTOGRAPH_API_KEY"] = "pk_live_env"
    try:
        assert resolve_api_key() == "pk_live_env"
    finally:
        os.environ.pop("PICTOGRAPH_API_KEY", None)


def test_resolve_base_url_precedence(runner: CliRunner, isolated_config: Path) -> None:
    """--base-url flag beats PICTOGRAPH_BASE_URL env beats config.toml."""
    write_config(api_key="pk_live_x", base_url="https://config.example.com")
    # config only
    assert resolve_base_url() == "https://config.example.com"
    os.environ["PICTOGRAPH_BASE_URL"] = "https://env.example.com"
    try:
        assert resolve_base_url() == "https://env.example.com"  # env beats config
        assert resolve_base_url("https://flag.example.com") == "https://flag.example.com"
    finally:
        os.environ.pop("PICTOGRAPH_BASE_URL", None)


def test_resolve_base_url_none_when_unset(isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")  # no base_url
    os.environ.pop("PICTOGRAPH_BASE_URL", None)
    assert resolve_base_url() is None


def test_get_client_uses_config_base_url(isolated_config: Path) -> None:
    """Regression: a base_url stored by `login --base-url` must reach the Client.
    Pre-fix get_client ignored config base_url and always hit the prod default."""
    from pictograph.cli import _client

    write_config(api_key="pk_live_x", base_url="https://staging.example.com")
    os.environ.pop("PICTOGRAPH_BASE_URL", None)
    captured: dict[str, Any] = {}
    with patch("pictograph.Client", lambda **kw: captured.update(kw) or MagicMock()):
        _client.get_client(None)
    assert captured["base_url"] == "https://staging.example.com"  # was None pre-fix


def test_load_config_returns_empty_when_missing(isolated_config: Path) -> None:
    cfg = load_config()
    assert cfg == CliConfig()


def test_write_then_load_round_trip(isolated_config: Path) -> None:
    path = write_config(api_key="pk_live_xyz", base_url="https://example.com")
    assert path.is_file()
    cfg = load_config()
    assert cfg.api_key == "pk_live_xyz"
    assert cfg.base_url == "https://example.com"


@pytest.mark.skipif(not hasattr(os, "fchmod"), reason="POSIX file-permission semantics only")
def test_write_config_sets_owner_only_permissions(isolated_config: Path) -> None:
    """The secret-bearing config file must be 0600 (owner read/write only)."""
    path = write_config(api_key="pk_live_secret")
    assert (path.stat().st_mode & 0o777) == 0o600


@pytest.mark.skipif(not hasattr(os, "fchmod"), reason="POSIX file-permission semantics only")
def test_write_config_repairs_loose_permissions_on_existing_file(
    isolated_config: Path,
) -> None:
    """A pre-existing world-readable config is tightened to 0600 on rewrite."""
    path = write_config(api_key="pk_live_first")
    path.chmod(0o644)  # simulate a loosely-permissioned legacy/copied file
    assert (path.stat().st_mode & 0o777) == 0o644
    write_config(api_key="pk_live_second")
    assert (path.stat().st_mode & 0o777) == 0o600
    assert load_config().api_key == "pk_live_second"


# ───────────── login command ─────────────


def test_login_writes_config(runner: CliRunner, isolated_config: Path) -> None:
    res = runner.invoke(app, ["login", "--api-key", "pk_live_test"])
    assert res.exit_code == 0
    assert "Saved API key" in res.stdout
    assert load_config().api_key == "pk_live_test"


# ───────────── init command ─────────────


def test_init_writes_agents_md(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    res = runner.invoke(app, ["init", "-o", str(target)])
    assert res.exit_code == 0
    assert target.is_file()
    body = target.read_text(encoding="utf-8")
    assert "Pictograph" in body
    assert "PICTOGRAPH_API_KEY" in body


def test_init_refuses_to_overwrite_without_force(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    target.write_text("existing", encoding="utf-8")
    res = runner.invoke(app, ["init", "-o", str(target)])
    assert res.exit_code == 1
    assert target.read_text() == "existing"


def test_init_force_overwrites(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    target.write_text("zzz_sentinel_zzz", encoding="utf-8")
    res = runner.invoke(app, ["init", "-o", str(target), "--force"])
    assert res.exit_code == 0
    assert "zzz_sentinel_zzz" not in target.read_text()
    assert "Pictograph" in target.read_text()


# ───────────── command dispatch (one happy-path per group) ─────────────


def test_datasets_list_renders_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.datasets.list.return_value = [_dataset("a"), _dataset("b")]
    with _patch_get_client_everywhere(client):
        res = runner.invoke(app, ["datasets", "list", "--json"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert len(payload) == 2
    client.datasets.list.assert_called_once_with(limit=100)


def test_datasets_get(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.datasets.get.return_value = _dataset("ds")
    with _patch_get_client_everywhere(client):
        res = runner.invoke(app, ["datasets", "get", "ds"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["name"] == "ds"


def test_credits_balance(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.credits.balance.return_value = CreditBalance(
        included_remaining_micro_usd=42_500_000,  # $42.50
        included_allowance_micro_usd=1_000_000_000,  # $1,000.00
        budget_micro_usd=5_000_000,  # $5.00
        period_overage_micro_usd=1_250_000,  # $1.25
        credits_reset_at=None,
        recent_history=[],
    )
    with _patch_get_client_everywhere(client):
        res = runner.invoke(app, ["credits", "balance"])
    assert res.exit_code == 0, res.stdout
    assert "Remaining:" in res.stdout
    assert "$42.50" in res.stdout
    assert "$1,000.00" in res.stdout  # monthly allowance, USD-formatted
    assert "Overage budget:" in res.stdout  # budget_micro_usd > 0 → shown


def test_agents_list_tools(runner: CliRunner) -> None:
    """Rich truncates long names in narrow terminals - assert on stable prefix."""
    res = runner.invoke(app, ["agents", "list-tools"])
    assert res.exit_code == 0, res.stdout
    # Tool names get truncated by Rich at ~22 chars; assert on a prefix present
    # in the rendered table regardless of column width.
    assert "Agent tools" in res.stdout
    assert "upload_dataset" in res.stdout
    assert "credit_balance" in res.stdout


def test_agents_export_tools_to_stdout(runner: CliRunner) -> None:
    res = runner.invoke(app, ["agents", "export-tools"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert isinstance(payload, list)
    assert len(payload) >= 25
    names = {t["name"] for t in payload}
    assert "upload_dataset_from_directory" in names


def test_agents_install_skill_to_custom_dir(runner: CliRunner, tmp_path: Path) -> None:
    res = runner.invoke(
        app,
        ["agents", "install-skill", "--target", "claude-code", "--output", str(tmp_path)],
    )
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["installed"] is True
    assert (tmp_path / "pictograph-cv" / "SKILL.md").is_file()


def test_agents_install_skill_invalid_target(runner: CliRunner) -> None:
    res = runner.invoke(
        app,
        ["agents", "install-skill", "--target", "nonsense"],
    )
    assert res.exit_code == 2


# ───────────── workflows + inference command groups ─────────────


def _workflow(name: str = "wf") -> Workflow:
    return Workflow(
        id="wf-1", organization_id="org-1", name=name, graph={"nodes": []}, status="draft"
    )


def _deployment() -> Deployment:
    return Deployment(
        id="dep-1",
        organization_id="org-1",
        model_id="m-1",
        name="d",
        status="active",
        compute_type="gpu",
        gpu_type="t4",
        min_containers=0,
        max_containers=1,
        scaledown_window=60,
        endpoint_url="https://endpoint.test/predict",
    )


def test_workflows_list_renders_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.workflows.list.return_value = [_workflow("a"), _workflow("b")]
    with _patch_get_client_everywhere(client):
        res = runner.invoke(app, ["workflows", "list", "--json"])
    assert res.exit_code == 0, res.stdout
    assert len(json.loads(res.stdout)) == 2
    client.workflows.list.assert_called_once_with()


def test_workflows_run_waits_for_completion(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.workflows.run.return_value = WorkflowRunCreated(run_id="run-1", deposit_micro_usd=1000)
    client.workflows.wait_for_run.return_value = WorkflowRun(
        id="run-1",
        organization_id="org-1",
        workflow_id="wf-1",
        status="completed",
    )
    with _patch_get_client_everywhere(client):
        res = runner.invoke(app, ["workflows", "run", "wf-1"])
    assert res.exit_code == 0, res.stdout
    assert json.loads(res.stdout)["status"] == "completed"
    client.workflows.run.assert_called_once_with("wf-1")
    client.workflows.wait_for_run.assert_called_once()


def test_workflows_run_no_wait_skips_poll(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.workflows.run.return_value = WorkflowRunCreated(run_id="run-1", deposit_micro_usd=2000)
    with _patch_get_client_everywhere(client):
        res = runner.invoke(app, ["workflows", "run", "wf-1", "--no-wait"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["run_id"] == "run-1" and payload["deposit_micro_usd"] == 2000
    client.workflows.wait_for_run.assert_not_called()


def test_deployments_list_renders_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.deployments.list.return_value = [_deployment()]
    with _patch_get_client_everywhere(client):
        res = runner.invoke(app, ["deployments", "list", "--json"])
    assert res.exit_code == 0, res.stdout
    assert json.loads(res.stdout)[0]["id"] == "dep-1"


def test_deployments_predict_calls_endpoint(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.deployments.get.return_value = _deployment()
    conn = MagicMock()
    # The CLI prints the endpoint's OWN json (infer_raw), not the typed result's
    # dump - someone debugging a deployment wants the wire body.
    conn.infer_raw.return_value = {"predictions": [{"name": "car", "confidence": 0.9}]}
    client.deployments.connect.return_value = conn
    with _patch_get_client_everywhere(client):
        res = runner.invoke(
            app,
            ["deployments", "predict", "dep-1", "https://img.test/x.jpg", "--token", "pk_deploy_t"],
        )
    assert res.exit_code == 0, res.stdout
    assert json.loads(res.stdout)["predictions"][0]["name"] == "car"
    client.deployments.connect.assert_called_once()


def test_auto_annotate_point(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.auto_annotate.point.return_value = PromptResult(status="success", annotations=[])
    with _patch_get_client_everywhere(client):
        res = runner.invoke(
            app, ["auto-annotate", "point", "ds", "img.jpg", "--x", "10", "--y", "20"]
        )
    assert res.exit_code == 0, res.stdout
    assert json.loads(res.stdout)["status"] == "success"
    client.auto_annotate.point.assert_called_once()


# ───────────── typing import (CliRunner dependency for type checker) ─────────────


# ───────────── main() error rendering (entry point contract) ─────────────
# The installed `pictograph` binary's [project.scripts] entry MUST target
# `main`, not `app`: main() catches an expected PictographError and renders a
# one-line `error: …` via print_error. Targeting `app` directly bypasses that
# handler, so Typer's excepthook prints a Rich traceback for every 401/402/404/
# 409/429 (a real shipped-CLI UX regression). These pin both halves.


def test_console_script_targets_main_not_app() -> None:
    """pyproject's entry point routes through main() (the error-wrapping shim),
    not the bare Typer app object."""
    pyproject = (Path(__file__).resolve().parents[3] / "pyproject.toml").read_text()
    assert 'pictograph = "pictograph.cli._app:main"' in pyproject
    assert 'pictograph = "pictograph.cli._app:app"' not in pyproject


def test_main_renders_pictograph_error_as_one_line(monkeypatch, capsys) -> None:
    """A PictographError surfacing from the app is caught and rendered as a
    clean `error: …` line with NO traceback; exit code is 1."""
    from pictograph.cli import _app
    from pictograph.exceptions import AuthError

    def _raising_app() -> None:
        raise AuthError("invalid api key (status=401)")

    monkeypatch.setattr(_app, "app", _raising_app)
    rc = _app.main()
    assert rc == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "error:" in combined
    assert "invalid api key (status=401)" in combined
    assert "Traceback" not in combined


def test_main_returns_zero_on_success(monkeypatch) -> None:
    """A normally-returning app (no exception) yields exit code 0."""
    from pictograph.cli import _app

    monkeypatch.setattr(_app, "app", lambda: None)
    assert _app.main() == 0


def test_main_propagates_typer_system_exit(monkeypatch) -> None:
    """Typer/Click signal exit codes via SystemExit (e.g. usage errors → 2,
    successful standalone exit → 0). main() must NOT swallow these - it only
    intercepts PictographError."""
    from pictograph.cli import _app

    def _exiting_app() -> None:
        raise SystemExit(2)

    monkeypatch.setattr(_app, "app", _exiting_app)
    with pytest.raises(SystemExit) as exc_info:
        _app.main()
    assert exc_info.value.code == 2


def test_install_skill_rejects_traversing_skill_name(tmp_path: Path) -> None:
    """`--skill ..` would rmtree the user's ~/.claude directory.

    dest = (~/.claude/skills) / ".." == ~/.claude, and install-skill removes an
    existing dest before copying. A slug is the only legal shape here.
    """
    from typer.testing import CliRunner

    from pictograph.cli._app import app

    runner = CliRunner()
    for hostile in ("..", "../evil", "/etc", "a/b"):
        result = runner.invoke(
            app, ["agents", "install-skill", "--skill", hostile, "--output", str(tmp_path)]
        )
        assert result.exit_code == 2, f"{hostile!r} was not rejected"
        assert "Invalid skill name" in result.output
    # The guard must not have deleted anything on the way out.
    assert tmp_path.is_dir()
