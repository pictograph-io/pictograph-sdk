"""The Claude Code plugin in ``claude-plugin/`` (installed with
``/plugin install pictograph --marketplace pictograph-io/pictograph-sdk``).

It carries a copy of the skill in ``src/pictograph/skills/pictograph-cv`` (one source; the
monorepo's ``scripts/sync_claude_plugin.py`` writes the copy), a manifest whose version is the
SDK's, a sensitive ``api_key`` option, and a SessionStart hook that hands that key to the
session's commands as ``PICTOGRAPH_API_KEY``. These tests fail when the copy drifts and pin the
hook's behaviour, which the plugin directory's credential policy depends on.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
from PIL import Image

import pictograph

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "claude-plugin"
SKILL_SRC = ROOT / "src" / "pictograph" / "skills" / "pictograph-cv"
SKILL_COPY = PLUGIN / "skills" / "pictograph-cv"
HOOK = PLUGIN / "hooks" / "session-start.sh"


def _files(base: Path) -> dict[str, bytes]:
    return {
        p.relative_to(base).as_posix(): p.read_bytes()
        for p in sorted(base.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }


def _manifest() -> dict[str, object]:
    return json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))


def test_the_plugins_skill_is_the_packaged_skill() -> None:
    want, have = _files(SKILL_SRC), _files(SKILL_COPY)
    fix = "copy src/pictograph/skills/pictograph-cv over claude-plugin/skills/pictograph-cv"
    assert sorted(want) == sorted(have), fix
    stale = [rel for rel in want if want[rel] != have[rel]]
    assert not stale, f"{stale} differ: {fix}"


def test_the_manifest_carries_the_sdk_version_and_a_sensitive_key_option() -> None:
    manifest = _manifest()
    assert manifest["name"] == "pictograph"
    assert manifest["version"] == pictograph.__version__
    assert manifest["license"] == "MIT"
    option = manifest["userConfig"]["api_key"]  # type: ignore[index]
    assert option["sensitive"] is True
    assert option["type"] == "string"


def test_the_icon_is_a_square_png_the_directory_accepts() -> None:
    manifest = _manifest()
    icon = PLUGIN / str(manifest["icon"])
    with Image.open(icon) as im:
        assert im.format == "PNG"
        width, height = im.size
    assert width == height and 512 <= width <= 2048
    assert icon.stat().st_size < 2 * 1024 * 1024


def test_the_marketplace_lists_this_plugin() -> None:
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    (entry,) = market["plugins"]
    assert entry["name"] == _manifest()["name"]
    assert entry["source"] == "./claude-plugin"


def test_the_skill_preapproves_only_the_cli_and_reading() -> None:
    front = (SKILL_SRC / "SKILL.md").read_text(encoding="utf-8").split("---")[1]
    (line,) = [ln for ln in front.splitlines() if ln.startswith("allowed-tools:")]
    assert line == "allowed-tools: Bash(pictograph *), Read"


# The plugin directory holds a plugin whose text hands a credential-named environment variable
# to a command beside a remote URL ("Uses a credential from the user's machine"): the skill takes
# a credential from the user as a value (or from stdin), never from their environment.
_ENV_CREDENTIAL = re.compile(r"\$\{?[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)\b")
_ENV_CREDENTIAL_READ = re.compile(
    r"environ(?:\.get)?\s*[\(\[]\s*[\"'][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)[\"']"
)


def test_the_gate_catches_what_the_directory_flagged() -> None:
    assert _ENV_CREDENTIAL.search("import_connector.py --provider v7 --api-key $V7_KEY --list")
    assert _ENV_CREDENTIAL.search('curl -H "Authorization: Bearer ${DEPLOY_TOKEN}" https://x')
    assert _ENV_CREDENTIAL_READ.search('default=os.environ.get("PICTOGRAPH_SOURCE_KEY"),')
    assert not _ENV_CREDENTIAL.search("`$PICTOGRAPH_CACHE_DIR`")


def test_the_skill_takes_no_credential_from_the_environment() -> None:
    hits = [
        f"{rel}:{n}: {line.strip()}"
        for rel, data in _files(SKILL_SRC).items()
        for n, line in enumerate(data.decode("utf-8").splitlines(), 1)
        if _ENV_CREDENTIAL.search(line) or _ENV_CREDENTIAL_READ.search(line)
    ]
    assert not hits, hits


def test_a_deployment_token_is_shown_going_to_pictographs_own_host() -> None:
    text = (SKILL_SRC / "references" / "inference.md").read_text(encoding="utf-8")
    assert "<your-deployment-endpoint>" not in text
    assert "https://workflows.pictograph.io/us-east/<deployment-slug>/predict" in text


def test_the_hook_runs_the_registered_script_by_its_literal_path() -> None:
    # Exec form (args present, no shell): the command is the script's literal path under
    # CLAUDE_PLUGIN_ROOT, the shape the plugin directory's reviewer can follow.
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    (group,) = hooks["hooks"]["SessionStart"]
    (hook,) = group["hooks"]
    assert hook == {
        "type": "command",
        "command": "${CLAUDE_PLUGIN_ROOT}/hooks/session-start.sh",
        "args": [],
    }
    assert os.access(HOOK, os.X_OK), "session-start.sh must be executable"
    assert HOOK.read_text(encoding="utf-8").startswith("#!/bin/sh\n")


def test_the_hook_reads_no_credential_but_the_plugin_option() -> None:
    code = "\n".join(
        ln
        for ln in HOOK.read_text(encoding="utf-8").splitlines()
        if not ln.lstrip().startswith("#")
    )
    assert "CLAUDE_PLUGIN_OPTION_API_KEY" in code
    assert "${PICTOGRAPH_API_KEY" not in code and "$PICTOGRAPH_API_KEY" not in code


# A key shaped like a real one (pk_live_ + token_urlsafe(32)), built at runtime.
KEY = "pk_live_" + "Ab3_-9" * 7 + "z"


def _run_hook(tmp_path: Path, env: dict[str, str]) -> tuple[subprocess.CompletedProcess[str], str]:
    env_file = tmp_path / "env.sh"
    env_file.write_text("", encoding="utf-8")
    base = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PICTOGRAPH_API_KEY", "CLAUDE_PLUGIN_OPTION_API_KEY")
    }
    full = {**base, "CLAUDE_ENV_FILE": str(env_file), **env}
    proc = subprocess.run(
        ["/bin/sh", str(HOOK)], env=full, capture_output=True, text=True, check=False
    )
    return proc, env_file.read_text(encoding="utf-8")


@pytest.mark.skipif(not Path("/bin/sh").exists(), reason="no /bin/sh")
class TestSessionStartHook:
    def test_it_sets_the_entered_key_and_prints_nothing(self, tmp_path: Path) -> None:
        proc, written = _run_hook(tmp_path, {"CLAUDE_PLUGIN_OPTION_API_KEY": KEY})
        assert proc.returncode == 0
        assert proc.stdout == "" and proc.stderr == ""
        assert written == f"export PICTOGRAPH_API_KEY={KEY}\n"

    def test_the_entered_key_is_used_beside_another_in_the_environment(
        self, tmp_path: Path
    ) -> None:
        # The key entered for the plugin is the one its commands use; the hook never reads a
        # credential from the user's environment.
        proc, written = _run_hook(
            tmp_path, {"CLAUDE_PLUGIN_OPTION_API_KEY": KEY, "PICTOGRAPH_API_KEY": "another-key"}
        )
        assert proc.returncode == 0
        assert written == f"export PICTOGRAPH_API_KEY={KEY}\n"

    @pytest.mark.parametrize("value", ["x$(touch pwned)", "a;b", "a b", "`id`", 'a"b', "a\nb"])
    def test_a_value_a_shell_could_read_as_code_is_never_written(
        self, tmp_path: Path, value: str
    ) -> None:
        proc, written = _run_hook(tmp_path, {"CLAUDE_PLUGIN_OPTION_API_KEY": value})
        assert proc.returncode == 0
        assert written == ""
        assert not (tmp_path / "pwned").exists()

    def test_no_key_entered_writes_nothing(self, tmp_path: Path) -> None:
        proc, written = _run_hook(tmp_path, {})
        assert proc.returncode == 0
        assert written == ""
