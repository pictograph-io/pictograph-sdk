"""Shared pytest fixtures and helpers for the Pictograph SDK test suite."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# ───────────── optional companion sources ─────────────
#
# A handful of tests cross-check this repository against material that is
# developed alongside it but does not ship here: the canonical inference
# wrappers the SDK vendors, the training image that produced the published
# weights, the artifact-policy module the API and the SDK must word identically,
# the agent-tool snapshot the API serves, and the published documentation.
#
# None of that is available in a public checkout or from an installed wheel, so
# every such check is OPT-IN: point the matching environment variable below at a
# local copy and the check runs; leave it unset (the normal case) and the check
# skips with a reason that names the variable. Resolving them here, once, keeps
# individual test modules free of any assumption about where those files live.

#: Directory holding the canonical ONNX inference wrappers that
#: ``pictograph.inference._wrappers`` is vendored from.
ENV_WRAPPERS_SOURCE = "PICTOGRAPH_WRAPPERS_SOURCE"

#: The training service module whose pins produced the published weights.
ENV_TRAINING_SERVICE_SOURCE = "PICTOGRAPH_TRAINING_SERVICE_SOURCE"

#: The service-side ``model_artifacts`` module the SDK mirrors verbatim.
ENV_ARTIFACT_POLICY_SOURCE = "PICTOGRAPH_ARTIFACT_POLICY_SOURCE"

#: The committed agent-tool JSON snapshot the API serves to discovery agents.
ENV_TOOLS_SNAPSHOT = "PICTOGRAPH_TOOLS_SNAPSHOT"

#: The published ``local-inference`` documentation page (markdown).
ENV_LOCAL_INFERENCE_DOC = "PICTOGRAPH_LOCAL_INFERENCE_DOC"

#: The generated REST route snapshot every SDK request is bound against.
ENV_REST_ROUTES_SNAPSHOT = "PICTOGRAPH_REST_ROUTES_SNAPSHOT"


#: Optional local file mapping the names above to paths, so a maintainer whose
#: checkout HAS the companion sources does not have to export six variables
#: before every run. One `KEY=path` per line; `#` comments and blanks ignored.
#: Gitignored, so it never ships and no path of ours appears in this repository.
#:
#: This exists because losing it silently is the exact failure these checks
#: guard against: the wire-contract module alone contributes 406 parametrized
#: tests, and it was written after a green suite plus a clean `mypy --strict`
#: shipped a broken wheel. A skipped check must be a choice, not an accident.
_LOCAL_PATHS_FILE = Path(__file__).parent / "companion-paths.env"


@lru_cache(maxsize=1)
def _local_companion_paths() -> dict[str, str]:
    if not _LOCAL_PATHS_FILE.is_file():
        return {}
    out: dict[str, str] = {}
    for line in _LOCAL_PATHS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value:
            out[key.strip()] = value
    return out


def companion_source(env_var: str) -> Path:
    """Resolve an optional companion source.

    Order: the environment variable, then the gitignored local overrides file.
    When neither supplies a value the return is a placeholder that deliberately
    does not exist, so the ``.exists()`` gate every caller pairs this with
    resolves to "skip", and the value still formats into a skip message.
    """
    raw = os.environ.get(env_var) or _local_companion_paths().get(env_var)
    return Path(raw).expanduser() if raw else Path(f"<unset:{env_var}>")


def companion_skip_reason(env_var: str) -> str:
    """Uniform skip text for a check gated on an optional companion source."""
    return f"optional companion source unavailable - set {env_var} to a local path"
