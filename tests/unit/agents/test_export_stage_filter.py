"""'all' is not an image stage, so neither the export tool nor the skill script may send it.

Both offered ``all`` and ``in_progress`` as export filters and passed the value straight to
``exports.create(status_filter=...)``. The API matches that value against an image's stage
(``new`` / ``annotate`` / ``review`` / ``complete``), so each built an EMPTY export with no
error. The fix lives in the SDK only: ``all`` is sent as no filter, ``in_progress`` (never a
stage) is gone, and each default is unchanged. The platform's own behavior is untouched.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from pictograph.agents import Toolkit
from pictograph.skills import skill_path

_STAGES = ("new", "annotate", "review", "complete")


def _export_call(status_filter: str | None) -> dict[str, object]:
    client = MagicMock()
    client.exports.create.return_value = {"id": "exp-1"}
    args: dict[str, object] = {"dataset_name": "shelf", "name": "v1"}
    if status_filter is not None:
        args["status_filter"] = status_filter
    Toolkit(client).dispatch("create_export", args)
    return dict(client.exports.create.call_args.kwargs)


def test_the_tool_sends_no_filter_for_all() -> None:
    assert _export_call("all")["status_filter"] is None


@pytest.mark.parametrize("stage", _STAGES)
def test_the_tool_passes_a_real_stage_through(stage: str) -> None:
    assert _export_call(stage)["status_filter"] == stage


def test_the_tool_keeps_its_default() -> None:
    # Unchanged behavior: a call that names no filter exports finished images, as before.
    assert _export_call(None)["status_filter"] == "complete"


def test_the_tool_rejects_a_value_that_is_not_a_stage() -> None:
    client = MagicMock()
    with pytest.raises(ValidationError):
        Toolkit(client).dispatch(
            "create_export", {"dataset_name": "shelf", "name": "v1", "status_filter": "in_progress"}
        )
    client.exports.create.assert_not_called()


def _load_export_script() -> ModuleType:
    path = skill_path("pictograph-cv") / "scripts" / "export.py"
    spec = importlib.util.spec_from_file_location("pictograph_cv_export_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_offers_only_real_stages_plus_all() -> None:
    script = _load_export_script()
    assert set(script.STAGES) == {"all", *_STAGES}


def test_the_script_sends_no_filter_for_all_and_passes_stages_through() -> None:
    script = _load_export_script()
    assert script.stage_filter("all") is None
    for stage in _STAGES:
        assert script.stage_filter(stage) == stage


@pytest.mark.parametrize(
    ("argv_filter", "sent"),
    [
        ([], "complete"),
        (["--status-filter", "all"], None),
        (["--status-filter", "review"], "review"),
    ],
)
def test_the_script_sends_what_the_api_takes(
    argv_filter: list[str],
    sent: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # main() end to end against a mocked client: the default is unchanged ("complete"),
    # "all" reaches the API as no filter, and a real stage passes through.
    script = _load_export_script()
    client = MagicMock()
    client.exports.create.return_value = MagicMock(
        id="exp-1", status="completed", image_count=3, annotation_count=9
    )
    client.exports.download.side_effect = lambda *_a, **k: Path(k["output_path"]).write_bytes(
        b"zip"
    )
    monkeypatch.setattr(script, "Client", lambda: client)
    out = tmp_path / "v1.zip"
    monkeypatch.setattr(
        sys,
        "argv",
        ["export.py", "--dataset", "shelf", "--name", "v1", "--output", str(out), *argv_filter],
    )
    assert script.main() == 0
    assert client.exports.create.call_args.kwargs["status_filter"] == sent
