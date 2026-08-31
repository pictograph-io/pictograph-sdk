"""Tests for the Claude / OpenAI adapters.

The adapter modules expose two surfaces each: (1) raw tool dicts (always
available, no SDK dep) and (2) decorator-wrapped tools (require the
optional ``[agents]`` extra). For (2) we don't import the actual SDKs
(they may not be installed in the test env) - instead we verify the
ImportError remediation message and the wrapping logic via a
hand-rolled fake decorator.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from pictograph.agents import (
    REGISTRY,
    Toolkit,
    for_anthropic_messages,
    for_openai_responses,
)

# ───────────── raw tool dicts (no extra deps) ─────────────


def test_for_anthropic_messages_matches_toolkit_method() -> None:
    """Module-level helper is a thin alias for the toolkit's method."""
    tk = Toolkit(MagicMock())
    assert for_anthropic_messages(tk) == tk.as_anthropic_tools()


def test_for_openai_responses_matches_toolkit_method() -> None:
    tk = Toolkit(MagicMock())
    assert for_openai_responses(tk) == tk.as_openai_tools()


# ───────────── claude SDK adapter (optional dep) ─────────────


def test_for_claude_agent_sdk_raises_clear_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without claude-agent-sdk installed, raise ImportError with install hint."""
    import builtins

    from pictograph.agents import for_claude_agent_sdk

    real_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("claude_agent_sdk"):
            raise ImportError("simulated missing claude-agent-sdk")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    tk = Toolkit(MagicMock())
    with pytest.raises(ImportError, match="claude-agent-sdk is required"):
        for_claude_agent_sdk(tk)


# ───────────── openai SDK adapter (optional dep) ─────────────


def test_for_openai_agents_raises_clear_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without openai-agents installed, raise ImportError with install hint."""
    import builtins

    from pictograph.agents import for_openai_agents

    real_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "agents":
            raise ImportError("simulated missing openai-agents")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    tk = Toolkit(MagicMock())
    with pytest.raises(ImportError, match="openai-agents is required"):
        for_openai_agents(tk)


# ───────────── adapter shape parity ─────────────


def test_anthropic_and_openai_export_same_tool_set() -> None:
    """Both adapters expose every registry entry - no silent omissions."""
    tk = Toolkit(MagicMock())
    anthropic_names = {t["name"] for t in tk.as_anthropic_tools()}
    openai_names = {t["name"] for t in tk.as_openai_tools()}
    assert anthropic_names == openai_names == {t.name for t in REGISTRY}


def test_input_schema_strips_pydantic_title() -> None:
    """Pydantic emits 'title' fields by default; we strip for clean schemas."""
    tk = Toolkit(MagicMock())
    for tool in tk.as_anthropic_tools():
        assert "title" not in tool["input_schema"]
    for tool in tk.as_openai_tools():
        assert "title" not in tool["parameters"]


# ───────────── decorator-wrap schema inlining ─────────────
#
# The @tool / function_tool wrap paths (for_claude_agent_sdk / for_openai_agents)
# must build their per-tool schema through the SAME `_to_json_schema` helper as
# the raw-dict adapters - NOT raw `model_json_schema()`. Otherwise a nested-model
# arg leaves a dangling `$ref`/`$defs` (and the always-present `title`) that the
# Claude Agent SDK can't resolve and OpenAI strict function-calling rejects.


class _NestedBox(BaseModel):
    x: int
    y: int


class _NestedArgs(BaseModel):
    label: str
    box: _NestedBox


def _fake_registry_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the adapters' lazily-imported ``get_tool`` at a synthetic
    nested-model tool so we exercise the $ref-emitting path."""
    descriptor = SimpleNamespace(
        name="nested_tool",
        description="A tool whose args nest a BaseModel (emits $defs/$ref).",
        args_schema=_NestedArgs,
    )
    monkeypatch.setattr(
        "pictograph.agents._registry.get_tool",
        lambda _name: descriptor,
    )


def test_claude_wrap_inlines_nested_refs(monkeypatch: pytest.MonkeyPatch) -> None:
    from pictograph.agents.claude import _wrap_for_claude

    _fake_registry_tool(monkeypatch)
    captured: dict[str, Any] = {}

    def _fake_tool(name: str, description: str, schema: dict[str, Any]) -> Any:
        captured["schema"] = schema
        return lambda handler: handler

    tk = Toolkit(MagicMock())
    _wrap_for_claude(tk, "nested_tool", _fake_tool)

    blob = json.dumps(captured["schema"])
    assert "$ref" not in blob
    assert "$defs" not in blob
    assert "title" not in captured["schema"]
    # the nested model's fields are inlined under properties.box.properties
    assert captured["schema"]["properties"]["box"]["properties"].keys() >= {"x", "y"}


def test_openai_wrap_inlines_nested_refs(monkeypatch: pytest.MonkeyPatch) -> None:
    from pictograph.agents.openai import _wrap_for_openai

    _fake_registry_tool(monkeypatch)
    captured: dict[str, Any] = {}

    def _fake_function_tool(invoke: Any, **kwargs: Any) -> Any:
        captured["schema"] = kwargs["params_json_schema"]
        return invoke

    tk = Toolkit(MagicMock())
    _wrap_for_openai(tk, "nested_tool", _fake_function_tool)

    blob = json.dumps(captured["schema"])
    assert "$ref" not in blob
    assert "$defs" not in blob
    assert "title" not in captured["schema"]
    assert captured["schema"]["properties"]["box"]["properties"].keys() >= {"x", "y"}
