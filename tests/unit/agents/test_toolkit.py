"""Tests for ``pictograph.agents._toolkit.Toolkit``.

The toolkit is the dispatcher seam: it validates agent input via
Pydantic, calls the right handler, and serialises the result. Adapter
output shape (Anthropic / OpenAI / JSON Schema) is also asserted here
since each adapter is a thin transform of the same registry.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from pictograph.agents import Toolkit, create_toolkit
from pictograph.agents._registry import REGISTRY
from pictograph.exceptions import ConfigurationError
from pictograph.models.credit import CreditBalance


def _balance(remaining_micro_usd: int = 1_234_000_000) -> CreditBalance:
    return CreditBalance(
        included_remaining_micro_usd=remaining_micro_usd,
        included_allowance_micro_usd=10_000_000_000,
        credits_reset_at=None,
        recent_history=[],
    )


# ───────────── construction ─────────────


def test_toolkit_construction_from_client() -> None:
    client = MagicMock()
    tk = Toolkit(client, max_response_tokens=10000)
    assert tk.client is client
    assert tk.max_response_tokens == 10000


def test_create_toolkit_uses_provided_client() -> None:
    client = MagicMock()
    tk = create_toolkit(client=client)
    assert tk.client is client


def test_create_toolkit_without_api_key_raises() -> None:
    """No api_key + no env var → ConfigurationError from Client init."""
    import os

    saved = os.environ.pop("PICTOGRAPH_API_KEY", None)
    try:
        with pytest.raises(ConfigurationError):
            create_toolkit()
    finally:
        if saved:
            os.environ["PICTOGRAPH_API_KEY"] = saved


# ───────────── dispatch ─────────────


def test_dispatch_calls_handler_with_validated_args() -> None:
    """Dispatch validates via Pydantic, calls handler, returns dump."""
    client = MagicMock()
    client.credits.balance.return_value = _balance(remaining_micro_usd=500_000_000)
    tk = Toolkit(client)
    result = tk.dispatch("get_credit_balance", {})
    assert result["included_remaining_micro_usd"] == 500_000_000


def test_dispatch_validates_args_against_schema() -> None:
    from pydantic import ValidationError

    client = MagicMock()
    tk = Toolkit(client)
    # Missing required fields → Pydantic ValidationError.
    with pytest.raises(ValidationError):
        tk.dispatch("get_dataset", {})


def test_dispatch_rejects_extra_args() -> None:
    """All schemas use extra='forbid' - typo'd arg names should fail."""
    from pydantic import ValidationError

    client = MagicMock()
    tk = Toolkit(client)
    with pytest.raises(ValidationError):
        tk.dispatch("list_datasets", {"limit": 5, "bogus_field": True})


def test_dispatch_unknown_tool_raises_keyerror() -> None:
    client = MagicMock()
    tk = Toolkit(client)
    with pytest.raises(KeyError, match="nonexistent_tool"):
        tk.dispatch("nonexistent_tool", {})


def test_dispatch_estimate_credit_cost_round_trip() -> None:
    """Dispatch flows through to client.credits.estimate with the raw kwargs."""
    from pictograph.models.credit import CreditEstimate

    client = MagicMock()
    client.credits.estimate.return_value = CreditEstimate(
        operation="training_a10g",
        micro_usd_per_unit=22_917,
        unit="minute",
        quantity=30,
        total_micro_usd=687_510,
        sufficient=True,
        remaining_micro_usd=1_000_000_000,
    )
    tk = Toolkit(client)
    result = tk.dispatch(
        "estimate_credit_cost",
        {"operation": "training_a10g", "quantity": 30},
    )
    client.credits.estimate.assert_called_once_with("training_a10g", quantity=30)
    assert result["total_micro_usd"] == 687_510
    assert result["sufficient"] is True


# ───────────── token-budget enforcement ─────────────


def test_dispatch_truncates_oversized_response() -> None:
    """Result over max_response_tokens collapses to a marker dict."""
    client = MagicMock()
    # 100k chars ≈ 25k tokens at 4 chars/token.
    big_history = [
        {
            "id": str(i),
            "operation": "x",
            "amount": 1,
            "balance_after": i,
            "description": None,
            "metadata": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        for i in range(5000)
    ]
    client.credits.balance.return_value = CreditBalance(
        included_remaining_micro_usd=0,
        included_allowance_micro_usd=0,
        credits_reset_at=None,
        recent_history=[],
    )
    # Patch the dump path: replace _dump indirectly by making balance() huge.
    # Simpler: call with a tiny budget so any real result triggers truncation.
    tk = Toolkit(client, max_response_tokens=1)
    client.credits.balance.return_value = _balance(remaining_micro_usd=42_000_000)
    result = tk.dispatch("get_credit_balance", {})
    assert result.get("_truncated") is True
    assert "max_response_tokens=1" in result["_message"]
    # Ensure big_history doesn't sneak through.
    assert big_history is not None  # hush ruff


def test_dispatch_zero_budget_disables_truncation() -> None:
    """max_response_tokens<=0 means 'no truncation' (debug mode)."""
    client = MagicMock()
    client.credits.balance.return_value = _balance(remaining_micro_usd=42_000_000)
    tk = Toolkit(client, max_response_tokens=0)
    result = tk.dispatch("get_credit_balance", {})
    assert "_truncated" not in result
    assert result["included_remaining_micro_usd"] == 42_000_000


# ───────────── adapter outputs ─────────────


def test_as_anthropic_tools_shape() -> None:
    """Anthropic tool dicts have name/description/input_schema."""
    client = MagicMock()
    tools = Toolkit(client).as_anthropic_tools()
    assert len(tools) == len(REGISTRY)
    for tool in tools:
        assert set(tool) == {"name", "description", "input_schema"}
        assert tool["input_schema"]["type"] == "object"
        # Pydantic 'title' is stripped for clean schemas.
        assert "title" not in tool["input_schema"]


def test_as_openai_tools_shape() -> None:
    """OpenAI tool dicts have type='function' + name/description/parameters."""
    client = MagicMock()
    tools = Toolkit(client).as_openai_tools()
    assert len(tools) == len(REGISTRY)
    for tool in tools:
        assert tool["type"] == "function"
        assert "name" in tool
        assert "description" in tool
        assert tool["parameters"]["type"] == "object"


def test_as_json_schema_includes_metadata() -> None:
    """JSON Schema export carries required_role / cost_micro_usd / idempotent."""
    client = MagicMock()
    tools = Toolkit(client).as_json_schema()
    assert len(tools) == len(REGISTRY)
    for tool in tools:
        assert {
            "name",
            "description",
            "input_schema",
            "required_role",
            "cost_micro_usd",
            "idempotent",
        }.issubset(tool)
        assert isinstance(tool["cost_micro_usd"], int)


def test_anthropic_tools_match_registry_names() -> None:
    """Anthropic adapter exports every tool - no silent omissions."""
    client = MagicMock()
    names = {t["name"] for t in Toolkit(client).as_anthropic_tools()}
    assert names == {t.name for t in REGISTRY}


def test_openai_tools_match_registry_names() -> None:
    client = MagicMock()
    names = {t["name"] for t in Toolkit(client).as_openai_tools()}
    assert names == {t.name for t in REGISTRY}


def test_anthropic_tool_schemas_are_json_serialisable() -> None:
    """Schemas survive json.dumps (no Python-only types leaked)."""
    client = MagicMock()
    for tool in Toolkit(client).as_anthropic_tools():
        json.dumps(tool)  # raises TypeError if not serialisable


# ───────────── JSON-schema $ref inlining ─────────────


def test_all_registry_tool_schemas_are_self_contained() -> None:
    """No tool's emitted schema may carry unresolved $ref/$defs - Anthropic and
    OpenAI strict function-calling reject those. Locks the invariant."""
    from pictograph.agents._registry import REGISTRY as _REG
    from pictograph.agents._toolkit import _to_json_schema

    for tool in _REG:
        blob = json.dumps(_to_json_schema(tool))
        assert "$ref" not in blob, tool.name
        assert "$defs" not in blob, tool.name


def test_to_json_schema_inlines_nested_model_refs() -> None:
    """A nested-Pydantic-model arg (which makes Pydantic emit $defs/$ref) is
    inlined into a self-contained schema rather than left with dangling refs."""
    from types import SimpleNamespace

    from pydantic import BaseModel

    from pictograph.agents._toolkit import _to_json_schema

    class _Box(BaseModel):
        x: float
        y: float

    class _NestedArgs(BaseModel):
        box: _Box
        label: str

    schema = _to_json_schema(SimpleNamespace(args_schema=_NestedArgs))
    blob = json.dumps(schema)
    assert "$ref" not in blob and "$defs" not in blob
    # the nested model's fields are inlined under properties.box
    assert schema["properties"]["box"]["properties"]["x"]["type"] == "number"
    assert schema["properties"]["label"]["type"] == "string"
