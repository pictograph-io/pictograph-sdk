"""Tests for ``pictograph.agents._registry``.

The registry is the single source of truth for adapter outputs, so the
contract here is: every entry has a coherent shape (name, description,
schema, handler), names are unique, and every Pydantic args schema
generates a valid JSON Schema. Adapter rendering is tested separately.
"""

from __future__ import annotations

from pydantic import BaseModel

from pictograph.agents import (
    REGISTRY,
    ToolDescriptor,
    get_tool,
    tool_names,
)


def test_registry_is_non_empty() -> None:
    """Sanity: the registry has tools (we ship 28 in v1)."""
    assert len(REGISTRY) >= 25
    assert all(isinstance(t, ToolDescriptor) for t in REGISTRY)


def test_registry_names_are_unique() -> None:
    """No two tools share a name - adapters dispatch by name."""
    names = [t.name for t in REGISTRY]
    assert len(names) == len(set(names)), "duplicate tool name"


def test_registry_names_are_snake_case_identifiers() -> None:
    """Tool names are valid Python identifiers and snake_case (no hyphens)."""
    for tool in REGISTRY:
        assert tool.name.isidentifier(), f"{tool.name!r} not a valid identifier"
        assert tool.name.islower(), f"{tool.name!r} not lowercase"
        assert "-" not in tool.name


def test_registry_descriptions_lead_with_use() -> None:
    """Descriptions start with 'Use' (Anthropic 'use when X' pattern)."""
    for tool in REGISTRY:
        assert tool.description.startswith("Use"), (
            f"{tool.name!r}: description should start with 'Use'"
        )


def test_registry_args_schemas_are_pydantic_models() -> None:
    for tool in REGISTRY:
        assert issubclass(tool.args_schema, BaseModel), (
            f"{tool.name!r}: args_schema is not a Pydantic model"
        )


def test_registry_args_schemas_emit_valid_json_schema() -> None:
    """Every Pydantic model generates a JSON Schema with ``properties`` (or no params)."""
    for tool in REGISTRY:
        schema = tool.args_schema.model_json_schema()
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        # GetCreditBalance has no params; the rest do.
        if tool.name != "get_credit_balance":
            assert schema.get("properties"), f"{tool.name!r}: schema missing 'properties'"


def test_registry_args_schemas_forbid_extra_fields() -> None:
    """All input schemas use extra='forbid' - agents must pass exactly the declared fields."""
    for tool in REGISTRY:
        cfg = tool.args_schema.model_config
        assert cfg.get("extra") == "forbid", f"{tool.name!r}: schema must forbid extra fields"


def test_registry_handlers_are_callable() -> None:
    for tool in REGISTRY:
        assert callable(tool.handler)


def test_registry_required_roles_are_valid() -> None:
    valid = {"viewer", "member", "admin", "owner"}
    for tool in REGISTRY:
        assert tool.required_role in valid


def test_registry_costs_are_non_negative() -> None:
    for tool in REGISTRY:
        assert tool.cost_micro_usd >= 0


def test_registry_paid_tools_have_micro_usd_cost() -> None:
    """The compute-intensive workflow tools carry a non-zero µUSD estimate."""
    paid = {"auto_annotate_dataset", "train_pipeline"}
    for tool in REGISTRY:
        if tool.name in paid:
            assert tool.cost_micro_usd > 0, f"{tool.name!r} should have a non-zero µUSD cost"


def test_registry_destructive_tools_require_admin() -> None:
    """Dataset deletion gates on admin (irreversible op)."""
    delete_dataset = get_tool("delete_dataset")
    assert delete_dataset.required_role == "admin"


def test_get_tool_lookup() -> None:
    for tool in REGISTRY:
        assert get_tool(tool.name) is tool


def test_get_tool_raises_keyerror_for_unknown() -> None:
    import pytest

    with pytest.raises(KeyError, match="not_a_real_tool"):
        get_tool("not_a_real_tool")


def test_tool_names_returns_registry_order() -> None:
    assert tool_names() == [t.name for t in REGISTRY]


def test_pipeline_tools_present() -> None:
    """The three staged pipeline entries are the headline agent surface."""
    expected = {
        "upload_dataset_from_directory",
        "auto_annotate_dataset",
        "train_pipeline",
    }
    names = set(tool_names())
    assert expected.issubset(names)


def test_full_pipeline_is_not_a_registered_tool() -> None:
    """``full_pipeline`` was removed outright - never hand an agent a tool that
    chains upload -> auto-annotate -> train blindly, with no review of the
    generated annotations and no training config. Agents stage the three
    explicit calls instead. Do NOT re-add it."""
    assert "full_pipeline" not in set(tool_names())
