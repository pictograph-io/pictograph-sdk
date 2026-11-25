"""``Toolkit`` - agent-facing entry point that owns a Client and the registry.

Adapters (Claude, OpenAI, dynamic-discovery via tools.json) build their
respective tool surfaces from this object. The toolkit also enforces
the ``max_response_tokens`` budget so large list/get operations don't
blow up the agent's context window.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pictograph.agents._registry import REGISTRY, ToolDescriptor, get_tool

if TYPE_CHECKING:
    from pictograph import Client


class Toolkit:
    """Agent-facing toolkit. Wraps a :class:`pictograph.Client` and the registry.

    Build via :func:`create_toolkit` - this constructor is for advanced use
    (already-instantiated client).

    Attributes:
        client: Authenticated :class:`pictograph.Client`. Handlers receive this.
        max_response_tokens: Soft budget for tool outputs. When a handler
            returns more than this (estimated at 4 chars/token), the
            toolkit truncates and adds a ``_truncated`` marker. Agents
            re-call with narrower filters (e.g. lower limit).
    """

    def __init__(self, client: Client, *, max_response_tokens: int = 25000) -> None:
        self.client = client
        self.max_response_tokens = max_response_tokens

    # ───────────── dispatch ─────────────

    def dispatch(self, name: str, args: dict[str, Any] | None = None) -> Any:
        """Validate ``args`` against the tool's schema and invoke its handler.

        Args:
            name: Registered tool name (see :func:`tool_names`).
            args: Raw kwargs from the agent. Validated through the tool's
                Pydantic schema; ``ValidationError`` is raised on bad input.

        Returns:
            The handler's JSON-serialisable result, possibly truncated to
            ``max_response_tokens``. Truncated payloads carry
            ``{"_truncated": true, "_message": "..."}``.

        Raises:
            KeyError: ``name`` is not a registered tool.
            ValidationError: ``args`` failed Pydantic validation.
        """
        tool = get_tool(name)
        validated = tool.args_schema.model_validate(args or {})
        result = tool.handler(self.client, validated)
        return self._enforce_token_budget(result, tool)

    def _enforce_token_budget(self, result: Any, tool: ToolDescriptor) -> Any:
        """Truncate result if its serialised size exceeds the budget.

        Token count is estimated via ``len(serialised) / 4`` - coarse but
        adequate for soft-cap enforcement. Agents that need more context
        re-call with tighter filters.
        """
        if self.max_response_tokens <= 0:
            return result
        serialised = json.dumps(result, default=str)
        if len(serialised) // 4 <= self.max_response_tokens:
            return result
        return {
            "_truncated": True,
            "_message": (
                f"Response exceeded max_response_tokens={self.max_response_tokens}. "
                f"Tool '{tool.name}' returned ~{len(serialised) // 4} tokens. "
                f"Re-call with narrower filters (lower limit, smaller class_filter, etc.)."
            ),
            "_size_chars": len(serialised),
        }

    # ───────────── adapter surfaces ─────────────

    def as_anthropic_tools(self) -> list[dict[str, Any]]:
        """Tool dicts in Anthropic's raw tool-use schema (``Anthropic.messages.create(tools=...)``).

        Returns one dict per registered tool with ``name``, ``description``,
        ``input_schema`` (JSON Schema). Pair with :meth:`dispatch` in the
        message loop.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": _to_json_schema(tool),
            }
            for tool in REGISTRY
        ]

    def as_openai_tools(self) -> list[dict[str, Any]]:
        """Tool dicts in OpenAI's function-calling schema (``OpenAI.responses.create(tools=...)``)."""
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": _to_json_schema(tool),
            }
            for tool in REGISTRY
        ]

    def as_json_schema(self) -> list[dict[str, Any]]:
        """Tool registry as plain JSON Schema array.

        This is the same shape served by the backend's
        ``GET /api/v1/developer/tools.json`` endpoint - agents using
        dynamic discovery (Vercel AI SDK, LangChain, etc.) consume this
        directly without a bespoke adapter.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": _to_json_schema(tool),
                "required_role": tool.required_role,
                "cost_micro_usd": tool.cost_micro_usd,
                "idempotent": tool.idempotent,
            }
            for tool in REGISTRY
        ]


def _inline_refs(node: Any, defs: dict[str, Any], _stack: tuple[str, ...] = ()) -> Any:
    """Recursively replace ``{"$ref": "#/$defs/X"}`` with the resolved ``$defs[X]``.

    Sibling keys on a ``$ref`` node (e.g. an overriding ``description``) are
    merged over the resolved definition. A cyclic reference (a def reachable
    from itself) is left as a bare ``$ref`` to avoid infinite recursion -
    none of the current arg models are recursive, but the guard keeps this safe.
    """
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref[len("#/$defs/") :]
            if name in _stack:  # cycle - don't recurse forever
                return dict(node)
            resolved = _inline_refs(defs.get(name, {}), defs, (*_stack, name))
            if isinstance(resolved, dict):
                merged = dict(resolved)
                merged.update({k: v for k, v in node.items() if k != "$ref"})
                return merged
            return resolved
        return {k: _inline_refs(v, defs, _stack) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(item, defs, _stack) for item in node]
    return node


def _to_json_schema(tool: ToolDescriptor) -> dict[str, Any]:
    """Generate a clean, self-contained JSON Schema from a tool's Pydantic args model.

    Pydantic's ``model_json_schema`` emits a top-level ``$defs`` section with
    local ``$ref`` pointers whenever a model uses a nested model. Anthropic and
    OpenAI both want self-contained schemas (and OpenAI strict function-calling
    rejects unresolved ``$ref``), so we inline every ``$ref`` and drop ``$defs``.
    Schemas without references (the current registry) pass through unchanged
    apart from the dropped ``title``.
    """
    schema: dict[str, Any] = tool.args_schema.model_json_schema()
    defs = schema.pop("$defs", None)
    if defs:
        schema = _inline_refs(schema, defs)
    schema.pop("title", None)
    return schema


def create_toolkit(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    max_response_tokens: int = 25000,
    client: Client | None = None,
) -> Toolkit:
    """Convenience constructor - builds a Client + Toolkit in one call.

    Args:
        api_key: API key. Reads from ``PICTOGRAPH_API_KEY`` env var if absent.
        base_url: Override base URL (e.g. for staging). Defaults to production.
        max_response_tokens: Soft budget for tool outputs (default 25k).
        client: Pre-built client. When provided, ``api_key`` and ``base_url``
            are ignored.

    Returns:
        :class:`Toolkit` ready for adapter consumption.

    Raises:
        ConfigurationError: ``api_key`` is missing and no env var is set.
    """
    if client is None:
        from pictograph import Client

        client = Client(api_key=api_key, base_url=base_url)
    return Toolkit(client, max_response_tokens=max_response_tokens)
