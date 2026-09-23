"""Claude Agent SDK adapter.

Two integration paths:

1. **Raw Anthropic tool dicts** (always available, no extra deps):

       toolkit = create_toolkit()
       message = anthropic_client.messages.create(
           model="claude-opus-4",
           tools=toolkit.as_anthropic_tools(),
           tool_choice={"type": "auto"},
           messages=...,
       )
       # Then dispatch tool_use blocks via toolkit.dispatch(...)

2. **Claude Agent SDK ``@tool`` decorator** (when ``claude-agent-sdk`` is
   installed via ``pip install pictograph[agents]``):

       toolkit = create_toolkit()
       agent_tools = for_claude_agent_sdk(toolkit)
       # Pass agent_tools to ClaudeAgent(...)

The adapter is a thin shim - the registry is the source of truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pictograph.agents._registry import REGISTRY

if TYPE_CHECKING:
    from collections.abc import Callable

    from pictograph.agents._toolkit import Toolkit


def for_claude_agent_sdk(toolkit: Toolkit) -> list[Any]:
    """Wrap every registry tool as a Claude Agent SDK ``@tool``.

    Requires ``claude-agent-sdk`` (install via ``pip install pictograph[agents]``).
    Raises ``ImportError`` with a clear remediation message when missing.

    Returns a list of ``@tool``-decorated callables ready to pass into
    ``ClaudeSDKClient(options=ClaudeAgentOptions(allowed_tools=[t.name for t in tools]))``
    or to register with an SDK MCP server.
    """
    try:
        from claude_agent_sdk import tool
    except ImportError as exc:
        raise ImportError(
            "claude-agent-sdk is required for for_claude_agent_sdk(). "
            "Install via: pip install 'pictograph[agents]'"
        ) from exc

    wrapped: list[Any] = []
    for descriptor in REGISTRY:
        # Closure captures the descriptor by name to avoid late-binding bugs.
        wrapped.append(_wrap_for_claude(toolkit, descriptor.name, tool))
    return wrapped


def _wrap_for_claude(
    toolkit: Toolkit,
    name: str,
    tool_decorator: Callable[..., Any],
) -> Any:
    """Build one Claude Agent SDK ``@tool`` for the given registry entry."""
    from pictograph.agents._registry import get_tool
    from pictograph.agents._toolkit import _to_json_schema

    descriptor = get_tool(name)
    # Route through the shared schema builder (NOT raw model_json_schema()) so the
    # @tool path inlines $ref/$defs and strips title exactly like the raw-dict
    # adapter (as_anthropic_tools). A nested-model arg would otherwise leave a
    # dangling $ref the Claude Agent SDK can't resolve.
    schema = _to_json_schema(descriptor)

    # Claude Agent SDK's @tool wants (name, description, input_schema, handler).
    # The handler signature is async (args: dict) -> dict.
    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        result = toolkit.dispatch(name, args)
        # Claude Agent SDK expects {"content": [{"type": "text", "text": "..."}]} format.
        import json

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, default=str, indent=2),
                }
            ]
        }

    return tool_decorator(name, descriptor.description, schema)(_handler)


def for_anthropic_messages(toolkit: Toolkit) -> list[dict[str, Any]]:
    """Return raw Anthropic tool dicts for ``client.messages.create(tools=...)``.

    Equivalent to ``toolkit.as_anthropic_tools()`` - exposed here for
    discoverability alongside the SDK adapter.
    """
    return toolkit.as_anthropic_tools()
