"""OpenAI agents adapter.

Two integration paths:

1. **Raw OpenAI function-tool dicts** (always available, no extra deps):

       toolkit = create_toolkit()
       response = openai_client.responses.create(
           model="gpt-5",
           input=...,
           tools=toolkit.as_openai_tools(),
       )
       # Then dispatch tool calls via toolkit.dispatch(...)

2. **openai-agents ``function_tool`` decorator** (when ``openai-agents`` is
   installed via ``pip install pictograph[agents]``):

       toolkit = create_toolkit()
       agent_tools = for_openai_agents(toolkit)
       agent = Agent(name="...", tools=agent_tools)

The adapter is a thin shim - the registry is the source of truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pictograph.agents._registry import REGISTRY

if TYPE_CHECKING:
    from collections.abc import Callable

    from pictograph.agents._toolkit import Toolkit


def for_openai_agents(toolkit: Toolkit) -> list[Any]:
    """Wrap every registry tool as an ``openai-agents`` ``function_tool``.

    Requires ``openai-agents`` (install via ``pip install pictograph[agents]``).
    Raises ``ImportError`` with a clear remediation message when missing.

    Returns a list of ``FunctionTool`` objects ready to pass into
    ``Agent(tools=tools)``.
    """
    try:
        from agents import function_tool
    except ImportError as exc:
        raise ImportError(
            "openai-agents is required for for_openai_agents(). "
            "Install via: pip install 'pictograph[agents]'"
        ) from exc

    wrapped: list[Any] = []
    for descriptor in REGISTRY:
        wrapped.append(_wrap_for_openai(toolkit, descriptor.name, function_tool))
    return wrapped


def _wrap_for_openai(
    toolkit: Toolkit,
    name: str,
    function_tool_decorator: Callable[..., Any],
) -> Any:
    """Build one openai-agents ``FunctionTool`` for the given registry entry.

    openai-agents' ``function_tool`` introspects the wrapped function's
    signature and docstring to build the OpenAI function schema. We
    bypass that by passing ``params_json_schema`` directly - Pydantic's
    schema is more accurate than what the decorator would infer.
    """
    from pictograph.agents._registry import get_tool
    from pictograph.agents._toolkit import _to_json_schema

    descriptor = get_tool(name)
    # Route through the shared schema builder (NOT raw model_json_schema()) so the
    # function-tool path inlines $ref/$defs and strips title exactly like the
    # raw-dict adapter (as_openai_tools). OpenAI strict function-calling rejects
    # an unresolved $ref, which a nested-model arg would otherwise emit.
    schema = _to_json_schema(descriptor)

    async def _invoke(_ctx: Any, args_json: str) -> str:
        import json

        args = json.loads(args_json) if args_json else {}
        result = toolkit.dispatch(name, args)
        return json.dumps(result, default=str)

    return function_tool_decorator(
        _invoke,
        name_override=name,
        description_override=descriptor.description,
        params_json_schema=schema,
    )


def for_openai_responses(toolkit: Toolkit) -> list[dict[str, Any]]:
    """Return raw OpenAI function-tool dicts for ``client.responses.create(tools=...)``.

    Equivalent to ``toolkit.as_openai_tools()`` - exposed here for
    discoverability alongside the SDK adapter.
    """
    return toolkit.as_openai_tools()
