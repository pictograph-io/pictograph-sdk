"""Pictograph agent toolkit - registry-driven adapters for Claude + OpenAI.

The toolkit exposes the SDK's high-leverage operations (workflows +
key resource methods) as language-agnostic tool descriptors. Adapters
translate the registry into the format each agent framework expects:

- :func:`for_anthropic_messages` / :func:`for_claude_agent_sdk` - Claude.
- :func:`for_openai_responses` / :func:`for_openai_agents` - OpenAI.
- :meth:`Toolkit.as_json_schema` - JSON Schema for dynamic-discovery
  agents (Vercel AI SDK, LangChain, etc.) via the
  ``GET /api/v1/developer/tools.json`` endpoint.

Quick start::

    from pictograph.agents import create_toolkit, for_anthropic_messages

    toolkit = create_toolkit(api_key="pk_live_...")
    tools = for_anthropic_messages(toolkit)

    # Pass tools to anthropic.messages.create(...)
    # Dispatch tool_use blocks via toolkit.dispatch(name, args)
"""

from __future__ import annotations

from pictograph.agents._registry import (
    REGISTRY,
    RequiredRole,
    ToolDescriptor,
    get_tool,
    tool_names,
)
from pictograph.agents._toolkit import Toolkit, create_toolkit
from pictograph.agents.claude import for_anthropic_messages, for_claude_agent_sdk
from pictograph.agents.openai import for_openai_agents, for_openai_responses

__all__ = [
    "REGISTRY",
    "RequiredRole",
    "ToolDescriptor",
    "Toolkit",
    "create_toolkit",
    "for_anthropic_messages",
    "for_claude_agent_sdk",
    "for_openai_agents",
    "for_openai_responses",
    "get_tool",
    "tool_names",
]
