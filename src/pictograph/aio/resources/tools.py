"""Async Tools resource - the vision tools your organization can call, and how.

Async twin of :class:`pictograph.resources.tools.Tools`: SAM3, each ready model, each
active deployment and each workflow, each with a card saying what it is and exactly
how to call it. See that module for the full description.
"""

from __future__ import annotations

from pictograph.models.tool import ToolCard, ToolKind, ToolSummary
from pictograph.resources._base import AsyncResource
from pictograph.resources.tools import _API_PATH, card_path, list_params


class AsyncTools(AsyncResource):
    """List the organization's vision tools and read the card for one (async)."""

    async def list(self, *, kind: ToolKind | None = None) -> list[ToolSummary]:
        """Every vision tool this organization can call right now.

        Args:
            kind: Only tools of this kind - ``"sam3"``, ``"model"``,
                ``"deployment"`` or ``"workflow"``.

        Returns:
            One :class:`~pictograph.models.tool.ToolSummary` per tool. Pass its
            ``tool`` to :meth:`describe`.
        """
        rows: list[ToolSummary] = []
        while True:
            response = await self._transport.request(
                "GET", f"{_API_PATH}/", params=list_params(kind, len(rows))
            )
            rows.extend(self._parse_list(ToolSummary, response.get("data", [])))
            if not response.get("pagination", {}).get("has_more") or not response.get("data"):
                return rows

    async def describe(self, tool: str) -> ToolCard:
        """One tool's card: what it is and exactly how to call it.

        Args:
            tool: The tool's handle - ``"sam3"``, ``"model/<name>"``,
                ``"deployment/<name>"`` or ``"workflow/<name>"``. A UUID works in
                place of a name.

        Returns:
            A :class:`~pictograph.models.tool.ToolCard`.

        Raises:
            ValueError: ``tool`` is not a tool handle.
            NotFoundError: No such tool in your organization.
            ConflictError: The model exists but is not ready yet.
        """
        response = await self._transport.request("GET", card_path(tool))
        return self._parse(ToolCard, response["data"])
