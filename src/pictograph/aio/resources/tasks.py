"""Async Tasks resource - annotation tasks and contribution tracking.

Async twin of :class:`pictograph.resources.tasks.Tasks`. Read-only: enumerate
the org's tasks and export a task's per-annotator contribution breakdown.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pictograph._http.pagination import AsyncOffsetPager
from pictograph.models.task import Task, TaskContributions
from pictograph.resources._base import AsyncResource

if TYPE_CHECKING:
    from collections.abc import Mapping

_API_PATH = "/api/v1/developer/tasks"


class AsyncTasks(AsyncResource):
    """List annotation tasks; read a task's contribution breakdown (async)."""

    async def list(self, *, limit: int = 50, offset: int = 0) -> list[Task]:
        """A single page of the organization's tasks, newest first."""
        response = await self._transport.request(
            "GET", f"{_API_PATH}", params={"limit": limit, "offset": offset}
        )
        return self._parse_list(Task, response.get("data", []))

    def iter(self, *, page_size: int = 100, max_total: int | None = None) -> AsyncOffsetPager[Task]:
        """Auto-paging async iterator across every task in the organization."""

        async def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            return cast(
                "Mapping[str, Any]",
                await self._transport.request(
                    "GET", f"{_API_PATH}", params={"offset": offset, "limit": limit}
                ),
            )

        return AsyncOffsetPager(
            fetch,
            items_key="data",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(Task, raw),
        )

    async def contributions(self, task_id: str) -> TaskContributions:
        """Per-annotator contribution breakdown for a task.

        Raises:
            NotFoundError: No task with that id in the key's organization.
        """
        response = await self._transport.request("GET", f"{_API_PATH}/{task_id}/contributions")
        return self._parse(TaskContributions, response["data"])
