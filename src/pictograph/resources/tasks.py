"""Tasks resource - annotation tasks and per-annotator contribution tracking.

A **task** assigns a frozen set of a dataset's images to org members for
annotation or review. This resource is read-only: enumerate the org's tasks and
export a task's per-annotator :class:`TaskContributions` breakdown (who annotated
how many images, active editing time, annotations authored) - the same aggregate
the app's task modal shows, so app and SDK never diverge on the numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pictograph._http.pagination import OffsetPager
from pictograph.models.task import Task, TaskContributions
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Mapping

_API_PATH = "/api/v1/developer/tasks"


class Tasks(Resource):
    """List annotation tasks; read a task's contribution breakdown."""

    def list(self, *, limit: int = 50, offset: int = 0) -> list[Task]:
        """A single page of the organization's tasks, newest first.

        Args:
            limit: Page size (backend cap: 100).
            offset: Page offset for manual pagination.
        """
        response = self._transport.request(
            "GET", f"{_API_PATH}", params={"limit": limit, "offset": offset}
        )
        return self._parse_list(Task, response.get("data", []))

    def iter(self, *, page_size: int = 100, max_total: int | None = None) -> OffsetPager[Task]:
        """Auto-paging iterator across every task in the organization."""

        def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            return cast(
                "Mapping[str, Any]",
                self._transport.request(
                    "GET", f"{_API_PATH}", params={"offset": offset, "limit": limit}
                ),
            )

        return OffsetPager(
            fetch,
            items_key="data",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(Task, raw),
        )

    def contributions(self, task_id: str) -> TaskContributions:
        """Per-annotator contribution breakdown for a task.

        Args:
            task_id: The task's id (tasks have no unique name, so id only).

        Returns:
            :class:`TaskContributions` - ``contributors`` (each with
            ``images_worked`` / ``active_seconds`` / ``images_completed`` /
            ``annotations_added`` / ``is_assignee``) plus the rollup totals.

        Raises:
            NotFoundError: No task with that id in the key's organization.
        """
        response = self._transport.request("GET", f"{_API_PATH}/{task_id}/contributions")
        return self._parse(TaskContributions, response["data"])
