"""Async Notifications resource - the organization event feed.

Async twin of :class:`pictograph.resources.notifications.Notifications`. Poll
job-lifecycle events (training complete, export ready) an agent kicked off.
"""

from __future__ import annotations

from pictograph.models.notification import Notification
from pictograph.resources._base import AsyncResource

_API_PATH = "/api/v1/developer/notifications"


class AsyncNotifications(AsyncResource):
    """List, count, and acknowledge organization notifications (async)."""

    async def list(
        self,
        *,
        unread_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Notification]:
        """A single page of notifications, newest first."""
        response = await self._transport.request(
            "GET",
            _API_PATH,
            params={"unread_only": unread_only, "limit": limit, "offset": offset},
        )
        return self._parse_list(Notification, response.get("notifications", []))

    async def unread_count(self) -> int:
        """The number of unread notifications for the organization."""
        response = await self._transport.request("GET", f"{_API_PATH}/unread-count")
        return int(response.get("unread_count", 0))

    async def mark_read(self, notification_id: str) -> None:
        """Mark one notification read (idempotent)."""
        await self._transport.request("POST", f"{_API_PATH}/{notification_id}/read")

    async def mark_all_read(self) -> int:
        """Mark every unread notification in the organization read.

        Returns the number newly marked read (idempotent - a second call
        returns ``0``).
        """
        response = await self._transport.request("POST", f"{_API_PATH}/read-all")
        return int(response.get("marked_count", 0))

    async def delete(self, notification_id: str) -> None:
        """Delete one notification (raises ``NotFoundError`` on an unknown id)."""
        await self._transport.request("DELETE", f"{_API_PATH}/{notification_id}")
