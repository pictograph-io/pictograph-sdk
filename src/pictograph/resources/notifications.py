"""Notifications resource - the organization event feed for agents.

Poll job-lifecycle events (training complete, export ready, batch auto-annotate
done) an agent kicked off, instead of tracking every run id. All calls are scoped
to the API key's organization server-side.

    client = Client()
    for n in client.notifications.list(unread_only=True):
        print(n.type, n.title)
        client.notifications.mark_read(n.id)
"""

from __future__ import annotations

from pictograph.models.notification import Notification
from pictograph.resources._base import Resource

_API_PATH = "/api/v1/developer/notifications"


class Notifications(Resource):
    """List, count, and acknowledge organization notifications."""

    def list(
        self,
        *,
        unread_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Notification]:
        """A single page of notifications, newest first.

        Args:
            unread_only: Only return notifications not yet marked read.
            limit: Page size (backend cap: 100).
            offset: Page offset for manual pagination.
        """
        response = self._transport.request(
            "GET",
            _API_PATH,
            params={"unread_only": unread_only, "limit": limit, "offset": offset},
        )
        return self._parse_list(Notification, response.get("notifications", []))

    def unread_count(self) -> int:
        """The number of unread notifications for the organization."""
        response = self._transport.request("GET", f"{_API_PATH}/unread-count")
        return int(response.get("unread_count", 0))

    def mark_read(self, notification_id: str) -> None:
        """Mark one notification read (idempotent)."""
        self._transport.request("POST", f"{_API_PATH}/{notification_id}/read")

    def mark_all_read(self) -> int:
        """Mark every unread notification in the organization read.

        Returns the number newly marked read (idempotent - a second call
        returns ``0``).
        """
        response = self._transport.request("POST", f"{_API_PATH}/read-all")
        return int(response.get("marked_count", 0))

    def delete(self, notification_id: str) -> None:
        """Delete one notification (raises ``NotFoundError`` on an unknown id)."""
        self._transport.request("DELETE", f"{_API_PATH}/{notification_id}")
