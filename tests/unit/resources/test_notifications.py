"""Tests for ``pictograph.resources.notifications.Notifications``.

Coverage: list parses typed Notification models + threads the unread_only/limit
params; unread_count returns the int; mark_read POSTs to the id path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.models.notification import Notification
from pictograph.resources.notifications import Notifications

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
PATH = f"{BASE}/api/v1/developer/notifications"


@pytest.fixture
def transport() -> Transport:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def notifications(transport: Transport) -> Notifications:
    return Notifications(transport)


def _note(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "n1",
        "organization_id": "org1",
        "user_id": "u1",
        "type": "training_complete",
        "title": "Training complete",
        "message": "Your model 'Swift Falcon' is ready.",
        "metadata": {"run_id": "r1"},
        "read": False,
        "created_at": "2026-07-07T00:00:00Z",
    }
    base.update(overrides)
    return base


def test_list_parses_typed_notifications(
    httpx_mock: HTTPXMock, notifications: Notifications
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{PATH}?unread_only=false&limit=50&offset=0",
        json={"notifications": [_note(), _note(id="n2", read=True)], "unread_count": 1},
    )
    items = notifications.list()
    assert len(items) == 2
    assert all(isinstance(n, Notification) for n in items)
    assert items[0].type == "training_complete"
    assert items[0].metadata == {"run_id": "r1"}
    assert items[1].read is True


def test_list_threads_unread_and_limit_params(
    httpx_mock: HTTPXMock, notifications: Notifications
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{PATH}?unread_only=true&limit=10&offset=0",
        json={"notifications": [_note()], "unread_count": 1},
    )
    items = notifications.list(unread_only=True, limit=10)
    assert len(items) == 1


def test_unread_count_returns_int(httpx_mock: HTTPXMock, notifications: Notifications) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{PATH}/unread-count",
        json={"unread_count": 7},
    )
    assert notifications.unread_count() == 7


def test_mark_read_posts_to_id_path(httpx_mock: HTTPXMock, notifications: Notifications) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{PATH}/n1/read",
        json={"success": True, "id": "n1", "read": True},
    )
    assert notifications.mark_read("n1") is None


def test_mark_all_read_returns_marked_count(
    httpx_mock: HTTPXMock, notifications: Notifications
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{PATH}/read-all",
        json={"success": True, "marked_count": 3},
    )
    assert notifications.mark_all_read() == 3


def test_mark_all_read_defaults_to_zero(
    httpx_mock: HTTPXMock, notifications: Notifications
) -> None:
    httpx_mock.add_response(method="POST", url=f"{PATH}/read-all", json={"success": True})
    assert notifications.mark_all_read() == 0


def test_delete_sends_delete_to_id_path(
    httpx_mock: HTTPXMock, notifications: Notifications
) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{PATH}/n1",
        json={"success": True, "notification_id": "n1", "message": "Notification deleted"},
    )
    assert notifications.delete("n1") is None


def test_delete_unknown_id_raises_not_found(
    httpx_mock: HTTPXMock, notifications: Notifications
) -> None:
    from pictograph.exceptions import NotFoundError

    httpx_mock.add_response(
        method="DELETE",
        url=f"{PATH}/missing",
        status_code=404,
        json={"detail": "Notification not found"},
    )
    with pytest.raises(NotFoundError):
        notifications.delete("missing")
