"""Async twin of ``tests/unit/resources/test_tasks.py``.

Proves ``AsyncTasks`` unwraps the ``{"data": ...}`` envelope and parses the same
typed models as the sync ``Tasks`` - two clients, one contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from pictograph.aio.resources.tasks import AsyncTasks
from pictograph.models.task import Task, TaskContributions

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

    from pictograph._http.async_transport import AsyncTransport

pytestmark = pytest.mark.anyio

PATH = "https://api.test.local/api/v1/developer/tasks"


def _task(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "t1",
        "project_id": "p1",
        "project": "Demo",
        "title": "Label the cars",
        "kind": "annotate",
        "status": "open",
        "created_at": "2026-08-05T00:00:00Z",
        "image_count": 8,
        "assignee_count": 2,
    }
    base.update(overrides)
    return base


async def test_async_list_parses(httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{PATH}?limit=50&offset=0",
        json={
            "data": [_task()],
            "pagination": {"total": 1, "limit": 50, "offset": 0, "has_more": False},
        },
    )
    items = await AsyncTasks(transport).list()
    assert len(items) == 1 and isinstance(items[0], Task)
    assert items[0].image_count == 8


async def test_async_contributions_parses(httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{PATH}/t1/contributions",
        json={
            "data": {
                "task_id": "t1",
                "contributors": [
                    {
                        "user_id": "u1",
                        "full_name": "Ann",
                        "email": None,
                        "avatar_url": None,
                        "is_assignee": True,
                        "active_seconds": 120,
                        "images_worked": 5,
                        "images_completed": 3,
                        "annotations_added": 20,
                    }
                ],
                "contributor_count": 1,
                "total_images": 8,
                "images_complete": 3,
                "total_active_seconds": 120,
            }
        },
    )
    c = await AsyncTasks(transport).contributions("t1")
    assert isinstance(c, TaskContributions)
    assert c.task_id == "t1" and c.total_images == 8
    assert c.contributors[0].active_minutes == 2.0  # 120s
