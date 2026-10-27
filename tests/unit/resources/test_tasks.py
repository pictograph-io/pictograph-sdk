"""Tests for ``pictograph.resources.tasks.Tasks``.

Coverage: list parses typed Task models from the ``{"data": [...]}`` envelope +
threads limit/offset; contributions parses the nested TaskContributions with its
minute properties; a missing task id raises NotFoundError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.models.task import Task, TaskContributions
from pictograph.resources.tasks import Tasks

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
PATH = f"{BASE}/api/v1/developer/tasks"


@pytest.fixture
def transport() -> Transport:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def tasks(transport: Transport) -> Tasks:
    return Tasks(transport)


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


def test_list_parses_typed_tasks(httpx_mock: HTTPXMock, tasks: Tasks) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{PATH}?limit=50&offset=0",
        json={
            "data": [_task(), _task(id="t2", status="done")],
            "pagination": {"total": 2, "limit": 50, "offset": 0, "has_more": False},
        },
    )
    items = tasks.list()
    assert len(items) == 2
    assert all(isinstance(t, Task) for t in items)
    assert items[0].title == "Label the cars"
    assert items[0].image_count == 8 and items[0].assignee_count == 2
    assert items[1].status == "done"


def test_list_threads_limit_offset(httpx_mock: HTTPXMock, tasks: Tasks) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{PATH}?limit=10&offset=20",
        json={
            "data": [_task()],
            "pagination": {"total": 21, "limit": 10, "offset": 20, "has_more": False},
        },
    )
    assert len(tasks.list(limit=10, offset=20)) == 1


def test_contributions_parses(httpx_mock: HTTPXMock, tasks: Tasks) -> None:
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
                        "email": "ann@x",
                        "avatar_url": None,
                        "is_assignee": True,
                        "active_seconds": 90,
                        "images_worked": 4,
                        "images_completed": 2,
                        "annotations_added": 12,
                    },
                    {
                        "user_id": "u2",
                        "full_name": "Bo",
                        "email": None,
                        "avatar_url": None,
                        "is_assignee": False,
                        "active_seconds": 0,
                        "images_worked": 1,
                        "images_completed": 0,
                        "annotations_added": 3,
                    },
                ],
                "contributor_count": 2,
                "total_images": 8,
                "images_complete": 2,
                "total_active_seconds": 90,
            }
        },
    )
    c = tasks.contributions("t1")
    assert isinstance(c, TaskContributions)
    assert c.task_id == "t1"
    assert c.total_images == 8 and c.images_complete == 2
    top = c.contributors[0]
    assert top.full_name == "Ann" and top.is_assignee is True
    assert top.active_seconds == 90 and top.active_minutes == 1.5  # 90s -> 1.5m
    assert c.total_active_minutes == 1.5
    # A non-assignee author with no tracked time still reports the images they authored.
    assert c.contributors[1].is_assignee is False and c.contributors[1].images_worked == 1


def test_contributions_unknown_id_raises_not_found(httpx_mock: HTTPXMock, tasks: Tasks) -> None:
    from pictograph.exceptions import NotFoundError

    httpx_mock.add_response(
        method="GET",
        url=f"{PATH}/missing/contributions",
        status_code=404,
        json={"detail": "Task 'missing' not found"},
    )
    with pytest.raises(NotFoundError):
        tasks.contributions("missing")
