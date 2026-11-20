"""The Live log on the async client: ``async for record in client.deployments.logs(...)`` - the sync twin's
arguments, checked when called (never at the first iteration), and every page read for the caller."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

from pictograph.aio.resources.deployments import AsyncDeployments
from pictograph.aio.resources.workflows import AsyncWorkflows
from pictograph.exceptions import NotFoundError
from pictograph.models.annotation import BBoxAnnotation
from pictograph.resources._live_log import LIVE_NOT_YET

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

    from pictograph._http.async_transport import AsyncTransport

pytestmark = pytest.mark.anyio

BASE = "https://api.test.local"
DEP_LOGS = re.compile(
    re.escape(f"{BASE}/api/v1/developer/deployments/loading-dock/logs") + r"(\?.*)?$"
)
WF_LOGS = re.compile(re.escape(f"{BASE}/api/v1/developer/workflows/crosswalk/logs") + r"(\?.*)?$")


def _frame(i: int) -> dict[str, Any]:
    return {
        "cursor": f"c-{i}",
        "kind": "frame",
        "time": "2026-11-20T18:22:11.099326Z",
        "deployment": "loading-dock",
        "session": "0123456789abcdef",
        "run": "0123456789abcdef",
        "seq": 1,
        "index": i,
        "annotations": [
            {
                "bounding_box": {"h": 1.0, "w": 1.0, "x": 0.0, "y": 0.0},
                "confidence": 0.9,
                "id": f"a-{i}",
                "name": "car",
                "type": "bbox",
            }
        ],
    }


def _page(records: list[dict[str, Any]], nxt: str | None) -> dict[str, Any]:
    return {
        "records": records,
        "next_cursor": nxt,
        "since": "2026-11-20T17:22:00Z",
        "until": "2026-11-20T18:22:00Z",
        "frames_logged": True,
    }


async def test_logs_is_an_async_iterator_over_every_page(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    httpx_mock.add_response(url=DEP_LOGS, json=_page([_frame(0), _frame(1)], "c-1"))
    httpx_mock.add_response(url=DEP_LOGS, json=_page([_frame(2)], None))
    got = [
        r
        async for r in AsyncDeployments(transport).logs("loading-dock", since="1h", classes=["car"])
    ]
    assert [r.index for r in got] == [0, 1, 2]
    assert isinstance(got[0].annotations[0], BBoxAnnotation)
    first, second = httpx_mock.get_requests()
    assert first.url.params["since"] == "1h" and first.url.params["classes"] == "car"
    assert (
        second.url.params["cursor"] == "c-1"
        and second.url.params["since"] == "2026-11-20T17:22:00Z"
    )


async def test_its_arguments_are_checked_when_called(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    with pytest.raises(ValueError, match="kinds are"):
        AsyncWorkflows(transport).logs("crosswalk", kinds=["frames"])  # not awaited, not iterated
    with pytest.raises(NotImplementedError) as e:
        AsyncDeployments(transport).logs("loading-dock", live=True)
    assert str(e.value) == LIVE_NOT_YET
    assert httpx_mock.get_requests() == []


async def test_a_workflows_log_and_a_refusal(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    httpx_mock.add_response(url=WF_LOGS, json=_page([_frame(0)], None))
    assert (
        len(
            [
                r
                async for r in AsyncWorkflows(transport).logs(
                    "crosswalk", kinds=["event", "capture"]
                )
            ]
        )
        == 1
    )
    assert httpx_mock.get_requests()[0].url.params.get_list("kinds") == ["event", "capture"]
    httpx_mock.add_response(
        url=DEP_LOGS, status_code=404, json={"detail": "Deployment 'loading-dock' not found"}
    )
    with pytest.raises(NotFoundError):
        async for _ in AsyncDeployments(transport).logs("loading-dock"):
            pass
