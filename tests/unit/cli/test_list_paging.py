"""``--limit`` on a list command means how many rows to return, not one request's size.

Until 1.69.103, ``models list -n 150`` sent ``limit=150`` in ONE request, and the API
(which serves at most 100 models a page) answered 422 - an agent asking for "all my
models" got an error instead of a list. ``models``, ``deployments``, ``train``,
``datasets`` and ``exports`` now page through the SDK's pagers; the page-oriented
commands that take ``--offset`` (``tasks``, ``credits``, webhook deliveries) and
``notifications`` refuse a page larger than the API serves, before any request.

The real client over a mocked HTTP transport, so the requests on the wire are what is
checked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs

import httpx
import pytest
from typer.testing import CliRunner

from pictograph.cli._app import app

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_test")
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    return lambda *args: runner.invoke(app, list(args))


def _model(index: int) -> dict[str, Any]:
    return {
        "id": f"00000000-0000-0000-0000-{index:012d}",
        "organization_id": "org",
        "name": f"model-{index}",
        "model_type": "object_detection",
        "status": "ready",
        "visibility": "private",
        "created_at": "2026-04-01T00:00:00Z",
        "updated_at": "2026-04-01T00:00:00Z",
    }


def test_models_list_pages_past_the_apis_page_size(run: Any, httpx_mock: HTTPXMock) -> None:
    total = 230  # what the organization has

    def page(request: httpx.Request) -> httpx.Response:
        query = parse_qs(request.url.query.decode())
        limit, offset = int(query["limit"][0]), int(query["offset"][0])
        assert limit <= 100, "never ask the API for more than one page"
        rows = [_model(i) for i in range(offset, min(offset + limit, total))]
        has_more = offset + len(rows) < total
        return httpx.Response(200, json={"data": rows, "pagination": {"has_more": has_more}})

    httpx_mock.add_callback(
        page, url=re.compile(rf"{BASE}/api/v1/developer/models/.*"), is_reusable=True
    )
    res = run("models", "list", "-n", "150", "--json")
    assert res.exit_code == 0, res.output
    names = [row["name"] for row in json.loads(res.stdout)]
    assert names == [f"model-{i}" for i in range(150)]
    offsets = [int(parse_qs(r.url.query.decode())["offset"][0]) for r in httpx_mock.get_requests()]
    assert offsets == [0, 100], "two pages for 150 rows, and nothing past them"


@pytest.mark.parametrize(
    ("command", "cap"),
    [
        (["tasks", "list"], 100),
        (["credits", "history"], 100),
        (["notifications", "list"], 100),
        (["webhooks", "deliveries"], 200),
    ],
)
def test_a_page_larger_than_the_api_serves_is_refused_up_front(
    run: Any, httpx_mock: HTTPXMock, command: list[str], cap: int
) -> None:
    res = run(*command, "-n", str(cap + 1))
    assert res.exit_code == 2, res.output
    assert f"{cap + 1} is not in the range 1<=x<={cap}" in " ".join(res.output.split())
    assert httpx_mock.get_requests() == []
