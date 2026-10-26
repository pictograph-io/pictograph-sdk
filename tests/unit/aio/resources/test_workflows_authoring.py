"""The async ``nodes`` / ``templates`` / ``validate`` / ``create(template_key=)`` make the sync
twins' calls (B533 #17); tests/unit/resources/test_workflows_authoring.py covers the contract."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from pictograph.aio.resources.workflows import AsyncWorkflows

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

    from pictograph._http.async_transport import AsyncTransport

pytestmark = pytest.mark.anyio

API = "https://api.test.local/api/v1/developer/workflows"


async def test_async_nodes_and_templates(httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/nodes",
        json={
            "graph": {},
            "nodes": [
                {
                    "type": "sink",
                    "label": "Sink",
                    "summary": "Outputs.",
                    "inputs": ["detections"],
                    "output": None,
                    "new_block": {"kinds": ["json"]},
                    "fields": [],
                    "rules": [],
                }
            ],
        },
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{API}/templates",
        json={
            "templates": [
                {
                    "key": "custom",
                    "title": "Custom",
                    "description": "Blank.",
                    "graph": {"version": 1},
                }
            ]
        },
    )
    wf = AsyncWorkflows(transport)
    assert (await wf.nodes()).card("sink").output is None
    assert [t.key for t in await wf.templates()] == ["custom"]


async def test_async_validate_and_template_create(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{API}/validate", json={"valid": True, "issues": []}
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/",
        json={"workflow": {"id": "w", "organization_id": "o", "name": "Lot", "validation": []}},
    )
    wf = AsyncWorkflows(transport)
    assert (await wf.validate({"version": 1, "nodes": [], "edges": []})).valid is True
    await wf.create("Lot", template_key="parking")
    validate_req, create_req = httpx_mock.get_requests()
    assert json.loads(validate_req.content) == {"graph": {"version": 1, "nodes": [], "edges": []}}
    assert json.loads(create_req.content) == {"name": "Lot", "template_key": "parking"}
