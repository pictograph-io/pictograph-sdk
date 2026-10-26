"""The async ``invoke`` takes what the sync one does: a link and bytes (B533 V4).

The sync twin's tests (``tests/unit/resources/test_workflow_invoke.py``) cover the
whole contract; these hold the async resource to the same wire calls, over the real
async transport.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from pictograph.aio.resources.workflows import AsyncWorkflows
from pictograph.resources.workflows import URL_INVOKE_TIMEOUT_SECONDS

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

    from pictograph._http.async_transport import AsyncTransport

pytestmark = pytest.mark.anyio

API = "https://api.test.local/api/v1/developer/workflows"
WF = "fedcba98-1111-2222-3333-444455556666"
CREATED = {"success": True, "run_id": "run-9", "deposit_micro_usd": 1000}
JPEG_HEAD = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"


async def test_async_link_is_fetched_by_the_server(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    httpx_mock.add_response(method="POST", url=f"{API}/{WF}/invoke", json=CREATED)
    await AsyncWorkflows(transport).invoke(WF, source="https://media.test/a/clip.webm")
    (request,) = httpx_mock.get_requests()
    assert json.loads(request.content) == {
        "source": {"kind": "video", "url": "https://media.test/a/clip.webm"}
    }
    assert request.extensions["timeout"]["read"] == URL_INVOKE_TIMEOUT_SECONDS


async def test_async_bytes_are_uploaded_then_run(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/upload-url",
        json={
            "success": True,
            "upload_url": "https://signed.test/put",
            "gcs_uri": "gs://b/o/u.jpg",
        },
    )
    httpx_mock.add_response(method="PUT", url="https://signed.test/put", status_code=200)
    httpx_mock.add_response(method="POST", url=f"{API}/{WF}/invoke", json=CREATED)
    await AsyncWorkflows(transport).invoke(WF, source=JPEG_HEAD)
    upload, put, invoke = httpx_mock.get_requests()
    assert json.loads(upload.content) == {"filename": "upload.jpg", "content_type": "image/jpeg"}
    assert put.content == JPEG_HEAD and put.headers["content-type"] == "image/jpeg"
    assert json.loads(invoke.content) == {"source": {"kind": "image", "gcs_uri": "gs://b/o/u.jpg"}}


async def test_async_refuses_unreadable_input_before_any_request(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    with pytest.raises(ValueError, match="not a video or an image"):
        await AsyncWorkflows(transport).invoke(WF, source=b"plain text")
    with pytest.raises(ValueError, match="Say which"):
        await AsyncWorkflows(transport).invoke(WF, source="https://media.test/stream")
    assert httpx_mock.get_requests() == []
