"""The async ``run`` takes what the sync one does: a link, bytes, a dataset or an export, a video's rate (B533 V4,
B606).

The sync twin's tests (``tests/unit/resources/test_workflow_run.py``) cover the whole contract; these hold the async
resource to the same wire calls, over the real async transport.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from pictograph.aio.resources.workflows import AsyncWorkflows
from pictograph.resources.workflows import URL_RUN_TIMEOUT_SECONDS

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
    httpx_mock.add_response(method="POST", url=f"{API}/{WF}/run", json=CREATED)
    await AsyncWorkflows(transport).run(WF, source="https://media.test/a/clip.webm")
    (request,) = httpx_mock.get_requests()
    assert json.loads(request.content) == {
        "source": {"kind": "video", "url": "https://media.test/a/clip.webm"}
    }
    assert request.extensions["timeout"]["read"] == URL_RUN_TIMEOUT_SECONDS


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
    httpx_mock.add_response(method="POST", url=f"{API}/{WF}/run", json=CREATED)
    await AsyncWorkflows(transport).run(WF, source=JPEG_HEAD)
    upload, put, invoke = httpx_mock.get_requests()
    assert json.loads(upload.content) == {"filename": "upload.jpg", "content_type": "image/jpeg"}
    assert put.content == JPEG_HEAD and put.headers["content-type"] == "image/jpeg"
    assert json.loads(invoke.content) == {"source": {"kind": "image", "image": "gs://b/o/u.jpg"}}


async def test_async_refuses_unreadable_input_before_any_request(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    with pytest.raises(ValueError, match="not a video or an image"):
        await AsyncWorkflows(transport).run(WF, source=b"plain text")
    with pytest.raises(ValueError, match="Say which"):
        await AsyncWorkflows(transport).run(WF, source="https://media.test/stream")
    assert httpx_mock.get_requests() == []


async def test_async_dataset_export_and_a_videos_rate(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    for _ in range(3):
        httpx_mock.add_response(method="POST", url=f"{API}/{WF}/run", json=CREATED)
    wf = AsyncWorkflows(transport)
    await wf.run(WF, source={"kind": "dataset", "dataset": "d", "directory": "/night"})
    await wf.run(WF, source={"kind": "export", "export": "e"})
    await wf.run(WF, source="https://media.test/a/clip.mp4", sample_fps=10)
    bodies = [json.loads(r.content) for r in httpx_mock.get_requests()]
    assert bodies == [
        {"source": {"kind": "dataset", "dataset": "d", "directory": "/night"}},
        {"source": {"kind": "export", "export": "e"}},
        {"source": {"kind": "video", "url": "https://media.test/a/clip.mp4", "sample_fps": 10.0}},
    ]
    with pytest.raises(ValueError, match="sample_fps is a video's"):
        await wf.run(WF, source={"kind": "export", "export": "e"}, sample_fps=10)


async def test_async_invoke_is_runs_old_name(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    httpx_mock.add_response(method="POST", url=f"{API}/{WF}/run", json=CREATED)
    created = await AsyncWorkflows(transport).invoke(WF, source={"kind": "dataset", "dataset": "d"})
    assert created.run_id == "run-9"
