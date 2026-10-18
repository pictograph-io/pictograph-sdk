"""Tests for ``client.workflows.invoke`` - the WF-D on-demand invoke (slice 1).

Covers: the per-mode request body (dataset + gcs-uri), None-field exclusion, the
client-side ``extra="forbid"`` guard on a dict source, the async return shape (run_id
+ status_url), and ``wait=True`` reusing ``wait_for_run`` (terminal + error).
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import ApiError
from pictograph.models.workflow import WorkflowInvokeSource, WorkflowRun, WorkflowRunCreated
from pictograph.resources.workflows import Workflows

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
_API = f"{BASE}/api/v1/developer/workflows"
WF = "fedcba98-1111-2222-3333-444455556666"  # a uuid -> no name-resolution GET


@pytest.fixture
def transport() -> Transport:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def workflows(transport: Transport) -> Workflows:
    return Workflows(transport)


def _run(**o: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "run-9",
        "organization_id": "org-1",
        "workflow_id": WF,
        "status": "processing",
        "progress": 0.0,
        "frames_total": None,
        "frames_done": 0,
        "sample_fps": None,
        "step_results": {},
        "artifacts": [],
        "warnings": [],
        "deposit_micro_usd": 1000,
        "final_micro_usd": None,
        "error": None,
        "created_at": "2026-09-23T00:00:00Z",
        "completed_at": None,
    }
    base.update(o)
    return base


def _created_json() -> dict[str, Any]:
    return {
        "success": True,
        "run_id": "run-9",
        "deposit_micro_usd": 1000,
        "status_url": "/api/v1/developer/workflows/runs/run-9",
    }


def test_invoke_dataset_async_returns_created(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(method="POST", url=f"{_API}/{WF}/invoke", json=_created_json())
    created = workflows.invoke(
        WF, source={"kind": "dataset", "dataset": "My Data", "directory_path": "/sub"}
    )
    assert isinstance(created, WorkflowRunCreated)
    assert created.run_id == "run-9"
    assert created.status_url == "/api/v1/developer/workflows/runs/run-9"
    body = _json.loads(httpx_mock.get_requests()[-1].content)
    assert body == {"source": {"kind": "dataset", "dataset": "My Data", "directory_path": "/sub"}}


def test_invoke_video_model_excludes_none_fields(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(method="POST", url=f"{_API}/{WF}/invoke", json=_created_json())
    src = WorkflowInvokeSource(kind="video", gcs_uri="gs://my-bucket/org/clip.mp4", sample_fps=2.0)
    workflows.invoke(WF, source=src)
    body = _json.loads(httpx_mock.get_requests()[-1].content)
    # dataset/directory_path/duration_seconds are None -> omitted from the wire body.
    assert body == {
        "source": {
            "kind": "video",
            "gcs_uri": "gs://my-bucket/org/clip.mp4",
            "sample_fps": 2.0,
        }
    }


def test_invoke_image_model_is_just_uri(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(method="POST", url=f"{_API}/{WF}/invoke", json=_created_json())
    workflows.invoke(
        WF, source=WorkflowInvokeSource(kind="image", gcs_uri="gs://my-bucket/org/a.png")
    )
    body = _json.loads(httpx_mock.get_requests()[-1].content)
    assert body == {"source": {"kind": "image", "gcs_uri": "gs://my-bucket/org/a.png"}}


def test_invoke_dict_source_forbids_unknown_field(workflows: Workflows) -> None:
    # A mistyped field is caught CLIENT-SIDE (extra="forbid"), before any request.
    with pytest.raises(ValidationError, match="datasets"):
        workflows.invoke(WF, source={"kind": "dataset", "datasets": "x"})


def test_invoke_wait_returns_terminal_run(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(method="POST", url=f"{_API}/{WF}/invoke", json=_created_json())
    # wait_for_run's FIRST get_run sees completed -> returns immediately (no sleep).
    httpx_mock.add_response(
        method="GET",
        url=f"{_API}/runs/run-9",
        json={"run": _run(status="completed", progress=100.0, frames_done=16, frames_total=16)},
    )
    run = workflows.invoke(
        WF, source={"kind": "dataset", "dataset": "d"}, wait=True, poll_interval=0.01, timeout=30.0
    )
    assert isinstance(run, WorkflowRun)
    assert run.status == "completed" and run.frames_done == 16


def test_invoke_wait_propagates_run_error(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(method="POST", url=f"{_API}/{WF}/invoke", json=_created_json())
    httpx_mock.add_response(
        method="GET", url=f"{_API}/runs/run-9", json={"run": _run(status="error", error="GPU OOM")}
    )
    with pytest.raises(ApiError, match="GPU OOM"):
        workflows.invoke(
            WF, source={"kind": "dataset", "dataset": "d"}, wait=True, poll_interval=0.01
        )


# --------------------------------------------------------------- local-file upload mode
def test_upload_source_streams_and_returns_uri(
    httpx_mock: HTTPXMock, workflows: Workflows, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00\x00")
    httpx_mock.add_response(
        method="POST",
        url=f"{_API}/upload-url",
        json={
            "success": True,
            "upload_url": "https://signed.example/put",
            "gcs_uri": "gs://b/org/clip.mp4",
        },
    )
    uploaded: list[Any] = []
    monkeypatch.setattr(
        workflows._transport,
        "upload_external",
        lambda url, path, *, content_type: uploaded.append((url, str(path), content_type)),
    )
    uri = workflows.upload_source(f)
    assert uri == "gs://b/org/clip.mp4"
    # It inferred video/mp4 from the extension and streamed to the signed URL.
    assert uploaded == [("https://signed.example/put", str(f), "video/mp4")]
    body = _json.loads(httpx_mock.get_requests()[-1].content)
    assert body == {"filename": "clip.mp4", "content_type": "video/mp4"}


def test_invoke_local_file_uploads_then_invokes(
    httpx_mock: HTTPXMock, workflows: Workflows, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    f = tmp_path / "scene.png"
    f.write_bytes(b"\x89PNG")
    httpx_mock.add_response(
        method="POST",
        url=f"{_API}/upload-url",
        json={
            "success": True,
            "upload_url": "https://signed.example/put",
            "gcs_uri": "gs://b/org/scene.png",
        },
    )
    httpx_mock.add_response(method="POST", url=f"{_API}/{WF}/invoke", json=_created_json())
    monkeypatch.setattr(workflows._transport, "upload_external", lambda *_a, **_k: None)
    created = workflows.invoke(WF, source=f)
    assert created.run_id == "run-9"
    # A .png local file -> uploaded -> invoked with an image gcs_uri source.
    invoke_body = _json.loads(httpx_mock.get_requests()[-1].content)
    assert invoke_body == {"source": {"kind": "image", "gcs_uri": "gs://b/org/scene.png"}}


def test_invoke_nonexistent_path_raises(workflows: Workflows) -> None:
    with pytest.raises(FileNotFoundError, match="not a local file"):
        workflows.invoke(WF, source="/no/such/file.mp4")


def test_prepare_invoke_source_infers_kind(tmp_path: Path) -> None:
    from pictograph.resources.workflows import _local_source_kind, _prepare_invoke_source

    vid = tmp_path / "a.mov"
    vid.write_bytes(b"x")
    assert _prepare_invoke_source(vid) == vid  # a local file -> needs upload
    assert _local_source_kind(vid) == "video"
    img = tmp_path / "a.WEBP"  # case-insensitive
    img.write_bytes(b"x")
    assert _local_source_kind(img) == "image"
    bad = tmp_path / "a.txt"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match="infer a video/image source"):
        _local_source_kind(bad)


# --------------------------------------------------------------- a link Pictograph downloads (B533 V4)
LINK = "https://media.test/clips/dock%20cam.MP4?sig=abc"


def test_invoke_link_sends_it_and_waits_for_the_download(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    from pictograph.resources.workflows import URL_INVOKE_TIMEOUT_SECONDS

    httpx_mock.add_response(method="POST", url=f"{_API}/{WF}/invoke", json=_created_json())
    workflows.invoke(WF, source=LINK)
    (request,) = httpx_mock.get_requests()  # no upload: the server fetches it
    assert _json.loads(request.content) == {"source": {"kind": "video", "url": LINK}}
    # The server answers once the file is DOWNLOADED (its deadline is ten minutes), so
    # this one request outwaits it; every other call keeps the client's own timeout.
    assert request.extensions["timeout"]["read"] == URL_INVOKE_TIMEOUT_SECONDS
    assert URL_INVOKE_TIMEOUT_SECONDS > 600


def test_an_image_link_is_an_image_and_http_is_a_link_too(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(method="POST", url=f"{_API}/{WF}/invoke", json=_created_json())
    workflows.invoke(WF, source="http://media.test/a/shelf.JPEG")
    body = _json.loads(httpx_mock.get_requests()[-1].content)
    assert body == {"source": {"kind": "image", "url": "http://media.test/a/shelf.JPEG"}}


def test_a_link_whose_kind_cant_be_read_is_refused_before_any_request(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    with pytest.raises(ValueError, match=r'source=\{"kind": "video", "url"'):
        workflows.invoke(WF, source="https://media.test/stream?id=4")
    assert httpx_mock.get_requests() == []


def test_a_link_with_its_kind_said_carries_the_frame_rate(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(method="POST", url=f"{_API}/{WF}/invoke", json=_created_json())
    workflows.invoke(
        WF, source={"kind": "video", "url": "https://media.test/live", "sample_fps": 2}
    )
    body = _json.loads(httpx_mock.get_requests()[-1].content)
    assert body == {
        "source": {"kind": "video", "url": "https://media.test/live", "sample_fps": 2.0}
    }


def test_a_stored_source_keeps_the_clients_own_timeout(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(method="POST", url=f"{_API}/{WF}/invoke", json=_created_json())
    workflows.invoke(WF, source={"kind": "dataset", "dataset": "d"})
    assert httpx_mock.get_requests()[-1].extensions["timeout"]["read"] == 10.0


# --------------------------------------------------------------- bytes (stdin, a buffer)
# The first bytes of files OpenCV 5.0's FFmpeg writer produced (captured 2026-10-17):
# what a phone, a camera or an encoder really emits, not a header typed from memory.
_MP4 = b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08free\x00\x00\x01\xee"
_MOV = b"\x00\x00\x00\x14ftypqt  \x00\x00\x02\x00qt  \x00\x00\x00\x08wide\x00\x00\x01\xeemdat"
_M4V = b"\x00\x00\x00\x1cftypM4V \x00\x00\x02\x00M4V isomiso2\x00\x00\x00\x08free\x00\x00\x01\xee"
_AVI = b"RIFFT\x1c\x00\x00AVI LIST\xec\x11\x00\x00hdrlavih8\x00\x00\x00\xa0\x86\x01\x00"
_MKV = (
    b"\x1aE\xdf\xa3\xa3B\x86\x81\x01B\xf7\x81\x01B\xf2\x81\x04B\xf3\x81\x08B\x82\x88matroskaB\x87"
)
_WEBM = (
    b"\x1aE\xdf\xa3\x9fB\x86\x81\x01B\xf7\x81\x01B\xf2\x81\x04B\xf3\x81\x08B\x82\x84webmB\x87\x81"
)


def _image(fmt: str) -> bytes:
    """A real image file, written by Pillow."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), (200, 40, 40)).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        (lambda: _image("JPEG"), ("image", "upload.jpg", "image/jpeg")),
        (lambda: _image("PNG"), ("image", "upload.png", "image/png")),
        (lambda: _image("GIF"), ("image", "upload.gif", "image/gif")),
        (lambda: _image("BMP"), ("image", "upload.bmp", "image/bmp")),
        (lambda: _image("WEBP"), ("image", "upload.webp", "image/webp")),
        (lambda: _MP4, ("video", "upload.mp4", "video/mp4")),
        (lambda: _MOV, ("video", "upload.mov", "video/quicktime")),
        (lambda: _M4V, ("video", "upload.m4v", "video/x-m4v")),
        (lambda: _AVI, ("video", "upload.avi", "video/x-msvideo")),
        (lambda: _MKV, ("video", "upload.mkv", "video/x-matroska")),
        (lambda: _WEBM, ("video", "upload.webm", "video/webm")),
    ],
    ids=["jpeg", "png", "gif", "bmp", "webp", "mp4", "mov", "m4v", "avi", "mkv", "webm"],
)
def test_bytes_are_read_by_their_signature(make: Any, expected: tuple[str, str, str]) -> None:
    from pictograph.resources.workflows import _sniff_media

    assert _sniff_media(make()) == expected


def test_invoke_bytes_uploads_them_then_runs(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    png = _image("PNG")
    httpx_mock.add_response(
        method="POST",
        url=f"{_API}/upload-url",
        json={
            "success": True,
            "upload_url": "https://signed.test/put",
            "gcs_uri": "gs://b/o/x.png",
        },
    )
    httpx_mock.add_response(method="PUT", url="https://signed.test/put", status_code=200)
    httpx_mock.add_response(method="POST", url=f"{_API}/{WF}/invoke", json=_created_json())
    created = workflows.invoke(WF, source=png)
    assert created.run_id == "run-9"
    upload, put, invoke = httpx_mock.get_requests()
    assert _json.loads(upload.content) == {"filename": "upload.png", "content_type": "image/png"}
    # The signed PUT carries the bytes themselves, typed as the URL was signed for.
    assert put.method == "PUT" and put.content == png
    assert put.headers["content-type"] == "image/png"
    assert put.headers["content-length"] == str(len(png))
    assert _json.loads(invoke.content) == {"source": {"kind": "image", "gcs_uri": "gs://b/o/x.png"}}


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (b"%PDF-1.7 not media", "not a video or an image"),
        (b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic", "HEIC and AVIF"),
        (b"\x00\x00\x00\x1cftypavif\x00\x00\x00\x00avifmif1", "HEIC and AVIF"),
        (b"", "empty"),
    ],
    ids=["pdf", "heic", "avif", "empty"],
)
def test_bytes_that_are_not_a_readable_video_or_image_are_refused_first(
    httpx_mock: HTTPXMock, workflows: Workflows, data: bytes, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        workflows.invoke(WF, source=data)
    assert httpx_mock.get_requests() == []


def test_upload_external_sends_bytes_and_reports_them_sent(
    httpx_mock: HTTPXMock, transport: Transport
) -> None:
    httpx_mock.add_response(method="PUT", url="https://signed.test/put", status_code=201)
    seen: list[tuple[int, int]] = []
    transport.upload_external(
        "https://signed.test/put",
        _MP4,
        content_type="video/mp4",
        progress=lambda sent, total: seen.append((sent, total)),
    )
    (put,) = httpx_mock.get_requests()
    assert put.content == _MP4 and put.headers["content-length"] == str(len(_MP4))
    assert seen == [(len(_MP4), len(_MP4))]
