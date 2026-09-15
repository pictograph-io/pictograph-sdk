"""Tests for ``pictograph.resources.video.Video``.

Coverage targets:
- ``upload``: 3-step flow (get-url + GCS PUT), file-not-found, GCS PUT failure.
- ``probe``: body shape, NotFound on missing path.
- ``extract_frames``: kicker body shape, wait=True polling, raises on failed.
- ``get_extraction``: typed response.
- ``wait_for_extraction``: argument validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import (
    ApiError,
    NotFoundError,
    PollTimeoutError,
)
from pictograph.models.video import (
    VideoExtractionJob,
    VideoMetadata,
    VideoUploadInfo,
)
from pictograph.resources.video import Video

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"


@pytest.fixture
def transport() -> Transport:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def video(transport: Transport) -> Video:
    return Video(transport)


def _upload_info(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "upload_url": "https://storage.googleapis.com/signed-url-here",
        "gcs_path": "org-uuid/temp/videos/abc_video.mp4",
        "gcs_uri": "gs://example-bucket/org-uuid/temp/videos/abc_video.mp4",
    }
    base.update(overrides)
    return base


def _probe_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "duration_seconds": 60.5,
        "native_fps": 29.97,
        "width": 1920,
        "height": 1080,
        "frame_count": 1813,
    }
    base.update(overrides)
    return base


def _job_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "job_id": "job-uuid-1",
        "status": "processing",
        "progress": 0,
        "frames_extracted": 0,
        "total_frames": 60,
        "error": None,
        "directory_path": None,
    }
    base.update(overrides)
    return base


# ───────────── upload ─────────────


def test_upload_3_step_flow(httpx_mock: HTTPXMock, video: Video, tmp_path: Path) -> None:
    """Upload calls get-url, then PUTs bytes to the signed URL."""
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/video/upload-url",
        json=_upload_info(),
    )
    httpx_mock.add_response(
        method="PUT",
        url="https://storage.googleapis.com/signed-url-here",
        status_code=200,
    )
    src = tmp_path / "video.mp4"
    src.write_bytes(b"fake mp4 bytes" * 100)
    info = video.upload(src)
    assert isinstance(info, VideoUploadInfo)
    assert info.gcs_path.endswith("video.mp4")

    requests = httpx_mock.get_requests()
    # 1) POST upload-url, 2) PUT to GCS
    assert len(requests) == 2
    assert requests[0].method == "POST"
    body = json.loads(requests[0].read())
    assert body == {"filename": "video.mp4", "content_type": "video/mp4"}
    assert requests[1].method == "PUT"
    assert str(requests[1].url) == "https://storage.googleapis.com/signed-url-here"
    # Regression: the GCS PUT MUST carry an explicit Content-Length (a GCS
    # signed-URL PUT rejects chunked transfer encoding). The upload routes
    # through Transport.upload_external rather than a generator body.
    assert requests[1].headers["content-length"] == str(len(b"fake mp4 bytes" * 100))
    assert requests[1].headers["content-type"] == "video/mp4"
    assert "transfer-encoding" not in {k.lower() for k in requests[1].headers}


def test_upload_file_not_found(video: Video) -> None:
    with pytest.raises(FileNotFoundError):
        video.upload("/does/not/exist.mp4")


def test_upload_gcs_put_failure(httpx_mock: HTTPXMock, video: Video, tmp_path: Path) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/video/upload-url",
        json=_upload_info(),
    )
    httpx_mock.add_response(
        method="PUT",
        url="https://storage.googleapis.com/signed-url-here",
        status_code=403,
        text="Signature mismatch",
    )
    src = tmp_path / "v.mp4"
    src.write_bytes(b"x" * 10)
    with pytest.raises(ApiError, match="Upload failed"):
        video.upload(src)


def test_upload_custom_content_type(httpx_mock: HTTPXMock, video: Video, tmp_path: Path) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/video/upload-url",
        json=_upload_info(),
    )
    httpx_mock.add_response(
        method="PUT",
        url="https://storage.googleapis.com/signed-url-here",
        status_code=200,
    )
    src = tmp_path / "v.mov"
    src.write_bytes(b"x" * 10)
    video.upload(src, content_type="video/quicktime")
    body = json.loads(httpx_mock.get_requests()[0].read())
    assert body["content_type"] == "video/quicktime"


# ───────────── probe ─────────────


def test_probe_returns_typed_metadata(httpx_mock: HTTPXMock, video: Video) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/video/probe",
        json=_probe_payload(),
    )
    metadata = video.probe("org-uuid/temp/videos/x.mp4")
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {"gcs_path": "org-uuid/temp/videos/x.mp4"}
    assert isinstance(metadata, VideoMetadata)
    assert metadata.duration_seconds == 60.5
    assert metadata.frame_count == 1813


def test_probe_404(httpx_mock: HTTPXMock, video: Video) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/video/probe",
        status_code=404,
        json={"detail": "Video file not found in GCS"},
    )
    with pytest.raises(NotFoundError):
        video.probe("missing-path")


# ───────────── extract_frames ─────────────


def test_extract_frames_kicker_body(httpx_mock: HTTPXMock, video: Video) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/video/extract-frames",
        json=_job_payload(),
    )
    job = video.extract_frames(
        "road-signs",
        "org-uuid/temp/videos/abc.mp4",
        directory_name="extracted",
        sample_fps=2.0,
        wait=False,
    )
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {
        "dataset_name": "road-signs",
        "gcs_path": "org-uuid/temp/videos/abc.mp4",
        "directory_name": "extracted",
        "sample_fps": 2.0,
        "parent_directory_path": "/",
    }
    assert isinstance(job, VideoExtractionJob)


def test_extract_frames_wait_polls_until_complete(
    httpx_mock: HTTPXMock, video: Video, monkeypatch: pytest.MonkeyPatch
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/video/extract-frames",
        json=_job_payload(),
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/video/extract-frames/job-uuid-1",
        json=_job_payload(status="processing", progress=50, frames_extracted=30),
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/video/extract-frames/job-uuid-1",
        json=_job_payload(
            status="complete",
            progress=100,
            frames_extracted=60,
            directory_path="/extracted",
        ),
    )
    sleeps: list[float] = []
    monkeypatch.setattr("pictograph.resources.video.time.sleep", lambda d: sleeps.append(d))
    job = video.extract_frames(
        "ds",
        "org-uuid/temp/videos/x.mp4",
        directory_name="frames",
        wait=True,
        poll_interval=1.0,
        timeout=60.0,
    )
    assert job.status == "complete"
    assert job.frames_extracted == 60
    assert job.directory_path == "/extracted"
    assert sleeps == [1.0]


def test_extract_frames_wait_raises_on_failed(
    httpx_mock: HTTPXMock, video: Video, monkeypatch: pytest.MonkeyPatch
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/video/extract-frames",
        json=_job_payload(),
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/video/extract-frames/job-uuid-1",
        json=_job_payload(
            status="failed",
            error="ffmpeg returned non-zero exit code",
        ),
    )
    monkeypatch.setattr("pictograph.resources.video.time.sleep", lambda _: None)
    with pytest.raises(ApiError, match="ffmpeg"):
        video.extract_frames(
            "ds",
            "org-uuid/temp/videos/x.mp4",
            directory_name="frames",
            wait=True,
            poll_interval=0.1,
            timeout=10.0,
        )


def test_extract_frames_wait_polltimeout(
    httpx_mock: HTTPXMock, video: Video, monkeypatch: pytest.MonkeyPatch
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/video/extract-frames",
        json=_job_payload(),
    )
    for _ in range(3):
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE}/api/v1/developer/video/extract-frames/job-uuid-1",
            json=_job_payload(status="processing", progress=20),
        )
    times = iter([100.0, 100.0, 105.0, 110.0])
    monkeypatch.setattr("pictograph.resources.video.time.monotonic", lambda: next(times))
    monkeypatch.setattr("pictograph.resources.video.time.sleep", lambda _: None)
    with pytest.raises(PollTimeoutError, match="did not complete"):
        video.extract_frames(
            "ds",
            "org-uuid/temp/videos/x.mp4",
            directory_name="frames",
            wait=True,
            poll_interval=0.1,
            timeout=10.0,
        )


def test_extract_frames_404_dataset(httpx_mock: HTTPXMock, video: Video) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/video/extract-frames",
        status_code=404,
        json={"detail": "Dataset 'missing' not found"},
    )
    with pytest.raises(NotFoundError):
        video.extract_frames(
            "missing",
            "org-uuid/temp/videos/x.mp4",
            directory_name="frames",
            wait=False,
        )


# ───────────── get_extraction / wait_for_extraction ─────────────


def test_get_extraction_returns_typed_job(httpx_mock: HTTPXMock, video: Video) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/video/extract-frames/job-x",
        json=_job_payload(status="complete", progress=100, frames_extracted=42),
    )
    job = video.get_extraction("job-x")
    assert job.frames_extracted == 42


def test_wait_for_extraction_argument_validation(video: Video) -> None:
    with pytest.raises(ValueError, match="poll_interval"):
        video.wait_for_extraction("j", poll_interval=0.0)
    with pytest.raises(ValueError, match="timeout"):
        video.wait_for_extraction("j", timeout=0.0)
