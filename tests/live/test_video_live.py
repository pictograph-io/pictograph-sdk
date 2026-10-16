"""Live: video - upload + probe.

Video upload + extract-frames is heavy; we smoke-test upload+probe only.
The full extract_frames path is tested via a separate manual run.
"""

from __future__ import annotations

import pytest

from pictograph import Client
from pictograph.models.video import VideoMetadata, VideoUploadInfo

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory) -> str | None:
    """Create a tiny test video using ffmpeg if available."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available - video tests require it")
    out = tmp_path_factory.mktemp("video") / "tiny.mp4"
    # 1s 64x64 solid-color video.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x64:d=1",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return str(out)


def test_video_upload_and_probe(client: Client, sample_video: str) -> None:
    info = client.video.upload(sample_video, content_type="video/mp4")
    assert isinstance(info, VideoUploadInfo)
    assert info.gcs_path

    meta = client.video.probe(info.gcs_path)
    assert isinstance(meta, VideoMetadata)
    assert meta.duration_seconds > 0
    assert meta.width == 64
    assert meta.height == 64
