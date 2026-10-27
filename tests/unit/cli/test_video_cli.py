"""Tests for the ``pictograph video`` command group.

Self-contained: a :class:`~typer.testing.CliRunner` invokes the command group
in-process, the SDK client is a :class:`~unittest.mock.MagicMock` patched at the
``video`` command-module boundary, and ``~/.pictograph`` is redirected to a
``tmp_path`` so the suite never touches real config (mirrors ``test_exports_cli.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pictograph.cli._config import write_config
from pictograph.cli.commands.video import app
from pictograph.models.video import VideoExtractionJob, VideoMetadata, VideoUploadInfo


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ~/.pictograph/* to tmp_path so tests don't touch real config."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("PICTOGRAPH_API_KEY", raising=False)
    monkeypatch.setattr(
        "pictograph.cli._config.CONFIG_DIR",
        fake_home / ".pictograph",
    )
    monkeypatch.setattr(
        "pictograph.cli._config.CONFIG_PATH",
        fake_home / ".pictograph" / "config.toml",
    )
    return fake_home


def _upload_info() -> VideoUploadInfo:
    return VideoUploadInfo(
        upload_url="https://gcs.test/signed-put",
        gcs_path="org-1/temp/videos/clip.mp4",
        gcs_uri="gs://example-bucket/org-1/temp/videos/clip.mp4",
    )


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        duration_seconds=12.5,
        native_fps=30.0,
        width=1920,
        height=1080,
        frame_count=375,
    )


def _job(*, status: str = "complete") -> VideoExtractionJob:
    return VideoExtractionJob(
        job_id="job-uuid",
        status=status,  # type: ignore[arg-type]
        progress=100,
        frames_extracted=25,
        total_frames=25,
        directory_path="/clip",
    )


def _patch_client(client: MagicMock) -> object:
    """Patch ``get_client`` in the video command-module namespace."""
    return patch("pictograph.cli.commands.video.get_client", return_value=client)


# ───────────── upload ─────────────


def test_upload(runner: CliRunner, isolated_config: Path, tmp_path: Path) -> None:
    write_config(api_key="pk_live_x")
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fakevideobytes")  # the resource is mocked; just needs to exist
    client = MagicMock()
    client.video.upload.return_value = _upload_info()
    with _patch_client(client):
        res = runner.invoke(app, ["upload", str(src)])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["gcs_path"] == "org-1/temp/videos/clip.mp4"
    client.video.upload.assert_called_once_with(src, content_type="video/mp4")


# ───────────── probe ─────────────


def test_probe_renders_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.video.probe.return_value = _metadata()
    with _patch_client(client):
        res = runner.invoke(app, ["probe", "org-1/temp/videos/clip.mp4", "--json"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["frame_count"] == 375
    client.video.probe.assert_called_once_with("org-1/temp/videos/clip.mp4")


# ───────────── extract-frames ─────────────


def test_extract_frames_waits_for_completion(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.video.extract_frames.return_value = _job()
    with _patch_client(client):
        res = runner.invoke(
            app,
            [
                "extract-frames",
                "ds",
                "org-1/temp/videos/clip.mp4",
                "--directory-name",
                "clip",
                "--sample-fps",
                "2",
            ],
        )
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["status"] == "complete"
    client.video.extract_frames.assert_called_once_with(
        "ds",
        "org-1/temp/videos/clip.mp4",
        directory_name="clip",
        sample_fps=2.0,
        parent_directory_path="/",
        wait=True,
        poll_interval=3.0,
        timeout=1800.0,
    )


def test_extract_frames_no_wait(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.video.extract_frames.return_value = _job(status="processing")
    with _patch_client(client):
        res = runner.invoke(
            app,
            [
                "extract-frames",
                "ds",
                "org-1/temp/videos/clip.mp4",
                "--directory-name",
                "clip",
                "--no-wait",
            ],
        )
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["status"] == "processing"
    assert client.video.extract_frames.call_args.kwargs["wait"] is False


# ───────────── status ─────────────


def test_status_renders_json(runner: CliRunner, isolated_config: Path) -> None:
    write_config(api_key="pk_live_x")
    client = MagicMock()
    client.video.get_extraction.return_value = _job()
    with _patch_client(client):
        res = runner.invoke(app, ["status", "job-uuid", "--json"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["job_id"] == "job-uuid"
    assert payload["status"] == "complete"
    client.video.get_extraction.assert_called_once_with("job-uuid")
