"""Async Video resource - upload, probe, extract frames into a dataset.

Async twin of :class:`pictograph.resources.video.Video`.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

from pictograph.exceptions import ApiError, PollTimeoutError
from pictograph.models.video import (
    VideoExtractionJob,
    VideoJobStatus,
    VideoMetadata,
    VideoUploadInfo,
)
from pictograph.resources._base import AsyncResource

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_API_PATH = "/api/v1/developer/video"
_DEFAULT_POLL_INTERVAL = 3.0
_DEFAULT_TIMEOUT = 1800.0  # 30 min - matches the extraction service's soft cap
_TERMINAL_STATUSES: frozenset[VideoJobStatus] = frozenset({"complete", "failed"})


class AsyncVideo(AsyncResource):
    """Upload videos and extract frames into a dataset's virtual directory (async)."""

    # ───────────── upload ─────────────

    async def upload(
        self,
        local_path: str | Path,
        *,
        content_type: str = "video/mp4",
    ) -> VideoUploadInfo:
        """Upload a local video file to temporary storage (get-url → PUT bytes → path).

        Args:
            local_path: Path to the local video file.
            content_type: MIME type (default ``video/mp4``).

        Returns:
            :class:`VideoUploadInfo` - pass ``gcs_path`` to :meth:`probe` /
            :meth:`extract_frames`.

        Raises:
            FileNotFoundError: ``local_path`` doesn't exist.
            NetworkError / RequestTimeoutError: The upload PUT failed.
        """
        path = Path(local_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        info_response = await self._transport.request(
            "POST",
            f"{_API_PATH}/upload-url",
            json={"filename": path.name, "content_type": content_type},
        )
        upload_info = self._parse(VideoUploadInfo, info_response)

        await self._transport.upload_external(
            upload_info.upload_url,
            path,
            content_type=content_type,
        )

        return upload_info

    # ───────────── probe ─────────────

    async def probe(self, gcs_path: str) -> VideoMetadata:
        """Probe an uploaded video for duration / fps / dimensions.

        Slow on large files (backend runs ``ffprobe`` on the full bytes) - call
        once after :meth:`upload`, don't poll.

        Raises:
            NotFoundError: Path doesn't exist or belongs to another org.
            ValidationError: Backend couldn't parse the file as a video.
        """
        response = await self._transport.request(
            "POST", f"{_API_PATH}/probe", json={"gcs_path": gcs_path}
        )
        return self._parse(VideoMetadata, response)

    # ───────────── extract_frames ─────────────

    async def extract_frames(
        self,
        dataset_name: str,
        gcs_path: str,
        *,
        directory_name: str,
        sample_fps: float = 1.0,
        parent_directory_path: str = "/",
        wait: bool = True,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> VideoExtractionJob:
        """Extract frames from an uploaded video into a dataset directory.

        Frames are decoded at ``sample_fps``, uploaded under
        ``{dataset}/{parent_directory_path}/{directory_name}/``, and registered with
        SigLIP embeddings spawned automatically.

        Args:
            dataset_name: Project name within your org.
            gcs_path: From :meth:`upload`.
            directory_name: Virtual directory name to create for the frames.
            sample_fps: Frames per second to extract (default 1.0).
            parent_directory_path: Parent directory in the dataset (default ``"/"``).
            wait: When ``True`` (default), poll until terminal status.
            poll_interval: Seconds between polls (default 3s).
            timeout: Max seconds to wait (default 1800 = 30 min).

        Raises:
            NotFoundError: Dataset or gcs_path missing.
            ApiError: Job ended in ``failed`` status (when ``wait=True``).
            PollTimeoutError: Deadline elapsed (when ``wait=True``).
        """
        body = {
            "dataset_name": dataset_name,
            "gcs_path": gcs_path,
            "directory_name": directory_name,
            "sample_fps": sample_fps,
            "parent_directory_path": parent_directory_path,
        }
        response = await self._transport.request("POST", f"{_API_PATH}/extract-frames", json=body)
        job = self._parse(VideoExtractionJob, response)
        if not wait:
            return job
        return await self.wait_for_extraction(
            job.job_id, poll_interval=poll_interval, timeout=timeout
        )

    async def get_extraction(self, job_id: str) -> VideoExtractionJob:
        """Fetch the current status of a frame-extraction job."""
        response = await self._transport.request("GET", f"{_API_PATH}/extract-frames/{job_id}")
        return self._parse(VideoExtractionJob, response)

    async def wait_for_extraction(
        self,
        job_id: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> VideoExtractionJob:
        """Poll a frame-extraction job until terminal status."""
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else asyncio.sleep
        deadline = time.monotonic() + timeout
        while True:
            job = await self.get_extraction(job_id)
            if job.status == "complete":
                return job
            if job.status == "failed":
                raise ApiError(
                    f"Video extraction job {job_id} failed: "
                    f"{job.error or 'no error message provided'}",
                    response=job.model_dump(mode="json"),
                )
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Video extraction job {job_id} did not complete within "
                    f"{timeout:.0f}s (status: {job.status}). Fetch later via "
                    f"client.video.get_extraction(...)."
                )
            await sleep_fn(poll_interval)
