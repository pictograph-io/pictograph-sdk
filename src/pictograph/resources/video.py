"""Video resource - upload, probe, extract frames into a dataset.

Three-step ingestion:

1. :meth:`Video.upload` - uploads a local video file to a temporary
   storage path (3-step: get-url → PUT bytes → return path). Returns
   :class:`VideoUploadInfo`.
2. :meth:`Video.probe` - runs ``ffprobe`` on the uploaded video and
   returns :class:`VideoMetadata`.
3. :meth:`Video.extract_frames` - kicks off a background frame-extraction
   job. Returns a :class:`VideoExtractionJob`. Pass ``wait=True`` (default)
   to poll until the job is terminal.

The uploaded video is temporary: the backend deletes it once extraction
completes, so ``video`` is not a durable handle.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from pictograph.exceptions import ApiError, PollTimeoutError, job_end_code
from pictograph.models.video import (
    VideoExtractionJob,
    VideoJobStatus,
    VideoMetadata,
    VideoUploadInfo,
)
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Callable

_API_PATH = "/api/v1/developer/video"
_DEFAULT_POLL_INTERVAL = 3.0
_DEFAULT_TIMEOUT = 1800.0  # 30 min - matches the extraction service's soft cap
_TERMINAL_STATUSES: frozenset[VideoJobStatus] = frozenset({"complete", "failed"})


class Video(Resource):
    """Upload videos and extract frames into a dataset's virtual directory."""

    # ───────────── upload ─────────────

    def upload(
        self,
        file_path: str | Path,
        *,
        content_type: str = "video/mp4",
    ) -> VideoUploadInfo:
        """Upload a local video file to a temporary storage path.

        Three-step internally: get signed URL → PUT bytes → return path.
        The returned ``video`` feeds :meth:`probe` and
        :meth:`extract_frames`. The temporary file is auto-deleted when
        extraction finishes.

        Args:
            file_path: Path to the local video file.
            content_type: MIME type (default ``video/mp4``). Set
                explicitly for non-MP4 sources (``video/quicktime``,
                ``video/webm``, etc.).

        Returns:
            :class:`VideoUploadInfo` - pass its ``video`` to subsequent
            video methods.

        Raises:
            FileNotFoundError: ``file_path`` doesn't exist.
            NetworkError / RequestTimeoutError: The upload PUT failed.
        """
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        # Step 1: get signed URL.
        info_response = self._transport.request(
            "POST",
            f"{_API_PATH}/upload-url",
            json={"filename": path.name, "content_type": content_type},
        )
        upload_info = self._parse(VideoUploadInfo, info_response)

        # Step 2: stream the bytes directly to object storage - no backend hop -
        # through the shared transport helper. It sets an explicit Content-Length
        # (a signed-URL PUT rejects chunked transfer encoding, which a
        # generator body would force) and maps transport/HTTP errors to the
        # SDK's typed NetworkError / RequestTimeoutError / ApiError. This is the
        # same proven path the image upload uses.
        self._transport.upload_external(
            upload_info.upload_url,
            path,
            content_type=content_type,
        )

        return upload_info

    # ───────────── probe ─────────────

    def probe(self, video: str) -> VideoMetadata:
        """Probe an uploaded video for duration / fps / dimensions.

        Slow on large files - backend downloads the full bytes and runs
        ``ffprobe``. Don't poll this; call once after :meth:`upload`.

        Args:
            video: The ``video`` returned from :meth:`upload`.

        Raises:
            NotFoundError: Path doesn't exist or belongs to another org.
            ValidationError: Backend couldn't parse the file as a video.
        """
        response = self._transport.request("POST", f"{_API_PATH}/probe", json={"video": video})
        return self._parse(VideoMetadata, response)

    # ───────────── extract_frames ─────────────

    def extract_frames(
        self,
        dataset: str,
        video: str,
        *,
        directory: str,
        sample_fps: float = 1.0,
        wait: bool = True,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> VideoExtractionJob:
        """Extract frames from an uploaded video into a dataset directory.

        Frames are decoded at ``sample_fps`` (lossy compression - pick the
        rate to match your annotation needs), saved into ``directory`` and
        registered as images, with their embeddings computed automatically.

        Args:
            dataset: Dataset name within your org.
            video: From :meth:`upload`.
            directory: The dataset directory the frames go into, e.g.
                ``"/clips/run-1"`` (created if missing).
            sample_fps: Frames per second to extract (default 1.0).
            wait: When ``True`` (default), poll until terminal status.
            poll_interval: Seconds between polls (default 3s).
            timeout: Max seconds to wait (default 1800 = 30 min).

        Returns:
            :class:`VideoExtractionJob` - terminal-state if ``wait=True``.

        Raises:
            NotFoundError: Dataset or video missing.
            ApiError: Job ended in ``failed`` status (when ``wait=True``).
            PollTimeoutError: Deadline elapsed (when ``wait=True``).
        """
        body = {
            "dataset": dataset,
            "video": video,
            "directory": directory,
            "sample_fps": sample_fps,
        }
        response = self._transport.request("POST", f"{_API_PATH}/extract-frames", json=body)
        job = self._parse(VideoExtractionJob, response)
        if not wait:
            return job
        return self.wait_for_extraction(job.job_id, poll_interval=poll_interval, timeout=timeout)

    def get_extraction(self, job: str) -> VideoExtractionJob:
        """Fetch the current status of a frame-extraction job."""
        response = self._transport.request("GET", f"{_API_PATH}/extract-frames/{job}")
        return self._parse(VideoExtractionJob, response)

    def wait_for_extraction(
        self,
        job: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        sleep: Callable[[float], None] | None = None,
    ) -> VideoExtractionJob:
        """Poll a frame-extraction current until terminal status."""
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else time.sleep
        deadline = time.monotonic() + timeout
        while True:
            current = self.get_extraction(job)
            if current.status == "complete":
                return current
            if current.status == "failed":
                raise ApiError(
                    f"Video extraction job {job} failed: "
                    f"{current.error or 'no error message provided'}",
                    response=current.model_dump(mode="json"),
                    code=job_end_code(current.status),
                )
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Video extraction job {job} did not complete within "
                    f"{timeout:.0f}s (status: {current.status}). Fetch later via "
                    f"client.video.get_extraction(...)."
                )
            sleep_fn(poll_interval)
