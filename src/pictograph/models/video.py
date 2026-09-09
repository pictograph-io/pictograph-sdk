"""Video Pydantic models - probe metadata + frame-extraction job state."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VideoJobStatus = Literal["processing", "complete", "failed"]
"""Lifecycle of a frame-extraction job."""


class VideoMetadata(BaseModel):
    """Probe result for an uploaded video.

    Returned by :meth:`pictograph.resources.video.Video.probe`.
    """

    model_config = ConfigDict(extra="ignore")

    duration_seconds: float = Field(ge=0.0)
    native_fps: float = Field(ge=0.0, description="Source video frame rate.")
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    frame_count: int = Field(ge=0, description="Computed total frames at native fps.")


class VideoUploadInfo(BaseModel):
    """Signed URL + temporary storage path returned by ``upload-url``.

    The SDK uses ``upload_url`` to PUT bytes directly to storage, then passes
    ``gcs_path`` to :meth:`pictograph.resources.video.Video.probe` and
    :meth:`pictograph.resources.video.Video.extract_frames`.
    """

    model_config = ConfigDict(extra="ignore")

    upload_url: str
    gcs_path: str
    gcs_uri: str


class VideoExtractionJob(BaseModel):
    """Snapshot of a frame-extraction job."""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    status: VideoJobStatus
    progress: int = Field(default=0, ge=0, le=100)
    frames_extracted: int = Field(default=0, ge=0)
    total_frames: int = Field(default=0, ge=0)
    error: str | None = None
    directory_path: str | None = Field(
        default=None,
        description="Virtual directory where extracted frames land. Set on success.",
    )
    warning: str | None = Field(
        default=None,
        description=(
            "Set when extraction was truncated to fit the organization's image quota. "
            "The job still completes - with fewer frames than requested."
        ),
    )
    image_ids: list[str] | None = Field(
        default=None,
        description=(
            "The extracted frames' image ids. Populated on the terminal 'complete' poll "
            "only (a long video is thousands of ids, and this endpoint is polled). Lets "
            "you chain extraction straight into auto-annotate without re-listing the "
            "directory."
        ),
    )
