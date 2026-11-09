"""Video Pydantic models - probe metadata + frame-extraction job state."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

VideoJobStatus = Literal["processing", "complete", "failed"]
"""Lifecycle of a frame-extraction job."""


class VideoMetadata(BaseModel):
    """Probe result for an uploaded video.

    Returned by :meth:`pictograph.resources.video.Video.probe`.
    """

    model_config = ConfigDict(extra="ignore")

    duration_seconds: float = Field(ge=0.0)
    native_fps: float = Field(ge=0.0, description="The video's frame rate.")
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    frame_count: int = Field(ge=0, description="Frames at its own rate.")


class VideoUploadInfo(BaseModel):
    """An uploaded video: ``video`` is the handle
    :meth:`~pictograph.resources.video.Video.probe` and
    :meth:`~pictograph.resources.video.Video.extract_frames` take."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # The signed URL the bytes were sent to - internal to the upload, never printed.
    upload_url: str = Field(exclude=True, repr=False)
    video: str = Field(validation_alias=AliasChoices("video", "gcs_path"))


class VideoExtractionJob(BaseModel):
    """Snapshot of a frame-extraction job."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # ``job_id``, or ``job`` from builds of 2026-08-01..10-02 (see BatchJob.job_id).
    job_id: str = Field(validation_alias=AliasChoices("job_id", "job"))
    status: VideoJobStatus
    progress: int = Field(default=0, ge=0, le=100)
    frames_extracted: int = Field(default=0, ge=0)
    total_frames: int = Field(default=0, ge=0)
    error: str | None = None
    directory_path: str | None = Field(
        default=None,
        description="The directory the frames went into, once it succeeds.",
    )
    warning: str | None = Field(
        default=None,
        description="Set when the image quota cut the frames short; the job still completes.",
    )
    image_ids: list[str] | None = Field(
        default=None,
        description="The frames' image ids, once the job is complete.",
    )
