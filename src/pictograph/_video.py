"""Which frames of a video to run: the rule a Pictograph workflow uses.

``pictograph models predict --video`` and ``pictograph deployments predict --video``
decode a video on this machine and run the model on sampled frames. They sample
exactly as a Pictograph workflow does, so ``frame`` and ``t`` on a CLI line name the
same frame a workflow over the same file runs:

* stride = ``max(1, round(native_fps / max(sample_fps, 0.01)))``;
* a container that reports no frame rate is read as 30 fps;
* every frame is ``grab()``-bed, and only the sampled ones are decoded
  (``retrieve()``), so a 30 fps video sampled at 1 fps decodes 1 frame in 30.

This module imports nothing from ``pictograph`` and loads OpenCV lazily, on
purpose: Pictograph's platform tests load THIS FILE by path and run it beside the
workflow runner on the same videos, so the two cannot drift apart. Keep it
dependency-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

DEFAULT_SAMPLE_FPS = 1.0
# The runner's own floor on the sampling rate (`max(sample_fps, 0.01)`).
MIN_SAMPLE_FPS = 0.01
# The runner's frame rate for a container that does not report one.
FALLBACK_FPS = 30.0


def stride_for(native_fps: float, sample_fps: float) -> int:
    """Run every ``stride``-th frame: the video's rate over the rate asked for."""
    return max(1, round(native_fps / max(sample_fps, MIN_SAMPLE_FPS)))


def sample_frames(capture: Any, sample_fps: float) -> Iterator[tuple[int, float, Any]]:
    """``(frame index, seconds, BGR frame)`` for every sampled frame, in order.

    ``capture`` is an open ``cv2.VideoCapture``; the caller releases it.
    """
    import cv2

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 0:
        native_fps = FALLBACK_FPS
    stride = stride_for(native_fps, sample_fps)
    index = 0
    while capture.grab():
        if index % stride == 0:
            ok, frame = capture.retrieve()
            if not ok:
                return
            yield index, index / native_fps, frame
        index += 1


__all__ = ["DEFAULT_SAMPLE_FPS", "FALLBACK_FPS", "MIN_SAMPLE_FPS", "sample_frames", "stride_for"]
