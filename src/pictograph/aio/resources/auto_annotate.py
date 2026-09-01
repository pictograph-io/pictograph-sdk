"""Async Auto-annotate resource - SAM3 point / box / text + batch jobs.

Async twin of :class:`pictograph.resources.auto_annotate.AutoAnnotate`.
Single-image prompts are fast; batch kicks off a background job you poll.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from pictograph.aio.resources import _resolve
from pictograph.exceptions import ApiError, PollTimeoutError
from pictograph.models.auto_annotate import (
    BatchClass,
    BatchJob,
    BatchJobStatus,
    BatchQuote,
    ProjectedImages,
    PromptResult,
)
from pictograph.resources._base import AsyncResource

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

_API_PATH = "/api/v1/developer/auto-annotate"
_DEFAULT_POLL_INTERVAL = 5.0
_DEFAULT_BATCH_TIMEOUT = 1800.0  # 30 minutes - SAM3 batch hard cap
_TERMINAL_STATUSES: frozenset[BatchJobStatus] = frozenset({"completed", "failed", "cancelled"})


class AsyncAutoAnnotate(AsyncResource):
    """SAM3 prompts (point / box / text) + batch jobs (async)."""

    # ───────────── single-image prompts ─────────────

    async def point(
        self,
        dataset_name: str,
        image_filename: str,
        *,
        x: int,
        y: int,
        name: str = "object",
        positive_points: Sequence[tuple[int, int]] | None = None,
        negative_points: Sequence[tuple[int, int]] | None = None,
        score_threshold: float = 0.75,
    ) -> PromptResult:
        """SAM3 point prompt → single polygon annotation.

        Args:
            dataset_name: Project name within your org.
            image_filename: Image filename (not the UUID).
            x, y: Anchor point coordinates in absolute pixels.
            name: Class label for the resulting annotation.
            positive_points: Extra positive anchors as ``[(x, y), ...]``.
            negative_points: Negative-prompt anchors (excluded regions).
            score_threshold: Minimum SAM3 score to include (0-1).

        Raises:
            NotFoundError: Dataset or image missing.
            PaymentRequiredError: Insufficient credits (3-credit minimum).
        """
        body: dict[str, Any] = {
            "dataset_name": dataset_name,
            "image_filename": image_filename,
            "x": x,
            "y": y,
            "name": name,
            "score_threshold": score_threshold,
        }
        if positive_points is not None:
            body["positive_points"] = [list(p) for p in positive_points]
        if negative_points is not None:
            body["negative_points"] = [list(p) for p in negative_points]
        response = await self._transport.request("POST", f"{_API_PATH}/sam3/point", json=body)
        ann = response.get("annotation")
        normalised = {
            "status": response.get("status", "success"),
            "annotations": [ann] if ann else [],
            "score": response.get("score"),
            "inference_time": response.get("inference_time"),
        }
        return self._parse(PromptResult, normalised)

    async def box(
        self,
        dataset_name: str,
        image_filename: str,
        *,
        box: dict[str, float],
        name: str,
        confidence_threshold: float = 0.5,
        return_polygon: bool = True,
        negative_boxes: Sequence[dict[str, float]] | None = None,
    ) -> PromptResult:
        """SAM3 box prompt → bbox + optional polygon annotation(s).

        Args:
            dataset_name: Project name.
            image_filename: Image filename.
            box: ``{"x", "y", "w", "h"}`` in absolute pixels.
            name: Class label.
            confidence_threshold: Minimum SAM3 confidence to include.
            return_polygon: When ``True`` (default), also include a polygon.
            negative_boxes: Boxes to exclude from segmentation.
        """
        body: dict[str, Any] = {
            "dataset_name": dataset_name,
            "image_filename": image_filename,
            "box": box,
            "name": name,
            "confidence_threshold": confidence_threshold,
            "return_polygon": return_polygon,
        }
        if negative_boxes is not None:
            body["negative_boxes"] = [dict(b) for b in negative_boxes]
        response = await self._transport.request("POST", f"{_API_PATH}/sam3/box", json=body)
        return self._parse(
            PromptResult,
            {
                "status": response.get("status", "success"),
                "annotations": response.get("annotations", []),
                "score": response.get("score"),
                "inference_time": response.get("inference_time"),
            },
        )

    async def text(
        self,
        dataset_name: str,
        image_filename: str,
        *,
        text_prompt: str,
        output_type: str = "polygon",
        confidence_threshold: float = 0.3,
        max_detections: int = 50,
    ) -> PromptResult:
        """SAM3 text prompt → list of detected annotations (sorted by confidence).

        Args:
            dataset_name, image_filename: As with :meth:`point` / :meth:`box`.
            text_prompt: Natural language description (e.g. ``"red car"``).
            output_type: ``"polygon"`` (default) or ``"bbox"``.
            confidence_threshold: Minimum confidence (0-1).
            max_detections: Cap on result count (1-100).
        """
        body: dict[str, Any] = {
            "dataset_name": dataset_name,
            "image_filename": image_filename,
            "text_prompt": text_prompt,
            "output_type": output_type,
            "confidence_threshold": confidence_threshold,
            "max_detections": max_detections,
        }
        response = await self._transport.request("POST", f"{_API_PATH}/sam3/text", json=body)
        return self._parse(
            PromptResult,
            {
                "status": response.get("status", "success"),
                "annotations": response.get("annotations", []),
                "score": None,
                "inference_time": response.get("inference_time"),
            },
        )

    # ───────────── batch ─────────────

    async def batch(
        self,
        dataset_name: str,
        image_filenames: Sequence[str],
        classes: Sequence[BatchClass | dict[str, Any]],
        *,
        confidence_threshold: float = 0.5,
        model: str | None = None,
        top_k: int = 1,
        sahi: bool = False,
        sahi_slice_size: int = 640,
        wait: bool = True,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_BATCH_TIMEOUT,
    ) -> BatchJob:
        """Spawn an auto-annotate batch job over many images.

        SAM3 path (``model_id=None``) requires ≥1 class; trained-model path
        (``model_id=<uuid>``) supports classification with empty classes.

        Args:
            dataset_name: Project name.
            image_filenames: Filenames to process (1-5000).
            classes: Class configs - :class:`BatchClass` instances or dicts.
            confidence_threshold: Minimum confidence to include.
            model: Trained model by NAME (a UUID also works), or ``None``
                (default) for SAM3.
            top_k: Classifier-only - predicted tags per image.
            sahi: SAHI sliced inference (SAM3 only).
            sahi_slice_size: SAHI tile edge in source pixels (256-1024).
            wait: Poll until terminal status (default ``True``).
            poll_interval: Seconds between checks (default 5s).
            timeout: Max seconds to wait (default 1800 = 30 min).

        Raises:
            NotFoundError: Dataset or no matching images.
            PaymentRequiredError: Insufficient credits for the estimate.
            ValidationError: SAM3 with no classes, SAHI with a trained model, etc.
            ApiError: Job ended in ``failed`` (when ``wait=True``).
            PollTimeoutError: Deadline elapsed (when ``wait=True``).
        """
        body: dict[str, Any] = {
            "dataset_name": dataset_name,
            "image_filenames": list(image_filenames),
            "classes": [
                c.model_dump(exclude_none=True) if isinstance(c, BatchClass) else dict(c)
                for c in classes
            ],
            "confidence_threshold": confidence_threshold,
            "top_k": top_k,
        }
        if model is not None:
            # "sam3" is a sentinel, not a model name - it must not be resolved.
            body["model_id"] = (
                model if model == "sam3" else await _resolve.model_id(self._transport, model)
            )
        if sahi:
            body["sahi_enabled"] = True
            body["sahi_slice_size"] = sahi_slice_size
        response = await self._transport.request("POST", f"{_API_PATH}/batch", json=body)
        job = self._parse(BatchJob, response)
        if not wait:
            return job
        return await self.wait_for_batch(job.job_id, poll_interval=poll_interval, timeout=timeout)

    async def quote(
        self,
        *,
        dataset_name: str | None = None,
        image_filenames: Sequence[str] = (),
        projected: Sequence[ProjectedImages | dict[str, Any]] = (),
        classes: Sequence[BatchClass | dict[str, Any]] = (),
        model: str | None = None,
        sahi: bool = False,
        sahi_slice_size: int = 640,
    ) -> BatchQuote:
        """Ask what a batch job would cost, WITHOUT running it.

        Async twin of :meth:`pictograph.resources.auto_annotate.AutoAnnotate.quote` -
        see it for the full contract. The quote and the charge come from one function on
        the server, so what you are told here is what :meth:`batch` will deduct.

        ``projected`` prices images that don't exist yet, which is how you price a video's
        frames (``floor(duration * fps)``) before uploading it.
        """
        body: dict[str, Any] = {
            "image_filenames": list(image_filenames),
            "projected": [
                p.model_dump(exclude_none=True) if isinstance(p, ProjectedImages) else dict(p)
                for p in projected
            ],
            "classes": [
                c.model_dump(exclude_none=True) if isinstance(c, BatchClass) else dict(c)
                for c in classes
            ],
        }
        if dataset_name is not None:
            body["dataset_name"] = dataset_name
        if model is not None:
            # "sam3" is a sentinel, not a model name - it must not be resolved.
            body["model_id"] = (
                model if model == "sam3" else await _resolve.model_id(self._transport, model)
            )
        if sahi:
            body["sahi_enabled"] = True
            body["sahi_slice_size"] = sahi_slice_size
        response = await self._transport.request("POST", f"{_API_PATH}/batch/quote", json=body)
        return self._parse(BatchQuote, response)

    async def get_batch(self, job_id: str) -> BatchJob:
        """Fetch the current status of a batch job."""
        response = await self._transport.request("GET", f"{_API_PATH}/batch/{job_id}")
        return self._parse(BatchJob, response)

    async def cancel_batch(self, job_id: str) -> BatchJob:
        """Cancel a pending or running batch job (does NOT refund pre-charge)."""
        response = await self._transport.request("POST", f"{_API_PATH}/batch/{job_id}/cancel")
        return self._parse(BatchJob, response)

    async def wait_for_batch(
        self,
        job_id: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_BATCH_TIMEOUT,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> BatchJob:
        """Poll a batch job until terminal status or timeout."""
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else asyncio.sleep
        deadline = time.monotonic() + timeout
        while True:
            job = await self.get_batch(job_id)
            if job.status == "completed":
                return job
            if job.status in ("failed", "cancelled"):
                raise ApiError(
                    f"Batch job {job_id} ended with status '{job.status}': "
                    f"{job.error_message or 'no error message provided'}",
                    response=job.model_dump(mode="json"),
                )
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Batch job {job_id} did not complete within {timeout:.0f}s "
                    f"(last status: {job.status}). Fetch later via "
                    f"client.auto_annotate.get_batch(...)."
                )
            await sleep_fn(poll_interval)
