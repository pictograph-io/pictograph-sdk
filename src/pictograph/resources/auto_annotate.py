"""Auto-annotate resource - SAM3 point / box / text + batch jobs.

Single-image SAM3 calls are synchronous and fast (subsecond after the
first warmup). The batch surface kicks off a background job and returns
a :class:`BatchJob` that callers poll to terminal status.

Annotations land in canonical Pictograph JSON - the same Pydantic models
:class:`pictograph.annotations.save` consumes. Common pattern:

    result = client.auto_annotate.box(
        "road-signs", "frame_001.jpg",
        box={"x": 100, "y": 100, "w": 200, "h": 200},
        name="stop_sign",
    )
    if result.status == "success":
        client.annotations.save("road-signs", "img-001.jpg", result.annotations)

Charging happens server-side, in micro-USD (1 USD = 1_000_000 µUSD). Do NOT reproduce the
pricing formula here - it depends on image dimensions, class count and container fan-out,
and the underlying rates can change at runtime. Ask instead::

    quote = client.auto_annotate.quote(
        dataset="road-signs",
        image_filenames=[...],
        classes=[BatchClass(name="car", output_type="bbox")],
    )
    quote.estimated_credits  # exactly what batch() will deduct

:meth:`AutoAnnotate.quote` runs the same function the deposit runs, so the estimate IS
the charge. It can also price images that don't exist yet (``projected``) - which is how
you find out what labelling a video's frames will cost before you upload it.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from pictograph.exceptions import (
    ApiError,
    PictographError,
    PollTimeoutError,
    job_end_code,
)
from pictograph.models.auto_annotate import (
    BatchClass,
    BatchJob,
    BatchJobStatus,
    BatchQuote,
    ProjectedImages,
    PromptResult,
    SimilarOutputType,
    SimilarQuote,
    SimilarScope,
)
from pictograph.resources import _resolve
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pictograph.models.annotation import Annotation
    from pictograph.models.image import Image

_API_PATH = "/api/v1/developer/auto-annotate"
_DEFAULT_POLL_INTERVAL = 5.0
_DEFAULT_BATCH_TIMEOUT = 1800.0  # 30 minutes - SAM3 batch hard cap
_TERMINAL_STATUSES: frozenset[BatchJobStatus] = frozenset({"completed", "failed", "cancelled"})

AnnotateMode = Literal["batch", "text"]
"""``batch`` is the default - one async job over many images. ``text``
runs a synchronous SAM3 text prompt per image (slower; useful for small
datasets or single-image debugging)."""

# Images listed per request while :meth:`AutoAnnotate.dataset` reads a dataset (the
# developer images list's own page cap). It pages to the end, so a dataset of any
# size is read in full - the single ``Datasets.get(include_images=True)`` call it
# used before stopped at 10,000 images and silently never saw the rest.
_LIST_PAGE_SIZE = 1000
# Backend per-call cap on the SAM3 batch endpoint's image_filenames.
_BATCH_IMAGE_CAP = 5000


def _similar_body(
    dataset: str,
    image: str,
    *,
    annotation_ids: Sequence[str],
    reference_class: str | None,
    scope: str,
    target_class: str | None,
    output_type: str,
    confidence: float | None,
) -> dict[str, Any]:
    """The ``/similar`` (and ``/similar/quote``) body - one builder for both twins.

    The examples are named one way, before any request: by id or by class.
    """
    ids = list(dict.fromkeys(annotation_ids))
    if ids and reference_class:
        raise ValueError("Name the examples one way: annotations or reference_class, not both.")
    if not ids and not reference_class:
        raise ValueError(
            "Name the example polygon(s): annotations, or reference_class for every "
            "polygon of that class on the image."
        )
    body: dict[str, Any] = {
        "dataset": dataset,
        "image": image,
        "scope": scope,
        "output_type": output_type,
    }
    if ids:
        body["annotations"] = ids
    else:
        body["reference_class"] = reference_class
    if target_class is not None:
        body["target_class"] = target_class
    if confidence is not None:
        # Omitted, the server's default applies - the app modal's, one number.
        body["confidence"] = confidence
    return body


class AnnotationFailure(BaseModel):
    """One image that failed to auto-annotate in :meth:`AutoAnnotate.dataset`.

    ``image_filename`` is ``"<batch>"`` for a failure of the whole batch job and
    ``"<batch-cap>"`` for eligible images held back by the per-job limit. When the
    failure was an SDK error, ``error_type`` names its class (``"PaymentRequiredError"``,
    ``"PollTimeoutError"``, ...) and ``code`` carries its ``ApiError.code``
    (``"insufficient_credits"``, ``"job_failed"``, ...) so a caller can branch on it.
    """

    model_config = ConfigDict(frozen=True)

    image_filename: str
    reason: str
    error_type: str | None = None
    code: str | None = None

    @classmethod
    def from_error(cls, image_filename: str, error: PictographError) -> AnnotationFailure:
        """A failure recording ``error``: its message, its class and its code."""
        return cls(
            image_filename=image_filename,
            reason=str(error),
            error_type=type(error).__name__,
            code=error.code if isinstance(error, ApiError) else None,
        )


class AnnotateReport(BaseModel):
    """Outcome of an :meth:`AutoAnnotate.dataset` call.

    ``job_id`` is set only when ``mode="batch"`` - the underlying
    ``BatchJob`` ID for follow-up polling or cancellation.
    """

    dataset_name: str
    images_attempted: int = 0
    images_processed: int = 0
    # Skipped ONLY because already annotated (when ``overwrite=False``) - does
    # NOT include images held back by ``max_images`` (those are ``images_capped``).
    images_skipped: int = 0
    # Eligible images held back this run - by ``max_images`` and/or the SAM3
    # per-batch cap (a >cap remainder also adds a ``<batch-cap>`` entry to
    # ``failures``, so ``success`` is False until the dataset is fully processed).
    images_capped: int = 0
    annotations_added: int = 0
    failures: list[AnnotationFailure] = Field(default_factory=list)
    job_id: str | None = None
    # The batch job's status when the call returned: ``"completed"`` (etc.) after a
    # wait, ``"pending"`` / ``"running"`` when submitted with ``wait=False``.
    job_status: str | None = None

    @property
    def success(self) -> bool:
        return not self.failures and self.images_processed > 0


class AutoAnnotate(Resource):
    """SAM3 prompts (point / box / text) + batch jobs."""

    # ───────────── single-image prompts ─────────────

    def point(
        self,
        dataset: str,
        image: str,
        *,
        x: int,
        y: int,
        name: str = "object",
        positive_points: Sequence[tuple[int, int]] | None = None,
        negative_points: Sequence[tuple[int, int]] | None = None,
        score_threshold: float = 0.75,
    ) -> PromptResult:
        """SAM3 point prompt → single polygon annotation.

        The first prompt on a new image triggers GPU embedding generation
        (~1-2s on a warm container). Subsequent prompts on the same image
        are sub-second.

        Args:
            dataset: Dataset name within your org.
            image: The image: its filename (the dataset root first), its
                path inside the dataset (``"val/x.jpg"``), or its id.
            x, y: Anchor point coordinates in absolute pixels.
            name: Class label for the resulting annotation.
            positive_points: Extra positive anchors as ``[(x, y), ...]``.
            negative_points: Negative-prompt anchors (excluded regions).
            score_threshold: Minimum SAM3 score to include (0-1).

        Returns:
            :class:`PromptResult` - ``status="success"`` carries one polygon
            annotation in ``annotations[0]``.

        Raises:
            NotFoundError: Dataset or image missing.
            PaymentRequiredError: Insufficient credits (3-credit minimum).
        """
        body: dict[str, Any] = {
            "dataset": dataset,
            "image": image,
            "x": x,
            "y": y,
            "name": name,
            "score_threshold": score_threshold,
        }
        if positive_points is not None:
            body["positive_points"] = [list(p) for p in positive_points]
        if negative_points is not None:
            body["negative_points"] = [list(p) for p in negative_points]
        response = self._transport.request("POST", f"{_API_PATH}/sam3/point", json=body)
        # Backend wraps the single annotation under "annotation" rather than
        # "annotations" - normalise into a list before parsing.
        ann = response.get("annotation")
        normalised = {
            "status": response.get("status", "success"),
            "annotations": [ann] if ann else [],
            "score": response.get("score"),
            "inference_time": response.get("inference_time"),
        }
        return self._parse(PromptResult, normalised)

    def box(
        self,
        dataset: str,
        image: str,
        *,
        box: dict[str, float],
        name: str,
        confidence_threshold: float = 0.5,
        return_polygon: bool = True,
        negative_boxes: Sequence[dict[str, float]] | None = None,
    ) -> PromptResult:
        """SAM3 box prompt → bbox + optional polygon annotation(s).

        Args:
            dataset: Dataset name.
            image: The image: its filename (the dataset root first), its
                path inside the dataset (``"val/x.jpg"``), or its id.
            box: ``{"x": float, "y": float, "w": float, "h": float}`` in
                absolute pixels.
            name: Class label.
            confidence_threshold: Minimum SAM3 confidence to include.
            return_polygon: When ``True`` (default), include a polygon
                annotation in addition to the refined bbox. Set ``False``
                if only the bbox is wanted.
            negative_boxes: Boxes to exclude from segmentation (e.g.
                shift-drag exclusion zones).

        Returns:
            :class:`PromptResult` - ``annotations`` may be empty if status
            is ``no_detection`` or ``below_threshold``.
        """
        body: dict[str, Any] = {
            "dataset": dataset,
            "image": image,
            "box": box,
            "name": name,
            "confidence_threshold": confidence_threshold,
            "return_polygon": return_polygon,
        }
        if negative_boxes is not None:
            body["negative_boxes"] = [dict(b) for b in negative_boxes]
        response = self._transport.request("POST", f"{_API_PATH}/sam3/box", json=body)
        return self._parse(
            PromptResult,
            {
                "status": response.get("status", "success"),
                "annotations": response.get("annotations", []),
                "score": response.get("score"),
                "inference_time": response.get("inference_time"),
            },
        )

    def text(
        self,
        dataset: str,
        image: str,
        *,
        text_prompt: str,
        output_type: str = "polygon",
        confidence_threshold: float = 0.3,
        max_detections: int = 50,
    ) -> PromptResult:
        """SAM3 text prompt → list of detected annotations.

        Searches the image for instances matching ``text_prompt`` (e.g.
        ``"person"``, ``"red car"``). Returns up to ``max_detections``
        bounding boxes or polygons, sorted by confidence.

        Args:
            dataset, image: As with :meth:`point` / :meth:`box`.
            text_prompt: Natural language description.
            output_type: ``"polygon"`` (default) or ``"bbox"``.
            confidence_threshold: Minimum confidence (0-1).
            max_detections: Cap on result count (1-100).
        """
        body: dict[str, Any] = {
            "dataset": dataset,
            "image": image,
            "text_prompt": text_prompt,
            "output_type": output_type,
            "confidence_threshold": confidence_threshold,
            "max_detections": max_detections,
        }
        response = self._transport.request("POST", f"{_API_PATH}/sam3/text", json=body)
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

    def batch(
        self,
        dataset: str,
        images: Sequence[str],
        *,
        classes: Sequence[BatchClass | dict[str, Any]],
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

        SAM3 path (``model=None`` or ``"sam3"``) requires at least one
        class. Trained-model path (``model=<name>``) supports
        classification models with empty classes (predicted classes land
        as tags).

        Args:
            dataset: Dataset name.
            images: The images to process (1-5000): each an id or a path
                inside the dataset (``"val/x.jpg"``) - exactly that image - or a
                bare filename, which names every image of that name.
            classes: Class configs - :class:`BatchClass` instances or raw
                dicts ``{"name": str, "output_type": "polygon"|"bbox"|"tag"}``.
            confidence_threshold: Minimum confidence to include.
            model: Trained model, by name or id, or ``None``
                (default) for SAM3.
            top_k: Classifier-only - how many predicted tags per image.
            sahi: SAHI sliced inference (SAM3 only). Slices each image into
                overlapping tiles plus one full-image pass so small objects
                are detected at near-native resolution; tile fragments are
                merged server-side into whole instances. Costs more (one
                grounding pass per tile) and is rejected with a 400 for
                trained-model jobs.
            sahi_slice_size: SAHI tile edge in source pixels (256-1024,
                default 640). Smaller tiles see smaller objects but produce
                more passes.
            wait: When ``True`` (default), poll until terminal status.
            poll_interval: Seconds between checks (default 5s).
            timeout: Max seconds to wait (default 1800 = 30 min).

        Returns:
            :class:`BatchJob`. If ``wait=True``, returns the terminal-state
            snapshot; if ``wait=False``, returns the kick-off snapshot.

        Raises:
            NotFoundError: Dataset or no matching images.
            PaymentRequiredError: Insufficient credits for the estimate.
            ValidationError: SAM3 path with no classes, SAHI with a trained
                model, or an out-of-range ``sahi_slice_size``.
            ApiError: Job ended in ``failed`` status (when ``wait=True``).
            PollTimeoutError: Deadline elapsed (when ``wait=True``).
        """
        body: dict[str, Any] = {
            "dataset": dataset,
            "images": list(images),
            "classes": [
                c.model_dump(exclude_none=True) if isinstance(c, BatchClass) else dict(c)
                for c in classes
            ],
            "confidence_threshold": confidence_threshold,
            "top_k": top_k,
        }
        if model is not None:
            # "sam3" is a sentinel, not a model name - it must not be resolved.
            body["model"] = model if model == "sam3" else _resolve.model_id(self._transport, model)
        if sahi:
            # Only sent when opted in - the request model is extra="forbid",
            # so omitting keeps requests compatible with pre-SAHI backends.
            body["sahi_enabled"] = True
            body["sahi_slice_size"] = sahi_slice_size
        response = self._transport.request("POST", f"{_API_PATH}/batch", json=body)
        job = self._parse(BatchJob, response)
        if not wait:
            return job
        return self.wait_for_batch(job.job_id, poll_interval=poll_interval, timeout=timeout)

    def quote(
        self,
        *,
        dataset: str | None = None,
        images: Sequence[str] = (),
        projected: Sequence[ProjectedImages | dict[str, Any]] = (),
        classes: Sequence[BatchClass | dict[str, Any]] = (),
        model: str | None = None,
        sahi: bool = False,
        sahi_slice_size: int = 640,
    ) -> BatchQuote:
        """Ask what a batch job would cost, WITHOUT running it.

        The quote and the charge come from one function on the server, so what you are
        told here is what :meth:`batch` will deduct. Previously the only way to learn a
        job's price was to start it - by which point the credits were already gone.

        ``projected`` prices images that DON'T EXIST YET. That is what lets you decide
        before you spend, most importantly for video: a video is one file but hundreds of
        frames, and the frames are what you pay for.

        Args:
            dataset: Required to price images that already exist.
            images: Existing images to price (an id, a path inside the
                dataset, or a bare filename - as :meth:`batch`). Their real stored
                dimensions drive SAHI tile pricing.
            projected: Groups of images that don't exist yet - an upload in flight, or a
                video's frames. Give each group its pixel size when you know it.
            classes: Class configs. Each extra class adds a grounding pass, so the class
                count changes the price.
            model: Trained model, by name or id, or ``None``
                (default) for SAM3.
            sahi: SAHI sliced inference (SAM3 only).
            sahi_slice_size: SAHI tile edge in source pixels (256-1024).

        Returns:
            :class:`BatchQuote` - the estimate, your remaining balance, whether it covers
            the job, and whether the job is over the per-job image ceiling.

        Example - price a video's frames before uploading it::

            meta = client.video.probe(client.video.upload("clip.mp4").video)
            frames = int(meta.duration_seconds * 5)  # sample at 5 fps
            quote = client.auto_annotate.quote(
                projected=[{"count": frames, "width": meta.width, "height": meta.height}],
                classes=[BatchClass(name="car", output_type="bbox")],
            )
            if quote.sufficient:
                job = client.video.extract_frames(..., sample_fps=5)
                client.auto_annotate.batch(...)  # over job.image_ids
        """
        body: dict[str, Any] = {
            "images": list(images),
            "projected": [
                p.model_dump(exclude_none=True) if isinstance(p, ProjectedImages) else dict(p)
                for p in projected
            ],
            "classes": [
                c.model_dump(exclude_none=True) if isinstance(c, BatchClass) else dict(c)
                for c in classes
            ],
        }
        if dataset is not None:
            body["dataset"] = dataset
        if model is not None:
            # "sam3" is a sentinel, not a model name - it must not be resolved.
            body["model"] = model if model == "sam3" else _resolve.model_id(self._transport, model)
        if sahi:
            body["sahi_enabled"] = True
            body["sahi_slice_size"] = sahi_slice_size
        response = self._transport.request("POST", f"{_API_PATH}/batch/quote", json=body)
        return self._parse(BatchQuote, response)

    def get_batch(self, job: str) -> BatchJob:
        """Fetch the current status of a batch job."""
        response = self._transport.request("GET", f"{_API_PATH}/batch/{job}")
        return self._parse(BatchJob, response)

    def cancel_batch(self, job: str) -> BatchJob:
        """Cancel a pending or running batch job.

        Does NOT refund pre-charged credits - by the time an SDK caller can
        cancel, the GPU job has already started and the compute is spent.
        """
        response = self._transport.request("POST", f"{_API_PATH}/batch/{job}/cancel")
        return self._parse(BatchJob, response)

    def wait_for_batch(
        self,
        job: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_BATCH_TIMEOUT,
        sleep: Callable[[float], None] | None = None,
    ) -> BatchJob:
        """Poll a batch current until terminal status or timeout."""
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else time.sleep
        deadline = time.monotonic() + timeout
        while True:
            current = self.get_batch(job)
            if current.status == "completed":
                return current
            if current.status in ("failed", "cancelled"):
                raise ApiError(
                    f"Batch job {job} ended with status '{current.status}': "
                    f"{current.error_message or 'no error message provided'}",
                    response=current.model_dump(mode="json"),
                    code=job_end_code(current.status),
                )
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Batch job {job} did not complete within {timeout:.0f}s "
                    f"(last status: {current.status}). Fetch later via "
                    f"client.auto_annotate.get_batch(...)."
                )
            sleep_fn(poll_interval)

    # ───────────── "Annotate similar" ─────────────

    def similar(
        self,
        dataset: str,
        image: str,
        *,
        annotations: Sequence[str] = (),
        reference_class: str | None = None,
        scope: SimilarScope = "directory",
        target_class: str | None = None,
        output_type: SimilarOutputType = "polygon",
        confidence: float | None = None,
        wait: bool = True,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_BATCH_TIMEOUT,
    ) -> BatchJob:
        """Label every object like your example polygons, across the image's folder or
        the whole dataset - the app's "Annotate similar".

        Draw one or more polygons around an example object on one image; SAM3 finds every
        object like it in the OTHER images of that image's folder (``scope="directory"``)
        or of the dataset (``scope="dataset"``) and saves each match as an annotation of
        ``target_class``. The labels are drafts: have a person review them before you
        export the dataset and train on the export.

        Charged the job's measured GPU time once, when it completes; a failed or cancelled
        job is free. The start is gated on the estimate - price it first with
        :meth:`quote_similar`. Progress and cancel go through the batch job calls
        (:meth:`get_batch`, :meth:`cancel_batch`) and ``client.jobs``.

        Args:
            dataset: Dataset name.
            image: The image the examples are drawn on: its filename (the dataset
                root), its path inside the dataset (``"val/x.jpg"``), or its id.
            annotations: The example polygons, by annotation id
                (``client.annotations.get`` lists them).
            reference_class: Instead of ``annotations``: every polygon of this class
                on the image is an example.
            scope: ``"directory"`` (default) or ``"dataset"``.
            target_class: The class to label the matches as - one of the dataset's
                classes. Defaults to the examples' class (they must share one).
            output_type: ``"polygon"`` (default; needs a polygon class), ``"bbox"`` or
                ``"obb"`` (a rotated box; both need a box class).
            confidence: SAM3's detection confidence, 0.2-0.8. Raise it if unrelated
                objects slip in; lower it to catch more. Omit for the app's default.
            wait: When ``True`` (default), poll until the job finishes.
            poll_interval: Seconds between checks (default 5s).
            timeout: Max seconds to wait (default 1800 = 30 min).

        Returns:
            :class:`BatchJob` - the finished job when ``wait=True``, else the kick-off
            snapshot (its ``estimated_credits`` set).

        Raises:
            ValueError: Neither or both of ``annotations`` / ``reference_class``.
            NotFoundError: The dataset, the image, or an example id.
            ValidationError: An example that is not a polygon, examples of several
                classes with no ``target_class``, a class the dataset lacks, nothing
                else in scope, or an image SAM3 cannot read (``image_unreadable``).
            PaymentRequiredError: The balance does not cover the estimate.
            ApiError: The job ended ``failed`` or ``cancelled`` (when ``wait=True``).
            PollTimeoutError: Deadline elapsed (when ``wait=True``).
        """
        body = _similar_body(
            dataset,
            image,
            annotation_ids=annotations,
            reference_class=reference_class,
            scope=scope,
            target_class=target_class,
            output_type=output_type,
            confidence=confidence,
        )
        response = self._transport.request("POST", f"{_API_PATH}/similar", json=body)
        job = self._parse(BatchJob, response)
        if not wait:
            return job
        return self.wait_for_batch(job.job_id, poll_interval=poll_interval, timeout=timeout)

    def quote_similar(
        self,
        dataset: str,
        image: str,
        *,
        annotations: Sequence[str] = (),
        reference_class: str | None = None,
        scope: SimilarScope = "directory",
        target_class: str | None = None,
        output_type: SimilarOutputType = "polygon",
        confidence: float | None = None,
    ) -> SimilarQuote:
        """What :meth:`similar` would run and cost, WITHOUT running it.

        The same checks the start makes - the examples, the target class, whether SAM3
        reads the example image, the images in scope - and the estimate the start is
        gated on, with your balance. Writes nothing, charges nothing; any role.

        Args:
            dataset: Dataset name.
            image: The image the examples are drawn on: its filename (the dataset
                root), its path inside the dataset (``"val/x.jpg"``), or its id.
            annotations: The example polygons, by annotation id
                (``client.annotations.get`` lists them).
            reference_class: Instead of ``annotations``: every polygon of this class
                on the image is an example.
            scope: ``"directory"`` (default) or ``"dataset"``.
            target_class: The class to label the matches as - one of the dataset's
                classes. Defaults to the examples' class (they must share one).
            output_type: ``"polygon"`` (default), ``"bbox"`` or ``"obb"``.
            confidence: SAM3's detection confidence, 0.2-0.8. Omit for the app's default.

        Returns:
            :class:`SimilarQuote` - the images it would label, the estimate (µUSD), your
            balance and whether it covers the estimate.

        Raises:
            ValueError: Neither or both of ``annotations`` / ``reference_class``.
            NotFoundError: The dataset, the image, or an example id.
            ValidationError: Anything :meth:`similar` would refuse before it starts.
        """
        body = _similar_body(
            dataset,
            image,
            annotation_ids=annotations,
            reference_class=reference_class,
            scope=scope,
            target_class=target_class,
            output_type=output_type,
            confidence=confidence,
        )
        response = self._transport.request("POST", f"{_API_PATH}/similar/quote", json=body)
        return self._parse(SimilarQuote, response)

    # ───────────── whole-dataset ─────────────

    def dataset(
        self,
        dataset: str,
        *,
        classes: Sequence[BatchClass | tuple[str, str] | dict[str, str]],
        mode: AnnotateMode = "batch",
        directory: str | None = None,
        model: str | None = None,
        confidence_threshold: float = 0.5,
        overwrite: bool = False,
        max_images: int | None = None,
        sahi: bool = False,
        sahi_slice_size: int = 640,
        wait: bool = True,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_BATCH_TIMEOUT,
    ) -> AnnotateReport:
        """Draft annotations for a whole dataset (or one folder of it) and save them.

        The dataset-wide form of :meth:`batch` / :meth:`text`: it reads every image
        of the dataset (or of ``directory``), holds back the ones already
        annotated (unless ``overwrite``), submits the rest, and saves what comes
        back. What it saves are DRAFTS: have a person review them (start with
        ``client.images.iter(dataset, min_confidence_lt=0.5)``, the ones the model
        was least sure of), then export the dataset and train on the export.

        Args:
            dataset: Dataset name within your org.
            classes: Class configs to detect. Accepts:
                - :class:`BatchClass` instances (canonical),
                - ``(name, output_type)`` tuples (shorthand),
                - ``{"name": ..., "output_type": ...}`` dicts.
                ``output_type`` defaults to ``"polygon"`` for tuples without one.
                With ``model``, the classes are optional (a classifier's predicted
                classes land as tags).
            mode: ``"batch"`` (default) - async batch job; ``"text"`` -
                synchronous per-image text prompt (SAM3 only).
            directory: Only the images in this folder (``"/val"``; ``"/"`` is
                the root) - exactly that folder, not its subfolders. ``None``
                (default) reads the whole dataset.
            model: A trained model by name (a UUID works) instead of SAM3.
                ``"batch"`` mode only.
            confidence_threshold: Confidence cutoff (0-1).
            overwrite: When ``False`` (default), skip images that already
                have at least one annotation. When ``True``, draft every image: in
                ``"batch"`` mode the new drafts are added beside the annotations an
                image has (one that duplicates an existing annotation of its class
                is dropped); ``"text"`` mode replaces them.
            max_images: Cap the number of images processed (useful for
                dry-runs). ``None`` means "all".
            sahi: SAHI sliced inference for small objects (SAM3, ``"batch"`` mode).
            sahi_slice_size: SAHI tile edge in pixels (256-1024).
            wait: ``"batch"`` mode only. ``True`` (default) waits for the job and
                reports what it saved; ``False`` submits it and returns at once,
                with ``job_id`` set (follow it with :meth:`get_batch` /
                :meth:`wait_for_batch`; do not run this again on the same images
                until it finishes, or they are drafted twice).
            poll_interval: ``"batch"`` mode only - seconds between polls.
            timeout: ``"batch"`` mode only - max seconds to wait.

        Returns:
            :class:`AnnotateReport`. Inspect ``failures`` for per-image errors; a
            failure of the batch job keeps its error's class and code.

        Raises:
            NotFoundError: ``dataset`` (or ``directory``) doesn't exist.
            ValueError: ``model`` / ``sahi`` with ``mode="text"``, or no classes for
                SAM3.

        Example:
            >>> client.auto_annotate.dataset("road-signs", [("stop_sign", "bbox")])
            AnnotateReport(dataset='road-signs', ...)
        """
        from pictograph.resources.annotations import Annotations
        from pictograph.resources.images import Images

        if mode == "text" and (model is not None or sahi):
            raise ValueError("model and sahi apply to mode='batch' only.")

        canonical_classes = _normalize_classes(classes, model)

        # Every image of the dataset (or of the one folder), read page by page to
        # the end: a dataset of any size is enumerated in full.
        all_images = list(
            Images(self._transport).iter(dataset, directory=directory, page_size=_LIST_PAGE_SIZE)
        )
        images, report = _plan_dataset_run(
            all_images, dataset=dataset, mode=mode, overwrite=overwrite, max_images=max_images
        )
        if not images:
            return report

        if mode == "batch":
            try:
                # Submit, then wait: the report names the job from the moment it
                # exists, so a wait that runs out or a job that ends failed still
                # says which job to follow or to look at.
                job = self.batch(
                    dataset=dataset,
                    # By ID: a filename is unique per FOLDER, so a bare name could
                    # also pick up an annotated same-named image elsewhere.
                    images=[img.id for img in images],
                    classes=canonical_classes,
                    confidence_threshold=confidence_threshold,
                    model=model,
                    sahi=sahi,
                    sahi_slice_size=sahi_slice_size,
                    wait=False,
                )
                report.job_id = job.job_id
                report.job_status = job.status
                if wait:
                    job = self.wait_for_batch(
                        job.job_id, poll_interval=poll_interval, timeout=timeout
                    )
                    report.job_status = job.status
                report.images_processed = job.processed_images
                report.annotations_added = job.total_annotations_added
                if job.failed_images:
                    report.failures.append(
                        AnnotationFailure(
                            image_filename="<batch>",
                            reason=(f"{job.failed_images} images failed during batch processing"),
                        )
                    )
            except PictographError as e:
                # Any SDK-domain error is a documented-as-report failure, NOT an
                # uncaught raise: dataset() "Returns AnnotateReport; inspect
                # failures", so a caller staging upload → annotate → train keeps the
                # report instead of losing it to a raise. This covers a batch
                # PollTimeoutError (str(e) carries the job id + how to fetch it
                # later), an ApiError, AND a transient NetworkError /
                # RequestTimeoutError mid-poll. The failure keeps the error's class
                # and code. Real Python bugs (not PictographError) still propagate.
                report.failures.append(AnnotationFailure.from_error("<batch>", e))
                ended = e.response if isinstance(e, ApiError) else None
                if isinstance(ended, dict) and isinstance(ended.get("status"), str):
                    report.job_status = ended["status"]  # the job ended failed/cancelled
            return report

        # mode == "text" - synchronous per-image SAM3 text prompt with each class.
        # IMPORTANT: collect EVERY class's annotations and write them in ONE save per
        # image. ``annotations.save`` is a *full replacement* of the image's
        # annotations, so saving once per class would overwrite each prior class's
        # result and leave only the LAST class's annotations on the image (silent
        # data loss for any multi-class run). This mirrors batch mode, which sends
        # all classes in a single job.
        annotations_resource = Annotations(self._transport)
        for img in images:
            collected: list[Annotation] = []
            per_image_failures: list[str] = []
            for cls in canonical_classes:
                try:
                    result = self.text(
                        dataset=dataset,
                        # By ID, the image the save below writes to (a bare name could
                        # resolve to a same-named image in another folder).
                        image=img.id,
                        text_prompt=cls.name,
                        output_type=(cls.output_type if cls.output_type != "tag" else "polygon"),
                        confidence_threshold=confidence_threshold,
                    )
                    if result.status == "success" and result.annotations:
                        collected.extend(result.annotations)
                except PictographError as e:
                    # Per-class SDK-domain failure (ApiError, a transient
                    # NetworkError / RequestTimeoutError from the text prompt, etc.)
                    # is recorded against this image and the loop continues - one bad
                    # class never aborts the dataset run.
                    per_image_failures.append(f"{cls.name}: {e}")

            # One full-replacement save per image with every class's annotations.
            if collected:
                try:
                    save_result = annotations_resource.save(dataset, img.id, annotations=collected)
                    # ``annotations_added`` = annotations this run WROTE (matches batch
                    # mode's server-computed ``total_annotations_added``, which is ge=0).
                    # ``save`` is a full replacement, so ``new_count`` IS what this run
                    # produced; a ``new_count - previous_count`` net-delta would
                    # under-count - and go NEGATIVE - when overwrite=True re-annotated
                    # an image that already had annotations (previous_count > 0).
                    report.annotations_added += save_result.new_count
                except PictographError as e:
                    per_image_failures.append(f"save: {e}")

            if per_image_failures:
                report.failures.append(
                    AnnotationFailure(
                        image_filename=img.filename,
                        reason="; ".join(per_image_failures),
                    )
                )
            else:
                report.images_processed += 1

        return report


def _normalize_classes(
    classes: Sequence[BatchClass | tuple[str, str] | dict[str, str]], model: str | None
) -> list[BatchClass]:
    """``dataset()``'s class specs as :class:`BatchClass` instances (shared by both twins)."""
    # ``BatchClass.output_type`` is a strict Literal - Pydantic re-validates whichever
    # string we pass through.
    canonical_classes: list[BatchClass] = []
    for c in classes:
        if isinstance(c, BatchClass):
            canonical_classes.append(c)
        elif isinstance(c, tuple):
            # Accept ``(name,)`` (→ default polygon) or ``(name, output_type)``. Any other
            # arity is a malformed spec: a 3-tuple would silently drop the caller's
            # output_type and a 0-tuple would crash on ``c[0]``. (``c`` is statically a
            # 2-tuple; widen it so the runtime arity guard is not pruned as unreachable.)
            spec = cast("tuple[Any, ...]", c)
            if len(spec) == 2:
                name, output_type = spec[0], spec[1]
            elif len(spec) == 1:
                name, output_type = spec[0], "polygon"
            else:
                raise ValueError(f"Unsupported class spec: {c!r}")
            canonical_classes.append(BatchClass(name=name, output_type=cast("Any", output_type)))
        elif isinstance(c, dict):
            canonical_classes.append(BatchClass(**cast("dict[str, Any]", c)))
        else:
            raise ValueError(f"Unsupported class spec: {c!r}")
    if not canonical_classes and model is None:
        raise ValueError("Give at least one class for SAM3 to find (or a model).")
    return canonical_classes


def _plan_dataset_run(
    all_images: list[Image],
    *,
    dataset: str,
    mode: AnnotateMode,
    overwrite: bool,
    max_images: int | None,
) -> tuple[list[Image], AnnotateReport]:
    """Which images ``dataset()`` submits, and its report so far (shared by both twins)."""
    eligible = all_images
    if not overwrite:
        eligible = [img for img in eligible if img.annotation_count == 0]
    images = eligible[:max_images] if max_images is not None else eligible

    # The SAM3 batch endpoint accepts at most _BATCH_IMAGE_CAP images per call. Cap
    # EXPLICITLY (never silently): a larger eligible set would 422 the whole batch, so
    # submit the first _BATCH_IMAGE_CAP and report the remainder as a failure (so
    # success=False) telling the caller to run again or pass max_images. ``text`` mode
    # is per-image and uncapped.
    batch_overflow = 0
    if mode == "batch" and len(images) > _BATCH_IMAGE_CAP:
        batch_overflow = len(images) - _BATCH_IMAGE_CAP
        images = images[:_BATCH_IMAGE_CAP]

    report = AnnotateReport(
        dataset_name=dataset,
        images_attempted=len(images),
        # Only images dropped for being already annotated - NOT the cap remainder.
        images_skipped=len(all_images) - len(eligible),
        # Eligible images held back this run - by max_images AND/OR the per-batch cap.
        images_capped=len(eligible) - len(images),
    )
    if batch_overflow:
        report.failures.append(
            AnnotationFailure(
                image_filename="<batch-cap>",
                reason=(
                    f"{batch_overflow} eligible image(s) exceed the "
                    f"{_BATCH_IMAGE_CAP}-image per-batch limit and were not "
                    "submitted - run it again once this job has finished to "
                    "draft the remainder, or pass max_images to control scope."
                ),
            )
        )
    return images, report
