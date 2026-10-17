"""The five task-typed local model classes.

ONE class per task, not one per task per runtime. A :class:`DetectionModel` is a
detection model whether an ONNX session, an ExecuTorch program, a TensorRT plan or a
``torch`` module is doing the work - ``.backend`` says which, and the typed result is
identical either way::

    onnx = get_model("Shelf Detector", task="object_detection")  # 'onnxruntime'
    pte = load_model("xnnpack-fp32.pte", cfg, task="object_detection")  # 'executorch'
    torch = get_model("Shelf Detector", task="object_detection", format="safetensors")

    for m in (onnx, pte, torch):
        result: DetectionResult = m.predict("photo.jpg")  # same type, same fields

That is what makes the swappable-runtime promise real rather than documented: every
engine feeds the SAME result builder from the SAME shared emitters - the three graph
runtimes literally share one preprocessing/postprocessing implementation and differ
only in the forward pass (see :mod:`pictograph.inference._engine`) - so a difference
between them is a bug with a test that catches it, not an expected variation.

Every model is a context manager and should be closed when you are done with it -
an ONNX session and a CUDA/MPS allocator both hold memory the garbage collector
will not promptly return.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, cast, runtime_checkable

from pictograph.inference.results import (
    AnyResult,
    ClassificationResult,
    DetectionResult,
    InferenceResult,
    InstanceSegmentationResult,
    KeypointResult,
    SemanticSegmentationResult,
    TaskName,
    build_result,
)
from pictograph.inference.runtime import empty_device_cache

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "TASK_MODEL_TYPES",
    "AnyModel",
    "ClassificationModel",
    "DetectionModel",
    "InferenceModel",
    "InstanceSegmentationModel",
    "KeypointModel",
    "SemanticSegmentationModel",
]

# A local path, an http(s) URL, raw image bytes, a decoded BGR numpy array, or a PIL
# image. Kept as `Any` in signatures for a clean, un-noisy API.
ImageInput = Any

R_co = TypeVar("R_co", bound=InferenceResult, covariant=True)


@runtime_checkable
class InferenceModel(Protocol[R_co]):
    """The common, swappable interface of every local model.

    Parameterized by the result type, so a function can accept any model and still
    know what it gets back::

        def annotate(model: InferenceModel[DetectionResult], path: str) -> None:
            for p in model.predict(path).predictions:
                print(p.bounding_box)

    ``predict_batch`` returns a ``Sequence``, not a ``list``, deliberately: a
    covariant type variable in an invariant container would stop the concrete
    classes from satisfying the protocol at all.
    """

    model_type: TaskName
    backend: str
    device: str
    providers: list[str]
    classes: list[str]

    def predict(self, image: ImageInput, *, confidence: float | None = ...) -> R_co: ...

    def predict_batch(
        self, images: list[ImageInput], *, confidence: float | None = ...
    ) -> Sequence[R_co]: ...

    def close(self) -> None: ...


class _TaskModel(Generic[R_co]):
    """Shared machinery: hold an engine, time the call, build the typed result."""

    model_type: TaskName

    def __init__(
        self,
        *,
        engine: Any,
        model_id: str,
        name: str,
        architecture: str,
        confidence: float,
    ) -> None:
        self._engine = engine
        self.id = model_id
        self.name = name
        self.architecture = architecture
        self._confidence = confidence
        self._lock = threading.Lock()

    # ── provenance, read off the engine so it reflects what actually loaded ──

    @property
    def backend(self) -> str:
        """The runtime that ran it - one of
        :data:`~pictograph.inference.runtime.RUNTIMES`:
        ``'pytorch'`` / ``'executorch'`` / ``'onnxruntime'`` / ``'tensorrt'``."""
        return str(self._engine.backend)

    @property
    def device(self) -> str:
        """The device that RAN it - ``cpu`` / ``cuda`` / ``mps`` / ``coreml``."""
        return str(self._engine.device)

    @property
    def providers(self) -> list[str]:
        """The runtime's own execution targets, as it resolved them.

        ORT providers on ``onnxruntime``, delegate backends on ``executorch``, the
        plan's TensorRT version + SM on ``tensorrt``, empty on ``pytorch``.
        """
        return list(self._engine.providers)

    @property
    def classes(self) -> list[str]:
        """Class names in the model's label order."""
        return list(self._engine.classes)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, backend={self.backend!r}, "
            f"device={self.device!r}, classes={len(self.classes)})"
        )

    # ── inference ──

    def _run(self, image: ImageInput, confidence: float | None, top_k: int = 1) -> R_co:
        conf = self._confidence if confidence is None else confidence
        started = time.perf_counter()
        raw = self._engine.infer(_decode_image(image), confidence=conf, top_k=top_k)
        elapsed = (time.perf_counter() - started) * 1000.0
        return self._build(raw, elapsed)

    def _run_batch(
        self, images: list[ImageInput], confidence: float | None, top_k: int = 1
    ) -> list[R_co]:
        conf = self._confidence if confidence is None else confidence
        decoded = [_decode_image(i) for i in images]
        started = time.perf_counter()
        raws = self._engine.infer_batch(decoded, confidence=conf, top_k=top_k)
        each = (time.perf_counter() - started) * 1000.0 / max(1, len(raws))
        return [self._build(r, each) for r in raws]

    def _build(self, raw: dict[str, Any], elapsed_ms: float) -> R_co:
        raise NotImplementedError

    def _meta(self, elapsed_ms: float) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "device": self.device,
            "providers": self.providers,
            "inference_ms": round(elapsed_ms, 3),
        }

    def close(self) -> None:
        """Release the underlying session / module and its device memory."""
        engine = self._engine
        if engine is not None:
            engine.close()
        empty_device_cache(getattr(engine, "device", "cpu"))

    def __enter__(self: _SelfT) -> _SelfT:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


_SelfT = TypeVar("_SelfT", bound=_TaskModel[Any])


def _build_typed(model: _TaskModel[Any], raw: dict[str, Any], elapsed_ms: float) -> AnyResult:
    """Local run → typed result, through the ONE shared builder.

    The remote path (:class:`pictograph.DeploymentClient`) calls the same
    function on the same payload shape, so a change to how predictions become a
    result lands on Edge and Remote together or on neither.

    ``source`` names this model in whatever `build_result` has to say about it.
    It used to read ``Classifier {name}`` for EVERY task, which went unnoticed
    only because the one message that consumed it was the classifier's - so a
    detector calling it "Classifier" was never printed. It is task-neutral now.
    """
    return build_result(
        raw,
        task=model.model_type,
        source=f"Model {model.name!r}",
        **model._meta(elapsed_ms),
    )


class DetectionModel(_TaskModel[DetectionResult]):
    """Object detection - returns :class:`~pictograph.DetectionResult`.

    Covers both the YOLOX and RF-DETR detection pipelines.
    """

    model_type: TaskName = "object_detection"

    def predict(self, image: ImageInput, *, confidence: float | None = None) -> DetectionResult:
        """Run the model on one image.

        Args:
            image: A file path, an http(s) URL, raw image bytes, a decoded **BGR**
                numpy array, or a PIL image.
            confidence: Minimum score to keep (0-1). Defaults to the model's own.
        """
        return self._run(image, confidence)

    def predict_batch(
        self, images: list[ImageInput], *, confidence: float | None = None
    ) -> list[DetectionResult]:
        """Run the model on several images. One result per image, in order."""
        return self._run_batch(images, confidence)

    def _build(self, raw: dict[str, Any], elapsed_ms: float) -> DetectionResult:
        return cast("DetectionResult", _build_typed(self, raw, elapsed_ms))


class InstanceSegmentationModel(_TaskModel[InstanceSegmentationResult]):
    """Instance segmentation - returns :class:`~pictograph.InstanceSegmentationResult`."""

    model_type: TaskName = "instance_segmentation"

    def predict(
        self, image: ImageInput, *, confidence: float | None = None
    ) -> InstanceSegmentationResult:
        """Run the model on one image."""
        return self._run(image, confidence)

    def predict_batch(
        self, images: list[ImageInput], *, confidence: float | None = None
    ) -> list[InstanceSegmentationResult]:
        """Run the model on several images. One result per image, in order."""
        return self._run_batch(images, confidence)

    def _build(self, raw: dict[str, Any], elapsed_ms: float) -> InstanceSegmentationResult:
        return cast("InstanceSegmentationResult", _build_typed(self, raw, elapsed_ms))


class SemanticSegmentationModel(_TaskModel[SemanticSegmentationResult]):
    """Semantic segmentation - returns :class:`~pictograph.SemanticSegmentationResult`."""

    model_type: TaskName = "semantic_segmentation"

    def predict(
        self, image: ImageInput, *, confidence: float | None = None
    ) -> SemanticSegmentationResult:
        """Run the model on one image."""
        return self._run(image, confidence)

    def predict_batch(
        self, images: list[ImageInput], *, confidence: float | None = None
    ) -> list[SemanticSegmentationResult]:
        """Run the model on several images. One result per image, in order."""
        return self._run_batch(images, confidence)

    def _build(self, raw: dict[str, Any], elapsed_ms: float) -> SemanticSegmentationResult:
        return cast("SemanticSegmentationResult", _build_typed(self, raw, elapsed_ms))


class KeypointModel(_TaskModel[KeypointResult]):
    """Keypoint detection - returns :class:`~pictograph.KeypointResult`."""

    model_type: TaskName = "keypoint_detection"

    def predict(self, image: ImageInput, *, confidence: float | None = None) -> KeypointResult:
        """Run the model on one image."""
        return self._run(image, confidence)

    def predict_batch(
        self, images: list[ImageInput], *, confidence: float | None = None
    ) -> list[KeypointResult]:
        """Run the model on several images. One result per image, in order."""
        return self._run_batch(images, confidence)

    def _build(self, raw: dict[str, Any], elapsed_ms: float) -> KeypointResult:
        return cast("KeypointResult", _build_typed(self, raw, elapsed_ms))


class ClassificationModel(_TaskModel[ClassificationResult]):
    """Whole-image classification - returns :class:`~pictograph.ClassificationResult`."""

    model_type: TaskName = "classification"

    def predict(
        self,
        image: ImageInput,
        *,
        confidence: float | None = None,
        top_k: int = 1,
    ) -> ClassificationResult:
        """Run the model on one image.

        Args:
            image: A file path, URL, bytes, BGR numpy array, or PIL image.
            confidence: Minimum score to keep (0-1).
            top_k: How many ranked classes to return. The same default on both
                backends - the torch path used to hardcode 5 while ONNX returned 1.
        """
        return self._run(image, confidence, top_k)

    def predict_batch(
        self,
        images: list[ImageInput],
        *,
        confidence: float | None = None,
        top_k: int = 1,
    ) -> list[ClassificationResult]:
        """Run the model on several images. One result per image, in order."""
        return self._run_batch(images, confidence, top_k)

    def _build(self, raw: dict[str, Any], elapsed_ms: float) -> ClassificationResult:
        # A payload that ranked nothing raises inside the shared builder: the
        # emitter always reports rank 1 whatever the threshold, so `classes` is
        # non-empty and `top` can stay non-optional.
        return cast("ClassificationResult", _build_typed(self, raw, elapsed_ms))


AnyModel = (
    DetectionModel
    | InstanceSegmentationModel
    | SemanticSegmentationModel
    | KeypointModel
    | ClassificationModel
)
"""Any local model. What a loader returns when the task is not given as a literal."""

TASK_MODEL_TYPES: dict[TaskName, type[Any]] = {
    "object_detection": DetectionModel,
    "instance_segmentation": InstanceSegmentationModel,
    "semantic_segmentation": SemanticSegmentationModel,
    "keypoint_detection": KeypointModel,
    "classification": ClassificationModel,
}
"""Task → model class. The single mapping every loader and test iterates."""


def _decode_image(image: ImageInput) -> Any:
    """Decode any supported input into a **BGR** numpy array.

    BGR is the SDK's one convention for a raw array, on BOTH engines: the ONNX
    wrappers are cv2-based and expect it, so ``cv2.imread(path)`` means the same
    thing whichever backend you hold. (The torch engine used to read the same array
    as RGB, so the two silently disagreed.)
    """
    import numpy as np

    if isinstance(image, np.ndarray):
        return image

    from pathlib import Path

    import cv2

    data: bytes
    if isinstance(image, (bytes, bytearray)):
        data = bytes(image)
    elif isinstance(image, (str, Path)):
        ref = str(image)
        if ref.startswith(("http://", "https://")):
            import httpx

            data = httpx.get(ref, timeout=30.0, follow_redirects=True).content
        else:
            img = cv2.imread(ref)
            if img is None:
                raise FileNotFoundError(f"Could not read image at {ref!r}")
            return img
    elif hasattr(image, "save"):  # PIL.Image
        arr = np.array(image.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        raise TypeError(
            "Unsupported image input; pass a path, URL, bytes, numpy array, or PIL image."
        )

    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Could not decode the provided image bytes.")
    return decoded
