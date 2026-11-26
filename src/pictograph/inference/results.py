"""Typed, per-task results from a local model run.

Every local runtime - ONNX Runtime, ExecuTorch, TensorRT and native PyTorch, reached
through :func:`pictograph.get_model` / :func:`pictograph.load_model` and their shared
``format=`` argument - returns one of the six task-specific subclasses below. The task determines the shape,
so the geometry a model actually predicts is on the type rather than in the docs::

    from pictograph import get_model, DetectionModel, DetectionResult

    model: DetectionModel = get_model("Woody Whirling Grouse", task="object_detection")
    result: DetectionResult = model.predict("photo.jpg")

    for p in result.predictions:  # list[BBoxAnnotation] - narrowed
        print(p.name, round(p.confidence, 2), p.bounding_box)

Why per-task subclasses rather than one result with optional fields: the old flat
``InferenceResult`` carried ``predictions`` / ``classes`` / ``tags`` together, so
every caller had to know which were populated, ``result.top`` was ``| None`` on a
classifier that always has a top class, and ``p.bounding_box`` was a union-attr
error because ``list[Annotation]`` includes ``KeypointAnnotation`` (which has no
box). Each subclass below declares ``predictions`` afresh with exactly the
annotation types its task can emit.

The base deliberately does **not** declare ``predictions``: pydantic (and mypy)
treat a list field as invariant, so a subclass cannot narrow an inherited
``list[Annotation]`` to ``list[BBoxAnnotation]``.

This module is pure pydantic and imports no optional dependency, so it can be
imported (and type-checked, and covered) without the ``[inference]`` extra.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, Field

from pictograph.models.annotation import (
    BBoxAnnotation,
    KeypointAnnotation,
    PolygonAnnotation,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

__all__ = [
    "TASK_RESULT_TYPES",
    "AnomalyMap",
    "AnomalyResult",
    "AnyResult",
    "BackendName",
    "ClassScore",
    "ClassificationResult",
    "DetectionResult",
    "InferenceResult",
    "InstanceSegmentationResult",
    "KeypointResult",
    "SemanticSegmentationResult",
    "TaskName",
    "build_result",
]

_LOG = logging.getLogger("pictograph.inference")

TaskName = Literal[
    "object_detection",
    "instance_segmentation",
    "semantic_segmentation",
    "keypoint_detection",
    "classification",
    "anomaly_detection",
]
"""The six model tasks the platform trains. One result type + one model type each.

``object_detection`` covers BOTH the YOLOX and RF-DETR detection pipelines - they
differ in architecture, not in what they emit, and the loader dispatches on
architecture internally.
"""


class ClassScore(BaseModel):
    """One ranked class from a classification model."""

    name: str
    confidence: float = Field(ge=0.0, le=1.0)


BackendName = Literal["pytorch", "executorch", "onnxruntime", "tensorrt"]
"""The runtime that executed a model. Named separately so callers that construct a
result (rather than receive one) can spell the field's type."""


class InferenceResult(BaseModel):
    """Fields shared by every task result - provenance for the run that produced it.

    Never returned directly; :meth:`predict` always returns one of the six
    subclasses. Use it as the type when you accept any result::

        def log(result: InferenceResult) -> None:
            print(result.model_type, result.backend, result.device, result.inference_ms)
    """

    model_type: TaskName
    backend: BackendName = "onnxruntime"
    """Which runtime ran it. All four are interchangeable and produce equivalent output.

    The four values are :data:`pictograph.inference.runtime.RUNTIMES`, and they name
    the RUNTIME rather than the file extension - that is what a caller selects, what
    the model card's Runtime row renders, and what stays meaningful if a runtime ever
    gains a second container format.
    """

    device: str = "cpu"
    """The resolved device - ``cpu`` / ``cuda`` / ``cuda:1`` / ``mps`` / ``coreml``.

    This is what actually ran, not what was requested: a CUDA session that silently
    fell back to CPU reports ``cpu``.
    """

    providers: list[str] = Field(default_factory=list)
    """The runtime's own execution targets, as IT resolved them.

    Per runtime: ONNX Runtime execution providers in resolution order
    (``onnxruntime``); the ``.pte``'s delegate backends (``executorch``); the plan's
    TensorRT version and SM target (``tensorrt``); empty on ``pytorch``, which has no
    such concept - :attr:`device` is the whole story there.
    """

    inference_ms: float | None = None
    """Wall-clock milliseconds for the forward pass + postprocess, excluding image decode."""


class DetectionResult(InferenceResult):
    """Object detection - axis-aligned boxes.

    Produced by the YOLOX and RF-DETR detection pipelines.
    """

    model_type: Literal["object_detection"] = "object_detection"
    predictions: list[BBoxAnnotation] = Field(default_factory=list)
    """Detected objects. Every entry carries ``bounding_box`` and ``confidence``."""


class InstanceSegmentationResult(InferenceResult):
    """Instance segmentation - one mask per detected object.

    ``predictions`` is a union because the RF-DETR segmentation head legitimately
    emits either: a class whose mask polygonizes yields a
    :class:`~pictograph.PolygonAnnotation`, and one whose mask is empty or below the
    area floor degrades to its :class:`~pictograph.BBoxAnnotation`. Both carry a
    ``bounding_box``, so box-only consumers can ignore the distinction.
    """

    model_type: Literal["instance_segmentation"] = "instance_segmentation"
    predictions: list[PolygonAnnotation | BBoxAnnotation] = Field(default_factory=list)

    @property
    def polygons(self) -> list[PolygonAnnotation]:
        """Only the instances that produced a real mask."""
        return [p for p in self.predictions if isinstance(p, PolygonAnnotation)]


class SemanticSegmentationResult(InferenceResult):
    """Semantic segmentation - per-class regions, not per-object instances.

    Each prediction is one connected region of a class. A class covering several
    disjoint areas yields several polygons; there is no object identity.
    """

    model_type: Literal["semantic_segmentation"] = "semantic_segmentation"
    predictions: list[PolygonAnnotation] = Field(default_factory=list)


class KeypointResult(InferenceResult):
    """Keypoint detection - points, optionally grouped into multi-joint objects.

      Pose estimation is only ONE case of this task. The other, equally common case is
      "keypoint-as-class": every class is a single point of arity 1 and there is no
      grouping anywhere in the model. Hence ``Keypoint``, not ``Pose`` - the name
      describes the task (``keypoint_detection``), not its multi-joint minority.

    **Every prediction is one point of one class.** ``predictions`` is a flat
      ``list[KeypointAnnotation]`` - no union, no per-class shape switch - because a
      joint is a CLASS and :attr:`~pictograph.KeypointAnnotation.instance_id` is the
      OBJECT. A 17-joint pose is 17 predictions sharing an ``instance_id``, and a
      3-person image is 51 predictions carrying ids 1, 2, 3.

      That is why :attr:`points` and :attr:`instances` both exist, and it is the same
      reason the old ``skeletons`` / ``points`` split did: the raw list answers "what
      did the model find", not "what OBJECTS did it find", and reconstructing the
      objects by hand (bucket by ``instance_id``, remember that ``None`` means
      unassociated rather than "all one group") is exactly the loop a caller gets
      subtly wrong. Iterate the view you actually want::

          for point in result.points:  # every joint, flat
              print(point.name, round(point.keypoint.x), round(point.keypoint.y))

          for obj in result.instances:  # one list per detected OBJECT
              print("object with", len(obj), "joints")
              for joint in obj:  # joint.name IS the joint class
                  print(" ", joint.name, round(joint.keypoint.x), round(joint.keypoint.y))

      Connectivity is deliberately absent from the wire: an edge list was a per-class
      template identical for every instance. To DRAW a pose, connect an instance's
      points through the class template on the model / project config.
    """

    model_type: Literal["keypoint_detection"] = "keypoint_detection"
    predictions: list[KeypointAnnotation] = Field(default_factory=list)

    @property
    def points(self) -> list[KeypointAnnotation]:
        """Every predicted point, flat and in emission order.

        Identical to :attr:`predictions`; kept because it names the unit ("points",
        not "objects") at the call site and because it is what the arity-1
        keypoint-as-class reader has always used.
        """
        return list(self.predictions)

    @property
    def instances(self) -> list[list[KeypointAnnotation]]:
        """Predictions grouped into OBJECTS, by ``instance_id``.

        Ordered by ``instance_id`` ascending; each point whose ``instance_id`` is
        ``None`` is its own single-element group and sorts last, in emission order.
        ``None`` means *unassociated* - a lone landmark - so fusing those into one
        group would invent an object the model never predicted.
        """
        groups: dict[tuple[int, int], list[KeypointAnnotation]] = {}
        order: list[tuple[int, int]] = []
        singleton_seq = 0
        for prediction in self.predictions:
            if prediction.instance_id is None:
                singleton_seq += 1
                key = (1, singleton_seq)  # (1, …) sorts after every (0, …)
            else:
                key = (0, prediction.instance_id)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(prediction)
        return [groups[key] for key in sorted(order)]


class ClassificationResult(InferenceResult):
    """Whole-image classification - ranked classes, no geometry.

    :attr:`classes` is ordered by descending confidence and is never empty, so
    :attr:`top` is non-optional and ``result.top.name`` needs no guard.
    """

    model_type: Literal["classification"] = "classification"
    classes: list[ClassScore] = Field(min_length=1)
    """Ranked classes, highest confidence first. Length is the ``top_k`` asked for."""

    @property
    def top(self) -> ClassScore:
        """The highest-confidence class. Always present."""
        return self.classes[0]

    @property
    def tags(self) -> list[str]:
        """Just the class names, in rank order - the shape the auto-tagger consumes."""
        return [c.name for c in self.classes]


class AnomalyMap(BaseModel):
    """An anomaly model's map of the image, at the model's input size - returned only when asked
    (``predict(..., anomaly_map="png")`` locally, the same option on a deployment).

    ``encoding`` says what ``data`` (base64) holds: ``"png"`` - the NORMALIZED map as an 8-bit
    grayscale PNG (each pixel ``round(score x 255)``), for drawing a heatmap; ``"float16"`` - the
    RAW map as little-endian half floats, row-major, for your own scoring. :meth:`to_array` decodes
    either.
    """

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    encoding: Literal["png", "float16"]
    data: str

    def to_array(self) -> Any:
        """The map as a ``(height, width)`` float32 numpy array: the raw map (``float16``), or the
        normalized map in [0, 1] (``png``: each pixel / 255). Needs numpy, and Pillow for ``png``."""
        import base64
        import io

        import numpy as np

        raw = base64.b64decode(self.data)
        if self.encoding == "float16":
            return (
                np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(self.height, self.width)
            )
        from PIL import Image

        gray = np.asarray(Image.open(io.BytesIO(raw)).convert("L"))
        if gray.shape != (self.height, self.width):
            raise ValueError(
                f"the anomaly map's PNG is {gray.shape[1]}x{gray.shape[0]}, "
                f"not the declared {self.width}x{self.height}"
            )
        return gray.astype(np.float32) / np.float32(255.0)


class AnomalyResult(InferenceResult):
    """Anomaly detection - the image's label (normal or anomalous), its score, and where.

    An anomaly model (PatchCore, EfficientAD) learned what NORMAL looks like. ``score`` is its
    normalized anomaly score in [0, 1] (anomalib's normalization, with the model's own calibrated
    threshold mapped to 0.5); the image is anomalous when ``score`` is OVER ``confidence`` (the
    default 0.5 is that calibrated threshold, so raise it for fewer, surer alarms).

    - :attr:`label` - the model's anomaly class on an anomalous image, else its normal class (a
      top-1 label, as a classifier's), at :attr:`label_confidence` (the score, or its complement);
      ``None`` when ``class_filter`` left that class out.
    - :attr:`is_anomalous` - the decision, kept even when the label is filtered away.
    - ``predictions`` - the anomalous REGIONS, polygons named the anomaly class, each at its highest
      normalized score; none on a normal image.
    - :attr:`anomaly_map` - the map itself, only when asked.
    """

    model_type: Literal["anomaly_detection"] = "anomaly_detection"
    predictions: list[PolygonAnnotation] = Field(default_factory=list)
    """The anomalous regions. Empty on an image labelled normal."""
    label: str | None = None
    label_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=1.0)
    is_anomalous: bool
    anomaly_map: AnomalyMap | None = None

    @property
    def tags(self) -> list[str]:
        """The label as image tags (one, or none when filtered) - the shape the auto-tagger consumes."""
        return [self.label] if self.label is not None else []


TASK_RESULT_TYPES: dict[TaskName, type[InferenceResult]] = {
    "object_detection": DetectionResult,
    "instance_segmentation": InstanceSegmentationResult,
    "semantic_segmentation": SemanticSegmentationResult,
    "keypoint_detection": KeypointResult,
    "classification": ClassificationResult,
    "anomaly_detection": AnomalyResult,
}
"""Task → result class. The single mapping every loader and test iterates."""


AnyResult = (
    DetectionResult
    | InstanceSegmentationResult
    | SemanticSegmentationResult
    | KeypointResult
    | ClassificationResult
    | AnomalyResult
)
"""Any task result. The static type when the task is not known at the call site.

Mirrors :data:`pictograph.inference.models.AnyModel`. Narrow it with
``isinstance`` or by matching on :attr:`InferenceResult.model_type`, both of which
mypy understands - the union is discriminated on that literal.
"""


def build_result(
    payload: Mapping[str, Any],
    *,
    task: TaskName,
    backend: BackendName = "onnxruntime",
    device: str = "cpu",
    providers: Sequence[str] = (),
    inference_ms: float | None = None,
    source: str = "This model",
) -> AnyResult:
    """Turn ONE raw prediction payload into its typed, task-specific result.

    This is the single conversion in the SDK. Both paths feed it the same dict:
    the local engines call ``pictograph.inference._wrappers.dispatch.infer_image``,
    and a deployed endpoint runs on PRT, Pictograph's realtime engine, which answers in
    exactly that function's shape (its values checked against ONNX Runtime's when the
    deployment's engine is built) and whose answer the inference gateway passes
    through verbatim. So "Edge and Remote agree" is one contract here, not a
    convention two builders are asked to keep.

    The payload is ``{"model_type": <task>, "predictions": [...]}`` for the four
    geometry tasks; an anomaly model's also carries ``tags`` / ``labels`` (its label),
    ``anomaly_score``, ``is_anomalous`` and, when asked, ``anomaly_map``; and it
    additionally carries ``tags`` for ``classification``,
    whose prediction entries are ``{"class": name, "confidence": p}`` - note the
    key is ``class``, not ``name``, which is why classification cannot simply be
    ``model_validate``d like the others.

    Args:
        payload: The raw engine / endpoint dict.
        task: Which result class to build. The caller decides - locally from the
            loaded model, remotely from the payload's own ``model_type``.
        backend: The runtime that ran it.
        device: The device that ran it. A caller that genuinely does not know
            (a remote endpoint that did not report it) should say so rather than
            let this default to ``"cpu"`` and assert something false.
        providers: The runtime's own execution targets, if known.
        inference_ms: Forward-pass + postprocess milliseconds, if measured.
            ``None`` means not measured - never a substituted round-trip time.
        source: How to name the model in an error or a log line. Task-neutral -
            it is read on every task, not only on the classifier refusal below.

    Raises:
        ValueError: A classifier payload that ranked no classes. The shared
            emitter always reports rank 1 whatever the threshold, so this means a
            malformed payload rather than a low-confidence image.
    """
    meta: dict[str, Any] = {
        "backend": backend,
        "device": device,
        "providers": list(providers),
        "inference_ms": inference_ms,
    }
    raw_predictions: list[dict[str, Any]] = list(payload.get("predictions") or [])

    if task == "classification":
        scores = [
            ClassScore(name=str(p["class"]), confidence=float(p.get("confidence", 0.0)))
            for p in raw_predictions
            if p.get("class") is not None
        ]
        if not scores:
            raise ValueError(
                f"{source} returned no classes - the model output could not be "
                "ranked. This indicates a malformed payload."
            )
        return ClassificationResult(classes=scores, **meta)

    if task == "anomaly_detection":
        labels = list(payload.get("labels") or [])
        label = labels[0] if labels else None
        regions = [p if p.get("id") else {**p, "id": str(uuid.uuid4())} for p in raw_predictions]
        return AnomalyResult.model_validate(
            {
                "predictions": [p for p in regions if not _is_degenerate(p)],
                "label": str(label["class"]) if label else None,
                "label_confidence": float(label.get("confidence", 0.0)) if label else None,
                "score": float(payload.get("anomaly_score") or 0.0),
                # An older engine said only its label: the anomaly class's label is the decision.
                "is_anomalous": bool(payload.get("is_anomalous", bool(raw_predictions))),
                "anomaly_map": payload.get("anomaly_map"),
                **meta,
            }
        )

    # Every emitter already assigns a uuid4 `id`; backfill only so a hand-rolled
    # or third-party payload still validates against the annotation models.
    kept = [p for p in raw_predictions if not _is_degenerate(p)]
    if len(kept) != len(raw_predictions):
        _LOG.debug(
            "Dropped %d of %s's %d predictions - no spatial extent.",
            len(raw_predictions) - len(kept),
            source,
            len(raw_predictions),
        )
    predictions = [p if p.get("id") else {**p, "id": str(uuid.uuid4())} for p in kept]
    result = TASK_RESULT_TYPES[task].model_validate({"predictions": predictions, **meta})
    # `TASK_RESULT_TYPES` is declared `type[InferenceResult]`, so the concrete
    # class the mapping guarantees is not visible statically. The exhaustiveness
    # test walks every task through here, which is what actually holds the cast.
    return cast("AnyResult", result)


def _positive_extent(geometry: Any) -> bool | None:
    """``True``/``False`` if this box has/lacks extent; ``None`` if it is not a box.

    ``None`` also covers a box whose ``w``/``h`` are missing or unparseable - those
    are MALFORMED rather than degenerate, and must still reach pydantic and raise.
    """
    if not isinstance(geometry, dict):
        return None
    try:
        return float(geometry["w"]) > 0 and float(geometry["h"]) > 0
    except (KeyError, TypeError, ValueError):
        return None


def _is_degenerate(prediction: Mapping[str, Any]) -> bool:
    """Does this prediction describe a shape with NO spatial extent?

    A model asked for its low-confidence tail returns junk - that is what the tail
    IS - and among the junk are zero-area boxes and one- or two-vertex "polygons".
    The annotation models correctly refuse to represent those (``BoundingBox.w`` and
    ``.h`` are ``gt=0``; a polygon ring needs three points), so before 1.69.15 ONE
    such prediction raised ``ValidationError`` out of :meth:`predict` and took the
    whole call - every good prediction alongside it included - down with it.
    Measured on ``fixture-rfdetr_detection`` at ``confidence=0.001``: prediction 77
    of 100 had ``h == 0.0``, and the other 99 were lost with it.

    Dropping is the only defensible reading. There is nothing to repair a zero-extent
    box INTO (any epsilon would be invented pixels), nothing to draw, and nothing to
    count - and a caller who sets a low threshold is asking for a longer list, not
    for an exception. Every real emitter already clamps with ``max(0.0, x2 - x1)``,
    so what arrives here is a box the model genuinely gave no width or height.

    Deliberately narrow: ONLY geometry that is parseable AND has no extent is
    dropped. A prediction that is malformed in any other way (missing ``name``,
    a non-numeric coordinate, an unknown ``type``) still reaches pydantic and still
    raises, because that is a defect in the producer rather than a weak detection.
    """
    box = _positive_extent(prediction.get("bounding_box"))
    if box is False:
        return True
    if _positive_extent(prediction.get("oriented_box")) is False:
        return True

    polygon = prediction.get("polygon")
    if isinstance(polygon, dict):
        rings = polygon.get("paths")
        # A ring under three points encloses no area, whether it is the outer
        # boundary or a hole. Dropping the prediction rather than pruning the bad
        # ring keeps this function from ever INVENTING a shape the model did not
        # emit - and costs one prediction where raising cost every prediction.
        if isinstance(rings, list) and any(isinstance(r, list) and len(r) < 3 for r in rings):
            return True

    return False
