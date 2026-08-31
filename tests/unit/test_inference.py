"""Unit tests for the local-inference layer (pictograph.inference).

The friendly result-shaping + config resolution are tested here with no ONNX
runtime. The provider auto-detect and image decoding need the [inference] extra,
so they ``importorskip`` and are skipped in the base gate. End-to-end model
loading + prediction is covered by tests/live/test_inference_live.py.

Result shaping used to go through one `_to_result()` function. It is gone --
each per-task model class (`pictograph.inference.models`) now builds its own
typed result via `_build()`, so these tests build one directly with a fake
engine standing in for a real ONNX session / torch module. See
`test_inference_results.py` for the result TYPES themselves (narrowing,
`.instances`/`.points`/`.polygons`/`.top`/`.tags`) and `test_task_dispatch.py`
for the `task=` verification these loaders lean on.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import pytest

from pictograph.inference import _classes_of, _input_size_of
from pictograph.inference.models import TASK_MODEL_TYPES
from pictograph.inference.results import (
    ClassificationResult,
    ClassScore,
    DetectionResult,
    InstanceSegmentationResult,
)
from pictograph.models.annotation import BBoxAnnotation, PolygonAnnotation
from pictograph.models.model import Model


def _model(**over: object) -> Model:
    base: dict[str, object] = {
        "id": "11111111-1111-1111-1111-111111111111",
        "organization_id": "org",
        "name": "My Detector",
        "model_type": "object_detection",
        "architecture": "YOLOX",
        "visibility": "private",
        "status": "ready",
        "class_mapping": {"classes": ["person", "car"]},
        "training_config": {"image_height": 512, "image_width": 512},
        "created_at": "2026-07-01T00:00:00Z",
        "updated_at": "2026-07-01T00:00:00Z",
    }
    base.update(over)
    return Model.model_validate(base)


class _FakeEngine:
    """The minimal engine surface `_TaskModel._meta` reads -- backend/device/
    providers/classes -- so a task model class's `_build()` can be exercised
    without a real ONNX session or torch module."""

    def __init__(
        self,
        *,
        backend: str = "onnxruntime",
        device: str = "cpu",
        providers: list[str] | None = None,
        classes: list[str] | None = None,
    ) -> None:
        self.backend = backend
        self.device = device
        self.providers = providers or []
        self.classes = classes or []

    def close(self) -> None:
        pass


def _build(task: str, raw: dict[str, Any], **engine_kwargs: Any) -> Any:
    """Build a typed result via the real per-task model class's `_build()` --
    the replacement for the deleted `_to_result()`."""
    cls = TASK_MODEL_TYPES[task]
    model = cls(
        engine=_FakeEngine(**engine_kwargs),
        model_id="m",
        name="Test Model",
        architecture="",
        confidence=0.5,
    )
    return model._build(raw, 1.23)


class TestResultShaping:
    def test_detection_maps_to_typed_annotations(self) -> None:
        raw = {
            "model_type": "object_detection",
            "predictions": [
                {
                    "id": "a",
                    "name": "person",
                    "type": "bbox",
                    "bounding_box": {"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0},
                    "confidence": 0.9,
                    "attributes": ["auto-annotate"],
                }
            ],
        }
        result = _build("object_detection", raw)
        assert isinstance(result, DetectionResult)
        assert result.model_type == "object_detection"
        assert len(result.predictions) == 1
        pred = result.predictions[0]
        assert isinstance(pred, BBoxAnnotation)
        assert pred.name == "person"
        assert pred.bounding_box.x == 1.0
        assert pred.confidence == 0.9
        # Per-task subclasses declare ONLY the fields their task can produce --
        # no shared `classes`/`top` leaking in from the classifier shape.
        assert not hasattr(result, "classes")
        assert not hasattr(result, "top")

    def test_polygon_prediction_maps_to_polygon_annotation(self) -> None:
        raw = {
            "model_type": "instance_segmentation",
            "predictions": [
                {
                    "name": "person",
                    "type": "polygon",
                    "polygon": {"paths": [[{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}]]},
                    "confidence": 0.8,
                }
            ],
        }
        result = _build("instance_segmentation", raw)
        assert isinstance(result, InstanceSegmentationResult)
        assert isinstance(result.predictions[0], PolygonAnnotation)
        # A missing id is filled in (the Annotation model requires one).
        assert result.predictions[0].id
        assert result.polygons == result.predictions

    def test_classification_maps_to_ranked_classes_and_tags(self) -> None:
        raw = {
            "model_type": "classification",
            "predictions": [
                {"class": "cat", "confidence": 0.98},
                {"class": "dog", "confidence": 0.02},
            ],
        }
        result = _build("classification", raw)
        assert isinstance(result, ClassificationResult)
        # ClassificationResult has no `predictions` at all -- geometry-shaped
        # tasks and the classifier no longer share a field name/shape.
        assert not hasattr(result, "predictions")
        assert result.classes == [
            ClassScore(name="cat", confidence=0.98),
            ClassScore(name="dog", confidence=0.02),
        ]
        assert result.top == ClassScore(name="cat", confidence=0.98)
        assert result.tags == ["cat", "dog"]

    def test_classification_with_no_scores_raises(self) -> None:
        """`ClassificationResult.classes` forbids an empty list (min_length=1);
        an engine payload that ranked nothing is a malformed payload, not a
        valid zero-class result."""
        raw = {"model_type": "classification", "predictions": []}
        with pytest.raises(ValueError, match="returned no classes"):
            _build("classification", raw)

    def test_predictions_round_trip_to_deployment_dict(self) -> None:
        raw = {
            "model_type": "object_detection",
            "predictions": [
                {
                    "name": "car",
                    "type": "bbox",
                    "bounding_box": {"x": 0, "y": 0, "w": 5, "h": 5},
                    "confidence": 0.7,
                }
            ],
        }
        dumped = (
            _build("object_detection", raw)
            .predictions[0]
            .model_dump(mode="json", exclude_none=True)
        )
        assert dumped["name"] == "car"
        assert dumped["type"] == "bbox"
        assert dumped["bounding_box"] == {"x": 0.0, "y": 0.0, "w": 5.0, "h": 5.0}

    def test_meta_fields_reflect_the_engine_not_a_hardcoded_backend(self) -> None:
        """The interchangeable-backend promise: the SAME `_build()` on the SAME
        task class reports whatever engine actually produced the raw dict."""
        raw = {"model_type": "object_detection", "predictions": []}
        result = _build("object_detection", raw, backend="pytorch", device="mps", providers=[])
        assert result.backend == "pytorch"
        assert result.device == "mps"
        assert result.providers == []
        assert result.inference_ms == 1.23


class TestConfigResolution:
    def test_classes_from_mapping(self) -> None:
        assert _classes_of(_model()) == ["person", "car"]

    def test_missing_classes_is_a_clear_error(self) -> None:
        with pytest.raises(ValueError, match="no class list"):
            _classes_of(_model(class_mapping={}))

    def test_input_size_from_training_config(self) -> None:
        assert _input_size_of(_model()) == (512, 512)

    def test_input_size_defaults_when_absent(self) -> None:
        assert _input_size_of(_model(training_config=None)) == (640, 640)


class TestProviders:
    """`resolve_providers` moved to `pictograph.inference.runtime` and now
    returns ORT's tuned tuple form for providers with options (e.g. CUDA's
    `cudnn_conv_algo_search`), not bare strings. See
    `test_inference_runtime.py` for the exhaustive, OS-independent suite;
    these two pin the same behaviour this module used to own directly."""

    def test_prefers_cuda_then_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ort = pytest.importorskip("onnxruntime")
        from pictograph.inference.runtime import resolve_providers

        monkeypatch.setattr(
            ort,
            "get_available_providers",
            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        resolved = resolve_providers("auto")
        assert resolved[0][0] == "CUDAExecutionProvider"
        assert resolved[-1] == "CPUExecutionProvider"

        monkeypatch.setattr(
            ort,
            "get_available_providers",
            lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        assert resolve_providers("auto") == ["CPUExecutionProvider"]  # CPU fallback, no CUDA

    def test_the_measurement_hatch_passes_providers_through(self) -> None:
        pytest.importorskip("onnxruntime")
        from pictograph.inference.runtime import resolve_providers

        assert resolve_providers(requested=["MyProvider"]) == ["MyProvider"]


class TestImageDecode:
    """`_decode_image` moved to `pictograph.inference.models` -- it is now the
    ONE shared decoder both the ONNX and torch engines call before handing
    off to their own (BGR-array-only) preprocessing."""

    def test_decode_ndarray_passthrough(self) -> None:
        np = pytest.importorskip("numpy")
        pytest.importorskip("cv2")
        from pictograph.inference.models import _decode_image

        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        assert _decode_image(arr) is arr

    def test_decode_bytes(self) -> None:
        np = pytest.importorskip("numpy")
        cv2 = pytest.importorskip("cv2")
        from pictograph.inference.models import _decode_image

        ok, buf = cv2.imencode(".png", np.zeros((4, 4, 3), dtype=np.uint8))
        assert ok
        out = _decode_image(bytes(buf))
        assert out.shape == (4, 4, 3)

    def test_decode_pil_image_converts_to_bgr(self) -> None:
        """Ported from the deleted `_decode_to_pil` -- the shared decoder now
        returns a BGR array (the SDK's one array convention) rather than a
        PIL image."""
        pytest.importorskip("numpy")
        pytest.importorskip("cv2")
        from PIL import Image

        from pictograph.inference.models import _decode_image

        rgba = Image.new("RGBA", (2, 2), (255, 0, 0, 255))  # solid red
        out = _decode_image(rgba)
        assert out.shape == (2, 2, 3)
        # Red in RGB is last-channel in BGR: (B, G, R) = (0, 0, 255).
        assert tuple(int(c) for c in out[0, 0]) == (0, 0, 255)

    def test_decode_missing_path_raises(self) -> None:
        """Ported from the deleted `_decode_to_pil`'s missing-file case."""
        pytest.importorskip("cv2")
        from pictograph.inference.models import _decode_image

        with pytest.raises(FileNotFoundError):
            _decode_image("/nowhere/does-not-exist.jpg")

    def test_unsupported_input_is_a_clear_error(self) -> None:
        pytest.importorskip("cv2")
        from pictograph.inference.models import _decode_image

        with pytest.raises(TypeError, match="path, URL, bytes"):
            _decode_image(1234)  # type: ignore[arg-type]


# ───────────── _true_input_size (live-caught, LOCKSTEP w/ service) ─────────────


class TestTrueInputSize:
    """The graph's static shape must beat the training_config guess - a model
    stored with a minimal config otherwise mis-sizes its inputs and ORT
    rejects the tensor (RF-DETR nano: 384 vs the 640 default)."""

    @staticmethod
    def _make(tmp_path, h, w):
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto, helper

        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, h, w])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, h, w])
        node = helper.make_node("Identity", ["input"], ["output"])
        graph = helper.make_graph([node], "g", [inp], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        p = tmp_path / f"m_{h}_{w}.onnx"
        onnx.save(model, str(p))
        return p

    def test_static_graph_shape_overrides_declared(self, tmp_path: Path) -> None:
        from pictograph.inference import _true_input_size

        path = self._make(tmp_path, 384, 384)
        assert _true_input_size(path, (640, 640)) == (384, 384)

    def test_dynamic_dims_keep_declared(self, tmp_path: Path) -> None:
        from pictograph.inference import _true_input_size

        path = self._make(tmp_path, "height", "width")
        assert _true_input_size(path, (512, 512)) == (512, 512)

    def test_unreadable_file_falls_back(self, tmp_path: Path) -> None:
        from pictograph.inference import _true_input_size

        bad = tmp_path / "bad.onnx"
        bad.write_bytes(b"not a protobuf")
        assert _true_input_size(bad, (640, 640)) == (640, 640)


class TestSemanticSegDispatchRegression:
    """`masks or []` on a numpy stack raised the ambiguous-truth ValueError,
    killing EVERY semantic-seg predict through dispatch.infer_image
    (deployment /infer, workflow model nodes, SDK local). Fixed 2026-07-17;
    this pins the ndarray-shaped return path.

    The stubs mirror the real wrapper's contract: ``infer_image`` asks for
    ``return_probs=True`` (so the emitter gets a real confidence rather than the
    1.0 default), which returns the ``(masks, probs)`` pair."""

    def test_infer_image_accepts_ndarray_mask_stack(self) -> None:
        np = pytest.importorskip("numpy")
        pytest.importorskip("cv2")
        from pictograph.inference._wrappers import dispatch

        class StubSmWrapper:
            def predict(self, _img: object, return_probs: bool = False) -> object:
                masks = np.zeros((2, 16, 16), dtype=np.uint8)
                masks[0, 4:12, 4:12] = 1  # one real component for class 0
                probs = np.full((2, 16, 16), 0.75, dtype=np.float32)
                return (masks, list(probs)) if return_probs else masks

        out = dispatch.infer_image(
            StubSmWrapper(),
            np.zeros((16, 16, 3), dtype=np.uint8),
            model_type="semantic_segmentation",
            architecture="unetplusplus",
            classes=["road", "car"],
            confidence=0.5,
        )
        assert out["model_type"] == "semantic_segmentation"
        names = {p["name"] for p in out["predictions"]}
        assert names == {"road"}
        # The probability map reaches the emitter, so the polygon carries a real
        # confidence instead of falling back to the pydantic default of 1.0.
        assert out["predictions"][0]["confidence"] == pytest.approx(0.75)

    def test_infer_image_accepts_none_masks(self) -> None:
        np = pytest.importorskip("numpy")
        pytest.importorskip("cv2")
        from pictograph.inference._wrappers import dispatch

        class StubNoneWrapper:
            def predict(self, _img: object, return_probs: bool = False) -> object:
                return (None, None) if return_probs else None

        out = dispatch.infer_image(
            StubNoneWrapper(),
            np.zeros((8, 8, 3), dtype=np.uint8),
            model_type="semantic_segmentation",
            architecture="unet",
            classes=["a"],
            confidence=0.5,
        )
        assert out["predictions"] == []
