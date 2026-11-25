"""``model.predict(img, confidence=X)`` returns what X asks for, whatever the model loaded with.

``load_model(..., confidence=0.5)`` builds its wrapper at 0.5, and ``predict(confidence=X)``
overrides that per call. Until the vendored ``dispatch.infer_image`` handed X to the wrapper,
an RF-DETR's wrapper gated at the LOAD confidence before dispatch filtered at X - so a
``predict`` below it silently lost every detection in between - and a semantic model ignored
X altogether (D-009 in pictograph-prt/docs/DECISIONS.md; the same defect on the server's
deployments and workflows).

Both run the REAL vendored wrapper over a stubbed ONNX session behind the real
``OnnxEngine`` and task model, so they exercise exactly the path ``predict`` takes.
"""

from __future__ import annotations

import types
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _inference_extra() -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("numpy")


class _StubSession:
    def __init__(self, outputs: list[Any]) -> None:
        self._outputs = outputs

    def get_inputs(self) -> list[Any]:
        return [types.SimpleNamespace(name="input")]

    def get_outputs(self) -> list[Any]:
        return [types.SimpleNamespace(name=f"out{i}") for i in range(len(self._outputs))]

    def run(self, _names: Any, _feed: Any) -> list[Any]:
        return [o.copy() for o in self._outputs]


def _detector(loaded_at: float) -> Any:
    """A real RFDETRDetector: class "a" at 0.95 and 0.40, class "b" at 0.60, far apart."""
    import numpy as np

    from pictograph.inference._wrappers.rfdetr_det_wrapper import RFDETRDetector

    boxes = np.array(
        [[[0.2, 0.2, 0.15, 0.15], [0.5, 0.5, 0.15, 0.15], [0.8, 0.8, 0.15, 0.15]]], dtype=np.float32
    )
    logits = np.full((1, 3, 3), -10.0, dtype=np.float32)
    for q, (col, p) in enumerate([(0, 0.95), (0, 0.40), (1, 0.60)]):
        logits[0, q, col] = np.log(p / (1.0 - p))
    w = object.__new__(RFDETRDetector)
    w.model_path, w.classes, w.dims = "stub", ["a", "b"], (32, 32)
    w.confidence_threshold, w.nms_threshold, w.providers = loaded_at, 0.5, []
    w.session = _StubSession([boxes, logits])
    w.input_name, w.output_names = "input", ["out0", "out1"]
    w._orig_height = w._orig_width = None
    return w


def _semantic(loaded_at: float) -> Any:
    """A real semantic wrapper: road blocks at 0.987 and 0.40, a car block at 0.60."""
    import numpy as np

    from pictograph.inference._wrappers.sm_pytorch_wrapper import SemanticSegmentationModelPyTorch

    logits = np.zeros((1, 3, 64, 64), dtype=np.float32)
    logits[0, 0] = 5.0
    logits[0, :, 8:24, 8:56] = np.array([0.0, 5.0, 0.0])[:, None, None]
    logits[0, :, 40:56, 8:28] = np.array([0.0, np.log(0.4 / 0.3), 0.0])[:, None, None]
    logits[0, :, 40:56, 36:56] = np.array([0.0, 0.0, np.log(3.0)])[:, None, None]
    w = object.__new__(SemanticSegmentationModelPyTorch)
    w.model_path, w.dims, w.ratio, w.classes, w.providers = (
        "stub",
        (64, 64),
        1.0,
        ["road", "car"],
        [],
    )
    w.mask_max_side, w.class_confidences, w.confidence_threshold = None, {}, loaded_at
    w.session = _StubSession([logits])
    return w


def _model(cls: Any, wrapper: Any, model_type: str, loaded_at: float) -> Any:
    from pictograph.inference._onnx import OnnxEngine

    engine = OnnxEngine(
        wrapper=wrapper,
        model_type=model_type,
        architecture="stub",
        classes=list(wrapper.classes),
        providers=["CPUExecutionProvider"],
    )
    return cls(engine=engine, model_id="m", name="m", architecture="stub", confidence=loaded_at)


def _image() -> Any:
    import numpy as np

    return np.zeros((64, 64, 3), dtype=np.uint8)


def test_detection_predict_below_the_load_confidence_returns_those_detections() -> None:
    from pictograph.inference.models import DetectionModel

    model = _model(DetectionModel, _detector(0.5), "object_detection", 0.5)
    scores = sorted(
        round(float(d.confidence), 2) for d in model.predict(_image(), confidence=0.2).predictions
    )
    assert scores == [0.4, 0.6, 0.95]
    # And the load confidence is still the default.
    assert sorted(round(float(d.confidence), 2) for d in model.predict(_image()).predictions) == [
        0.6,
        0.95,
    ]


def test_semantic_predict_gates_its_pixels_at_the_requested_confidence() -> None:
    from pictograph.inference.models import SemanticSegmentationModel

    model = _model(SemanticSegmentationModel, _semantic(0.5), "semantic_segmentation", 0.5)
    assert sorted(p.name for p in model.predict(_image(), confidence=0.2).predictions) == [
        "car",
        "road",
        "road",
    ]
    assert sorted(p.name for p in model.predict(_image(), confidence=0.7).predictions) == ["road"]
    assert sorted(p.name for p in model.predict(_image()).predictions) == ["car", "road"]
