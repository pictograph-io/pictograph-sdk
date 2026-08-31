"""Local PyTorch keypoint inference - the raw-forward → canonical-decode path.

RF-DETR keypoint models do NOT go through ``rfdetr.predict()``: that returns
per-``(query x class)`` hypotheses and rescales their scores, which is not what
the server does. ``TorchEngine._predict_rfdetr_keypoint`` instead runs the
module's raw forward and hands ``pred_boxes`` / ``pred_logits`` /
``pred_keypoints`` to the canon's own ``decode_keypoint_outputs`` - the same
function the ONNX wrapper delegates to - then to the shared
``_keypoint_to_annotations``. These tests pin the keypoint-vs-skeleton decision
(arity 1 → point) both from the threaded schema and from the joint-count
fallback. Cross-backend equality itself is pinned in
``test_keypoint_backend_parity.py``.

That path needs the whole ``[inference]`` extra (numpy + cv2 + onnxruntime) plus
torch, which the base CI gate deliberately does not install, so these tests
``importorskip`` them exactly as ``test_inference.py`` does. The
``TestCkptSchemaExtraction`` tests below are pure-python on purpose and must stay
that way, so the base gate keeps real coverage of the arity guard.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pictograph.inference._torch import (
    TorchEngine,
    _rfdetr_ckpt_num_keypoints_per_class,
)


def _logit(p: float) -> float:
    import numpy as np

    return float(np.log(p / (1.0 - p)))


class _StubModule:
    """The rebuilt rfdetr nn.Module, replaced by fixed raw outputs.

    ``keypoints`` is the padded ``(1, Q, C * max_K, D)`` block RF-DETR emits;
    ``logits`` carries the trailing BACKGROUND column the real head has.
    """

    def __init__(self, boxes: Any, logits: Any, keypoints: Any) -> None:
        self._outputs = (boxes, logits, keypoints)

    def eval(self) -> None:
        return None

    def __call__(self, _tensor: Any) -> dict[str, Any]:
        import torch

        boxes, logits, keypoints = (torch.from_numpy(a) for a in self._outputs)
        return {"pred_boxes": boxes, "pred_logits": logits, "pred_keypoints": keypoints}


def _engine(
    classes: list[str],
    npc: list[int] | None = None,
    module: Any = None,
    confidence: float = 0.05,
) -> TorchEngine:
    torch = pytest.importorskip("torch")
    return TorchEngine(
        module=SimpleNamespace(model=module) if module is not None else object(),
        family="rfdetr",
        device="cpu",
        dtype=torch.float32,
        checkpoint_path=Path("unused.pth"),
        model_type="keypoint_detection",
        architecture="RF-DETR Keypoint Preview",
        classes=classes,
        input_size=(32, 32),
        num_keypoints_per_class=npc,
        confidence_threshold=confidence,
    )


def _image(size: int = 100) -> Any:
    import numpy as np

    return np.zeros((size, size, 3), dtype=np.uint8)


class TestKeypointPredict:
    """``_predict_rfdetr_keypoint`` - raw forward through the canonical decode.

    It delegates to ``_wrappers``, and importing that package eagerly pulls in
    every ONNX wrapper (cv2 + onnxruntime), so these need the whole
    ``[inference]`` extra, not just numpy."""

    @pytest.fixture(autouse=True)
    def _inference_extra(self) -> None:
        pytest.importorskip("cv2")
        pytest.importorskip("onnxruntime")
        pytest.importorskip("torch")

    def test_single_keypoint_class_emits_keypoint(self) -> None:
        np = pytest.importorskip("numpy")
        classes = ["head", "torso", "l_hand", "r_hand", "l_foot", "r_foot"]
        boxes = np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype=np.float32)
        # 6 foreground columns + background; 'head' wins.
        logits = np.array([[[_logit(0.8)] + [_logit(0.1)] * 5 + [_logit(0.05)]]], dtype=np.float32)
        # 6 classes x 1 joint: slot 0 is 'head' at normalized (1.0, 0.5).
        kps = np.zeros((1, 1, 6, 3), dtype=np.float32)
        kps[0, 0, 0] = [1.0, 0.5, _logit(0.9)]
        m = _engine(classes, npc=[1] * 6, module=_StubModule(boxes, logits, kps))

        preds = m.infer(_image(), confidence=0.05)["predictions"]

        assert len(preds) == 1
        assert preds[0]["type"] == "keypoint"
        assert preds[0]["name"] == "head"
        assert preds[0]["keypoint"] == {"x": 100.0, "y": 50.0}
        assert "skeleton" not in preds[0]

    def test_multi_keypoint_class_emits_one_point_per_joint(self) -> None:
        np = pytest.importorskip("numpy")
        boxes = np.array([[[0.5, 0.5, 0.4, 0.4]]], dtype=np.float32)
        logits = np.array([[[_logit(0.8), _logit(0.05)]]], dtype=np.float32)
        kps = np.array(
            [
                [
                    [
                        [0.1, 0.1, _logit(0.9)],
                        [0.2, 0.2, _logit(0.9)],
                        [0.3, 0.3, _logit(0.9)],
                    ]
                ]
            ],
            dtype=np.float32,
        )
        m = _engine(["person"], npc=[3], module=_StubModule(boxes, logits, kps))

        preds = m.infer(_image(), confidence=0.05)["predictions"]

        # A joint is a CLASS; instance_id is the OBJECT. 3 joints -> 3 annotations
        # sharing one id, and the torch engine must agree with the ONNX engine
        # exactly (they call the same emitter).
        assert len(preds) == 3
        assert {p["type"] for p in preds} == {"keypoint"}
        assert {p["instance_id"] for p in preds} == {1}
        assert all("skeleton" not in p for p in preds)

    def test_arity_falls_back_to_joint_count_without_schema(self) -> None:
        """No threaded schema → the geometry is inferred from the slot count, and
        a 1-joint class emits a ``keypoint`` named for the CLASS itself."""
        np = pytest.importorskip("numpy")
        boxes = np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype=np.float32)
        logits = np.array([[[_logit(0.1), _logit(0.8), _logit(0.05)]]], dtype=np.float32)
        kps = np.zeros((1, 1, 2, 3), dtype=np.float32)  # 2 classes x 1 joint
        kps[0, 0, 1] = [0.25, 0.25, _logit(0.9)]
        m = _engine(["kp0", "kp1"], npc=None, module=_StubModule(boxes, logits, kps))

        preds = m.infer(_image(), confidence=0.05)["predictions"]

        assert preds[0]["type"] == "keypoint"
        assert preds[0]["name"] == "kp1"

    def test_below_confidence_dropped(self) -> None:
        np = pytest.importorskip("numpy")
        boxes = np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype=np.float32)
        logits = np.array([[[_logit(0.1), _logit(0.05)]]], dtype=np.float32)
        kps = np.zeros((1, 1, 1, 3), dtype=np.float32)
        m = _engine(["head"], npc=[1], module=_StubModule(boxes, logits, kps), confidence=0.5)

        assert m.infer(_image(), confidence=0.5)["predictions"] == []

    def test_empty_result(self) -> None:
        """A graph with no query above the gate returns nothing, not an error."""
        np = pytest.importorskip("numpy")
        boxes = np.zeros((1, 0, 4), dtype=np.float32)
        logits = np.zeros((1, 0, 2), dtype=np.float32)
        kps = np.zeros((1, 0, 1, 3), dtype=np.float32)
        m = _engine(["head"], npc=[1], module=_StubModule(boxes, logits, kps))

        assert m.infer(_image(), confidence=0.05)["predictions"] == []


class TestCkptSchemaExtraction:
    def test_clean_active_first_accepted(self) -> None:
        ckpt = {"args": {"num_keypoints_per_class": [1, 1, 1]}}
        assert _rfdetr_ckpt_num_keypoints_per_class(ckpt, 3) == [1, 1, 1]

    def test_background_first_rejected(self) -> None:
        # A leading 0 (background-first) is rejected → fall back to joint count.
        ckpt = {"args": {"num_keypoints_per_class": [0, 17]}}
        assert _rfdetr_ckpt_num_keypoints_per_class(ckpt, 2) is None

    def test_length_mismatch_rejected(self) -> None:
        ckpt = {"args": {"num_keypoints_per_class": [1, 1]}}
        assert _rfdetr_ckpt_num_keypoints_per_class(ckpt, 3) is None

    def test_missing_returns_none(self) -> None:
        assert _rfdetr_ckpt_num_keypoints_per_class({"args": {}}, 2) is None
        assert _rfdetr_ckpt_num_keypoints_per_class(None, 2) is None
