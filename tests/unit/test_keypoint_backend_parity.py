"""The two keypoint backends must reduce the SAME raw model output identically.

This is the regression guard for a real, confirmed divergence. ``rfdetr.predict()``
does not reduce the way the server does:

* its ``PostProcess._select_topk`` flattens ``(queries x classes)`` sigmoid
  probabilities and takes a GLOBAL top-k, so ONE query is returned once per class
  it scores on - the same object arrives several times under several labels, and
  the trailing background column competes as if it were a class;
* it then rescales the surviving scores by a keypoint-uncertainty factor;
* it never runs class-aware NMS.

The ONNX wrapper - which is what the batch inference service, per-deployment
serving and the workflow runner all run, i.e. the CANON - instead takes a
per-query argmax over the FOREGROUND columns only, gates on confidence, and
suppresses per class. Measured on a real 6-class RF-DETR Keypoint model at
conf=0.05: the torch path emitted 100 predictions where the canon emitted 19, and
put ``head`` / ``l_foot`` where the canon put ``l_hand`` on the same coordinate.

The fix routes the torch engine through the canon's own
``decode_keypoint_outputs``. These tests pin that with FAKES - no network, no
rfdetr, no trained weights - by feeding both engines one fixed set of raw
hypothesis tensors and requiring byte-identical annotations out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("onnxruntime")

import numpy as np

from pictograph.inference._torch import TorchEngine
from pictograph.inference._wrappers import dispatch
from pictograph.inference._wrappers.rfdetr_kp_wrapper import (
    RFDETRKeypointDetector,
    decode_keypoint_outputs,
)

CLASSES = ["head", "torso", "l_hand"]
COUNTS = [1, 1, 1]
IMAGE_HW = (100, 200)  # (h, w) - deliberately non-square so a swap would show
INPUT_HW = (32, 32)


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _raw_outputs() -> list[np.ndarray]:
    """Three queries whose logits are AMBIGUOUS across classes on purpose.

    Every query scores respectably on more than one class, which is exactly the
    shape that makes a global (query x class) top-k return the same object
    repeatedly under different labels. Query 2 is below any sane gate.

    Keypoint block layout is ``C * max_K`` slots: with 3 classes x 1 joint the
    slot index IS the class index, so reading the wrong class block moves the
    emitted point to a visibly different place.
    """
    boxes = np.array(
        [
            [[0.25, 0.25, 0.20, 0.20], [0.75, 0.75, 0.30, 0.30], [0.50, 0.50, 0.10, 0.10]],
        ],
        dtype=np.float32,
    )
    # Columns: head, torso, l_hand, BACKGROUND (RF-DETR's trailing slot).
    logits = np.array(
        [
            [
                [_logit(0.40), _logit(0.35), _logit(0.30), _logit(0.90)],  # bg outscores all
                [_logit(0.20), _logit(0.60), _logit(0.55), _logit(0.10)],
                [_logit(0.02), _logit(0.01), _logit(0.03), _logit(0.99)],  # below any gate
            ]
        ],
        dtype=np.float32,
    )
    row = lambda x, y: [x, y, _logit(0.95), 0.0]  # noqa: E731 - D=4, tail ignored
    keypoints = np.array(
        [
            [
                [row(0.10, 0.10), row(0.20, 0.20), row(0.30, 0.30)],
                [row(0.60, 0.60), row(0.70, 0.70), row(0.80, 0.80)],
                [row(0.40, 0.40), row(0.45, 0.45), row(0.50, 0.50)],
            ]
        ],
        dtype=np.float32,
    )
    return [boxes, logits, keypoints]


def _onnx_wrapper(conf: float) -> RFDETRKeypointDetector:
    """The canon wrapper with a FAKE session that replays fixed outputs.

    Built with ``object.__new__`` so no ONNX file is needed; ``preprocess`` and
    ``postprocess`` are the real ones.
    """
    wrapper = object.__new__(RFDETRKeypointDetector)
    wrapper.classes = list(CLASSES)
    wrapper.dims = INPUT_HW
    wrapper.confidence_threshold = conf
    wrapper.nms_threshold = 0.5
    wrapper.num_keypoints_per_class = list(COUNTS)
    wrapper.keypoint_names = {c: [c] for c in CLASSES}
    wrapper.skeleton_edges = {}
    wrapper.keypoint_threshold = 0.5
    wrapper.session = _FakeSession()
    wrapper.input_name = "input"
    wrapper.output_names = ["boxes", "logits", "keypoints"]
    return wrapper


class _FakeSession:
    """Records the tensor it was fed, returns the fixed raw outputs."""

    def __init__(self) -> None:
        self.seen: Any = None

    def run(self, _names: Any, feed: dict[str, Any]) -> list[np.ndarray]:
        self.seen = next(iter(feed.values()))
        return _raw_outputs()


class _FakeModule:
    """Stands in for the rebuilt rfdetr nn.Module: records its input tensor and
    returns the SAME raw outputs the fake ONNX session returns."""

    def __init__(self, as_tuple: bool = False) -> None:
        self.seen: Any = None
        self.eval_calls = 0
        self.as_tuple = as_tuple

    def eval(self) -> None:
        self.eval_calls += 1

    def __call__(self, tensor: Any) -> Any:
        import torch

        self.seen = tensor
        boxes, logits, keypoints = (torch.from_numpy(a) for a in _raw_outputs())
        if self.as_tuple:
            return (boxes, logits, keypoints)
        return {"pred_boxes": boxes, "pred_logits": logits, "pred_keypoints": keypoints}


def _torch_engine(conf: float, module: Any) -> TorchEngine:
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    return TorchEngine(
        module=SimpleNamespace(model=module),
        family="rfdetr",
        device="cpu",
        dtype=torch.float32,
        checkpoint_path=Path("unused.safetensors"),
        model_type="keypoint_detection",
        architecture="RF-DETR Keypoint Preview",
        classes=list(CLASSES),
        input_size=INPUT_HW,
        num_keypoints_per_class=list(COUNTS),
        keypoint_names={c: [c] for c in CLASSES},
        skeleton_edges={},
        confidence_threshold=conf,
    )


def _image() -> np.ndarray:
    """A deterministic BGR image - content is irrelevant (both fakes replay fixed
    outputs) but it must be IDENTICAL for the preprocessing comparison."""
    rng = np.random.default_rng(1234)
    return rng.integers(0, 255, (IMAGE_HW[0], IMAGE_HW[1], 3), dtype=np.uint8)


def _strip_ids(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in p.items() if k != "id"} for p in preds]


# ───────────── the reduction itself ─────────────


class TestSharedReduction:
    """``decode_keypoint_outputs`` is the ONE reduction; the wrapper delegates."""

    @pytest.mark.parametrize("conf", [0.05, 0.3, 0.5, 0.9])
    def test_wrapper_postprocess_is_the_shared_decode(self, conf: float) -> None:
        wrapper = _onnx_wrapper(conf)
        via_method = wrapper.postprocess(_raw_outputs(), IMAGE_HW)
        via_function = decode_keypoint_outputs(
            _raw_outputs(),
            IMAGE_HW,
            classes=CLASSES,
            num_keypoints_per_class=COUNTS,
            confidence_threshold=conf,
            nms_threshold=0.5,
        )
        for a, b in zip(via_method[:3], via_function[:3], strict=True):
            np.testing.assert_array_equal(a, b)
        assert len(via_method[3]) == len(via_function[3])
        for a_kp, b_kp in zip(via_method[3], via_function[3], strict=True):
            np.testing.assert_array_equal(a_kp, b_kp)

    def test_one_hypothesis_per_query_not_one_per_query_times_class(self) -> None:
        """THE bug. rfdetr's own reduction returns ``queries x classes``
        hypotheses; the canon returns at most one per query."""
        boxes, scores, class_ids, _ = decode_keypoint_outputs(
            _raw_outputs(),
            IMAGE_HW,
            classes=CLASSES,
            num_keypoints_per_class=COUNTS,
            confidence_threshold=0.05,
        )
        # 3 queries in, and the third is below the gate → 2 detections, NOT the
        # 3 x 4 = 12 hypotheses a flattened top-k would surface.
        assert len(boxes) == 2
        assert len(scores) == len(class_ids) == 2

    def test_the_predicted_class_is_the_foreground_argmax_not_the_global_max(self) -> None:
        """Query 0's BACKGROUND column (0.90) outscores every class. rfdetr's
        top-k would return it as a detection labelled ``__background__``; the
        canon reports the best real class (``head``, 0.40) instead."""
        _, scores, class_ids, keypoints = decode_keypoint_outputs(
            _raw_outputs(),
            IMAGE_HW,
            classes=CLASSES,
            num_keypoints_per_class=COUNTS,
            confidence_threshold=0.05,
        )
        by_class = dict(zip((int(c) for c in class_ids), (float(s) for s in scores), strict=True))
        assert by_class[0] == pytest.approx(0.40, abs=1e-5)  # head, not background
        assert by_class[1] == pytest.approx(0.60, abs=1e-5)  # torso, not l_hand (0.55)
        # And the joint read is the PREDICTED class's own slot: query 1 -> class 1
        # -> slot 1 -> normalized (0.70, 0.70) scaled by (w=200, h=100).
        ordered = sorted(zip(class_ids, keypoints, strict=True), key=lambda p: int(p[0]))
        np.testing.assert_allclose(ordered[1][1][0, :2], [0.70 * 200, 0.70 * 100], rtol=1e-4)


# ───────────── engine-level parity ─────────────


class TestEngineParity:
    """Same image, same raw model output → byte-identical annotations from both
    engines, through their FULL paths (preprocess → run → decode → emit)."""

    @pytest.mark.parametrize("conf", [0.05, 0.3, 0.5])
    def test_both_engines_emit_identical_annotations(self, conf: float) -> None:
        pytest.importorskip("torch")
        image = _image()

        wrapper = _onnx_wrapper(conf)
        onnx_preds = dispatch.infer_image(
            wrapper,
            image,
            model_type="keypoint_detection",
            architecture="RF-DETR Keypoint Preview",
            classes=CLASSES,
            confidence=conf,
        )["predictions"]

        module = _FakeModule()
        torch_preds = _torch_engine(conf, module).infer(image, confidence=conf)["predictions"]

        assert _strip_ids(onnx_preds) == _strip_ids(torch_preds)

    def test_both_engines_are_fed_the_identical_preprocessed_tensor(self) -> None:
        """Preprocessing parity, proven rather than asserted in a comment: a
        resize/channel-order/normalization difference silently disagrees about
        the same image even when the reduction is shared."""
        pytest.importorskip("torch")
        image = _image()

        wrapper = _onnx_wrapper(0.05)
        wrapper.predict(image)

        module = _FakeModule()
        _torch_engine(0.05, module).infer(image, confidence=0.05)

        np.testing.assert_array_equal(np.asarray(wrapper.session.seen), module.seen.cpu().numpy())

    def test_the_module_is_put_in_eval_mode_before_every_forward(self) -> None:
        """rfdetr's own ``predict`` re-asserts ``eval()`` on each call because
        training reassigns the module; bypassing ``predict`` must not lose that,
        or dropout runs during inference."""
        pytest.importorskip("torch")
        module = _FakeModule()
        engine = _torch_engine(0.05, module)
        engine.infer(_image(), confidence=0.05)
        engine.infer(_image(), confidence=0.05)
        assert module.eval_calls == 2

    def test_a_tuple_returning_module_decodes_the_same_as_a_dict_returning_one(self) -> None:
        """A compiled / export-shim rfdetr returns ``(boxes, logits, keypoints)``
        instead of a dict. Both must land on the same annotations."""
        pytest.importorskip("torch")
        image = _image()
        as_dict = _torch_engine(0.05, _FakeModule()).infer(image, confidence=0.05)
        as_tuple = _torch_engine(0.05, _FakeModule(as_tuple=True)).infer(image, confidence=0.05)
        assert _strip_ids(as_dict["predictions"]) == _strip_ids(as_tuple["predictions"])

    def test_per_call_confidence_narrows_both_engines_the_same_way(self) -> None:
        """The load-time threshold gates the decode on both engines; a HIGHER
        per-call value then narrows the emitter identically."""
        pytest.importorskip("torch")
        image = _image()

        wrapper = _onnx_wrapper(0.05)
        onnx_preds = dispatch.infer_image(
            wrapper,
            image,
            model_type="keypoint_detection",
            architecture="RF-DETR Keypoint Preview",
            classes=CLASSES,
            confidence=0.5,
        )["predictions"]
        torch_preds = _torch_engine(0.05, _FakeModule()).infer(image, confidence=0.5)["predictions"]

        assert len(onnx_preds) == 1  # only the 0.60 torso survives 0.5
        assert _strip_ids(onnx_preds) == _strip_ids(torch_preds)
