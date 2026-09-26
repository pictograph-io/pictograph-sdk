"""The two RF-DETR backends must reduce the SAME raw model output identically.

The detection / instance-segmentation twin of ``test_keypoint_backend_parity``.
Keypoint was moved onto the canon's shared decode when its divergence was found;
**detection and segmentation never were**, and they carried the identical defect
until this gate was added.

``rfdetr.predict()`` does not reduce the way the server does: its
``PostProcess._select_topk`` flattens the ``(queries x classes)`` sigmoid grid and
takes a GLOBAL top-k, so ONE query is returned once per class it scores on. The
ONNX wrapper - what the batch inference service, per-deployment serving and the
workflow runner all run, i.e. the CANON - instead takes a per-query argmax over
the FOREGROUND columns only, gates on confidence, and suppresses per class.

Measured offline on the dev-org fixture models at conf=0.05, on one 1440x810
photo, before the fix:

===================  =====  =====
model                 ONNX  torch
===================  =====  =====
fixture-rfdetr_detection      62    126
fixture-rfdetr_segmentation    2      6
===================  =====  =====

After it, both read 62 and 2, with matched-pair IoU >= 0.99 and no query
changing either its survival or its argmax class.

These tests pin that with FAKES - no network, no trained weights - by feeding both
engines one fixed set of raw hypothesis tensors and requiring byte-identical
annotations out.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("onnxruntime")

import numpy as np

from pictograph.inference._torch import TorchEngine
from pictograph.inference._wrappers.rfdetr_det_wrapper import (
    RFDETRDetector,
    decode_detection_outputs,
    preprocess_detection_image,
)
from pictograph.inference._wrappers.rfdetr_seg_wrapper import (
    RFDETRSegDetector,
    decode_segmentation_outputs,
    preprocess_segmentation_image,
)

CLASSES = ["pallet", "forklift", "crate"]
IMAGE_HW = (100, 200)  # (h, w) - deliberately non-square so a swap would show
INPUT_HW = (32, 32)
MASK_HW = (16, 16)


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _boxes() -> np.ndarray:
    """Six queries: four well-separated, then two that OVERLAP query 0.

    The overlapping pair is load-bearing, not padding. With only separated boxes
    NMS never suppresses anything, so disabling it entirely is a no-op and a test
    that "pins the shared decode" passes over a divergent copy - verified: a
    mutation setting the wrapper's ``nms_threshold`` to 1.01 was caught by
    nothing until these two rows existed. ``test_nms_actually_suppresses_on_this_
    fixture`` keeps that property from being edited away.

    Scores are far enough apart (0.40 / 0.38 / 0.36) that NMS ordering is never a
    near-tie, so the expected survivor is deterministic.
    """
    return np.array(
        [
            [
                [0.20, 0.20, 0.10, 0.10],
                [0.80, 0.20, 0.10, 0.10],
                [0.20, 0.80, 0.10, 0.10],
                [0.50, 0.50, 0.06, 0.06],
                [0.21, 0.21, 0.10, 0.10],  # ~IoU 0.82 with query 0, same class
                [0.19, 0.22, 0.11, 0.11],  # ~IoU 0.68 with query 0, same class
            ]
        ],
        dtype=np.float32,
    )


def _logits() -> np.ndarray:
    """Logits that are AMBIGUOUS across classes on purpose.

    Every query scores respectably on more than one class - exactly the shape
    that makes a global (query x class) top-k return the same object repeatedly
    under different labels. The trailing column is RF-DETR's BACKGROUND slot and
    must never win; on query 0 it deliberately outscores every class.
    Query 3 is below any sane gate.
    """
    return np.array(
        [
            [
                # pallet  forklift  crate   BACKGROUND
                [_logit(0.40), _logit(0.35), _logit(0.30), _logit(0.95)],
                [_logit(0.20), _logit(0.60), _logit(0.55), _logit(0.10)],
                [_logit(0.70), _logit(0.65), _logit(0.10), _logit(0.05)],
                [_logit(0.02), _logit(0.01), _logit(0.03), _logit(0.99)],
                [_logit(0.38), _logit(0.12), _logit(0.09), _logit(0.20)],  # loses NMS to q0
                [_logit(0.36), _logit(0.11), _logit(0.08), _logit(0.20)],  # loses NMS to q0
            ]
        ],
        dtype=np.float32,
    )


def _masks() -> np.ndarray:
    """One mask per query - a filled quadrant each, so a mis-paired mask lands
    somewhere visibly different."""
    n = _boxes().shape[1]
    m = np.full((1, n, MASK_HW[0], MASK_HW[1]), -8.0, dtype=np.float32)
    half_h, half_w = MASK_HW[0] // 2, MASK_HW[1] // 2
    quadrants = [
        (slice(None, half_h), slice(None, half_w)),
        (slice(None, half_h), slice(half_w, None)),
        (slice(half_h, None), slice(None, half_w)),
        (slice(half_h, None), slice(half_w, None)),
    ]
    for i in range(n):
        rows, cols = quadrants[i % len(quadrants)]
        m[0, i, rows, cols] = 8.0
    return m


def _det_outputs() -> list[np.ndarray]:
    return [_boxes(), _logits()]


def _seg_outputs() -> list[np.ndarray]:
    return [_boxes(), _logits(), _masks()]


class _FakeSession:
    """Records the tensor it was fed, returns the fixed raw outputs."""

    def __init__(self, outputs: Any) -> None:
        self.seen: Any = None
        self._outputs = outputs

    def run(self, _names: Any, feed: dict[str, Any]) -> list[np.ndarray]:
        self.seen = next(iter(feed.values()))
        return self._outputs()


class _FakeModule:
    """Stands in for the rebuilt rfdetr nn.Module: records its input tensor and
    returns the SAME raw outputs the fake ONNX session returns."""

    def __init__(self, outputs: Any, *, as_tuple: bool = False, mask_key: bool = False) -> None:
        self.seen: Any = None
        self.eval_calls = 0
        self._outputs = outputs
        self.as_tuple = as_tuple
        self.mask_key = mask_key

    def eval(self) -> None:
        self.eval_calls += 1

    def __call__(self, tensor: Any) -> Any:
        import torch

        self.seen = tensor
        arrays = [torch.from_numpy(a) for a in self._outputs()]
        if self.as_tuple:
            return tuple(arrays)
        out = {"pred_boxes": arrays[0], "pred_logits": arrays[1]}
        if len(arrays) == 3:
            out["pred_masks" if self.mask_key else "pred_keypoints"] = arrays[2]
        return out


def _det_wrapper(conf: float) -> RFDETRDetector:
    """The canon wrapper with a FAKE session, so no ONNX file is needed;
    ``preprocess`` and ``postprocess`` are the real ones."""
    w = object.__new__(RFDETRDetector)
    w.classes = list(CLASSES)
    w.dims = INPUT_HW
    w.confidence_threshold = conf
    w.nms_threshold = 0.5
    w.session = _FakeSession(_det_outputs)
    w.input_name = "input"
    w.output_names = ["boxes", "logits"]
    w._orig_height = None
    w._orig_width = None
    return w


def _seg_wrapper(conf: float) -> RFDETRSegDetector:
    w = object.__new__(RFDETRSegDetector)
    w.classes = list(CLASSES)
    w.dims = INPUT_HW
    w.confidence_threshold = conf
    w.nms_threshold = 0.5
    w.mask_threshold = 0.0
    w.session = _FakeSession(_seg_outputs)
    w.input_name = "input"
    w.output_names = ["boxes", "logits", "masks"]
    w._orig_height = None
    w._orig_width = None
    return w


def _torch_engine(conf: float, module: Any, model_type: str) -> TorchEngine:
    torch = pytest.importorskip("torch")

    return TorchEngine(
        module=SimpleNamespace(model=module),
        family="rfdetr",
        device="cpu",
        dtype=torch.float32,
        checkpoint_path=Path("unused.pth"),
        model_type=model_type,
        architecture="RF-DETR Nano",
        classes=list(CLASSES),
        input_size=INPUT_HW,
        confidence_threshold=conf,
    )


def _image() -> np.ndarray:
    """A deterministic BGR image - content is irrelevant (both fakes replay fixed
    outputs) but it must be IDENTICAL for the preprocessing comparison."""
    rng = np.random.default_rng(4321)
    return rng.integers(0, 255, (IMAGE_HW[0], IMAGE_HW[1], 3), dtype=np.uint8)


def _onnx_infer(wrapper: Any, image: np.ndarray, model_type: str, conf: float) -> dict[str, Any]:
    """The canon path exactly as the inference service calls it."""
    from pictograph.inference._wrappers import dispatch

    return dispatch.infer_image(
        wrapper,
        image,
        model_type=model_type,
        architecture="RF-DETR Nano",
        classes=CLASSES,
        confidence=conf,
    )


def _strip_ids(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in p.items() if k != "id"} for p in preds]


# ───────────── the reduction itself ─────────────


class TestSharedReduction:
    """The free ``decode_*_outputs`` is the ONE reduction; the wrapper delegates."""

    @pytest.mark.parametrize("conf", [0.05, 0.3, 0.5, 0.9])
    def test_detection_postprocess_is_the_shared_decode(self, conf: float) -> None:
        wrapper = _det_wrapper(conf)
        via_method = wrapper.postprocess(_det_outputs(), IMAGE_HW)
        via_function = decode_detection_outputs(
            _det_outputs(),
            IMAGE_HW,
            classes=CLASSES,
            confidence_threshold=conf,
            nms_threshold=0.5,
        )
        assert len(via_method) == len(via_function) == 3
        for a, b in zip(via_method, via_function, strict=True):
            np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("conf", [0.05, 0.3, 0.5, 0.9])
    def test_segmentation_postprocess_is_the_shared_decode(self, conf: float) -> None:
        wrapper = _seg_wrapper(conf)
        wrapper._orig_height, wrapper._orig_width = IMAGE_HW
        via_method = wrapper.postprocess(_seg_outputs())
        via_function = decode_segmentation_outputs(
            _seg_outputs(),
            IMAGE_HW,
            classes=CLASSES,
            confidence_threshold=conf,
            nms_threshold=0.5,
            mask_threshold=0.0,
        )
        assert len(via_method) == len(via_function) == 4
        for a, b in zip(via_method, via_function, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_background_column_never_wins(self) -> None:
        """Query 0's BACKGROUND logit outscores every class; it must still be
        labelled by the best FOREGROUND class, not dropped and not called
        background."""
        boxes, scores, class_ids = decode_detection_outputs(
            _det_outputs(), IMAGE_HW, classes=CLASSES, confidence_threshold=0.05
        )
        assert len(class_ids) > 0
        assert {int(c) for c in class_ids} <= set(range(len(CLASSES)))
        # the 0.95 background probability must appear nowhere
        assert float(np.max(scores)) < 0.95

    def test_a_query_is_emitted_at_most_once(self) -> None:
        """The heart of it. A per-query argmax emits each query ONCE; the
        global (query x class) top-k this replaced emitted one query once per
        class it scored on, which is how 62 detections became 126."""
        n_queries = _boxes().shape[1]
        boxes, scores, class_ids = decode_detection_outputs(
            _det_outputs(), IMAGE_HW, classes=CLASSES, confidence_threshold=0.05
        )
        # One query per row at most - never one row per (query x class) pair,
        # which is what turned 62 real detections into 126.
        assert 0 < len(boxes) <= n_queries
        assert len(boxes) < n_queries * len(CLASSES)
        corners = {(round(float(b[0]), 3), round(float(b[1]), 3)) for b in boxes}
        assert len(corners) == len(boxes), "the same query came back more than once"

    def test_nms_actually_suppresses_on_this_fixture(self) -> None:
        """Guards the fixture itself.

        Queries 4 and 5 overlap query 0 on the same class so that NMS has real
        work to do. Without them, disabling NMS entirely changes nothing and
        every "both engines share the decode" assertion below passes over a
        divergent copy. If a future edit separates the boxes, this fails here
        rather than silently hollowing out the rest of the file.
        """
        with_nms = decode_detection_outputs(
            _det_outputs(), IMAGE_HW, classes=CLASSES, confidence_threshold=0.05
        )[0]
        without_nms = decode_detection_outputs(
            _det_outputs(),
            IMAGE_HW,
            classes=CLASSES,
            confidence_threshold=0.05,
            nms_threshold=1.01,
        )[0]
        assert len(without_nms) > len(with_nms), "NMS suppresses nothing on this fixture"


# ───────────── the two engines, end to end ─────────────


class TestBackendsAgree:
    @pytest.mark.parametrize("conf", [0.05, 0.3, 0.5])
    @pytest.mark.parametrize("as_tuple", [False, True])
    def test_detection_backends_emit_identical_annotations(
        self, conf: float, as_tuple: bool
    ) -> None:
        image = _image()
        wrapper = _det_wrapper(conf)
        module = _FakeModule(_det_outputs, as_tuple=as_tuple)
        engine = _torch_engine(conf, module, "object_detection")

        onnx_preds = _onnx_infer(wrapper, image, "object_detection", conf)
        torch_preds = engine.infer(image, confidence=conf)

        assert _strip_ids(torch_preds["predictions"]) == _strip_ids(onnx_preds["predictions"])
        assert onnx_preds["predictions"], "the fixture must produce something to compare"

    @pytest.mark.parametrize("conf", [0.05, 0.3, 0.5])
    @pytest.mark.parametrize("as_tuple", [False, True])
    def test_segmentation_backends_emit_identical_annotations(
        self, conf: float, as_tuple: bool
    ) -> None:
        image = _image()
        wrapper = _seg_wrapper(conf)
        module = _FakeModule(_seg_outputs, as_tuple=as_tuple, mask_key=True)
        engine = _torch_engine(conf, module, "instance_segmentation")

        onnx_preds = _onnx_infer(wrapper, image, "instance_segmentation", conf)
        torch_preds = engine.infer(image, confidence=conf)

        assert _strip_ids(torch_preds["predictions"]) == _strip_ids(onnx_preds["predictions"])
        assert onnx_preds["predictions"], "the fixture must produce something to compare"

    def test_both_engines_are_fed_the_same_tensor(self) -> None:
        """Not just the same decode - the same INPUT.

        The torch path used torchvision's antialiased bilinear where the canon
        uses cv2 ``INTER_LINEAR``; the two differ in edge handling and sample
        positioning, so the engines were reducing different pixels (RF-DETR was
        the last family carrying that residual).
        """
        image = _image()
        wrapper = _det_wrapper(0.05)
        module = _FakeModule(_det_outputs)
        engine = _torch_engine(0.05, module, "object_detection")

        _onnx_infer(wrapper, image, "object_detection", 0.05)
        engine.infer(image, confidence=0.05)

        onnx_tensor = np.asarray(wrapper.session.seen)
        torch_tensor = module.seen.cpu().numpy()
        assert onnx_tensor.shape == torch_tensor.shape
        np.testing.assert_array_equal(onnx_tensor, torch_tensor)

    def test_the_module_is_put_in_eval_mode(self) -> None:
        """rfdetr reassigns the module during training, so ``predict`` re-asserts
        ``eval()`` on every call. Skipping it runs dropout during inference."""
        module = _FakeModule(_det_outputs)
        engine = _torch_engine(0.05, module, "object_detection")
        engine.infer(_image(), confidence=0.05)
        assert module.eval_calls >= 1

    def test_a_change_to_the_shared_decode_moves_both_engines(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The point of sharing: there is no way to change one side alone.

        Raising the decode's gate must empty BOTH engines. If either kept its own
        copy of the reduction, one of them would still return annotations.
        """
        image = _image()
        wrapper = _det_wrapper(0.05)
        module = _FakeModule(_det_outputs)
        engine = _torch_engine(0.05, module, "object_detection")

        from pictograph.inference._wrappers import rfdetr_det_wrapper

        real = rfdetr_det_wrapper.decode_detection_outputs

        def gated(outputs: Any, shape: Any, **kw: Any) -> Any:
            kw["confidence_threshold"] = 0.999
            return real(outputs, shape, **kw)

        monkeypatch.setattr(rfdetr_det_wrapper, "decode_detection_outputs", gated)

        assert _onnx_infer(wrapper, image, "object_detection", 0.05)["predictions"] == []
        assert engine.infer(image, confidence=0.05)["predictions"] == []


# ───────────── the preprocess is shared too ─────────────


class TestSharedPreprocess:
    @pytest.mark.parametrize(
        ("wrapper_factory", "free_fn"),
        [
            (_det_wrapper, preprocess_detection_image),
            (_seg_wrapper, preprocess_segmentation_image),
        ],
    )
    def test_wrapper_preprocess_is_the_shared_function(
        self, wrapper_factory: Any, free_fn: Any
    ) -> None:
        image = _image()
        wrapper = wrapper_factory(0.05)
        np.testing.assert_array_equal(wrapper.preprocess(image), free_fn(image, INPUT_HW, True))

    def test_preprocess_stashes_the_original_shape(self) -> None:
        """The seg wrapper's ``postprocess`` reads ``_orig_*`` off the instance,
        so delegating ``preprocess`` must not stop recording them."""
        wrapper = _seg_wrapper(0.05)
        wrapper.preprocess(_image())
        assert (wrapper._orig_height, wrapper._orig_width) == IMAGE_HW
