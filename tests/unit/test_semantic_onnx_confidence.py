"""The ONNX semantic-segmentation path must emit a REAL per-region confidence.

THE GAP THIS PINS
-----------------
Both engines calibrate their raw semantic output to probabilities and gate on
it (2026-07-30), but only the TORCH engine ever handed the resulting probability
maps to the emitter::

    torch: semantic_masks_from_logits(full, conf) -> (masks, prob_maps)
           _semantic_seg_to_annotations(masks, classes, None, prob_maps)  # PRESENT
    onnx:  masks = wrapper.predict(img_bgr)
           _semantic_seg_to_annotations(masks, classes, class_filter)     # ABSENT

``_semantic_seg_to_annotations`` only computes a region-mean confidence when it
is GIVEN a probability map, so every ONNX semantic polygon silently fell back to
the pydantic default of 1.0 - reporting total certainty for a region the model
was, on the measured real model, about 20% sure of. The fix is
``SemanticSegmentationModelPyTorch.predict(..., return_probs=True)`` (opt-in, so
every other caller is untouched) plumbed through ``dispatch.infer_image``.

These tests run the REAL ``predict`` / ``postprocess`` / ``infer_image`` code
over a stubbed ONNX session - no weights, no network - so they fail if the
plumbing is removed, not merely if the emitter changes.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _inference_extra() -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("numpy")


# A 3-channel head (background + 2 classes) over a 64x64 image. The foreground
# block sits well inside the border because `mask_to_instance_polygons` zeroes
# the outermost ring before running connected components.
_SIZE = 64
_BLOCK = slice(16, 48)

# Deliberately MARGINAL evidence: softmax([0.0, 0.8, 0.0])[1] == 0.5267, i.e.
# "road" wins its argmax and clears the default 0.5 gate, but only just. That is
# the whole point - a correct pipeline reports ~0.53 here, and the broken one
# reported 1.0.
_BACKGROUND_LOGIT = 0.0
_ROAD_LOGIT = 0.8
_CAR_LOGIT = 0.0


def _expected_road_probability() -> float:
    import numpy as np

    logits = np.array([_BACKGROUND_LOGIT, _ROAD_LOGIT, _CAR_LOGIT])
    exp = np.exp(logits - logits.max())
    return float((exp / exp.sum())[1])


def _logits_chw():
    """(3, 64, 64) raw logits: one marginal "road" block, background elsewhere.

    The off-block values are NEGATIVE on purpose. ``semantic_probabilities``
    identifies logits by the presence of values outside [0, 1] (see
    ``semantic_calibration``'s docstring), so an all-in-[0, 1] fixture would be
    read as an already-calibrated probability map and this test would silently
    stop exercising the softmax path it exists to cover.
    """
    import numpy as np

    chw = np.zeros((3, _SIZE, _SIZE), dtype=np.float32)
    chw[0, :, :] = 2.0  # background wins everywhere by default
    chw[1, :, :] = -1.0
    chw[2, :, :] = -1.0
    chw[0, _BLOCK, _BLOCK] = _BACKGROUND_LOGIT
    chw[1, _BLOCK, _BLOCK] = _ROAD_LOGIT
    chw[2, _BLOCK, _BLOCK] = _CAR_LOGIT
    return chw


class _StubInput:
    name = "input"


class _StubSession:
    """Stands in for the ONNX session: returns fixed logits, shaped (1, C, H, W)
    exactly as a real semantic graph does."""

    def __init__(self, chw):
        import numpy as np

        self._output = np.expand_dims(chw, axis=0)

    def get_inputs(self):
        return [_StubInput()]

    def run(self, _output_names, _feeds):
        return [self._output]


def _wrapper(chw, classes=("road", "car")):
    """A real ``SemanticSegmentationModelPyTorch`` with a stubbed session, built
    without ``__init__`` so no ONNX file is needed."""
    from pictograph.inference._wrappers.sm_pytorch_wrapper import (
        SemanticSegmentationModelPyTorch,
    )

    wrapper = object.__new__(SemanticSegmentationModelPyTorch)
    wrapper.classes = list(classes)
    wrapper.dims = (_SIZE, _SIZE)
    wrapper.class_confidences = {}
    wrapper.confidence_threshold = 0.5
    wrapper.session = _StubSession(chw)
    return wrapper


def _image():
    import numpy as np

    return np.zeros((_SIZE, _SIZE, 3), dtype=np.uint8)


class TestOnnxSemanticPathSuppliesProbMaps:
    def test_infer_image_emits_a_real_confidence_not_the_1_0_default(self) -> None:
        """THE regression: every ONNX semantic polygon used to carry no
        ``confidence`` at all, so it deserialized to the pydantic default 1.0."""
        from pictograph.inference._wrappers import dispatch

        result = dispatch.infer_image(
            _wrapper(_logits_chw()),
            _image(),
            model_type="semantic_segmentation",
            architecture="unetplusplus",
            classes=["road", "car"],
        )

        predictions = result["predictions"]
        assert predictions, "the marginal block should still clear the 0.5 gate"
        assert all("confidence" in p for p in predictions), (
            "ONNX semantic polygons carry no confidence - infer_image is not "
            "passing prob_maps to _semantic_seg_to_annotations"
        )
        for prediction in predictions:
            assert prediction["confidence"] < 0.9, (
                "a region the model is ~53% sure of must not report near-certainty"
            )
            assert prediction["confidence"] == pytest.approx(_expected_road_probability(), abs=1e-4)

    def test_the_emitted_confidence_is_the_softmax_probability(self) -> None:
        """Pin the VALUE, not just its presence: it is the mean calibrated
        probability over the region's own pixels."""
        from pictograph.inference._wrappers import dispatch

        result = dispatch.infer_image(
            _wrapper(_logits_chw()),
            _image(),
            model_type="semantic_segmentation",
            architecture="unetplusplus",
            classes=["road", "car"],
        )

        assert result["predictions"][0]["name"] == "road"
        assert result["predictions"][0]["confidence"] == pytest.approx(0.52674, abs=1e-4)


class TestBothEnginesReportTheSameConfidence:
    """The parity claim, stated as a test: given the SAME raw logits, the ONNX
    engine (``infer_image`` -> wrapper) and the torch engine
    (``semantic_masks_from_logits`` -> emitter, exactly as
    ``_torch.TorchEngine._predict_smp`` does it) emit the same confidences."""

    def test_onnx_and_torch_confidences_agree(self) -> None:
        from pictograph.inference._wrappers import dispatch

        chw = _logits_chw()
        classes = ["road", "car"]

        onnx = dispatch.infer_image(
            _wrapper(chw),
            _image(),
            model_type="semantic_segmentation",
            architecture="unetplusplus",
            classes=classes,
        )["predictions"]

        masks, prob_maps = dispatch.semantic_masks_from_logits(chw, 0.5)
        torch = dispatch._semantic_seg_to_annotations(masks, classes, None, prob_maps)

        assert len(onnx) == len(torch) == 1
        assert [p["name"] for p in onnx] == [p["name"] for p in torch]
        assert onnx[0]["confidence"] == pytest.approx(torch[0]["confidence"], abs=1e-6)


class TestReturnProbsIsOptIn:
    """Backward compatibility is the constraint that shaped the fix: the other
    production callers (``run_job``'s warmup, the per-deployment app's warmup)
    call ``predict(img)`` positionally and must keep getting masks alone."""

    def test_predict_without_the_flag_returns_masks_only(self) -> None:
        import numpy as np

        masks = _wrapper(_logits_chw()).predict(_image())

        assert isinstance(masks, np.ndarray), (
            "predict() must still return the bare mask stack by default"
        )
        assert masks.shape == (2, _SIZE, _SIZE)

    def test_predict_with_the_flag_returns_masks_and_probs(self) -> None:
        import numpy as np

        masks, probs = _wrapper(_logits_chw()).predict(_image(), return_probs=True)

        assert masks.shape == (2, _SIZE, _SIZE)
        assert len(probs) == len(masks), "prob maps are aligned 1:1 with masks"
        for prob in probs:
            assert prob.shape == (_SIZE, _SIZE)
            assert prob.min() >= 0.0 and prob.max() <= 1.0, "probs, not logits"
        # The road channel's probability inside the block is the marginal value.
        assert float(np.asarray(probs[0])[_BLOCK, _BLOCK].mean()) == pytest.approx(
            _expected_road_probability(), abs=1e-5
        )

    def test_return_probs_without_postprocess_is_rejected(self) -> None:
        """There are no probability maps without the calibrate+gate step, so the
        combination raises rather than returning a differently-shaped result a
        caller would unpack as (masks, probs) and get raw logits from."""
        with pytest.raises(ValueError, match="postprocess"):
            _wrapper(_logits_chw()).predict(_image(), postprocess=False, return_probs=True)

    def test_postprocess_without_the_flag_returns_masks_only(self) -> None:
        """`postprocess` is called directly by the cross-engine parity tests in
        test_shared_emitters.py - its default return shape is load-bearing."""
        import numpy as np

        chw = _logits_chw()
        masks = _wrapper(chw).postprocess(np.transpose(chw, (1, 2, 0)))

        assert isinstance(masks, np.ndarray)
        assert masks.shape == (2, _SIZE, _SIZE)

    def test_single_channel_head_also_supplies_a_prob_map(self) -> None:
        """The (H, W) single-class branch of ``postprocess`` returns its map too,
        so a single-class Segformer/Unet gets a real confidence as well."""
        import numpy as np

        hw = np.full((_SIZE, _SIZE), -4.0, dtype=np.float32)
        hw[_BLOCK, _BLOCK] = 0.25  # sigmoid(0.25) == 0.5622 - marginal, clears 0.5
        wrapper = _wrapper(hw[None, :, :], classes=("only",))

        masks, probs = wrapper.predict(_image(), return_probs=True)

        assert masks.shape == (1, _SIZE, _SIZE)
        assert len(probs) == 1
        assert float(np.asarray(probs[0])[_BLOCK, _BLOCK].mean()) == pytest.approx(
            0.562177, abs=1e-5
        )
