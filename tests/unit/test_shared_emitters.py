"""Unit tests for the shared dispatch emitters in
`pictograph.inference._wrappers.dispatch` -- `_classification_to_result`,
`semantic_masks_from_logits`, and `_semantic_seg_to_annotations` -- plus the
logit-vs-probability calibration underneath them
(`_wrappers.semantic_calibration`).

These are the functions BOTH engines call (the ONNX wrappers directly, the
torch engine via the same module -- see `_torch.py::TorchEngine._predict_smp`
/ `_classification_result`), so a difference between backends would be a bug
these tests catch rather than an "expected variation."

Importing `dispatch` pulls in the whole `_wrappers` package (every ONNX
wrapper needs cv2 + onnxruntime + numpy), so the whole module needs the full
`[inference]` extra -- exactly like `test_pytorch_keypoint.py`.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _inference_extra() -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("numpy")


def _blob_mask():
    """A 16x16 binary mask with one solid 8x8 component, clear of the border
    (`_clear_edges` zeroes the outermost ring) and well above the instance
    emitter's default `min_area_ratio` floor."""
    import numpy as np

    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1
    return mask


class TestClassificationToResult:
    """`_classification_to_result`'s rank-1-is-always-kept, ranks-2..k-pruned
    semantic is deliberate and load-bearing (see the function's own
    docstring): a classifier's answer is "which class", so returning nothing
    for a low-confidence image is a worse answer than a real class with a low
    score the caller can inspect. This is what makes `ClassificationResult.top`
    non-optional."""

    def test_top_rank_is_kept_even_when_its_own_score_is_below_conf(self) -> None:
        import numpy as np

        from pictograph.inference._wrappers import dispatch

        classes = ["cat", "dog", "bird"]
        logits = np.array([1.0, 0.5, -0.5])
        probs = dispatch._softmax(logits)
        conf = float(probs[0]) + 0.01  # strictly above every class's own score
        result = dispatch._classification_to_result(logits, classes, conf, 3, "classification")

        assert result["model_type"] == "classification"
        assert len(result["predictions"]) == 1
        assert result["predictions"][0]["class"] == "cat"
        assert result["predictions"][0]["confidence"] == pytest.approx(float(probs[0]))
        assert result["tags"] == ["cat"]

    def test_lower_ranks_are_pruned_below_threshold_and_kept_above(self) -> None:
        import numpy as np

        from pictograph.inference._wrappers import dispatch

        classes = ["cat", "dog", "bird"]
        logits = np.array([2.0, 1.9, -5.0])  # cat/dog close together, bird far behind
        probs = dispatch._softmax(logits)
        conf = float(probs[2]) + 0.01  # clears bird's score, not cat's or dog's
        result = dispatch._classification_to_result(logits, classes, conf, 3, "classification")

        assert [p["class"] for p in result["predictions"]] == ["cat", "dog"]

    def test_top_k_limits_how_many_ranks_are_even_considered(self) -> None:
        import numpy as np

        from pictograph.inference._wrappers import dispatch

        classes = ["cat", "dog", "bird"]
        logits = np.array([2.0, 1.9, 1.8])  # all close, all would clear conf=0
        result = dispatch._classification_to_result(logits, classes, 0.0, 1, "classification")
        assert [p["class"] for p in result["predictions"]] == ["cat"]

    def test_class_filter_can_drop_the_top_rank_entirely(self) -> None:
        """The "always keep rank 1" rule does not resurrect a class the
        explicit class_filter removed -- the filter is checked first."""
        import numpy as np

        from pictograph.inference._wrappers import dispatch

        classes = ["cat", "dog"]
        logits = np.array([1.0, 0.0])  # cat is rank 1
        result = dispatch._classification_to_result(
            logits, classes, 0.0, 2, "classification", class_filter=["dog"]
        )
        assert [p["class"] for p in result["predictions"]] == ["dog"]

    def test_out_of_range_rank_index_is_skipped_not_an_error(self) -> None:
        """top_k larger than the class list must not index classes[] out of
        range -- extra ranks beyond len(classes) are silently dropped."""
        import numpy as np

        from pictograph.inference._wrappers import dispatch

        classes = ["only"]
        logits = np.array([1.0, 2.0, 3.0])
        result = dispatch._classification_to_result(logits, classes, 0.0, 3, "classification")
        assert [p["class"] for p in result["predictions"]] == ["only"]

    def test_model_type_is_echoed_back(self) -> None:
        import numpy as np

        from pictograph.inference._wrappers import dispatch

        result = dispatch._classification_to_result(
            np.array([1.0]), ["only"], 0.0, 1, "classification"
        )
        assert result["model_type"] == "classification"


class TestSemanticCalibration:
    """`semantic_probabilities` -- the ONE place "is this output a logit or a
    probability?" is answered, for both engines.

    The defect it fixes: `create_model` passes `activation` ONLY to smp.Unet /
    smp.UnetPlusPlus, and only for a SINGLE-class head. Every multi-class model
    and every Segformer is built with no activation and emits raw unbounded
    logits -- which both postprocess paths then compared against a
    PROBABILITY-scale confidence (0.5). Measured on the real 81-channel
    UnetPlusPlus model 56dc0d04: output range [-3.62, +3.44], 54% of values
    negative, per-pixel channel sums from -16.9 to +6.8 (a softmax sums to 1.0).
    """

    def test_multi_channel_logits_are_softmaxed(self) -> None:
        import numpy as np

        from pictograph.inference._wrappers.semantic_calibration import (
            semantic_probabilities,
        )

        logits = np.array([[-2.0, 1.0], [3.0, -1.0], [0.0, 0.5]])  # (C=3, N=2)
        probs = semantic_probabilities(logits, channel_axis=0)

        np.testing.assert_allclose(probs.sum(axis=0), [1.0, 1.0], atol=1e-6)
        assert probs.min() >= 0.0 and probs.max() <= 1.0
        # Monotonic: the winning class per pixel is unchanged. This is why the
        # defect was invisible for multi-class -- only the GATE was wrong.
        np.testing.assert_array_equal(np.argmax(probs, axis=0), np.argmax(logits, axis=0))

    def test_a_2d_stack_is_not_mistaken_for_a_single_channel_map(self) -> None:
        """A `(C, N)` stack and an `(H, W)` map are both 2-D and need OPPOSITE
        treatment, so the channel axis is stated by the caller, never inferred
        from `ndim`. Getting this wrong would softmax across image rows."""
        import numpy as np

        from pictograph.inference._wrappers.semantic_calibration import (
            semantic_probabilities,
        )

        logits = np.array([[-2.0, 1.0], [3.0, -1.0], [0.0, 0.5]])
        stacked = semantic_probabilities(logits, channel_axis=0)
        as_one_map = semantic_probabilities(logits, channel_axis=None)

        np.testing.assert_allclose(stacked.sum(axis=0), [1.0, 1.0], atol=1e-6)
        # The sigmoid reading does NOT normalise across the first axis.
        assert not np.allclose(as_one_map.sum(axis=0), 1.0)

    def test_single_channel_logits_are_sigmoided(self) -> None:
        import numpy as np

        from pictograph.inference._wrappers.semantic_calibration import (
            semantic_probabilities,
        )

        logits = np.array([[-2.0, 0.0, 2.0]])  # (C=1, N=3)
        probs = semantic_probabilities(logits, channel_axis=0)

        np.testing.assert_allclose(probs, [[0.1192029, 0.5, 0.8807971]], atol=1e-6)

    def test_an_already_probability_scaled_map_is_returned_unchanged(self) -> None:
        """The inverse error is the dangerous one: sigmoid over an already
        sigmoid-ed map compresses [0, 1] into [0.50, 0.73], so a 0.5 gate would
        pass the ENTIRE image."""
        import numpy as np

        from pictograph.inference._wrappers.semantic_calibration import (
            semantic_probabilities,
        )

        probs_in = np.array([[0.02, 0.5, 0.99]])
        np.testing.assert_array_equal(semantic_probabilities(probs_in, channel_axis=0), probs_in)

    def test_per_channel_sigmoid_output_is_not_re_softmaxed(self) -> None:
        """Multi-channel values inside [0, 1] that do NOT sum to 1 are a
        per-channel sigmoid head, not logits -- softmaxing them would rescale a
        correct probability map."""
        import numpy as np

        from pictograph.inference._wrappers.semantic_calibration import (
            semantic_probabilities,
        )

        sigmoided = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
        np.testing.assert_array_equal(semantic_probabilities(sigmoided, channel_axis=0), sigmoided)

    def test_float64_input_is_not_narrowed(self) -> None:
        import numpy as np

        from pictograph.inference._wrappers.semantic_calibration import (
            semantic_probabilities,
        )

        probs_in = np.array([[0.1, 0.9]], dtype=np.float64)
        out = semantic_probabilities(probs_in, channel_axis=0)
        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, probs_in)

    def test_is_probability_scaled_separates_the_two_regimes(self) -> None:
        import numpy as np

        from pictograph.inference._wrappers.semantic_calibration import (
            is_probability_scaled,
        )

        assert is_probability_scaled(np.array([0.0, 0.5, 1.0]))
        assert not is_probability_scaled(np.array([-0.01, 0.5]))
        assert not is_probability_scaled(np.array([0.5, 1.01]))
        # Float slack, so an ONNX graph's float32 round-trip of 1.0 still reads
        # as a probability.
        assert is_probability_scaled(np.array([1.0 + 1e-7]))
        assert is_probability_scaled(np.array([]))

    def test_large_magnitude_logits_do_not_overflow(self) -> None:
        import numpy as np

        from pictograph.inference._wrappers.semantic_calibration import (
            semantic_probabilities,
        )

        extreme = np.array([[-800.0, 800.0]])
        single = semantic_probabilities(extreme, channel_axis=None)
        assert np.isfinite(single).all()
        np.testing.assert_allclose(single, [[0.0, 1.0]], atol=1e-9)

        multi = semantic_probabilities(np.array([[-800.0], [800.0]]), channel_axis=0)
        assert np.isfinite(multi).all()
        np.testing.assert_allclose(multi.sum(axis=0), [1.0], atol=1e-9)


class TestSemanticMasksFromLogits:
    """Mirrors `SemanticSegmentationModelPyTorch.postprocess` exactly, per the
    function's own docstring, so both engines threshold identically."""

    def test_single_channel_segformer_logits_are_gated_as_probabilities(self) -> None:
        """The straightforwardly-wrong case: a single-class Segformer carries NO
        baked sigmoid, so a logit of 0.0 IS probability 0.5. Gating the raw logit
        at 0.5 cut the mask at sigmoid(0.5) == 0.62 instead."""
        import numpy as np

        from pictograph.inference._wrappers import dispatch

        full = np.array([[[-2.0, 0.0], [2.0, -0.5]]])  # (1, 2, 2) raw logits
        masks, probs = dispatch.semantic_masks_from_logits(full, conf=0.5)

        # sigmoid -> [[0.119, 0.500], [0.881, 0.378]]; the 0.0 logit is exactly
        # at the threshold and belongs IN the mask. The old raw gate excluded it.
        np.testing.assert_array_equal(masks[0], np.array([[0, 1], [1, 0]], dtype=np.uint8))
        # ...and what the emitter reports as confidence is a probability now.
        np.testing.assert_allclose(probs[0], [[0.1192029, 0.5], [0.8807971, 0.3775407]], atol=1e-6)

    def test_multi_channel_logits_are_softmaxed_before_the_per_channel_gate(self) -> None:
        """Raw logits big enough to clear a naive `>= 0.5` comparison, whose
        actual probabilities do not."""
        import numpy as np

        from pictograph.inference._wrappers import dispatch

        background = np.array([2.0, 0.0])
        road = np.array([1.9, 3.0])  # pixel 0: > 0.5 as a logit, ~0.24 as a probability
        car = np.array([1.8, -1.0])
        full = np.stack([background, road, car], axis=0)

        masks, probs = dispatch.semantic_masks_from_logits(full, conf=0.5)

        road_mask, car_mask = masks
        # pixel 0: background wins the argmax anyway, and road's own probability
        # (~0.30) is nowhere near the gate -- the raw value 1.9 would have been.
        assert road_mask[0] == 0 and car_mask[0] == 0
        # pixel 1: road wins AND its real probability (~0.94) clears the gate.
        assert road_mask[1] == 1 and car_mask[1] == 0
        assert probs[0].max() <= 1.0 and probs[0].min() >= 0.0

    def test_the_argmax_is_untouched_by_calibration(self) -> None:
        """Softmax is monotonic, so the per-pixel class assignment is identical
        before and after -- the fix moves the GATE, not the labels."""
        import numpy as np

        from pictograph.inference._wrappers import dispatch

        rng = np.random.default_rng(0)
        full = rng.normal(size=(5, 32, 32)) * 3.0
        masks, _probs = dispatch.semantic_masks_from_logits(full, conf=0.0)

        assigned = np.argmax(full, axis=0)
        for class_idx, mask in enumerate(masks, start=1):
            np.testing.assert_array_equal(mask, (assigned == class_idx).astype(np.uint8))

    def test_a_probability_input_is_thresholded_unchanged(self) -> None:
        """A single-class Unet/UnetPlusPlus DOES carry a baked sigmoid, so its
        output must pass through the calibration untouched."""
        import numpy as np

        from pictograph.inference._wrappers import dispatch

        full = np.array([[[0.9, 0.2], [0.4, 0.6]]])
        masks, probs = dispatch.semantic_masks_from_logits(full, conf=0.5)

        np.testing.assert_array_equal(masks[0], np.array([[1, 0], [0, 1]], dtype=np.uint8))
        np.testing.assert_array_equal(probs[0], full[0])

    def test_single_channel_is_a_binary_gate_against_conf(self) -> None:
        import numpy as np

        from pictograph.inference._wrappers import dispatch

        full = np.array([[[0.9, 0.2], [0.4, 0.6]]])  # shape (1, 2, 2)
        masks, probs = dispatch.semantic_masks_from_logits(full, conf=0.5)

        assert len(masks) == 1
        np.testing.assert_array_equal(masks[0], np.array([[1, 0], [0, 1]], dtype=np.uint8))
        assert masks[0].dtype == np.uint8
        np.testing.assert_array_equal(probs[0], full[0])

    def test_multi_channel_argmax_with_background_and_per_channel_gate(self) -> None:
        import numpy as np

        from pictograph.inference._wrappers import dispatch

        # channel 0 = background, channel 1 = road, channel 2 = car.
        background = np.array([0.1, 0.1, 0.9, 0.2])
        road = np.array([0.6, 0.3, 0.05, 0.45])
        car = np.array([0.3, 0.6, 0.05, 0.35])
        full = np.stack([background, road, car], axis=0)

        masks, probs = dispatch.semantic_masks_from_logits(full, conf=0.5)

        # One mask per NON-background channel -- background never gets its own.
        assert len(masks) == 2
        road_mask, car_mask = masks

        # pixel 0: road wins the argmax (0.6) and clears the gate.
        assert road_mask[0] == 1 and car_mask[0] == 0
        # pixel 1: car wins the argmax (0.6) and clears the gate.
        assert car_mask[1] == 1 and road_mask[1] == 0
        # pixel 2: background wins the argmax -- neither foreground class fires.
        assert road_mask[2] == 0 and car_mask[2] == 0
        # pixel 3: road wins the argmax (0.45) but ITS OWN value is below conf --
        # the per-channel gate applies even to the argmax winner.
        assert road_mask[3] == 0 and car_mask[3] == 0

        np.testing.assert_array_equal(probs[0], road)
        np.testing.assert_array_equal(probs[1], car)

    def test_masks_are_uint8_binary(self) -> None:
        import numpy as np

        from pictograph.inference._wrappers import dispatch

        full = np.stack([np.array([0.9, 0.1]), np.array([0.1, 0.9])], axis=0)
        masks, _probs = dispatch.semantic_masks_from_logits(full, conf=0.5)
        assert masks[0].dtype == np.uint8
        assert set(np.unique(masks[0]).tolist()) <= {0, 1}


class TestBothEnginesThresholdIdentically:
    """`dispatch.semantic_masks_from_logits` (torch engine) and
    `SemanticSegmentationModelPyTorch.postprocess` (ONNX engine) are two
    implementations of one rule. They now share `semantic_probabilities`, and
    this is the test that says so in both directions."""

    @staticmethod
    def _wrapper(classes: list[str]):
        """The wrapper WITHOUT its ONNX session -- `postprocess` is pure."""
        from pictograph.inference._wrappers.sm_pytorch_wrapper import (
            SemanticSegmentationModelPyTorch,
        )

        wrapper = object.__new__(SemanticSegmentationModelPyTorch)
        wrapper.classes = classes
        wrapper.class_confidences = {}
        wrapper.confidence_threshold = 0.5
        return wrapper

    def test_multi_class_logits_agree_between_the_engines(self) -> None:
        import numpy as np

        from pictograph.inference._wrappers import dispatch

        rng = np.random.default_rng(5)
        chw = rng.normal(size=(4, 24, 24)) * 3.0  # (C, H, W) raw logits

        torch_masks, _probs = dispatch.semantic_masks_from_logits(chw, conf=0.5)
        onnx_masks = self._wrapper(["a", "b", "c"]).postprocess(np.transpose(chw, (1, 2, 0)))

        np.testing.assert_array_equal(np.stack(torch_masks), onnx_masks)

    def test_single_class_logits_agree_between_the_engines(self) -> None:
        import numpy as np

        from pictograph.inference._wrappers import dispatch

        rng = np.random.default_rng(6)
        hw = rng.normal(size=(24, 24)) * 3.0  # a single-class Segformer's map

        torch_masks, _probs = dispatch.semantic_masks_from_logits(hw[None, :, :], conf=0.5)
        onnx_masks = self._wrapper(["only"]).postprocess(hw)

        np.testing.assert_array_equal(np.stack(torch_masks), onnx_masks)
        # ...and it is genuinely the sigmoid gate, not the raw one.
        assert not np.array_equal(onnx_masks[0], (hw >= 0.5).astype(np.uint8))


class TestSemanticSegToAnnotations:
    def test_emits_polygon_annotations_with_auto_annotate_attribute(self) -> None:
        from pictograph.inference._wrappers import dispatch

        anns = dispatch._semantic_seg_to_annotations([_blob_mask()], ["road"], None)
        assert len(anns) == 1
        assert anns[0]["name"] == "road"
        assert anns[0]["type"] == "polygon"
        assert anns[0]["attributes"] == ["auto-annotate"]
        assert "id" in anns[0] and "bounding_box" in anns[0] and "polygon" in anns[0]

    def test_no_prob_maps_means_no_confidence_key(self) -> None:
        from pictograph.inference._wrappers import dispatch

        anns = dispatch._semantic_seg_to_annotations([_blob_mask()], ["road"], None)
        assert "confidence" not in anns[0]

    def test_prob_maps_give_a_real_mean_confidence_over_the_regions_own_pixels(self) -> None:
        """Without `prob_maps` every semantic polygon used to report a
        hardcoded 1.0 regardless of how marginal the evidence was -- this pins
        that the confidence is the MEAN over the region's own pixels, not a
        blend with the rest of the (very different-valued) probability map."""
        import numpy as np

        from pictograph.inference._wrappers import dispatch

        mask = _blob_mask()
        prob_map = np.full((16, 16), 0.3, dtype=np.float32)
        prob_map[4:12, 4:12] = 0.9  # the region's own pixels
        anns = dispatch._semantic_seg_to_annotations([mask], ["road"], None, prob_maps=[prob_map])

        assert len(anns) == 1
        # Exactly 0.9 (the region's own value), not ~0.45 (a blend across the
        # whole 16x16 map) -- proves the mean is scoped to the mask's pixels.
        assert anns[0]["confidence"] == pytest.approx(0.9, abs=1e-6)

    def test_class_index_out_of_range_is_skipped(self) -> None:
        from pictograph.inference._wrappers import dispatch

        # Two masks but only one class name -- the second (index 1) is out of
        # range and must be skipped rather than raise an IndexError.
        anns = dispatch._semantic_seg_to_annotations([_blob_mask(), _blob_mask()], ["road"], None)
        assert {a["name"] for a in anns} == {"road"}

    def test_class_filter_drops_unfiltered_classes(self) -> None:
        from pictograph.inference._wrappers import dispatch

        anns = dispatch._semantic_seg_to_annotations(
            [_blob_mask(), _blob_mask()], ["road", "car"], ["car"]
        )
        assert {a["name"] for a in anns} == {"car"}

    def test_empty_mask_yields_no_annotations(self) -> None:
        import numpy as np

        from pictograph.inference._wrappers import dispatch

        empty = np.zeros((16, 16), dtype=np.uint8)
        anns = dispatch._semantic_seg_to_annotations([empty], ["road"], None)
        assert anns == []
