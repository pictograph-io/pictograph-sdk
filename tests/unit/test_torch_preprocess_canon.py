"""The torch path's resizes must be the SERVER-SIDE CANON's, not PIL's.

`_torch.py` fed `PIL.Image.Resampling.BILINEAR` to three helpers whose own
docstrings claimed cv2 ("matching the ONNX wrapper's cv2 call", "matching
training"). PIL's BILINEAR is support-scaled - it widens the kernel and
antialiases on a downscale; cv2's INTER_LINEAR samples a fixed 2x2 neighbourhood
and aliases. Real inputs ARE downscales, so the `pytorch` runtime reproduced
neither the training preprocessing (`ClassificationDataset` decodes with cv2,
`get_validation_augmentation` resizes with albumentations' cv2 INTER_LINEAR) nor
its own sibling backends (every graph runtime serves through the cv2 wrappers).

Measured on a real 2-class resnet18 before the fix - onnxruntime/CPU vs
pytorch/CPU, max |Δp| on the softmax vector:

    already-224x224 input ....... 0.000000    (no resize -> no divergence)
    every other input ........... 0.0028 - 0.019827

against `FP16_CLS_PROB_ATOL = 2e-2`. After: 8.9e-08. The exact zero at native
size is what identifies the resize as the sole cause - and is also why every
pre-existing parity test missed this, because they all feed native size.

So these tests deliberately feed NON-NATIVE sizes. A native-size-only test here
would pass against the bug it exists to catch.
"""

from __future__ import annotations

import pytest

# numpy ships in the [inference] extra, not the base install. Importing it
# directly RAISES on a base venv, and a raise during collection interrupts the
# WHOLE run - so `pytest`, the SDK's documented gate, collected 2788 tests and
# then aborted on all of them. Skip like every sibling module here already does.
np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("PIL")

from pictograph.inference._torch import (  # noqa: E402
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    _bgr_to_pil,
    _normalized_chw,
    _resize_bgr,
    _resize_channel,
)

# Every case is a real downscale/upscale, never the identity. 224 is the
# classification input; the sources are shaped like actual photographs.
_NON_NATIVE = [
    ((480, 640), (224, 224)),  # a 4:3 photo into a classifier - the common case
    ((613, 997), (224, 224)),  # awkward, non-integer ratio both axes
    ((259, 321), (224, 224)),  # small source, mild downscale
    ((224, 224), (384, 384)),  # upscale
]


def _rng_bgr(h: int, w: int) -> np.ndarray:
    # Structured noise, not flat: a constant image resizes identically under
    # every filter and would make these tests vacuous.
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _pil_bilinear_chw(pil, width: int, height: int) -> np.ndarray:
    """What the pre-fix implementation did - the thing that must NOT come back."""
    from PIL import Image

    arr = (
        np.asarray(pil.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    )
    mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
    std = np.array(_IMAGENET_STD, dtype=np.float32)
    return ((arr - mean) / std).astype(np.float32).transpose(2, 0, 1)


class TestNormalizedChwIsCv2:
    """`_normalized_chw` - the classification + semantic-seg INPUT preprocess."""

    @pytest.mark.parametrize("src,dst", _NON_NATIVE)
    def test_matches_the_wrapper_preprocess_exactly(self, src, dst):
        """Byte-identical to the canonical `classifier_wrapper.preprocess`."""
        h, w = src
        out_h, out_w = dst
        bgr = _rng_bgr(h, w)

        got = _normalized_chw(_bgr_to_pil(bgr), out_w, out_h)

        # the shipped wrapper's chain, inlined
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        rgb = rgb.astype(np.float32) / 255.0
        expected = (
            (
                (rgb - np.array(_IMAGENET_MEAN, dtype=np.float32))
                / np.array(_IMAGENET_STD, dtype=np.float32)
            )
            .astype(np.float32)
            .transpose(2, 0, 1)
        )

        assert got.shape == (3, out_h, out_w)
        np.testing.assert_array_equal(got, expected)

    @pytest.mark.parametrize("src,dst", _NON_NATIVE)
    def test_is_measurably_not_pil(self, src, dst):
        """The bug is only visible off-native, so prove the two really differ here.

        Without this, `test_matches_the_wrapper_preprocess_exactly` could be
        satisfied by a PIL implementation on any size where the filters happen
        to agree.
        """
        h, w = src
        out_h, out_w = dst
        pil = _bgr_to_pil(_rng_bgr(h, w))
        assert (
            np.abs(_normalized_chw(pil, out_w, out_h) - _pil_bilinear_chw(pil, out_w, out_h)).max()
            > 1e-3
        )

    def test_native_size_is_identity(self):
        """The control: at native size there is no resize, so nothing can diverge -
        which is exactly why native-size-only parity tests missed this."""
        bgr = _rng_bgr(224, 224)
        pil = _bgr_to_pil(bgr)
        np.testing.assert_array_equal(
            _normalized_chw(pil, 224, 224), _pil_bilinear_chw(pil, 224, 224)
        )


class TestResizeBgrIsCv2:
    """`_resize_bgr` - the YOLOX letterbox INPUT resize."""

    @pytest.mark.parametrize("src,dst", _NON_NATIVE)
    def test_matches_cv2_inter_linear(self, src, dst):
        h, w = src
        out_h, out_w = dst
        bgr = _rng_bgr(h, w)
        expected = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        got = _resize_bgr(bgr, out_w, out_h)
        assert got.dtype == np.uint8
        np.testing.assert_array_equal(got, expected)


class TestResizeChannelIsCv2:
    """`_resize_channel` - the semantic-seg OUTPUT logit resize back to source size."""

    @pytest.mark.parametrize("src,dst", _NON_NATIVE)
    def test_matches_cv2_inter_linear(self, src, dst):
        h, w = src
        out_h, out_w = dst
        rng = np.random.default_rng(1)
        channel = rng.standard_normal((h, w), dtype=np.float32)
        expected = cv2.resize(channel, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        got = _resize_channel(channel, out_w, out_h)
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got, expected)
