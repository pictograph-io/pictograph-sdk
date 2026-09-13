"""Calibrating a semantic-segmentation model's RAW output to probabilities.

THE DEFECT THIS EXISTS TO FIX
-----------------------------
``pipelines/sm_pytorch/train_semantic_seg.py::create_model`` bakes an activation
into the graph for exactly ONE case::

    n_classes = 1 if len(classes) == 1 else len(classes) + 1
    activation = model_config.get("activation", "sigmoid" if n_classes == 1 else None)
    # ...and `activation` is passed ONLY to smp.Unet / smp.UnetPlusPlus.
    # smp.Segformer is constructed WITHOUT it.

So a SINGLE-class Unet/UnetPlusPlus emits probabilities, and **everything else -
every multi-class model of any architecture, and every Segformer - emits raw,
unbounded logits.**

Both postprocess paths used to threshold that raw output directly against a
PROBABILITY-scale confidence (default 0.5):

    ``SemanticSegmentationModelPyTorch.postprocess``   (the ONNX engine)
    ``dispatch.semantic_masks_from_logits``            (the torch engine)

For a multi-class model the argmax is unaffected - softmax is monotonic - but
the per-channel gate was comparing a logit against 0.5, which is not a
probability at all. For a single-class Segformer it is straightforwardly wrong:
a logit of 0.0 IS probability 0.5, so the mask was cut at sigmoid(0.5) ≈ 0.62.

MEASURED, not assumed (2026-07-30, model 56dc0d04 - an 81-channel
UnetPlusPlus/resnet34 semantic model, its real published ONNX and its real
``.pth`` both run on CPU):

===============================  =========================  =====================
                                 ONNX graph                 torch module
===============================  =========================  =====================
output range                     [-3.0991, +3.0934]         [-3.0156, +3.1279]
fraction of values below zero    53.6%                      53.6%
per-pixel sum over channels      -3.00  (softmax → 1.0)     -3.02
===============================  =========================  =====================

i.e. both engines were gating raw logits. Applying the softmax first moved the
same image from 9.1% of (pixel, class) cells clearing a 0.5 gate to 0.0% - the
model is genuinely unconfident on that input, which the raw gate hid.

HOW THE ACTIVATION IS RECOVERED (self-describing, not an architecture table)
---------------------------------------------------------------------------
The tensor says which it is. A probability map is bounded to [0, 1] by
construction - sigmoid and softmax cannot leave it - while a logit map from a
trained model essentially always contains negatives (measured: 53.6% of values).
So: **within [0, 1] everywhere ⇒ already probability-scaled; otherwise ⇒
logits.** That covers softmax outputs, per-channel sigmoid outputs, and
externally-supplied ONNX graphs alike, with no list of architecture names to
keep in lockstep with anything.

KNOWN FAILURE MODE, stated plainly: a raw-logit map whose values ALL happen to
land inside [0, 1] is indistinguishable from a probability map, and is read as
one. That means a maximally-unconfident model on one particular image (every
logit in [0, 1] ⇒ every true probability in [0.50, 0.73]) is gated at 0.5 on the
logit instead, which emits FEWER pixels than it should. The failure is
one-directional and it is the safe direction: it under-emits rather than
flooding the image with false regions, which is what the inverse error - running
sigmoid over an already-sigmoid map, compressing [0, 1] to [0.50, 0.73] - would
do to every single-class model at the default threshold.
"""

# Slack on the [0, 1] bound, for float32 round-tripping through an ONNX graph and
# through the bilinear upscale both engines run before postprocessing. Bilinear
# interpolation is a convex combination, so it cannot push a probability map out
# of [0, 1] - this is purely for representation error.
PROBABILITY_BOUND_EPS = 1e-4


def _as_float(raw):
    """The input as a float array, preserving float64 rather than narrowing it.

    Narrowing to float32 here would change the VALUES a caller passed in (a
    probability map that is already float64 would come back subtly different),
    so only a non-float input is promoted.
    """
    import numpy as np

    arr = np.asarray(raw)
    return arr if arr.dtype.kind == "f" else arr.astype(np.float32)


def is_probability_scaled(arr) -> bool:
    """True when every value lies within [0, 1] (± float slack).

    The whole detection: a sigmoid or softmax output cannot leave the unit
    interval, and a trained model's raw logits reliably do. See the module
    docstring for the measurement and for the one case this cannot separate.
    """
    if arr.size == 0:
        return True
    return bool(arr.min() >= -PROBABILITY_BOUND_EPS and arr.max() <= 1.0 + PROBABILITY_BOUND_EPS)


def _sigmoid(arr):
    """Numerically stable logistic - no ``exp`` overflow on large-magnitude logits."""
    import numpy as np

    positive = arr >= 0
    out = np.empty_like(arr)
    out[positive] = 1.0 / (1.0 + np.exp(-arr[positive]))
    exp_negative = np.exp(arr[~positive])
    out[~positive] = exp_negative / (1.0 + exp_negative)
    return out


def _softmax_along(arr, axis: int):
    """Numerically stable softmax over one axis (max-shifted)."""
    import numpy as np

    shifted = np.exp(arr - np.max(arr, axis=axis, keepdims=True))
    return shifted / np.sum(shifted, axis=axis, keepdims=True)


def semantic_probabilities(raw, channel_axis: "int | None" = 0):
    """Raw semantic-segmentation output -> per-class probabilities, same shape.

    THE single place the logit-vs-probability question is answered, so the ONNX
    wrapper and the torch engine cannot drift apart on it. Callers threshold the
    RESULT against their probability-scale confidence.

    Args:
        raw: The model's own output for one image - ``(C, H, W)``, ``(H, W, C)``
            or ``(H, W)``, at whatever resolution the caller has already resized
            it to. (Both engines upscale the raw channels BEFORE calibrating;
            that ordering is deliberate and shared, so the two agree bit for
            bit, and bilinear interpolation preserves the [0, 1] bound anyway.)
        channel_axis: Which axis holds the class channels, or ``None`` for an
            array that HAS no channel axis - a single-class head's ``(H, W)``
            map. This must be stated, not inferred from ``ndim``: a ``(H, W)``
            map and a ``(C, N)`` stack are both 2-D and need opposite
            treatment, and picking the wrong one would softmax across image rows.

    Returns:
        An array of the same shape with every value in [0, 1]: the input itself
        (clipped) when it is already probability-scaled, else a softmax across
        ``channel_axis`` for a multi-channel head, or a sigmoid for a
        single-channel one.
    """
    import numpy as np

    arr = _as_float(raw)
    if is_probability_scaled(arr):
        # Already a probability map - clip only the float slack, so a caller's
        # values survive unchanged. Re-activating it here is the harmful error:
        # a second sigmoid would compress [0, 1] into [0.50, 0.73] and a 0.5
        # gate would then pass the ENTIRE image.
        return np.clip(arr, 0.0, 1.0)
    if channel_axis is None or arr.shape[channel_axis] == 1:
        return _sigmoid(arr)
    return _softmax_along(arr, channel_axis)
