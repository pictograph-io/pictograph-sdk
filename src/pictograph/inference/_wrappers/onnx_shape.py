"""ONNX graph shape introspection - the canonical, shared implementation.

The sizing fix (2026-07-16): a model stored with a minimal training_config
(API-created runs) used to be sized from the config's image_height/width guess
(default 640), and any graph whose STATIC input differs (RF-DETR nano = 384)
died with ORT INVALID_ARGUMENT in EVERY consumer. The first pass fixed the
inference service + the SDK - but the WORKFLOW runner is a THIRD ONNX engine
with the same bug (found live 2026-07-16: a workflow over a minimal-config
rfdetr model errored on its first frame). The implementation
now lives here, in the package every ONNX-running image already mounts, so a
fourth engine can't fork the logic again. kept in sync: the SDK's
`pictograph.inference._true_input_size` calls `true_input_shape` here directly
(via the vendored `_wrappers.onnx_shape` copy) rather than keeping a second
reimplementation.
"""

import logging

logger = logging.getLogger(__name__)


def true_input_shape(local_path: str, declared: tuple) -> tuple[int, int]:
    """The ONNX graph's static (H, W), when it declares one - else ``declared``.

    Proto-only read (``load_external_data=False``), no session build. Dynamic
    dims (symbolic/-1) keep the declared shape; introspection failures fall
    back loudly rather than blocking a load.
    """
    try:
        import onnx

        m = onnx.load(local_path, load_external_data=False)
        dims = m.graph.input[0].type.tensor_type.shape.dim
        if len(dims) == 4:
            h, w = dims[2], dims[3]
            if (
                h.HasField("dim_value")
                and w.HasField("dim_value")
                and h.dim_value > 0
                and w.dim_value > 0
            ):
                true_shape = (int(h.dim_value), int(w.dim_value))
                if true_shape != tuple(declared):
                    logger.debug(
                        "[load] ONNX graph input %dx%d overrides configured %s",
                        true_shape[0],
                        true_shape[1],
                        tuple(declared),
                    )
                return true_shape
    except Exception as e:
        logger.debug(
            "[load] input-shape introspection failed (%s) - using configured %s", e, tuple(declared)
        )
    return tuple(declared)  # type: ignore[return-value]


def rfdetr_foreground_columns(logits_width: int, class_count: int) -> int:
    """How many LEADING ``pred_logits`` columns are real classes.

    RF-DETR's head is built as ``num_classes + 1`` (``lwdetr.py``: *"the
    `num_classes` naming here is somewhat misleading. It indeed corresponds to
    `max_obj_id + 1`"*), and the extra TRAILING slot is background - quoting
    ``lwdetr.py``: *"the background slot (index detection_num_classes-1)"*.
    Measured on real trained models: 82 classes -> width 83, 80 -> 81, 1 -> 2.
    Detection, segmentation and keypoint all share one ``class_embed``, so the
    convention is identical for all three.

    An ``argmax`` over EVERY column can therefore return the background index.
    Downstream that index fails the ``cid >= len(classes)`` guard and the
    detection is DROPPED - silently, with no exception - so a genuine object
    whose background score merely happens to be higher is simply lost, and the
    reported confidence for surviving rows is whatever won the argmax rather
    than the object score. Restricting the argmax to the foreground columns is
    strictly additive: it can only ever recover detections, never invent them.

    Lives here - the one module every ONNX engine already mounts - so the four
    engines (deployments / batch auto-annotate / workflows / SDK local
    inference) cannot fork three copies of the rule.
    """
    if logits_width <= 1:
        return max(logits_width, 0)
    if 0 < class_count < logits_width:
        return class_count  # class list known → background is what's left over
    return logits_width - 1  # unknown → assume the single trailing background slot
