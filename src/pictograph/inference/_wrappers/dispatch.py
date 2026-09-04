"""Single-image inference dispatch shared by the batch inference service and
the per-deployment serving app.

`build_wrapper()` mirrors the batch service's wrapper construction; `infer_image()`
runs one already-decoded BGR image through the right wrapper and returns Pictograph
annotation dicts (or top-k class predictions for classifiers) - the same shapes the
batch service produces, so deployment output == batch output.
"""

from __future__ import annotations

import uuid
from typing import Any


def build_wrapper(
    *,
    model_type: str,
    architecture: str,
    model_path: str,
    classes: list[str],
    input_shape: tuple,
    confidence_threshold: float,
    providers,
    sess_options,
    keypoint_schema: dict[str, Any] | None = None,
):
    """Construct the correct ONNX wrapper for a model. Mirrors the batch service.

    ``keypoint_schema`` is the ``_pictograph.keypoint_schema`` block from the
    model's ``config.json`` - ``{class_names, num_keypoints_per_class, keypoint_names,
    skeleton}``. Optional and ignored by every non-keypoint wrapper, so existing
    callers are unaffected; a keypoint model WITHOUT it still runs, but its joints
    come back positionally named and assumed full-arity.
    """
    from . import (
        PytorchImageClassifier,
        RFDETRDetector,
        RFDETRKeypointDetector,
        RFDETRSegDetector,
        SemanticSegmentationModelPyTorch,
        YOLOXDetector,
    )

    arch = (architecture or "").lower()
    if model_type == "keypoint_detection":
        schema = keypoint_schema or {}
        return RFDETRKeypointDetector(
            model_path=model_path,
            classes=classes,
            input_shape=input_shape,
            confidence_threshold=confidence_threshold,
            num_keypoints_per_class=schema.get("num_keypoints_per_class"),
            keypoint_names=schema.get("keypoint_names"),
            skeleton_edges=schema.get("skeleton"),
            providers=providers,
            sess_options=sess_options,
        )
    if model_type == "object_detection" and arch.startswith("yolox"):
        return YOLOXDetector(
            model_path=model_path,
            input_shape=input_shape,
            confidence=confidence_threshold,
            providers=providers,
            sess_options=sess_options,
        )
    if model_type == "object_detection":
        return RFDETRDetector(
            model_path=model_path,
            input_shape=input_shape,
            confidence_threshold=confidence_threshold,
            providers=providers,
            sess_options=sess_options,
            classes=classes,
        )
    if model_type == "instance_segmentation":
        return RFDETRSegDetector(
            model_path=model_path,
            classes=classes,
            input_shape=input_shape,
            confidence_threshold=confidence_threshold,
            providers=providers,
            sess_options=sess_options,
        )
    if model_type == "semantic_segmentation":
        return SemanticSegmentationModelPyTorch(
            model_path=model_path,
            classes=classes,
            input_shape=input_shape,
            confidence_threshold=confidence_threshold,
            providers=providers,
            sess_options=sess_options,
        )
    if model_type == "classification":
        return PytorchImageClassifier(
            model_path=model_path,
            class_names=classes,
            input_shape=input_shape,
            providers=providers,
            sess_options=sess_options,
        )
    raise ValueError(f"Unknown model_type/architecture: {model_type}/{architecture}")


def keypoint_schema_from_config(config: dict[str, Any] | None) -> dict[str, Any] | None:
    """Pull the keypoint schema out of a model's ``config.json``.

    Accepts the full artifact (``{"_pictograph": {...}, "config": {...}}``) or a
    bare envelope, mirroring the SDK's ``_keypoint_schema_of``. Pure: it parses a
    dict a caller already fetched, so this package stays I/O-free and torch-free
    and can keep being vendored into the SDK unchanged.

    Returns ``None`` for every non-keypoint model - the schema is written
    conditionally, so its absence is the normal case, never an error. A keypoint
    model served WITHOUT it still runs; its joints just come back positionally
    named (``point_0``…) instead of ``nose`` / ``left_eye``.
    """
    if not isinstance(config, dict):
        return None
    env = config.get("_pictograph") if isinstance(config.get("_pictograph"), dict) else config
    if not isinstance(env, dict):
        return None
    schema = env.get("keypoint_schema")
    return schema if isinstance(schema, dict) else None


def _softmax(logits):
    import numpy as np

    shifted = logits - np.max(logits)
    e = np.exp(shifted)
    return e / np.sum(e)


# YOLOX's ONNX graph decodes the full anchor grid but does NOT suppress, so the
# caller must run NMS or get dozens of overlapping boxes per object. 0.45 is the
# YOLOX inference default (nmsthre). RF-DETR wrappers already suppress
# internally, so this is YOLOX-specific.
YOLOX_NMS_IOU = 0.45


def multiclass_nms(boxes_xyxy, class_ids, scores, iou_threshold: float) -> list[int]:
    """Greedy per-class Non-Maximum Suppression. Returns indices to keep.

    Class-aware: boxes only suppress others of the SAME class, so an overlapping
    'person' and 'car' both survive. Shared by the deployment serving path and
    the batch inference service so YOLOX output is identical in both."""
    import numpy as np

    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    keep: list[int] = []
    for c in np.unique(class_ids):
        order = np.where(class_ids == c)[0]
        order = order[scores[order].argsort()[::-1]]
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[iou <= iou_threshold]
    return keep


def _yolox_to_annotations(
    boxes_xyxy, scores_mat, classes, class_filter, conf
) -> list[dict[str, Any]]:
    import numpy as np

    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return []
    class_ids = np.argmax(scores_mat, axis=1)
    confidences = scores_mat[np.arange(len(class_ids)), class_ids]
    keep = confidences >= conf
    boxes, class_ids, confidences = boxes_xyxy[keep], class_ids[keep], confidences[keep]
    # Suppress duplicate overlapping detections (YOLOX emits a dense grid).
    keep_idx = multiclass_nms(boxes, class_ids, confidences, YOLOX_NMS_IOU)
    boxes, class_ids, confidences = boxes[keep_idx], class_ids[keep_idx], confidences[keep_idx]
    filter_set = set(class_filter) if class_filter is not None else None
    anns: list[dict[str, Any]] = []
    for box, cid, score in zip(boxes, class_ids, confidences):
        if cid >= len(classes):
            continue
        name = classes[int(cid)]
        if filter_set is not None and name not in filter_set:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        anns.append(
            {
                "id": str(uuid.uuid4()),
                "name": name,
                "type": "bbox",
                "bounding_box": {"x": x1, "y": y1, "w": max(0.0, x2 - x1), "h": max(0.0, y2 - y1)},
                "confidence": float(score),
                "attributes": ["auto-annotate"],
            }
        )
    return anns


def _instance_seg_to_annotations(
    boxes, scores, labels, masks, classes, class_filter, conf, instance_output_map
) -> list[dict[str, Any]]:
    from . import mask_to_all_polygons

    if boxes is None or len(boxes) == 0:
        return []
    filter_set = set(class_filter) if class_filter is not None else None
    instance_output_map = instance_output_map or {}
    anns: list[dict[str, Any]] = []
    for i in range(len(boxes)):
        score = float(scores[i])
        cid = int(labels[i])
        if score < conf or cid < 0 or cid >= len(classes):
            continue
        name = classes[cid]
        if filter_set is not None and name not in filter_set:
            continue
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        bbox = {"x": x1, "y": y1, "w": max(0.0, x2 - x1), "h": max(0.0, y2 - y1)}
        output_type = instance_output_map.get(name, "polygon")
        mask_i = masks[i] if (masks is not None and len(masks) > i) else None
        poly_body = None
        if output_type != "bbox" and mask_i is not None:
            try:
                poly_body = mask_to_all_polygons(mask_i)
            except Exception:
                poly_body = None
        if output_type == "bbox" or poly_body is None:
            anns.append(
                {
                    "id": str(uuid.uuid4()),
                    "name": name,
                    "type": "bbox",
                    "bounding_box": bbox,
                    "confidence": score,
                    "attributes": ["auto-annotate"],
                }
            )
        else:
            anns.append(
                {
                    "id": str(uuid.uuid4()),
                    "name": name,
                    "type": "polygon",
                    "bounding_box": poly_body["bounding_box"],
                    "polygon": poly_body["polygon"],
                    "confidence": score,
                    "attributes": ["auto-annotate"],
                }
            )
    return anns


def semantic_masks_from_logits(full, conf: float):
    """Raw (C, H, W) semantic output at FULL resolution -> (binary masks, prob maps).

    Mirrors SemanticSegmentationModelPyTorch.postprocess exactly: the raw output
    is CALIBRATED to probabilities first, then a single-channel model is a binary
    mask over that, while multi-class takes the argmax across channels (channel 0
    is background) and keeps a pixel only where its own channel also clears the
    confidence gate. Shared so both engines threshold identically.

    The calibration is what makes `conf` mean what it says. `create_model` bakes
    a sigmoid in for a SINGLE-class Unet/UnetPlusPlus and for nothing else, so
    every multi-class model and every Segformer emits raw unbounded logits - and
    both engines used to compare those directly against a probability-scale
    threshold. `semantic_calibration.semantic_probabilities` is the one place
    that is decided; see its docstring for the measurement.

    The returned prob maps are now real probabilities too, so the per-region mean
    confidence `_semantic_seg_to_annotations` derives from them is a probability
    rather than a logit clipped into [0, 1].
    """
    import numpy as np

    from .semantic_calibration import semantic_probabilities

    probabilities = semantic_probabilities(full, channel_axis=0)
    if probabilities.shape[0] == 1:
        return [(probabilities[0] >= conf).astype(np.uint8)], [probabilities[0]]
    assigned = np.argmax(probabilities, axis=0)
    masks, probs = [], []
    for class_idx in range(1, probabilities.shape[0]):
        gate = probabilities[class_idx] >= conf
        masks.append(((assigned == class_idx) & gate).astype(np.uint8))
        probs.append(probabilities[class_idx])
    return masks, probs


def _present_classes(masks):
    """Which classes have ANY pixel, in ONE vectorised pass. None if undecidable.

     **The dtype branch is the whole optimisation, not a micro-detail.**

    `mask_to_instance_polygons` runs `connectedComponentsWithStats` at FULL
    RESOLUTION per class - including for classes that are entirely empty, which on
    a real 80-class model is most of them (measured: only 37 of 80 present on one
    1440x810 frame). Skipping those is worth ~2x.

    But the OBVIOUS way to find them is slower than the thing it saves. MEASURED
    on that same 80x810x1440 uint8 stack:

        (stack > 0.5).reshape(n, -1).any(axis=1)   68.2 ms   <- allocates 93 MB
        stack.reshape(n, -1).any(axis=1)            2.5 ms   <- allocates nothing

    The first form made the emitter SLOWER at 1440x810 on the first attempt. For
    any integer or boolean dtype `!= 0` and `> 0.5` are the SAME predicate, so
    taking `any` directly is exact rather than an approximation. Floats still need
    the threshold.

    `tests/test_semantic_seg_emitter.py` pins both halves: that the result is
    correct for every dtype, and that an integer stack is never compared against
    0.5 - which is the regression that would quietly give the time back.
    """
    import numpy as np

    if masks is None:
        return None
    try:
        stack = np.asarray(masks)
        if stack.ndim < 2:
            return None
        flat = stack.reshape(stack.shape[0], -1)
        if stack.dtype == bool or np.issubdtype(stack.dtype, np.integer):
            return flat.any(axis=1)
        return (flat > 0.5).any(axis=1)
    except Exception:
        # A ragged or otherwise unstackable `masks` is not worth failing over -
        # the caller falls back to doing the per-class work, exactly as before.
        return None


def _semantic_seg_to_annotations(
    masks, classes, class_filter, prob_maps=None
) -> list[dict[str, Any]]:
    """Per-class semantic masks -> polygon annotations. Shared by both engines.

    ``prob_maps`` (optional, same order as ``masks``) supplies a real per-region
    confidence - the MEAN probability over the region's own pixels. Without it every
    semantic polygon reported a hardcoded 1.0 regardless of how marginal the
    evidence was.
    """
    import numpy as np

    from . import mask_to_instance_polygons

    filter_set = set(class_filter) if class_filter is not None else None
    anns: list[dict[str, Any]] = []

    # Skip classes with no pixels before doing any per-class work - see
    # `_present_classes`. Classes that ARE present take exactly the path they
    # always took, so this cannot change any output.
    present = _present_classes(masks)

    for cid, mask in enumerate(masks if masks is not None else []):
        if cid >= len(classes):
            continue
        name = classes[cid]
        if filter_set is not None and name not in filter_set:
            continue
        if present is not None and not present[cid]:
            # No pixel anywhere in this class - `mask_to_instance_polygons` would
            # label an empty image and return [] after the full-resolution pass.
            continue
        try:
            poly_bodies = mask_to_instance_polygons(mask)
        except Exception:
            poly_bodies = []
        prob = prob_maps[cid] if prob_maps is not None and len(prob_maps) > cid else None
        for poly_body in poly_bodies:
            ann = {
                "id": str(uuid.uuid4()),
                "name": name,
                "type": "polygon",
                "bounding_box": poly_body["bounding_box"],
                "polygon": poly_body["polygon"],
                "attributes": ["auto-annotate"],
            }
            if prob is not None:
                region = np.asarray(mask, dtype=bool)
                if region.any():
                    ann["confidence"] = float(np.clip(np.asarray(prob)[region].mean(), 0.0, 1.0))
            anns.append(ann)
    return anns


def _classification_to_result(
    logits, classes, conf: float, top_k: int, model_type: str, class_filter=None
) -> dict[str, Any]:
    """Raw classifier logits -> the ranked-class result dict. Shared by both engines."""
    import numpy as np

    # A classifier ALWAYS has a best class, so at least one prediction must be
    # reported (the "top-1 is always reported" invariant below). `top_k` is
    # CLIENT-CONTROLLABLE - the deployment /predict request forwards it straight
    # into this decode - and top_k <= 0 used to return `{"predictions": [],
    # "tags": []}` for a perfectly valid image: a silent "the model found
    # nothing" for a model that cannot find nothing. A NEGATIVE top_k was worse -
    # `np.argsort(probs)[::-1][:top_k]` mis-sliced and dropped the LOWEST-prob
    # class instead of capping the count. Clamp to >= 1.
    top_k = max(1, int(top_k))
    probs = _softmax(np.asarray(logits, dtype=np.float64))
    filter_set = set(class_filter) if class_filter is not None else None
    preds: list[dict[str, Any]] = []
    tags: list[str] = []
    for rank, idx in enumerate(np.argsort(probs)[::-1][:top_k]):
        i = int(idx)
        if i >= len(classes):
            continue
        score = float(probs[i])
        name = classes[i]
        if filter_set is not None and name not in filter_set:
            continue
        # The single best class is ALWAYS reported, whatever the threshold: a
        # classifier's answer is "which class", and returning nothing (or a
        # synthesized placeholder) for a low-confidence image is a worse answer
        # than a real class with a low score the caller can inspect. The
        # threshold still prunes ranks 2..k.
        if rank > 0 and score < conf:
            continue
        preds.append({"class": name, "confidence": score})
        tags.append(name)
    return {"model_type": model_type, "predictions": preds, "tags": tags}


def _keypoint_to_annotations(
    boxes, scores, class_ids, keypoints, wrapper, classes, class_filter, conf
) -> list[dict[str, Any]]:
    """RF-DETR keypoint outputs -> Pictograph ``keypoint`` annotation dicts.

    **A joint is a CLASS; ``instance_id`` is the OBJECT.** The ``skeleton``
    primitive is gone: its edge list was a per-class template identical for every
    instance, and the only thing it uniquely carried was instance identity. So a
    detection of a K-joint class emits **K separate ``keypoint`` annotations**, each
    named after its own joint class, all sharing one ``instance_id``::

        {id, name: "nose", type: "keypoint", keypoint: {x, y},
         confidence, attributes: ["auto-annotate"], instance_id: 1}

    Connectivity is postprocessing - group by ``instance_id``, connect via the
    class template in ``project_config``. Nothing here re-derives it.

    ``instance_id`` is **1-based and scoped to the image**, allocated in DETECTION
    order, and only consumed by a detection that actually emits a joint, so the ids
    of one image are contiguous ``1..N``.

    Every emitted annotation carries **NO ``bounding_box``** - a point has no
    extent, and both ``KeypointAnnotation`` twins (the SDK's and the API's) are
    ``extra="forbid"``, so an extra key is not ignored, it RAISES. The instance's
    box is DERIVED at export/training time from its placed points, which is
    exactly what COCO's own person-keypoints entries carry.

    **A joint below the findability threshold is OMITTED, not zeroed.** There is no
    ``visibility`` field on a keypoint annotation, so absence IS the encoding of
    "not found" - and the grouping rule re-materializes it as COCO's ``0, 0, 0`` at
    the template index. To keep a real object from vanishing entirely (the
    silent-drop class this emitter has been bitten by), a detection whose joints are
    ALL sub-threshold still emits its single best joint - the structural successor
    of the old "fall back to the detector's box" arm.

    ``confidence`` is the DETECTION's score on every joint of an instance, not the
    per-joint findability score: it is the object's score, which is what the removed
    ``skeleton`` carried and what the arity-1 branch has always carried.

    The arity is read from the model's own schema (``num_keypoints_per_class``), not
    from the returned joint count, so a class's shape never flickers with a single
    missed joint. Arity 1 ("keypoint-as-class") keeps its own naming - the CLASS is
    the joint - and its detector-box-centre fallback, since a point must exist.
    """
    anns: list[dict[str, Any]] = []
    filter_set = set(class_filter) if class_filter is not None else None
    if boxes is None or len(boxes) == 0:
        return anns

    kp_threshold = float(getattr(wrapper, "keypoint_threshold", 0.5))
    per_class_counts = list(getattr(wrapper, "num_keypoints_per_class", []) or [])
    instance_id = 0
    for idx, (box, cid, score) in enumerate(zip(boxes, class_ids, scores)):
        if float(score) < conf or int(cid) < 0 or int(cid) >= len(classes):
            continue
        name = classes[int(cid)]
        if filter_set is not None and name not in filter_set:
            continue

        joints = keypoints[idx] if idx < len(keypoints) else []
        # The class's trained joint count. Falls back to the returned joint count
        # when the schema is absent (an externally-supplied ONNX).
        arity = per_class_counts[int(cid)] if int(cid) < len(per_class_counts) else len(joints)

        # ── keypoint-as-class: a single-joint class → one point named for the class ──
        if arity == 1:
            x1, y1, x2, y2 = [float(v) for v in box]
            if len(joints) >= 1 and float(joints[0][2]) >= kp_threshold:
                kx, ky = float(joints[0][0]), float(joints[0][1])
            else:
                # Joint below findability (or none returned): the detection box is
                # the point's neighborhood, so its center is the best estimate.
                kx, ky = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            instance_id += 1
            anns.append(
                {
                    "id": str(uuid.uuid4()),
                    "name": name,
                    "type": "keypoint",
                    "keypoint": {"x": kx, "y": ky},
                    "confidence": float(score),
                    "attributes": ["auto-annotate"],
                    "instance_id": instance_id,
                }
            )
            continue

        # ── multi-joint object → one `keypoint` per joint, sharing an instance_id ──
        node_names = wrapper.node_names_for(name, len(joints)) if len(joints) else []
        placed = [j for j, (_, _, jconf) in enumerate(joints) if float(jconf) >= kp_threshold]
        if not placed and len(joints):
            # Nothing cleared findability, but the OBJECT cleared `conf`. Keep it
            # alive as its single most-findable joint rather than dropping a real
            # detection on the floor.
            placed = [max(range(len(joints)), key=lambda j: float(joints[j][2]))]
        if not placed:
            continue

        instance_id += 1
        for j in placed:
            jx, jy = float(joints[j][0]), float(joints[j][1])
            anns.append(
                {
                    "id": str(uuid.uuid4()),
                    "name": node_names[j],
                    "type": "keypoint",
                    "keypoint": {"x": jx, "y": jy},
                    "confidence": float(score),
                    "attributes": ["auto-annotate"],
                    "instance_id": instance_id,
                }
            )
    return anns


def infer_image(
    wrapper,
    img_bgr,
    *,
    model_type: str,
    architecture: str,
    classes: list[str],
    class_filter: list[str] | None = None,
    confidence: float = 0.5,
    instance_output_map: dict[str, str] | None = None,
    top_k: int = 1,
) -> dict[str, Any]:
    """Run one BGR image through `wrapper`. Returns the per-deployment /infer
    response: detection/seg models -> {predictions:[ann...]}, classifier ->
    {predictions:[{class,confidence}], tags:[...]}.
    """
    import numpy as np

    arch = (architecture or "").lower()

    if model_type == "object_detection" and arch.startswith("yolox"):
        tensor = wrapper.preprocess(img_bgr)
        input_name = wrapper.session.get_inputs()[0].name
        outputs = wrapper.session.run(
            None, {input_name: np.stack([tensor], axis=0).astype(np.float32)}
        )[0]
        boxes_xyxy, scores_mat = wrapper.postprocess(outputs[0:1])
        return {
            "model_type": model_type,
            "predictions": _yolox_to_annotations(
                boxes_xyxy, scores_mat, classes, class_filter, confidence
            ),
        }

    if model_type == "object_detection":  # RF-DETR
        boxes, scores, class_ids = wrapper.predict(img_bgr)
        anns: list[dict[str, Any]] = []
        filter_set = set(class_filter) if class_filter is not None else None
        if boxes is not None:
            for box, cid, score in zip(boxes, class_ids, scores):
                if float(score) < confidence or int(cid) < 0 or int(cid) >= len(classes):
                    continue
                name = classes[int(cid)]
                if filter_set is not None and name not in filter_set:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                anns.append(
                    {
                        "id": str(uuid.uuid4()),
                        "name": name,
                        "type": "bbox",
                        "bounding_box": {
                            "x": x1,
                            "y": y1,
                            "w": max(0.0, x2 - x1),
                            "h": max(0.0, y2 - y1),
                        },
                        "confidence": float(score),
                        "attributes": ["auto-annotate"],
                    }
                )
        return {"model_type": model_type, "predictions": anns}

    if model_type == "keypoint_detection":  # rfdetr-keypoint
        boxes, scores, class_ids, keypoints = wrapper.predict(img_bgr)
        return {
            "model_type": model_type,
            "predictions": _keypoint_to_annotations(
                boxes, scores, class_ids, keypoints, wrapper, classes, class_filter, confidence
            ),
        }

    if model_type == "instance_segmentation":  # rfdetr-seg
        boxes, scores, labels, masks = wrapper.predict(img_bgr)
        return {
            "model_type": model_type,
            "predictions": _instance_seg_to_annotations(
                boxes, scores, labels, masks, classes, class_filter, confidence, instance_output_map
            ),
        }

    if model_type == "semantic_segmentation":
        # `masks` is a numpy stack, not a list - `masks or []` raises the
        # ambiguous-truth ValueError on every multi-mask result, which killed
        # ALL semantic-seg inference through this path (deployment /infer,
        # workflow model nodes, SDK local). The shared emitter None-checks instead.
        #
        # `return_probs=True` is what makes this branch reach PARITY with the torch
        # engine (`_torch._predict_smp`, which passes the maps
        # `semantic_masks_from_logits` already hands back). Without them the emitter
        # has no evidence to derive a confidence from and every ONNX semantic polygon
        # fell back to the pydantic default of 1.0, however marginal the region was.
        masks, prob_maps = wrapper.predict(img_bgr, return_probs=True)
        return {
            "model_type": model_type,
            "predictions": _semantic_seg_to_annotations(masks, classes, class_filter, prob_maps),
        }

    if model_type == "classification":
        tensor = wrapper.preprocess(img_bgr)
        outputs = wrapper.session.run(
            None, {wrapper.input_name: np.stack([tensor], axis=0).astype(np.float32)}
        )[0]
        return _classification_to_result(
            outputs[0], classes, confidence, top_k, model_type, class_filter
        )

    raise ValueError(f"Unsupported model_type/architecture: {model_type}/{architecture}")


def infer_batch(
    wrapper,
    imgs,
    *,
    model_type: str,
    architecture: str,
    classes: list[str],
    class_filter: list[str] | None = None,
    confidence: float = 0.5,
    instance_output_map: dict[str, str] | None = None,
    top_k: int = 1,
) -> list[dict[str, Any]]:
    """Run a list of BGR frames; returns one ``{predictions: [...]}`` per image, in
    order. Batches YOLOX (one ``session.run`` over a stacked batch - the throughput
    win for video); every other architecture loops :func:`infer_image`. Used by the
    workflow runner so detection amortizes ONNX overhead across a frame window
    while the stateful tracker still consumes frames sequentially."""
    import numpy as np

    arch = (architecture or "").lower()
    if model_type == "object_detection" and arch.startswith("yolox") and len(imgs) > 1:
        tensors = []
        ratios = []
        for im in imgs:
            tensors.append(wrapper.preprocess(im))
            ratios.append(wrapper.ratio)
        input_name = wrapper.session.get_inputs()[0].name
        batch = np.stack(tensors, axis=0).astype(np.float32)
        outputs = wrapper.session.run(None, {input_name: batch})[0]
        out: list[dict[str, Any]] = []
        for i in range(len(imgs)):
            wrapper.ratio = ratios[i]
            boxes_xyxy, scores_mat = wrapper.postprocess(outputs[i : i + 1])
            out.append(
                {
                    "model_type": model_type,
                    "predictions": _yolox_to_annotations(
                        boxes_xyxy, scores_mat, classes, class_filter, confidence
                    ),
                }
            )
        return out
    return [
        infer_image(
            wrapper,
            im,
            model_type=model_type,
            architecture=architecture,
            classes=classes,
            class_filter=class_filter,
            confidence=confidence,
            instance_output_map=instance_output_map,
            top_k=top_k,
        )
        for im in imgs
    ]
