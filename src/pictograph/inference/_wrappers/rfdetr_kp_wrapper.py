"""
RF-DETR Keypoint (pose) Inference.

ONNX Runtime wrapper for RF-DETR keypoint models. The pose sibling of
``rfdetr_det_wrapper`` / ``rfdetr_seg_wrapper``; pure numpy/cv2/onnxruntime, no
torch, matching the rest of this package.

**The output layout, which is the whole reason this wrapper is not the detection
one.** The keypoint head adds a third tensor beside ``pred_logits`` /
``pred_boxes``::

    pred_keypoints: (B, Q, C * max_K, D)

Keypoint slots are **padded per class**: every query carries a full slot block for
*every* class, and only the block belonging to the query's PREDICTED class is
meaningful. ``max_K`` is the largest per-class node count; classes with fewer
nodes leave their tail unused. So decoding is: reshape to
``(Q, C, max_K, D)`` -> index by the predicted label -> take that class's first
``num_keypoints_per_class[label]`` slots. Reading the tensor without the schema -
which is why the schema is persisted on the model - yields plausible garbage
rather than an error, since the shape is still valid.

``D`` is at least 3: ``x``, ``y`` (both NORMALIZED to [0,1], scaled here by the
ORIGINAL image size) and a findability logit that needs a sigmoid. When
``D >= 7``, slots ``4:7`` carry the NLL-Cholesky precision parameters - RF-DETR
Keypoint's calibrated per-keypoint uncertainty. We do not surface those: they are
a training-time construct, and the per-keypoint confidence a Pictograph annotation
carries is ``sigmoid(d[2])``.

This mirrors ``rfdetr/models/postprocess.py::_postprocess_keypoints`` from the
pinned rfdetr 1.8.3, which is the reference implementation for the arithmetic here.

**The decode is module-level, not a method, because a SECOND engine runs it.**
:func:`preprocess_keypoint_image` and :func:`decode_keypoint_outputs` are the
canonical preprocess + reduction; the class below is a thin ONNX-session shell
around them, and the SDK's native-PyTorch engine
(``pictograph.inference._torch``) feeds its raw module output through the SAME
two functions. That is deliberate: ``rfdetr.predict()`` returns
per-``(query x class)`` hypotheses (its ``PostProcess._select_topk`` flattens
``(Q, C)`` and takes the global top-k, so ONE query appears once per class it
scores on, and the background column competes as a class), while this decode
reduces to ONE hypothesis per query via a foreground-only argmax and then
suppresses per class. Running the torch module through ``predict()`` instead of
this function produced ~5x the detections with DIFFERENT class labels on the
same coordinates - see ``tests/unit/test_keypoint_backend_parity.py``.
"""

from collections.abc import Sequence

import cv2
import numpy as np
import onnxruntime as ort

from .onnx_shape import rfdetr_foreground_columns

# ImageNet normalization - identical for every RF-DETR family wrapper.
KEYPOINT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
KEYPOINT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Query cap applied before the confidence gate. Mirrors rfdetr's own
# ``PostProcess(num_select=300)``; a graph emits at most a few hundred queries,
# so this only ever bounds pathological outputs.
KEYPOINT_NUM_SELECT = 300


def preprocess_keypoint_image(
    image: np.ndarray, dims: tuple[int, int], bgr2rgb: bool = True
) -> np.ndarray:
    """BGR image -> normalized ``(1, 3, H, W)`` float32. ``dims`` is ``(H, W)``.

    Module-level so the torch engine feeds its module the byte-identical tensor
    this wrapper feeds its ONNX session - a preprocessing difference is a silent
    disagreement between the two backends about the same image.
    """
    resized = cv2.resize(image, (dims[1], dims[0]), interpolation=cv2.INTER_LINEAR)
    if bgr2rgb:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - KEYPOINT_MEAN) / KEYPOINT_STD
    return np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0).astype(np.float32)


def split_keypoint_outputs(
    outputs: Sequence[np.ndarray | None],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Identify (boxes, logits, keypoints) by SHAPE, never by name or order.

    The export names outputs from a dict whose key order is not contractual
    (the detection wrapper makes the same choice for the same reason):
      * keypoints - rank 4 ``(B, Q, S, D)``
      * boxes     - rank 3 with last dim exactly 4
      * logits    - the remaining rank-3 tensor

    Shape-keyed identification is also what lets the torch engine hand over its
    raw ``{pred_boxes, pred_logits, pred_keypoints}`` in any order.
    """
    boxes = logits = keypoints = None
    for out in outputs:
        if out is None:
            continue
        if out.ndim == 4:
            keypoints = out
        elif out.ndim == 3:
            if out.shape[-1] == 4 and boxes is None:
                boxes = out
            else:
                logits = out
    # A 2-output graph (no keypoint head) still decodes as plain detection.
    if boxes is None and logits is not None and logits.shape[-1] == 4:
        boxes, logits = logits, None
    return boxes, logits, keypoints


def keypoint_geometry(
    slots: int, classes: Sequence[str], num_keypoints_per_class: Sequence[int] | None
) -> tuple[int, int, list[int]]:
    """``(num_keypoint_classes, max_K, per_class_active_counts)`` for a padded
    slot dimension of ``slots``.

    **This comes from the SCHEMA, never from the logits width.** The keypoint
    tensor's class dimension counts KEYPOINT classes; ``pred_logits`` is
    wider, because RF-DETR (like DETR) carries an extra class slot. Measured
    on a real 1-class trained model: ``pred_logits`` is ``(1, 100, 2)`` while
    ``pred_keypoints`` is ``(1, 100, 5, 8)`` - 1 class x 5 joints. Deriving
    the stride as ``slots // logits_width`` gives ``5 // 2 = 2``, the
    ``slots == C * max_K`` identity then fails, and every pose comes back
    EMPTY with no error anywhere. rfdetr's own ``_postprocess_keypoints``
    uses ``len(num_keypoints_per_class)`` and ``max(num_keypoints_per_class)``
    for exactly this reason.

    With no schema at all, assume ONE keypoint class spanning every slot -
    the only assumption that cannot silently DROP a joint.
    """
    counts = [int(c) for c in (num_keypoints_per_class or []) if int(c) >= 0]
    if not counts:
        # No schema. Prefer the known class count when it divides the slot
        # dimension evenly (the usual multi-class case); otherwise assume ONE
        # class spanning every slot. Both are "read the widest block we can
        # justify", which over-reads into padding at worst - and padded slots
        # decode to zero confidence, which the emit step drops. Guessing a
        # SMALLER stride would instead discard joints the model predicted.
        n = len(classes)
        if n > 1 and slots % n == 0:
            per = slots // n
            return n, per, [per] * n
        return 1, slots, [slots]
    num_kp_classes = len(counts)
    max_k = max(counts) if counts else 0
    return num_kp_classes, max_k, [max(0, min(c, max_k)) for c in counts]


def keypoint_nms(boxes, scores, class_ids, iou_threshold: float) -> np.ndarray:
    """CLASS-AWARE NMS - identical policy to the detection wrapper, so an
    overlapping person and hand both survive."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[iou <= iou_threshold]
    return np.array(sorted(keep), dtype=np.int32)


def decode_keypoint_outputs(
    outputs: Sequence[np.ndarray | None],
    original_shape: tuple[int, int],
    *,
    classes: Sequence[str],
    num_keypoints_per_class: Sequence[int] | None = None,
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """**THE canonical RF-DETR keypoint reduction.** Raw graph/module outputs ->
    ``(boxes_xyxy, scores, class_ids, keypoints_per_detection)``.

    ``keypoints_per_detection[i]`` is an ``(K_i, 3)`` array of
    ``(x_px, y_px, confidence)`` for detection ``i``, already sliced to that
    detection's own class arity.

    The reduction, in order - this is what makes N raw queries into N' objects,
    and every step of it is why the two backends must share this function:

    1. ``sigmoid(pred_logits)``, then **restrict to the FOREGROUND columns**
       (the trailing background slot is not a class).
    2. **Per-query argmax** over those columns - ONE hypothesis per query, its
       best class. This is the de-duplication. rfdetr's own postprocess instead
       flattens ``(Q, C)`` and takes a global top-k, so the same query surfaces
       once per class and one object arrives labelled several different ways.
    3. Top ``KEYPOINT_NUM_SELECT`` queries by score, then the confidence gate.
    4. cxcywh -> xyxy, scaled to the ORIGINAL image and clipped to it.
    5. The predicted class's own padded keypoint block, truncated to its arity.
    6. **Class-aware NMS** on the boxes; each pose rides its detection.
    """
    orig_h, orig_w = original_shape
    boxes_t, logits_t, kps_t = split_keypoint_outputs(outputs)
    empty: tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]] = (
        np.array([]),
        np.array([]),
        np.array([]),
        [],
    )
    if boxes_t is None or logits_t is None:
        return empty

    boxes_cxcywh = boxes_t[0] if boxes_t.ndim == 3 else boxes_t
    logits = logits_t[0] if logits_t.ndim == 3 else logits_t

    # Sigmoid (RF-DETR is multi-label focal, NOT softmax) -> per-query best class.
    #
    # The LAST logit column is RF-DETR's BACKGROUND slot, not a class:
    # `lwdetr.py` defines `foreground_num_classes = detection_num_classes - 1`
    # and notes "the background slot (index detection_num_classes-1)".
    # Measured on a real 1-class trained model, `pred_logits` is width 2 and
    # the background column wins every query on an image with no object -
    # which is correct behaviour, but an argmax over ALL columns then reports
    # background as the predicted class, so a genuine detection whose
    # background score merely happens to be higher is silently lost.
    # Restrict the argmax to the foreground columns.
    scores_all = 1.0 / (1.0 + np.exp(-logits))
    n_fg = rfdetr_foreground_columns(logits.shape[-1], len(classes))
    foreground = scores_all[:, :n_fg]
    class_ids = np.argmax(foreground, axis=1)
    confidences = np.max(foreground, axis=1)

    keep = np.argsort(confidences)[::-1][:KEYPOINT_NUM_SELECT]
    mask = confidences[keep] >= confidence_threshold
    keep = keep[mask]
    if keep.size == 0:
        return empty

    boxes_cxcywh = boxes_cxcywh[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]

    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]] * orig_w, 0, orig_w)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]] * orig_h, 0, orig_h)

    # ── keypoints ────────────────────────────────────────────────────────
    per_detection: list[np.ndarray] = []
    if kps_t is not None:
        kps = kps_t[0][keep]  # (N, C*max_K, D)
        slots, depth = kps.shape[1], kps.shape[2]
        num_kp_classes, max_k, counts = keypoint_geometry(slots, classes, num_keypoints_per_class)
        if max_k > 0 and slots == num_kp_classes * max_k and depth >= 3:
            reshaped = kps.reshape(kps.shape[0], num_kp_classes, max_k, depth)
            for i, cid in enumerate(class_ids):
                c = int(cid)
                # A predicted label can exceed the keypoint-class count (the
                # logits carry an extra slot); such a detection simply has no
                # pose block - same guard as rfdetr's `labels_i < C`.
                if c < 0 or c >= num_kp_classes:
                    per_detection.append(np.zeros((0, 3), dtype=np.float32))
                    continue
                active = counts[c]
                block = reshaped[i, c, :active, :]  # (K_i, D)
                if block.size == 0:
                    per_detection.append(np.zeros((0, 3), dtype=np.float32))
                    continue
                xy_conf = np.empty((block.shape[0], 3), dtype=np.float32)
                xy_conf[:, 0] = block[:, 0] * orig_w
                xy_conf[:, 1] = block[:, 1] * orig_h
                xy_conf[:, 2] = 1.0 / (1.0 + np.exp(-block[:, 2]))
                per_detection.append(xy_conf)
        else:
            per_detection = [np.zeros((0, 3), dtype=np.float32)] * len(class_ids)
    else:
        per_detection = [np.zeros((0, 3), dtype=np.float32)] * len(class_ids)

    # Class-aware NMS on the boxes; the pose rides its detection.
    keep_idx = keypoint_nms(boxes_xyxy, confidences, class_ids, nms_threshold)
    return (
        boxes_xyxy[keep_idx],
        confidences[keep_idx],
        class_ids[keep_idx],
        [per_detection[i] for i in keep_idx],
    )


def keypoint_node_names(
    keypoint_names: dict[str, list[str]] | None, class_name: str, count: int
) -> list[str]:
    """The class's canonical joint names, padded positionally if short.

    A name array shorter than the model's arity is not an error - an
    externally-trained ONNX may carry no schema at all - so the tail degrades
    to ``point_{i}`` rather than dropping joints the model did predict.
    """
    names = list((keypoint_names or {}).get(class_name) or [])
    if len(names) < count:
        names += [f"point_{i}" for i in range(len(names), count)]
    return names[:count]


class RFDETRKeypointDetector:
    """ONNX Runtime wrapper for RF-DETR keypoint models.

    Args:
        model_path: Path to the ONNX file.
        classes: Ordered class names - index == position (the training contract).
        num_keypoints_per_class: Active joint count per class, in the SAME order.
            Falls back to a uniform split of the padded slot dimension when the
            model row carries no schema (an externally-supplied ONNX).
        keypoint_names: Optional ``{class: [joint class names]}`` - the ordered
            joint-class template the model was trained against. It is what NAMES
            each emitted point (a joint is a CLASS), so index ``j`` of a decoded
            pose becomes an annotation of class ``keypoint_names[cls][j]``.
            Missing names degrade to ``point_{i}`` - the annotation is still
            structurally valid and editable.
        skeleton_edges: Optional ``{class: [[i, j], ...]}`` - the class TEMPLATE's
            connectivity, 1-indexed on the wire (COCO's convention) and stored
            0-indexed here. It is model METADATA, not an annotation primitive:
            emitted points carry only their joint class + ``instance_id``, and a
            consumer that wants to DRAW the pose connects the points of one
            instance through these edges. (The ``skeleton`` annotation type no
            longer exists; the per-class template is all that remains of it.)
    """

    MEAN = KEYPOINT_MEAN
    STD = KEYPOINT_STD

    def __init__(
        self,
        model_path: str,
        classes: list[str],
        input_shape: tuple[int, int] = (576, 576),
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        num_keypoints_per_class: list[int] | None = None,
        keypoint_names: dict[str, list[str]] | None = None,
        skeleton_edges: dict[str, list[list[int]]] | None = None,
        keypoint_threshold: float = 0.5,
        providers: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"],
        sess_options: ort.SessionOptions | None = None,
    ):
        self.model_path = model_path
        self.classes = list(classes or [])
        self.dims = input_shape
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.num_keypoints_per_class = list(num_keypoints_per_class or [])
        self.keypoint_names = dict(keypoint_names or {})
        # COCO's `skeleton` is 1-INDEXED; Pictograph's edges are 0-indexed. The
        # -1 happens exactly once, here, mirroring the writer's single +1.
        self.skeleton_edges = {
            cls: [[int(i) - 1, int(j) - 1] for i, j in (edges or []) if int(i) > 0 and int(j) > 0]
            for cls, edges in (skeleton_edges or {}).items()
        }
        # A joint below this findability score is emitted as visibility 0 ("not
        # labelled") rather than dropped: the node list stays TEMPLATE-COMPLETE,
        # which is the invariant every exporter and the editor rely on.
        self.keypoint_threshold = keypoint_threshold
        self.providers = providers

        if sess_options is None:
            sess_options = ort.SessionOptions()

        self.session = ort.InferenceSession(
            self.model_path, providers=self.providers, sess_options=sess_options
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Prefer the graph's own static H/W over the declared shape - the seg
        # wrapper does the same, and a mismatch here silently distorts every
        # coordinate rather than raising.
        try:
            shape = self.session.get_inputs()[0].shape
            h, w = shape[2], shape[3]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                self.dims = (h, w)
        except (IndexError, TypeError):
            pass

    # ── preprocessing ────────────────────────────────────────────────────────
    def preprocess(self, image: np.ndarray, bgr2rgb: bool = True) -> np.ndarray:
        """BGR image -> normalized (1, 3, H, W) float32. Identical to the twins.

        Delegates to :func:`preprocess_keypoint_image` so the torch engine feeds
        its module the byte-identical tensor.
        """
        return preprocess_keypoint_image(image, self.dims, bgr2rgb)

    # ── output identification ────────────────────────────────────────────────
    @staticmethod
    def _split_outputs(outputs: list[np.ndarray]):
        """Identify (boxes, logits, keypoints) by SHAPE - see
        :func:`split_keypoint_outputs`, which this delegates to."""
        return split_keypoint_outputs(outputs)

    # ── postprocessing ───────────────────────────────────────────────────────
    def postprocess(
        self, outputs: list[np.ndarray], original_shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
        """Decode to (boxes_xyxy, scores, class_ids, keypoints_per_detection).

        The arithmetic lives in :func:`decode_keypoint_outputs` - module-level so
        the SDK's native-PyTorch engine runs the SAME reduction over its own raw
        module output rather than ``rfdetr.predict()``'s per-(query x class)
        hypotheses.
        """
        return decode_keypoint_outputs(
            outputs,
            original_shape,
            classes=self.classes,
            num_keypoints_per_class=self.num_keypoints_per_class,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
        )

    def _nms(self, boxes, scores, class_ids, iou_threshold) -> np.ndarray:
        """CLASS-AWARE NMS - see :func:`keypoint_nms`, which this delegates to."""
        return keypoint_nms(boxes, scores, class_ids, iou_threshold)

    def predict(self, image: np.ndarray):
        """Full pipeline: (boxes_xyxy, scores, class_ids, keypoints_per_detection)."""
        original_shape = image.shape[:2]
        # A real failure here (e.g. an execution provider that can't compile
        # the graph) must propagate - swallowing it into an empty result
        # reports "your model found nothing" instead of the actual error.
        tensor = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        return self.postprocess(outputs, original_shape)

    # ── annotation emission ──────────────────────────────────────────────────
    def node_names_for(self, class_name: str, count: int) -> list[str]:
        """The class's canonical joint names, padded positionally if short -
        see :func:`keypoint_node_names`, which this delegates to."""
        return keypoint_node_names(self.keypoint_names, class_name, count)
