"""
RF-DETR Object Detection Inference

ONNX Runtime wrapper for RF-DETR object detection models. Pure ONNX - numpy, cv2
and onnxruntime only, no dependency on the rfdetr package.

Example usage:
    from inference_wrappers import RFDETRDetector

    detector = RFDETRDetector(
        model_path=weights,
        input_shape=(576, 576),
        confidence_threshold=0.5,
    )
    boxes, scores, class_ids = detector.predict(image_bgr)

The caller owns what happens next: both consumers decode these arrays into
Pictograph annotations with their own class map, threshold and class filter, so
this module deliberately stops at the arrays.
"""

import cv2
import numpy as np
import onnxruntime as ort

from .onnx_shape import rfdetr_foreground_columns

# ImageNet normalization - module level so the free functions below and the
# class share ONE copy. `RFDETRDetector.MEAN`/`.STD` alias these.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_detection_image(
    image: np.ndarray, dims: tuple[int, int], bgr2rgb: bool = True
) -> np.ndarray:
    """A BGR image → the (1, C, H, W) float32 tensor RF-DETR's graph takes.

    Free-standing so a caller holding a torch module - which has no ONNX session
    - can build the SAME tensor the graph is fed. `RFDETRDetector.preprocess`
    delegates here, so the two cannot drift.

    `dims` is (H, W); cv2's `dsize` is (W, H) - hence the swap.
    """
    resized = cv2.resize(image, (dims[1], dims[0]), interpolation=cv2.INTER_LINEAR)
    if bgr2rgb:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - _MEAN) / _STD
    transposed = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(transposed, axis=0).astype(np.float32)


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(N, 4) normalized [cx, cy, w, h] → normalized [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def detection_nms(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """CLASS-AWARE Non-Maximum Suppression → indices to keep.

    A box only suppresses others of the SAME class, so a legitimately
    overlapping 'person' and 'backpack' both survive. The previous
    class-agnostic pass dropped the lower-scoring box even when it was a
    DIFFERENT object - over-suppression for a multi-class detector. Mirrors
    dispatch.multiclass_nms (the YOLOX path's class-aware NMS), whose comment
    already assumed the RF-DETR wrappers "suppress internally" per-object.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    keep = []
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
            # +1e-9 avoids 0/0 -> NaN on twin zero-area boxes (a prediction
            # clipped fully outside the image), which would silently drop them.
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[iou <= iou_threshold]

    return np.array(keep, dtype=np.int32)


def decode_detection_outputs(
    outputs: list[np.ndarray],
    original_shape: tuple[int, int],
    *,
    classes: list[str] | None = None,
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RF-DETR detection outputs → (boxes_xyxy, scores, class_ids).

    **The one reduction both engines share.** A per-query FOREGROUND argmax, a
    top-300 sort, the confidence gate, cxcywh→xyxy in the ORIGINAL image's pixel
    space, and class-aware NMS.

    It exists as a free function, rather than only as `RFDETRDetector.postprocess`,
    so the torch runtime can call it too. rfdetr's own `PostProcess` does
    something materially different - a GLOBAL top-k over the flattened (Q, C)
    sigmoid grid, which returns one query once per class it scores on - and the
    two disagreed by 2-3x on the same model and image. The torch path now
    runs its raw forward and hands the tensors here, so a decode change lands on
    both engines at once or on neither.

    ``outputs`` are the graph's tensors in any order; they are identified by
    shape. ``original_shape`` is (height, width) of the image as the caller
    supplied it, pre-resize.
    """
    orig_h, orig_w = original_shape
    classes = list(classes or [])

    if len(outputs) >= 2:
        out0, out1 = outputs[0], outputs[1]

        # Remove batch dimension
        if len(out0.shape) == 3:
            out0 = out0[0]
        if len(out1.shape) == 3:
            out1 = out1[0]

        # Identify boxes (4 values) vs scores (more values)
        if out0.shape[-1] == 4:
            boxes_cxcywh = out0
            logits = out1
        else:
            boxes_cxcywh = out1
            logits = out0

        # Apply sigmoid to convert logits to probabilities
        scores = 1 / (1 + np.exp(-logits))  # sigmoid

        # Get class predictions and confidences over the FOREGROUND columns
        # only - the trailing column is RF-DETR's background slot, not a
        # class. Argmaxing over it silently DROPS the detection
        # downstream (`cid >= len(classes)`), so this only ever adds rows.
        n_fg = rfdetr_foreground_columns(scores.shape[-1], len(classes))
        foreground = scores[:, :n_fg]
        class_ids = np.argmax(foreground, axis=1)
        confidences = np.max(foreground, axis=1)

    else:
        # Single output format
        detections = outputs[0]
        if len(detections.shape) == 3:
            detections = detections[0]

        boxes_cxcywh = detections[:, :4]
        if detections.shape[-1] > 4:
            confidences = 1 / (1 + np.exp(-detections[:, 4]))
            class_ids = (
                detections[:, 5].astype(np.int32)
                if detections.shape[-1] > 5
                else np.zeros(len(detections), dtype=np.int32)
            )
        else:
            confidences = np.ones(len(detections))
            class_ids = np.zeros(len(detections), dtype=np.int32)

    # Sort by confidence (descending) and limit to top detections
    sorted_indices = np.argsort(confidences)[::-1][:300]
    boxes_cxcywh = boxes_cxcywh[sorted_indices]
    confidences = confidences[sorted_indices]
    class_ids = class_ids[sorted_indices]

    # Filter by confidence threshold
    mask = confidences >= confidence_threshold
    boxes_cxcywh = boxes_cxcywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes_cxcywh) == 0:
        return np.array([]), np.array([]), np.array([])

    # Convert from cxcywh to xyxy format
    boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh)

    # Scale from normalized [0,1] coordinates to pixel coordinates
    boxes_xyxy[:, [0, 2]] *= orig_w
    boxes_xyxy[:, [1, 3]] *= orig_h

    # Clip to image bounds
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

    # Apply NMS (class-aware - see detection_nms)
    keep_indices = detection_nms(boxes_xyxy, confidences, class_ids, nms_threshold)
    return boxes_xyxy[keep_indices], confidences[keep_indices], class_ids[keep_indices]


class RFDETRDetector:
    """
    ONNX Runtime wrapper for RF-DETR object detection models.

    Handles preprocessing, inference, and postprocessing for RF-DETR ONNX models.
    """

    # ImageNet normalization parameters (aliases of the module-level pair the
    # free functions use, so a caller reading them off the class sees the same
    # numbers the preprocess actually applies).
    MEAN = _MEAN
    STD = _STD

    def __init__(
        self,
        model_path: str,
        input_shape: tuple[int, int] = (576, 576),
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        providers: list[str] = [
            "CoreMLExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        sess_options: ort.SessionOptions | None = None,
        classes: list[str] | None = None,
    ):
        """
        Initialize RF-DETR detector.

        Args:
            model_path: Path to ONNX model file
            input_shape: Model input dimensions (height, width)
            confidence_threshold: Minimum confidence for detections
            nms_threshold: IoU threshold for NMS
            providers: ONNX Runtime execution providers (in order of preference)
            sess_options: Optional ONNX Runtime session options
            classes: Optional class names. Only used to locate RF-DETR's trailing
                background logit column; omit it and the column is inferred
                from the logits width instead, so existing callers are unaffected.
        """
        self.model_path = model_path
        self.classes = list(classes or [])
        self.dims = input_shape
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.providers = providers

        if sess_options is None:
            sess_options = ort.SessionOptions()

        self.session = ort.InferenceSession(
            self.model_path, providers=self.providers, sess_options=sess_options
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Store original image dimensions for scaling
        self._orig_height = None
        self._orig_width = None

    def preprocess(self, image: np.ndarray, bgr2rgb: bool = True) -> np.ndarray:
        """
        Preprocess image for RF-DETR inference.

        Args:
            image: Input image (BGR format from cv2)
            bgr2rgb: Whether to convert BGR to RGB

        Returns:
            Preprocessed image tensor (1, C, H, W)
        """
        self._orig_height, self._orig_width = image.shape[:2]
        return preprocess_detection_image(image, self.dims, bgr2rgb)

    def postprocess(
        self, outputs: list[np.ndarray], original_shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Postprocess RF-DETR outputs.

        RF-DETR ONNX outputs:
        - outputs[0]: boxes (batch, num_queries, 4) in normalized cxcywh format
        - outputs[1]: class logits (batch, num_queries, num_classes)

        Args:
            outputs: Raw ONNX model outputs
            original_shape: Original image shape (height, width)

        Returns:
            Tuple of (boxes_xyxy, scores, class_ids)
        """
        return decode_detection_outputs(
            outputs,
            original_shape,
            classes=self.classes,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
        )

    def _nms(
        self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float
    ) -> np.ndarray:
        """
        Apply CLASS-AWARE Non-Maximum Suppression.

        A box only suppresses others of the SAME class, so a legitimately
        overlapping 'person' and 'backpack' both survive. The previous
        class-agnostic pass dropped the lower-scoring box even when it was a
        DIFFERENT object - over-suppression for a multi-class detector. Mirrors
        dispatch.multiclass_nms (the YOLOX path's class-aware NMS), whose comment
        already assumed the RF-DETR wrappers "suppress internally" per-object.

        Args:
            boxes: Bounding boxes (N, 4) in xyxy format
            scores: Confidence scores (N,)
            class_ids: Per-box class ids (N,)
            iou_threshold: IoU threshold for suppression

        Returns:
            Indices of boxes to keep (into the input arrays)
        """
        return detection_nms(boxes, scores, class_ids, iou_threshold)

    def predict(
        self, image: np.ndarray, preprocess: bool = True, postprocess: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run full inference pipeline.

        Args:
            image: Input image (BGR format)
            preprocess: Whether to apply preprocessing
            postprocess: Whether to apply postprocessing

        Returns:
            Tuple of (boxes_xyxy, scores, class_ids)
        """
        original_shape = image.shape[:2]

        if preprocess:
            input_tensor = self.preprocess(image)
        else:
            input_tensor = image

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        if postprocess:
            return self.postprocess(outputs, original_shape)
        return outputs
