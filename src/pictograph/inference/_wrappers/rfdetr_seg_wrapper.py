"""
RF-DETR Instance Segmentation Inference

ONNX Runtime wrapper for RF-DETR instance-segmentation models. Pure ONNX - numpy,
cv2 and onnxruntime only, no dependency on the rfdetr package.

Example usage:
    from inference_wrappers import RFDETRSegDetector

    detector = RFDETRSegDetector(
        model_path=weights,
        classes=class_names,
        input_shape=(576, 576),
        confidence_threshold=0.5,
    )
    boxes, scores, labels, masks = detector.predict(image_bgr)

The caller owns what happens next: both consumers turn these masks into polygons
and Pictograph annotations with their own thresholds, so this module stops at the
arrays.
"""

import logging

import cv2
import numpy as np
import onnxruntime as ort

from .onnx_shape import rfdetr_foreground_columns

logger = logging.getLogger(__name__)


# ImageNet normalization - module level so the free functions below and the
# class share ONE copy. `RFDETRSegDetector.MEAN`/`.STD` alias these.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_segmentation_image(
    image: np.ndarray, dims: tuple[int, int], bgr2rgb: bool = True
) -> np.ndarray:
    """A BGR image → the (1, C, H, W) float32 tensor RF-DETR-Seg's graph takes.

    Free-standing so a caller holding a torch module - which has no ONNX session
    - can build the SAME tensor the graph is fed. `RFDETRSegDetector.preprocess`
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


def seg_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(N, 4) normalized [cx, cy, w, h] → normalized [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def segmentation_nms(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """CLASS-AWARE Non-Maximum Suppression → indices to keep.

    A mask only suppresses others of the SAME class, so a legitimately
    overlapping 'pallet' and 'forklift' both survive. Mirrors the detection
    twin's `detection_nms`.
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
            # +1e-9 avoids 0/0 -> NaN on twin zero-area boxes.
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[iou <= iou_threshold]

    return np.array(keep, dtype=np.int32)


def decode_segmentation_outputs(
    outputs: list[np.ndarray],
    original_shape: tuple[int, int],
    *,
    classes: list[str] | None = None,
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.5,
    mask_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RF-DETR segmentation outputs → (boxes_xyxy, scores, class_ids, masks).

    **The one reduction both engines share.** Same shape as the detection twin
    plus mask resize + threshold: a per-query FOREGROUND argmax, a top-300 sort,
    the confidence gate, cxcywh→xyxy in the ORIGINAL image's pixel space,
    class-aware NMS, and each surviving mask resized to the original frame.

    It exists as a free function, rather than only as
    `RFDETRSegDetector.postprocess`, so the torch runtime can call it too.
    rfdetr's own `PostProcess` does something materially different - a GLOBAL
    top-k over the flattened (Q, C) sigmoid grid - and the two disagreed 3x on
    the same model and image.

    RF-DETR segmentation ONNX outputs (3 tensors, identified by SHAPE because the
    export script's ``[::-1]`` can reverse their names):
    - boxes: (1, N, 4) normalized cxcywh
    - class logits: (1, N, num_classes + 1)
    - masks: (1, N, mask_h, mask_w) raw logits
    """
    orig_h, orig_w = original_shape
    classes = list(classes or [])

    boxes_raw = None
    logits_raw = None
    masks_raw = None

    for out in outputs:
        if out is None:
            continue
        if len(out.shape) == 4:
            # 4D tensor -> masks (1, N, mask_h, mask_w)
            masks_raw = out
        elif len(out.shape) == 3 and out.shape[-1] == 4 and boxes_raw is None:
            # 3D with last dim 4 -> boxes (1, N, 4).
            #
            # `boxes_raw is None` is LOAD-BEARING, not defensive. The
            # graph emits (boxes, logits, masks) POSITIONALLY, and logits is
            # `len(classes) + 1` wide - so a THREE-class model makes logits
            # exactly 4 wide too. Without this guard the logits tensor
            # re-enters this branch, overwrites the boxes, and leaves
            # `logits_raw is None`, which the check below turns into an empty
            # result: a 3-class instance-segmentation model returned ZERO
            # detections on every engine, silently and with no exception.
            # Boxes always arrive first, so first-wins is the correct rule.
            # The keypoint twin has carried this guard from the start.
            boxes_raw = out
        elif len(out.shape) == 3:
            # Remaining 3D -> class logits (1, N, num_classes + 1)
            logits_raw = out

    empty = (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int32),
        np.zeros((0, orig_h, orig_w), dtype=np.uint8),
    )
    if boxes_raw is None or logits_raw is None:
        return empty

    # Remove batch dimensions
    boxes_cxcywh = boxes_raw[0]  # (N, 4)
    logits = logits_raw[0]  # (N, num_classes)
    masks_logits = masks_raw[0] if masks_raw is not None else None  # (N, mask_h, mask_w)

    # Sigmoid on logits -> class probabilities
    scores_all = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    # Get class predictions and confidences over the FOREGROUND columns only -
    # the trailing column is RF-DETR's background slot, not a class.
    n_fg = rfdetr_foreground_columns(scores_all.shape[-1], len(classes))
    foreground = scores_all[:, :n_fg]
    class_ids = np.argmax(foreground, axis=1)
    confidences = np.max(foreground, axis=1)

    # Sort by confidence (descending) and limit to top 300
    sorted_indices = np.argsort(confidences)[::-1][:300]
    boxes_cxcywh = boxes_cxcywh[sorted_indices]
    confidences = confidences[sorted_indices]
    class_ids = class_ids[sorted_indices]
    if masks_logits is not None:
        masks_logits = masks_logits[sorted_indices]

    # Filter by confidence threshold
    keep = confidences >= confidence_threshold
    boxes_cxcywh = boxes_cxcywh[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep].astype(np.int32)
    if masks_logits is not None:
        masks_logits = masks_logits[keep]

    if len(boxes_cxcywh) == 0:
        return empty

    # Convert from cxcywh to xyxy format
    boxes_xyxy = seg_cxcywh_to_xyxy(boxes_cxcywh)

    # Scale from normalized [0,1] coordinates to pixel coordinates
    boxes_xyxy[:, [0, 2]] *= orig_w
    boxes_xyxy[:, [1, 3]] *= orig_h

    # Clip to image bounds
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

    # Apply NMS to suppress overlapping detections (class-aware)
    keep = segmentation_nms(boxes_xyxy, confidences, class_ids, nms_threshold)
    boxes_xyxy = boxes_xyxy[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]
    if masks_logits is not None:
        masks_logits = masks_logits[keep]

    if len(boxes_xyxy) == 0:
        return empty

    # Process masks: resize from model resolution to full original image size
    if masks_logits is not None and len(masks_logits) > 0:
        processed_masks = []
        for mask in masks_logits:
            # Resize mask logits to original image size
            mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            # Threshold logits at mask_threshold (default 0.0 = probability 0.5)
            processed_masks.append((mask_resized > mask_threshold).astype(np.uint8))
        masks = np.array(processed_masks)
    else:
        masks = np.zeros((len(boxes_xyxy), orig_h, orig_w), dtype=np.uint8)

    return boxes_xyxy, confidences, class_ids, masks


class RFDETRSegDetector:
    """
    ONNX Runtime wrapper for RF-DETR instance segmentation models.

    Handles preprocessing, inference, and postprocessing for RF-DETR segmentation ONNX models.
    No dependency on the rfdetr Python package.
    """

    # ImageNet normalization parameters
    MEAN = _MEAN
    STD = _STD

    def __init__(
        self,
        model_path: str,
        classes: list[str],
        input_shape: tuple[int, int] = (576, 576),
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        mask_threshold: float = 0.0,
        providers: list[str] = [
            "CoreMLExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        sess_options: ort.SessionOptions | None = None,
    ):
        self.model_path = model_path
        self.classes = classes
        self.num_classes = len(classes)
        self.dims = input_shape
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.mask_threshold = mask_threshold
        self.providers = providers

        if sess_options is None:
            sess_options = ort.SessionOptions()

        self.session = ort.InferenceSession(
            self.model_path,
            providers=self.providers,
            sess_options=sess_options,
        )

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Auto-detect input shape from ONNX model
        onnx_input_shape = self.session.get_inputs()[0].shape
        if len(onnx_input_shape) == 4:  # [B, C, H, W]
            h, w = onnx_input_shape[2], onnx_input_shape[3]
            if isinstance(h, int) and isinstance(w, int):
                self.dims = (h, w)

        logger.debug("RF-DETR Segmentation model loaded:")
        logger.debug("  Input: %s %s", self.input_name, onnx_input_shape)
        logger.debug("  Target dimensions: %s", self.dims)
        logger.debug("  Outputs: %s", self.output_names)
        logger.debug("  Classes: %s", self.classes)

        # Store original image dimensions for postprocessing
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
        return preprocess_segmentation_image(image, self.dims, bgr2rgb)

    def postprocess(
        self, outputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Postprocess RF-DETR segmentation outputs.

        RF-DETR segmentation ONNX outputs (3 tensors):
        - boxes: (1, N, 4) normalized cxcywh
        - class logits: (1, N, num_classes) raw logits
        - masks: (1, N, mask_h, mask_w) raw logits

        Note: Output names may be reversed due to [::-1] in the export script,
        so we identify outputs by shape rather than name.

        Returns:
            Tuple of (boxes, scores, labels, masks)
                boxes: (N, 4) in original image coordinates [x1, y1, x2, y2]
                scores: (N,) confidence scores
                labels: (N,) class indices
                masks: (N, orig_h, orig_w) binary masks in original image size
        """
        return decode_segmentation_outputs(
            outputs,
            (self._orig_height, self._orig_width),
            classes=self.classes,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            mask_threshold=self.mask_threshold,
        )

    def _nms(
        self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float
    ) -> np.ndarray:
        """
        Apply CLASS-AWARE Non-Maximum Suppression.

        A box only suppresses others of the SAME class, so a legitimately
        overlapping instance of a different class survives. The previous
        class-agnostic pass dropped the lower-scoring box even when it was a
        DIFFERENT object - over-suppression for a multi-class detector. Mirrors
        dispatch.multiclass_nms (the YOLOX path's class-aware NMS).

        Args:
            boxes: Bounding boxes (N, 4) in xyxy format
            scores: Confidence scores (N,)
            class_ids: Per-box class ids (N,)
            iou_threshold: IoU threshold for suppression

        Returns:
            Indices of boxes to keep (into the input arrays)
        """
        return segmentation_nms(boxes, scores, class_ids, iou_threshold)

    def predict(
        self, image: np.ndarray, preprocess: bool = True, postprocess: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | list[np.ndarray]:
        """
        Run full inference pipeline.

        Args:
            image: Input image (BGR format from cv2)
            preprocess: Whether to apply preprocessing
            postprocess: Whether to apply postprocessing

        Returns:
            Tuple of (boxes_xyxy, scores, class_ids, masks)
        """
        if preprocess:
            input_tensor = self.preprocess(image)
        else:
            input_tensor = image

        # Run inference. A real failure here (e.g. an execution provider that
        # can't compile the graph) must propagate - swallowing it into an
        # empty result reports "your model found nothing" instead of the
        # actual error.
        outputs = self.session.run(None, {self.input_name: input_tensor})

        if postprocess:
            return self.postprocess(outputs)
        return outputs
