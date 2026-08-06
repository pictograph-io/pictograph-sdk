"""
YOLOX Object Detection Inference

ONNX Runtime wrapper for YOLOX detection models. Pure ONNX - numpy, cv2 and
onnxruntime only, no dependency on the YOLOX training package.

Example usage:
    from inference_wrappers import YOLOXDetector

    detector = YOLOXDetector(model_path=weights, classes=class_names)
    boxes, scores = detector.predict(image_bgr)

The caller decodes these arrays into Pictograph annotations with its own class
map and thresholds, so this module stops at the arrays.
"""

import cv2
import numpy as np
import onnxruntime as ort


class YOLOXDetector:
    def __init__(
        self,
        model_path: str,
        input_shape: tuple[int] = (640, 640),
        confidence: float = 0.6,
        providers: list[str] = [
            "CoreMLExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        sess_options=ort.SessionOptions(),
    ):
        self.model_path: str = model_path
        self.dims: tuple[int] = input_shape
        self.ratio: float = 1.0
        self.confidence_threshold: float = confidence
        self.classes: list[str] = []
        self.providers: list[str] = providers
        self.session = ort.InferenceSession(
            self.model_path,
            providers=self.providers,
            sess_options=sess_options,
        )

    def preprocess(
        self, image: np.ndarray, bgr2rgb: bool = True
    ):  # Note: Flip bgr2rgb: bool = False if performance is poor.
        """Preprocess image for YOLOX model."""
        if len(image.shape) == 3:
            padded_image = np.ones((self.dims[0], self.dims[1], 3), dtype=np.uint8) * 114
        else:
            padded_image = np.ones(self.dims, dtype=np.uint8) * 114

        if bgr2rgb:
            padded_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)

        self.ratio = min(self.dims[0] / image.shape[0], self.dims[1] / image.shape[1])
        resized_image = cv2.resize(
            image,
            (int(image.shape[1] * self.ratio), int(image.shape[0] * self.ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_image[: int(image.shape[0] * self.ratio), : int(image.shape[1] * self.ratio)] = (
            resized_image
        )

        padded_image = padded_image.transpose((2, 0, 1))
        padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)
        return padded_image

    def postprocess(self, outputs, p64=False):
        """Post-process YOLOX model outputs into usable bounding boxes and scores."""
        # `infer_batch` hands this a VIEW into the shared batch buffer
        # (`outputs[i:i+1]`), never a copy - basic numpy slicing aliases the
        # original array. Decoding in place below would mutate that shared
        # buffer through the alias, so copy first.
        outputs = outputs.copy()
        grids = []
        expanded_strides = []
        strides = [8, 16, 32] if not p64 else [8, 16, 32, 64]

        hsizes = [self.dims[0] // stride for stride in strides]
        wsizes = [self.dims[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        outputs = outputs[0]

        boxes = outputs[:, :4]
        scores = outputs[:, 4:5] * outputs[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= self.ratio
        return boxes_xyxy, scores

    def predict(self, image: np.ndarray, preprocess: bool = True, postprocess: bool = True):
        """Run YOLOX detector on an image and return detected bounding boxes and scores."""
        if preprocess:
            image = self.preprocess(image=image)
        onnx_pred = self.session.run(
            None, {self.session.get_inputs()[0].name: np.expand_dims(image, axis=0)}
        )[0]
        if postprocess:
            return self.postprocess(onnx_pred)
        return onnx_pred
