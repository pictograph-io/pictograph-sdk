"""
Image Classification Inference Module

Unified ONNX model wrapper for all backbone families (ResNet, EfficientNet,
MobileNet, ConvNeXt, ViT) trained with the torchvision training pipeline.

All backbones use identical ImageNet normalization:
- mean = [0.485, 0.456, 0.406]
- std = [0.229, 0.224, 0.225]

SCOPE: this wrapper owns CONSTRUCTION and PREPROCESSING only. Both callers - the
SDK's `dispatch.infer_image` and this service's `_run_classification_batched` -
drive `session.run` themselves so they can batch across images, and each decodes
logits with its own top-k/threshold/class-filter policy. A `predict`,
`predict_batch`, `postprocess` and a `Classification` result model used to live
here to serve a third caller that no longer exists; nothing referenced them, so
they were removed rather than left as a second, silently diverging decode path.
"""

import cv2
import numpy as np
import onnxruntime as ort


class PytorchImageClassifier:
    """
    Unified ONNX image classifier for all backbone families.

    Supports: ResNet, EfficientNet, MobileNet, ConvNeXt, ViT
    All models use identical ImageNet normalization.
    """

    # ImageNet normalization constants (pre-computed as float32 for speed)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    #: Tried in order. A tuple, not a list, because a mutable default is shared
    #: across every instance that omits the argument.
    DEFAULT_PROVIDERS: tuple[str, ...] = (
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    )

    def __init__(
        self,
        model_path: str,
        class_names: list[str],
        input_shape: tuple[int, int] = (224, 224),
        providers: list[str] | None = None,
        sess_options: ort.SessionOptions | None = None,
    ):
        """
        Initialize the image classifier.

        Args:
            model_path: Path to ONNX model file
            class_names: List of class names in order (index = class_id)
            input_shape: Model input shape as (height, width)
            providers: ONNX Runtime execution providers, tried in order.
                Defaults to CUDA -> CoreML -> CPU.
            sess_options: Optional ONNX Runtime session options
        """
        self.model_path = model_path
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.height, self.width = input_shape
        self.providers = list(providers) if providers is not None else list(self.DEFAULT_PROVIDERS)

        if sess_options is None:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.model_path,
            providers=self.providers,
            sess_options=sess_options,
        )

        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image: np.ndarray, bgr2rgb: bool = True) -> np.ndarray:
        """
        Preprocess image for model inference.

        Args:
            image: Input image (H, W, C) in BGR format (uint8)
            bgr2rgb: Whether to convert BGR to RGB

        Returns:
            Preprocessed image (C, H, W) as float32
        """
        if bgr2rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - self.MEAN) / self.STD
        image = image.transpose(2, 0, 1)

        return np.ascontiguousarray(image, dtype=np.float32)

    def __repr__(self) -> str:
        return (
            f"PytorchImageClassifier(model='{self.model_path}', "
            f"classes={self.num_classes}, input=({self.height}, {self.width}))"
        )
