"""ONNX inference wrappers for user-trained models.

Each module wraps one model family (detection, segmentation, classification) with
the pre- and post-processing needed to run its exported ONNX graph and return
Pictograph annotations. Pure ONNX (numpy / cv2 / onnxruntime) - no torch
dependency; execution providers and session options are supplied by the caller at
construction time.
"""

from .classifier_wrapper import PytorchImageClassifier
from .mask_to_polygon import mask_to_all_polygons, mask_to_instance_polygons, mask_to_polygon
from .rfdetr_det_wrapper import RFDETRDetector
from .rfdetr_kp_wrapper import RFDETRKeypointDetector
from .rfdetr_seg_wrapper import RFDETRSegDetector
from .sm_pytorch_wrapper import SemanticSegmentationModelPyTorch
from .yolox_wrapper import YOLOXDetector

__all__ = [
    "PytorchImageClassifier",
    "RFDETRDetector",
    "RFDETRKeypointDetector",
    "RFDETRSegDetector",
    "SemanticSegmentationModelPyTorch",
    "YOLOXDetector",
    "mask_to_all_polygons",
    "mask_to_instance_polygons",
    "mask_to_polygon",
]
