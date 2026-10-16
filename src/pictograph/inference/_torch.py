"""The native-PyTorch inference engine.

The torch twin of the ONNX engine: it downloads a model's native container -
``format="pytorch"`` (the ``.pth``) or ``format="safetensors"`` - rebuilds the EXACT
architecture its training pipeline built, strict-loads the weights, and produces the
same raw annotation dicts the ONNX wrappers produce - so both engines feed one shared
result builder and a caller can swap between them without touching their code.

Rebuild recipes here mirror the training pipelines' own model builders and are
Must stay in sync with them:

============================  ==========================================================
this module                   the pipeline it mirrors
============================  ==========================================================
``_YOLOX_SIZES``              ``pipelines/yolox/train_yolox.py::MODEL_CONFIGS``
                              (the table itself now lives with the vendored
                              architecture, ``._yolox.YOLOX_SIZES``)
``_CLS_HEADS`` + head shape   ``pipelines/classification/train_classification.py``
``_build_smp``                ``pipelines/sm_pytorch/train_semantic_seg.py::create_model``
                              AND ``pictograph_training_service.py`` normalizer defaults
============================  ==========================================================

**Nothing here asks the caller to install a third-party package.** Every framework a
rebuild needs is either declared by the ``[inference]`` extra (torch, torchvision,
safetensors, and ``segmentation-models-pytorch`` PINNED in lockstep with the training
image) or vendored into the wheel (``._rfdetr``, ``._yolox``). That is a hard rule,
pinned by ``tests/unit/test_pytorch_local.py::test_no_hint_names_a_third_party_package``
- which greps this very file, so the rule cannot be quietly relaxed. The app's install
snippet mirrors these ``ImportError`` hints, and for a long time it appended a SECOND
install line naming torch and the segmentation framework directly (and, for YOLOX, a
git URL plus five transitive packages) beneath our own. Those were our undeclared
dependencies handed over as the reader's homework.

Not every pipeline publishes both containers. ``rfdetr_keypoint`` publishes ONNX +
``model.safetensors`` only (its trainer writes no ``checkpoint_best_*.pth``, so
``models.gcs_pytorch_weights_path`` is NULL and ``download(format="pytorch")``
returns 409). Asking for a container the model does not have is REFUSED with the
list of formats it does have - never quietly served from the other one. See
:func:`_fetch_native_weights`.

Preprocessing is deliberately identical to the ONNX wrappers, per family, because
a difference here means the two backends silently disagree about the same image.
The three that used to differ, and now do not:

- **YOLOX is fed BGR**, not RGB. The pipeline subclasses upstream ``yolox.exp.Exp``
  and uses its stock cv2 loader, which performs no channel swap - so BGR is what
  the model trained on and what the ONNX wrapper feeds. Classification and
  semantic-seg DO convert to RGB in training, so those two stay RGB.
- **Resize is bilinear.** ``PIL.Image.resize`` with no ``resample`` is BICUBIC,
  while every training transform and every ONNX wrapper uses bilinear.
- **The classifier returns ``top_k`` classes**, defaulting to the same value the
  ONNX path defaults to, rather than a hardcoded 5.

**Keypoint models do not go through ``rfdetr.predict()`` at all.** That method
returns per-``(query x class)`` hypotheses - its ``PostProcess._select_topk``
flattens ``(Q, C)`` and takes a global top-k, so one query surfaces once per class
it scores on (background included) - and applies its own uncertainty-fusion to the
scores. Measured against the ONNX canon on a real 6-class model, that produced 100
predictions where the canon produced 19, with DIFFERENT class labels on the same
coordinate. So :meth:`TorchEngine._predict_rfdetr_keypoint` runs the module's raw
forward and hands ``pred_boxes`` / ``pred_logits`` / ``pred_keypoints`` to the
canon's own ``decode_keypoint_outputs`` - the same function the ONNX wrapper's
``postprocess`` delegates to, fed the same preprocessed tensor. One reduction, one
implementation, no room to drift.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pictograph._path_safety import safe_path_component
from pictograph.exceptions import ConflictError, NotFoundError
from pictograph.inference._safe_load import safe_torch_load
from pictograph.inference._yolox import YOLOX_SIZES
from pictograph.inference.runtime import Device, empty_device_cache, resolve_torch_device

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

    from pictograph.models.model import Model
    from pictograph.resources.models import Models

__all__ = ["NativeSpec", "TorchEngine", "build_local_torch_engine", "build_torch_engine"]

_LOG = logging.getLogger("pictograph.inference")

# YOLOX depth/width multipliers per size. The table itself lives with the
# VENDORED architecture (`._yolox.YOLOX_SIZES`) - it is part of the rebuild
# recipe, so it belongs next to the modules it parameterises - and is kept in sync
# with the training pipeline's MODEL_CONFIGS. Re-exported here under the old
# private name so the rest of this module reads unchanged.
_YOLOX_SIZES = YOLOX_SIZES

# Torchvision classifier-head placement per backbone - Must stay in sync with the
# classification pipeline's BACKBONE_CONFIGS. `(attr, kind)`; the positional
# index this table used to carry is GONE, because the index was the bug:
# it encoded the LAST Linear, which is the head's input width only for
# EfficientNet. See `_sequential_head_split` for the rule that replaced it.
_CLS_HEADS: dict[str, tuple[str, str]] = {
    "resnet18": ("fc", "linear"),
    "resnet34": ("fc", "linear"),
    "resnet50": ("fc", "linear"),
    "resnet101": ("fc", "linear"),
    "efficientnet_b0": ("classifier", "sequential"),
    "efficientnet_b1": ("classifier", "sequential"),
    "efficientnet_b2": ("classifier", "sequential"),
    "efficientnet_b3": ("classifier", "sequential"),
    "efficientnet_b4": ("classifier", "sequential"),
    "mobilenet_v3_small": ("classifier", "sequential"),
    "mobilenet_v3_large": ("classifier", "sequential"),
    "convnext_tiny": ("classifier", "sequential"),
    "convnext_small": ("classifier", "sequential"),
    "vit_b_16": ("heads", "vit"),
    "vit_b_32": ("heads", "vit"),
}

# ImageNet normalization - the constant every pipeline's eval transform uses.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# The semantic-seg pipeline's defaults, as applied by the training service's config
# normalizer - NOT smp's own defaults. A run submitted without an explicit
# architecture/encoder trains a Segformer + efficientnet-b1, so rebuilding a Unet +
# resnet34 (the previous behaviour) fails the strict load outright.
_SMP_DEFAULT_ARCH = "segformer"
_SMP_DEFAULT_ENCODER = "efficientnet-b1"

_SMP_ARCH_CLASSES = {
    "unet": "Unet",
    "unetplusplus": "UnetPlusPlus",
    "unet++": "UnetPlusPlus",
    "segformer": "Segformer",
}

# RF-DETR variant class names, per model_type - Must stay in sync with each pipeline's
# own MODEL_CLASSES map (`pipelines/rfdetr_{detection,segmentation,keypoint}/`)
# and with rfdetr's `_CHECKPOINT_MODEL_NAME_CLASS_SYMBOLS`. Only consulted when
# rebuilding from safetensors, where the class name has to be supplied because
# the bare tensors carry no `model_name`. A wrong pick is CAUGHT, not guessed
# past - see `_verify_rfdetr_load`.
_RFDETR_SIZE_LABELS = (
    "2xlarge",
    "xxlarge",
    "xlarge",
    "preview",
    "nano",
    "small",
    "medium",
    "large",
    "base",
)
_RFDETR_VARIANT_PREFIX = {
    "instance_segmentation": "RFDETRSeg",
    # RF-DETR ships keypoint as ONE preview variant (rfdetr 1.8.3); the size is
    # therefore fixed rather than parsed.
    "keypoint_detection": "RFDETRKeypoint",
    "object_detection": "RFDETR",
}
_RFDETR_DEFAULT_SIZE = {"keypoint_detection": "preview"}


def _require(module: str, hint: str) -> Any:
    """Import an optional framework, or raise with the exact install command."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the local env
        raise ImportError(hint) from exc


def _require_torch() -> Any:
    return _require(
        "torch",
        "Local PyTorch inference needs torch. Install it with:\n"
        '    pip install "pictograph[inference]"',
    )


def _require_safetensors() -> Any:
    return _require(
        "safetensors.torch",
        "This model publishes its native weights as safetensors. Install the "
        "local-inference extra:\n"
        '    pip install "pictograph[inference]"',
    )


class TorchEngine:
    """Runs a rebuilt ``torch`` model and emits raw annotation dicts.

    Not constructed directly - :func:`build_torch_engine` builds it, and one of the
    task model classes in :mod:`pictograph.inference.models` wraps it.
    """

    backend = "pytorch"

    def __init__(
        self,
        *,
        module: Any,
        family: str,
        device: str,
        dtype: Any,
        checkpoint_path: Path,
        model_type: str,
        architecture: str,
        classes: list[str],
        input_size: tuple[int, int],
        num_keypoints_per_class: list[int] | None = None,
        keypoint_names: dict[str, list[str]] | None = None,
        skeleton_edges: dict[str, list[list[int]]] | None = None,
        keypoint_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.module = module
        self.family = family
        self.device = device
        self.dtype = dtype
        self.providers: list[str] = []
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.architecture = architecture
        self.classes = classes
        self._input_size = input_size
        # The LOAD-TIME threshold the decode gates on, mirroring the ONNX engine
        # exactly: `build_onnx_engine(confidence=...)` becomes the wrapper's
        # `confidence_threshold`, while `predict(confidence=...)` gates the shared
        # emitter. Both engines therefore reduce the same raw queries to the same
        # detections; a per-call value only ever narrows what is emitted.
        self.confidence_threshold = confidence_threshold
        # Keypoint schema - the ground truth for per-class arity and joint CLASS
        # names, so `self` can stand in as the `wrapper` the shared
        # `_keypoint_to_annotations` reads (a joint is a class, so the names are
        # what each emitted point is called). Without them a pose model's joints
        # come back as `point_0..point_N`. `skeleton_edges` is the class TEMPLATE's
        # connectivity - model metadata a consumer uses to DRAW an instance's
        # points, never stamped onto an annotation.
        self.num_keypoints_per_class = list(num_keypoints_per_class or [])
        self.keypoint_names = dict(keypoint_names or {})
        self.skeleton_edges = dict(skeleton_edges or {})
        self.keypoint_threshold = keypoint_threshold
        # The rebuilt modules carry per-call state and torch modules are not
        # re-entrant, so one engine serializes its own inference - matching the
        # ONNX engine's documented guarantee.
        self._lock = threading.Lock()

    def node_names_for(self, class_name: str, count: int) -> list[str]:
        """A keypoint class's joint names, positionally padded - mirrors the ONNX
        wrapper's method so `_keypoint_to_annotations` treats `self` as a wrapper.

        Delegates to the canon's own ``keypoint_node_names`` when the vendored
        wrappers are importable, so the padding rule cannot drift; falls back to an
        inline copy when they are not (the base gate installs no ``[inference]``
        extra, and this method is pure-python coverage there)."""
        try:
            from ._wrappers.rfdetr_kp_wrapper import keypoint_node_names
        except ImportError:  # pragma: no cover - only without the [inference] extra
            names = list(self.keypoint_names.get(class_name) or [])
            if len(names) < count:
                names += [f"point_{i}" for i in range(len(names), count)]
            return names[:count]
        return keypoint_node_names(self.keypoint_names, class_name, count)

    def __repr__(self) -> str:
        return (
            f"TorchEngine(family={self.family!r}, device={self.device!r}, "
            f"type={self.model_type!r}, classes={len(self.classes)})"
        )

    # ───────────── public entry points ─────────────

    def infer(self, image_bgr: Any, *, confidence: float, top_k: int = 1) -> dict[str, Any]:
        """One BGR numpy image → the raw annotation dict the result builder parses."""
        if self.module is None:
            raise RuntimeError("This model has been closed and can no longer predict.")
        with self._lock:
            # RF-DETR bypasses `rfdetr.predict()` entirely - see the module
            # docstring. Every family takes the RAW BGR array, because the canon's
            # own preprocess is what builds their tensor.
            if self.model_type == "keypoint_detection":
                return self._predict_rfdetr_keypoint(image_bgr, confidence)
            if self.family == "rfdetr":
                return self._predict_rfdetr(image_bgr, confidence)
            if self.family == "yolox":
                return self._predict_yolox(image_bgr, confidence)
            if self.family == "segmentation_models_pytorch":
                return self._predict_smp(_bgr_to_pil(image_bgr), confidence)
            return self._predict_classifier(_bgr_to_pil(image_bgr), confidence, top_k)

    def infer_batch(
        self, images_bgr: list[Any], *, confidence: float, top_k: int = 1
    ) -> list[dict[str, Any]]:
        """Several BGR images. Batches where the family allows it, else loops."""
        if not images_bgr:
            return []
        if len(images_bgr) == 1:
            return [self.infer(images_bgr[0], confidence=confidence, top_k=top_k)]
        if self.family == "torchvision":
            if self.module is None:
                raise RuntimeError("This model has been closed and can no longer predict.")
            with self._lock:
                return self._predict_classifier_batch(
                    [_bgr_to_pil(i) for i in images_bgr], confidence, top_k
                )
        return [self.infer(i, confidence=confidence, top_k=top_k) for i in images_bgr]

    def close(self) -> None:
        """Drop the module and release its device memory."""
        self.module = None
        empty_device_cache(self.device)

    # ───────────── tensor helpers ─────────────

    def _tensor(self, chw: Any) -> Any:
        """A (C,H,W) float array → a batched tensor on the model's device + dtype.

        This is the fix for the defect that made every non-CPU load unusable: the
        module was moved to CUDA/MPS but its input was left on the CPU, so the
        first forward raised a device mismatch.
        """
        torch = _require_torch()
        import numpy as np

        return (
            torch.from_numpy(np.ascontiguousarray(chw))
            .unsqueeze(0)
            .to(device=self.device, dtype=self.dtype)
        )

    def _forward(self, tensor: Any) -> Any:
        torch = _require_torch()

        with torch.inference_mode():
            return self.module(tensor)

    # ───────────── per-family inference ─────────────

    def _predict_rfdetr(self, image_bgr: Any, conf: float) -> dict[str, Any]:
        """RF-DETR detection / instance segmentation: raw forward → the CANONICAL
        ONNX reduction.

        Like the keypoint twin below, this deliberately does NOT call
        ``rfdetr``'s ``predict()``. That method's postprocessor
        (``PostProcess._select_topk``) flattens the ``(Q, C)`` sigmoid grid and
        takes a GLOBAL top-k, so one query is returned once per class it scores
        on. The server never did that - it takes a per-query FOREGROUND argmax -
        and the two disagreed on the same model and image by 2-3x: measured on
        the ``fixture-rfdetr_detection`` / ``fixture-rfdetr_segmentation`` models
        at conf=0.05, **126 detections vs the canon's 62** and **6 vs 2**.

        It also resized with torchvision's antialiased bilinear where the canon
        resizes with cv2 ``INTER_LINEAR`` - the same class of divergence already
        removed from the other three families, and the last place it survived.

        So: preprocess with the canon's own cv2 preprocess, run the module's raw
        forward (the exact tensors the ONNX graph exports), and hand them to the
        canon's own decode. Same input, same reduction, same shared emitter; the
        two backends cannot disagree without a change to the one function they
        share.
        """
        import numpy as np

        from ._wrappers.dispatch import _instance_seg_to_annotations
        from ._wrappers.rfdetr_det_wrapper import (
            decode_detection_outputs,
            preprocess_detection_image,
        )
        from ._wrappers.rfdetr_seg_wrapper import (
            decode_segmentation_outputs,
            preprocess_segmentation_image,
        )

        segmentation = self.model_type == "instance_segmentation"
        torch = _require_torch()
        inner = _rfdetr_inner(self.module)
        if inner is None or not callable(inner):  # pragma: no cover - defensive
            raise RuntimeError(
                "This RF-DETR model exposes no inner module to run - its "
                "checkpoint did not rebuild into a callable rfdetr model."
            )
        # rfdetr's own predict() re-asserts eval() on every call because training
        # reassigns the module; do the same, or dropout runs during inference.
        inner.eval()

        source = np.asarray(image_bgr)
        preprocess = preprocess_segmentation_image if segmentation else preprocess_detection_image
        chw = preprocess(source, self._input_size)[0]
        with torch.inference_mode():
            raw = inner(self._tensor(chw))

        outputs = _rfdetr_raw_outputs(raw)
        if outputs is None:  # pragma: no cover - depends on rfdetr version
            _LOG.warning(
                "RF-DETR forward returned %s, which carries no "
                "pred_boxes/pred_logits - returning no predictions.",
                type(raw).__name__,
            )
            return {"model_type": self.model_type, "predictions": []}
        # A detection model has no third head; the decode identifies its tensors
        # by SHAPE, so a `None` in the list would be an unreadable `.shape`.
        outputs = [o for o in outputs if o is not None]

        original_shape = (int(source.shape[0]), int(source.shape[1]))
        # `nms_threshold` (and `mask_threshold`) are deliberately LEFT AT THE
        # SHARED DEFAULT rather than restated here: the ONNX wrapper's own
        # defaults are the same constants and no caller overrides them, so
        # passing one would be a second place to change.
        masks: Any = None
        if segmentation:
            boxes, scores, class_ids, masks = decode_segmentation_outputs(
                outputs,
                original_shape,
                classes=self.classes,
                confidence_threshold=self.confidence_threshold,
            )
        else:
            boxes, scores, class_ids = decode_detection_outputs(
                outputs,
                original_shape,
                classes=self.classes,
                confidence_threshold=self.confidence_threshold,
            )

        # Delegate to the SHARED emitter so both backends produce byte-identical
        # dicts (same `attributes`, same confidence handling, same polygon fallback).
        preds = _instance_seg_to_annotations(
            boxes,
            scores,
            class_ids,
            list(masks) if masks is not None else None,
            self.classes,
            None,
            conf,
            None,
        )
        return {"model_type": self.model_type, "predictions": preds}

    def _predict_rfdetr_keypoint(self, image_bgr: Any, conf: float) -> dict[str, Any]:
        """RF-DETR keypoint: raw forward → the CANONICAL ONNX reduction.

        This deliberately does NOT call ``rfdetr.predict()``. That method's
        postprocessor (``PostProcess._select_topk``) flattens ``(Q, C)`` sigmoid
        probabilities and takes a GLOBAL top-k, so a single query is returned once
        per class it scores on - including the background column - and it then
        rescales the surviving scores by a keypoint-uncertainty factor. None of
        that is what the server does. Measured against the ONNX canon on a real
        6-class model at conf=0.05: 100 predictions vs the canon's 19, with the
        same coordinate arriving under several different class labels.

        So: preprocess with the canon's own :func:`preprocess_keypoint_image`, run
        the module's raw forward (the exact three tensors the ONNX graph exports),
        and hand them to the canon's own :func:`decode_keypoint_outputs` - a
        per-query foreground argmax plus class-aware NMS. Same input, same
        reduction, same shared emitter; the two backends cannot disagree without a
        change to the one function they share.
        """
        import numpy as np

        from ._wrappers.dispatch import _keypoint_to_annotations
        from ._wrappers.rfdetr_kp_wrapper import (
            decode_keypoint_outputs,
            preprocess_keypoint_image,
        )

        torch = _require_torch()
        inner = _rfdetr_inner(self.module)
        if inner is None or not callable(inner):  # pragma: no cover - defensive
            raise RuntimeError(
                "This RF-DETR keypoint model exposes no inner module to run - its "
                "checkpoint did not rebuild into a callable rfdetr model."
            )
        # rfdetr's own predict() re-asserts eval() on every call because training
        # reassigns the module; do the same, or dropout runs during inference.
        inner.eval()

        source = np.asarray(image_bgr)
        chw = preprocess_keypoint_image(source, self._input_size)[0]
        with torch.inference_mode():
            raw = inner(self._tensor(chw))

        outputs = _rfdetr_raw_outputs(raw)
        if outputs is None:  # pragma: no cover - depends on rfdetr version
            _LOG.warning(
                "RF-DETR keypoint forward returned %s, which carries no "
                "pred_boxes/pred_logits - returning no predictions.",
                type(raw).__name__,
            )
            return {"model_type": self.model_type, "predictions": []}

        # `nms_threshold` is deliberately LEFT AT THE SHARED DEFAULT rather than
        # restated here: the ONNX wrapper's own default is the same constant and
        # no caller overrides it, so passing one would be a second place to change.
        boxes, scores, class_ids, keypoints = decode_keypoint_outputs(
            outputs,
            (int(source.shape[0]), int(source.shape[1])),
            classes=self.classes,
            num_keypoints_per_class=self.num_keypoints_per_class,
            confidence_threshold=self.confidence_threshold,
        )
        preds = _keypoint_to_annotations(
            boxes, scores, class_ids, keypoints, self, self.classes, None, conf
        )
        return {"model_type": self.model_type, "predictions": preds}

    def _predict_yolox(self, image_bgr: Any, conf: float) -> dict[str, Any]:
        """YOLOX letterbox → forward → shared decode.

        Fed BGR, matching training (upstream's stock cv2 loader does no channel
        swap) and matching the ONNX wrapper. Feeding RGB here silently swapped red
        and blue relative to the other backend.
        """
        import numpy as np

        from ._wrappers import dispatch

        in_h, in_w = self._input_size
        src = np.asarray(image_bgr, dtype=np.uint8)
        ratio = min(in_h / src.shape[0], in_w / src.shape[1])
        new_h, new_w = int(src.shape[0] * ratio), int(src.shape[1] * ratio)
        padded = np.full((in_h, in_w, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = _resize_bgr(src, new_w, new_h)
        tensor = self._tensor(padded.transpose(2, 0, 1).astype(np.float32))

        out = self._forward(tensor)
        out = out[0] if isinstance(out, (tuple, list)) else out
        decoded = out.float().cpu().numpy()[0]  # (anchors, 5 + C): cx cy w h obj cls…

        boxes = decoded[:, :4]
        scores_mat = decoded[:, 4:5] * decoded[:, 5:]
        boxes_xyxy = np.empty_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio

        preds = dispatch._yolox_to_annotations(boxes_xyxy, scores_mat, self.classes, None, conf)
        return {"model_type": self.model_type, "predictions": preds}

    def _predict_smp(self, pil: Any, conf: float) -> dict[str, Any]:
        import numpy as np

        from ._wrappers.dispatch import _semantic_seg_to_annotations, semantic_masks_from_logits

        in_h, in_w = self._input_size
        orig_w, orig_h = pil.size
        chw = _normalized_chw(pil, in_w, in_h)
        out = self._forward(self._tensor(chw))
        logits = out[0].float().cpu().numpy()  # (C, H, W) or (1, H, W)

        # Mirror the ONNX wrapper EXACTLY: upscale each raw output channel to the
        # original resolution first (bilinear, preserving the value distribution),
        # then argmax across channels - channel 0 is background for multi-class.
        full = np.stack(
            [_resize_channel(logits[i], orig_w, orig_h) for i in range(logits.shape[0])], axis=0
        )
        masks, prob_maps = semantic_masks_from_logits(full, conf)
        preds = _semantic_seg_to_annotations(masks, self.classes, None, prob_maps)
        return {"model_type": self.model_type, "predictions": preds}

    def _predict_classifier(self, pil: Any, conf: float, top_k: int) -> dict[str, Any]:
        in_h, in_w = self._input_size
        out = self._forward(self._tensor(_normalized_chw(pil, in_w, in_h)))
        return self._classification_result(out[0].float().cpu().numpy(), conf, top_k)

    def _predict_classifier_batch(
        self, pils: list[Any], conf: float, top_k: int
    ) -> list[dict[str, Any]]:
        """Stack the batch into ONE forward - the real throughput win for classifiers."""
        torch = _require_torch()
        import numpy as np

        in_h, in_w = self._input_size
        stacked = np.stack([_normalized_chw(p, in_w, in_h) for p in pils], axis=0)
        tensor = torch.from_numpy(np.ascontiguousarray(stacked)).to(
            device=self.device, dtype=self.dtype
        )
        out = self._forward(tensor)
        logits = out.float().cpu().numpy()
        return [self._classification_result(logits[i], conf, top_k) for i in range(len(pils))]

    def _classification_result(self, logits: Any, conf: float, top_k: int) -> dict[str, Any]:
        from ._wrappers.dispatch import _classification_to_result

        return _classification_to_result(logits, self.classes, conf, top_k, self.model_type)


# ───────────── loader ─────────────


@dataclass(frozen=True)
class NativeSpec:
    """Everything rebuilding a checkpoint needs, independent of WHERE it came from.

    The seam that makes the two loaders twins on the native formats. A ``.pth`` is
    rebuilt from its training pipeline's own model definition, and that definition is
    selected by the task, the architecture label and the training config - three
    facts that live on the API's model record AND in the ``config.json`` a pipeline
    writes beside the weights. Naming them once, here, is what lets
    :func:`build_torch_engine` (online, from the record) and
    :func:`build_local_torch_engine` (offline, from the file) share every line of the
    actual rebuild instead of growing a second copy that drifts.
    """

    model_type: str
    architecture: str
    training_config: dict[str, Any]
    classes: list[str]
    name: str


def build_torch_engine(
    model: Model,
    *,
    models: Models,
    weight_format: Literal["pytorch", "safetensors"] = "safetensors",
    cache_dir: Path,
    device: Device = "auto",
    keypoint_schema: dict[str, Any] | None = None,
    input_size: tuple[int, int] | None = None,
    confidence: float = 0.5,
) -> TorchEngine:
    """Download, rebuild and strict-load a model's native checkpoint.

    ``weight_format`` names WHICH native container to rebuild from - ``.pth`` or
    ``model.safetensors``. The two hold the same tensors and produce the same
    module; see :func:`_fetch_native_weights` for why they are nonetheless a real
    choice, and why one is never substituted for the other.

    ``confidence`` is the LOAD-TIME threshold, and exists for the same reason
    ``build_onnx_engine`` takes one: the RF-DETR keypoint decode gates the raw
    queries before they ever reach the shared emitter, so both engines must be
    handed the same number or they reduce the same image differently.
    """
    if model.status != "ready":
        raise ValueError(
            f"Model {model.name!r} is {model.status!r}, not 'ready' - it can't be loaded yet."
        )
    spec = NativeSpec(
        model_type=model.model_type,
        architecture=model.architecture or "",
        training_config=model.training_config or {},
        classes=_classes_of(model),
        name=model.name,
    )
    weights, ckpt = _fetch_native_weights(
        model, models=models, cache_dir=cache_dir, weight_format=weight_format
    )
    return _engine_from_checkpoint(
        weights,
        ckpt,
        spec,
        device=device,
        cache_dir=cache_dir,
        keypoint_schema=keypoint_schema,
        input_size=input_size,
        confidence=confidence,
    )


def build_local_torch_engine(
    weights: Path,
    *,
    weight_format: Literal["pytorch", "safetensors"],
    spec: NativeSpec,
    cache_dir: Path,
    device: Device = "auto",
    keypoint_schema: dict[str, Any] | None = None,
    confidence: float = 0.5,
) -> TorchEngine:
    """Rebuild a checkpoint that is already on disk - the fully offline twin.

    Identical to :func:`build_torch_engine` from the checkpoint onwards; it differs
    only in not fetching anything. ``spec`` comes from the ``config.json`` written
    beside the weights instead of from the API's model record, which is the only
    thing the network was ever needed for.
    """
    return _engine_from_checkpoint(
        weights,
        _load(weights, weight_format),
        spec,
        device=device,
        cache_dir=cache_dir,
        keypoint_schema=keypoint_schema,
        input_size=None,
        confidence=confidence,
    )


def _engine_from_checkpoint(
    weights: Path,
    ckpt: Any,
    spec: NativeSpec,
    *,
    device: Device,
    cache_dir: Path,
    keypoint_schema: dict[str, Any] | None,
    input_size: tuple[int, int] | None,
    confidence: float,
) -> TorchEngine:
    """Rebuild the module from a loaded checkpoint and put it on ``device``.

    The ONE rebuild, shared by the online and offline entry points so a ``.pth``
    cannot behave differently depending on which door it came through.
    """
    torch = _require_torch()
    classes = list(spec.classes)
    config = spec.training_config

    family = _resolve_family(spec.model_type, spec.architecture, ckpt)
    if family == "rfdetr":
        # rfdetr can only be rebuilt from a `.pth` container (RFDETR.from_checkpoint
        # reads `args` / `model_name` off it and hands the PATH back to its own
        # weight loader). A safetensors artifact is wrapped into that container
        # once, in the cache - see `_rfdetr_container`.
        container = _rfdetr_container(weights, ckpt, spec, classes, cache_dir)
        module = _build_rfdetr(container)
        if container is not weights:
            _verify_rfdetr_load(module, ckpt, container)
            # A safetensors artifact is rebuilt through a container we SYNTHESIZE
            # from the record, so "the artifact decides the resolution" is vacuous
            # on that path unless it is checked: rfdetr silently interpolates the
            # trained position embeddings to whatever `model_config.resolution`
            # asks for. Measured on this fixture pair - a record claiming 576 for
            # a model trained at 312 loaded, predicted, and returned 3 detections
            # where the `.pth` returned 1. Re-derive what the STATE DICT
            # was trained at and prefer it.
            module = _rebuild_rfdetr_at_artifact_resolution(
                module, ckpt, weights, spec, classes, cache_dir
            )
        # The checkpoint's own class list is the training truth - prefer it over
        # the record's class_mapping when present (the same "artifact beats config"
        # rule the ONNX loader applies to the input shape). A safetensors artifact
        # carries no names at all, so there the record IS the only source - which
        # is why the same box came back as `Forklift` from one container and the
        # record's label from the other. One rule, stated once: artifact when it
        # has an answer, record otherwise, and a disagreement is never silent.
        ckpt_names = _rfdetr_ckpt_class_names(ckpt)
        if ckpt_names:
            if list(spec.classes) and list(spec.classes) != ckpt_names:
                _LOG.warning(
                    "Class names disagree: this checkpoint was trained on %s but the "
                    "model record says %s. Using the checkpoint's - it is what the "
                    "weights actually predict. A record-only container (safetensors) "
                    "would have used the record's names and mislabelled every box.",
                    ckpt_names,
                    list(spec.classes),
                )
            classes = ckpt_names
    elif family == "yolox":
        module = _build_yolox(ckpt, config, spec.architecture, len(classes))
    elif family == "segmentation_models_pytorch":
        module = _build_smp(ckpt, config, spec.architecture, classes)
    else:
        module = _build_torchvision(ckpt, config, spec.architecture, len(classes))

    resolved_device = resolve_torch_device(device)
    resolved_size = input_size
    if family == "rfdetr":
        # rfdetr wraps its nn.Module twice; read the dtype and the trained
        # resolution off the LIVE module rather than the wrapper (the wrapper has
        # no `.parameters()`, so an fp16 model otherwise reported float32 and its
        # first forward raised "expected Half but found Float").
        inner = _rfdetr_inner(module)
        dtype = _checkpoint_dtype(inner if inner is not None else module, torch)
        resolved_size = resolved_size or _rfdetr_module_resolution(module)
        _move_rfdetr(module, resolved_device)
    else:
        dtype = _checkpoint_dtype(module, torch)
        module = module.to(device=resolved_device, dtype=dtype)
        module.eval()

    kp_counts = (
        _rfdetr_ckpt_num_keypoints_per_class(ckpt, len(classes))
        if spec.model_type == "keypoint_detection"
        else None
    )
    schema = keypoint_schema or {}
    return TorchEngine(
        module=module,
        family=family,
        device=resolved_device,
        dtype=dtype,
        checkpoint_path=weights,
        model_type=spec.model_type,
        architecture=spec.architecture,
        classes=classes,
        input_size=resolved_size or _pytorch_input_size(spec.model_type, config),
        num_keypoints_per_class=schema.get("num_keypoints_per_class") or kp_counts,
        keypoint_names=schema.get("keypoint_names"),
        skeleton_edges=schema.get("skeleton"),
        confidence_threshold=confidence,
    )


def _cache_stem(model: Model) -> str:
    """Cache key for a model's weights - id PLUS the version it currently serves.

    Keying on the id alone means a retrained (or rolled-back) model keeps predicting
    with whatever was downloaded first, forever, on that machine.
    """
    version = getattr(model, "current_version_id", None) or getattr(model, "updated_at", None)
    # Both halves are SERVER-SUPPLIED, so both are reduced to a safe component
    # before they become a filename. See pictograph._path_safety.
    ident = safe_path_component(model.id, fallback="model")
    if version is None:
        return ident
    return f"{ident}-{safe_path_component(str(version)[:32], fallback='v')}"


def _load_checkpoint(weights: Path, *, allow_unsafe_pickle: bool = False) -> Any:
    """Read a checkpoint WITHOUT executing pickled code.

    Delegates to :func:`pictograph.inference._safe_load.safe_torch_load`, which is
    the single place any checkpoint is read. See that module for why the previous
    automatic ``weights_only=False`` fallback was removed.
    """
    _require_torch()
    return safe_torch_load(weights, allow_unsafe_pickle=allow_unsafe_pickle)


def _load_safetensors(weights: Path) -> dict[str, Any]:
    """Load a ``.safetensors`` artifact into a plain state dict.

    safetensors is data-only by construction - there is no pickle and therefore
    no arbitrary-code path, which is exactly why `_load_checkpoint`'s
    ``weights_only`` dance has no counterpart here.
    """
    safetensors_torch = _require_safetensors()
    loaded: dict[str, Any] = safetensors_torch.load_file(str(weights), device="cpu")
    return loaded


def _fetch_native_weights(
    model: Model,
    *,
    models: Models,
    cache_dir: Path,
    weight_format: Literal["pytorch", "safetensors"] = "safetensors",
) -> tuple[Path, Any]:
    """The model's native weights on disk, plus the loaded object.

    **The requested container is the one fetched. There is no fallback.** A model
      that does not publish it is refused with a message naming what it DOES publish
      - quietly handing back the other container would mean ``format="pytorch"`` and
      ``format="safetensors"`` could return the same bytes, which makes the argument
      a label rather than a selection.

      Why the two containers are a genuine choice and not an implementation detail:

      * ``model.safetensors`` is published ONLY after a publish-BLOCKING parity gate
        has compared it against that version's ONNX graph; a version whose gate
        failed ships no safetensors at all. It is *verified to be the model the graph
        runs*, which is why it is the recommended native format.
      * The ``.pth`` is the raw training checkpoint and has never been gated.
        Formerly ``rfdetr_detection`` glob-picked it out of ``output_dir`` (best/EMA)
        while exporting its ONNX from the live post-train wrapper, so the two
        published artifacts of one version encoded DIFFERENT models - measured on a
        real published pair (``b148-det-med-b16``, RFDETRMedium, fp32) against its
        own published ONNX on three real dataset images: max|delta| 0.63 / 0.82 /
        0.95 across 24 / 37 / 25 SURVIVING boxes, i.e. essentially the full dynamic
        range. The pipeline is fixed going forward; models trained before the fix
        still carry that ``.pth`` in storage, so ``format="safetensors"`` is the one
        that reproduces the graph on them.

      Not every version has both. ``rfdetr_keypoint`` publishes ONNX +
      ``model.safetensors`` only - its trainer finds no
      ``checkpoint_best_{total,regular,ema}.pth`` / ``checkpoint.pth`` in
      ``output_dir``, so ``gcs_pytorch_weights_path`` is NULL and the wire
      ``format="pytorch"`` answers 409. That refusal now reaches the caller naming
      ``safetensors`` as the format that exists.

      Both containers cache under the SAME version-aware stem (`_cache_stem`), so a
      retrained or rolled-back model re-downloads rather than predicting forever with
      whatever landed first.
    """
    stem = _cache_stem(model)
    cached = cache_dir / f"{stem}{'.safetensors' if weight_format == 'safetensors' else '.pth'}"
    if cached.exists():
        return cached, _load(cached, weight_format)

    try:
        if weight_format == "safetensors":
            # Resolved by FORMAT, never by the literal name
            # `model.safetensors`. That name was hardcoded here because every
            # model published under it, which is the same fact that made it a
            # cache-collision hazard; artifacts are now named after their model
            # (`my-model.safetensors`), so a by-name fetch would 404 on
            # everything trained from 2026-07-31 on. The `format=` route
            # resolves `gcs_safetensors_path` off the version row and is
            # therefore correct for BOTH naming eras.
            models.download(model_id=model.id, output_path=cached, format="safetensors")
        else:
            models.download(model_id=model.id, output_path=cached, format="pytorch")
    except (ConflictError, NotFoundError) as exc:
        raise _missing_native_format(model, weight_format, exc, models) from exc
    return cached, _load(cached, weight_format)


def _load(weights: Path, weight_format: Literal["pytorch", "safetensors"]) -> Any:
    """Read a native container with the loader that container needs."""
    if weight_format == "safetensors":
        return _load_safetensors(weights)
    return _load_checkpoint(weights)


def _missing_native_format(
    model: Model, weight_format: str, exc: Exception, models: Models
) -> ConflictError:
    """The typed error for a native format this model does not publish.

    Stays a :class:`~pictograph.exceptions.ConflictError` (409) and NAMES THE
    FORMATS THAT EXIST, read off the model's own files manifest. "This model has no
    .pth" is a dead end; "this model publishes onnx, safetensors" is the next call.
    A manifest lookup that itself fails degrades to the bare reason rather than
    inventing a list - a wrong list is worse than none.
    """
    available = _available_formats(models, model)
    alternatives = (
        f" This model publishes: {', '.join(available)}."
        + (
            f" Load it with format={available[0]!r}."
            if len(available) == 1
            else " Pass one of those as format=."
        )
        if available
        else ""
    )
    return ConflictError(
        f"Model {model.name!r} publishes no {weight_format!r} weights ({exc}).{alternatives}",
        status_code=409,
        fix=(
            "Formats are never substituted for one another - ask for one the model "
            "actually has, or retrain/rebuild to publish this one."
        ),
    )


#: Files-manifest ``format`` token → the SDK's ``format=`` value. The manifest speaks
#: the wire's vocabulary (§ 5.2 of the artifact contract); this is the same
#: translation ``runtime.wire_format`` does, read in the other direction. Rows whose
#: format is not a loadable weight (``json``, ``markdown``) are simply absent here.
#: Insertion order IS the order alternatives are listed in - the product's, matching
#: ``runtime.WEIGHT_FORMATS``.
_FORMAT_FROM_MANIFEST = {
    "pytorch": "pytorch",
    "safetensors": "safetensors",
    "pte": "pytorch_engine",
    "onnx": "onnx",
    "engine": "tensorrt_engine",
}


def _available_formats(models: Models, model: Model) -> list[str]:
    """Which ``format=`` values this model actually has, for the version it serves.

    Scoped to the effective version, because "this model has a .pth" is false and
    misleading when the ``.pth`` belongs to a version that is no longer served. If
    no row carries that version id, every row is considered rather than reporting
    nothing.

    Deliberately total: ANY failure (an older backend, a transport error, a stub
    without ``files``) yields ``[]`` and the caller omits the list. This runs only
    on an already-failing path, so it must never turn a clear 409 into a traceback.
    """
    try:
        entries = list(models.files(model_id=model.id).files)
    except Exception as exc:  # advisory only - never the failure the caller sees
        _LOG.debug("Could not list %s's artifacts to name the alternatives (%s).", model.name, exc)
        return []
    version = getattr(model, "current_version_id", None)
    scoped = [e for e in entries if getattr(e, "version_id", None) == version] if version else []
    seen = {
        _FORMAT_FROM_MANIFEST[entry.format]
        for entry in (scoped or entries)
        if getattr(entry, "format", None) in _FORMAT_FROM_MANIFEST
    }
    return [fmt for fmt in _FORMAT_FROM_MANIFEST.values() if fmt in seen]


def _rfdetr_variant(spec: NativeSpec) -> tuple[str, str]:
    """``(RFDETR class name, pretrain_weights token)`` for this model.

    rfdetr resolves its variant class from the checkpoint's ``model_name``, or -
    as a fallback - by substring-matching ``args.pretrain_weights`` against its
    own ``_CHECKPOINT_MODEL_MAP_ENTRIES``. A safetensors artifact carries
    neither, so both are synthesised here. Both are written, so a future rename
    on either side still resolves.

    The SIZE comes from the architecture label first, because
    the training service ALWAYS stamps it there for the RF-DETR pipelines
    ("RF-DETR Seg Medium", "RF-DETR Keypoint Preview"), and only then from
    ``training_config["model_size"]``.
    """
    model_type = spec.model_type
    prefix = _RFDETR_VARIANT_PREFIX.get(model_type, "RFDETR")
    label = f"{spec.architecture} {spec.training_config.get('model_size') or ''}"
    label = label.strip().lower()
    size = next((s for s in _RFDETR_SIZE_LABELS if s in label), None)
    if size is None:
        size = _RFDETR_DEFAULT_SIZE.get(model_type, "medium")
    if model_type == "keypoint_detection":
        # One keypoint variant exists (rfdetr 1.8.3) and its map key is the
        # compound "keypoint-preview"; do not synthesise sizes it has no class for.
        return "RFDETRKeypointPreview", "rf-detr-keypoint-preview.pth"
    token_family = "seg-" if model_type == "instance_segmentation" else ""
    return f"{prefix}{size.title()}", f"rf-detr-{token_family}{size}.pth"


def _rfdetr_container(
    weights: Path, ckpt: Any, spec: NativeSpec, classes: list[str], cache_dir: Path
) -> Path:
    """A path ``RFDETR.from_checkpoint`` can consume.

      A ``.pth`` is already one, and is returned untouched. A safetensors state
      dict is wrapped ONCE, in the cache, into the container rfdetr reads:
      ``{"model": <state dict>, "model_name": ..., "args": {...}}``.

      Why this shape and not "construct the variant class and ``load_state_dict``":
      ``from_checkpoint`` is the path that infers ``num_classes`` from
      ``class_embed.weight`` and ``num_keypoints_per_class`` from
      ``_kp_active_mask``, re-inits the detection/keypoint heads to match, and
      interpolates position embeddings for a resolution change. Reconstructing
      that by hand would be a second, drifting copy of rfdetr's own loader.

      ``model_config`` carries the trained resolution when the record knows it;
      rfdetr treats those keys as checkpoint-derived (not user overrides), which
      is exactly right for a reload.

      The container's BASENAME is deliberately the cache stem, never a variant
      filename: ``load_pretrain_weights`` calls ``download_pretrain_weights`` on
      whatever path it is handed, and a name that matches rfdetr's asset registry
      would make it re-download over our file.

    **The name is keyed on the model, not on the artifact's filename**
      (:func:`_rfdetr_container_key`) - see there for what went wrong when it was
      not.
    """
    if weights.suffix != ".safetensors":
        return weights

    torch = _require_torch()
    variant, token = _rfdetr_variant(spec)
    args: dict[str, Any] = {
        "num_classes": len(classes),
        "class_names": list(classes),
        "pretrain_weights": token,
    }
    payload: dict[str, Any] = {"model": ckpt, "model_name": variant, "args": args}
    resolution = _rfdetr_resolution(spec)
    if resolution is not None:
        payload["model_config"] = {"resolution": resolution}

    key = _rfdetr_container_key(weights, variant, token, classes, resolution)
    container = cache_dir / f"{weights.stem}.{key}.rfdetr.pth"
    if container.exists():
        return container

    tmp = container.with_name(container.name + ".part")
    torch.save(payload, str(tmp))
    tmp.replace(container)
    _LOG.debug(
        "Wrapped %s into an rfdetr checkpoint as %s (%s).", weights.name, container.name, variant
    )
    return container


def _rfdetr_container_key(
    weights: Path, variant: str, token: str, classes: list[str], resolution: int | None
) -> str:
    """A cache key that is unique per MODEL, not per artifact FILENAME.

    The container was named ``{weights.stem}.rfdetr.pth``. Every Pictograph model
    publishes its native weights under the one filename ``model.safetensors``, so
    offline - where ``weights`` is the file the user downloaded rather than this
    cache's own version-stamped copy - EVERY model in an organization resolved to
    the same ``model.rfdetr.pth``. Load two models in one session and the second
    read the FIRST's architecture, class list and resolution out of the cache.

    That is the shape ``services/ImageThumbnailCache.ts`` documents on the frontend:
    a key scoped to the shared container rather than to the thing being cached
    collides across every tenant of that container. Same mistake, same fix - key on
    what makes the entry DIFFERENT.

    Everything the container's payload is built from is in the key:

    * the source file, as ``(resolved path, size, mtime_ns)`` - the standard
      cache-validity triple. Two different models cannot be the same file; a file
      REPLACED in place changes size or mtime and invalidates its entry. The
      tensors themselves are deliberately not hashed: they are a few hundred MB and
      the digest would be recomputed on every load, hit or miss, to defend against
      an in-place overwrite that preserved both size and mtime.
    * ``variant`` / ``token`` / ``classes`` / ``resolution`` - the rest of the
      payload. These come from ``config.json``, which a user can edit WITHOUT
      touching the weights file, so the triple above does not cover them.
    """
    digest = hashlib.sha256()
    try:
        stat = weights.stat()
        digest.update(f"{weights.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    except OSError:  # pragma: no cover - defensive; the file was just read
        digest.update(str(weights).encode())
    digest.update(f"|{variant}|{token}|{resolution}|".encode())
    digest.update("\x00".join(classes).encode())
    return digest.hexdigest()[:16]


def _rfdetr_resolution(spec: NativeSpec) -> int | None:
    """The square input resolution the run trained at, if the config states one."""
    config = spec.training_config
    for key in ("resolution", "image_height", "image_width"):
        raw = config.get(key)
        try:
            value = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _rfdetr_inner(module: Any) -> Any:
    """The live ``nn.Module`` inside an rfdetr wrapper (``RFDETR.model.model``)."""
    inner = getattr(getattr(module, "model", None), "model", None)
    return inner if inner is not None else getattr(module, "model", None)


def _rfdetr_module_resolution(module: Any) -> tuple[int, int] | None:
    """The square input resolution the rebuilt rfdetr model actually runs at.

    ARTIFACT BEATS CONFIG, the same rule the ONNX loader applies (it reads the
    graph's static input shape over the stored ``image_height``/``image_width``).
    ``_pytorch_input_size`` would otherwise hand a keypoint model the 640 default
    while the model was trained - and exported - at 576, and RF-DETR's backbone
    additionally requires the resolution be divisible by ``patch_size *
    num_windows``, so a wrong value is a hard failure or silently distorted
    coordinates rather than a small numeric difference.
    """
    raw = getattr(getattr(module, "model", None), "resolution", None)
    try:
        value = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return (value, value) if value > 0 else None


_POS_EMBED_SUFFIX = "embeddings.position_embeddings"


def _rfdetr_pos_embed_tokens(state: Any) -> int | None:
    """The patch-token count in an RF-DETR backbone's position embeddings.

    ``(1, 1 + S*S, D)`` - one CLS token plus an ``S x S`` grid - so this is the
    artifact's own record of the grid it was trained on, independent of any
    config. Returns the grid area ``S*S``, or None if the tensor is absent or not
    a perfect square (a non-square backbone would make the arithmetic below
    meaningless, and guessing is worse than declining).
    """
    if not isinstance(state, dict):
        return None
    for key, value in state.items():
        if not isinstance(key, str) or not key.endswith(_POS_EMBED_SUFFIX):
            continue
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) != 3 or int(shape[1]) < 2:
            continue
        tokens = int(shape[1]) - 1
        side = math.isqrt(tokens)
        return tokens if side * side == tokens else None
    return None


def _rebuild_rfdetr_at_artifact_resolution(
    module: Any,
    ckpt: Any,
    weights: Path,
    spec: NativeSpec,
    classes: list[str],
    cache_dir: Path,
) -> Any:
    """Put a safetensors rebuild back on the resolution its WEIGHTS were trained at.

    The patch size is taken from the module rfdetr just built rather than from a
    table: that module reports both its resolution ``R`` and its own token grid
    ``S x S``, so ``R / S`` is rfdetr's patch geometry as rfdetr computes it. The
    artifact's resolution is then ``S_artifact * (R / S)``. Nothing here restates
    rfdetr's geometry, so nothing here can drift from it - which is the same
    reason :func:`_rfdetr_container` wraps a state dict rather than reconstructing
    the loader by hand.

    Returns the original module when the two already agree (the overwhelmingly
    common case - the record is usually right), or when the arithmetic cannot be
    done confidently.
    """
    artifact_tokens = _rfdetr_pos_embed_tokens(ckpt)
    built_tokens = _rfdetr_pos_embed_tokens(_rfdetr_state_of(module))
    built = _rfdetr_module_resolution(module)
    if artifact_tokens is None or built_tokens is None or built is None:
        return module
    if artifact_tokens == built_tokens:
        return module

    side_built = math.isqrt(built_tokens)
    side_artifact = math.isqrt(artifact_tokens)
    if side_built == 0:
        return module  # pragma: no cover - guarded by the perfect-square check
    patch = built[0] / side_built
    resolution = round(side_artifact * patch)
    if resolution <= 0 or resolution == built[0]:  # pragma: no cover - defensive
        return module

    _LOG.warning(
        "Resolution disagreement: these weights were trained at %dx%d (a %dx%d "
        "patch grid) but the model record asked for %dx%d, so rfdetr interpolated "
        "the position embeddings and you would have been served a different model "
        "than the one that trained. Rebuilding at %d - the artifact decides.",
        resolution,
        resolution,
        side_artifact,
        side_artifact,
        built[0],
        built[0],
        resolution,
    )
    forced = NativeSpec(
        model_type=spec.model_type,
        architecture=spec.architecture,
        training_config={**spec.training_config, "resolution": resolution},
        classes=list(spec.classes),
        name=spec.name,
    )
    container = _rfdetr_container(weights, ckpt, forced, classes, cache_dir)
    rebuilt = _build_rfdetr(container)
    _verify_rfdetr_load(rebuilt, ckpt, container)
    return rebuilt


def _rfdetr_state_of(module: Any) -> dict[str, Any] | None:
    """The live module's state dict, for comparing against the artifact's."""
    inner = _rfdetr_inner(module)
    target = inner if inner is not None else module
    getter = getattr(target, "state_dict", None)
    if not callable(getter):  # pragma: no cover - defensive
        return None
    try:
        return dict(getter())
    except Exception:  # pragma: no cover - defensive
        return None


def _rfdetr_raw_outputs(raw: Any) -> list[Any] | None:
    """An rfdetr module's forward output → the array list the canon's decode reads.

    Handles both shapes rfdetr itself handles: the eager module returns a dict
    (``pred_logits`` / ``pred_boxes`` / ``pred_keypoints`` / ``pred_masks``) while
    a compiled or export shim returns the positional tuple. Order does not matter
    - every canon decode identifies its tensors by SHAPE - so this only has to
    find them and hand over numpy.

    The third slot is whichever HEAD this model has: keypoints for a pose model,
    masks for a segmentation one. A detection model has neither and it stays
    ``None``, which every decode treats as "no third tensor".
    """
    import numpy as np

    def _np(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "detach"):
            return value.detach().float().cpu().numpy()
        return np.asarray(value)

    if isinstance(raw, dict):
        if "pred_boxes" not in raw or "pred_logits" not in raw:
            return None
        head = raw.get("pred_keypoints")
        if head is None:
            head = raw.get("pred_masks")
        return [_np(raw["pred_boxes"]), _np(raw["pred_logits"]), _np(head)]
    if isinstance(raw, (tuple, list)):
        arrays = [_np(v) for v in raw]
        return arrays if len(arrays) >= 2 else None
    return None  # pragma: no cover - depends on rfdetr version


def _verify_rfdetr_load(module: Any, state_dict: Any, container: Path) -> None:
    """Prove the rebuilt rfdetr actually ACCEPTED the safetensors tensors.

    rfdetr loads with ``strict=False`` and only WARNS on a partial load, so a
    wrong variant guess would hand back a mostly-randomly-initialised model that
    predicts confident nonsense. Comparing the artifact's tensor names+shapes
    against the built module's own turns that into an error at load time.

    A tolerance (not an exact match) because rfdetr legitimately remaps a few
    keys during load (e.g. ``remap_projector_to_cross_attn``); a WRONG variant
    misses far more than that - a different depth/width mismatches most shapes.
    """
    inner = _rfdetr_inner(module)
    if inner is None or not hasattr(inner, "state_dict") or not isinstance(state_dict, dict):
        return  # pragma: no cover - defensive; nothing to compare against
    built = inner.state_dict()
    total = len(state_dict)
    if total == 0:
        return  # pragma: no cover - defensive
    matched = sum(
        1
        for name, tensor in state_dict.items()
        if name in built and tuple(built[name].shape) == tuple(getattr(tensor, "shape", ()))
    )
    if matched >= total * 0.9:
        return
    variant = container.name
    raise ValueError(
        f"Rebuilding this model from its safetensors artifact loaded only "
        f"{matched}/{total} tensors - the RF-DETR variant inferred for it "
        f"({variant}) does not match the trained architecture, so the model "
        f"would predict from partly-uninitialised weights. Use the ONNX "
        f"backend (pictograph.get_model / client.models.load) for this model."
    )


def _checkpoint_dtype(module: Any, torch: Any) -> Any:
    """The dtype the module's own parameters carry.

    An fp16-published model has half weights; pinning float32 inputs against them
    raises ``expected scalar type Half but found Float``.
    """
    try:
        for param in module.parameters():
            return param.dtype
    except (AttributeError, StopIteration):  # pragma: no cover - rfdetr wrapper
        pass
    return torch.float32


def _move_rfdetr(module: Any, device: str) -> None:
    """Move an rfdetr model wrapper onto ``device`` if it exposes a way to.

    RF-DETR owns its own device handling; previously the requested device was
    silently dropped for this family.
    """
    for attr in ("to", "model"):
        target = getattr(module, attr, None)
        if attr == "to" and callable(target):
            try:
                target(device)
                return
            except Exception:  # pragma: no cover - depends on rfdetr version
                _LOG.debug("rfdetr model would not move to %s; leaving it where it is.", device)
                return
        inner = getattr(target, "model", None) if target is not None else None
        if inner is not None and hasattr(inner, "to"):
            try:
                inner.to(device)
            except Exception:  # pragma: no cover - depends on rfdetr version
                _LOG.debug("rfdetr inner module would not move to %s.", device)
            return


def _classes_of(model: Model) -> list[str]:
    mapping = model.class_mapping or {}
    classes = mapping.get("classes")
    if isinstance(classes, list) and classes:
        return [str(c) for c in classes]
    raise ValueError(f"Model {model.name!r} has no class list to run inference with.")


def _pytorch_input_size(model_type: str, config: dict[str, Any]) -> tuple[int, int]:
    """The eval-time input size per family (falls back to each pipeline's default)."""
    default = (
        224
        if model_type == "classification"
        else (512 if model_type == "semantic_segmentation" else 640)
    )
    try:
        height = int(config.get("image_height") or default)
        width = int(config.get("image_width") or default)
    except (TypeError, ValueError):
        height = width = default
    return height, width


def _resolve_family(model_type: str, architecture: str, ckpt: Any) -> str:
    """Resolve which framework trained this model.

    ``object_detection`` is ambiguous for models stored with a bare ``model_size``
    architecture (both YOLOX and RF-DETR use size labels), so the CHECKPOINT itself
    is the tiebreak: RF-DETR's stripped best-checkpoint carries
    ``{"model", "args", "model_name"}``, YOLOX's carries ``"model"`` without ``args``.
    """
    if model_type == "classification":
        return "torchvision"
    if model_type == "semantic_segmentation":
        return "segmentation_models_pytorch"
    if model_type == "instance_segmentation":
        return "rfdetr"
    if model_type == "keypoint_detection":
        # The only keypoint pipeline is RF-DETR (Must stay in sync with
        # dispatch.build_wrapper, which routes keypoint_detection to
        # RFDETRKeypointDetector unconditionally). Stated rather than left to
        # the architecture/checkpoint sniffing below, because a safetensors
        # artifact is a BARE state dict with no `args` to sniff.
        return "rfdetr"
    arch = architecture.lower()
    if arch.startswith("yolox"):
        return "yolox"
    if "rf-detr" in arch or "rfdetr" in arch:
        return "rfdetr"
    if isinstance(ckpt, dict):
        if "args" in ckpt and "model_name" in ckpt:
            return "rfdetr"
        if "model" in ckpt:
            return "yolox"
    return "rfdetr"


def _rfdetr_ckpt_class_names(ckpt: Any) -> list[str] | None:
    """The class list an RF-DETR checkpoint carries in its own args, if any."""
    if not isinstance(ckpt, dict):
        return None
    args = ckpt.get("args")
    if args is None:
        return None
    data = args if isinstance(args, dict) else getattr(args, "__dict__", {})
    names = data.get("class_names")
    if isinstance(names, list) and names and all(isinstance(n, str) for n in names):
        return list(names)
    return None


def _rfdetr_ckpt_num_keypoints_per_class(ckpt: Any, num_classes: int) -> list[int] | None:
    """A keypoint checkpoint's per-class active joint counts, if cleanly active-first.

    Only returned when it is a per-class active-first schema - exactly one entry per
    class, every entry ``>= 1``. A background-first / padded form is rejected so the
    emitter falls back to the returned joint count rather than mis-indexing arity.
    """
    if not isinstance(ckpt, dict):
        return None
    args = ckpt.get("args")
    data = args if isinstance(args, dict) else getattr(args, "__dict__", {})
    raw = data.get("num_keypoints_per_class") if isinstance(data, dict) else None
    if (
        isinstance(raw, (list, tuple))
        and len(raw) == num_classes
        and all(isinstance(x, (int, float)) and int(x) >= 1 for x in raw)
    ):
        return [int(x) for x in raw]
    return _kp_counts_from_active_mask(ckpt, num_classes)


def _kp_counts_from_active_mask(ckpt: Any, num_classes: int) -> list[int] | None:
    """Per-class arity read off the ``_kp_active_mask`` buffer, ``[classes, max_K]``.

    The tensor route matters for the SAFETENSORS path: a bare state dict has no
    ``args`` to read the schema from, but it does carry the buffer rfdetr itself
    infers ``num_keypoints_per_class`` from (``detr.py::from_checkpoint``).
    Accepted only when it is cleanly active-first - one row per class, every row
    with at least one active slot - mirroring the ``args`` rule above.

    Read through :func:`_bare_state_dict` rather than off ``ckpt`` directly,
    because the buffer's DEPTH depends on the container: a safetensors artifact
    is the bare mapping and carries it at the top level, while a ``.pth`` nests
    it under ``"model"``. Measured on the shipped RF-DETR checkpoints, a real
    ``args`` carries no ``num_keypoints_per_class`` at all - so for a ``.pth``
    this tensor route is the ONLY route, and reading the top level found
    nothing and silently dropped every class's arity.
    """
    state = _bare_state_dict(ckpt)
    mask: Any = state.get("_kp_active_mask") if state is not None else None
    shape = getattr(mask, "shape", None)
    if shape is None or len(shape) != 2 or int(shape[0]) != num_classes:
        return None
    try:
        counts = [int(row) for row in mask.sum(dim=1).tolist()]
    except (AttributeError, TypeError, ValueError):  # pragma: no cover - defensive
        return None
    return counts if all(c >= 1 for c in counts) else None


def _build_rfdetr(weights: Path) -> Any:
    """Rebuild an RF-DETR checkpoint into a runnable module. Needs nothing installed.

    The architecture comes from :mod:`pictograph.inference._rfdetr`, vendored from
    rfdetr 1.8.3 - the version the training image pins - so a user never has to
    install `rfdetr` (and, transitively, `transformers>=5.1,<6` + `supervision`)
    to run weights we published to them.
    """
    from ._rfdetr import from_checkpoint

    return from_checkpoint(str(weights))


def _build_yolox(ckpt: Any, config: dict[str, Any], architecture: str, num_classes: int) -> Any:
    """Rebuild a YOLOX checkpoint into a runnable module. Needs nothing installed.

    The architecture comes from :mod:`pictograph.inference._yolox`, vendored from
    YOLOX at the commit the training image pins, so a user never has to install
    `yolox` - which they could not do from a declared dependency anyway: PyPI's
    only release is a 2022 sdist whose `setup.py` aborts unless torch is already
    importable, and PyPI rejects the `git+https://…@<sha>` direct reference that
    would otherwise pin the right commit. See that package's NOTICE.
    """
    from ._yolox import build_yolox

    size = _yolox_size(config, architecture)
    depth, width = _YOLOX_SIZES[size]
    model = build_yolox(depth, width, num_classes)
    # A .pth nests the weights under "model"; a safetensors artifact IS the bare
    # state dict (it is serialised from `module.state_dict()` directly).
    state = _bare_state_dict(ckpt)
    if state is None:
        raise ValueError("YOLOX checkpoint has no 'model' state_dict - is this a YOLOX .pth?")
    model.load_state_dict(state, strict=True)
    model.eval()
    model.head.decode_in_inference = True
    return model


def _state_for_load(ckpt: Any) -> Any:
    """The tensors to strict-load, accepted in EITHER container shape.

    The two container shapes are not a per-family property, they are a per-
    FORMAT one: ``safetensors.torch.load_file`` always returns the bare mapping,
    while a ``.pth`` may nest it under ``"model"``. Normalising here (exactly as
    :func:`_build_yolox` already did) is what lets one family load from both
    formats instead of only from whichever one its pipeline happens to publish
    today.

    Anything this cannot identify as a tensor mapping is handed through
    UNCHANGED, so ``load_state_dict`` still raises its own (already precise)
    error about what is actually missing - this normalises a container, it does
    not add a validation step.
    """
    state = _bare_state_dict(ckpt)
    return ckpt if state is None else state


def _bare_state_dict(ckpt: Any) -> dict[str, Any] | None:
    """The tensor mapping inside a checkpoint, whatever container it arrived in.

    Shape-based, never format-name-based: a ``.pth`` nests its weights under
    ``"model"`` (or ``"model_state_dict"``), a safetensors artifact is already the
    bare mapping.

    ``model_state_dict`` is the shape our OWN classification pipeline writes for its
    resume checkpoint (``train_classification.py`` saves the published
    ``checkpoint_best_*.pth`` bare and the resumable one nested). That distinction is
    invisible from the filename, and now that :func:`load_model` accepts a checkpoint
    off a user's disk rather than only the artifact we published, handing over the
    wrong one of the two is an easy and completely silent mistake - the strict load
    fails with a wall of "Missing key(s)" naming every layer in the backbone, which
    says nothing about the container.
    """
    if not isinstance(ckpt, dict):
        return None
    for key in ("model", "model_state_dict", "state_dict"):
        inner = ckpt.get(key)
        if isinstance(inner, dict):
            return inner
    return ckpt if all(hasattr(v, "shape") for v in list(ckpt.values())[:3]) else None


def _yolox_size(config: dict[str, Any], architecture: str) -> str:
    raw = str(config.get("model_size") or architecture or "m").lower()
    raw = raw.removeprefix("yolox-").removeprefix("yolox_").removeprefix("yolox ")
    if raw in _YOLOX_SIZES:
        return raw
    raise ValueError(f"Unknown YOLOX size {raw!r} - expected one of {sorted(_YOLOX_SIZES)}.")


def _build_smp(ckpt: Any, config: dict[str, Any], architecture: str, classes: list[str]) -> Any:
    smp = _require(
        "segmentation_models_pytorch",
        # Declared + PINNED by the [inference] extra, so this only fires on an
        # environment that installed the SDK without it. The hint therefore names
        # the extra, never the third-party package: a user who pip-installs a
        # bare `segmentation-models-pytorch` gets whatever version is newest,
        # which is not necessarily the one whose module shapes our .pth files
        # were written from.
        "Semantic-segmentation models need the local-inference extra:\n"
        '    pip install "pictograph[inference]"',
    )
    arch = str(config.get("architecture") or architecture or _SMP_DEFAULT_ARCH).lower()
    cls_name = _SMP_ARCH_CLASSES.get(arch)
    if cls_name is None:
        raise ValueError(
            f"Unknown segmentation architecture {arch!r} - "
            f"expected one of {sorted(set(_SMP_ARCH_CLASSES))}."
        )
    encoder = str(config.get("encoder") or config.get("encoder_name") or _SMP_DEFAULT_ENCODER)
    # Must stay in sync with the pipeline's create_model: single-class = 1 channel;
    # multi-class = classes + a background channel.
    n_classes = 1 if len(classes) == 1 else len(classes) + 1
    kwargs: dict[str, Any] = {
        "encoder_name": encoder,
        "encoder_weights": None,
        "in_channels": 3,
        "classes": n_classes,
    }
    # The pipeline passes `activation` ONLY to Unet / UnetPlusPlus. Segformer is
    # built without one, so adding a sigmoid here would apply an activation the
    # trained model and the exported graph do not have.
    if cls_name != "Segformer":
        kwargs["activation"] = "sigmoid" if n_classes == 1 else None
    model = getattr(smp, cls_name)(**kwargs)
    model.load_state_dict(_state_for_load(ckpt), strict=True)
    model.eval()
    return model


def _build_torchvision(
    ckpt: Any, config: dict[str, Any], architecture: str, num_classes: int
) -> Any:
    _require_torch()
    tv_models = _require(
        "torchvision.models",
        'Classification models need torchvision:\n    pip install "pictograph[inference]"',
    )
    import torch.nn as nn

    backbone = str(config.get("backbone") or architecture or "").lower()
    if backbone not in _CLS_HEADS:
        raise ValueError(
            f"Unknown classification backbone {backbone!r} - expected one of {sorted(_CLS_HEADS)}."
        )
    attr, kind = _CLS_HEADS[backbone]
    model = getattr(tv_models, backbone)(weights=None)

    # Must stay in sync with train_classification.py::ModelFactory._sequential_head_split.
    # The head's input is the FIRST nn.Linear, and every structural module before
    # it must be PRESERVED. Taking the last Linear (what this used to do) is right
    # only for EfficientNet: MobileNetV3's classifier is
    # (Linear(960,1280), Hardswish, Dropout, Linear(1280,1000)) so the real input
    # is 960, not 1280; and ConvNeXt's is (LayerNorm2d, Flatten, Linear(768,1000)),
    # where replacing the WHOLE Sequential also deletes the LayerNorm2d and Flatten
    # and the head then receives an un-flattened (N, 768, 1, 1).
    keep: list[Any] = []
    if kind == "linear":
        in_features = getattr(model, attr).in_features
    elif kind == "sequential":
        in_features, keep = _sequential_head_split(getattr(model, attr))
    else:  # vit
        in_features = model.heads.head.in_features

    state = _state_for_load(ckpt)
    dropout = float(config.get("dropout_rate") or 0.5)
    # The head's hidden width comes from the CHECKPOINT, not from the config: the
    # training service's normalizer drops `hidden_units`, so the config always says
    # 256 even when the run trained something else - and a mismatch fails the
    # strict load. Artifact beats config.
    hidden = _classifier_hidden_units(state, attr) or int(config.get("hidden_units") or 256)
    new_head = nn.Sequential(
        *keep,
        nn.Dropout(p=dropout),
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(hidden, num_classes),
    )
    setattr(model, attr, new_head)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _sequential_head_split(classifier: Any) -> tuple[int, list[Any]]:
    """``(in_features, modules to KEEP)`` for a Sequential classifier.

    Must stay in sync with ``train_classification.py::ModelFactory._sequential_head_split``.
    The head's input is the first ``nn.Linear``; everything structural before it is
    preserved. ``Dropout`` is excluded from that prefix because the replacement head
    brings its own - which keeps EfficientNet byte-identical to its previous
    (working) behaviour.
    """
    import torch.nn as nn

    modules = list(classifier)
    first_linear = next((i for i, m in enumerate(modules) if isinstance(m, nn.Linear)), None)
    if first_linear is None:
        raise ValueError(f"Sequential classifier {classifier!r} contains no nn.Linear")
    keep = [m for m in modules[:first_linear] if not isinstance(m, nn.Dropout)]
    return modules[first_linear].in_features, keep


def _classifier_hidden_units(ckpt: Any, attr: str) -> int | None:
    """The trained head's hidden width, read off the checkpoint's own tensor shapes.

    The head is ``[*preserved prefix,] Dropout, Linear(in, hidden), ReLU, Dropout,
    Linear(hidden, n)``. A fixed ``{attr}.1.weight`` lookup only works when the prefix
    is empty - ConvNeXt preserves ``LayerNorm2d`` + ``Flatten``, which shifts every
    index. So find the FIRST 2-D weight under ``attr`` instead: a preserved structural
    prefix contributes only 1-D weights (LayerNorm) or none (Flatten), so the first
    2-D one is always the head's first Linear, whatever its index.
    """
    if not isinstance(ckpt, dict):
        return None
    best: tuple[int, int] | None = None
    for key, value in ckpt.items():
        if not (key.startswith(f"{attr}.") and key.endswith(".weight")):
            continue
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) != 2:
            continue
        part = key.split(".")[1]
        if not part.isdigit():
            continue
        idx = int(part)
        if best is None or idx < best[0]:
            best = (idx, int(shape[0]))
    return best[1] if best else None


# ───────────── image helpers ─────────────


def _bgr_to_pil(image_bgr: Any) -> Any:
    """A BGR numpy array → an RGB PIL image.

    BGR is the SDK's one documented convention for a raw array on both engines, so
    `cv2.imread(path)` means the same thing whichever backend a caller holds.
    """
    import numpy as np
    from PIL import Image

    arr = np.asarray(image_bgr)
    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    return Image.fromarray(arr[:, :, ::-1].astype("uint8"), mode="RGB")


# These three resizes MUST be cv2's, not PIL's - all three docstrings
# already claimed cv2 ("matching the ONNX wrapper's cv2 call", "matching
# training"), and all three used `PIL.Image.Resampling.BILINEAR`.
#
# The two are not interchangeable. PIL's BILINEAR is support-scaled: on a
# downscale it widens the kernel and antialiases. cv2's INTER_LINEAR samples a
# fixed 2x2 neighbourhood and aliases. Real inputs are downscales (a 640x480
# photo into a 224x224 classifier), which is precisely where they diverge.
#
# The server-side wrappers are the canon - they are what the pipelines TRAIN
# against (`ClassificationDataset` decodes with cv2 and `get_validation_
# augmentation` resizes with albumentations' cv2 INTER_LINEAR) and what every
# graph runtime serves through. So a PIL resize here made the `pytorch` runtime
# the only path that reproduced neither the training preprocessing nor its own
# sibling backends.
#
# Measured on a real 2-class resnet18 (10 images, onnxruntime/CPU vs
# pytorch/CPU, max |Δp| on the softmax vector):
#   already-224x224 input .................. 0.000000   (no resize -> no divergence)
#   every other input ...................... 0.0028 - 0.019827
# The tolerance these are asserted against is FP16_CLS_PROB_ATOL = 2e-2, so the
# torch path was passing on 99% of its budget, on a binary classifier whose
# probabilities sit near 0.5 - i.e. one borderline image away from a flipped
# class. That the native-size case is EXACTLY zero is what identifies the resize
# as the sole cause, and is also why every pre-existing parity test missed it:
# they all feed native size.
#
# cv2 is not a new dependency: it is already required by the `inference` extra
# and `models._decode_image` decodes through it on this very path.


def _cv2_resize(arr: Any, width: int, height: int, dtype: Any = None) -> Any:
    """The ONE resize for torch-path image data - cv2 INTER_LINEAR, the canon."""
    import cv2
    import numpy as np

    out = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.asarray(out, dtype=dtype) if dtype is not None else out


def _resize_bgr(src: Any, width: int, height: int) -> Any:
    """Bilinear resize of a BGR uint8 array, matching the ONNX wrapper's cv2 call."""
    return _cv2_resize(src.astype("uint8"), width, height, dtype="uint8")


def _normalized_chw(pil: Any, width: int, height: int) -> Any:
    """RGB PIL → ImageNet-normalized (C,H,W) float32. Bilinear, matching training."""
    import numpy as np

    arr = _cv2_resize(np.asarray(pil, dtype=np.uint8), width, height)
    arr = arr.astype(np.float32) / 255.0
    mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
    std = np.array(_IMAGENET_STD, dtype=np.float32)
    normalized = ((arr - mean) / std).astype(np.float32)
    return normalized.transpose(2, 0, 1)


def _resize_channel(channel: Any, width: int, height: int) -> Any:
    """Bilinear-resize one float output channel, matching the wrapper's cv2 call."""
    import numpy as np

    return _cv2_resize(np.asarray(channel, dtype=np.float32), width, height, dtype="float32")
