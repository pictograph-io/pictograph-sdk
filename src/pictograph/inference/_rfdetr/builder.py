# ------------------------------------------------------------------------
# Pictograph - original work. The checkpoint-resolution rules and the predict
# pipeline below are transcribed from rfdetr 1.8.3's `detr.py`
# (`RFDETR.from_checkpoint`, `RFDETR.predict`) and `inference.py`
# (`_build_model_context`), Copyright (c) 2025 Roboflow, Apache-2.0. See ./NOTICE.
# ------------------------------------------------------------------------
"""Rebuild an RF-DETR model from one of our own checkpoints, with no `rfdetr` install.

This is the replacement for `rfdetr.RFDETR.from_checkpoint(path).predict(...)`.
It reproduces three upstream behaviours and deliberately drops everything else:

**Variant resolution.** `model_name` in the checkpoint, else a substring match on
`args.pretrain_weights`, else the checkpoint's filename - the same three-step order
and the same ordered map (seg entries before base entries, so ``seg-nano`` cannot be
shadowed by ``nano``).

**Schema inference from the tensors.** `num_classes` from ``class_embed.weight``'s
row count minus the background row, and `num_keypoints_per_class` from the
``_kp_active_mask`` buffer. Both override a stale `model_config`, because the head
shape is what the weights actually encode.

**The predict pipeline.** ``to_tensor`` → resize to the model's square resolution →
ImageNet normalize → forward → the vendored `PostProcess` → threshold. Identical
arithmetic to upstream, returning a plain record instead of a `supervision`
`Detections`, which is the second dependency this removes.

Dropped, because a reload never reaches them: training, export, ONNX/TensorRT
optimisation, LoRA, Roboflow deployment, the `rfdetr_plus` XLarge/2XLarge detection
variants (which live in a separate package upstream too), and the COCO sparse-ID
class mapping (our checkpoints always carry their own `class_names`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from pictograph.inference._rfdetr.config import (
    ModelConfig,
    RFDETRBaseConfig,
    RFDETRKeypointPreviewConfig,
    RFDETRLargeConfig,
    RFDETRMediumConfig,
    RFDETRNanoConfig,
    RFDETRSeg2XLargeConfig,
    RFDETRSegLargeConfig,
    RFDETRSegMediumConfig,
    RFDETRSegNanoConfig,
    RFDETRSegPreviewConfig,
    RFDETRSegSmallConfig,
    RFDETRSegXLargeConfig,
    RFDETRSmallConfig,
    TrainConfig,
)
from pictograph.inference._rfdetr.models.lwdetr import build_model
from pictograph.inference._rfdetr.models.postprocess import PostProcess
from pictograph.inference._rfdetr.models.weights import load_pretrain_weights
from pictograph.inference._safe_load import safe_torch_load

__all__ = ["Detections", "RFDETRModel", "from_checkpoint"]

_LOG = logging.getLogger("pictograph.inference")

# ImageNet statistics - RF-DETR's `RFDETR.means` / `RFDETR.stds`.
_MEANS = [0.485, 0.456, 0.406]
_STDS = [0.229, 0.224, 0.225]

# `model_name` (as written into the checkpoint by the training stack) → its config.
# Must stay in sync with rfdetr 1.8.3 `variants.py`'s `_model_config_class` assignments.
_VARIANT_CONFIGS: dict[str, type[ModelConfig]] = {
    "RFDETRBase": RFDETRBaseConfig,
    "RFDETRNano": RFDETRNanoConfig,
    "RFDETRSmall": RFDETRSmallConfig,
    "RFDETRMedium": RFDETRMediumConfig,
    "RFDETRLarge": RFDETRLargeConfig,
    "RFDETRKeypointPreview": RFDETRKeypointPreviewConfig,
    "RFDETRSegPreview": RFDETRSegPreviewConfig,
    "RFDETRSegNano": RFDETRSegNanoConfig,
    "RFDETRSegSmall": RFDETRSegSmallConfig,
    "RFDETRSegMedium": RFDETRSegMediumConfig,
    "RFDETRSegLarge": RFDETRSegLargeConfig,
    "RFDETRSegXLarge": RFDETRSegXLargeConfig,
    "RFDETRSeg2XLarge": RFDETRSeg2XLargeConfig,
}

# Substring → variant, in PRIORITY ORDER. rfdetr 1.8.3's
# `_CHECKPOINT_MODEL_MAP_ENTRIES`, reordered exactly as `from_checkpoint` reorders
# it at runtime: seg entries, then keypoint, then base. The order is load-bearing -
# a flat dict would let "nano" match "rf-detr-seg-nano.pth" first.
_NAME_MATCH_ORDER: tuple[tuple[str, str], ...] = (
    ("seg-2xlarge", "RFDETRSeg2XLarge"),
    ("seg-xxlarge", "RFDETRSeg2XLarge"),
    ("seg-xlarge", "RFDETRSegXLarge"),
    ("seg-large", "RFDETRSegLarge"),
    ("seg-medium", "RFDETRSegMedium"),
    ("seg-small", "RFDETRSegSmall"),
    ("seg-nano", "RFDETRSegNano"),
    ("seg-preview", "RFDETRSegPreview"),
    ("keypoint-preview", "RFDETRKeypointPreview"),
    ("large", "RFDETRLarge"),
    ("medium", "RFDETRMedium"),
    ("small", "RFDETRSmall"),
    ("nano", "RFDETRNano"),
    ("base", "RFDETRBase"),
)

# Detection XLarge / 2XLarge are not in the open-source RF-DETR package - upstream
# resolves them from `rfdetr_plus`. Named explicitly so a checkpoint that wants one
# gets a straight answer instead of being mis-resolved to Large.
_PLUS_ONLY = ("RFDETRXLarge", "RFDETR2XLarge")


@dataclass
class Detections:
    """What `predict` returns, in place of a `supervision.Detections`.

    Field names match the four attributes the engine reads off that class, so the
    call site is unchanged.
    """

    xyxy: Any
    confidence: Any
    class_id: Any
    mask: Any | None = None

    def __len__(self) -> int:
        return len(self.xyxy)


@dataclass
class _ModelContext:
    """The subset of rfdetr's `ModelContext` this package produces and consumes."""

    model: torch.nn.Module
    postprocess: PostProcess
    device: torch.device
    resolution: int
    args: Any
    class_names: list[str] | None = None
    model_config: ModelConfig | None = None
    inference_model: None = field(default=None, init=False)


def _bare_weights(ckpt: Any) -> dict[str, Any]:
    """The tensor mapping inside a checkpoint, in either container shape.

    `{"model": {...}}` is what our trainers write; `{"state_dict": {"model.…": …}}`
    is PyTorch Lightning's native form, whose keys carry a `model.` prefix and,
    under `torch.compile`, an additional `_orig_mod.` segment.
    """
    weights = ckpt.get("model") or {}
    if weights or "state_dict" not in ckpt:
        return dict(weights)
    out: dict[str, Any] = {}
    for key, value in ckpt["state_dict"].items():
        if not key.startswith("model."):
            continue
        stripped = key[len("model.") :]
        if stripped.startswith("_orig_mod."):
            stripped = stripped[len("_orig_mod.") :]
        out[stripped] = value
    return out


def _resolve_variant(path: str, ckpt: dict[str, Any]) -> str:
    """The variant name for this checkpoint: `model_name`, else weights name, else filename."""
    saved = ckpt.get("model_name")
    if isinstance(saved, str) and saved.strip() in _VARIANT_CONFIGS:
        return saved.strip()

    args = ckpt.get("args")
    data = args if isinstance(args, dict) else getattr(args, "__dict__", {}) or {}
    weights_name = str(data.get("pretrain_weights", "") or "").strip().lower()
    if weights_name in {"", "none", "null"}:
        weights_name = Path(path).name.lower()

    if isinstance(saved, str) and saved.strip() in _PLUS_ONLY:
        raise ValueError(
            f"Checkpoint requests the {saved.strip()!r} RF-DETR variant, which is not part "
            f"of the open-source RF-DETR architecture. Run this model through its ONNX "
            f"export instead."
        )
    for token, variant in _NAME_MATCH_ORDER:
        if token in weights_name:
            return variant
    raise ValueError(
        f"Could not infer the RF-DETR variant for checkpoint {path!r} "
        f"(model_name={saved!r}, pretrain_weights={weights_name!r}). Run this model "
        f"through its ONNX export instead."
    )


def from_checkpoint(path: str, **overrides: Any) -> RFDETRModel:
    """Rebuild the RF-DETR model a checkpoint encodes, and load its weights.

    Args:
        path: A `.pth` checkpoint carrying `{"model": state_dict, "args": {...}}`.
      **overrides: Config fields that win over anything inferred from the file.

    Returns:
        A loaded, eval-mode :class:`RFDETRModel` on the CPU.
    """
    ckpt: dict[str, Any] = safe_torch_load(path)
    if not isinstance(ckpt, dict):  # pragma: no cover - defensive
        raise ValueError(f"{path!r} is not an RF-DETR checkpoint (loaded a {type(ckpt).__name__}).")

    variant = _resolve_variant(path, ckpt)
    config_cls = _VARIANT_CONFIGS[variant]
    fields_ = getattr(config_cls, "model_fields", {}) or {}

    kwargs: dict[str, Any] = {}
    saved_config = ckpt.get("model_config")
    if isinstance(saved_config, dict):
        for key, value in saved_config.items():
            if key != "pretrain_weights" and (not fields_ or key in fields_):
                kwargs[key] = value

    args = ckpt.get("args")
    args_data = args if isinstance(args, dict) else getattr(args, "__dict__", {}) or {}
    if args_data.get("num_classes") is not None and "num_classes" not in overrides:
        kwargs["num_classes"] = args_data["num_classes"]

    # The TENSORS are authoritative over any stored config: a run fine-tuned onto a
    # different class count or keypoint schema leaves `model_config` stale, and a head
    # built to the stale shape then loads partially and predicts from noise.
    weights = _bare_weights(ckpt)
    if weights:
        mask = weights.get("_kp_active_mask")
        if (
            "num_keypoints_per_class" not in overrides
            and (not fields_ or "num_keypoints_per_class" in fields_)
            and isinstance(mask, torch.Tensor)
            and mask.ndim == 2
        ):
            kwargs["num_keypoints_per_class"] = [int(n) for n in mask.sum(dim=1).tolist()]
        class_embed = weights.get("class_embed.weight")
        if (
            "num_classes" not in overrides
            and isinstance(class_embed, torch.Tensor)
            and class_embed.ndim == 2
        ):
            kwargs["num_classes"] = int(class_embed.shape[0]) - 1  # the extra row is background

    kwargs.update(overrides)
    kwargs["pretrain_weights"] = str(path)

    config = config_cls(**kwargs)
    # Upstream clears checkpoint-derived keys out of `model_fields_set` so that
    # TRAINING-time head adaptation still sees them as unset. `load_pretrain_weights`
    # reads the same set to decide whether the checkpoint's class count is
    # authoritative - which, on a pure reload, it always is.
    derived = (set(kwargs) - set(overrides)) - {"pretrain_weights"}
    fields_set = getattr(config, "model_fields_set", None)
    if fields_set is not None:
        fields_set.difference_update(derived)

    return RFDETRModel(_build_context(config), variant=variant)


def _build_context(config: ModelConfig) -> _ModelContext:
    """rfdetr's `_build_model_context`: namespace → module → weights → postprocessor."""
    from pictograph.inference._rfdetr._namespace import _namespace_from_configs

    args = _namespace_from_configs(config, TrainConfig(dataset_dir=".", output_dir="."))
    module = build_model(args)

    class_names: list[str] = []
    if config.pretrain_weights is not None:
        class_names = load_pretrain_weights(module, config)
        # `load_pretrain_weights` can realign the config to the checkpoint's schema;
        # the namespace feeds `PostProcess`, so it has to follow.
        if getattr(args, "num_classes", None) != config.num_classes:
            args.num_classes = config.num_classes
        kp = list(getattr(config, "num_keypoints_per_class", []) or [])
        if list(getattr(args, "num_keypoints_per_class", []) or []) != kp:
            args.num_keypoints_per_class = kp

    postprocess = PostProcess(
        num_select=args.num_select,
        num_keypoints_per_class=getattr(args, "num_keypoints_per_class", []),
        trace_alpha=getattr(args, "postprocess_trace_alpha", 0.2),
    )
    return _ModelContext(
        model=module,
        postprocess=postprocess,
        device=torch.device("cpu"),
        resolution=config.resolution,
        args=args,
        class_names=class_names or None,
        model_config=config,
    )


class RFDETRModel:
    """A rebuilt RF-DETR model, exposing the surface the torch engine drives.

    Mirrors `rfdetr.RFDETR`'s attribute shape (`.model.model`, `.model.resolution`,
    `.to()`, `.predict()`, `.class_names`), so the engine's rfdetr helpers -
    `_rfdetr_inner`, `_rfdetr_module_resolution`, `_move_rfdetr` - need no change.
    """

    def __init__(self, context: _ModelContext, *, variant: str) -> None:
        self.model = context
        self.variant = variant
        self.means = list(_MEANS)
        self.stds = list(_STDS)
        num_channels = getattr(context.model_config, "num_channels", 3)
        if num_channels != 3:
            from itertools import cycle

            self.means = [v for _, v in zip(range(num_channels), cycle(_MEANS))]
            self.stds = [v for _, v in zip(range(num_channels), cycle(_STDS))]
        context.model.eval()

    @property
    def class_names(self) -> list[str]:
        return list(self.model.class_names or [])

    @property
    def model_config(self) -> ModelConfig | None:
        return self.model.model_config

    def to(self, device: Any) -> RFDETRModel:
        """Move the module (and the device the postprocessor targets) onto `device`."""
        resolved = torch.device(device)
        self.model.model.to(resolved)
        self.model.device = resolved
        return self

    def __repr__(self) -> str:
        return f"RFDETRModel(variant={self.variant!r}, resolution={self.model.resolution})"

    @torch.inference_mode()
    def predict(self, image: Any, threshold: float = 0.5) -> Detections:
        """One RGB image → thresholded boxes (+ masks, for a segmentation model).

        The arithmetic is rfdetr's `predict` for the single-image case: `to_tensor`
        puts the image in [0, 1], `resize` takes it to the model's square resolution
        with torchvision's default bilinear + antialias, and `normalize` applies the
        ImageNet statistics. Boxes come back in the ORIGINAL image's pixel space
        because `target_sizes` is the pre-resize shape.
        """
        import numpy as np

        # `F` is torchvision's own documented alias for this module.
        from torchvision.transforms import functional as F  # noqa: N812

        module = self.model.model
        module.eval()

        tensor = image if isinstance(image, torch.Tensor) else F.to_tensor(image)
        if tensor.shape[0] != getattr(self.model.model_config, "num_channels", 3):
            raise ValueError(
                f"Expected a {getattr(self.model.model_config, 'num_channels', 3)}-channel "
                f"image, got shape {tuple(tensor.shape)}."
            )
        original_hw = (int(tensor.shape[1]), int(tensor.shape[2]))

        device = next(module.parameters()).device
        dtype = next(module.parameters()).dtype
        resolution = int(self.model.resolution)
        batch = F.resize(tensor.to(device), [resolution, resolution]).unsqueeze(0)
        batch = F.normalize(batch, self.means, self.stds).to(dtype=dtype)

        raw = module(batch)
        if isinstance(raw, (tuple, list)):
            # A compiled or export shim returns positional tensors rather than a dict.
            outputs: dict[str, Any] = {"pred_boxes": raw[0], "pred_logits": raw[1]}
            if len(raw) == 3:
                key = (
                    "pred_keypoints"
                    if getattr(self.model.model_config, "use_grouppose_keypoints", False)
                    else "pred_masks"
                )
                outputs[key] = raw[2]
        else:
            outputs = raw

        target_sizes = torch.tensor([original_hw], device=device)
        result = self.model.postprocess(outputs, target_sizes=target_sizes)[0]

        scores = result["scores"]
        keep = scores > threshold
        masks = result.get("masks")
        return Detections(
            xyxy=result["boxes"][keep].float().cpu().numpy(),
            confidence=scores[keep].float().cpu().numpy(),
            class_id=result["labels"][keep].cpu().numpy().astype(np.int64),
            mask=masks[keep].squeeze(1).cpu().numpy() if masks is not None else None,
        )
