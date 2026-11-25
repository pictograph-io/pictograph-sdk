# ------------------------------------------------------------------------
# Pictograph - original work, not vendored.
# ------------------------------------------------------------------------
"""The anomaly-detection models' inference definitions (PatchCore, EfficientAD), from anomalib.

Pictograph trains its anomaly models with anomalib (pinned at :data:`ANOMALIB_VENDORED_VERSION`
in the training image), but a trained model is RUN everywhere else - the ONNX export, the fp16
graph, the ``.pte`` and Core ML builds, this SDK's local ``.pth`` / ``.safetensors`` - and none
of those installs anomalib (its Lightning, CLI and metrics stack is a training framework, and
its 2.7 line declares torch 2.10+ for its own extras). So the two models' INFERENCE paths live
here, extracted from anomalib's source with its Apache-2.0 notice (``NOTICE``): plain
``torch.nn`` modules, ImageNet normalization inside, raw outputs.

**One definition, proven at every training run.** The training pipeline copies anomalib's
trained state into these modules (``*_state_from_anomalib``), holds their answers to anomalib's
on held-out images, and exports the ONNX graph FROM them; every other runtime rebuilds from the
same state with :func:`build_anomaly_module`. A disagreement fails the run.

**The contract** (``config.json`` ``_pictograph.anomaly``; ROADMAP Program N): input
``(B, 3, S, S)`` RGB in [0, 1] (a stretch resize of the image); outputs ``anomaly_map``
``(B, 1, S, S)`` and ``pred_score`` ``(B, 1)``, raw. The decoders normalize them with anomalib's
post-processing numbers (the block's ``image_threshold`` / ``pixel_threshold`` / ``*_min`` /
``*_max``); nothing here does.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: The anomalib release these definitions were extracted from and the training image pins
#: (LOCKSTEP with that pin; the platform's tests hold the two equal).
ANOMALIB_VENDORED_VERSION = "2.7.1"

#: The anomaly families, as a model's ``_pictograph.anomaly.family`` names them (and its
#: ``pipeline``).
FAMILIES = ("patchcore", "efficientad")


def anomaly_meta_of(config: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The ``_pictograph.anomaly`` block of a model's ``config.json`` (the full artifact or a bare
    envelope), or None for a model that is not an anomaly model."""
    if not isinstance(config, Mapping):
        return None
    env = config.get("_pictograph") if isinstance(config.get("_pictograph"), Mapping) else config
    if not isinstance(env, Mapping):
        return None
    meta = env.get("anomaly")
    return dict(meta) if isinstance(meta, Mapping) else None


def build_anomaly_module(meta: Mapping[str, Any]) -> Any:
    """An untrained module of the family ``meta`` names, shaped by it (load the trained state into
    it): PatchCore's backbone, layers, memory-bank size and neighbours; EfficientAD's size."""
    family = str(meta.get("family") or "")
    if family == "patchcore":
        from .patchcore import PatchcoreInference

        return PatchcoreInference(
            backbone=str(meta.get("backbone") or "wide_resnet50_2"),
            layers=tuple(meta.get("layers") or ("layer2", "layer3")),
            memory_bank_size=int(meta["memory_bank_size"]),
            num_neighbors=int(meta.get("num_neighbors") or 9),
            sigma=float(meta.get("sigma") or 4.0),
        )
    if family == "efficientad":
        from .efficient_ad import EfficientAdInference

        return EfficientAdInference(
            model_size=str(meta.get("model_size") or "small"),
            teacher_out_channels=int(meta.get("teacher_out_channels") or 384),
            padding=bool(meta.get("padding", False)),
            pad_maps=bool(meta.get("pad_maps", True)),
        )
    raise ValueError(f"unknown anomaly family {family!r}; one of {FAMILIES}")


def load_anomaly_module(meta: Mapping[str, Any], state_dict: Mapping[str, Any]) -> Any:
    """:func:`build_anomaly_module` with the trained ``state_dict`` loaded STRICTLY (every key
    present, none extra), in eval mode."""
    module = build_anomaly_module(meta)
    module.load_state_dict(dict(state_dict), strict=True)
    return module.eval()


__all__ = [
    "ANOMALIB_VENDORED_VERSION",
    "FAMILIES",
    "anomaly_meta_of",
    "build_anomaly_module",
    "load_anomaly_module",
]
