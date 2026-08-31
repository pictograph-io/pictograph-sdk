# ------------------------------------------------------------------------
# Pictograph - original work, not vendored.
# ------------------------------------------------------------------------
"""The YOLOX architecture, vendored from YOLOX @ ``6ddff48`` (Apache-2.0). See NOTICE.

A trained YOLOX checkpoint is a state dict: to run it natively you must first
rebuild the exact ``nn.Module`` it was trained as. Upstream that means installing
the ``yolox`` package - and there is no way to do that from ``pip install
"pictograph[inference]"``. Three independent reasons, each measured:

1. **PyPI's ``yolox`` cannot be built as a dependency at all.** The newest release
   is 0.3.0 (2022-04-22), **sdist only**, and its ``setup.py`` compiles a C++
   extension: ``get_ext_modules()`` does ``assert TORCH_AVAILABLE, "torch is
   required for pre-compiling ops"`` on every non-Windows platform. pip generates
   a dependency's metadata in an isolated build env *before* it has installed
   torch, so the build aborts::

       $ pip install yolox
       AssertionError: torch is required for pre-compiling ops, please install it first.

   That is a hard failure in a clean venv, not a warning.

2. **0.3.0 is not the architecture our weights encode.** The training image pins
   the git SHA below, ~3 years newer than 0.3.0, and the two differ inside
   ``yolo_head.py`` - including ``decode_outputs``, the function that turns raw
   head output into boxes.

3. **Its dependency set is incompatible with ours.** ``requirements.txt`` (which
   ``setup.py`` turns into ``install_requires``) pulls ``onnx-simplifier==0.4.10``,
   ``pycocotools``, ``tensorboard``, ``thop``, ``ninja``, ``tqdm``, ``loguru`` and
   - critically - ``opencv_python``, a SECOND distribution of the top-level ``cv2``
   package beside the ``opencv-python-headless`` in ``[inference]``. Which OpenCV
   then wins is decided by pip's overwrite order, the exact hazard documented in
   ``pictograph_training_service.py`` (Layer 8b). That is why the training image
   installs YOLOX ``--no-deps`` from git.

The remaining alternative - declaring ``yolox @ git+https://…@<sha>`` - is not
one: PEP 440 direct references are rejected by PyPI at upload, so a published
wheel cannot carry one.

So the model-construction subset lives here instead, and ``pip install
"pictograph[inference]"`` is the whole requirement.

What is vendored is the BUILD path only: ``yolox/models/`` minus its ``build.py``
(model-zoo downloads via ``torch.hub``) and ``yolo_fpn.py`` (the Darknet-53 FPN
neck, which no Pictograph pipeline trains). Data loading, the ``Exp`` system,
the trainer, evaluators, COCO metrics, the distributed/EMA/MLflow utilities and
the ONNX/TensorRT export tools are not.

**The SHA is load-bearing.** These modules are ``6ddff48``, matching
`the training service`'s ``YOLOX_SHA``,
because the weights we ship encode that architecture. Re-vendoring from a
different commit without retraining is how a checkpoint starts loading cleanly
and predicting nonsense. Re-sync both together.
"""

from __future__ import annotations

from typing import Any

#: The upstream commit this tree was vendored from - Must stay in sync with
#: ``pictograph_training_service.py``'s ``YOLOX_SHA``.
YOLOX_VENDORED_SHA = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"

#: Depth/width multipliers per size - Must stay in sync with the training pipeline's
#: ``pipelines/yolox/train_yolox.py::MODEL_CONFIGS``. The pipeline's ``DynamicExp``
#: builds every size with standard convs + silu, so the rebuild is uniform (no
#: upstream depthwise-nano special case).
YOLOX_SIZES: dict[str, tuple[float, float]] = {
    "nano": (0.33, 0.25),
    "tiny": (0.33, 0.375),
    "s": (0.33, 0.50),
    "m": (0.67, 0.75),
    "l": (1.00, 1.00),
    "x": (1.33, 1.25),
}

__all__ = ["YOLOX_SIZES", "YOLOX_VENDORED_SHA", "build_yolox"]


def build_yolox(depth: float, width: float, num_classes: int) -> Any:
    """The bare ``YOLOX(YOLOPAFPN, YOLOXHead)`` module, weightless.

    Mirrors what the training pipeline's ``DynamicExp.get_model`` assembles, so a
    checkpoint it wrote strict-loads into this. The caller loads the state dict
    and sets ``eval()`` - this only builds the shape.

    Returns ``Any`` rather than ``nn.Module`` deliberately: the caller reaches
    ``model.head.decode_in_inference``, which is a YOLOX attribute and not an
    ``nn.Module`` one, so the narrower annotation would be a type mypy has to be
    argued out of at every use.
    """
    from .yolo_head import YOLOXHead
    from .yolo_pafpn import YOLOPAFPN
    from .yolox import YOLOX

    backbone: Any = YOLOPAFPN(depth, width, act="silu")
    head: Any = YOLOXHead(num_classes, width, act="silu")
    return YOLOX(backbone, head)
