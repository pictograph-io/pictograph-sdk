# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# VENDORED INTO THE PICTOGRAPH SDK from rfdetr 1.8.3 (Apache-2.0).
# Modified by Pictograph: imports rewritten onto this package, and the
# HuggingFace `transformers` base classes replaced by the local shim in
# `_compat.py`. Training-only code paths are removed. See ../_rfdetr/NOTICE.
# ------------------------------------------------------------------------

from pictograph.inference._rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn

__all__ = [
    "MSDeformAttn",
]
