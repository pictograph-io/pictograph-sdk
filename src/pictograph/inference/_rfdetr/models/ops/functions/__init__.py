# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------
# VENDORED INTO THE PICTOGRAPH SDK from rfdetr 1.8.3 (Apache-2.0).
# Modified by Pictograph: imports rewritten onto this package, and the
# HuggingFace `transformers` base classes replaced by the local shim in
# `_compat.py`. Training-only code paths are removed. See ../_rfdetr/NOTICE.
# ------------------------------------------------------------------------
"""ms_deform_attn_func."""

from pictograph.inference._rfdetr.models.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch

__all__ = [
    "ms_deform_attn_core_pytorch",
]
