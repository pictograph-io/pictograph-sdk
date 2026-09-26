#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
#
# MODIFIED BY PICTOGRAPH (Apache-2.0 § 4b). See ./NOTICE.
#
# The four helpers `yolox/models/yolo_head.py` imports from `yolox.utils`,
# gathered into one module so the vendored tree does not have to carry
# `yolox/utils/` - whose package `__init__` pulls in the distributed-training,
# MLflow, EMA, LR-scheduler and COCO-metric modules, none of which a rebuilt
# checkpoint touches.
#
# Each function below is VERBATIM from the pinned upstream commit, save for ONE
# line noted beside `bboxes_iou`; only the file they live in changed:
#
#     meshgrid         yolox/utils/compat.py
#     bboxes_iou       yolox/utils/boxes.py     (one line - see below)
#     cxcywh2xyxy      yolox/utils/boxes.py
#     random_color     yolox/utils/demo_utils.py
#     visualize_assign yolox/utils/demo_utils.py
#
# `logger` is the ONE substitution: upstream binds `loguru.logger`, which would
# make `loguru` a runtime dependency of `pip install pictograph` for two log
# calls on the label-assignment path - code a rebuilt, `.eval()`-mode checkpoint
# never reaches. It is bound to the standard library's logger for the SDK's own
# `pictograph.inference` channel instead, which has the same `.info`/`.error`
# surface those two call sites use.
from __future__ import annotations

import logging
import random

import cv2
import numpy as np
import torch

__all__ = [
    "bboxes_iou",
    "cxcywh2xyxy",
    "logger",
    "meshgrid",
    "random_color",
    "visualize_assign",
]

logger = logging.getLogger("pictograph.inference")

_TORCH_VER = [int(x) for x in torch.__version__.split(".")[:2]]


def meshgrid(*tensors):
    if _TORCH_VER >= [1, 10]:
        return torch.meshgrid(*tensors, indexing="ij")
    else:
        return torch.meshgrid(*tensors)


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    # MODIFIED (Apache-2.0 § 4b): upstream writes `(tl < br).type(tl.type())`.
    # Same dtype cast, spelled without a legacy tensor type string - `Tensor.type()`
    # manufactures 'torch.mps.FloatTensor' on Apple silicon and `Tensor.type()`
    # then refuses to parse it, because legacy type strings exist only for CPU and
    # CUDA. `tl` is already on the target device, so this is a pure dtype cast.
    en = (tl < br).to(tl.dtype).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def cxcywh2xyxy(bboxes):
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    return bboxes


def random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)


def visualize_assign(img, boxes, coords, match_results, save_name=None) -> np.ndarray:
    """visualize label assign result.

    Args:
        img: img to visualize
        boxes: gt boxes in xyxy format
        coords: coords of matched anchors
        match_results: match results of each gt box and coord.
        save_name: name of save image, if None, image will not be saved. Default: None.
    """
    for box_id, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = random_color()
        assign_coords = coords[match_results == box_id]
        if assign_coords.numel() == 0:
            # unmatched boxes are red
            color = (0, 0, 255)
            cv2.putText(
                img, "unmatched", (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1
            )
        else:
            for coord in assign_coords:
                # draw assigned anchor
                cv2.circle(img, (int(coord[0]), int(coord[1])), 3, color, -1)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    if save_name is not None:
        cv2.imwrite(save_name, img)

    return img
