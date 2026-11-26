# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Pictograph (2026-11-24): the INFERENCE path of anomalib 2.7.1's EfficientAD
# (``anomalib/models/image/efficient_ad/torch_model.py``): the patch description networks, the
# autoencoder and ``EfficientAdModel.forward`` in eval mode with its quantile normalization, as
# one export-ready module. The training losses, the random augmentation and the "is the
# statistic set" branches are gone (a trained model always carries its statistics, as plain
# buffers). Network definitions and parameter names are anomalib's, unchanged. See NOTICE.
"""EfficientAD, inference only: an image in [0, 1] -> (anomaly map, image score), raw.

The arithmetic is anomalib 2.7.1's, operation for operation (the training run gates it):

1. The teacher's output, normalized by its channel mean and std over the training images.
2. ``map_st``: the mean over channels of (teacher - student[:, :C])^2.
3. ``map_stae``: the mean over channels of (autoencoder - student[:, C:])^2.
4. Both zero-padded by 4 (``pad_maps``), resized to the input (bilinear), each mapped by its
   validation quantiles, ``0.1 * (m - q_a) / (q_b - q_a)``.
5. The anomaly map, ``0.5 * map_st + 0.5 * map_stae``; the score, its maximum.

Every network normalizes its input with ImageNet statistics itself (anomalib's
``imagenet_norm_batch``), so the module takes an image in [0, 1].
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

#: The sizes anomalib's EfficientAD comes in (``EfficientAdModelSize``).
MODEL_SIZES = ("small", "medium")


def imagenet_norm_batch(x: torch.Tensor) -> torch.Tensor:
    """anomalib's ``imagenet_norm_batch``, its constants in the input's dtype: identical at float32 (anomalib's), and a
    network halved for an FP16 graph stays half instead of being promoted back to float32 here (ROADMAP Program N)."""
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(
        device=x.device, dtype=x.dtype
    )
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(
        device=x.device, dtype=x.dtype
    )
    return (x - mean) / std


class SmallPatchDescriptionNetwork(nn.Module):
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        out: torch.Tensor = self.conv4(x)
        return out


class MediumPatchDescriptionNetwork(nn.Module):
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.conv6 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0 * pad_mult
        )
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        out: torch.Tensor = self.conv6(x)
        return out


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enconv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        out: torch.Tensor = self.enconv6(x)
        return out


class Decoder(nn.Module):
    def __init__(self, out_channels: int, padding: int) -> None:
        super().__init__()
        self.padding = padding
        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        # anomalib's six Dropout(0.2) layers are identities at inference and hold no state.

    def forward(self, x: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        last_upsample = (
            math.ceil(image_size[0] / 4) if self.padding else math.ceil(image_size[0] / 4) - 8,
            math.ceil(image_size[1] / 4) if self.padding else math.ceil(image_size[1] / 4) - 8,
        )
        x = F.interpolate(
            x, size=(image_size[0] // 64 - 1, image_size[1] // 64 - 1), mode="bilinear"
        )
        x = F.relu(self.deconv1(x))
        x = F.interpolate(x, size=(image_size[0] // 32, image_size[1] // 32), mode="bilinear")
        x = F.relu(self.deconv2(x))
        x = F.interpolate(
            x, size=(image_size[0] // 16 - 1, image_size[1] // 16 - 1), mode="bilinear"
        )
        x = F.relu(self.deconv3(x))
        x = F.interpolate(x, size=(image_size[0] // 8, image_size[1] // 8), mode="bilinear")
        x = F.relu(self.deconv4(x))
        x = F.interpolate(x, size=(image_size[0] // 4 - 1, image_size[1] // 4 - 1), mode="bilinear")
        x = F.relu(self.deconv5(x))
        x = F.interpolate(x, size=(image_size[0] // 2 - 1, image_size[1] // 2 - 1), mode="bilinear")
        x = F.relu(self.deconv6(x))
        x = F.interpolate(x, size=last_upsample, mode="bilinear")
        x = F.relu(self.deconv7(x))
        out: torch.Tensor = self.deconv8(x)
        return out


class AutoEncoder(nn.Module):
    def __init__(self, out_channels: int, padding: int) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(out_channels, padding)

    def forward(self, x: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        x = imagenet_norm_batch(x)
        x = self.encoder(x)
        out: torch.Tensor = self.decoder(x, image_size)
        return out


class EfficientAdInference(nn.Module):
    """EfficientAD's inference path. Input ``(B, 3, H, W)`` RGB in [0, 1]; output
    ``(anomaly_map (B, 1, H, W), pred_score (B, 1))``, raw (the caller normalizes)."""

    teacher_mean: torch.Tensor
    teacher_std: torch.Tensor
    qa_st: torch.Tensor
    qb_st: torch.Tensor
    qa_ae: torch.Tensor
    qb_ae: torch.Tensor

    def __init__(
        self,
        *,
        model_size: str = "small",
        teacher_out_channels: int = 384,
        padding: bool = False,
        pad_maps: bool = True,
    ) -> None:
        super().__init__()
        if model_size not in MODEL_SIZES:
            raise ValueError(f"unknown EfficientAD size {model_size!r}; one of {MODEL_SIZES}")
        self.model_size = model_size
        self.teacher_out_channels = int(teacher_out_channels)
        self.pad_maps = bool(pad_maps)
        pdn = (
            MediumPatchDescriptionNetwork
            if model_size == "medium"
            else SmallPatchDescriptionNetwork
        )
        self.teacher = pdn(out_channels=self.teacher_out_channels, padding=padding)
        self.student = pdn(out_channels=self.teacher_out_channels * 2, padding=padding)
        self.ae = AutoEncoder(out_channels=self.teacher_out_channels, padding=int(padding))
        c = self.teacher_out_channels
        self.register_buffer("teacher_mean", torch.zeros((1, c, 1, 1)))
        self.register_buffer("teacher_std", torch.ones((1, c, 1, 1)))
        for name in ("qa_st", "qb_st", "qa_ae", "qb_ae"):
            self.register_buffer(name, torch.tensor(0.0))
        self.qb_st.fill_(1.0)
        self.qb_ae.fill_(1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_size = (int(x.shape[-2]), int(x.shape[-1]))
        c = self.teacher_out_channels
        teacher_output = (self.teacher(x) - self.teacher_mean) / self.teacher_std
        student_output = self.student(x)
        distance_st = torch.pow(teacher_output - student_output[:, :c, :, :], 2)
        ae_output = self.ae(x, image_size)
        map_st = torch.mean(distance_st, dim=1, keepdim=True)
        map_stae = torch.mean((ae_output - student_output[:, c:]) ** 2, dim=1, keepdim=True)
        if self.pad_maps:
            map_st = F.pad(map_st, (4, 4, 4, 4))
            map_stae = F.pad(map_stae, (4, 4, 4, 4))
        map_st = F.interpolate(map_st, size=image_size, mode="bilinear")
        map_stae = F.interpolate(map_stae, size=image_size, mode="bilinear")
        map_st = 0.1 * (map_st - self.qa_st) / (self.qb_st - self.qa_st)
        map_stae = 0.1 * (map_stae - self.qa_ae) / (self.qb_ae - self.qa_ae)
        anomaly_map = 0.5 * map_st + 0.5 * map_stae
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return anomaly_map, pred_score


def efficientad_state_from_anomalib(
    anomalib_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """A trained anomalib ``EfficientAdModel``'s state dict -> this module's: the networks keep
    their names; ``mean_std.{mean,std}`` and ``quantiles.{qa_st,...}`` (parameter dicts there)
    become plain buffers here. Refuses a state whose statistics were never set (anomalib's own
    ``is_set``: any non-zero value)."""
    out: dict[str, torch.Tensor] = {}
    for key, value in anomalib_state.items():
        if key.startswith(("teacher.", "student.", "ae.")):
            out[key] = value
        elif key == "mean_std.mean":
            out["teacher_mean"] = value
        elif key == "mean_std.std":
            out["teacher_std"] = value
        elif key.startswith("quantiles."):
            out[key.split(".", 1)[1]] = value
    if not bool((out["teacher_mean"] != 0).any() or (out["teacher_std"] != 0).any()):
        raise ValueError("the anomalib state's teacher statistics were never set")
    if not any(bool(out[q] != 0) for q in ("qa_st", "qb_st", "qa_ae", "qb_ae")):
        raise ValueError("the anomalib state's map quantiles were never set")
    return out
