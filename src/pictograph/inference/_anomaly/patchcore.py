# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Pictograph (2026-11-24): the INFERENCE path of anomalib 2.7.1's PatchCore
# (``anomalib/models/image/patchcore/torch_model.py`` ``PatchcoreModel.forward`` in eval mode,
# ``anomaly_map.py`` ``AnomalyMapGenerator`` and ``models/components/filters/blur.py``
# ``GaussianBlur2d``), extracted as one export-ready module: the timm feature extractor is a
# torchvision ResNet trunk carrying the same weights under the same names, ImageNet
# normalization (anomalib's pre-processor) moved inside the module, no training path, no
# chunked search, no tiler. See NOTICE.
"""PatchCore, inference only: an image in [0, 1] -> (anomaly map, image score), raw.

The arithmetic is anomalib 2.7.1's, operation for operation, so a model trained by anomalib
and copied in here answers what anomalib answers (the training run gates it):

1. ImageNet normalization, ``(x - mean) / std`` (torchvision's ``Normalize``).
2. The backbone's ``layer2`` and ``layer3`` outputs (BatchNorm on its running statistics).
3. Each passed through ``AvgPool2d(3, 1, 1)`` (padding counted, PyTorch's default).
4. ``layer3`` resized to ``layer2``'s grid (bilinear, ``align_corners=False``), concatenated.
5. Every patch's Euclidean distance to its nearest memory-bank entry,
   ``sqrt(max(|a|^2 - 2 a.b + |b|^2, 0))`` (anomalib's ``euclidean_dist``).
6. The image score: the farthest patch's distance, reweighted by the softmax over its nearest
   neighbour's ``num_neighbors`` support samples (the paper's eq. 7; anomalib's
   ``compute_anomaly_score``).
7. The anomaly map: the patch distances resized to the input (nearest), then a Gaussian blur
   (sigma 4, a 33-tap kernel, reflect padding).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

if TYPE_CHECKING:
    from collections.abc import Sequence

#: ImageNet statistics, as anomalib's pre-processor normalizes with them.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Backbone(NamedTuple):
    """A torchvision ResNet recipe (the timm model of the same name has the same layout and
    parameter names) and each stage's output channels."""

    block: str  # "bottleneck" | "basic"
    stages: tuple[int, int, int, int]
    width_per_group: int
    channels: dict[str, int]


#: The backbones a PatchCore model may be built on.
BACKBONES: dict[str, Backbone] = {
    "wide_resnet50_2": Backbone(
        "bottleneck",
        (3, 4, 6, 3),
        128,
        {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048},
    ),
    "resnet18": Backbone(
        "basic",
        (2, 2, 2, 2),
        64,
        {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512},
    ),
}

_LAYER_ORDER = ("layer1", "layer2", "layer3", "layer4")


def feature_dim(backbone: str, layers: Sequence[str]) -> int:
    """The memory bank's width: the channels of every layer concatenated."""
    return sum(BACKBONES[backbone].channels[layer] for layer in layers)


def gaussian_kernel(sigma: float) -> torch.Tensor:
    """anomalib's blur kernel: ``2 * int(4 * sigma + 0.5) + 1`` taps of ``exp(-n^2 / 2 sigma^2)``,
    each axis normalized to sum 1 (kornia's ``get_gaussian_kernel1d``), their outer product
    normalized again (``normalize_kernel2d``). Shape ``(1, 1, k, k)``, float32.

    A trained model carries the kernel anomalib built as a buffer, so this only seeds a module
    built without weights (it agrees with kornia's to float32 rounding: tested)."""
    size = 2 * int(4.0 * sigma + 0.5) + 1
    taps = torch.arange(size, dtype=torch.float32) - (size // 2)
    g = torch.exp(-(taps**2) / (2.0 * float(sigma) ** 2))
    g = g / g.sum()
    k2 = g[:, None] * g[None, :]
    k2 = k2 / k2.abs().sum()
    return k2.view(1, 1, size, size)


class ResNetTrunk(nn.Module):
    """A torchvision ResNet's stem and its first ``depth`` stages, returning the named stages'
    outputs. Parameter names match torchvision's (and timm's) ``conv1`` / ``bn1`` / ``layerN``."""

    def __init__(self, backbone: str, layers: Sequence[str]) -> None:
        super().__init__()
        from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

        recipe = BACKBONES[backbone]
        block = Bottleneck if recipe.block == "bottleneck" else BasicBlock
        net = ResNet(block, list(recipe.stages), width_per_group=recipe.width_per_group)
        self.out_layers = tuple(layers)
        self.depth = max(_LAYER_ORDER.index(layer) for layer in layers) + 1
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        for i in range(self.depth):
            setattr(self, _LAYER_ORDER[i], getattr(net, _LAYER_ORDER[i]))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        outs: dict[str, torch.Tensor] = {}
        for i in range(self.depth):
            name = _LAYER_ORDER[i]
            x = getattr(self, name)(x)
            outs[name] = x
        return [outs[name] for name in self.out_layers]


class PatchcoreInference(nn.Module):
    """PatchCore's inference path. Input ``(B, 3, H, W)`` RGB in [0, 1] (H and W multiples of 8);
    output ``(anomaly_map (B, 1, H, W), pred_score (B, 1))``, raw (the caller normalizes)."""

    pixel_mean: torch.Tensor
    pixel_std: torch.Tensor
    memory_bank: torch.Tensor
    blur_kernel: torch.Tensor

    def __init__(
        self,
        *,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3"),
        memory_bank_size: int = 1,
        num_neighbors: int = 9,
        sigma: float = 4.0,
    ) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown PatchCore backbone {backbone!r}; one of {sorted(BACKBONES)}")
        self.backbone = backbone
        self.layers = tuple(layers)
        self.num_neighbors = int(num_neighbors)
        dim = feature_dim(backbone, self.layers)
        # Constants, not trained state: kept out of the state dict (persistent=False), so a
        # trained state holds only what training produced.
        self.register_buffer(
            "pixel_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.trunk = ResNetTrunk(backbone, self.layers)
        self.feature_pooler = nn.AvgPool2d(3, 1, 1)
        # The bank's size is the module's shape, fixed when it is built (a trained state of
        # another size refuses to load), so the support search's k is a constant of the graph.
        self.memory_bank_size = int(memory_bank_size)
        self.register_buffer("memory_bank", torch.zeros(self.memory_bank_size, dim))
        kernel = gaussian_kernel(sigma)
        self.register_buffer("blur_kernel", kernel)
        self.blur_padding = (kernel.shape[-1] - 1) // 2

    @staticmethod
    def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """anomalib's ``PatchcoreModel.euclidean_dist``, the same operations in the same order. Its floor is
        written ``0.0`` where anomalib writes ``0``: the same value on a float tensor and the same ONNX graph, and
        the form Core ML converts (coremltools refuses an integer bound on a float tensor)."""
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)
        res = torch.matmul(x, y.transpose(-2, -1))
        res = res * -2
        res = res + x_norm
        res = res + y_norm.transpose(-2, -1)
        return res.clamp_min(0.0).sqrt()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """The patch embedding ``(B, D, h, w)`` of an image in [0, 1]."""
        x = (x - self.pixel_mean) / self.pixel_std
        feats: list[torch.Tensor] = [self.feature_pooler(f) for f in self.trunk(x)]
        embedding = feats[0]
        for f in feats[1:]:
            f = F.interpolate(f, size=embedding.shape[-2:], mode="bilinear")
            embedding = torch.cat((embedding, f), 1)
        return embedding

    def blur(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        p = self.blur_padding
        padded = F.pad(anomaly_map, (p, p, p, p), mode="reflect")
        blurred: torch.Tensor = F.conv2d(padded, self.blur_kernel, groups=1, padding=0, stride=1)
        return blurred

    def score(
        self, patch_scores: torch.Tensor, locations: torch.Tensor, embedding: torch.Tensor
    ) -> torch.Tensor:
        """anomalib's ``compute_anomaly_score``: ``(B, P)`` patch scores -> ``(B,)``."""
        if self.num_neighbors == 1:
            return patch_scores.amax(1)
        # anomalib indexes with ``[torch.arange(B), max_patches]``; ``gather`` / ``index_select``
        # read the same elements and export as plain Gather nodes on every runtime.
        batch_size, num_patches = patch_scores.shape
        dim = embedding.shape[-1]
        max_patches = torch.argmax(patch_scores, dim=1, keepdim=True)  # (B, 1)
        features = embedding.reshape(batch_size, num_patches, dim)
        max_patches_features = features.gather(1, max_patches.unsqueeze(-1).expand(-1, -1, dim))
        score = patch_scores.gather(1, max_patches).squeeze(1)
        nn_index = locations.gather(1, max_patches).squeeze(1)
        nn_sample = self.memory_bank.index_select(0, nn_index)
        k = min(self.num_neighbors, self.memory_bank_size)
        distances = self.euclidean_dist(nn_sample, self.memory_bank)
        _, support_samples = distances.topk(k=k, largest=False, dim=1)
        support = self.memory_bank.index_select(0, support_samples.reshape(-1))
        distances = self.euclidean_dist(max_patches_features, support.reshape(batch_size, k, dim))
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        return weights * score

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output_size = x.shape[-2:]
        embedding = self.embed(x)
        batch_size, dim, height, width = embedding.shape
        flat = embedding.permute(0, 2, 3, 1).reshape(-1, dim)
        distances = self.euclidean_dist(flat, self.memory_bank)
        patch_scores, locations = distances.min(1)
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))
        pred_score = self.score(patch_scores, locations, flat)
        patch_map = patch_scores.reshape((batch_size, 1, height, width))
        anomaly_map = self.blur(F.interpolate(patch_map, size=(output_size[0], output_size[1])))
        return anomaly_map, pred_score.reshape(batch_size, 1)


def patchcore_state_from_anomalib(
    anomalib_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """A trained anomalib ``PatchcoreModel``'s state dict -> this module's: the timm feature
    extractor's stages (``feature_extractor.feature_extractor.<name>``) become the trunk's
    (``trunk.<name>``), the memory bank and the blur kernel keep their values. Anything else
    anomalib keeps (the embedding store is not state) is not inference."""
    out: dict[str, torch.Tensor] = {}
    prefix = "feature_extractor.feature_extractor."
    for key, value in anomalib_state.items():
        if key.startswith(prefix):
            out["trunk." + key[len(prefix) :]] = value
        elif key == "memory_bank":
            out["memory_bank"] = value
        elif key == "anomaly_map_generator.blur.kernel":
            out["blur_kernel"] = value
    if "memory_bank" not in out or not math.prod(out["memory_bank"].shape):
        raise ValueError("the anomalib state carries no memory bank: was the model fitted?")
    return out
