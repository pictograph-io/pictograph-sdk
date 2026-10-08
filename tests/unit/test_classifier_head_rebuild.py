"""The `.pth` loader must rebuild EVERY supported classification backbone.

LOCKSTEP with the training pipeline's own classifier-head split.

The rule: a Sequential classifier's head input is the FIRST ``nn.Linear``, and every
structural module before it must be PRESERVED. The SDK previously used a positional
index into the Sequential, which encodes the LAST Linear - correct only for
EfficientNet. The training pipeline had the identical bug and it made four backbones
untrainable outright:

    mobilenet_v3_small   mat1 and mat2 shapes cannot be multiplied (2x576 and 1024x256)
    mobilenet_v3_large   ... (2x960 and 1280x256)
    convnext_tiny        ... (1536x1 and 768x256)
    convnext_small       ... (1536x1 and 768x256)

MobileNetV3's classifier is ``(Linear(960,1280), Hardswish, Dropout, Linear(1280,1000))``
- the head's real input is 960, not the last Linear's 1280. ConvNeXt's is
``(LayerNorm2d, Flatten, Linear(768,1000))`` - the width is right, but replacing the
WHOLE Sequential deletes the LayerNorm2d and Flatten, so the head receives an
un-flattened ``(N, 768, 1, 1)``.

Those four can now be trained, so the loader MUST be able to rebuild them or a user
trains a model the SDK cannot load.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
tv = pytest.importorskip("torchvision.models")

import torch.nn as nn  # noqa: E402

from pictograph.inference._torch import (  # noqa: E402
    _CLS_HEADS,
    _build_torchvision,
    _classifier_hidden_units,
    _sequential_head_split,
)

# Measured from torchvision, and corroborated by the shapes in the training
# errors above. `keep` is the count of structural modules preserved ahead of the head.
EXPECTED: dict[str, tuple[int, int]] = {
    "resnet18": (512, 0),
    "resnet34": (512, 0),
    "resnet50": (2048, 0),
    "resnet101": (2048, 0),
    "efficientnet_b0": (1280, 0),
    "efficientnet_b1": (1280, 0),
    "efficientnet_b2": (1408, 0),
    "efficientnet_b3": (1536, 0),
    "efficientnet_b4": (1792, 0),
    "mobilenet_v3_small": (576, 0),
    "mobilenet_v3_large": (960, 0),
    "convnext_tiny": (768, 2),
    "convnext_small": (768, 2),
    "vit_b_16": (768, 0),
    "vit_b_32": (768, 0),
}

PREVIOUSLY_UNTRAINABLE = (
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "convnext_tiny",
    "convnext_small",
)


def _head_input(model: object, attr: str, kind: str) -> tuple[int, list[object]]:
    if kind == "sequential":
        return _sequential_head_split(getattr(model, attr))
    if kind == "linear":
        return getattr(model, attr).in_features, []
    return model.heads.head.in_features, []  # type: ignore[attr-defined]


class TestEveryBackboneRebuilds:
    @pytest.mark.parametrize("backbone", sorted(_CLS_HEADS))
    def test_round_trips_through_the_pth_loader(self, backbone: str) -> None:
        """Build a head the way the PIPELINE does, then rebuild it from its state dict."""
        attr, kind = _CLS_HEADS[backbone]
        model = getattr(tv, backbone)(weights=None)
        in_features, keep = _head_input(model, attr, kind)

        expected_in, expected_keep = EXPECTED[backbone]
        assert in_features == expected_in, f"{backbone}: head input width"
        assert len(keep) == expected_keep, f"{backbone}: preserved structural modules"

        setattr(
            model,
            attr,
            nn.Sequential(
                *keep,
                nn.Dropout(0.5),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 3),
            ),
        )
        state = model.state_dict()

        rebuilt = _build_torchvision(
            state, {"backbone": backbone, "hidden_units": 256}, backbone, 3
        )
        with torch.inference_mode():
            out = rebuilt(torch.zeros(2, 3, 224, 224))
        assert tuple(out.shape) == (2, 3)


class TestTheFourThatCouldNotTrain:
    """These are the whole point - they were unbuildable before the fix."""

    @pytest.mark.parametrize("backbone", PREVIOUSLY_UNTRAINABLE)
    def test_head_width_is_not_the_last_linear(self, backbone: str) -> None:
        attr, kind = _CLS_HEADS[backbone]
        classifier = getattr(getattr(tv, backbone)(weights=None), attr)
        first, _ = _sequential_head_split(classifier)
        linears = [m for m in classifier if isinstance(m, nn.Linear)]
        last = linears[-1].in_features
        # For all four, the OLD rule (last Linear) disagrees with the correct one,
        # which is exactly why they raised at the first forward pass.
        assert first == EXPECTED[backbone][0]
        if backbone.startswith("mobilenet"):
            assert first != last, "mobilenet's first and last Linear widths must differ"

    def test_convnext_preserves_its_structural_prefix(self) -> None:
        """Dropping LayerNorm2d + Flatten leaves the head an un-flattened 4-D input."""
        classifier = tv.convnext_tiny(weights=None).classifier
        _, keep = _sequential_head_split(classifier)
        assert [type(m).__name__ for m in keep] == ["LayerNorm2d", "Flatten"]


class TestHiddenWidthProbe:
    """A preserved prefix shifts every index, so a fixed `{attr}.1.weight` lookup breaks."""

    def test_finds_the_first_2d_weight_past_a_structural_prefix(self) -> None:
        # ConvNeXt shape: LayerNorm2d(1-D weight) at .0, Flatten at .1 (no params),
        # Dropout at .2, the head's first Linear at .3.
        state = {
            "classifier.0.weight": torch.zeros(768),  # LayerNorm - 1-D, must be ignored
            "classifier.3.weight": torch.zeros(256, 768),  # the head's first Linear
            "classifier.6.weight": torch.zeros(3, 256),  # the output Linear
        }
        assert _classifier_hidden_units(state, "classifier") == 256

    def test_no_prefix_still_works(self) -> None:
        state = {
            "classifier.1.weight": torch.zeros(512, 1280),
            "classifier.4.weight": torch.zeros(7, 512),
        }
        assert _classifier_hidden_units(state, "classifier") == 512

    def test_absent_returns_none_so_the_config_fallback_applies(self) -> None:
        assert _classifier_hidden_units({"backbone.0.weight": torch.zeros(8, 8)}, "fc") is None
