"""Unit tests for the local PyTorch loader (pictograph.inference._torch).

Family resolution, rebuild-recipe constants, checkpoint introspection, and
image decoding run with NO framework installed (fakes via sys.modules).
End-to-end download → rebuild → strict-load → predict was live-verified
against real trained models of all five pipelines (2026-07-17); the
recipes here are pinned so a drift from the pipelines' builders is caught.

Image-input decoding (dispatch over path/bytes/PIL/URL/ndarray, formerly
`_decode_to_pil`) moved to the ONE shared `pictograph.inference.models._decode_image`
both engines now call before handing off to their own preprocessing --
covered in `test_inference.py::TestImageDecode`. This module's own
`_bgr_to_pil` only ever receives an already-decoded BGR array from that
shared decoder; its behaviour is exercised through the keypoint adapter
tests in `test_pytorch_keypoint.py`.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from pictograph.inference._torch import (
    _CLS_HEADS,
    _YOLOX_SIZES,
    _pytorch_input_size,
    _resolve_family,
    _rfdetr_ckpt_class_names,
    _yolox_size,
)


class TestResolveFamily:
    def test_unambiguous_model_types(self) -> None:
        assert _resolve_family("classification", "resnet18", {}) == "torchvision"
        assert (
            _resolve_family("semantic_segmentation", "unetplusplus", {})
            == "segmentation_models_pytorch"
        )
        assert _resolve_family("instance_segmentation", "nano", {}) == "rfdetr"

    def test_detection_by_architecture(self) -> None:
        assert _resolve_family("object_detection", "YOLOX-S", {}) == "yolox"
        assert _resolve_family("object_detection", "yolox", {}) == "yolox"
        assert _resolve_family("object_detection", "RF-DETR Medium", {}) == "rfdetr"
        assert _resolve_family("object_detection", "rfdetr-base", {}) == "rfdetr"

    def test_bare_size_falls_back_to_checkpoint_keys(self) -> None:
        """The ambiguity class that mis-dispatched SDK-created yolox models:
        a bare size label resolves by the checkpoint's own shape."""
        rfdetr_ckpt = {"model": {}, "args": object(), "model_name": "rfdetr_medium"}
        yolox_ckpt = {"model": {}, "start_epoch": 1}
        assert _resolve_family("object_detection", "s", yolox_ckpt) == "yolox"
        assert _resolve_family("object_detection", "nano", rfdetr_ckpt) == "rfdetr"
        assert _resolve_family("object_detection", "medium", rfdetr_ckpt) == "rfdetr"

    def test_unrecognizable_checkpoint_defaults_to_rfdetr(self) -> None:
        assert _resolve_family("object_detection", "medium", "not-a-dict") == "rfdetr"


class TestYoloxSize:
    def test_config_model_size_wins(self) -> None:
        assert _yolox_size({"model_size": "m"}, "YOLOX-S") == "m"

    def test_architecture_prefix_stripped(self) -> None:
        assert _yolox_size({}, "YOLOX-S") == "s"
        assert _yolox_size({}, "yolox-nano") == "nano"
        assert _yolox_size({}, "l") == "l"

    def test_unknown_size_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown YOLOX size"):
            _yolox_size({}, "yolox-huge")

    def test_sizes_match_training_pipeline(self) -> None:
        # LOCKSTEP with the YOLOX training pipeline's own size table.
        assert _YOLOX_SIZES == {
            "nano": (0.33, 0.25),
            "tiny": (0.33, 0.375),
            "s": (0.33, 0.50),
            "m": (0.67, 0.75),
            "l": (1.00, 1.00),
            "x": (1.33, 1.25),
        }


class TestClassifierHeads:
    def test_backbone_table_matches_training_pipeline(self) -> None:
        """LOCKSTEP with the classification pipeline's backbone table - `(attr, kind)`.

        The positional index this table used to carry is GONE, and its absence is the
        point: the index encoded the LAST Linear in the Sequential, which is the
        head's input width only for EfficientNet. This test previously asserted
        `mobilenet_v3_large == (..., 3)` and `convnext_tiny == (..., 2)` - the exact
        values that made those backbones unbuildable, pinned as if correct. The head
        input is now derived by `_sequential_head_split` (first `nn.Linear`, structural
        prefix preserved); see tests/unit/test_classifier_head_rebuild.py, which
        round-trips all 15 backbones through the real loader.
        """
        assert _CLS_HEADS["resnet50"] == ("fc", "linear")
        assert _CLS_HEADS["efficientnet_b4"] == ("classifier", "sequential")
        assert _CLS_HEADS["mobilenet_v3_large"] == ("classifier", "sequential")
        assert _CLS_HEADS["convnext_tiny"] == ("classifier", "sequential")
        assert _CLS_HEADS["vit_b_16"] == ("heads", "vit")
        assert len(_CLS_HEADS) == 15
        # No entry may carry a positional index again.
        assert all(len(v) == 2 for v in _CLS_HEADS.values())


class TestCkptClassNames:
    def test_names_from_args_namespace(self) -> None:
        args = types.SimpleNamespace(class_names=["cat", "dog"])
        assert _rfdetr_ckpt_class_names({"args": args}) == ["cat", "dog"]

    def test_names_from_args_dict(self) -> None:
        assert _rfdetr_ckpt_class_names({"args": {"class_names": ["a"]}}) == ["a"]

    def test_absent_or_malformed(self) -> None:
        assert _rfdetr_ckpt_class_names({}) is None
        assert _rfdetr_ckpt_class_names({"args": {}}) is None
        assert _rfdetr_ckpt_class_names({"args": {"class_names": [1, 2]}}) is None
        assert _rfdetr_ckpt_class_names("nope") is None


class TestInputSize:
    def test_family_defaults(self) -> None:
        assert _pytorch_input_size("classification", {}) == (224, 224)
        assert _pytorch_input_size("semantic_segmentation", {}) == (512, 512)
        assert _pytorch_input_size("object_detection", {}) == (640, 640)

    def test_config_overrides(self) -> None:
        assert _pytorch_input_size("classification", {"image_height": 299, "image_width": 299}) == (
            299,
            299,
        )

    def test_garbage_config_falls_back(self) -> None:
        assert _pytorch_input_size("object_detection", {"image_height": "x"}) == (640, 640)


class _FakeSmpModel:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.loaded: Any = None

    def load_state_dict(self, sd: Any, strict: bool = False) -> None:
        assert strict is True
        self.loaded = sd

    def eval(self) -> None:
        pass


class TestBuildSmp:
    def test_mirrors_pipeline_create_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multi-class = classes + background channel, encoder_weights None,
        activation None - LOCKSTEP with train_semantic_seg.create_model."""
        fake = types.ModuleType("segmentation_models_pytorch")
        fake.UnetPlusPlus = _FakeSmpModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "segmentation_models_pytorch", fake)

        from pictograph.inference._torch import _build_smp

        model = _build_smp({"raw": "sd"}, {"encoder": "resnet34"}, "unetplusplus", ["a", "b", "c"])
        assert model.kwargs == {
            "encoder_name": "resnet34",
            "encoder_weights": None,
            "in_channels": 3,
            "classes": 4,
            "activation": None,
        }
        assert model.loaded == {"raw": "sd"}

    def test_single_class_is_sigmoid_binary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = types.ModuleType("segmentation_models_pytorch")
        fake.Unet = _FakeSmpModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "segmentation_models_pytorch", fake)

        from pictograph.inference._torch import _build_smp

        model = _build_smp({}, {}, "unet", ["only"])
        assert model.kwargs["classes"] == 1
        assert model.kwargs["activation"] == "sigmoid"

    def test_unknown_architecture_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = types.ModuleType("segmentation_models_pytorch")
        monkeypatch.setitem(sys.modules, "segmentation_models_pytorch", fake)

        from pictograph.inference._torch import _build_smp

        with pytest.raises(ValueError, match="Unknown segmentation architecture"):
            _build_smp({}, {}, "hrnet", ["a"])


class TestMissingFrameworkMessages:
    def test_rfdetr_needs_no_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RF-DETR must rebuild with `pictograph` alone - no `rfdetr`, no `transformers`.

        This used to assert the opposite: that a missing `rfdetr` raised with a
        `pip install rfdetr` hint. The architecture is vendored now
        (`pictograph.inference._rfdetr`), so making either package unimportable must
        change nothing. The checkpoint below does not exist, so the assertion is
        that we get that far - a `FileNotFoundError` about the PATH, never an
        `ImportError` about a package.
        """
        # `_build_rfdetr` reaches the vendored architecture, which needs torch. CI
        # does not install the [inference] extra, so this ERRORed instead of
        # skipping - the one test in this module that did.
        pytest.importorskip("torch", reason="needs the [inference] extra")

        for absent in ("rfdetr", "transformers", "supervision"):
            monkeypatch.setitem(sys.modules, absent, None)  # any import raises

        from pictograph.inference._torch import _build_rfdetr

        with pytest.raises(FileNotFoundError):
            _build_rfdetr(Path("/tmp/pictograph-nonexistent-checkpoint.pth"))

    def test_no_install_hint_mentions_rfdetr(self) -> None:
        """No error message anywhere in the engine may tell a user to install rfdetr."""
        from pictograph.inference import _torch

        source = Path(_torch.__file__).read_text()
        assert "pip install rfdetr" not in source

    def test_yolox_needs_no_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """YOLOX must rebuild with `pictograph` alone - no `yolox`, no `loguru`.

        This used to assert the opposite: that a missing `yolox` raised with a
        `pip install git+https://github.com/Megvii-BaseDetection/YOLOX.git` hint.
        The architecture is vendored now (`pictograph.inference._yolox`), so making
        every upstream package unimportable must change nothing. The checkpoint
        below carries an empty state dict, so the assertion is that we get as far
        as the strict load and fail THERE - on missing tensors, never on an
        `ImportError` about a package.
        """
        for absent in ("yolox", "yolox.models", "loguru", "thop", "tabulate"):
            monkeypatch.setitem(sys.modules, absent, None)  # any import raises

        pytest.importorskip("torch")
        from pictograph.inference._torch import _build_yolox

        with pytest.raises(RuntimeError, match="Missing key"):
            _build_yolox({"model": {}}, {"model_size": "s"}, "s", 3)

    def test_no_install_hint_mentions_yolox(self) -> None:
        """No error message anywhere in the engine may tell a user to install YOLOX."""
        from pictograph.inference import _torch

        source = Path(_torch.__file__).read_text()
        assert "pip install yolox" not in source
        assert "Megvii-BaseDetection/YOLOX.git" not in source

    def test_no_hint_names_a_third_party_package(self) -> None:
        """Every `pip install` the engine can print must be a `pictograph[...]` extra.

        The defect this pins (2026-07-31): the app's install snippet told a user to
        `pip install "pictograph[inference]"` and then, on a second line,
        `pip install torch segmentation-models-pytorch` - our own undeclared
        dependencies, handed over as the reader's homework. The snippet copied
        those lines FROM these hints, so the durable fix has to hold here too:
        whatever a model needs is declared by an extra or vendored into the wheel,
        and the only install command we are allowed to print names `pictograph`.
        """
        # The ONE sanctioned exception, and it is a substitution rather than an
        # addition: `onnxruntime-gpu` REPLACES `onnxruntime` (installing both puts
        # two providers of the same module in one environment), so it cannot be an
        # extra alongside it. It is also not needed to load anything - it only
        # makes an already-working model faster.
        allowed = {"onnxruntime-gpu"}

        # Only real COMMANDS, not prose that mentions the words: `pip install`
        # followed by something that looks like a requirement token.
        command = re.compile(r"""pip install\s+["']?([A-Za-z][A-Za-z0-9_.,\[\]-]*)""")

        offenders: list[tuple[str, str]] = []
        root = Path(_torch_module_root())
        for path in sorted(root.rglob("*.py")):
            # The vendored trees carry upstream provenance, not instructions.
            if "_rfdetr" in path.parts or "_yolox" in path.parts:
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                for target in command.findall(line):
                    if target.startswith("pictograph") or target in allowed:
                        continue
                    offenders.append((path.name, line.strip()))
        assert not offenders, f"install hints naming a third-party package: {offenders}"


def _torch_module_root() -> str:
    """The `pictograph/inference/` directory, resolved from the installed package."""
    from pictograph import inference

    return str(Path(inference.__file__).parent)
