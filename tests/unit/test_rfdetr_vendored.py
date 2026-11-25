"""Guards on the vendored RF-DETR architecture (``pictograph.inference._rfdetr``).

RF-DETR weights can only be run natively by first rebuilding the exact ``nn.Module``
they were trained as. That architecture used to come from ``pip install rfdetr``,
which also pulls ``transformers>=5.1,<6`` and ``supervision`` into a user's
environment. It is vendored now, and these tests exist so it stays that way:

1. **No dependency, in either direction.** Nothing in the shipped SDK may import
   `rfdetr`/`transformers`/`supervision` on the RF-DETR path, declare them as a
   dependency, or tell a user to install one.
2. **The pinned version is the training image's version.** The weights encode the
   architecture of the release the trainer used; re-vendoring from a different one
   is how a checkpoint starts loading cleanly and predicting nonsense.
3. **Variant resolution stays correct.** The substring map is ORDER-DEPENDENT -
   ``seg-nano`` must not be shadowed by ``nano``.
4. **Apache-2.0 attribution ships in the wheel.**

The end-to-end proof (rebuild a real checkpoint, compare against its ONNX twin) is
not here - it needs real weights and torch. It lives in ``tests/live/``.

5. **Bit-identical to the release it claims to be** (opt-in, like the YOLOX twin).
   Point ``PICTOGRAPH_RFDETR_UPSTREAM`` at the directory that holds that release's
   ``rfdetr`` package (an unpacked wheel or a site-packages; it needs
   ``transformers``, ``supervision`` and ``pyDeprecate`` importable), and every
   variant is built from the SAME state dict through both trees and required to give
   ``torch.equal`` eval and export forwards. No weights are downloaded::

       PICTOGRAPH_RFDETR_UPSTREAM=/tmp/rfdetr-1.11.2 pytest tests/unit/test_rfdetr_vendored.py -k upstream
"""

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import ENV_TRAINING_SERVICE_SOURCE, companion_skip_reason, companion_source

# tomllib is stdlib in 3.11+; the SDK floor is 3.10, where a bare `import tomllib`
# raises and aborts collection. tomli (the [cli] back-compat pin) covers 3.10.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - the 3.10 back-compat path
    import tomli as tomllib

_SDK_ROOT = Path(__file__).resolve().parents[2]
_VENDORED = _SDK_ROOT / "src" / "pictograph" / "inference" / "_rfdetr"
_SRC = _SDK_ROOT / "src" / "pictograph"
# The training image that produced the weights. It is not part of this
# repository, so the pin comparison below is opt-in (see tests/conftest.py).
_TRAINING_SERVICE = companion_source(ENV_TRAINING_SERVICE_SOURCE)

#: Packages a user must never be required to install to run RF-DETR weights.
_FORBIDDEN = ("rfdetr", "transformers", "supervision")


def _vendored_version() -> str:
    from pictograph.inference._rfdetr import RFDETR_VENDORED_VERSION

    return RFDETR_VENDORED_VERSION


class TestNoDependency:
    def test_not_declared_in_project_metadata(self) -> None:
        """Neither a runtime dependency nor an extra may name any of the three."""
        pyproject = tomllib.loads((_SDK_ROOT / "pyproject.toml").read_text())
        project = pyproject["project"]
        declared = list(project.get("dependencies", []))
        for extra in project.get("optional-dependencies", {}).values():
            declared.extend(extra)

        for spec in declared:
            name = re.split(r"[<>=!~\[; ]", spec.strip(), maxsplit=1)[0].lower()
            assert name not in _FORBIDDEN, f"{name!r} is declared in pyproject: {spec!r}"

    def test_no_module_imports_them_at_top_level(self) -> None:
        """The whole shipped tree, statically: no unconditional import of the three.

        A *deferred* import inside a function is allowed and is used deliberately
        (`transformers.AutoBackbone` on the hub-pretrained branch a reload never
        takes), so this walks only module-level and class-level statements.
        """
        offenders: list[str] = []
        for path in sorted(_SRC.rglob("*.py")):
            tree = ast.parse(path.read_text(), str(path))
            for node in tree.body:
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    names = [node.module]
                for name in names:
                    if name.split(".")[0] in _FORBIDDEN:
                        offenders.append(f"{path.relative_to(_SRC)}: {name}")
        assert not offenders, "top-level imports of a removed dependency:\n" + "\n".join(offenders)

    @pytest.mark.parametrize("package", _FORBIDDEN)
    def test_no_install_hint_in_shipped_source(self, package: str) -> None:
        """No message a USER can see may tell them to install one of them.

        Scoped to runtime string literals - the text of an exception, a log line, a
        `fix=` hint. Docstrings are exempt on purpose: several of them explain what
        `pip install rfdetr` used to pull in and why it no longer does, and that
        prose is the documentation of this decision, not an instruction.
        """
        pattern = re.compile(rf"(pip install|uv add|poetry add)[^\n]*\b{package}\b")
        offenders: list[str] = []
        for path in sorted(_SRC.rglob("*.py")):
            tree = ast.parse(path.read_text(), str(path))
            docstrings = {
                id(node.body[0].value)
                for node in ast.walk(tree)
                if isinstance(
                    node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                )
                and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            }
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docstrings
                    and pattern.search(node.value)
                ):
                    offenders.append(f"{path.relative_to(_SRC)}:{node.lineno}")
        assert not offenders, (
            f"a user-visible install hint for {package!r} survives at: {offenders}"
        )


# Needs the vendored architecture to actually import, which needs torch - not in
# CI's `[dev,cli,agents,cache,telemetry]` install, so these ERRORed rather than
# skipping and left this module RED on main.
@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs the [inference] extra")
class TestPinnedVersion:
    def test_matches_the_training_image(self) -> None:
        """The vendored release must be the one the training image installs.

        A drift here means the shipped weights and the architecture that rebuilds
        them came from different versions of RF-DETR. Re-vendor rather than
        loosening this.
        """
        if not _TRAINING_SERVICE.exists():
            pytest.skip(companion_skip_reason(ENV_TRAINING_SERVICE_SOURCE))
        pinned = re.search(r"rfdetr==([0-9][0-9A-Za-z.]*)", _TRAINING_SERVICE.read_text())
        assert pinned, "no `rfdetr==` pin found in the training service"
        assert _vendored_version() == pinned.group(1), (
            f"vendored RF-DETR is {_vendored_version()} but the training image pins "
            f"{pinned.group(1)} - re-vendor the architecture from the pinned release."
        )

    def test_notice_states_the_same_version(self) -> None:
        notice = (_VENDORED / "NOTICE").read_text()
        assert f"rfdetr=={_vendored_version()}" in notice


# Needs the vendored architecture to actually import, which needs torch - not in
# CI's `[dev,cli,agents,cache,telemetry]` install, so these ERRORed rather than
# skipping and left this module RED on main.
@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs the [inference] extra")
class TestVariantResolution:
    def test_every_variant_maps_to_a_config(self) -> None:
        from pictograph.inference._rfdetr.builder import _NAME_MATCH_ORDER, _VARIANT_CONFIGS

        for _token, variant in _NAME_MATCH_ORDER:
            assert variant in _VARIANT_CONFIGS, f"{variant!r} has no config class"

    @pytest.mark.parametrize(
        ("weights_name", "expected"),
        [
            # The ordering hazard: a plain-substring map would resolve every one of
            # these seg/keypoint names to its base variant instead.
            ("rf-detr-seg-nano.pth", "RFDETRSegNano"),
            ("rf-detr-seg-small.pth", "RFDETRSegSmall"),
            ("rf-detr-seg-medium.pth", "RFDETRSegMedium"),
            ("rf-detr-seg-large.pth", "RFDETRSegLarge"),
            ("rf-detr-seg-xlarge.pth", "RFDETRSegXLarge"),
            ("rf-detr-seg-2xlarge.pth", "RFDETRSeg2XLarge"),
            ("rf-detr-keypoint-preview.pth", "RFDETRKeypointPreview"),
            ("rf-detr-nano.pth", "RFDETRNano"),
            ("rf-detr-large.pth", "RFDETRLarge"),
        ],
    )
    def test_resolves_from_pretrain_weights(self, weights_name: str, expected: str) -> None:
        from pictograph.inference._rfdetr.builder import _resolve_variant

        ckpt = {"args": {"pretrain_weights": weights_name}}
        assert _resolve_variant("/tmp/checkpoint_best_total.pth", ckpt) == expected

    def test_model_name_wins_over_weights_name(self) -> None:
        from pictograph.inference._rfdetr.builder import _resolve_variant

        ckpt = {"model_name": "RFDETRSegMedium", "args": {"pretrain_weights": "rf-detr-nano.pth"}}
        assert _resolve_variant("/tmp/x.pth", ckpt) == "RFDETRSegMedium"

    def test_falls_back_to_the_filename(self) -> None:
        """Our own trainers name the artifact, and some checkpoints store no hint."""
        from pictograph.inference._rfdetr.builder import _resolve_variant

        for sentinel in ("", "none", "null"):
            ckpt = {"args": {"pretrain_weights": sentinel}}
            assert _resolve_variant("/w/rf-detr-seg-large.pth", ckpt) == "RFDETRSegLarge"

    def test_unresolvable_checkpoint_names_the_way_out(self) -> None:
        from pictograph.inference._rfdetr.builder import _resolve_variant

        with pytest.raises(ValueError, match="ONNX export"):
            _resolve_variant("/w/mystery.pth", {"args": {"pretrain_weights": "mystery"}})

    def test_plus_only_variant_is_rejected_explicitly(self) -> None:
        """Detection XLarge/2XLarge live in `rfdetr_plus`, so say so rather than
        silently resolving to Large and loading a mismatched head."""
        from pictograph.inference._rfdetr.builder import _resolve_variant

        ckpt = {"model_name": "RFDETR2XLarge", "args": {"pretrain_weights": "rf-detr-2xlarge.pth"}}
        with pytest.raises(ValueError, match="not part of the open-source"):
            _resolve_variant("/w/x.pth", ckpt)


# Needs the vendored architecture to actually import, which needs torch - not in
# CI's `[dev,cli,agents,cache,telemetry]` install, so these ERRORed rather than
# skipping and left this module RED on main.
@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs the [inference] extra")
class TestSchemaInferenceContainers:
    """`_bare_weights` must find the tensors in either container shape."""

    def test_reads_the_plain_model_key(self) -> None:
        from pictograph.inference._rfdetr.builder import _bare_weights

        assert _bare_weights({"model": {"a": 1}}) == {"a": 1}

    def test_reads_a_lightning_state_dict(self) -> None:
        from pictograph.inference._rfdetr.builder import _bare_weights

        assert _bare_weights({"state_dict": {"model.a": 1, "optimizer.b": 2}}) == {"a": 1}

    def test_strips_the_torch_compile_prefix(self) -> None:
        from pictograph.inference._rfdetr.builder import _bare_weights

        assert _bare_weights({"state_dict": {"model._orig_mod.a": 1}}) == {"a": 1}


class TestAttribution:
    def test_licence_and_notice_are_present(self) -> None:
        assert "Apache License" in (_VENDORED / "LICENSE").read_text()
        notice = (_VENDORED / "NOTICE").read_text()
        assert "Copyright (c) 2025 Roboflow" in notice
        assert "NOTICE OF MODIFICATION" in notice

    def test_they_are_packaged_into_the_wheel(self) -> None:
        """Apache-2.0 requires the licence + notices to travel with the code, and
        the DINOv2 JSON configs are READ AT BUILD TIME - a wheel missing them
        cannot rebuild a model at all."""
        artifacts = tomllib.loads((_SDK_ROOT / "pyproject.toml").read_text())["tool"]["hatch"][
            "build"
        ]["targets"]["wheel"]["artifacts"]
        for required in (
            "src/pictograph/inference/_rfdetr/LICENSE",
            "src/pictograph/inference/_rfdetr/NOTICE",
            "src/pictograph/inference/_rfdetr/models/backbone/dinov2_configs/*.json",
        ):
            assert required in artifacts, f"{required} is not shipped in the wheel"

    def test_dinov2_configs_exist(self) -> None:
        configs = sorted((_VENDORED / "models" / "backbone" / "dinov2_configs").glob("*.json"))
        assert configs, "the DINOv2 backbone configs are missing from the vendored tree"

    def test_upstream_derived_files_carry_the_modification_stamp(self) -> None:
        """Apache-2.0 §4(b): a modified file must say it was modified."""
        unstamped = [
            str(path.relative_to(_VENDORED))
            for group in ("models", "utilities")
            for path in (_VENDORED / group).rglob("*.py")
            if path.stat().st_size and "VENDORED INTO THE PICTOGRAPH SDK" not in path.read_text()
        ]
        assert not unstamped, f"vendored files without the modification notice: {unstamped}"


# Needs torch, like every class above that builds or loads anything.
@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs the [inference] extra")
class TestBuilderFollowsTheVendoredRelease:
    """`builder.py` transcribes rfdetr's own resolution and predict rules; these pin the
    ones the 1.8.3 -> 1.11.2 re-vendor added or corrected."""

    def test_a_plus_only_weights_name_is_refused_too(self) -> None:
        """Upstream refuses a detection XLarge / 2XLarge by WEIGHTS NAME as well as by
        `model_name`. Without that, "rf-detr-xlarge.pth" substring-matches "large" and
        rebuilds as RFDETRLarge, loading a mismatched head."""
        from pictograph.inference._rfdetr.builder import _resolve_variant

        for name in ("rf-detr-xlarge.pth", "rf-detr-2xlarge.pth", "rf-detr-xxlarge.pth"):
            with pytest.raises(ValueError, match="not part of the open-source"):
                _resolve_variant("/w/x.pth", {"args": {"pretrain_weights": name}})

    def test_seg_and_keypoint_xlarge_names_still_resolve(self) -> None:
        """The guard is detection-only: the segmentation XLarge / 2XLarge ARE open source."""
        from pictograph.inference._rfdetr.builder import _resolve_variant

        assert _resolve_variant(
            "/w/x.pth", {"args": {"pretrain_weights": "rf-detr-seg-xlarge.pth"}}
        ) == ("RFDETRSegXLarge")
        assert _resolve_variant(
            "/w/x.pth", {"args": {"pretrain_weights": "rf-detr-seg-2xlarge.pth"}}
        ) == ("RFDETRSeg2XLarge")

    def test_large_deprecated_resolves_by_name_only(self) -> None:
        from pictograph.inference._rfdetr.builder import _resolve_variant

        by_name = {
            "model_name": "RFDETRLargeDeprecated",
            "args": {"pretrain_weights": "rf-detr-large.pth"},
        }
        assert _resolve_variant("/w/x.pth", by_name) == "RFDETRLargeDeprecated"
        by_weights = {"args": {"pretrain_weights": "rf-detr-large.pth"}}
        assert _resolve_variant("/w/x.pth", by_weights) == "RFDETRLarge"

    def test_the_trained_architecture_is_restored_but_never_the_device(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`model_config` from the checkpoint wins over class defaults, except
        `pretrain_weights` (the file is the weights) and `device`, which names the
        TRAINING host - a GPU-trained checkpoint must load on a CPU-only machine."""
        import torch

        from pictograph.inference._rfdetr import builder
        from pictograph.inference._rfdetr.config import RFDETRNanoConfig

        captured: list[Any] = []

        class _Context:
            def __init__(self, config: Any) -> None:
                self.model = torch.nn.Identity()
                self.model_config = config
                self.class_names = None

        def _fake_build(config: Any) -> _Context:
            captured.append(config)
            return _Context(config)

        monkeypatch.setattr(builder, "_build_context", _fake_build)
        ckpt = tmp_path / "checkpoint_best_total.pth"
        torch.save(
            {
                "model": {"class_embed.weight": torch.zeros(4, 8)},
                "model_name": "RFDETRNano",
                "args": {"num_classes": 3},
                "model_config": {"resolution": 448, "num_select": 50, "device": "cuda"},
            },
            ckpt,
        )
        builder.from_checkpoint(str(ckpt))
        (config,) = captured
        assert config.resolution == 448
        assert config.num_select == 50
        assert config.num_classes == 3
        assert config.device == RFDETRNanoConfig().device, "the training host's device leaked in"
        assert config.pretrain_weights.endswith("checkpoint_best_total.pth")

    def test_predict_resizes_without_antialias_and_hands_on_the_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """rfdetr 1.10+ resizes with ``antialias=False`` (its training resize is cv2
        ``INTER_LINEAR``, upstream issue #1203) and passes the threshold to
        `PostProcess`, which then upsamples only the masks it keeps."""
        import torch
        from torchvision.transforms import functional as tv_functional

        from pictograph.inference._rfdetr.builder import RFDETRModel, _ModelContext

        resize_calls: list[dict[str, Any]] = []
        real_resize = tv_functional.resize

        def _recording_resize(*args: Any, **kwargs: Any) -> Any:
            resize_calls.append(kwargs)
            return real_resize(*args, **kwargs)

        monkeypatch.setattr(tv_functional, "resize", _recording_resize)

        class _Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x: Any) -> dict[str, Any]:  # noqa: ARG002 - fixed outputs
                return {"pred_boxes": torch.zeros(1, 2, 4), "pred_logits": torch.zeros(1, 2, 3)}

        post_calls: list[dict[str, Any]] = []

        class _Post:
            def __call__(self, outputs: Any, **kwargs: Any) -> list[dict[str, Any]]:  # noqa: ARG002
                post_calls.append(kwargs)
                return [
                    {
                        "scores": torch.tensor([0.9, 0.2]),
                        "labels": torch.tensor([1, 2]),
                        "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0], [1.0, 1.0, 2.0, 2.0]]),
                    }
                ]

        context = _ModelContext(
            model=_Module(),
            postprocess=_Post(),  # type: ignore[arg-type]
            device=torch.device("cpu"),
            resolution=32,
            args=None,
        )
        result = RFDETRModel(context, variant="RFDETRNano").predict(
            torch.rand(3, 48, 64), threshold=0.5
        )

        assert resize_calls and all(call.get("antialias") is False for call in resize_calls)
        assert post_calls[0]["score_threshold"] == 0.5
        assert len(result) == 1 and result.class_id.tolist() == [1]


_RFDETR_UPSTREAM = os.environ.get("PICTOGRAPH_RFDETR_UPSTREAM")

# Every variant our training pipelines build. RFDETRBase / RFDETRLargeDeprecated are left
# out: their patch-14 DINOv2 backbone would fetch hub weights to initialise, and this test
# downloads nothing.
_PARITY_VARIANTS = (
    "RFDETRNano",
    "RFDETRSmall",
    "RFDETRMedium",
    "RFDETRLarge",
    "RFDETRSegNano",
    "RFDETRSegSmall",
    "RFDETRSegMedium",
    "RFDETRSegLarge",
    "RFDETRSegXLarge",
    "RFDETRSeg2XLarge",
    "RFDETRKeypointPreview",
)


def _assert_identical(left: Any, right: Any, where: str) -> int:
    """Recursively require `torch.equal` on every tensor; returns how many were compared."""
    import torch

    if isinstance(left, dict):
        assert sorted(left) == sorted(right), f"{where}: keys {sorted(left)} != {sorted(right)}"
        return sum(_assert_identical(left[k], right[k], f"{where}.{k}") for k in left)
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right), f"{where}: length {len(left)} != {len(right)}"
        return sum(
            _assert_identical(a, b, f"{where}[{i}]")
            for i, (a, b) in enumerate(zip(left, right, strict=True))
        )
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), f"{where}: max |delta| {(left - right).abs().max().item()}"
        return 1
    assert left == right, f"{where}: {left!r} != {right!r}"
    return 0


@pytest.mark.skipif(_RFDETR_UPSTREAM is None, reason="PICTOGRAPH_RFDETR_UPSTREAM not set")
@pytest.mark.parametrize("variant", _PARITY_VARIANTS)
def test_rebuild_is_bit_identical_to_upstream(variant: str) -> None:
    """Same state dict, same input: the vendored tree must compute EXACTLY what the
    release it was vendored from computes (`NOTICE`: no edit here changes numerics)."""
    torch = pytest.importorskip("torch")
    assert _RFDETR_UPSTREAM is not None
    if _RFDETR_UPSTREAM not in sys.path:
        sys.path.insert(0, _RFDETR_UPSTREAM)
    try:
        upstream_version = importlib.metadata.version("rfdetr")
    except importlib.metadata.PackageNotFoundError:
        upstream_version = None
    if upstream_version is not None:
        assert upstream_version == _vendored_version(), (
            f"PICTOGRAPH_RFDETR_UPSTREAM holds rfdetr {upstream_version}; this tree is "
            f"{_vendored_version()}"
        )

    # Upstream's import emits its own warnings (the `torch.jit.script` deprecation on
    # torch >= 2.12 that the vendored tree removed, pyDeprecate notices); they are not
    # this test's subject, and the suite runs with warnings as errors.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from rfdetr import config as up_config  # type: ignore[import-not-found]
        from rfdetr._namespace import (
            _namespace_from_configs as up_namespace,  # type: ignore[import-not-found]
        )
        from rfdetr.models.lwdetr import build_model as up_build  # type: ignore[import-not-found]

    from pictograph.inference._rfdetr import config as v_config
    from pictograph.inference._rfdetr._namespace import _namespace_from_configs as v_namespace
    from pictograph.inference._rfdetr.models.lwdetr import build_model as v_build

    def build(config_module: Any, namespace: Any, build_model: Any) -> Any:
        # A non-None pretrain_weights keeps the backbone from reaching for hub weights;
        # build_model never opens the file.
        cfg = getattr(config_module, f"{variant}Config")(
            pretrain_weights="unused.pth", device="cpu"
        )
        train = config_module.TrainConfig(dataset_dir=None, output_dir="output")
        torch.manual_seed(0)
        return build_model(namespace(cfg, train)).eval(), int(cfg.resolution)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        upstream, resolution = build(up_config, up_namespace, up_build)
    vendored, vendored_resolution = build(v_config, v_namespace, v_build)
    assert vendored_resolution == resolution

    up_state = upstream.state_dict()
    assert {k: tuple(v.shape) for k, v in up_state.items()} == {
        k: tuple(v.shape) for k, v in vendored.state_dict().items()
    }
    vendored.load_state_dict(up_state, strict=True)

    image = torch.rand(1, 3, resolution, resolution, generator=torch.Generator().manual_seed(1234))
    with torch.no_grad():
        compared = _assert_identical(upstream(image), vendored(image), "eval")
        upstream.export()
        vendored.export()
        compared += _assert_identical(upstream(image), vendored(image), "export")
    assert compared >= 4, "nothing was compared"
