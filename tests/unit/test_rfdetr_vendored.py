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
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

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
