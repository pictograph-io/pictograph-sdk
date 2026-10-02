"""The TensorRT loader, and the refusal that keeps it from being a support ticket.

A TensorRT plan is valid for exactly one GPU architecture x TensorRT version x
precision. Copying ``model.engine`` to another machine and running it produces a raw
deserialization crash - which a user experiences as "Pictograph gave me a broken
file". The whole point of the loader is that this never happens: it reads what the
artifact was built for, compares it against the local device, and refuses with a
sentence that names both.

**Everything here runs without a GPU and without TensorRT installed**, deliberately.
The compatibility check happens BEFORE any TensorRT call, which is what makes the
refusal testable on any machine - and it is the single most important behaviour in
the module, because it is the one a user hits when they have made a mistake.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from pictograph.inference._tensorrt import (
    GPU_SM,
    EngineTarget,
    _sidecar_target,
    check_engine_compatibility,
    detect_local_target,
    engine_mismatch_message,
    parse_engine_filename,
)
from pictograph.inference.runtime import RUNTIMES, runtime_for_weights, staleness_blocks_load
from tests.conftest import ENV_ARTIFACT_POLICY_SOURCE, companion_skip_reason, companion_source

# The contract's own example, used throughout so the tests and the spec agree.
_ENGINE_NAME = "sm75-trt10.13.3.9-fp16.engine"
_BUILT = EngineTarget(sm="sm80", toolchain_version="trt-10.13.3.9", precision="fp32")


class TestFilenameIsTheIdentity:
    """The basename encodes the binding - ``{sm}-trt{version}-{precision}.engine``."""

    def test_parses_the_contract_shape(self) -> None:
        target = parse_engine_filename(_ENGINE_NAME)
        assert target == EngineTarget("sm75", "trt-10.13.3.9", "fp16")
        assert target is not None
        assert target.trt_version == "10.13.3.9"

    @pytest.mark.parametrize("precision", ["fp32", "fp16"])
    def test_both_precisions_round_trip(self, precision: str) -> None:
        target = parse_engine_filename(f"sm90-trt10.13.3.9-{precision}.engine")
        assert target is not None
        assert target.precision == precision

    def test_dot_plan_parses_too(self) -> None:
        """``.plan`` is TensorRT's other conventional extension for the SAME
        serialized plan, and ``runtime_for_weights`` already routes it here - so a
        contract-shaped ``.plan`` must keep its binding rather than losing the check
        over a file extension."""
        assert parse_engine_filename("sm75-trt10.13.3.9-fp16.plan") == EngineTarget(
            "sm75", "trt-10.13.3.9", "fp16"
        )

    @pytest.mark.parametrize(
        "name",
        [
            "model.engine",  # the name a user gets after "save as"
            "sm75-fp16.engine",  # no toolchain
            "trt10.13.3.9-fp16.engine",  # no arch
            "sm75-trt10.13.3.9-int8.engine",  # a precision we do not build
            "sm75-tensorrt10.13.3.9-fp16.engine",  # not the contract's `trt` prefix
        ],
    )
    def test_returns_none_rather_than_guessing(self, name: str) -> None:
        """A wrong guess about the binding is worse than admitting we do not know."""
        assert parse_engine_filename(name) is None

    def test_the_model_named_form_parses(self) -> None:
        """The file is named after its MODEL and the binding is a SUFFIX.

        ``fixture-rfdetr_segmentation-v2-sm75-trt10.13.3.9.engine``. The old
        target-named form said nothing about WHICH model it was, so two models'
        engines were byte-different files with the same name.
        """
        target = parse_engine_filename("fixture-rfdetr_segmentation-v2-sm75-trt10.13.3.9.engine")
        assert target == EngineTarget("sm75", "trt-10.13.3.9", "fp32")

    def test_fp32_is_implied_by_the_absence_of_a_precision_token(self) -> None:
        """fp32 is the default a reader assumes, so spelling it out added
        length to every filename to say nothing. Absent MUST mean fp32, not
        unknown: the loader refuses on a mismatch, so a wrong default is a
        refused-but-valid engine."""
        target = parse_engine_filename("my-model-sm80-trt10.13.3.9.engine")
        assert target is not None
        assert target.precision == "fp32"

    def test_the_fp16_token_sits_before_the_binding(self) -> None:
        target = parse_engine_filename("my-model-fp16-sm86-trt10.13.3.9.engine")
        assert target == EngineTarget("sm86", "trt-10.13.3.9", "fp16")

    def test_a_model_name_containing_a_dash_or_digits_still_parses(self) -> None:
        """Models can be called anything; the binding is the LAST `sm…-trt…`
        run in the stem, which is what keeps a name like `b235-profile-5k-aug`
        from eating it."""
        target = parse_engine_filename("b235-profile-5k-aug-fp16-sm90-trt10.13.3.9.engine")
        assert target == EngineTarget("sm90", "trt-10.13.3.9", "fp16")

    def test_the_legacy_target_named_form_still_parses(self) -> None:
        """The rename covered NEW artifacts only - nothing in storage was migrated, so
        an engine built before 2026-07-31 must keep its binding."""
        assert parse_engine_filename("sm75-trt10.13.3.9-fp16.engine") == EngineTarget(
            "sm75", "trt-10.13.3.9", "fp16"
        )
        assert parse_engine_filename("sm80-trt10.13.3.9-fp32.engine") == EngineTarget(
            "sm80", "trt-10.13.3.9", "fp32"
        )

    def test_sidecar_carries_the_binding_for_a_renamed_file(self, tmp_path: Path) -> None:
        """A user who renames the engine keeps the check, via the manifest row."""
        engine = tmp_path / "my-model.engine"
        engine.write_bytes(b"not-a-real-plan")
        engine.with_suffix(".json").write_text(
            json.dumps(
                {
                    "target_key": "sm86",
                    "toolchain_version": "trt-10.13.3.9",
                    "precision": "fp16",
                    "runtime": "tensorrt",
                }
            )
        )
        assert _sidecar_target(engine) == EngineTarget("sm86", "trt-10.13.3.9", "fp16")

    def test_absent_or_unusable_sidecar_is_not_an_error(self, tmp_path: Path) -> None:
        engine = tmp_path / "my-model.engine"
        engine.write_bytes(b"x")
        assert _sidecar_target(engine) is None
        engine.with_suffix(".json").write_text("{ not json")
        assert _sidecar_target(engine) is None
        engine.with_suffix(".json").write_text(json.dumps({"unrelated": True}))
        assert _sidecar_target(engine) is None


class TestTheRefusal:
    """The behaviour that decides whether a mismatch is a clear message or a crash."""

    def test_wrong_architecture_is_refused_with_both_targets_named(self) -> None:
        detected = EngineTarget(sm="sm75", toolchain_version="trt-10.13.3.9")
        with pytest.raises(RuntimeError) as excinfo:
            check_engine_compatibility(_BUILT, detected)
        message = str(excinfo.value)
        # The two facts a user needs in order to act.
        assert "sm80" in message, "must name what the engine was BUILT for"
        assert "sm75" in message, "must name what this DEVICE is"
        assert "rebuild the engine for your device" in message

    def test_wrong_tensorrt_minor_is_refused(self) -> None:
        """A TRT minor bump invalidates every previously serialized plan."""
        detected = EngineTarget(sm="sm80", toolchain_version="trt-10.14.0.1")
        with pytest.raises(RuntimeError) as excinfo:
            check_engine_compatibility(_BUILT, detected)
        assert "trt-10.13.3.9" in str(excinfo.value)
        assert "trt-10.14.0.1" in str(excinfo.value)

    def test_a_matching_target_loads(self) -> None:
        check_engine_compatibility(_BUILT, EngineTarget("sm80", "trt-10.13.3.9"))

    def test_patch_difference_warns_but_does_not_refuse(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Plans DO load across patch builds. Refusing would block a working engine."""
        with caplog.at_level(logging.WARNING, logger="pictograph.inference"):
            check_engine_compatibility(_BUILT, EngineTarget("sm80", "trt-10.13.3.10"))
        assert "should deserialize" in caplog.text

    def test_unlabelled_engine_warns_and_defers_to_tensorrt(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No metadata is not the same as a mismatch - refusing would block a
        legitimately-correct engine on the basis of a missing filename convention."""
        with caplog.at_level(logging.WARNING, logger="pictograph.inference"):
            check_engine_compatibility(None, EngineTarget("sm80", "trt-10.13.3.9"))
        assert "no target metadata" in caplog.text

    def test_undetectable_device_still_reports_what_the_engine_needs(self) -> None:
        """Half the actionable information survives even with no GPU to inspect."""
        message = engine_mismatch_message(
            built_target="sm80", built_toolchain="trt-10.13.3.9", detected_target=None
        )
        assert "sm80" in message
        assert "unknown" in message

    def test_an_unknown_device_does_not_manufacture_a_mismatch(self) -> None:
        """On a machine with no detectable GPU there is nothing to contradict, so the
        check must stay silent and let TensorRT speak - inventing a refusal here would
        block every load in a container where the probe simply did not work."""
        check_engine_compatibility(_BUILT, EngineTarget("unknown", "unknown"))


class TestLoaderIntegration:
    """The loader-level guarantees, all reachable without TensorRT installed."""

    def test_suffix_dispatch_routes_engine_and_plan_to_tensorrt(self) -> None:
        assert runtime_for_weights(Path("a.engine")) == "tensorrt"
        assert runtime_for_weights(Path("a.plan")) == "tensorrt"
        assert runtime_for_weights(Path("a.pte")) == "executorch"
        assert runtime_for_weights(Path("a.onnx")) == "onnxruntime"

    def test_a_checkpoint_resolves_to_the_pytorch_runtime_not_an_engine(self) -> None:
        """A ``.pth`` is a recognised format - it is simply not one `load_model`
        can execute, and THAT refusal is where the next step is named."""
        assert runtime_for_weights(Path("checkpoint_best.pth")) == "pytorch"

    def test_the_mismatch_is_raised_before_tensorrt_is_even_needed(self, tmp_path: Path) -> None:
        """The check runs ahead of any TensorRT call - which is exactly why a user
        gets an explanation instead of a deserialization crash, and why this is
        testable on a machine with no GPU."""
        from pictograph.inference._tensorrt import build_tensorrt_engine

        engine = tmp_path / "sm80-trt10.13.3.9-fp32.engine"
        engine.write_bytes(b"definitely-not-a-plan")

        def _fake_local() -> EngineTarget:
            return EngineTarget("sm75", "trt-10.13.3.9")

        import pictograph.inference._tensorrt as trt_module

        original = trt_module.detect_local_target
        trt_module.detect_local_target = _fake_local
        try:
            with pytest.raises(RuntimeError, match="sm80"):
                build_tensorrt_engine(
                    weights=engine,
                    model_type="classification",
                    architecture="resnet18",
                    classes=["a"],
                    input_shape=(224, 224),
                    confidence=0.5,
                )
        finally:
            trt_module.detect_local_target = original

    def test_device_cpu_is_a_contradiction_and_says_so(self, tmp_path: Path) -> None:
        from pictograph.inference._tensorrt import build_tensorrt_engine

        engine = tmp_path / "sm75-trt10.13.3.9-fp32.engine"
        engine.write_bytes(b"x")
        with pytest.raises(ValueError, match="no cpu form at all"):
            build_tensorrt_engine(
                weights=engine,
                model_type="classification",
                architecture="resnet18",
                classes=["a"],
                input_shape=(224, 224),
                confidence=0.5,
                device="cpu",
            )

    def test_the_cpu_refusal_points_at_the_onnx_artifact(self) -> None:
        from pictograph.inference._tensorrt import check_tensorrt_device

        with pytest.raises(ValueError, match="format='onnx'"):
            check_tensorrt_device("cpu")
        with pytest.raises(ValueError, match="format='onnx'"):
            check_tensorrt_device("mps")

    def test_cuda_and_auto_pass_and_an_index_survives(self) -> None:
        """`cuda:1` is a real choice on a multi-GPU box - the plan, its context, its
        stream and every buffer must land on the GPU that was named."""
        from pictograph.inference._tensorrt import check_tensorrt_device

        assert check_tensorrt_device("auto") is None
        assert check_tensorrt_device("cuda") is None
        assert check_tensorrt_device("cuda:2") == 2

    def test_install_hint_names_the_exact_command_and_the_platform_limit(self) -> None:
        pytest.importorskip  # noqa: B018 - documents that this needs nothing installed
        from pictograph.inference import _tensorrt

        if _tensorrt._local_toolchain() is not None:  # pragma: no cover - NVIDIA machines
            pytest.skip("tensorrt is installed here, so the ImportError cannot be raised")
        with pytest.raises(ImportError) as excinfo:
            _tensorrt._require_tensorrt()
        message = str(excinfo.value)
        assert 'pip install "pictograph[inference,tensorrt]"' in message
        assert "NVIDIA-only" in message
        # No naked third-party install line - the only command we print is ours.
        assert "pip install tensorrt" not in message

    def test_detect_local_target_never_raises(self) -> None:
        """It runs on every platform, including ones with no GPU and no nvidia-smi."""
        target = detect_local_target()
        assert isinstance(target.sm, str)
        assert isinstance(target.toolchain_version, str)


class TestSharedVocabulary:
    """The SDK's copies of the contract's constants must match the backend's."""

    def test_gpu_sm_map_is_the_priced_set(self) -> None:
        assert GPU_SM == {
            "t4": "sm75",
            "l4": "sm89",
            "a10g": "sm86",
            "a100": "sm80",
            "h100": "sm90",
        }

    def test_only_tensorrt_staleness_blocks_a_load(self) -> None:
        """Flattening the three would either withdraw two working artifact classes
        or serve one unloadable plan."""
        assert staleness_blocks_load("tensorrt") is True
        assert staleness_blocks_load("executorch") is False
        assert staleness_blocks_load("onnxruntime") is False
        assert staleness_blocks_load("pytorch") is False

    def test_the_runtime_vocabulary_is_the_owner_settled_order(self) -> None:
        """The contract settles both the set and the order; the UI renders this row."""
        assert RUNTIMES == ("pytorch", "executorch", "onnxruntime", "tensorrt")


class TestTheDoNotUndoSurvivesThisRuntimeExisting:
    """Shipping a TensorRT runtime does NOT license re-adding the ORT provider.

    This is the misreading the whole module invites: "we support TensorRT now, so
    ``device='cuda'`` on an ``.onnx`` should use it." It must not, and the two facts
    are unrelated - so they are asserted together, here, where someone about to make
    that change will be reading.

    ORT's ``TensorrtExecutionProvider`` JIT-builds an engine on FIRST INFERENCE
    (8-65 s, measured) and is listed by ``get_available_providers()`` whether or not
    ``libnvinfer`` is loadable - and when it is listed-but-unloadable ORT discards
    the WHOLE provider list and silently falls back to CPU. This module is the
    ahead-of-time alternative to exactly that: an ``.engine`` is built once, on
    purpose, by a user who chose a GPU. Explicit stays explicit.

    ``tests/unit/test_inference_runtime.py::TestCudaAndTensorrt`` holds the ladder
    assertions themselves; this pins the RELATIONSHIP.
    """

    def test_the_onnx_ladder_still_excludes_tensorrt_for_every_device(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pictograph.inference.runtime as runtime_module
        from pictograph.inference.runtime import resolve_providers

        monkeypatch.setattr(
            runtime_module,
            "_available",
            lambda: {
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            },
        )
        for device in ("auto", "cuda", "cpu"):
            resolved = resolve_providers(device, architecture="yolox")
            names = [p if isinstance(p, str) else p[0] for p in resolved]
            assert "TensorrtExecutionProvider" not in names, (
                f"device={device!r} put TensorRT back in the ONNX ladder. The AOT "
                f".engine runtime (device='cuda' + format='tensorrt_engine') is the "
                f"supported TensorRT path (see the DO-NOT-UNDO in runtime.py)."
            )

    def test_the_measured_rationale_is_still_recorded_in_the_source(self) -> None:
        """The comment IS the artifact - it carries the measurements that justify the
        decision, and deleting it is how the decision gets quietly reversed later."""
        source = Path(runtime_source()).read_text(encoding="utf-8")
        assert "TensorRT is deliberately NOT in this ladder" in source
        assert "DISCARDS THE WHOLE PROVIDER LIST" in source


def runtime_source() -> str:
    import pictograph.inference.runtime as runtime_module

    assert runtime_module.__file__ is not None
    return runtime_module.__file__


# ── LOCKSTEP gate: the SDK and the service must say the SAME sentence ───────────

#: The service-side ``model_artifacts`` module. Not part of this repository, so
#: the two comparisons below are opt-in (see ``tests/conftest.py``).
_ARTIFACT_POLICY = companion_source(ENV_ARTIFACT_POLICY_SOURCE)


@pytest.mark.skipif(
    not _ARTIFACT_POLICY.exists(),
    reason=companion_skip_reason(ENV_ARTIFACT_POLICY_SOURCE),
)
def test_mismatch_message_is_byte_identical_to_the_backends() -> None:
    """The refusal wording is duplicated on purpose and must not drift.

    ``engine_mismatch_message`` exists in both the service's ``model_artifacts``
    module and the SDK, because the SDK ships as its own wheel and cannot import the
    service. The contract requires all three surfaces - API, SDK, UI - to say ONE
    sentence, so a user who hits this in a REST 409, in a traceback and in the app's
    model dialog reads the same thing rather than three paraphrases. This test is
    what makes "LOCKSTEP" enforceable instead of aspirational.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_service_model_artifacts", _ARTIFACT_POLICY)
    assert spec is not None and spec.loader is not None
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)

    cases = [
        {"built_target": "sm80", "built_toolchain": "trt-10.13.3.9"},
        {
            "built_target": "sm80",
            "built_toolchain": "trt-10.13.3.9",
            "detected_target": "sm75",
        },
        {
            "built_target": "sm90",
            "built_toolchain": "trt-10.13.3.9",
            "detected_target": "sm86",
            "detected_toolchain": "trt-10.14.0.1",
        },
    ]
    for case in cases:
        assert engine_mismatch_message(**case) == backend.engine_mismatch_message(**case), (
            f"SDK and service engine_mismatch_message disagree for {case}. They are "
            f"LOCKSTEP - change both in one commit."
        )


@pytest.mark.skipif(
    not _ARTIFACT_POLICY.exists(),
    reason=companion_skip_reason(ENV_ARTIFACT_POLICY_SOURCE),
)
def test_shared_constants_match_the_backend() -> None:
    """The GPU→SM map and the staleness rule are the service's; mirroring them wrong
    means the SDK refuses an engine the API happily served, or vice versa."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_service_model_artifacts2", _ARTIFACT_POLICY)
    assert spec is not None and spec.loader is not None
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)

    assert GPU_SM == backend.GPU_SM
    for runtime in ("tensorrt", "executorch", "onnxruntime", "pytorch"):
        assert staleness_blocks_load(runtime) == backend.staleness_blocks_load(runtime)
