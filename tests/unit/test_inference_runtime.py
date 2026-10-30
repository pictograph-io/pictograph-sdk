"""Unit tests for `pictograph.inference.runtime` -- the provider/device ladder.

`resolve_providers` builds the ONNX execution-provider list; `resolve_torch_device`
resolves the `torch` device string. Both take the ONE `Device` vocabulary
(`"auto"`/`"cpu"`/`"cuda"`/`"cuda:N"`/`"mps"`) and map it onto their own runtime's
mechanism, and both report what actually ran rather than what was requested
(`device_label`, `warn_on_fallback`).

The rule these tests exist to pin: **`auto` may degrade and warn, a NAMED device
raises.** Silently handing back CPU when CUDA was asked for is the failure mode the
whole argument was collapsed to prevent.

These tests monkeypatch `runtime._available()` directly (bypassing the real
`onnxruntime.get_available_providers()` call) and `platform.system()`, so the
whole ladder is deterministic on any OS/hardware and does not need a real GPU
-- or even a real onnxruntime install, since `_available` is never actually
called once patched. `resolve_torch_device` / `empty_device_cache` DO need a
real `torch` (they call `torch.cuda.is_available()` etc.), so those
`importorskip`.
"""

from __future__ import annotations

import logging

import pytest

from pictograph.inference import runtime


def _set_available(monkeypatch: pytest.MonkeyPatch, providers: set[str]) -> None:
    monkeypatch.setattr(runtime, "_available", lambda: set(providers))


class TestNormalizeDevice:
    """The ONE parser, so every runtime agrees on what was asked before deciding."""

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("auto", ("auto", None)),
            ("cpu", ("cpu", None)),
            ("CPU", ("cpu", None)),
            (" cuda ", ("cuda", None)),
            ("cuda:0", ("cuda", 0)),
            ("cuda:3", ("cuda", 3)),
            ("mps", ("mps", None)),
            ("coreml", ("mps", None)),  # same silicon, ORT's name for it
        ],
    )
    def test_parses_and_canonicalizes(self, given: str, expected: tuple[str, int | None]) -> None:
        assert runtime.normalize_device(given) == expected

    @pytest.mark.parametrize("bad", ["gpu", "CUDA_0", "mps:1", "cpu:0", "nvidia", ""])
    def test_an_unknown_device_raises_naming_the_vocabulary(self, bad: str) -> None:
        with pytest.raises(ValueError, match="not a device this SDK knows"):
            runtime.normalize_device(bad)

    @pytest.mark.parametrize("removed", ["max", "off"])
    def test_the_removed_accelerate_values_get_a_migration_hint(self, removed: str) -> None:
        """`accelerate="off"` was the documented way to pin CPU; a user carrying that
        habit forward must be told the new spelling, not just that it is invalid."""
        with pytest.raises(ValueError, match="no 'accelerate' argument any more"):
            runtime.normalize_device(removed)

    def test_only_cuda_takes_an_index(self) -> None:
        with pytest.raises(ValueError, match="Only 'cuda' takes an index"):
            runtime.normalize_device("mps:0")

    def test_is_explicit_is_the_one_predicate_behind_the_no_fallback_rule(self) -> None:
        assert not runtime.is_explicit("auto")
        for named in ("cpu", "cuda", "cuda:1", "mps", "coreml"):
            assert runtime.is_explicit(named), named


class TestResolveProvidersMeasurementHatch:
    """`requested=` is the private hatch `benchmarks/inference_bench.py` pins a
    provider config with. It is deliberately NOT on either public loader."""

    def test_explicit_requested_passes_through_untouched(self) -> None:
        requested = [("CUDAExecutionProvider", {"foo": "bar"}), "CPUExecutionProvider"]
        resolved = runtime.resolve_providers(requested=requested)
        assert resolved == requested
        assert resolved is not requested  # a fresh list, not the same object

    def test_explicit_requested_skips_available_probe_entirely(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom() -> set[str]:
            raise AssertionError("_available() must not be called when requested is explicit")

        monkeypatch.setattr(runtime, "_available", _boom)
        assert runtime.resolve_providers(requested=["MyProvider"]) == ["MyProvider"]


class TestResolveProvidersCpu:
    def test_cpu_returns_cpu_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_available(monkeypatch, {"CUDAExecutionProvider", "CoreMLExecutionProvider"})
        assert runtime.resolve_providers("cpu") == ["CPUExecutionProvider"]

    def test_cpu_skips_available_probe_entirely(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The instant-load property `accelerate="off"` was chosen for in CI: naming
        the CPU must not probe the accelerator stack at all."""

        def _boom() -> set[str]:
            raise AssertionError("_available() must not be called under device='cpu'")

        monkeypatch.setattr(runtime, "_available", _boom)
        assert runtime.resolve_providers("cpu") == ["CPUExecutionProvider"]


class TestNamedDeviceRaisesRatherThanFallingBack:
    """The load-bearing rule. A named device ORT cannot reach is an ERROR."""

    def test_cuda_without_the_provider_raises_naming_what_is_there(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_available(monkeypatch, {"CPUExecutionProvider"})
        with pytest.raises(ValueError) as exc:
            runtime.resolve_providers("cuda")
        assert "CUDAExecutionProvider" in str(exc.value)
        assert "CPUExecutionProvider" in str(exc.value), "must name what IS available"
        assert "onnxruntime-gpu" in str(exc.value), "must say how to fix it"

    def test_auto_in_the_same_situation_quietly_uses_cpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The contrast that makes the rule meaningful - `auto` never raises for
        want of an accelerator, because it never promised one."""
        _set_available(monkeypatch, {"CPUExecutionProvider"})
        assert runtime.resolve_providers("auto") == ["CPUExecutionProvider"]

    def test_mps_off_darwin_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_available(monkeypatch, {"CoreMLExecutionProvider", "CPUExecutionProvider"})
        monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
        with pytest.raises(ValueError, match="Apple hardware"):
            runtime.resolve_providers("mps")

    def test_a_delegate_only_device_raises_pointing_at_the_pte(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_available(monkeypatch, {"CPUExecutionProvider"})
        with pytest.raises(ValueError, match="pytorch_engine"):
            runtime.resolve_providers("qnn")

    def test_cuda_index_becomes_ort_device_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`cuda:1` must actually reach ORT - without `device_id` a multi-GPU box
        silently lands on GPU 0 and the request is ignored."""
        _set_available(monkeypatch, {"CUDAExecutionProvider", "CPUExecutionProvider"})
        resolved = runtime.resolve_providers("cuda:1")
        assert resolved == [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC", "device_id": "1"}),
            "CPUExecutionProvider",
        ]

    def test_named_mps_pays_the_slow_build_auto_skips(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """This is `accelerate="max"`'s entire measured behaviour, preserved under
        the honest name: `auto` skips RF-DETR's 31.7s CoreML build, naming the
        device pays it. Nothing was removed - it moved."""
        _set_available(monkeypatch, {"CoreMLExecutionProvider", "CPUExecutionProvider"})
        monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
        assert runtime.resolve_providers("auto", architecture="rfdetr-base") == [
            "CPUExecutionProvider"
        ]
        named = runtime.resolve_providers("mps", architecture="rfdetr-base")
        assert any(isinstance(p, tuple) and p[0] == "CoreMLExecutionProvider" for p in named)


class TestCheckDeviceHonoured:
    """The second gate: ORT HAS the provider, but did the built session keep it?"""

    def test_a_dropped_named_provider_raises(self) -> None:
        with pytest.raises(RuntimeError) as exc:
            runtime.check_device_honoured("cuda", ["CPUExecutionProvider"])
        assert "did not keep" in str(exc.value)
        assert "driver" in str(exc.value)

    def test_a_kept_provider_is_silent(self) -> None:
        runtime.check_device_honoured("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"])

    def test_auto_and_cpu_never_raise(self) -> None:
        runtime.check_device_honoured("auto", ["CPUExecutionProvider"])
        runtime.check_device_honoured("cpu", ["CPUExecutionProvider"])

    def test_mps_is_satisfied_by_coreml(self) -> None:
        """The user asked for Apple silicon and got it; the mechanism's name
        differing from the hardware's must not read as a failure."""
        runtime.check_device_honoured("mps", ["CoreMLExecutionProvider", "CPUExecutionProvider"])


class TestCheckArtifactDevice:
    """The AOT gate - a fixed target is confirmed or refused, never ignored."""

    def test_auto_accepts_whatever_the_artifact_runs_on(self) -> None:
        runtime.check_artifact_device("auto", "coreml", artifact="x", remedy="y")

    def test_a_matching_request_is_confirmed(self) -> None:
        runtime.check_artifact_device("cpu", "cpu", artifact="x", remedy="y")
        runtime.check_artifact_device("mps", "coreml", artifact="x", remedy="y")

    def test_a_mismatch_names_both_sides_and_the_remedy(self) -> None:
        with pytest.raises(ValueError) as exc:
            runtime.check_artifact_device(
                "cpu",
                "coreml",
                artifact="'a.pte' is a program that",
                remedy="Load the XNNPACK one.",
            )
        assert "device='cpu'" in str(exc.value)
        assert "coreml" in str(exc.value)
        assert "Load the XNNPACK one." in str(exc.value)


class TestCpuAlwaysLast:
    """CPU is the guaranteed fallback: every ladder ends with it, exactly once,
    regardless of what else is available."""

    @pytest.mark.parametrize(
        "available",
        [
            {"CPUExecutionProvider"},
            {"CUDAExecutionProvider", "CPUExecutionProvider"},
            {"CoreMLExecutionProvider", "CPUExecutionProvider"},
            {
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CoreMLExecutionProvider",
                "DmlExecutionProvider",
                "ROCMExecutionProvider",
                "CPUExecutionProvider",
            },
        ],
        ids=["cpu-only", "cuda", "coreml", "everything"],
    )
    @pytest.mark.parametrize("system", ["Darwin", "Linux", "Windows"])
    def test_cpu_is_last_and_appears_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        available: set[str],
        system: str,
    ) -> None:
        _set_available(monkeypatch, available)
        monkeypatch.setattr(runtime.platform, "system", lambda: system)
        resolved = runtime.resolve_providers("auto")
        assert resolved[-1] == "CPUExecutionProvider"
        assert sum(1 for p in resolved if p == "CPUExecutionProvider") == 1


class TestCoreMlFormatTable:
    """Per-architecture CoreML model format, LOCKSTEP with the measured table
    in runtime.py's module docstring."""

    @pytest.mark.parametrize(
        ("architecture", "model_type", "expected_format"),
        [
            ("yolox-s", "object_detection", "MLProgram"),
            ("YOLOX-Nano", "object_detection", "MLProgram"),
            ("rfdetr-base", "instance_segmentation", "MLProgram"),
            ("", "object_detection", "MLProgram"),  # falls back to model_type
            ("", "instance_segmentation", "MLProgram"),
            ("", "keypoint_detection", "MLProgram"),
            ("resnet50", "classification", "NeuralNetwork"),
            ("", "classification", "NeuralNetwork"),
            ("unetplusplus", "semantic_segmentation", "NeuralNetwork"),
            ("", "semantic_segmentation", "NeuralNetwork"),
            ("something_unknown", "classification", "NeuralNetwork"),  # model_type fallback
        ],
    )
    def test_format_per_architecture(
        self,
        monkeypatch: pytest.MonkeyPatch,
        architecture: str,
        model_type: str,
        expected_format: str,
    ) -> None:
        _set_available(monkeypatch, {"CoreMLExecutionProvider", "CPUExecutionProvider"})
        monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
        # device="mps" so a slow-build architecture (rfdetr) is not skipped here
        # -- that skip is its own test below.
        resolved = runtime.resolve_providers(
            "mps", architecture=architecture, model_type=model_type
        )
        coreml = [p for p in resolved if isinstance(p, tuple) and p[0] == "CoreMLExecutionProvider"]
        assert len(coreml) == 1
        assert coreml[0][1]["ModelFormat"] == expected_format

    def test_coreml_only_selected_on_darwin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_available(monkeypatch, {"CoreMLExecutionProvider", "CPUExecutionProvider"})
        for system in ("Linux", "Windows"):
            monkeypatch.setattr(runtime.platform, "system", lambda system=system: system)
            assert runtime.resolve_providers("auto") == ["CPUExecutionProvider"]


class TestCoreMlSlowBuildGate:
    """RF-DETR's CoreML session build is measured at ~31s -- `auto` must not
    pay that on every load; naming `device="mps"` opts in explicitly.

    This is the behaviour `accelerate="max"` used to select. It is intact; only the
    way you ask for it changed, from a tradeoff word to the hardware's own name."""

    @pytest.mark.parametrize("architecture", ["rfdetr-base", "RF-DETR Medium", "rf-detr-nano"])
    def test_auto_skips_coreml_for_rfdetr(
        self, monkeypatch: pytest.MonkeyPatch, architecture: str
    ) -> None:
        _set_available(monkeypatch, {"CoreMLExecutionProvider", "CPUExecutionProvider"})
        monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
        resolved = runtime.resolve_providers("auto", architecture=architecture)
        assert resolved == ["CPUExecutionProvider"]

    @pytest.mark.parametrize("architecture", ["rfdetr-base", "RF-DETR Medium", "rf-detr-nano"])
    def test_naming_mps_includes_coreml_for_rfdetr(
        self, monkeypatch: pytest.MonkeyPatch, architecture: str
    ) -> None:
        _set_available(monkeypatch, {"CoreMLExecutionProvider", "CPUExecutionProvider"})
        monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
        resolved = runtime.resolve_providers("mps", architecture=architecture)
        assert any(isinstance(p, tuple) and p[0] == "CoreMLExecutionProvider" for p in resolved)

    def test_auto_does_not_skip_coreml_for_fast_architectures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_available(monkeypatch, {"CoreMLExecutionProvider", "CPUExecutionProvider"})
        monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
        resolved = runtime.resolve_providers("auto", architecture="yolox-s")
        assert any(isinstance(p, tuple) and p[0] == "CoreMLExecutionProvider" for p in resolved)


class TestCoreMlCacheDir:
    def test_cache_dir_is_created_and_wired_into_options(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        _set_available(monkeypatch, {"CoreMLExecutionProvider", "CPUExecutionProvider"})
        monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
        resolved = runtime.resolve_providers("auto", architecture="yolox-s", cache_dir=tmp_path)
        _provider, opts = next(p for p in resolved if isinstance(p, tuple))
        cache_dir = opts["ModelCacheDirectory"]
        assert cache_dir == str(tmp_path / runtime.COREML_CACHE_SUBDIR)
        assert (tmp_path / runtime.COREML_CACHE_SUBDIR).is_dir()

    def test_no_cache_dir_means_no_cache_option(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_available(monkeypatch, {"CoreMLExecutionProvider", "CPUExecutionProvider"})
        monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
        resolved = runtime.resolve_providers("auto", architecture="yolox-s")
        _provider, opts = next(p for p in resolved if isinstance(p, tuple))
        assert "ModelCacheDirectory" not in opts


class TestCudaAndTensorrt:
    def test_cuda_gets_the_heuristic_algo_search_option(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_available(monkeypatch, {"CUDAExecutionProvider", "CPUExecutionProvider"})
        resolved = runtime.resolve_providers("auto")
        assert resolved == [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
            "CPUExecutionProvider",
        ]

    def test_tensorrt_is_never_auto_selected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TensorRT must never enter the ONNX ladder on availability alone.

        MEASURED ON A TESLA T4: `get_available_providers()` lists TensorRT whether or
        not libnvinfer is loadable. When it is listed but unloadable, ORT does not skip
        it and continue - it DISCARDS THE ENTIRE PROVIDER LIST and falls back to
        CPU-only - the SLOWEST option - on a stock `pip install onnxruntime-gpu`.
        Since this set is exactly what a GPU box reports, this test reproduces that
        situation. `device="cuda"` + `format="tensorrt_engine"` is the supported way
        to reach TensorRT, with the build cost paid at publish time.
        """
        _set_available(
            monkeypatch,
            {"TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"},
        )
        for device in ("auto", "cuda"):
            resolved = runtime.resolve_providers(device)
            assert "TensorrtExecutionProvider" not in resolved, device
            assert resolved[0] == (
                "CUDAExecutionProvider",
                {"cudnn_conv_algo_search": "HEURISTIC"},
            ), device
            assert resolved[-1] == "CPUExecutionProvider", device

    def test_the_measurement_hatch_can_still_pin_tensorrt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The benchmark harness measures the TRT provider it excludes, which is how
        the DO-NOT-UNDO above stays an evidenced claim rather than folklore."""
        _set_available(monkeypatch, {"TensorrtExecutionProvider", "CPUExecutionProvider"})
        asked = ["TensorrtExecutionProvider", "CPUExecutionProvider"]
        assert runtime.resolve_providers(requested=asked) == asked


class TestOtherProviders:
    def test_directml_and_rocm_included_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_available(monkeypatch, {"DmlExecutionProvider", "ROCMExecutionProvider"})
        assert runtime.resolve_providers("auto") == [
            "DmlExecutionProvider",
            "ROCMExecutionProvider",
            "CPUExecutionProvider",
        ]


class TestDeviceLabel:
    def test_cpu_only_is_cpu(self) -> None:
        assert runtime.device_label(["CPUExecutionProvider"]) == "cpu"

    def test_empty_is_cpu(self) -> None:
        assert runtime.device_label([]) == "cpu"

    def test_cuda(self) -> None:
        assert runtime.device_label(["CUDAExecutionProvider", "CPUExecutionProvider"]) == "cuda"

    def test_tensorrt_maps_to_cuda(self) -> None:
        assert runtime.device_label(["TensorrtExecutionProvider", "CPUExecutionProvider"]) == "cuda"

    def test_coreml(self) -> None:
        assert runtime.device_label(["CoreMLExecutionProvider", "CPUExecutionProvider"]) == "coreml"

    def test_dml(self) -> None:
        assert runtime.device_label(["DmlExecutionProvider", "CPUExecutionProvider"]) == "dml"

    def test_rocm(self) -> None:
        assert runtime.device_label(["ROCMExecutionProvider", "CPUExecutionProvider"]) == "rocm"

    def test_unrecognized_provider_falls_through_to_cpu(self) -> None:
        assert runtime.device_label(["SomeFutureExecutionProvider"]) == "cpu"

    def test_reports_what_ran_not_what_was_first_in_a_failed_request(self) -> None:
        """`resolved` is `session.get_providers()` -- the providers ORT actually
        KEPT. A CUDA request that failed to load and left only CPU must report
        cpu, not silently claim cuda."""
        assert runtime.device_label(["CPUExecutionProvider"]) == "cpu"


class TestResolveTorchDevice:
    """`auto` searches; a named device is VERIFIED against what torch can see."""

    def test_cpu_is_honoured_regardless_of_hardware(self) -> None:
        pytest.importorskip("torch")
        assert runtime.resolve_torch_device("cpu") == "cpu"

    def test_auto_prefers_cuda_over_everything(self, monkeypatch: pytest.MonkeyPatch) -> None:
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert runtime.resolve_torch_device("auto") == "cuda"

    def test_auto_prefers_mps_when_no_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        torch = pytest.importorskip("torch")
        mps = getattr(torch.backends, "mps", None)
        if mps is None:
            pytest.skip("this torch build has no torch.backends.mps")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(mps, "is_available", lambda: True)
        monkeypatch.setattr(mps, "is_built", lambda: True)
        assert runtime.resolve_torch_device("auto") == "mps"

    def test_auto_requires_mps_both_available_and_built(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A real op gap: MPS reporting available but NOT built (or vice versa)
        must not be auto-selected."""
        torch = pytest.importorskip("torch")
        mps = getattr(torch.backends, "mps", None)
        if mps is None:
            pytest.skip("this torch build has no torch.backends.mps")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(mps, "is_available", lambda: True)
        monkeypatch.setattr(mps, "is_built", lambda: False)
        assert runtime.resolve_torch_device("auto") == "cpu"

    def test_auto_falls_back_to_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        mps = getattr(torch.backends, "mps", None)
        if mps is not None:
            monkeypatch.setattr(mps, "is_available", lambda: False)
        assert runtime.resolve_torch_device("auto") == "cpu"

    def test_named_cuda_without_cuda_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole point of the collapse: on a `.pth` a silent demotion to CPU is
        a 4-10x slowdown with no signal at all."""
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(ValueError) as exc:
            runtime.resolve_torch_device("cuda")
        assert "torch.cuda.is_available() is False" in str(exc.value)
        assert "device='cpu'" in str(exc.value), "must name a way forward"

    def test_the_cuda_refusal_points_at_mps_when_that_is_the_real_accelerator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        torch = pytest.importorskip("torch")
        mps = getattr(torch.backends, "mps", None)
        if mps is None:
            pytest.skip("this torch build has no torch.backends.mps")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(mps, "is_available", lambda: True)
        monkeypatch.setattr(mps, "is_built", lambda: True)
        with pytest.raises(ValueError, match="device='mps'"):
            runtime.resolve_torch_device("cuda")

    def test_an_out_of_range_gpu_index_raises_naming_the_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        with pytest.raises(ValueError, match="1 CUDA device"):
            runtime.resolve_torch_device("cuda:3")

    def test_an_in_range_gpu_index_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
        assert runtime.resolve_torch_device("cuda:1") == "cuda:1"

    def test_named_mps_without_mps_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        torch = pytest.importorskip("torch")
        mps = getattr(torch.backends, "mps", None)
        if mps is None:
            pytest.skip("this torch build has no torch.backends.mps")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(mps, "is_available", lambda: False)
        with pytest.raises(ValueError, match="MPS backend"):
            runtime.resolve_torch_device("mps")

    def test_a_graph_only_device_raises_pointing_at_the_graph_formats(self) -> None:
        pytest.importorskip("torch")
        with pytest.raises(ValueError, match="format='onnx'"):
            runtime.resolve_torch_device("dml")


class TestWarnOnFallback:
    def test_no_requested_provider_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger="pictograph.inference")
        runtime.warn_on_fallback(None, ["CPUExecutionProvider"])
        assert caplog.records == []

    def test_empty_requested_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger="pictograph.inference")
        runtime.warn_on_fallback([], ["CPUExecutionProvider"])
        assert caplog.records == []

    def test_cpu_only_request_never_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger="pictograph.inference")
        runtime.warn_on_fallback(["CPUExecutionProvider"], ["CPUExecutionProvider"])
        assert caplog.records == []

    def test_matched_provider_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger="pictograph.inference")
        runtime.warn_on_fallback(
            ["CUDAExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        assert caplog.records == []

    def test_missing_requested_provider_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger="pictograph.inference")
        runtime.warn_on_fallback(["CUDAExecutionProvider"], ["CPUExecutionProvider"])
        assert len(caplog.records) == 1
        assert "CUDAExecutionProvider" in caplog.text

    def test_tuple_form_requested_provider_is_matched_by_name(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="pictograph.inference")
        requested = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})]
        runtime.warn_on_fallback(requested, ["CPUExecutionProvider"])
        assert len(caplog.records) == 1
        assert "CUDAExecutionProvider" in caplog.text

    def test_tuple_form_requested_provider_that_matched_is_silent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="pictograph.inference")
        requested = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})]
        runtime.warn_on_fallback(requested, ["CUDAExecutionProvider", "CPUExecutionProvider"])
        assert caplog.records == []


class TestSessionOptions:
    def test_default_leaves_ort_thread_count_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("onnxruntime")
        monkeypatch.delenv("PICTOGRAPH_INFERENCE_THREADS", raising=False)
        opts = runtime.session_options()
        assert opts.intra_op_num_threads == 0  # ORT's own "size from the host" default

    def test_env_var_sets_thread_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("onnxruntime")
        monkeypatch.setenv("PICTOGRAPH_INFERENCE_THREADS", "4")
        assert runtime.session_options().intra_op_num_threads == 4

    def test_explicit_threads_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("onnxruntime")
        monkeypatch.setenv("PICTOGRAPH_INFERENCE_THREADS", "4")
        assert runtime.session_options(threads=2).intra_op_num_threads == 2

    def test_invalid_env_var_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("onnxruntime")
        monkeypatch.setenv("PICTOGRAPH_INFERENCE_THREADS", "not-a-number")
        assert runtime.session_options().intra_op_num_threads == 0

    def test_graph_optimization_is_all(self) -> None:
        ort = pytest.importorskip("onnxruntime")
        opts = runtime.session_options()
        assert opts.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL


class TestEmptyDeviceCache:
    def test_cpu_is_a_no_op(self) -> None:
        runtime.empty_device_cache("cpu")  # must not raise

    def test_unrecognized_device_is_a_no_op(self) -> None:
        runtime.empty_device_cache("some-future-device")  # must not raise

    def test_cuda_empties_the_caching_allocator_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        torch = pytest.importorskip("torch")
        calls: list[str] = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("emptied"))
        runtime.empty_device_cache("cuda")
        assert calls == ["emptied"]

    def test_cuda_without_availability_does_not_call_empty_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        torch = pytest.importorskip("torch")
        calls: list[str] = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("emptied"))
        runtime.empty_device_cache("cuda")
        assert calls == []

    def test_mps_empties_the_backend_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        torch = pytest.importorskip("torch")
        backend = getattr(torch, "mps", None)
        if backend is None or not hasattr(backend, "empty_cache"):
            pytest.skip("this torch build exposes no torch.mps.empty_cache")
        calls: list[str] = []
        monkeypatch.setattr(backend, "empty_cache", lambda: calls.append("emptied"))
        runtime.empty_device_cache("mps")
        assert calls == ["emptied"]

    def test_missing_torch_is_a_no_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`empty_device_cache` must not blow up when torch isn't installed at
        all -- torch is an optional extra, not a base dependency."""
        import builtins

        real_import = builtins.__import__

        def _no_torch(name: str, *args: object, **kwargs: object) -> object:
            if name == "torch":
                raise ImportError("simulated: torch not installed")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", _no_torch)
        runtime.empty_device_cache("cuda")  # must not raise
