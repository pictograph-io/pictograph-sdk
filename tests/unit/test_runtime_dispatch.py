"""The loader surface: format dispatch, argument honesty, cache keys.

Covers the decisions that make ``format=`` a real choice rather than a label:

* the weights suffix decides the format, so an artifact is self-describing and
  there is never a second source of truth to disagree with the file;
* the runtime is DERIVED from the format and never asked for separately, so the
  two can never be given contradicting answers;
* a ``(device, format)`` pair that does not exist RAISES instead of being ignored,
  because a silently-dropped device request reads as "the GPU didn't work";
* derived artifacts are cached under a key that includes everything that makes them
  a different file, so an fp16 sm80 engine cannot be served where an fp32 sm75 one
  was asked for.

None of it needs a GPU, ExecuTorch or TensorRT installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pictograph.inference import (
    DEFAULT_PTE_TARGET,
    PORTABLE_TARGET,
    _artifact_request,
    _artifact_stem,
    _check_native_precision,
)
from pictograph.inference._engine import NodeArg, RuntimeSession, input_hw_from
from pictograph.inference.runtime import (
    DEVICES,
    DEVICES_BY_RUNTIME,
    RUNTIMES,
    WEIGHT_FORMATS,
    check_device_supported,
    format_for_weights,
    runtime_for_format,
    runtime_for_weights,
)
from pictograph.models.model import Model, ModelFileEntry


def _model(**over: Any) -> Model:
    base: dict[str, Any] = {
        "id": "11111111-1111-1111-1111-111111111111",
        "organization_id": "org",
        "name": "My Detector",
        "model_type": "object_detection",
        "architecture": "YOLOX",
        "visibility": "private",
        "status": "ready",
        "precision": "fp32",
        "class_mapping": {"classes": ["person", "car"]},
        "created_at": "2026-07-01T00:00:00Z",
        "updated_at": "2026-07-01T00:00:00Z",
    }
    base.update(over)
    return Model.model_validate(base)


class TestSuffixDispatch:
    @pytest.mark.parametrize(
        ("name", "fmt", "runtime"),
        [
            ("model.onnx", "onnx", "onnxruntime"),
            ("xnnpack-fp32.pte", "pytorch_engine", "executorch"),
            ("sm75-trt10.13.3.9-fp16.engine", "tensorrt_engine", "tensorrt"),
            ("model.plan", "tensorrt_engine", "tensorrt"),
            ("MODEL.ONNX", "onnx", "onnxruntime"),
            ("checkpoint_best_042.pth", "pytorch", "pytorch"),
            ("model.safetensors", "safetensors", "pytorch"),
        ],
    )
    def test_the_artifact_says_what_it_is_and_the_runtime_follows(
        self, name: str, fmt: str, runtime: str
    ) -> None:
        """All five formats are recognised, including the two native containers -
        answering "unknown" for a ``.pth`` would simply be false. Whether a given
        entry point can EXECUTE it is asked separately."""
        assert format_for_weights(Path(name)) == fmt
        assert runtime_for_weights(Path(name)) == runtime

    def test_an_unknown_suffix_lists_what_is_supported(self) -> None:
        with pytest.raises(ValueError, match=r"\.engine"):
            format_for_weights(Path("weights.bin"))

    def test_a_near_miss_suffix_gets_the_convention_spelled_out(self) -> None:
        with pytest.raises(ValueError, match=r"conventionally '\.pth'"):
            format_for_weights(Path("last.ckpt"))

    def test_unknown_format_names_the_five_and_says_they_are_not_runtimes(self) -> None:
        """The likeliest wrong value is a RUNTIME name, so the message says so."""
        with pytest.raises(ValueError, match="pytorch, safetensors, pytorch_engine"):
            runtime_for_format("tflite")
        with pytest.raises(ValueError, match="the runtime follows from it"):
            runtime_for_format("onnxruntime")
        assert {runtime_for_format(f) for f in WEIGHT_FORMATS} == set(RUNTIMES)


class TestArgumentHonesty:
    """A ``(device, format)`` pair that does not exist must raise, never be dropped.

    The messages speak in ``format=``, the vocabulary the caller passed - naming the
    runtime alone would send them looking for an argument that does not exist.
    """

    def test_a_tensorrt_plan_has_no_cpu_form(self) -> None:
        with pytest.raises(ValueError, match="format='tensorrt_engine'"):
            check_device_supported("cpu", "tensorrt", "tensorrt_engine")

    def test_the_refusal_names_the_artifact_to_load_instead(self) -> None:
        with pytest.raises(ValueError, match=r"\.onnx"):
            check_device_supported("cpu", "tensorrt", "tensorrt_engine")
        with pytest.raises(ValueError, match="tensorrt_engine"):
            check_device_supported("cuda", "executorch", "pytorch_engine")

    def test_no_cuda_pte_lowering_is_published(self) -> None:
        with pytest.raises(ValueError, match="format='pytorch_engine'"):
            check_device_supported("cuda", "executorch", "pytorch_engine")

    def test_torch_cannot_reach_a_delegate_only_device(self) -> None:
        with pytest.raises(ValueError, match="format='pytorch'"):
            check_device_supported("vulkan", "pytorch", "pytorch")

    def test_auto_is_supported_by_every_runtime(self) -> None:
        """`auto` is the default, so a runtime it could not serve would be a format
        nobody can load without naming hardware they may not have."""
        for runtime in RUNTIMES:
            check_device_supported("auto", runtime, "onnx")
            assert "auto" in DEVICES_BY_RUNTIME[runtime]

    def test_every_taught_device_is_reachable_by_some_runtime(self) -> None:
        """A value in the taught vocabulary that no format can run on would be a
        documented dead end."""
        reachable = {d for devices in DEVICES_BY_RUNTIME.values() for d in devices}
        assert set(DEVICES) <= reachable

    def test_the_pairs_that_do_exist_pass(self) -> None:
        check_device_supported("cuda", "onnxruntime", "onnx")
        check_device_supported("cuda:1", "pytorch", "pytorch")
        check_device_supported("mps", "onnxruntime", "onnx")
        check_device_supported("mps", "pytorch", "safetensors")
        check_device_supported("cuda", "tensorrt", "tensorrt_engine")
        check_device_supported("cpu", "executorch", "pytorch_engine")

    def test_native_precision_mismatch_explains_the_alternatives(self) -> None:
        """A checkpoint is the one artifact with no derived form, so fp16 is a real
        'cannot', and the message has to say what CAN be done instead."""
        with pytest.raises(ValueError, match="no derived"):
            _check_native_precision(_model(precision="fp32"), "safetensors", "fp16")

    def test_the_alternatives_are_named_as_formats_not_runtimes(self) -> None:
        with pytest.raises(ValueError, match="format='onnx' / 'pytorch_engine'"):
            _check_native_precision(_model(precision="fp32"), "pytorch", "fp16")

    def test_matching_or_absent_precision_is_fine(self) -> None:
        _check_native_precision(_model(precision="fp32"), "pytorch", "fp32")
        _check_native_precision(_model(precision="fp32"), "pytorch", None)


class TestArtifactSelection:
    """Which file ``get_model(format=…)`` actually fetches, and under which WIRE
    format name - the SDK's ``pytorch_engine`` / ``tensorrt_engine`` are translated
    to the route's ``pte`` / ``engine`` exactly here and nowhere else."""

    def test_pytorch_engine_defaults_to_the_portable_lowering(self) -> None:
        fmt, precision, target, suffix, derived = _artifact_request(
            _model(), "pytorch_engine", None, None
        )
        assert (fmt, precision, target, suffix, derived) == (
            "pte",
            "fp32",
            DEFAULT_PTE_TARGET,
            ".pte",
            True,
        )

    def test_shipped_onnx_is_not_a_derived_fetch(self) -> None:
        """The version's own fp32 graph comes from the 1:1 column, so the request
        carries no precision/target - which is also what keeps it working against a
        backend that predates the derived-artifact params."""
        fmt, _, target, _, derived = _artifact_request(_model(precision="fp32"), "onnx", None, None)
        assert (fmt, target, derived) == ("onnx", "", False)
        _, _, _, _, still_shipped = _artifact_request(
            _model(precision="fp32"), "onnx", "fp32", None
        )
        assert still_shipped is False

    def test_a_derived_fp16_onnx_is_a_portable_artifact_row(self) -> None:
        """Contract amendment 2026-07-30: a version has ONE gcs_weights_path (fp32,
        never overwritten), so an fp16 graph derived after training is a
        model_artifacts row with the `portable` sentinel - not a second column."""
        fmt, precision, target, _, derived = _artifact_request(
            _model(precision="fp32"), "onnx", "fp16", None
        )
        assert (fmt, precision, target, derived) == ("onnx", "fp16", PORTABLE_TARGET, True)

    def test_an_fp16_version_serves_its_own_graph(self) -> None:
        """If the VERSION is fp16, its 1:1 column already holds the fp16 graph and
        there is nothing to derive."""
        _, _, _, _, derived = _artifact_request(_model(precision="fp16"), "onnx", "fp16", None)
        assert derived is False

    def test_tensorrt_engine_defaults_to_this_machines_architecture(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fetching any other engine would be downloading a file that provably
        cannot load here."""
        import pictograph.inference._tensorrt as trt_module
        from pictograph.inference._tensorrt import EngineTarget

        monkeypatch.setattr(
            trt_module, "detect_local_target", lambda: EngineTarget("sm86", "trt-10.13.3.9")
        )
        fmt, _, target, suffix, derived = _artifact_request(
            _model(), "tensorrt_engine", "fp16", None
        )
        assert (fmt, target, suffix, derived) == ("engine", "sm86", ".engine", True)

    def test_tensorrt_engine_without_a_detectable_gpu_asks_rather_than_guesses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pictograph.inference._tensorrt as trt_module
        from pictograph.inference._tensorrt import EngineTarget

        monkeypatch.setattr(
            trt_module, "detect_local_target", lambda: EngineTarget("unknown", "unknown")
        )
        with pytest.raises(ValueError, match="target='sm75'"):
            _artifact_request(_model(), "tensorrt_engine", None, None)

    def test_an_explicit_target_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pictograph.inference._tensorrt as trt_module
        from pictograph.inference._tensorrt import EngineTarget

        monkeypatch.setattr(
            trt_module, "detect_local_target", lambda: EngineTarget("sm86", "trt-10.13.3.9")
        )
        _, _, target, _, _ = _artifact_request(_model(), "tensorrt_engine", None, "sm90")
        assert target == "sm90"


class _RecordingModels:
    """A stand-in for ``client.models`` that records the download it was asked for."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def download(self, **kwargs: Any) -> Path:
        self.calls.append(kwargs)
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"stub")
        return out

    def download_file(self, **_kwargs: Any) -> Path:  # pragma: no cover - keypoint only
        raise AssertionError("no config fetch expected for a non-keypoint model")


class TestTheFetchRequest:
    """WHICH artifact the loader asks the API for - the wire contract, recorded.

    These assert the exact query the backend's ``/download`` resolver receives, so a
    drift between the SDK's request and the contract's parameters is caught here
    rather than as a 404 on a user's machine.
    """

    def _fetch(self, model: Model, tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
        from pictograph.inference import _build_graph_model

        models = _RecordingModels()
        # The engine build is irrelevant here - the download is what is under test.
        # The stub bytes are not a real graph, so the engine build fails - which is
        # fine and deliberate: the DOWNLOAD REQUEST is what is under test here.
        with pytest.raises(Exception):
            _build_graph_model(
                model,
                models=models,
                task=None,
                confidence=0.5,
                device="cpu",
                cache_dir=tmp_path,
                **kwargs,
            )
        assert len(models.calls) == 1, "the loader must fetch exactly one artifact"
        return models.calls[0]

    def test_the_shipped_onnx_is_requested_without_the_derived_params(self, tmp_path: Path) -> None:
        """An older backend that predates precision/target must still serve this."""
        call = self._fetch(
            _model(model_type="classification", architecture="resnet18"),
            tmp_path,
            format="onnx",
            precision=None,
            target=None,
        )
        assert call["format"] == "onnx"
        assert "precision" not in call
        assert "target" not in call

    def test_a_derived_fp16_onnx_asks_for_the_portable_artifact_row(self, tmp_path: Path) -> None:
        """Matches the row the platform actually builds: runtime='onnxruntime',
        format='onnx', target_key='portable', precision='fp16' at
        ``{artifact_dir}/onnx/portable-fp16.onnx``."""
        call = self._fetch(
            _model(model_type="classification", architecture="resnet18", precision="fp32"),
            tmp_path,
            format="onnx",
            precision="fp16",
            target=None,
        )
        assert call["format"] == "onnx"
        assert call["precision"] == "fp16"
        assert call["target"] == PORTABLE_TARGET

    def test_pytorch_engine_asks_for_the_portable_lowering_by_default(self, tmp_path: Path) -> None:
        call = self._fetch(
            _model(model_type="classification", architecture="resnet18"),
            tmp_path,
            format="pytorch_engine",
            precision=None,
            target=None,
        )
        assert (call["format"], call["precision"], call["target"]) == (
            "pte",
            "fp32",
            DEFAULT_PTE_TARGET,
        )

    def test_tensorrt_engine_asks_for_this_machines_architecture(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import pictograph.inference._tensorrt as trt_module
        from pictograph.inference._tensorrt import EngineTarget

        monkeypatch.setattr(
            trt_module, "detect_local_target", lambda: EngineTarget("sm75", "trt-10.13.3.9")
        )
        call = self._fetch(
            _model(model_type="classification", architecture="resnet18"),
            tmp_path,
            format="tensorrt_engine",
            precision="fp16",
            target=None,
        )
        assert (call["format"], call["precision"], call["target"]) == ("engine", "fp16", "sm75")


class TestCacheKeying:
    """Two different artifacts of one model must never share a cache filename."""

    def test_the_shipped_graph_keeps_the_pre_existing_stem(self) -> None:
        """So an existing ONNX cache entry is not invalidated by the multi-runtime
        work - a user upgrading the SDK must not silently re-download every model."""
        from pictograph.inference import _cache_stem

        model = _model()
        assert _artifact_stem(model, "onnx", "fp32", "") == _cache_stem(model)

    def test_every_derived_dimension_is_in_the_key(self) -> None:
        model = _model()
        stems = {
            _artifact_stem(model, "tensorrt_engine", "fp16", "sm80"),
            _artifact_stem(model, "tensorrt_engine", "fp32", "sm80"),
            _artifact_stem(model, "tensorrt_engine", "fp16", "sm75"),
            _artifact_stem(model, "pytorch_engine", "fp16", "xnnpack"),
            _artifact_stem(model, "onnx", "fp16", "portable"),
        }
        assert len(stems) == 5, "two distinct artifacts collided on one cache name"

    def test_the_key_still_carries_the_served_version(self) -> None:
        """Keying on the id alone let a retrained model keep predicting with whatever
        was downloaded first, forever, on that machine."""
        a = _artifact_stem(
            _model(updated_at="2026-07-01T00:00:00Z"), "tensorrt_engine", "fp16", "sm80"
        )
        b = _artifact_stem(
            _model(updated_at="2026-07-02T00:00:00Z"), "tensorrt_engine", "fp16", "sm80"
        )
        assert a != b


class _StubSession(RuntimeSession):
    """A session whose forward returns fixed arrays, to test the shim's contract."""

    def __init__(self, inputs: list[NodeArg], outputs: list[NodeArg], values: list[Any]) -> None:
        super().__init__(inputs=inputs, outputs=outputs)
        self._values = values

    def _forward(self, tensor: Any) -> list[Any]:  # noqa: ARG002 - fixed outputs
        return list(self._values)


class TestSessionShim:
    """The ORT-shaped surface the vendored wrappers drive."""

    def _session(self) -> _StubSession:
        return _StubSession(
            [NodeArg("input", [1, 3, 224, 224])],
            [NodeArg("boxes", [1, 5, 4]), NodeArg("logits", [1, 5, 3])],
            ["BOXES", "LOGITS"],
        )

    def test_run_with_none_returns_every_output_in_graph_order(self) -> None:
        assert self._session().run(None, {"input": object()}) == ["BOXES", "LOGITS"]

    def test_run_with_names_returns_them_in_the_requested_order(self) -> None:
        """RF-DETR asks by name, so this path has to be exact."""
        session = self._session()
        assert session.run(["logits", "boxes"], {"input": object()}) == ["LOGITS", "BOXES"]

    def test_the_feed_key_is_ignored_because_these_graphs_are_single_input(self) -> None:
        """YOLOX hardcodes ``{"input": ...}`` rather than reading get_inputs()[0].name,
        so keying strictly would work for five families and KeyError for one."""
        assert self._session().run(None, {"whatever": object()}) == ["BOXES", "LOGITS"]

    def test_an_empty_feed_is_an_error_not_a_silent_no_op(self) -> None:
        with pytest.raises(ValueError, match="No input tensor"):
            self._session().run(None, {})

    def test_get_inputs_and_outputs_expose_ort_shaped_nodeargs(self) -> None:
        session = self._session()
        assert session.get_inputs()[0].name == "input"
        assert session.get_inputs()[0].shape == [1, 3, 224, 224]
        assert [o.name for o in session.get_outputs()] == ["boxes", "logits"]


class TestInputShapeFromArtifact:
    """A compiled artifact's input shape beats the config's - it is the ONLY shape
    it accepts."""

    def test_a_concrete_nchw_shape_wins(self) -> None:
        session = _StubSession([NodeArg("input", [1, 3, 384, 512])], [], [])
        assert input_hw_from(session, (640, 640)) == (384, 512)

    @pytest.mark.parametrize(
        "shape",
        [[1, 3, -1, -1], [1, 3, 0, 0], ["batch", 3, 224, 224, 1], [1, 3], []],
    )
    def test_anything_less_than_concrete_falls_back_to_the_config(self, shape: list[Any]) -> None:
        session = _StubSession([NodeArg("input", shape)], [], [])
        assert input_hw_from(session, (640, 640)) == (640, 640)

    def test_no_inputs_falls_back(self) -> None:
        assert input_hw_from(_StubSession([], [], []), (321, 123)) == (321, 123)


class TestManifestFields:
    """The five additive fields the files manifest gained (contract § 5.2)."""

    def test_an_artifact_row_parses_with_its_binding(self) -> None:
        row = ModelFileEntry.model_validate(
            {
                "version_id": "v1",
                "name": "sm75-trt10.13.3.9-fp16.engine",
                "kind": "weights",
                "format": "engine",
                "runtime": "tensorrt",
                "precision": "fp16",
                "target_key": "sm75",
                "toolchain_version": "trt-10.13.3.9",
                "stale": False,
                "artifact_id": "a1",
            }
        )
        assert row.runtime == "tensorrt"
        assert row.target_key == "sm75"
        assert row.stale is False

    def test_an_older_backend_that_omits_them_still_parses(self) -> None:
        """A new SDK against an older API degrades to 'unknown binding', not a crash."""
        row = ModelFileEntry.model_validate(
            {"version_id": "v1", "name": "m.onnx", "kind": "weights", "format": "onnx"}
        )
        assert row.runtime is None
        assert row.target_key is None
        assert row.stale is False
