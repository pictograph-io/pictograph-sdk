"""A session build that FAILS must degrade to CPU, not kill the load.

The provider ladder's promise is that CPU-last makes every configuration
recoverable. ONNX Runtime only honours half of that on its own:

- a provider that fails to REGISTER (missing library, wrong driver) is dropped and
  the session still builds - graceful, no help needed;
- a provider that registers and then fails to COMPILE THE MODEL **raises**, taking
  the whole load down even though CPU was right there in the list.

That second case is real, not defensive. MEASURED on macOS 15.5 / onnxruntime 1.26
against the shipped 1.68.0 wheel: loading the RF-DETR keypoint export through CoreML
raised ``Failed to create MLModel ... error code: -7`` from CoreML's MLProgram
compiler, reproduced against a freshly-cleared compiled-model cache - so it is the
graph, not a stale artifact. A user's model simply would not load.

The retry is scoped to ``device="auto"``. When the caller NAMED the device, the same
retry would hand back a model running 4-10x slower than the one they asked for while
reporting success, so it raises instead - the no-silent-fallback rule, applied to the
one failure mode that CPU-last cannot cover.
"""

from __future__ import annotations

from typing import Any

import pytest

# Every test here drives a real ORT session build, so without the [inference]
# extra all 7 ERROR on `No module named 'onnxruntime'` rather than skipping. CI
# installs `[dev,cli,agents,cache,telemetry]` deliberately - onnxruntime, torch
# and numpy are not in it - so this module was RED on main and the suite's other
# ~38 importorskip sites are why nothing else was.
pytest.importorskip("onnxruntime", reason="needs the [inference] extra")

from pictograph.inference import _onnx

CPU = "CPUExecutionProvider"
COREML = "CoreMLExecutionProvider"
CUDA = "CUDAExecutionProvider"


class _FakeDispatch:
    """Records each build attempt; fails on any non-CPU-only provider list."""

    def __init__(self, *, fail_non_cpu: bool = True) -> None:
        self.attempts: list[list[Any]] = []
        self._fail_non_cpu = fail_non_cpu

    def build_wrapper(self, *, providers: list[Any], **_: Any) -> str:
        self.attempts.append(providers)
        names = [p if isinstance(p, str) else p[0] for p in providers]
        if self._fail_non_cpu and names != [CPU]:
            raise RuntimeError("Failed to create MLModel, error: ... error code: -7")
        return "wrapper"


class TestBuildFallback:
    def test_compile_failure_retries_on_cpu_and_returns_a_wrapper(self) -> None:
        dispatch = _FakeDispatch()
        wrapper, attempted = _onnx._build_with_fallback(
            dispatch, [(COREML, {"ModelFormat": "MLProgram"}), CPU], named_device=None
        )
        assert wrapper == "wrapper"
        assert attempted == [CPU]
        # It tried the real ladder FIRST, then CPU - not CPU straight away.
        assert len(dispatch.attempts) == 2
        assert [p if isinstance(p, str) else p[0] for p in dispatch.attempts[0]] == [COREML, CPU]
        assert dispatch.attempts[1] == [CPU]

    def test_a_working_ladder_is_not_retried(self) -> None:
        dispatch = _FakeDispatch(fail_non_cpu=False)
        ladder = [(CUDA, {"cudnn_conv_algo_search": "HEURISTIC"}), CPU]
        wrapper, attempted = _onnx._build_with_fallback(dispatch, ladder, named_device=None)
        assert wrapper == "wrapper"
        assert attempted == ladder
        assert len(dispatch.attempts) == 1

    def test_a_cpu_only_failure_propagates(self) -> None:
        """Nothing left to fall back to - the real error must reach the caller."""

        class _AlwaysFails:
            def build_wrapper(self, **_: Any) -> None:
                raise RuntimeError("genuinely broken model")

        with pytest.raises(RuntimeError, match="genuinely broken model"):
            _onnx._build_with_fallback(_AlwaysFails(), [CPU], named_device=None)

    def test_the_retry_is_warned_not_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="pictograph.inference"):
            _onnx._build_with_fallback(_FakeDispatch(), [(COREML, {}), CPU], named_device=None)
        # getMessage(), not .message - the latter is only populated once a formatter
        # has run, so it raises AttributeError on a record captured straight off the
        # logger.
        assert any("retrying on CPU" in r.getMessage() for r in caplog.records)

    def test_attempted_providers_suppress_the_misleading_second_warning(self) -> None:
        """After falling back we report CPU as what we ASKED for.

        Otherwise the caller's silent-fallback check fires a second warning telling
        the user to install a runtime that IS installed and merely could not compile
        this graph - two warnings, one of them wrong.
        """
        _, attempted = _onnx._build_with_fallback(
            _FakeDispatch(), [(COREML, {}), CPU], named_device=None
        )
        # warn_on_fallback(attempted, resolved) must see no discrepancy.
        import logging

        from pictograph.inference.runtime import warn_on_fallback

        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        logger = logging.getLogger("pictograph.inference")
        handler = _Capture()
        logger.addHandler(handler)
        try:
            warn_on_fallback(attempted, [CPU])
        finally:
            logger.removeHandler(handler)
        assert records == []


class TestNamedDeviceDoesNotDegrade:
    """A device the caller NAMED must not be quietly swapped for a slower one."""

    def test_a_named_device_raises_instead_of_retrying_on_cpu(self) -> None:
        dispatch = _FakeDispatch()
        with pytest.raises(RuntimeError) as exc:
            _onnx._build_with_fallback(dispatch, [(COREML, {}), CPU], named_device="mps")
        assert "device='mps'" in str(exc.value)
        assert len(dispatch.attempts) == 1, "the CPU retry must not be attempted"

    def test_the_refusal_distinguishes_a_bad_graph_from_a_missing_install(self) -> None:
        """These need different fixes, and the accelerator's own compiler error is
        the actionable half - a user told to reinstall onnxruntime-gpu here would be
        chasing the wrong thing."""
        with pytest.raises(RuntimeError) as exc:
            _onnx._build_with_fallback(_FakeDispatch(), [(COREML, {}), CPU], named_device="mps")
        message = str(exc.value)
        assert "not a missing install" in message
        assert "error code: -7" in message, "the underlying compiler error must survive"
        assert "device='cpu'" in message, "and a way forward must be named"
