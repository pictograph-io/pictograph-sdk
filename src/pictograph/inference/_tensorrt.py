"""The TensorRT inference engine - runs a prebuilt ``.engine`` plan.

A TensorRT engine is **not a model file**. It is a compiled, hardware-specific
execution plan, bound simultaneously to the GPU architecture (SM compute
capability) it was built on, the exact TensorRT version that built it, the precision
it was built for, and its build-time shape profile. An engine built on an A100
(``sm80``) does not deserialize on a T4 (``sm75``) - it fails at LOAD, not at
accuracy.

That is why most of this module is not inference. **The predictable support burden
here is a user who copies ``model.engine`` to another machine and gets a raw
deserialization crash**, which reads as "Pictograph gave me a broken file". So the
loader reads what the artifact was built for, compares it against the local device,
and refuses FIRST with a sentence that names both - see :func:`engine_mismatch_message`,
which is a verbatim mirror of the backend's
the server's artifact-compatibility contract so the API,
the SDK and the UI all say the same thing. **kept in sync - change both together.**

Local vs. platform staleness - a distinction worth keeping straight
-------------------------------------------------------------------
The backend calls an engine *stale* when its ``toolchain_version`` differs from the
platform's currently pinned TensorRT (``model_artifacts.is_stale``). That is the
signal the UI uses to offer a rebuild.

**This module does not use that rule to refuse a load**, and deliberately so: what
decides whether a plan will actually deserialize is the TensorRT installed on the
machine doing the loading, not the one the platform happens to pin this week. A user
whose local TensorRT matches an "outdated" engine can run it perfectly well, and
refusing would be a lie. So the load gate compares **built vs. detected**; platform
staleness stays an advisory concern of the manifest.

Everything except the forward pass is the shared vendored wrapper - see
:mod:`pictograph.inference._engine`.

TensorRT stays OUT of the automatic ONNX provider ladder
--------------------------------------------------------
``device="cuda"`` + ``format="tensorrt_engine"`` is how a caller reaches TensorRT,
and this module is that pairing. It exists precisely because an ahead-of-time
``.engine`` removes the 8-65 s first-inference build cost that made ORT's
``TensorrtExecutionProvider`` a bad automatic default. It does **not** license
re-adding that provider to ``device="auto"``'s ONNX ladder - see the measured
DO-NOT-UNDO in :mod:`pictograph.inference.runtime`, which is still in force.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from pictograph.inference._engine import (
    NodeArg,
    RuntimeSession,
    WrapperEngine,
    build_wrapper_with_session,
    input_hw_from,
)
from pictograph.inference.runtime import normalize_device

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pictograph.inference.runtime import Device

__all__ = [
    "GPU_SM",
    "EngineTarget",
    "TensorRTEngine",
    "TensorRTSession",
    "build_tensorrt_engine",
    "check_tensorrt_device",
    "detect_local_target",
    "engine_mismatch_message",
    "parse_engine_filename",
]

_LOG = logging.getLogger("pictograph.inference")

_INSTALL_HINT = (
    "Running a .engine plan needs the TensorRT Python runtime. Install it with:\n"
    '    pip install "pictograph[inference,tensorrt]"\n'
    "TensorRT is NVIDIA-only - there is no CPU, Apple or AMD build, and a .engine "
    "cannot be run on those platforms at all. Use the .onnx or .pte artifact there."
)

_TORCH_HINT = (
    "The TensorRT backend uses torch to allocate its GPU buffers, so it needs a "
    "CUDA-enabled torch. Install it with:\n"
    '    pip install "pictograph[inference,tensorrt]"'
)

# GPU trade name → SM compute capability. Must stay in sync with
# the server's GPU-compatibility table and the contract's § 4 table.
# The engine's target is the SM, NEVER the trade name: L4 and L40S are both sm89 and
# their plans are genuinely interchangeable, so keying on "l4" would mint two
# identical engines and then refuse to serve one of them.
GPU_SM: dict[str, str] = {
    "t4": "sm75",
    "l4": "sm89",
    "a10g": "sm86",
    "a100": "sm80",
    "h100": "sm90",
}

# The contract's basename convention, in BOTH of its published forms.
#
# Current (from 2026-07-31): the file is named after its MODEL, and the
# binding is a SUFFIX - `my-model-sm75-trt10.13.3.9.engine`, or
# `my-model-fp16-sm75-trt10.13.3.9.engine` at fp16 (fp32 carries no precision
# token, being the default a reader assumes).
#
# Legacy: `sm75-trt10.13.3.9-fp16.engine`, named after the build target,
# precision last. Still matched - those artifacts were never migrated and must
# keep loading.
#
# The TRT version carries dots and never dashes, which is what keeps both
# unambiguous. `sm\d+` is anchored to a dash-boundary so a model legitimately
# called `sm75-something` cannot be mistaken for the binding: the binding is the
# LAST `-sm…-trt…` run in the stem.
_FILENAME_RE = re.compile(
    r"^(?:.*?-)??(?P<sm>sm\d+)-trt(?P<version>[\d.]+)"
    r"(?:-(?P<precision>fp32|fp16))?$",
    re.IGNORECASE,
)

# The precision token in the CURRENT form sits before the SM, as the last thing
# the model-derived stem carries.
_STEM_PRECISION_RE = re.compile(r"-(?P<precision>fp32|fp16)$", re.IGNORECASE)


class EngineTarget(NamedTuple):
    """What a plan was built for, or what a device is.

    ``toolchain_version`` uses the contract's string form - ``trt-10.13.3.9``,
    ``{runtime_short}-{version}``, no ``v`` prefix - so it compares directly against
    the value stored on ``model_artifacts.toolchain_version``.
    """

    sm: str
    toolchain_version: str
    precision: str = "fp32"

    @property
    def trt_version(self) -> str:
        """Just the numeric part - ``10.13.3.9``."""
        return self.toolchain_version.split("-", 1)[-1]

    def __str__(self) -> str:
        return f"{self.sm} / {self.toolchain_version} / {self.precision}"


def engine_mismatch_message(
    *,
    built_target: str,
    built_toolchain: str,
    detected_target: str | None = None,
    detected_toolchain: str | None = None,
) -> str:
    """The ONE sentence the backend, the SDK and the UI all say on a mismatch.

     Verbatim mirror of the server's model-artifact contract's function of
    the same name. The SDK ships as its own wheel and cannot import the backend, so
    the wording is duplicated on purpose - **kept in sync, change both together.** A user
    who hits this in the API, in the SDK and in the modal must read one sentence,
    not three paraphrases of it.
    """
    detected = detected_target or "unknown"
    parts = [
        f"This TensorRT engine was built for {built_toolchain} on {built_target}; "
        f"this device is {detected}"
    ]
    if detected_toolchain and detected_toolchain != built_toolchain:
        parts.append(f" running {detected_toolchain}")
    parts.append(
        ". A TensorRT plan is compiled for one GPU architecture and one "
        "TensorRT version and cannot be loaded anywhere else - rebuild the "
        "engine for your device."
    )
    return "".join(parts)


def parse_engine_filename(weights: Path | str) -> EngineTarget | None:
    """The target encoded in a contract-shaped ``.engine`` basename.

    ``my-model-fp16-sm75-trt10.13.3.9.engine`` →
    ``EngineTarget('sm75', 'trt-10.13.3.9', 'fp16')`` (named after the model,
    binding as a suffix, fp32 implied by the absence of a precision token).

    The legacy target-named form ``sm75-trt10.13.3.9-fp16.engine`` parses to the same
    thing - those artifacts were never migrated, so both must resolve.

    Returns ``None`` for a renamed or hand-made file - the caller then falls back to
    a sidecar, to explicit arguments, or to a documented "unknown" path.
    """
    stem = Path(weights).stem
    match = _FILENAME_RE.match(stem)
    if match is None:
        return None
    precision = match.group("precision")
    if precision is None:
        # Current form: the token, when present at all, is the tail of the
        # model-derived prefix. Absent ⇒ fp32, which is what "no suffix means
        # the default" buys at the cost of exactly this lookup.
        prefix = stem[: match.start("sm")].rstrip("-")
        tail = _STEM_PRECISION_RE.search(prefix)
        precision = tail.group("precision") if tail else "fp32"
    return EngineTarget(
        sm=match.group("sm").lower(),
        toolchain_version=f"trt-{match.group('version')}",
        precision=precision.lower(),
    )


def _sidecar_target(weights: Path) -> EngineTarget | None:
    """A ``<stem>.json`` beside the engine, as the files manifest describes it.

    Lets a user who renamed the file keep the binding, and lets a caller persist the
    manifest row next to the download rather than re-deriving it.
    """
    sidecar = weights.with_suffix(".json")
    if not sidecar.exists():
        return None
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _LOG.debug("Engine sidecar %s is unreadable (%s).", sidecar, exc)
        return None
    target = raw.get("target_key")
    toolchain = raw.get("toolchain_version")
    if not (isinstance(target, str) and isinstance(toolchain, str)):
        return None
    precision = raw.get("precision")
    return EngineTarget(
        sm=target,
        toolchain_version=toolchain,
        precision=precision if isinstance(precision, str) else "fp32",
    )


def _require_tensorrt() -> Any:
    """Import the TensorRT runtime, or raise with the exact install command."""
    try:
        import tensorrt as trt
    except ImportError as exc:  # pragma: no cover - depends on the local env
        raise ImportError(_INSTALL_HINT) from exc
    return trt


def _local_toolchain() -> str | None:
    """The installed TensorRT, in the contract's ``trt-{version}`` form."""
    try:
        import tensorrt as trt
    except ImportError:  # pragma: no cover - depends on the local env
        return None
    version = getattr(trt, "__version__", None)
    return f"trt-{version}" if version else None


def _local_sm() -> str | None:
    """This machine's GPU compute capability as ``smXY``, or ``None`` with no GPU.

    torch first because it is already an SDK extra and answers directly; ``nvidia-smi``
    second so a TensorRT-only environment (no torch) still gets a real answer instead
    of an "unknown device" in the refusal message, which is the one place a vague
    answer costs the user the most.
    """
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"sm{major}{minor}"
    except Exception as exc:  # pragma: no cover - torch optional / driver issues
        _LOG.debug("torch could not report a CUDA capability (%s).", exc)

    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 - resolved absolute path, fixed argv
            [smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - defensive
        _LOG.debug("nvidia-smi could not be run (%s).", exc)
        return None
    first = out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ""
    if re.fullmatch(r"\d+\.\d+", first):
        return "sm" + first.replace(".", "")
    return None


def detect_local_target() -> EngineTarget:
    """This machine's ``(sm, TensorRT version)``, for the compatibility check.

    Never raises: an undetectable device is reported as ``unknown`` so the mismatch
    message can still say what the engine WAS built for, which is the actionable half.
    """
    return EngineTarget(
        sm=_local_sm() or "unknown",
        toolchain_version=_local_toolchain() or "unknown",
    )


def check_engine_compatibility(built: EngineTarget | None, detected: EngineTarget) -> None:
    """Refuse a plan this device cannot deserialize, BEFORE trying to.

    Two checks, and they fail for genuinely different reasons:

    - **SM.** A plan carries architecture-specific compiled kernels. A mismatch is
      absolute.
    - **TensorRT major.minor.** A minor bump invalidates every previously serialized
      plan. A differing PATCH/build is *not* refused - plans do load across patch
      releases - but it is warned about, because it is the shape of a problem that
      otherwise surfaces as an unexplained crash.

    ``built is None`` means the file carried no binding at all (renamed, no sidecar).
    That is not refused: refusing would block a legitimately-correct engine on missing
    metadata. It warns and lets TensorRT be the judge - the one case where a raw
    deserialization error is the honest outcome, because we genuinely do not know.
    """
    if built is None:
        _LOG.warning(
            "This .engine carries no target metadata - its filename does not follow "
            "the {sm}-trt{version}-{precision}.engine convention and there is no "
            "sidecar .json beside it. It cannot be checked against this device (%s) "
            "before loading, so a mismatch will surface as a TensorRT deserialization "
            "error rather than a clear message.",
            detected,
        )
        return

    sm_differs = detected.sm != "unknown" and built.sm != detected.sm
    built_mm = built.trt_version.split(".")[:2]
    detected_mm = detected.trt_version.split(".")[:2]
    version_differs = detected.toolchain_version != "unknown" and built_mm != detected_mm

    if sm_differs or version_differs:
        raise RuntimeError(
            engine_mismatch_message(
                built_target=built.sm,
                built_toolchain=built.toolchain_version,
                detected_target=detected.sm,
                detected_toolchain=detected.toolchain_version,
            )
        )

    if (
        detected.toolchain_version != "unknown"
        and built.toolchain_version != detected.toolchain_version
    ):
        _LOG.warning(
            "This engine was built by %s and this machine has %s. The major.minor "
            "match, so it should deserialize, but TensorRT does not guarantee it "
            "across builds - rebuild for this device if the load fails.",
            built.toolchain_version,
            detected.toolchain_version,
        )


class TensorRTSession(RuntimeSession):
    """A deserialized TensorRT plan, behind the session interface.

    Device buffers are torch CUDA tensors rather than a pycuda allocation: torch is
    already an SDK extra, its caching allocator makes the per-call buffer reuse free,
    and it keeps this module from owning a second CUDA memory story.
    """

    def __init__(
        self, weights: Path, *, target: EngineTarget | None, index: int | None = None
    ) -> None:
        trt = _require_tensorrt()
        torch = _require_torch()

        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA device is available, so a TensorRT engine cannot be run "
                "here. TensorRT is NVIDIA-only; use this model's .onnx or .pte "
                "artifact on this machine instead."
            )
        count = int(torch.cuda.device_count())
        if index is not None and index >= count:
            raise ValueError(
                f"device='cuda:{index}' was requested, but torch reports {count} CUDA "
                f"device{'s' if count != 1 else ''} "
                f"(valid: cuda:0{f'-cuda:{count - 1}' if count > 1 else ''})."
            )
        # Everything below - deserialization, the execution context, the stream and
        # every buffer - must land on the SAME GPU, so the whole construction runs
        # inside one device scope rather than trusting the ambient current device.
        self._device = f"cuda:{index}" if index is not None else "cuda"

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with torch.cuda.device(self._device):
            try:
                engine = runtime.deserialize_cuda_engine(weights.read_bytes())
            except Exception as exc:
                raise RuntimeError(_deserialize_failure(weights, target, exc)) from exc
            if engine is None:
                raise RuntimeError(_deserialize_failure(weights, target, None))

            self._trt = trt
            self._torch = torch
            self._engine = engine
            self._context = engine.create_execution_context()
            self._stream = torch.cuda.Stream()
        self.target = target

        inputs: list[NodeArg] = []
        outputs: list[NodeArg] = []
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = [int(d) for d in engine.get_tensor_shape(name)]
            arg = NodeArg(name, shape)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(arg)
                self._input_names.append(name)
            else:
                outputs.append(arg)
                self._output_names.append(name)
        super().__init__(inputs=inputs, outputs=outputs)
        self.providers = [
            f"TensorRT {target.trt_version}" if target else "TensorRT",
            target.sm if target else "cuda",
        ]

    def _forward(self, tensor: Any) -> list[Any]:
        import numpy as np

        if self._context is None:
            raise RuntimeError("This TensorRT engine has been closed.")
        torch = self._torch
        trt = self._trt

        name = self._input_names[0]
        dtype = trt.nptype(self._engine.get_tensor_dtype(name))
        device_in = torch.from_numpy(np.ascontiguousarray(tensor, dtype=dtype)).to(self._device)
        self._context.set_input_shape(name, tuple(int(d) for d in device_in.shape))
        self._context.set_tensor_address(name, int(device_in.data_ptr()))

        held: list[Any] = [device_in]
        results: list[Any] = []
        for out_name in self._output_names:
            shape = tuple(int(d) for d in self._context.get_tensor_shape(out_name))
            out_dtype = trt.nptype(self._engine.get_tensor_dtype(out_name))
            buffer = torch.empty(
                shape, dtype=_torch_dtype(torch, np.dtype(out_dtype)), device=self._device
            )
            held.append(buffer)
            self._context.set_tensor_address(out_name, int(buffer.data_ptr()))
            results.append(buffer)

        with torch.cuda.stream(self._stream):
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed for this input.")
        self._stream.synchronize()
        return [r.cpu().numpy() for r in results]

    def close(self) -> None:
        self._context = None
        self._engine = None


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the local env
        raise ImportError(_TORCH_HINT) from exc
    return torch


def _torch_dtype(torch: Any, np_dtype: Any) -> Any:
    """numpy dtype → torch dtype, for the output buffers TensorRT writes into."""
    import numpy as np

    table = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.bool_): torch.bool,
    }
    return table.get(np_dtype, torch.float32)


def _deserialize_failure(weights: Path, target: EngineTarget | None, exc: object) -> str:
    """The message for a plan that got past the pre-check and still would not load.

    Reaching here means the pre-check could not prove the mismatch - usually an
    unlabelled file, occasionally a corrupt download or a shape profile this input
    falls outside. Say which, rather than re-raising TensorRT's own text alone.
    """
    detected = detect_local_target()
    if target is not None:
        return engine_mismatch_message(
            built_target=target.sm,
            built_toolchain=target.toolchain_version,
            detected_target=detected.sm,
            detected_toolchain=detected.toolchain_version,
        ) + (f" (TensorRT reported: {exc})" if exc else "")
    return (
        f"TensorRT could not deserialize {weights.name!r} on this device "
        f"({detected}). The file carries no target metadata, so it could not be "
        f"checked first - it was most likely built for a different GPU architecture "
        f"or a different TensorRT version, or the download is incomplete."
        + (f" (TensorRT reported: {exc})" if exc else "")
    )


class TensorRTEngine(WrapperEngine):
    """Runs a TensorRT plan and emits raw annotation dicts."""

    backend = "tensorrt"


def build_tensorrt_engine(
    *,
    weights: Path,
    model_type: str,
    architecture: str,
    classes: list[str],
    input_shape: tuple[int, int],
    confidence: float,
    device: Device = "auto",
    keypoint_schema: dict[str, Any] | None = None,
    target: EngineTarget | None = None,
) -> TensorRTEngine:
    """Build a :class:`TensorRTEngine` from a prebuilt ``.engine`` + resolved config.

    Args:
        target: What the plan was built for. Defaults to reading it from the
            filename, then from a ``<stem>.json`` sidecar. Pass it explicitly when
            you have the manifest row and the file has been renamed.
        device: ``"auto"`` or ``"cuda"`` - a plan is compiled for one NVIDIA GPU and
            has no other form, so those are the only coherent values and anything
            else raises. ``"cuda:1"`` IS honoured: the index picks which GPU
            deserializes and runs the plan, which is a real choice on a multi-GPU box.
    """
    index = check_tensorrt_device(device)

    resolved = target or parse_engine_filename(weights) or _sidecar_target(weights)
    check_engine_compatibility(resolved, detect_local_target())

    session = TensorRTSession(weights, target=resolved, index=index)
    wrapper = build_wrapper_with_session(
        session,
        model_type=model_type,
        architecture=architecture,
        model_path=str(weights),
        classes=classes,
        input_shape=input_hw_from(session, input_shape),
        confidence_threshold=confidence,
        keypoint_schema=keypoint_schema,
        providers=session.providers,
        sess_options=None,
    )
    return TensorRTEngine(
        wrapper=wrapper,
        model_type=model_type,
        architecture=architecture,
        classes=classes,
        providers=session.providers,
        # The GPU that actually holds the plan, indexed when one was named.
        device=f"cuda:{index}" if index is not None else "cuda",
        session=session,
    )


def check_tensorrt_device(device: Device) -> int | None:
    """The GPU ordinal a ``.engine`` should run on, or a refusal.

    A plan has exactly one coherent hardware family, so this is the shortest of the
    four device gates: ``auto`` and ``cuda`` pass, an index passes through, and
    everything else is a contradiction worth naming rather than ignoring.

    Raises:
        ValueError: ``device`` names hardware a TensorRT plan cannot run on.
    """
    family, index = normalize_device(device)
    if family in ("auto", "cuda"):
        return index
    raise ValueError(
        f"device={device!r} was requested, but a TensorRT engine is compiled for a "
        f"specific NVIDIA GPU and has no {family} form at all - not a slower one, "
        f"none. Load this model's .onnx artifact (format='onnx') to run on {family}."
    )
