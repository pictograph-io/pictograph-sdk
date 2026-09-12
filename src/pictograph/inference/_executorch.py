"""The ExecuTorch inference engine - runs a ``.pte`` program.

ExecuTorch is PyTorch's edge runtime. A ``.pte`` is an **ahead-of-time lowered
program**: ``torch.export`` captures the graph, a partitioner hands eligible
subgraphs to a delegate backend (XNNPACK for portable CPU, CoreML on Apple, Vulkan,
QNN on Qualcomm), and the result is serialized with its weights inside. That is the
opposite of ONNX Runtime, where the provider is chosen when the session is built.

The one thing that follows from it, and the thing most likely to surprise a caller:
**a ``.pte`` cannot be re-targeted at load time.** A CoreML-lowered program does not
run on a Qualcomm NPU, and no ``device=`` value can make an XNNPACK program use the
GPU. You pick the accelerator by choosing which ``.pte`` you load - so here
``device=`` is a CHECK on the artifact you picked, and it raises rather than
pretending. The portable XNNPACK build is the default published artifact precisely
because it is the one that runs anywhere.

Everything except the forward pass is the shared vendored wrapper - see
:mod:`pictograph.inference._engine` for why, and for how the session is substituted.
So an ExecuTorch model letterboxes, normalizes, decodes and polygonizes through the
exact same code as the ONNX path, and a numerical difference between the two can
only come from the forward pass itself.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pictograph.inference._engine import (
    NodeArg,
    RuntimeSession,
    WrapperEngine,
    build_wrapper_with_session,
    input_hw_from,
)
from pictograph.inference.runtime import check_artifact_device

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pictograph.inference.runtime import Device

__all__ = [
    "ExecuTorchEngine",
    "ExecuTorchSession",
    "build_executorch_engine",
]

_LOG = logging.getLogger("pictograph.inference")

_INSTALL_HINT = (
    "Running a .pte program needs the ExecuTorch runtime. Install it with:\n"
    '    pip install "pictograph[inference,executorch]"\n'
    "ExecuTorch's prebuilt runtime is compiled against a specific torch release; if "
    "the import succeeds but loading a program raises a missing-symbol error, install "
    "the torch version that ExecuTorch's metadata pins."
)

# Delegate-backend ids, as ExecuTorch serializes them into the .pte flatbuffer, mapped
# to the device they actually execute on. Read by `_delegates` - see its docstring for
# why the file is scanned rather than queried.
_DELEGATE_DEVICE: dict[str, str] = {
    "XnnpackBackend": "cpu",
    "CoreMLBackend": "coreml",
    "MPSBackend": "mps",
    "VulkanBackend": "vulkan",
    "QnnBackend": "qnn",
    "NeuropilotBackend": "neuropilot",
    "ArmBackend": "cpu",
    "CadenceBackend": "cpu",
    "VelaBackend": "cpu",
    "OpenvinoBackend": "openvino",
}

# The .pte flatbuffer is read whole for the delegate scan. Programs are weights-inside
# and can be large, so the scan is capped - delegate ids live in the schema tables near
# the head, not after the weight blobs, so a bounded prefix is enough in practice and a
# miss degrades to "portable", never to an error.
_DELEGATE_SCAN_BYTES = 1 << 20


def _require_executorch() -> Any:
    """Import the ExecuTorch runtime, or raise with the exact install command."""
    try:
        from executorch.runtime import Runtime
    except ImportError as exc:  # pragma: no cover - depends on the local env
        raise ImportError(_INSTALL_HINT) from exc
    return Runtime


def _delegates(weights: Path) -> list[str]:
    """The delegate backends a ``.pte`` was lowered to, in file order.

    ExecuTorch's public ``Runtime`` API exposes methods and their tensor metadata but
    not the program's delegate list, so this reads the ids straight out of the
    flatbuffer. They are stored as plain strings (``XnnpackBackend``,
    ``CoreMLBackend``, …), which makes a bounded byte scan both reliable and cheap.

    Purely informational - it drives ``.providers`` and ``.device`` so a caller can
    see what a program was built for. A miss returns ``[]`` and the model still runs.
    """
    try:
        with Path(weights).open("rb") as handle:
            blob = handle.read(_DELEGATE_SCAN_BYTES)
    except OSError as exc:  # pragma: no cover - defensive
        _LOG.debug("Could not scan %s for delegate ids (%s).", weights, exc)
        return []
    found = [name for name in _DELEGATE_DEVICE if name.encode("ascii") in blob]
    return sorted(found)


def _device_for(delegates: list[str]) -> str:
    """The device a lowered program actually executes on.

    A program may carry several delegates (a CoreML-partitioned graph still runs its
    unpartitioned ops on CPU). The first non-CPU delegate is the interesting one, for
    the same reason :func:`~pictograph.inference.runtime.device_label` prefers a
    non-CPU provider: reporting ``cpu`` for a CoreML program would hide the accelerator.
    """
    for name in delegates:
        device = _DELEGATE_DEVICE.get(name, "cpu")
        if device != "cpu":
            return device
    return "cpu"


class ExecuTorchSession(RuntimeSession):
    """An ExecuTorch ``forward`` method, behind the session interface.

    Holds the loaded program for its whole life: ExecuTorch memory-plans at load, so
    re-loading per call would pay the planning cost on every image.
    """

    def __init__(self, weights: Path, *, providers: list[str]) -> None:
        runtime_cls = _require_executorch()
        self._runtime = runtime_cls.get()
        try:
            self._program = self._runtime.load_program(str(weights))
            self._method = self._program.load_method("forward")
        except Exception as exc:
            raise RuntimeError(
                f"ExecuTorch could not load {Path(weights).name!r} as a program. "
                f"A .pte is produced by torch.export -> to_edge -> to_executorch; a "
                f"file that is not one, or one exported by a newer ExecuTorch than the "
                f"installed runtime, fails here. Underlying error: {exc}"
            ) from exc

        meta = self._method.metadata
        inputs = [
            NodeArg("input" if i == 0 else f"input_{i}", _sizes(meta, "input", i))
            for i in range(_count(meta, "input"))
        ] or [NodeArg("input", [])]
        outputs = [
            NodeArg(f"output_{i}", _sizes(meta, "output", i)) for i in range(_count(meta, "output"))
        ] or [NodeArg("output_0", [])]
        super().__init__(inputs=inputs, outputs=outputs)
        self.providers = providers

    def _forward(self, tensor: Any) -> list[Any]:
        import numpy as np
        import torch

        if self._method is None:
            raise RuntimeError("This ExecuTorch program has been closed.")
        # `.execute` takes torch tensors. The wrappers hand over float32 NCHW numpy,
        # which shares memory with the tensor, so this is a view rather than a copy.
        out = self._method.execute([torch.from_numpy(np.ascontiguousarray(tensor))])
        if isinstance(out, (list, tuple)):
            return [_to_numpy(o) for o in out]
        return [_to_numpy(out)]

    def close(self) -> None:
        self._method = None
        self._program = None


def _to_numpy(value: Any) -> Any:
    """One ExecuTorch output → a numpy array the wrappers can postprocess."""
    import numpy as np

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _count(meta: Any, kind: str) -> int:
    """How many inputs/outputs the method declares, across ExecuTorch API spellings."""
    for attr in (f"num_{kind}s", f"num_{kind}"):
        getter = getattr(meta, attr, None)
        if getter is None:
            continue
        try:
            return int(getter() if callable(getter) else getter)
        except Exception as exc:  # pragma: no cover - depends on the ExecuTorch version
            _LOG.debug("ExecuTorch metadata attr %r was unreadable (%s).", attr, exc)
    return 0


def _sizes(meta: Any, kind: str, index: int) -> list[int]:
    """One input/output's static shape, or ``[]`` when the runtime does not report it.

    Only the two RF-DETR wrappers read a shape off the session (to learn the graph's
    trained resolution), and both fall back to their configured ``input_shape`` when
    it is not a concrete list - so an empty shape is safe, not fatal.
    """
    getter = getattr(meta, f"{kind}_tensor_meta", None)
    if getter is None:
        return []
    try:
        return [int(v) for v in getter(index).sizes()]
    except Exception:  # pragma: no cover - depends on the ExecuTorch version
        return []


class ExecuTorchEngine(WrapperEngine):
    """Runs an ExecuTorch program and emits raw annotation dicts."""

    backend = "executorch"


def build_executorch_engine(
    *,
    weights: Path,
    model_type: str,
    architecture: str,
    classes: list[str],
    input_shape: tuple[int, int],
    confidence: float,
    device: Device = "auto",
    keypoint_schema: dict[str, Any] | None = None,
) -> ExecuTorchEngine:
    """Build an :class:`ExecuTorchEngine` from a ``.pte`` + resolved config.

    ``device`` is a CHECK here, not a selector, and that is the whole story of this
    runtime: a ``.pte``'s delegate was chosen when it was BUILT, so the only honest
    answers to a caller who names one are "yes, that is what this program runs on"
    and an exception. ``device="auto"`` accepts whatever the program was lowered to.

    The refusal matters most in the direction it originally shipped for: ``cpu`` is a
    REPRODUCIBILITY request (CI, numerical-parity work), and quietly handing back a
    CoreML-lowered program would defeat it while looking like success.
    """
    delegates = _delegates(weights)
    actual = _device_for(delegates)
    check_artifact_device(
        device,
        actual,
        artifact=(
            f"{Path(weights).name!r} is an ExecuTorch program lowered to the "
            f"{delegates[0] if delegates else 'portable'!r} delegate, so it"
        ),
        remedy=(
            f"Load the {'portable (XNNPACK)' if actual != 'cpu' else 'accelerated'} "
            f".pte for this model instead, or pass device='auto' to run the one you have."
        ),
    )

    providers = delegates or ["portable"]
    session = ExecuTorchSession(weights, providers=providers)
    wrapper = build_wrapper_with_session(
        session,
        model_type=model_type,
        architecture=architecture,
        model_path=str(weights),
        classes=classes,
        input_shape=input_hw_from(session, input_shape),
        confidence_threshold=confidence,
        keypoint_schema=keypoint_schema,
        providers=providers,
        sess_options=None,
    )
    return ExecuTorchEngine(
        wrapper=wrapper,
        model_type=model_type,
        architecture=architecture,
        classes=classes,
        providers=providers,
        # What RAN - the program's own delegate device, never the request. A caller
        # who passed device='mps' and got a CoreML program reads 'coreml' here, which
        # is the more specific truth, not a contradiction.
        device=actual,
        session=session,
    )
