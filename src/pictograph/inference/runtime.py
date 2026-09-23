"""Where a local model actually runs - execution providers, devices, session tuning.

ONE policy for every engine. The ONNX engine resolves an ordered execution-provider
ladder; the PyTorch engine resolves a ``torch`` device; ExecuTorch and TensorRT read
the target their artifact was BUILT for. All of them report back what they actually
got, so a silent fall back to CPU is visible rather than mysterious::

    model = get_model("Detector", task="object_detection")
    model.backend  # 'pytorch' | 'executorch' | 'onnxruntime' | 'tensorrt'
    model.device  # 'coreml' | 'cuda' | 'mps' | 'cpu'  - what RAN, not what was asked
    model.providers  # the runtime's own execution targets, as IT resolved them

The four runtimes
-----------------
:data:`RUNTIMES` is the ordered vocabulary - ``pytorch``, ``executorch``,
``onnxruntime``, ``tensorrt`` - and that order is the product's, not alphabetical
or historical: **native source → PyTorch-family edge runtime → cross-platform
runtime → NVIDIA-only AOT-compiled engine**, i.e. increasing specialisation and
decreasing portability, with the artifact that is valid for exactly one GPU
architecture last. The UI's Runtime row renders this order; do not re-sort it.

**A runtime is never asked for - it is derived.** The loaders take
:data:`WeightFormat` (``format="onnx"`` / ``"safetensors"`` / ``"pytorch"`` /
``"pytorch_engine"`` / ``"tensorrt_engine"``) and resolve the runtime from it via
:func:`runtime_for_format`, because a ``.engine`` is executed by TensorRT and by
nothing else. Two selectors would be two sources of truth that can disagree; one
cannot. ``Runtime`` remains the vocabulary of what RAN - the value
``model.backend`` reports.

Each names the artifact it executes and how it reports itself:

=================  =====================  ==================  =========================
``.backend``       artifact               ``.device``         ``.providers``
=================  =====================  ==================  =========================
``pytorch``        ``.pth`` / safetensors  cuda / mps / cpu   ``[]`` - torch has none
``executorch``     ``.pte``                cpu / coreml       delegate backends
``onnxruntime``    ``.onnx``               what ORT kept      ORT providers, in order
``tensorrt``       ``.engine``             cuda               the engine's built-for key
=================  =====================  ==================  =========================

One argument picks the hardware: ``device=``
--------------------------------------------
:data:`Device` is the SINGLE user-facing hardware selector, identical on both
loaders and meaning the same thing on all four runtimes. It names the HARDWARE;
each runtime is responsible for reaching it whichever way it actually can:

===============  ==========================  ================================
``device=``      pytorch                     onnxruntime
===============  ==========================  ================================
``auto``         cuda → mps → cpu            the measured ladder below
``cpu``          ``cpu``                     ``CPUExecutionProvider``
``cuda``         ``cuda``                    ``CUDAExecutionProvider``
``cuda:1``       ``cuda:1``                  … + ``device_id=1``
``mps``          ``mps``                     ``CoreMLExecutionProvider``
===============  ==========================  ================================

===============  ==========================  ================================
``device=``      executorch                  tensorrt
===============  ==========================  ================================
``auto``         whatever the ``.pte`` was   ``cuda`` - the only thing a plan
                 lowered to                  can run on
``cpu``          only an XNNPACK-class       raises: a plan is compiled for
                 program; else raises        one GPU and has no CPU form
``cuda[:N]``     raises (no CUDA delegate    ``cuda`` / that GPU
                 is published)
``mps``          only a CoreML/MPS-lowered   raises: NVIDIA-only
                 program; else raises
===============  ==========================  ================================

Two rules make that table safe to rely on, and they are the whole reason this is
one argument rather than three:

1. **A device that cannot be honoured RAISES, naming what IS available.** Never a
   silent fall back to CPU. The failure this exists to kill: ask for CUDA, hit a
   driver mismatch, get CPU speed and no signal at all.
2. **The two AOT runtimes have their target fixed when the ARTIFACT is built**, so
   ``device=`` there is a CHECK, not a selector - it either matches the program and
   is honoured, or it says so. Pick the accelerator by choosing which ``.pte`` you
   load, and ``device=`` will confirm you got it.

``device="auto"`` policy (ONNX Runtime)
---------------------------------------
``auto`` (the default) turns on the platform's accelerator when it is a measured
win and the session build cost is modest, and stays on CPU when it is not. It is
deliberately not "always use the GPU": on this platform CoreML is between a 3.8x
win and a 1.5x LOSS depending on the architecture. **Naming a device explicitly
overrides every one of those judgement calls** - if you ask for it, you get it or
you get an exception.

Measured on an M-series Mac (macOS 15.5, onnxruntime 1.26.0, median of 20 runs
after 5 warmups, batch 1) against real trained Pictograph models. TWO columns,
because they answer different questions and conflating them is misleading:

- **fwd** is the raw ``session.run()`` forward pass. This is what the provider
  choice actually changes, so it is what this policy table is derived from.
- **e2e** is what ``model.predict()`` returns to a caller - forward pass PLUS
  decode and postprocess.

====================  ========  ========  ===========  ==============  ====================
model                 CPU fwd   CPU e2e   CoreML-NN    CoreML-MLProg   default
                                          fwd / e2e    fwd / e2e
====================  ========  ========  ===========  ==============  ====================
classifier @224        12.0 ms   13.1 ms  0.8 / 1.5    1.8 / 6.1       CoreML NeuralNetwork
semantic-seg @512     463.7 ms  751.8 ms  21 / 337     40 / 378        CoreML NeuralNetwork
YOLOX @640             91.4 ms   96.0 ms  26.8 / 22.6  14.6 / 23.5     CoreML MLProgram
RF-DETR @384           85.9 ms   91.2 ms  132 / 143    22.3 / 28.4     CoreML MLProgram
====================  ========  ========  ===========  ==============  ====================

Read the semantic-seg row carefully before quoting a speedup: its postprocess is a
per-class ``cv2`` contour extraction over up to 80 classes that costs a roughly
CONSTANT ~300 ms whatever ran the forward pass (verified against both noise and
smooth-block images, so it is the fixed per-class loop, not mask fragmentation).
CoreML's advantage there is therefore ~2.2x end-to-end, NOT the ~22x the forward
pass alone suggests. Optimizing that loop would buy more than any provider change.

On NVIDIA (Tesla T4, onnxruntime-gpu 1.22, torch 2.7+cu126, CUDA 12.6), same
methodology, end-to-end ``predict()``:

====================  ==========  ==========  ==========  ==========
model                 onnx-cpu    onnx-cuda   torch-cpu   torch-cuda
====================  ==========  ==========  ==========  ==========
classifier @224        24.8 ms      6.1 ms     76.1 ms     16.6 ms
YOLOX @640            176.5 ms     24.7 ms    417.1 ms     42.4 ms
Unet++ semseg @512   2589.6 ms   2316.9 ms   3569.6 ms   1348.4 ms
====================  ==========  ==========  ==========  ==========

CPU->CUDA is 4.1-9.8x for classification and detection, but only **1.12x** for
semantic segmentation on ONNX - the same fixed per-class contour postprocess that
caps the CoreML win caps the CUDA one, and it dominates harder at higher
resolution. Do not quote a GPU speedup for segmentation from the forward pass.

CPU and CUDA agree numerically to float noise, verified per family: classifier
|dconf| 1.7e-09, YOLOX max |dconf| 6.4e-07 / max |dbox| 1.7e-04 px, semantic-seg
identical prediction set with confidences differing in the 6th decimal.

Three traps this table exists to encode, all of which a naive "enable CoreML"
would walk straight into:

1. **NeuralNetwork is SLOWER than CPU on RF-DETR** (143 ms vs 91 ms end-to-end).
   Turning CoreML on without picking the format per architecture makes it worse.
2. **MLProgram's session build is not free** - 0.3-0.6 s for most graphs but a
   measured 31.7 s for RF-DETR. :data:`COREML_CACHE_SUBDIR` keeps ORT's compiled
   artifact so the cost is paid once per machine per model (measured: 31.7 s cold,
   5.8 s warm, ~426 MiB on disk). This is why ``auto`` skips CoreML for the
   slow-building architectures - and why ``device="mps"`` pays it without asking:
   a 31 s surprise is unacceptable when we chose, and expected when you did.
3. **NeuralNetwork's build cost also varies by architecture** (0.46 s → 4.36 s),
   so a cheap-to-build provider is not automatically a cheap-to-build model.

Re-measure with ``python -m benchmarks.inference_bench`` and update the table
above together with :data:`_COREML_FORMAT` - they are one decision recorded twice.
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "DEVICES",
    "DEVICES_BY_RUNTIME",
    "RUNTIMES",
    "WEIGHT_FORMATS",
    "Device",
    "Runtime",
    "WeightFormat",
    "check_artifact_device",
    "check_device_honoured",
    "check_device_supported",
    "device_label",
    "format_for_weights",
    "is_explicit",
    "normalize_device",
    "resolve_providers",
    "resolve_torch_device",
    "runtime_for_format",
    "runtime_for_weights",
    "session_options",
    "staleness_blocks_load",
    "suffix_for_format",
    "wire_format",
]

_LOG = logging.getLogger("pictograph.inference")

Runtime = Literal["pytorch", "executorch", "onnxruntime", "tensorrt"]
"""Which runtime executes the model - the value :attr:`InferenceModel.backend` reports.

One stable, lowercase token per runtime. It names the RUNTIME, not the file
extension, because that is the thing a caller chooses and the UI renders.
"""

RUNTIMES: tuple[Runtime, ...] = ("pytorch", "executorch", "onnxruntime", "tensorrt")
"""The four runtimes, in the order the product presents them. See the module docstring."""


WeightFormat = Literal["pytorch", "safetensors", "pytorch_engine", "onnx", "tensorrt_engine"]
"""Which WEIGHT FILE to load - the one thing a caller actually chooses.

A runtime is not a choice, it is a consequence: a ``.engine`` is executed by
TensorRT and by nothing else. So the loaders take ``format=`` and DERIVE the
runtime from it (:func:`runtime_for_format`), rather than asking for both and
having to reconcile two answers that can disagree.

============================  ==================  ===================================
``format=``                   file                runtime that executes it
============================  ==================  ===================================
``pytorch``                   ``.pth``            ``pytorch`` - the live ``nn.Module``
``safetensors``               ``.safetensors``    ``pytorch`` - same module, gated container
``pytorch_engine``            ``.pte``            ``executorch`` - PyTorch's edge runtime
``onnx``                      ``.onnx``           ``onnxruntime`` - the default
``tensorrt_engine``           ``.engine``         ``tensorrt`` - NVIDIA, AOT-compiled
============================  ==================  ===================================

That order is the product's, matching :data:`RUNTIMES`: native source → the two
PyTorch-family containers → PyTorch's own edge engine → cross-platform runtime →
NVIDIA-only compiled plan, i.e. increasing specialisation and decreasing
portability. Do not re-sort it.

Two formats map onto the ``pytorch`` runtime because a checkpoint has two
containers for the SAME tensors: ``.pth`` is what the training loop wrote, and
``model.safetensors`` is what a publish-BLOCKING parity gate signed off against
that version's ONNX graph. They are a genuine choice, so they are two values -
and asking for one never silently hands back the other.
"""

WEIGHT_FORMATS: tuple[WeightFormat, ...] = (
    "pytorch",
    "safetensors",
    "pytorch_engine",
    "onnx",
    "tensorrt_engine",
)
"""The five loadable weight formats, in the order the product presents them."""

# format → the runtime that executes it. The ONE place this is decided. Keyed by
# `str`, not `WeightFormat`, because it is also the RUNTIME VALIDATOR: a caller
# without a type checker can pass anything, and this table is what turns that into
# a message naming the five rather than a KeyError three frames down.
_RUNTIME_BY_FORMAT: dict[str, Runtime] = {
    "pytorch": "pytorch",
    "safetensors": "pytorch",
    "pytorch_engine": "executorch",
    "onnx": "onnxruntime",
    "tensorrt_engine": "tensorrt",
}

# format → the suffix its file carries.
_SUFFIX_BY_FORMAT: dict[WeightFormat, str] = {
    "pytorch": ".pth",
    "safetensors": ".safetensors",
    "pytorch_engine": ".pte",
    "onnx": ".onnx",
    "tensorrt_engine": ".engine",
}

# format → the token the platform's `/models/{id}/download?format=` route takes.
# Only two differ, and deliberately: the SDK names the FAMILY a caller reasons in
# ("PyTorch's engine", "TensorRT's engine"), the wire names the FILE (`pte`,
# `engine`). Translating in one table beats leaking the wire spelling into every
# call site - and beats renaming either vocabulary to match the other.
_WIRE_BY_FORMAT: dict[WeightFormat, str] = {
    "pytorch": "pytorch",
    "safetensors": "safetensors",
    "pytorch_engine": "pte",
    "onnx": "onnx",
    "tensorrt_engine": "engine",
}


def runtime_for_format(fmt: str) -> Runtime:
    """The runtime that executes ``fmt``.

    Raises:
        ValueError: ``fmt`` is not one of :data:`WEIGHT_FORMATS`. The message
            lists them, since the value is almost always a typo or a runtime
            name used where a format belongs.
    """
    resolved = _RUNTIME_BY_FORMAT.get(fmt)
    if resolved is None:
        raise ValueError(
            f"Unknown format {fmt!r}. This SDK loads {', '.join(WEIGHT_FORMATS)}. "
            f"(These name the WEIGHTS FILE, not the runtime - the runtime follows "
            f"from it: {', '.join(RUNTIMES)}.)"
        )
    return resolved


def suffix_for_format(fmt: WeightFormat) -> str:
    """The file suffix ``fmt``'s artifact carries (``"pytorch_engine"`` → ``".pte"``)."""
    return _SUFFIX_BY_FORMAT[fmt]


def wire_format(fmt: WeightFormat) -> str:
    """The ``format=`` token the platform's download route takes for ``fmt``."""
    return _WIRE_BY_FORMAT[fmt]


def staleness_blocks_load(runtime: str) -> bool:
    """Whether a STALE artifact of this runtime must be refused rather than flagged.

     Must stay in sync with the server's model-artifact contract - same name, same
    rule. The SDK ships as its own wheel and cannot import the backend, so the
    predicate is mirrored; if one changes, both change.

    An artifact is *stale* when the platform's pinned toolchain for its runtime has
    moved on (the files manifest computes this and reports it as ``stale``). The
    three graph runtimes differ genuinely in what that costs, and flattening them
    would either withdraw two working artifact classes or serve one unloadable plan:

    - **tensorrt - BLOCKING.** TensorRT refuses to deserialize a plan built by a
      different minor version, so a stale engine is not a degraded artifact, it is an
      unusable one. Refuse first, with a message; never let it crash.
    - **executorch - ADVISORY.** The ``.pte`` schema is versioned and a newer runtime
      loads an older program. It still works; offer a rebuild.
    - **onnxruntime - ADVISORY.** A graph exported by an older torch still loads under
      a newer ORT.

    Note this is the *platform's* staleness. Whether a TensorRT plan will actually
    load HERE is decided by the TensorRT installed on this machine, which is what
    :mod:`pictograph.inference._tensorrt` checks at load time - see its docstring for
    why the two notions are deliberately kept apart.
    """
    return runtime == "tensorrt"


# Weights suffix → the format it holds. This is what lets `load_model` stay a
# two-argument call: the artifact already says what it is, so `format=` is an
# OVERRIDE for a renamed file rather than a second thing to keep in step.
# `.plan` is TensorRT's other conventional extension for the same serialized plan.
_FORMAT_BY_SUFFIX: dict[str, WeightFormat] = {
    ".onnx": "onnx",
    ".pte": "pytorch_engine",
    ".engine": "tensorrt_engine",
    ".plan": "tensorrt_engine",
    ".pth": "pytorch",
    ".safetensors": "safetensors",
}


def format_for_weights(weights: Path) -> WeightFormat:
    """The format ``weights`` holds, from its suffix.

    Every one of the five formats is recognised here, including the two native
    containers - a ``.pth`` IS the ``pytorch`` format, and answering "unknown" for
    it would be false. Whether a given ENTRY POINT can execute that format is a
    separate question, asked separately, so its refusal can be specific.

    Raises:
        ValueError: The suffix names none of the five. The message lists them,
            because the usual cause is a renamed file or a directory.
    """
    suffix = Path(weights).suffix.lower()
    resolved = _FORMAT_BY_SUFFIX.get(suffix)
    if resolved is None:
        known = ", ".join(sorted(_FORMAT_BY_SUFFIX))
        extra = (
            " (A training checkpoint is conventionally '.pth' here.)"
            if suffix in (".pt", ".ckpt", ".bin")
            else ""
        )
        raise ValueError(
            f"Cannot tell what {Path(weights).name!r} is: the suffix "
            f"{suffix or '(none)'!r} names none of {known}.{extra}"
        )
    return resolved


def runtime_for_weights(weights: Path) -> Runtime:
    """The runtime that executes ``weights``, from its suffix.

    Composed of the two decisions rather than a third table: the suffix says which
    FORMAT the file holds, and the format says which runtime runs it.
    """
    return runtime_for_format(format_for_weights(weights))


Device = str
"""Which HARDWARE to run on - the SINGLE hardware selector, on both loaders.

The four values the docs teach are :data:`DEVICES`:

``auto``
    Default. The best hardware this machine has, per the measured ladder above,
    without paying a session build that would read as a hang. Never raises for
    want of an accelerator - it is the value that means "whatever you've got".
``cpu``
    The CPU, and nothing else. Reproducible and instant to load - the right choice
    for CI, for numerical-parity work, and when a single image is all you will ever
    run. (This is what ``accelerate="off"`` used to spell.)
``cuda`` / ``cuda:1``
    An NVIDIA GPU; the indexed form picks one on a multi-GPU box. Honoured by every
    runtime that can reach CUDA at all, and an error from the ones that cannot.
``mps``
    Apple's accelerator. ``torch`` reaches it as Metal Performance Shaders, ONNX
    Runtime and ExecuTorch reach the same silicon through CoreML - one name, and
    each runtime maps it to its own mechanism. Asking for it explicitly ALSO buys
    what ``accelerate="max"`` used to: a slow-to-build CoreML session (measured
    31.7 s cold for RF-DETR) is paid rather than skipped, because you named it.

Any label :attr:`InferenceModel.device` can REPORT is accepted too - ``coreml``,
``rocm``, ``dml``, and the ExecuTorch delegate devices - so ``device=model.device``
always round-trips. ``coreml`` is the same hardware as ``mps`` and normalizes to it;
the rest stand for themselves.

Deliberately NOT a value: anything naming a latency/build-cost tradeoff rather than
a piece of hardware. ``accelerate="max"`` was exactly that, and reading it in a list
next to ``cpu`` and ``mps`` is what made three overlapping arguments feel like one
confused one. Its behaviour survives under ``device="mps"``; its name does not.
"""

DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda", "mps")
"""The four device values the docs teach, in the order they are presented."""

# Also accepted, but not taught: every label `model.device` can REPORT. Accepting
# them is what makes `device=model.device` a total round-trip rather than a trap on
# the platforms where a runtime's mechanism name differs from the hardware's.
#  Superset of `_executorch._DELEGATE_DEVICE`'s values and of `_DEVICE_BY_PROVIDER`'s
# - `test_inference_runtime.py` asserts that, so a new delegate cannot become a
# device this function then rejects.
_ALSO_ACCEPTED: tuple[str, ...] = (
    "coreml",
    "rocm",
    "dml",
    "vulkan",
    "qnn",
    "neuropilot",
    "openvino",
)

# Two names for one piece of silicon. `mps` is torch's, `coreml` is ORT's and
# ExecuTorch's; a user asking for one and being refused because the runtime spells
# it the other way would be the exact silent-mismatch this argument exists to end.
_SAME_HARDWARE: dict[str, str] = {"coreml": "mps"}


DEVICES_BY_RUNTIME: dict[Runtime, tuple[str, ...]] = {
    "pytorch": ("auto", "cpu", "cuda", "mps"),
    "onnxruntime": ("auto", "cpu", "cuda", "mps", "rocm", "dml"),
    "executorch": ("auto", "cpu", "mps", "vulkan", "qnn", "neuropilot", "openvino"),
    "tensorrt": ("auto", "cuda"),
}
"""Which hardware each runtime can reach - THE ``(device, format)`` table.

The user names two things and the SDK derives the third: ``format=`` says which
WEIGHTS, ``device=`` says which HARDWARE, and the execution provider falls out of
the pair. Nobody names a provider.

===============  ==================================================
``(device, format)``                pairs onto
================================    ==================================
``cpu`` + anything                  the CPU
``cuda`` + ``onnx``                 ``CUDAExecutionProvider``
``cuda`` + ``tensorrt_engine``      the TensorRT runtime (that plan's GPU)
``cuda`` + ``pytorch``              torch's ``cuda`` device
``mps`` + ``onnx``                  ``CoreMLExecutionProvider``
``mps`` + ``pytorch``               torch's ``mps`` device
``mps`` + ``pytorch_engine``        the program's CoreML/MPS delegate
``auto`` + anything                 search that format's ladder, best available
================================    ==================================

Reading it as a table rather than a chain of special cases is what makes the two
empty cells honest: a ``.engine`` has no CPU form and a ``.pte`` has no CUDA
lowering, so those pairs RAISE - naming what the format CAN run on - instead of
quietly running somewhere else.

The rows are supersets: this says the pairing is *coherent*, not that the hardware
is present. Whether this machine actually has it is the engines' question, asked
separately so the refusal can name what it found.
"""


def check_device_supported(device: Device, runtime: Runtime, fmt: str) -> None:
    """Raise when a weights format cannot run on the requested hardware at all.

    The FIRST of the two device gates, and the cheap one: it reads
    :data:`DEVICES_BY_RUNTIME` and fails before any runtime is imported, so
    ``load_model("sm75-trt10.13.3.9-fp16.engine", cfg, device="cpu")`` says what is
    wrong immediately rather than after a TensorRT import and a deserialization.

    The second gate lives in each engine and asks the harder question - is that
    hardware present, and did the session keep it.

    Raises:
        ValueError: This format has no form that runs on this device.
    """
    family, _ = normalize_device(device)
    supported = DEVICES_BY_RUNTIME[runtime]
    if family in supported:
        return
    runs_on = ", ".join(d for d in supported if d != "auto")
    raise ValueError(
        f"format={fmt!r} runs on {runtime}, which cannot use device={device!r}. "
        f"A {suffix_for_format(fmt) if fmt in _SUFFIX_BY_FORMAT else fmt} runs on: "
        f"{runs_on} (or device='auto'). "
        + _FORMAT_FOR_DEVICE.get((runtime, family), "Load a different format for that device.")
    )


# The concrete "load this instead" for the pairs that do not exist. A refusal that
# names the alternative artifact is the difference between a dead end and a redirect.
_FORMAT_FOR_DEVICE: dict[tuple[str, str], str] = {
    ("tensorrt", "cpu"): "Load this model's .onnx (format='onnx') to run on the CPU.",
    ("tensorrt", "mps"): "TensorRT is NVIDIA-only; load format='onnx' on Apple hardware.",
    ("executorch", "cuda"): (
        "No CUDA .pte lowering is published; load format='tensorrt_engine' for an "
        "NVIDIA GPU, or format='onnx' which runs on CUDA too."
    ),
}


def normalize_device(device: Device) -> tuple[str, int | None]:
    """``"cuda:1"`` → ``("cuda", 1)``; validate and canonicalize everything else.

    The ONE place a device string is parsed, so every runtime agrees on what a
    caller asked for before any of them decides whether it can be honoured.

    Returns:
        ``(family, index)``. ``family`` is canonical (``coreml`` normalizes to
        ``mps`` - see :data:`_SAME_HARDWARE`), and ``index`` is the GPU ordinal for
        the ``cuda:N`` form, else ``None``.

    Raises:
        ValueError: The value names no device this SDK knows. The message lists the
            taught vocabulary, since the cause is almost always a near-miss
            (``"gpu"``, ``"CUDA"``, or the removed ``"max"``).
    """
    raw = str(device).strip().lower()
    family, _, suffix = raw.partition(":")
    index: int | None = None
    if suffix:
        if family != "cuda" or not suffix.isdigit():
            raise ValueError(
                f"device={device!r} is not a device this SDK knows. Only 'cuda' takes "
                f"an index (device='cuda:1' picks the second GPU); "
                f"{', '.join(DEVICES)} are the rest."
            )
        index = int(suffix)
    family = _SAME_HARDWARE.get(family, family)
    if family not in DEVICES and family not in _ALSO_ACCEPTED:
        hint = ""
        if raw in ("max", "off", "on", "gpu"):
            hint = (
                " (There is no 'accelerate' argument any more - device='cpu' is what "
                "'off' meant, and device='mps' or 'cuda' is what 'max' meant.)"
                if raw in ("max", "off")
                else " (Name the vendor: device='cuda' or device='mps'.)"
            )
        raise ValueError(
            f"device={device!r} is not a device this SDK knows. Pass one of "
            f"{', '.join(DEVICES)} (or 'cuda:1' for a specific GPU).{hint}"
        )
    return family, index


def is_explicit(device: Device) -> bool:
    """Whether the caller NAMED hardware, as opposed to leaving it to ``auto``.

    The single predicate behind the SDK's no-silent-fallback rule: under ``auto``
    a runtime may degrade down its ladder and merely warn, and under a named device
    the same degradation must raise instead. Every engine asks this rather than
    re-deriving it from a string comparison.
    """
    return normalize_device(device)[0] != "auto"


COREML_CACHE_SUBDIR = "coreml-cache"
"""Compiled-CoreML cache, under the same root as the weights cache."""

# Per-architecture CoreML model format. Derived from the measurements in the module
# docstring; Must stay in sync with that table. Key is matched as a prefix against the
# lowercased architecture, falling back to the model task.
_COREML_FORMAT: dict[str, str] = {
    "yolox": "MLProgram",
    "rfdetr": "MLProgram",
    "rf-detr": "MLProgram",
    "object_detection": "MLProgram",
    "instance_segmentation": "MLProgram",
    "keypoint_detection": "MLProgram",
    "classification": "NeuralNetwork",
    "semantic_segmentation": "NeuralNetwork",
}

# Architectures whose CoreML session build is slow enough that `auto` keeps them on
# CPU and only `max` pays it. Measured: RF-DETR segmentation took 31.4 s cold.
_COREML_SLOW_BUILD = ("rfdetr", "rf-detr")

_CUDA = "CUDAExecutionProvider"
_TENSORRT = "TensorrtExecutionProvider"
_COREML = "CoreMLExecutionProvider"
_DIRECTML = "DmlExecutionProvider"
_ROCM = "ROCMExecutionProvider"
_CPU = "CPUExecutionProvider"

_DEVICE_BY_PROVIDER = {
    _TENSORRT: "cuda",
    _CUDA: "cuda",
    _ROCM: "rocm",
    _COREML: "coreml",
    _DIRECTML: "dml",
    _CPU: "cpu",
}


def _available() -> set[str]:
    import onnxruntime as ort

    return set(ort.get_available_providers())


def _coreml_format(architecture: str, model_type: str) -> str:
    arch = (architecture or "").lower()
    for key, fmt in _COREML_FORMAT.items():
        if arch.startswith(key):
            return fmt
    return _COREML_FORMAT.get(model_type, "NeuralNetwork")


def _coreml_is_slow_to_build(architecture: str) -> bool:
    arch = (architecture or "").lower()
    return arch.startswith(_COREML_SLOW_BUILD)


def resolve_providers(
    device: Device = "auto",
    *,
    architecture: str = "",
    model_type: str = "",
    cache_dir: Path | None = None,
    requested: Sequence[Any] | None = None,
) -> list[Any]:
    """The ordered execution-provider list to hand ONNX Runtime, for ``device``.

    This is the ONNX half of the one-argument mapping: a hardware name in, ORT's own
    vocabulary out. A named device resolves to that provider (plus CPU last, for the
    ops it cannot take) and RAISES if ORT does not have it; ``auto`` walks the
    measured ladder and never raises for want of an accelerator.

    CPU is ALWAYS last. That covers the common failure - a provider that fails to
    REGISTER (missing library, wrong driver) is dropped by ORT and the session still
    builds. It does NOT cover a provider that registers and then fails to COMPILE the
    model: ORT raises there, so `_onnx.build_onnx_engine` retries CPU-only on a build
    failure **under ``auto`` only**. MEASURED: CoreML's MLProgram compiler raises
    `Failed to create MLModel ... error code: -7` on the RF-DETR keypoint export.

    Args:
        device: See :data:`Device`.
        architecture: The model's architecture, used to pick the CoreML format
            per the measured table in this module's docstring.
        model_type: The task, used as the fallback key when architecture is blank.
        cache_dir: Root for the compiled-CoreML cache. Omitted → no cache, and
            every session pays the full build.
        requested: An explicit ORT provider list, passed straight through. **Not a
            public argument** - it is the escape hatch `benchmarks/inference_bench.py`
            uses to pin one exact provider configuration for measurement, which is how
            the tables in this module's docstring are produced. Accepts ORT's tuned
            form, ``[("CUDAExecutionProvider", {...}), "CPUExecutionProvider"]``.

    Returns:
        Providers in priority order, CPU last.

    Raises:
        ValueError: ``device`` names hardware ONNX Runtime cannot reach here. The
            message names the providers it DID find.
    """
    if requested is not None:
        return list(requested)

    family, index = normalize_device(device)
    if family == "cpu":
        return [_CPU]

    available = _available()
    if family != "auto":
        return _explicit_providers(family, index, available, architecture, model_type, cache_dir)

    chosen: list[Any] = []

    # NVIDIA. TensorRT is deliberately NOT in this ladder - the SDK's TensorRT story
    # is the first-class `format="tensorrt_engine"` path (an AOT-compiled plan), not
    # ORT's provider. Two measured reasons, both on a Tesla T4 (onnxruntime-gpu 1.22,
    # CUDA 12.6):
    #
    # 1. It is a LANDMINE here. `get_available_providers()` lists TensorRT whether or
    #    not libnvinfer is actually loadable, so there is no way to tell from that
    #    call alone. When it is listed but unloadable, ORT does not skip it and keep
    #    going - it DISCARDS THE WHOLE PROVIDER LIST and falls back to CPU-only.
    #    Measured: with it in the ladder, all three model families resolved to
    #    device='cpu' - the slowest option - on a stock `pip install onnxruntime-gpu`.
    # 2. It would not be worth it even when it loads. With tensorrt-cu12 installed,
    #    steady-state vs CUDA-only was 5.40 vs 6.08 ms (classifier), 22.95 vs
    #    24.68 ms (YOLOX), 2340 vs 2317 ms (semantic-seg - no gain), while the engine
    #    build costs 8-65 s on FIRST INFERENCE (not session build, so it surprises).
    #
    # DO NOT re-add it here. `format="tensorrt_engine"` gives the same hardware with
    # the build cost paid at publish time instead of on the user's first image.
    if _CUDA in available:
        chosen.append(_cuda_provider(None))

    # Apple. The lever is the model format, which is architecture-dependent (see the
    # docstring table).
    if _COREML in available and platform.system() == "Darwin":
        if not _coreml_is_slow_to_build(architecture):
            chosen.append(_coreml_provider(architecture, model_type, cache_dir))
        else:
            _LOG.debug(
                "CoreML skipped for architecture %r under device='auto' - its session "
                "build is slow (measured 31.7s cold for RF-DETR), which reads as a hang "
                "when nobody asked for it. Pass device='mps' to pay it deliberately.",
                architecture,
            )

    if _DIRECTML in available:
        chosen.append(_DIRECTML)
    if _ROCM in available:
        chosen.append(_ROCM)

    chosen.append(_CPU)
    return chosen


def _cuda_provider(index: int | None) -> tuple[str, dict[str, str]]:
    """ORT's CUDA provider, optionally pinned to one GPU.

    ``device_id`` is how ``device="cuda:1"`` reaches ONNX Runtime - the same request
    torch honours by name. Without it a multi-GPU box always lands on GPU 0 and a
    caller who asked for another one is silently ignored.
    """
    opts = {"cudnn_conv_algo_search": "HEURISTIC"}
    if index is not None:
        opts["device_id"] = str(index)
    return (_CUDA, opts)


def _coreml_provider(
    architecture: str, model_type: str, cache_dir: Path | None
) -> tuple[str, dict[str, str]]:
    """ORT's CoreML provider, with the per-architecture model format and its cache."""
    opts: dict[str, str] = {"ModelFormat": _coreml_format(architecture, model_type)}
    if cache_dir is not None:
        cache = Path(cache_dir) / COREML_CACHE_SUBDIR
        cache.mkdir(parents=True, exist_ok=True)
        opts["ModelCacheDirectory"] = str(cache)
    return (_COREML, opts)


# device family → the ORT provider that reaches it. `mps` maps to CoreML because that
# IS how ONNX Runtime reaches Apple silicon; ORT has no MPS provider of its own, and
# refusing `device="mps"` on that technicality would be the SDK failing to translate
# between two names for one chip - which is the whole job of this argument.
_PROVIDER_BY_DEVICE: dict[str, str] = {
    "cuda": _CUDA,
    "mps": _COREML,
    "rocm": _ROCM,
    "dml": _DIRECTML,
}


def _explicit_providers(
    family: str,
    index: int | None,
    available: set[str],
    architecture: str,
    model_type: str,
    cache_dir: Path | None,
) -> list[Any]:
    """The provider list for a NAMED device - or a refusal saying what ORT has.

    Two differences from the ``auto`` ladder, both deliberate:

    - **It raises.** A device the caller named and ORT cannot reach is an error, not
      a quiet demotion to CPU. The message lists ``get_available_providers()`` so the
      next step (install ``onnxruntime-gpu``, or use another device) is obvious.
    - **No cost heuristics.** ``auto`` skips CoreML for the slow-building
      architectures; a caller who NAMED ``mps`` gets it, 31.7 s build and all. That
      is the behaviour ``accelerate="max"`` used to select, reached now by naming the
      hardware instead of a tradeoff.

    CPU still trails the list - ORT needs a fallback for ops the accelerator cannot
    take, which is graph-level partitioning, not a device change. Whether the named
    provider actually survived the session build is checked afterwards, by
    :func:`check_device_honoured`.
    """
    provider = _PROVIDER_BY_DEVICE.get(family)
    if provider is None:
        raise ValueError(
            f"ONNX Runtime cannot target device={family!r}. It reaches "
            f"{', '.join(sorted(_PROVIDER_BY_DEVICE))} and the CPU; "
            f"{family!r} is an ExecuTorch delegate target, so load that model's .pte "
            f"with format='pytorch_engine' instead."
        )
    if provider not in available:
        raise ValueError(
            f"device={family!r} needs {provider}, which this ONNX Runtime install does "
            f"not have. It offers: {', '.join(sorted(available))}. "
            + _INSTALL_FOR_DEVICE.get(family, "")
            + " Or pass device='cpu' to run here anyway."
        )
    if provider is _COREML and platform.system() != "Darwin":
        raise ValueError(
            f"device={family!r} is Apple hardware and this is {platform.system()}. "
            f"Pass device='cuda' if this machine has an NVIDIA GPU, or device='cpu'."
        )

    if provider is _CUDA:
        return [_cuda_provider(index), _CPU]
    if provider is _COREML:
        return [_coreml_provider(architecture, model_type, cache_dir), _CPU]
    return [provider, _CPU]


# What to install to make a named device reachable. Kept beside the refusal that
# needs it, because a refusal that does not say how to fix it is half an error.
_INSTALL_FOR_DEVICE: dict[str, str] = {
    "cuda": "Install ONNX Runtime's CUDA build: `pip install onnxruntime-gpu`.",
    "rocm": "ROCm needs a ROCm-enabled onnxruntime build.",
    "dml": "DirectML needs a DirectML-enabled onnxruntime build, on Windows.",
}


def check_device_honoured(device: Device, resolved: Sequence[str]) -> None:
    """Raise when a NAMED device did not survive the session build.

    :func:`resolve_providers` proves ORT *has* the provider; this proves ORT *kept*
    it. They are different failures - a provider can register and then be dropped
    when the graph is compiled, or fail to load its shared library at session time -
    and only this one is visible after the fact.

    The failure it exists to kill, verbatim from the brief: the difference between
    "cuda was ignored" and "you have no CUDA". Under ``auto`` this is a warning
    (:func:`warn_on_fallback`), because degrading down a ladder nobody named is the
    ladder working. Under a named device it is an error.

    Raises:
        RuntimeError: The device was named and is not in ``resolved``.
    """
    family, _ = normalize_device(device)
    if family in ("auto", "cpu"):
        return
    wanted = _PROVIDER_BY_DEVICE.get(family)
    if wanted is None or wanted in resolved:
        return
    raise RuntimeError(
        f"device={device!r} was requested, but ONNX Runtime did not keep {wanted} - "
        f"it built the session on {', '.join(resolved) or 'nothing'} instead, which "
        f"would run at {device_label(resolved)!r} speed while reporting success. "
        f"This usually means the provider is installed but its driver/library "
        f"versions do not match. Fix the install, or pass device='cpu' (or 'auto') "
        f"to accept what this machine can actually do."
    )


def check_artifact_device(device: Device, actual: str, *, artifact: str, remedy: str) -> None:
    """Confirm a NAMED device against an artifact whose target is already fixed.

    The AOT half of the ``device=`` contract. A ``.pte``'s delegate and a ``.engine``'s
    GPU were chosen when the artifact was BUILT, so there is nothing to select at load
    time - which leaves exactly two honest options for a caller who names a device,
    and "ignore it" is not one of them. Either it matches what the artifact runs on
    and we confirm it, or it does not and we say so, naming both sides.

    ``auto`` always passes: it means "whatever this artifact runs on", which for an
    AOT artifact is the only answer there was.

    Args:
        device: What the caller asked for.
        actual: What the artifact actually runs on (its delegate device, or ``cuda``).
        artifact: How to name the artifact in the message (its filename).
        remedy: The concrete next step - which OTHER artifact to load instead.

    Raises:
        ValueError: The named device is not the one the artifact was built for.
    """
    family, _ = normalize_device(device)
    if family == "auto":
        return
    if _SAME_HARDWARE.get(actual, actual) == family:
        return
    raise ValueError(
        f"device={device!r} was requested, but {artifact} runs on {actual!r} - its "
        f"target was compiled in when the artifact was BUILT and cannot be changed at "
        f"load time. {remedy}"
    )


def device_label(resolved: Sequence[str]) -> str:
    """Map the providers ORT actually resolved to a short device name.

    ``resolved`` is ``session.get_providers()`` - the providers ORT KEPT, so a CUDA
    request that failed to load reports ``cpu`` here rather than lying.
    """
    for provider in resolved:
        label = _DEVICE_BY_PROVIDER.get(provider)
        if label is not None and label != "cpu":
            return label
    return "cpu"


def warn_on_fallback(requested: Sequence[Any] | None, resolved: Sequence[str]) -> None:
    """Warn when a non-CPU provider was asked for and ORT did not keep it.

    The failure this exists to make visible: install ``onnxruntime-gpu``, ask for
    CUDA, hit a driver/cuDNN mismatch, and get CPU speed with no signal at all.
    """
    if not requested:
        return
    wanted = {p if isinstance(p, str) else p[0] for p in requested} - {_CPU}
    missing = wanted - set(resolved)
    if missing:
        _LOG.warning(
            "Requested execution provider(s) %s were not loaded by ONNX Runtime - "
            "running on %s instead. Check that the matching runtime is installed "
            "(e.g. `pip install onnxruntime-gpu` for CUDA) and that its driver "
            "version matches.",
            sorted(missing),
            sorted(set(resolved)),
        )


def session_options(*, threads: int | None = None) -> Any:
    """A tuned :class:`onnxruntime.SessionOptions`.

    Two things a bare ``SessionOptions()`` gets wrong for SDK use:

    - ``intra_op_num_threads`` defaults to 0, which sizes ORT's pool from the HOST
      core count rather than the cgroup CPU limit. In a 2-vCPU container that
      over-subscribes badly. We honour ``PICTOGRAPH_INFERENCE_THREADS`` and
      otherwise leave ORT's default alone.
    - Graph optimization is left at its default level rather than ``ALL``.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    resolved = threads
    if resolved is None:
        raw = os.environ.get("PICTOGRAPH_INFERENCE_THREADS")
        if raw:
            try:
                resolved = int(raw)
            except ValueError:
                _LOG.debug("Ignoring non-integer PICTOGRAPH_INFERENCE_THREADS=%r", raw)
    if resolved and resolved > 0:
        opts.intra_op_num_threads = resolved
    return opts


def resolve_torch_device(device: Device = "auto") -> str:
    """The ``torch`` device string to load and predict on, for ``device``.

    The torch half of the one-argument mapping, and the twin of
    :func:`resolve_providers` - same vocabulary in, this runtime's own form out.
    ``auto`` resolves CUDA, then MPS (Apple Silicon), then CPU; MPS is a real
    accelerator for these models but has genuine op gaps, so it is only
    auto-selected when torch reports it BUILT and available.

    A NAMED device is verified against what torch can actually see and raises if it
    cannot be honoured - never a silent demotion to CPU, which on a ``.pth`` is a
    4-10x slowdown a caller would have no way to notice.

    Args:
        device: See :data:`Device`.

    Raises:
        ValueError: ``device`` names hardware torch cannot reach here. The message
            says what torch DOES report, so the next call is informed rather than
            another guess.
    """
    import torch

    family, index = normalize_device(device)
    if family == "cpu":
        return "cpu"

    cuda_ok = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_ok = bool(mps_backend is not None and mps_backend.is_available() and mps_backend.is_built())

    if family == "auto":
        if cuda_ok:
            return "cuda"
        return "mps" if mps_ok else "cpu"

    if family == "cuda":
        if not cuda_ok:
            raise ValueError(
                f"device={device!r} was requested, but torch.cuda.is_available() is "
                f"False - this build of torch sees no CUDA GPU. "
                + (
                    "This machine's accelerator is Apple's; pass device='mps'. "
                    if mps_ok
                    else "Install a CUDA build of torch, or run on another machine. "
                )
                + "Pass device='cpu' to run here anyway."
            )
        count = int(torch.cuda.device_count())
        if index is not None and index >= count:
            raise ValueError(
                f"device={device!r} was requested, but torch reports "
                f"{count} CUDA device{'s' if count != 1 else ''} "
                f"(valid: cuda:0{f'-cuda:{count - 1}' if count > 1 else ''})."
            )
        return f"cuda:{index}" if index is not None else "cuda"

    if family == "mps":
        if not mps_ok:
            built = bool(mps_backend is not None and mps_backend.is_built())
            raise ValueError(
                f"device={device!r} was requested, but torch's MPS backend is "
                + ("built and reporting unavailable" if built else "not built into this torch")
                + " - Apple's accelerator is not reachable here. "
                + (
                    "This machine has CUDA; pass device='cuda'. "
                    if cuda_ok
                    else "Pass device='cpu' to run on the CPU."
                )
            )
        return "mps"

    raise ValueError(
        f"torch cannot target device={family!r}. It runs on cuda, mps and cpu; "
        f"{family!r} is reachable only through a graph runtime - load this model's "
        f".onnx (format='onnx') or .pte (format='pytorch_engine') instead."
    )


def empty_device_cache(device: str) -> None:
    """Release cached allocator blocks for ``device``.

    ``del model`` drops the tensors but torch's caching allocator keeps the blocks,
    so ``nvidia-smi`` shows no change - which is exactly what a user closing models
    in a loop is watching. Called from ``close()``.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional
        return
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.startswith("mps"):
        backend = getattr(torch, "mps", None)
        if backend is not None and hasattr(backend, "empty_cache"):
            backend.empty_cache()
