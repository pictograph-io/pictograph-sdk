"""Run your trained Pictograph models locally.

A one-call, task-typed local inference API. Load a model by name or id and predict
on an image - weights download and cache automatically on first use, and the result
is a per-task type whose ``predictions`` are already narrowed to the annotations
that task can produce::

    from pictograph import get_model, DetectionModel, DetectionResult

    model: DetectionModel = get_model("Woody Whirling Grouse", task="object_detection")
    result: DetectionResult = model.predict("photo.jpg")

    for p in result.predictions:
        print(p.name, round(p.confidence, 2), p.bounding_box)

``task=`` is what lets the annotation typecheck without a cast, and it is verified:
the loader reads the model's real task and raises if they disagree, so it can never
silently lie. Omit it and you get the :data:`~pictograph.inference.models.AnyModel`
union to narrow with ``isinstance``.

Classifiers return ranked classes instead of geometry, and ``top`` is never ``None``::

    from pictograph import get_model, ClassificationResult

    result: ClassificationResult = get_model("My Classifier", task="classification").predict(
        "cat.jpg"
    )
    print(result.top.name, round(result.top.confidence, 2))

One argument picks the weights: ``format=``
-------------------------------------------
The same trained weights are published in every executable form your edge actually
runs, and every one of them returns the SAME task class and the SAME typed result.
``format=`` names the WEIGHTS FILE you want; the runtime that executes it follows
from that and is never asked for separately::

    from pictograph import get_model

    get_model("Detector", task="object_detection")  # format="onnx", the default
    get_model("Detector", task="object_detection", format="pytorch_engine")
    get_model("Detector", task="object_detection", format="tensorrt_engine")
    get_model("Detector", task="object_detection", format="safetensors")
    get_model("Detector", task="object_detection", format="pytorch")

====================  ==================  ==============  =========================
``format=``           file                runtime         what it is for
====================  ==================  ==============  =========================
``pytorch``           ``.pth``            ``pytorch``     the training checkpoint,
                                                          fine-tuning, research
``safetensors``       ``.safetensors``    ``pytorch``     the same module from the
                                                          parity-gated container
``pytorch_engine``    ``.pte``            ``executorch``  portable edge - phones,
                                                          Jetson CPU, ARM boards
``onnx``              ``.onnx``           ``onnxruntime`` the default; runs anywhere
``tensorrt_engine``   ``.engine``         ``tensorrt``    NVIDIA, lowest latency
====================  ==================  ==============  =========================

That order is the product's - native source → PyTorch's own edge engine →
cross-platform runtime → NVIDIA-only AOT-compiled engine, i.e. increasing
specialisation and decreasing portability. ``model.backend`` reports which runtime
actually ran.

**A format the model does not publish is refused, never substituted.** The refusal
names what the model DOES have, so the next call is obvious rather than a guess.

:func:`load_model` is the fully offline twin and needs no API key. It reads the
same ``format=``, defaulting to the weights' own suffix, so the call shape never
changes::

    load_model("model.onnx", "config.json", task="classification")
    load_model("xnnpack-fp32.pte", "config.json", task="classification")
    load_model("sm75-trt10.13.3.9-fp16.engine", "config.json", task="classification")

 **A ``.engine`` is not portable.** A TensorRT plan is compiled for one GPU
architecture, one TensorRT version and one precision; copying it to another machine
fails at LOAD. The loader checks before it tries and refuses with a message naming
both the built-for target and your device - see :mod:`pictograph.inference._tensorrt`.
A ``.pte``, by contrast, IS portable across devices for its lowering backend.

Requires the optional inference extra::

    pip install "pictograph[inference]"              # .onnx, .pth AND .safetensors
    pip install "pictograph[inference,executorch]"   # + .pte  (ExecuTorch)
    pip install "pictograph[inference,tensorrt]"     # + .engine (TensorRT, NVIDIA only)

One line is the whole install for the three formats every model family publishes
natively - ``[inference]`` carries torch, torchvision, safetensors and the pinned
``segmentation-models-pytorch`` alongside the ONNX stack, and the YOLOX and RF-DETR
architectures are vendored into the wheel. There is deliberately no second
``pip install`` of a third-party package anywhere in this SDK's docs, error
messages or the app's install snippet. The two engine runtimes stay separate
because they cannot be folded in: ExecuTorch pins one exact ``torch`` minor, and
``tensorrt`` has no non-CUDA distribution at all.

One argument picks the hardware: ``device=``
-------------------------------------------
``device="auto"`` (the default) selects automatically where there is a choice to
make: CUDA on NVIDIA, CoreML on macOS, MPS for the torch runtime on Apple Silicon,
CPU everywhere. Name one and you get it or an exception - never a silent demotion::

    get_model("Detector", task="object_detection", device="cuda")
    load_model("weights.pth", cfg, task="classification", device="cuda:1")
    load_model("model.onnx", cfg, task="classification", device="cpu")

**``device=`` and ``format=`` are the only two knobs, and they are orthogonal**:
``format`` picks the WEIGHTS, ``device`` picks the HARDWARE, and the execution
provider is DERIVED from the pair - you never name one. ``mps`` reaches torch's MPS
backend for a ``.pth`` and CoreML for an ``.onnx``; ``cuda`` reaches TensorRT for an
``.engine`` and CUDA for an ``.onnx``. A pairing that does not exist (a ``.engine``
on the CPU) raises and names what that format CAN run on.

``model.device`` reports what actually RAN, which is sometimes more specific than
what you asked for - ``device="mps"`` on an ``.onnx`` reports ``"coreml"``, because
that is the mechanism that got you there. That is the property that tells you a CUDA
request really landed on CUDA. See :mod:`pictograph.inference.runtime` for the
measured policy, the full ``(device, format)`` table, and why TensorRT is
deliberately not in the automatic ONNX ladder.
"""

from __future__ import annotations

import json as _json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

from pictograph._path_safety import safe_path_component
from pictograph.inference.models import (
    TASK_MODEL_TYPES,
    AnyModel,
    ClassificationModel,
    DetectionModel,
    InferenceModel,
    InstanceSegmentationModel,
    KeypointModel,
    SemanticSegmentationModel,
)
from pictograph.inference.results import (
    AnyResult,
    ClassificationResult,
    ClassScore,
    DetectionResult,
    InferenceResult,
    InstanceSegmentationResult,
    KeypointResult,
    SemanticSegmentationResult,
    TaskName,
)
from pictograph.inference.runtime import (
    DEVICES,
    RUNTIMES,
    WEIGHT_FORMATS,
    Device,
    Runtime,
    WeightFormat,
    check_device_supported,
    format_for_weights,
    runtime_for_format,
    suffix_for_format,
    wire_format,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pictograph.client import Client
    from pictograph.models.model import Model

__all__ = [
    "DEVICES",
    "RUNTIMES",
    "WEIGHT_FORMATS",
    "AnyModel",
    "AnyResult",
    "ClassScore",
    "ClassificationModel",
    "ClassificationResult",
    "DetectionModel",
    "DetectionResult",
    "Device",
    "InferenceModel",
    "InferenceResult",
    "InstanceSegmentationModel",
    "InstanceSegmentationResult",
    "KeypointModel",
    "KeypointResult",
    "Runtime",
    "SemanticSegmentationModel",
    "SemanticSegmentationResult",
    "TaskName",
    "WeightFormat",
    "get_model",
    "load_model",
]

ImageInput = Any

_LOG = logging.getLogger("pictograph.inference")


# ───────────── get_model (by name/id, any format) ─────────────


@overload
def get_model(
    name: str,
    *,
    task: Literal["object_detection"],
    format: WeightFormat = ...,
    precision: Literal["fp32", "fp16"] | None = ...,
    target: str | None = ...,
    api_key: str | None = ...,
    client: Client | None = ...,
    confidence: float = ...,
    cache_dir: str | Path | None = ...,
    device: Device = ...,
) -> DetectionModel: ...
@overload
def get_model(
    name: str,
    *,
    task: Literal["instance_segmentation"],
    format: WeightFormat = ...,
    precision: Literal["fp32", "fp16"] | None = ...,
    target: str | None = ...,
    api_key: str | None = ...,
    client: Client | None = ...,
    confidence: float = ...,
    cache_dir: str | Path | None = ...,
    device: Device = ...,
) -> InstanceSegmentationModel: ...
@overload
def get_model(
    name: str,
    *,
    task: Literal["semantic_segmentation"],
    format: WeightFormat = ...,
    precision: Literal["fp32", "fp16"] | None = ...,
    target: str | None = ...,
    api_key: str | None = ...,
    client: Client | None = ...,
    confidence: float = ...,
    cache_dir: str | Path | None = ...,
    device: Device = ...,
) -> SemanticSegmentationModel: ...
@overload
def get_model(
    name: str,
    *,
    task: Literal["keypoint_detection"],
    format: WeightFormat = ...,
    precision: Literal["fp32", "fp16"] | None = ...,
    target: str | None = ...,
    api_key: str | None = ...,
    client: Client | None = ...,
    confidence: float = ...,
    cache_dir: str | Path | None = ...,
    device: Device = ...,
) -> KeypointModel: ...
@overload
def get_model(
    name: str,
    *,
    task: Literal["classification"],
    format: WeightFormat = ...,
    precision: Literal["fp32", "fp16"] | None = ...,
    target: str | None = ...,
    api_key: str | None = ...,
    client: Client | None = ...,
    confidence: float = ...,
    cache_dir: str | Path | None = ...,
    device: Device = ...,
) -> ClassificationModel: ...
@overload
def get_model(
    name: str,
    *,
    task: None = ...,
    format: WeightFormat = ...,
    precision: Literal["fp32", "fp16"] | None = ...,
    target: str | None = ...,
    api_key: str | None = ...,
    client: Client | None = ...,
    confidence: float = ...,
    cache_dir: str | Path | None = ...,
    device: Device = ...,
) -> AnyModel: ...


def get_model(
    name: str,
    *,
    task: TaskName | None = None,
    format: WeightFormat = "onnx",
    precision: Literal["fp32", "fp16"] | None = None,
    target: str | None = None,
    api_key: str | None = None,
    client: Client | None = None,
    confidence: float = 0.5,
    device: Device = "auto",
    cache_dir: str | Path | None = None,
) -> Any:
    """Load one of your trained models for local inference, in any published format.

    Downloads and caches the weights file ``format`` names, then hands back the
    task's model class. The return type and the result type do not depend on the
    format - that is the whole point of the taxonomy.

    Args:
        name: The model's name, e.g. ``"My Detector"``.
        task: The model's task. Pass it as a literal and the return type narrows to
            that task's model class, so ``model: DetectionModel = ...`` typechecks
            with no cast. It is VERIFIED against the model's real task, never
            assumed. Omit for the :data:`AnyModel` union.
        format: Which weights file to load - ``"onnx"`` (default),
            ``"safetensors"``, ``"pytorch"``, ``"pytorch_engine"`` (the ExecuTorch
            ``.pte``) or ``"tensorrt_engine"`` (the TensorRT ``.engine``). The
            runtime follows from it and is not asked for separately; see
            :data:`~pictograph.inference.runtime.WeightFormat`. A format the model
            does not publish is refused with a message naming the ones it does.
        precision: ``"fp32"`` or ``"fp16"``. Selects WHICH artifact to fetch for the
            formats published per precision. ``"pytorch"`` and ``"safetensors"``
            have exactly one file per model version, so a value that disagrees with
            the version's own precision raises rather than silently handing back the
            other one.
        target: Which binding to fetch. For ``"tensorrt_engine"`` this is the GPU
            architecture (``"sm75"``…) and it DEFAULTS TO THIS MACHINE'S - asking for
            "the engine for my GPU" is the overwhelmingly common case, and fetching
            any other one would produce a file that cannot load here. For
            ``"pytorch_engine"`` it is the lowering backend, defaulting to
            ``"xnnpack"`` (portable CPU).
        api_key: Your Pictograph API key. Falls back to ``PICTOGRAPH_API_KEY``.
            Ignored if ``client`` is given.
        client: An existing :class:`~pictograph.client.Client` to use.
        confidence: Default minimum score for ``predict`` (0-1).
        device: Which HARDWARE to run on - ``"auto"`` (default), ``"cpu"``,
            ``"cuda"`` (or ``"cuda:1"`` for a specific GPU) or ``"mps"``. The SAME
            argument with the SAME values on every format: the execution provider is
            DERIVED from ``(device, format)``, never named by you - ``mps`` reaches
            CoreML for an ``.onnx`` and torch's MPS for a ``.pth``, ``cuda`` reaches
            TensorRT for an ``.engine`` and CUDA for an ``.onnx``. A device this
            machine (or this artifact) cannot honour RAISES, naming what is
            available - it is never silently downgraded to CPU. See
            :data:`~pictograph.inference.runtime.Device`.
        cache_dir: Where downloaded weights are cached. Defaults to
            ``$PICTOGRAPH_CACHE_DIR`` or ``~/.pictograph/models``.

    Returns:
        The task's model class, ready to predict. ``model.device`` reports what
        actually RAN, which may be more specific than what you asked for
        (``device="mps"`` on an ``.onnx`` reports ``"coreml"``).
    """
    resolved_client = client if client is not None else _default_client(api_key)
    return _load_by_name(
        name,
        models=resolved_client.models,
        task=task,
        format=format,
        precision=precision,
        target=target,
        confidence=confidence,
        device=device,
        cache_dir=cache_dir,
    )


def _load_by_name(
    name: str,
    *,
    models: Any,
    task: TaskName | None,
    format: WeightFormat,
    precision: Literal["fp32", "fp16"] | None,
    target: str | None,
    confidence: float,
    device: Device,
    cache_dir: str | Path | None,
) -> Any:
    """The body of :func:`get_model`, shared with ``client.models.load``.

    Both entry points must resolve the format, check it against the requested device
    and build the engine identically - a second copy is how the client-bound twin
    drifts into being a different loader.
    """
    # Validates `format` (a caller without a type checker can pass anything) and is
    # the ONLY thing that decides which runtime executes it.
    runtime = runtime_for_format(format)
    # The `(device, format)` gate, before the network call: asking for a .engine on
    # the CPU is wrong no matter what the model record says, so say so now rather
    # than after a download.
    check_device_supported(device, runtime, format)
    model = models.get_by_name(name)

    if format in ("pytorch", "safetensors"):
        _check_native_precision(model, format, precision)
        return _build_torch_model(
            model,
            models=models,
            task=task,
            weight_format=format,
            confidence=confidence,
            device=device,
            cache_dir=cache_dir,
        )

    return _build_graph_model(
        model,
        models=models,
        task=task,
        format=format,
        precision=precision,
        target=target,
        confidence=confidence,
        device=device,
        cache_dir=cache_dir,
    )


# ───────────── load_model (ONNX, fully offline) ─────────────


@overload
def load_model(
    weights: str | Path,
    config: str | Path | dict[str, Any],
    *,
    task: Literal["object_detection"],
    format: WeightFormat | None = ...,
    confidence: float = ...,
    device: Device = ...,
    cache_dir: str | Path | None = ...,
) -> DetectionModel: ...
@overload
def load_model(
    weights: str | Path,
    config: str | Path | dict[str, Any],
    *,
    task: Literal["instance_segmentation"],
    format: WeightFormat | None = ...,
    confidence: float = ...,
    device: Device = ...,
    cache_dir: str | Path | None = ...,
) -> InstanceSegmentationModel: ...
@overload
def load_model(
    weights: str | Path,
    config: str | Path | dict[str, Any],
    *,
    task: Literal["semantic_segmentation"],
    format: WeightFormat | None = ...,
    confidence: float = ...,
    device: Device = ...,
    cache_dir: str | Path | None = ...,
) -> SemanticSegmentationModel: ...
@overload
def load_model(
    weights: str | Path,
    config: str | Path | dict[str, Any],
    *,
    task: Literal["keypoint_detection"],
    format: WeightFormat | None = ...,
    confidence: float = ...,
    device: Device = ...,
    cache_dir: str | Path | None = ...,
) -> KeypointModel: ...
@overload
def load_model(
    weights: str | Path,
    config: str | Path | dict[str, Any],
    *,
    task: Literal["classification"],
    format: WeightFormat | None = ...,
    confidence: float = ...,
    device: Device = ...,
    cache_dir: str | Path | None = ...,
) -> ClassificationModel: ...
@overload
def load_model(
    weights: str | Path,
    config: str | Path | dict[str, Any],
    *,
    task: None = ...,
    format: WeightFormat | None = ...,
    confidence: float = ...,
    device: Device = ...,
    cache_dir: str | Path | None = ...,
) -> AnyModel: ...


def load_model(
    weights: str | Path,
    config: str | Path | dict[str, Any],
    *,
    task: TaskName | None = None,
    format: WeightFormat | None = None,
    confidence: float = 0.5,
    device: Device = "auto",
    cache_dir: str | Path | None = None,
) -> Any:
    """Load a trained model from its LOCAL files - fully offline, no API call.

      The offline twin of :func:`get_model`. Point it at the ``weights`` and the
      ``config.json`` a Pictograph training pipeline wrote (both live in a model
      version's file bundle - fetch them from the model's Files tab, or with
      ``client.models.download`` / ``download_file``)::

          from pictograph import load_model, DetectionModel

          model: DetectionModel = load_model(
              "yolox-a1b2c3.onnx", "config.json", task="object_detection"
          )
          result = model.predict("photo.jpg")

    **``format`` defaults to the weights' own suffix**, so the call shape is the same
      for all three compiled formats and the artifact stays the single source of truth
      for what it is::

          load_model("model.onnx", cfg, task="classification")  # ONNX Runtime
          load_model("xnnpack-fp32.pte", cfg, task="classification")  # ExecuTorch
          load_model("sm75-trt10.13.3.9-fp16.engine", cfg, task="classification")  # TensorRT

      A ``.engine`` is checked against this machine BEFORE it is deserialized: its
      filename (or a ``<stem>.json`` sidecar) says which GPU architecture and TensorRT
      version it was built for, and a mismatch raises a message naming both rather
      than surfacing a raw deserialization crash.

      ``device=`` works here exactly as it does on :func:`get_model` - same values,
      same meaning, same refusals - so a local ``.pth`` can be pinned to a GPU without
      a round trip to the API::

          load_model("weights.pth", cfg, task="classification", device="cuda:1")
          load_model("model.onnx", cfg, task="classification", device="cpu")

      Args:
          weights: Path to the model's ``.onnx``, ``.pte``, ``.engine``, ``.pth`` or
              ``.safetensors`` file.
          config: Path to (or the parsed dict of) the model's ``config.json``.
          task: See :func:`get_model`.
          format: Which format ``weights`` holds - the same vocabulary
              :func:`get_model` takes. Defaults to the file's suffix, which is what
              keeps the artifact self-describing; pass it only for a renamed file.
          device: Which HARDWARE to run on - identical to :func:`get_model`'s.
              ``"auto"`` (default), ``"cpu"``, ``"cuda"`` / ``"cuda:1"``, ``"mps"``.
          cache_dir: Where the compiled-CoreML cache lives (``format="onnx"`` only).

      Raises:
          ValueError: The suffix names no format, or ``device`` names hardware this
              format cannot run on / this machine does not have.
          RuntimeError: A ``.engine`` built for a different GPU architecture or
              TensorRT version than this machine's.
          ImportError: The runtime's package is not installed. The message carries
              the exact ``pip install`` command.
    """
    raw = (
        config
        if isinstance(config, dict)
        else _json.loads(Path(config).read_text(encoding="utf-8"))
    )
    weights_path = Path(weights)
    resolved_format = format if format is not None else format_for_weights(weights_path)
    resolved_runtime = runtime_for_format(resolved_format)
    check_device_supported(device, resolved_runtime, resolved_format)

    model_type, architecture, classes, input_shape, name = _parse_model_config(raw)
    resolved_task = _verify_task(task, model_type, name or weights_path.stem)
    root = _cache_dir(cache_dir)

    if resolved_runtime == "pytorch":
        from pictograph.inference._torch import NativeSpec, build_local_torch_engine

        engine: Any = build_local_torch_engine(
            weights_path,
            weight_format=resolved_format,  # type: ignore[arg-type]
            spec=NativeSpec(
                model_type=model_type,
                architecture=architecture,
                training_config=_training_config_of(raw, input_shape),
                classes=classes,
                name=name or weights_path.stem,
            ),
            cache_dir=root,
            device=device,
            keypoint_schema=_keypoint_schema_of(raw),
            confidence=confidence,
        )
    else:
        engine = _build_engine(
            runtime=resolved_runtime,
            weights=weights_path,
            model_type=model_type,
            architecture=architecture,
            classes=classes,
            input_shape=input_shape,
            confidence=confidence,
            device=device,
            cache_dir=root,
            keypoint_schema=_keypoint_schema_of(raw),
        )
    return _wrap(resolved_task, engine, name or weights_path.stem, name, architecture, confidence)


def _build_torch_model(
    model: Model,
    *,
    models: Any,
    task: TaskName | None,
    weight_format: Literal["pytorch", "safetensors"],
    confidence: float,
    device: Device,
    cache_dir: str | Path | None,
) -> Any:
    """Rebuild a model's live ``nn.Module`` from one of its two native containers."""
    resolved_task = _verify_task(task, model.model_type, model.name)

    from pictograph.inference._torch import build_torch_engine

    engine = build_torch_engine(
        model,
        models=models,
        weight_format=weight_format,
        cache_dir=_cache_dir(cache_dir),
        device=device,
        keypoint_schema=_fetch_keypoint_schema(models, model, _cache_dir(cache_dir)),
        confidence=confidence,
    )
    return _wrap(resolved_task, engine, model.id, model.name, model.architecture or "", confidence)


# ───────────── internals ─────────────


def _wrap(
    task: TaskName,
    engine: Any,
    model_id: str,
    name: str,
    architecture: str,
    confidence: float,
) -> Any:
    """Put the right task class around a built engine."""
    cls = TASK_MODEL_TYPES[task]
    return cls(
        engine=engine,
        model_id=model_id,
        name=name,
        architecture=architecture,
        confidence=confidence,
    )


def _verify_task(declared: TaskName | None, actual: str, name: str) -> TaskName:
    """Check a caller-declared ``task`` against the model's real one.

    ``task=`` is a typing device, so it MUST NOT be able to lie: a mismatch here
    would hand back a class whose ``predict`` returns a different shape than the
    annotation promises. Raising keeps the annotation honest.
    """
    if actual not in TASK_MODEL_TYPES:
        raise ValueError(
            f"Model {name!r} has task {actual!r}, which this SDK cannot run locally. "
            f"Expected one of {sorted(TASK_MODEL_TYPES)}."
        )
    resolved: TaskName = actual
    if declared is not None and declared != resolved:
        raise ValueError(
            f"Model {name!r} is a {resolved!r} model, but task={declared!r} was requested. "
            f"Pass task={resolved!r} (or omit it) - the annotation would otherwise be wrong."
        )
    return resolved


def _check_native_precision(model: Model, fmt: str, precision: str | None) -> None:
    """A native container holds ONE checkpoint, at the version's own precision.

    ``.pth`` / ``model.safetensors`` are the only artifacts with no derived form:
    ``.pte``, ``.engine`` and - since the 2026-07-30 contract amendment - a derived
    fp16 ``.onnx`` are all published per precision, but a checkpoint is the raw
    trained tensors and there is nothing to derive it into. So a precision that
    disagrees with the version's cannot be honoured, and saying so beats handing back
    the fp32 file while the caller believes they are benchmarking fp16.
    """
    actual = getattr(model, "precision", None)
    if precision is None or actual is None or precision == actual:
        return
    raise ValueError(
        f"Model {model.name!r} was trained at {actual}, and format={fmt!r} serves "
        f"that checkpoint as-is - a native checkpoint has no derived {precision} "
        f"form. To run {precision}, build a {precision} artifact for it and load "
        f"that (format='onnx' / 'pytorch_engine' / 'tensorrt_engine'), or select a "
        f"model version trained at {precision}."
    )


def _build_engine(
    *,
    runtime: Runtime,
    weights: Path,
    model_type: str,
    architecture: str,
    classes: list[str],
    input_shape: tuple[int, int],
    confidence: float,
    device: Device,
    cache_dir: Path,
    keypoint_schema: dict[str, Any] | None,
) -> Any:
    """The ONE place a graph runtime is chosen and its engine constructed.

    Shared by :func:`load_model` and :func:`_build_graph_model` so the offline and
    the by-name paths cannot build a model differently - the same defect class the
    ONNX loader's single ``build_onnx_engine`` entry point already prevents.
    """
    if runtime == "executorch":
        from pictograph.inference._executorch import build_executorch_engine

        return build_executorch_engine(
            weights=weights,
            model_type=model_type,
            architecture=architecture,
            classes=classes,
            input_shape=input_shape,
            confidence=confidence,
            device=device,
            keypoint_schema=keypoint_schema,
        )
    if runtime == "tensorrt":
        from pictograph.inference._tensorrt import build_tensorrt_engine

        return build_tensorrt_engine(
            weights=weights,
            model_type=model_type,
            architecture=architecture,
            classes=classes,
            input_shape=input_shape,
            confidence=confidence,
            device=device,
            keypoint_schema=keypoint_schema,
        )

    from pictograph.inference._onnx import build_onnx_engine

    return build_onnx_engine(
        weights=weights,
        model_type=model_type,
        architecture=architecture,
        classes=classes,
        # ONNX declares its input shape rather than compiling it in, so the graph is
        # introspected here. The two AOT runtimes read theirs off the loaded artifact
        # instead (`_engine.input_hw_from`), which is strictly more authoritative.
        input_shape=_true_input_size(weights, input_shape),
        confidence=confidence,
        device=device,
        cache_dir=cache_dir,
        keypoint_schema=keypoint_schema,
    )


#: The ``target_key`` sentinel for an artifact bound to no particular hardware.
#: Must stay in sync with the server's model-artifact contract - an ONNX graph runs on
#: every device onnxruntime supports, so there is nothing to bind it to, and the
#: sentinel keeps the column NOT NULL without inventing a fake hardware target.
PORTABLE_TARGET = "portable"

#: The ExecuTorch lowering published by default: the portable CPU backend, the one
#: ``.pte`` that runs on x86, ARM, a Jetson's CPU and an iPhone alike.
DEFAULT_PTE_TARGET = "xnnpack"


def _artifact_request(
    model: Model, fmt: WeightFormat, precision: str | None, target: str | None
) -> tuple[str, str, str, str, bool]:
    """``(wire_format, precision, target, suffix, is_derived)`` for a compiled format.

    The first element is the token the platform's ``/download`` route takes, which is
    the one place the SDK's ``format`` vocabulary is translated to the wire's.

    The defaults are the ones that make ``get_model(format=...)`` do the obviously
    right thing:

    - ``tensorrt_engine`` defaults ``target`` to **this machine's** SM. Any other
      engine is a file that provably cannot load here, so defaulting to anything else
      would be choosing a broken download.
    - ``pytorch_engine`` defaults to :data:`DEFAULT_PTE_TARGET`, the portable-CPU
      lowering that is the default published artifact and the only one that runs
      everywhere.
    - ``onnx`` serves the version's own SHIPPED graph unless a precision that
      differs from the version's is asked for, in which case it is a DERIVED
      artifact and is fetched like any other. That distinction is the whole point of
      the 2026-07-30 contract amendment: a version has exactly one
      ``gcs_weights_path`` (fp32, never overwritten) and one ``precision``, so a
      derived fp16 graph has no 1:1 column to live in and is a ``model_artifacts``
      row with ``target_key='portable'``.
    """
    resolved_precision = precision or "fp32"
    wire = wire_format(fmt)
    suffix = suffix_for_format(fmt)
    if fmt == "pytorch_engine":
        return wire, resolved_precision, target or DEFAULT_PTE_TARGET, suffix, True
    if fmt == "tensorrt_engine":
        from pictograph.inference._tensorrt import detect_local_target

        resolved_target = target or detect_local_target().sm
        if resolved_target == "unknown":
            raise ValueError(
                "Could not detect this machine's GPU architecture, so there is no way "
                "to tell which TensorRT engine to fetch - an engine built for the "
                "wrong architecture will not load. Pass target='sm75' (T4) / 'sm80' "
                "(A100) / 'sm86' (A10G) / 'sm89' (L4) / 'sm90' (H100) explicitly, or "
                "use format='onnx' on a machine without an NVIDIA GPU."
            )
        return wire, resolved_precision, resolved_target, suffix, True

    shipped = getattr(model, "precision", None) or "fp32"
    derived = precision is not None and precision != shipped
    return (
        wire,
        resolved_precision,
        PORTABLE_TARGET if derived else "",
        suffix,
        derived,
    )


def _build_graph_model(
    model: Model,
    *,
    models: Any,
    task: TaskName | None,
    format: WeightFormat,
    precision: str | None,
    target: str | None,
    confidence: float,
    device: Device,
    cache_dir: str | Path | None,
) -> Any:
    """Fetch-and-build for the three compiled formats, behind :func:`get_model`."""
    if model.status != "ready":
        raise ValueError(
            f"Model {model.name!r} is {model.status!r}, not 'ready' - it can't be run yet."
        )
    resolved_task = _verify_task(task, model.model_type, model.name)
    classes = _classes_of(model)
    root = _cache_dir(cache_dir)
    wire, resolved_precision, resolved_target, suffix, derived = _artifact_request(
        model, format, precision, target
    )

    weights = root / f"{_artifact_stem(model, format, resolved_precision, resolved_target)}{suffix}"
    if not weights.exists():
        if derived:
            models.download(
                model_id=model.id,
                output_path=weights,
                format=wire,
                precision=resolved_precision,
                target=resolved_target,
            )
        else:
            # The version's own shipped graph. Sent WITHOUT precision/target so an
            # older backend - one that predates the derived-artifact params - still
            # serves the ONNX download it has always served.
            models.download(model_id=model.id, output_path=weights, format=wire)

    engine = _build_engine(
        runtime=runtime_for_format(format),
        weights=weights,
        model_type=model.model_type,
        architecture=model.architecture or "",
        classes=classes,
        input_shape=_input_size_of(model),
        confidence=confidence,
        device=device,
        cache_dir=root,
        keypoint_schema=_fetch_keypoint_schema(models, model, root),
    )
    return _wrap(resolved_task, engine, model.id, model.name, model.architecture or "", confidence)


def _artifact_stem(model: Model, fmt: str, precision: str, target: str) -> str:
    """Cache filename stem, keyed by everything that makes the artifact a DIFFERENT file.

    ``_cache_stem`` already keys on model id + served version. A DERIVED artifact adds
    three more dimensions - an fp16 sm80 engine and an fp32 sm75 engine are both "this
    model", and caching them under one name would serve whichever landed first and
    then refuse to load it on the other machine.

    An empty ``target`` means the version's own shipped graph, which keeps the bare
    stem so an existing cache entry is not invalidated by this change.
    """
    base = _cache_stem(model)
    if not target:
        return base
    return "-".join([base, fmt, precision, target])


def _cache_stem(model: Model) -> str:
    """Cache key - the model id PLUS the version it currently serves.

    Keying on the id alone meant a retrained (or rolled-back) model kept predicting
    with whatever was downloaded first, forever, on that machine.
    """
    version = getattr(model, "current_version_id", None) or getattr(model, "updated_at", None)
    # Both halves are SERVER-SUPPLIED and become a directory name, so both are
    # reduced to a safe component first. This is the twin of the guarded helper
    # in pictograph.inference._torch; stripping ':' alone left "../" intact.
    ident = safe_path_component(model.id, fallback="model")
    if version is None:
        return ident
    return f"{ident}-{safe_path_component(str(version)[:32], fallback='v')}"


def _fetch_keypoint_schema(models: Any, model: Model, cache_dir: Path) -> dict[str, Any] | None:
    """The model's ``keypoint_schema`` block, for keypoint models only.

    Without it a multi-joint model's joints come back positionally named
    (``point_0``…) and with no class template - the model still runs and its points
    still group by ``instance_id``, but every joint is anonymous and the template's
    connectivity (the only thing left that can DRAW a pose) is lost. The block also
    carries the per-class arity, so a keypoint-as-class model needs it too. Only
    keypoint configs carry it, so a miss on any other task is normal and never an
    error.

    ``download_file`` streams to disk and returns a path (it does not return bytes),
    so the config is cached alongside the weights and re-read on later loads.
    """
    if model.model_type != "keypoint_detection":
        return None
    target = cache_dir / f"{_cache_stem(model)}-config.json"
    if not target.exists():
        try:
            models.download_file(model_id=model.id, file_name="config.json", output_path=target)
        except Exception as exc:
            _LOG.warning(
                "Could not fetch config.json for keypoint model %s (%s) - its joints "
                "will be positionally named (point_0…) with no class template.",
                model.name,
                exc,
            )
            return None
    try:
        parsed = _json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _LOG.warning("config.json for %s is unreadable (%s).", model.name, exc)
        return None
    schema = _keypoint_schema_of(parsed)
    if schema is None:
        _LOG.warning(
            "Keypoint model %s has a config.json with no keypoint_schema - joints "
            "will be positionally named.",
            model.name,
        )
    return schema


def _parse_model_config(
    raw: dict[str, Any],
) -> tuple[str, str, list[str], tuple[int, int], str]:
    """Read the fields :func:`load_model` needs from a pipeline's ``config.json``.

    Accepts the full artifact (``{"_pictograph": {...}, "config": {...}}``) or a bare
    metadata dict, so a downloaded config or a hand-built one both work.
    """
    env = raw.get("_pictograph") if isinstance(raw.get("_pictograph"), dict) else raw
    if not isinstance(env, dict):
        raise ValueError("config must be a JSON object (a config.json dict).")

    model_type = str(env.get("model_type") or "").strip()
    if not model_type:
        raise ValueError(
            "config.json has no 'model_type' - is this a Pictograph model config.json?"
        )
    architecture = str(env.get("architecture") or "")

    mapping = env.get("class_mapping")
    classes = mapping.get("classes") if isinstance(mapping, dict) else None
    if not classes:
        classes = env.get("class_names")
    if not (isinstance(classes, list) and classes):
        raise ValueError(
            "config.json has no class list (expected class_mapping.classes or class_names)."
        )
    class_list = [str(c) for c in classes]

    height, width = _declared_input_shape(env) or (640, 640)

    # The model's own name, when the artifact carries one. Before that it
    # had to be inferred from the export or the dataset it was trained on, which
    # named the DATA rather than the model.
    name = str(env.get("name") or env.get("export_name") or env.get("dataset_name") or "").strip()
    return model_type, architecture, class_list, (height, width), name


def _training_config_of(raw: dict[str, Any], input_shape: tuple[int, int]) -> dict[str, Any]:
    """The training config a native rebuild needs, out of a pipeline's ``config.json``.

     The offline counterpart of the API record's ``training_config``, and the reason
     :func:`load_model` can rebuild a ``.pth`` at all. A pipeline writes the run's
     hyperparameters under ``config`` beside the ``_pictograph`` metadata envelope, and
     those are exactly the keys the rebuild reads to pick a model definition:
     ``model_size`` (YOLOX, RF-DETR), ``architecture`` / ``encoder`` (SMP), ``backbone``
     / ``dropout_rate`` / ``hidden_units`` (torchvision classifiers), ``resolution``.

     ``image_height`` / ``image_width`` are backfilled from the metadata's
     ``input_shape`` when the config block omits them, so the input size resolves to the
     graph's real resolution rather than the 640 default - the same "artifact beats
     config" rule :func:`_true_input_size` applies on the ONNX path.

    **Only a DECLARED ``input_shape`` is backfilled.** ``_parse_model_config``
     has to return a concrete tuple, so it substitutes ``(640, 640)`` when the
     artifact states nothing. Backfilling THAT wrote a guess into the config under
     the same keys a known value uses, and every downstream reader then treated it
     as fact:

     * RF-DETR's ``_rfdetr_resolution`` found 640, rebuilt the module at 640, and
       the DINOv2 backbone asserts its input be divisible by 24 - so
       ``load_model(format="safetensors")`` raised outright for any model whose
       config.json carries ``input_shape: null``. Measured on three published
       fixtures. It also defeated ``_rfdetr_module_resolution``, which exists to
       recover exactly this: it only consults the module when the size is unknown,
       and the guess made it look known.
     * ``_pytorch_input_size`` defaults per family - 224 classification, 512
       semantic segmentation - and a backfilled 640 silently overrode both, so
       those ran at the wrong scale instead of crashing.

     Leaving the keys unset when nothing was declared hands each family its own
     default, which is correct by construction for all of them.
    """
    # The block is `training`; `config` is what every artifact published
    # before 2026-07-31 called it. Both are read, in that order, because this
    # rebuild must work for a model downloaded at any point in the product's life.
    block = raw.get("training")
    if not isinstance(block, dict):
        block = raw.get("config")
    config: dict[str, Any] = dict(block) if isinstance(block, dict) else {}
    env = raw.get("_pictograph") if isinstance(raw.get("_pictograph"), dict) else raw
    if isinstance(env, dict):
        for key in ("model_size", "encoder", "backbone", "resolution"):
            if key not in config and env.get(key) is not None:
                config[key] = env[key]
    if _declared_input_shape(env) is not None:
        config.setdefault("image_height", input_shape[0])
        config.setdefault("image_width", input_shape[1])
    return config


def _declared_input_shape(env: Any) -> tuple[int, int] | None:
    """The ``(height, width)`` the artifact DECLARES, or None if it declares none.

    ONE definition of "declared", used by both :func:`_parse_model_config` (which
    substitutes the 640 default when this returns None) and
    :func:`_training_config_of` (which backfills only when it does not). Written
    as a second copy of the rule first, which promptly drifted from the original
    in two ways - a good demonstration of why it is one function:

    * ``"512x512"`` - a STRING is a sequence, so indexing it yielded ``"5"`` and
      ``"1"`` and the model was built at 5x1 rather than falling back.
    * ``[0, 0]`` - parsed without error, so a zero shape passed straight through
      instead of being rejected.

    Both now read as undeclared, which is what they are.
    """
    if not isinstance(env, dict):
        return None
    shape = env.get("input_shape")
    # JSON gives lists; a str/bytes is indexable and would silently yield digits.
    if not isinstance(shape, (list, tuple)):
        return None
    try:
        height, width = int(shape[0]), int(shape[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return (height, width) if height > 0 and width > 0 else None


def _keypoint_schema_of(raw: dict[str, Any]) -> dict[str, Any] | None:
    """The ``keypoint_schema`` block, when the config carries one."""
    env = raw.get("_pictograph") if isinstance(raw.get("_pictograph"), dict) else raw
    if not isinstance(env, dict):
        return None
    schema = env.get("keypoint_schema")
    return schema if isinstance(schema, dict) else None


def _default_client(api_key: str | None) -> Client:
    from pictograph.client import Client

    return Client(api_key=api_key) if api_key else Client()


def _true_input_size(weights: Path, declared: tuple[int, int]) -> tuple[int, int]:
    """The ONNX graph's static (H, W) when it declares one - else ``declared``.

    A model stored with a minimal config otherwise mis-sizes its inputs and ORT
    rejects the tensor (RF-DETR nano: 384 vs the 640 default). Delegates to the
    vendored ``onnx_shape`` rather than keeping a second copy of the same rule.
    """
    try:
        from ._wrappers.onnx_shape import true_input_shape

        return true_input_shape(str(weights), declared)
    except Exception as exc:
        _LOG.debug("input-shape introspection failed (%s) - using %s", exc, tuple(declared))
        return (int(declared[0]), int(declared[1]))


def _classes_of(model: Model) -> list[str]:
    mapping = model.class_mapping or {}
    classes = mapping.get("classes")
    if isinstance(classes, list) and classes:
        return [str(c) for c in classes]
    raise ValueError(f"Model {model.name!r} has no class list to run inference with.")


def _input_size_of(model: Model) -> tuple[int, int]:
    config = getattr(model, "training_config", None) or {}
    try:
        height = int(config.get("image_height") or 640)
        width = int(config.get("image_width") or 640)
    except (TypeError, ValueError):
        height, width = 640, 640
    return height, width


def _cache_dir(cache_dir: str | Path | None) -> Path:
    base = (
        Path(cache_dir)
        if cache_dir is not None
        else Path(os.environ.get("PICTOGRAPH_CACHE_DIR", Path.home() / ".pictograph")) / "models"
    )
    base.mkdir(parents=True, exist_ok=True)
    return base
