#!/usr/bin/env python3
"""Local-inference runtime-config benchmark (produces the numbers in
``pictograph.inference.runtime``'s module docstring).

Benchmarks the matrix (model family) x (runtime config) for the SDK's local
inference stack (:mod:`pictograph.inference`). For each cell it measures, and
reports SEPARATELY:

- **session/model BUILD time** - cold (no compiled-model cache yet) and warm
  (the SAME on-disk cache reused). This matters: CoreML's MLProgram format
  compiles the graph on first build, and that cost varies enormously by
  architecture (measured: 0.3-0.6 s for most graphs, 31.4 s for RF-DETR).
  "Cold" here means "no cached compiled artifact for this model yet" - the
  same definition :mod:`pictograph.inference.runtime` uses - not a fresh OS
  process; provider-library loading (e.g. the CoreML EP's own dylib) is a
  one-time-per-process cost this harness pays once, up front, before any
  timed cell, so it never leaks into one cell's numbers unfairly.
- **steady-state per-image latency** - p50 / p95 via nearest-rank (reusing
  :func:`benchmarks.load_bench.summarize`), after ``--warmup`` passes.

Runtime configs covered, each auto-skipped (with a printed reason) when the
machine cannot run it:

    onnx-cpu, onnx-coreml-neuralnetwork, onnx-coreml-mlprogram,
    onnx-cuda, onnx-tensorrt, torch-cpu, torch-mps, torch-cuda

Model families are read from whatever of the four Pictograph-trained ONNX
exports are present in ``~/.pictograph/models/`` (override with
``--models-dir`` or ``$PICTOGRAPH_CACHE_DIR``) - classifier, semantic
segmentation, YOLOX detection, RF-DETR detection - plus each one's matching
``.pth`` checkpoint for the torch backend, when present. A download names each
cached file after the model's id, so the filename stems are yours, not ours:
give each family's stem in ``$PICTOGRAPH_BENCH_<FAMILY>`` (see :func:`_stem`).
A model or checkpoint that isn't on disk is SKIPPED, loudly, never silently
omitted.

**Nothing here is trusted from a label.** The class count and input
resolution used to build each model are read straight off the ONNX graph (a
live CPU forward pass, since some graphs declare dynamic output dims that
static protobuf introspection can't resolve) - a wrong class count silently
drops every prediction downstream, which is exactly the failure mode this
guards against.

No API key, no network: every model is built straight from local files via
:func:`pictograph.load_model` (ONNX) or the private
:func:`pictograph.inference._torch.build_torch_engine` (native checkpoint,
driven directly since :func:`pictograph.get_model` with a native ``format=``
normally resolves a model through the API client - see the module README).

Operator-run, needs the ``[inference]`` extra (and ``torch`` for the torch
rows). Runs fine without them too - those rows just report SKIPPED::

    python -m benchmarks.inference_bench
    python benchmarks/inference_bench.py --quick
    python benchmarks/inference_bench.py --warmup 10 --iters 50 --json out.json
    python benchmarks/inference_bench.py --only-models yolox,rf-detr
    python benchmarks/inference_bench.py --only-configs onnx-cpu,onnx-coreml-mlprogram

Re-run this after any change to the provider ladder or the CoreML format table
in :mod:`pictograph.inference.runtime` and reconcile its docstring table with
what comes out here - they are one decision recorded twice.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

# `benchmarks/` is an (unpackaged) namespace dir, invoked either as a script
# (`python benchmarks/inference_bench.py`, sys.path[0] == benchmarks/) or as a
# module (`python -m benchmarks.inference_bench`, cwd == repo root). Try both
# so `summarize` - the nearest-rank percentile helper - is only ever defined
# once, in load_bench.py.
try:
    from load_bench import summarize
except ImportError:
    from benchmarks.load_bench import summarize

__all__ = [
    "MODEL_SPECS",
    "RUNTIME_CONFIGS",
    "BenchRow",
    "ModelSpec",
    "RuntimeConfig",
    "main",
]


# ───────────── the model matrix ─────────────


@dataclass(frozen=True)
class ModelSpec:
    """One known Pictograph-trained model family present in the local cache."""

    label: str
    onnx_stem: str
    """Filename stem (no extension) of the ``.onnx`` export."""
    pth_stem: str | None
    """Filename stem of the matching ``.pth`` checkpoint, or ``None`` if this
    family was only ever exported to ONNX."""
    model_type: str
    """A :data:`pictograph.inference.results.TaskName` value."""
    architecture: str
    """Fed to the CoreML-format policy (:mod:`pictograph.inference.runtime`)
    and to the torch rebuild recipe (:mod:`pictograph.inference._torch`)."""
    torch_family: str
    """Which optional framework the torch backend needs for this family -
    a key into :data:`_TORCH_FRAMEWORK_IMPORT`."""
    torch_config_extra: dict[str, Any] = field(default_factory=dict)
    """Extra ``training_config`` keys the torch rebuild recipe needs beyond
    image size (e.g. the SMP encoder name - not derivable from the checkpoint
    without a much larger auto-detection effort than this harness needs)."""


def _stem(label: str) -> str:
    """The local filename stem for one benchmark family.

    A download names each cached artifact after the model's id, so these stems
    are per-operator: point ``PICTOGRAPH_BENCH_CLASSIFIER`` /
    ``PICTOGRAPH_BENCH_SEMANTIC_SEG`` / ``PICTOGRAPH_BENCH_YOLOX`` /
    ``PICTOGRAPH_BENCH_RF_DETR`` at yours. Unset, the stem falls back to the
    family label, so an unconfigured family finds no file and SKIPs loudly -
    the same path as any absent artifact.
    """
    return os.environ.get(f"PICTOGRAPH_BENCH_{label.upper().replace('-', '_')}", label)


# The four local files this harness benchmarks (see the module docstring).
# Architecture/backbone/encoder values below were VERIFIED by inspecting each
# checkpoint's own tensor shapes (resnet block counts, head widths) against
# pictograph.inference._torch's rebuild recipes - not guessed. Class counts are
# NEVER hardcoded here; see `_infer_num_classes`.
MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        label="classifier",
        onnx_stem=_stem("classifier"),
        pth_stem=_stem("classifier"),
        model_type="classification",
        architecture="resnet18",
        torch_family="torchvision",
    ),
    ModelSpec(
        label="semantic-seg",
        onnx_stem=_stem("semantic-seg"),
        pth_stem=_stem("semantic-seg"),
        model_type="semantic_segmentation",
        architecture="unetplusplus",
        torch_family="segmentation_models_pytorch",
        torch_config_extra={"encoder": "resnet34"},
    ),
    ModelSpec(
        label="yolox",
        onnx_stem=_stem("yolox"),
        pth_stem=_stem("yolox"),
        model_type="object_detection",
        architecture="yolox-s",
        torch_family="yolox",
    ),
    ModelSpec(
        label="rf-detr",
        onnx_stem=_stem("rf-detr"),
        pth_stem=None,  # ONNX-only export on disk - torch rows SKIP, loudly.
        model_type="object_detection",
        architecture="rfdetr-nano",
        torch_family="rfdetr",
    ),
]

_TORCH_FRAMEWORK_IMPORT: dict[str, tuple[str, str]] = {
    # family -> (module to probe, install hint)
    "torchvision": ("torchvision", "pip install torch torchvision"),
    "segmentation_models_pytorch": (
        "segmentation_models_pytorch",
        "pip install segmentation-models-pytorch",
    ),
    "yolox": (
        "yolox.models",
        "pip install git+https://github.com/Megvii-BaseDetection/YOLOX.git --no-deps "
        "&& pip install loguru tabulate psutil ninja opencv-python-headless",
    ),
    "rfdetr": ("rfdetr", "pip install rfdetr"),
}


@dataclass(frozen=True)
class RuntimeConfig:
    """One runtime configuration in the benchmark matrix."""

    name: str
    engine: str  # "onnx" | "torch"
    onnx_provider: str | None = None
    onnx_format: str | None = None  # CoreML "NeuralNetwork" | "MLProgram"
    torch_device: str | None = None


RUNTIME_CONFIGS: list[RuntimeConfig] = [
    RuntimeConfig("onnx-cpu", engine="onnx", onnx_provider="CPUExecutionProvider"),
    RuntimeConfig(
        "onnx-coreml-neuralnetwork",
        engine="onnx",
        onnx_provider="CoreMLExecutionProvider",
        onnx_format="NeuralNetwork",
    ),
    RuntimeConfig(
        "onnx-coreml-mlprogram",
        engine="onnx",
        onnx_provider="CoreMLExecutionProvider",
        onnx_format="MLProgram",
    ),
    RuntimeConfig("onnx-cuda", engine="onnx", onnx_provider="CUDAExecutionProvider"),
    RuntimeConfig("onnx-tensorrt", engine="onnx", onnx_provider="TensorrtExecutionProvider"),
    RuntimeConfig("torch-cpu", engine="torch", torch_device="cpu"),
    RuntimeConfig("torch-mps", engine="torch", torch_device="mps"),
    RuntimeConfig("torch-cuda", engine="torch", torch_device="cuda"),
]


# ───────────── result row ─────────────


@dataclass
class BenchRow:
    """One (model, runtime config) cell's outcome - always emitted, never
    dropped, so a skip is as visible in the report as a fast measurement."""

    model: str
    config: str
    status: str  # "ok" | "skip" | "error"
    device: str = ""
    reason: str = ""
    cache_used: bool = False
    build_cold_ms: float | None = None
    build_warm_ms: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    mean_ms: float | None = None
    n: int = 0


def _skip_row(model: str, config: str, reason: str) -> BenchRow:
    return BenchRow(model=model, config=config, status="skip", reason=reason)


def _error_row(model: str, config: str, reason: str) -> BenchRow:
    return BenchRow(model=model, config=config, status="error", reason=reason)


# ───────────── availability checks - every one returns (ok, reason) ─────────────


def _onnxruntime_available() -> tuple[bool, str]:
    try:
        import onnxruntime  # noqa: F401
    except ImportError as exc:
        return False, f"onnxruntime not installed ({exc}); pip install 'pictograph[inference]'"
    return True, ""


def _onnx_provider_available(name: str) -> tuple[bool, str]:
    ok, reason = _onnxruntime_available()
    if not ok:
        return ok, reason
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if name not in available:
        return False, f"{name} not in onnxruntime.get_available_providers() ({sorted(available)})"
    return True, ""


def _coreml_platform_ok() -> tuple[bool, str]:
    if platform.system() != "Darwin":
        return False, "CoreMLExecutionProvider only runs on macOS"
    return True, ""


def _onnx_config_availability(rc: RuntimeConfig) -> tuple[bool, str]:
    if rc.onnx_format:  # CoreML
        ok, reason = _coreml_platform_ok()
        if not ok:
            return ok, reason
    return _onnx_provider_available(rc.onnx_provider or "")


def _torch_available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        return False, f"torch not installed ({exc}); pip install torch"
    return True, ""


def _torch_device_available(device: str) -> tuple[bool, str]:
    ok, reason = _torch_available()
    if not ok:
        return ok, reason
    import torch

    if device == "cpu":
        return True, ""
    if device == "cuda":
        ok = torch.cuda.is_available()
        return (True, "") if ok else (False, "torch.cuda.is_available() is False")
    if device == "mps":
        backend = getattr(torch.backends, "mps", None)
        ok = bool(backend is not None and backend.is_available() and backend.is_built())
        return (True, "") if ok else (False, "torch.backends.mps not available/built")
    return False, f"unknown torch device {device!r}"  # pragma: no cover - defensive


def _torch_framework_available(family: str) -> tuple[bool, str]:
    entry = _TORCH_FRAMEWORK_IMPORT.get(family)
    if entry is None:  # pragma: no cover - defensive, registry bug
        return False, f"unknown torch family {family!r}"
    module, hint = entry
    try:
        import importlib

        importlib.import_module(module)
    except ImportError as exc:
        return False, f"{module} not installed ({exc}); {hint}"
    return True, ""


# ───────────── ONNX graph introspection - never trust a declared label ─────────────


def _probe_onnx_graph(weights: Path) -> dict[str, Any]:
    """Run one CPU forward pass to read the graph's TRUE input/output shapes.

    Static protobuf introspection (``onnx.load`` + shape inference) leaves
    YOLOX's output dims symbolic (data-dependent Reshape/Concat), so a live
    forward pass on a zero tensor is the only reliable way to read every
    family's real output width - which is exactly what class-count inference
    needs.
    """
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    dims = inp.shape
    height = dims[2] if len(dims) == 4 and isinstance(dims[2], int) else 640
    width = dims[3] if len(dims) == 4 and isinstance(dims[3], int) else 640
    dummy = np.zeros((1, 3, height, width), dtype=np.float32)
    outputs = session.run(None, {inp.name: dummy})
    names = [o.name for o in session.get_outputs()]
    del session
    return {
        "input_hw": (int(height), int(width)),
        "output_names": names,
        "output_shapes": [tuple(int(d) for d in o.shape) for o in outputs],
    }


def _infer_num_classes(model_type: str, probe: dict[str, Any]) -> int:
    """The model's true class count, read off the graph's output width.

    Conventions (all verified against the graphs shipped in this repo, not
    assumed - see the module docstring):

    - classification: the one output IS the per-class logit vector.
    - semantic segmentation: channel 0 of the (C,H,W) output is background
      for a multi-class head (:mod:`pictograph.inference._torch`'s
      ``_build_smp``: ``n_classes = classes+1`` unless single-class).
    - a single-output detection head (YOLOX): last dim is
      ``4 box + 1 objectness + classes``.
    - a multi-output detection/instance-seg head (RF-DETR family): the
      per-box class-logits output has a TRAILING background slot (see
      ``pictograph.inference._wrappers.onnx_shape.rfdetr_foreground_columns``,
      which documents the same convention).
    """
    shapes = probe["output_shapes"]
    names = probe["output_names"]
    if model_type == "classification":
        return int(shapes[0][-1])
    if model_type == "semantic_segmentation":
        channels = int(shapes[0][1])
        return channels - 1 if channels > 1 else 1
    if len(shapes) == 1:
        width = int(shapes[0][-1])
        inferred = width - 5
        if inferred > 0:
            return inferred
    for _name, shape in zip(names, shapes, strict=True):
        if len(shape) == 3 and shape[-1] > 4:
            return int(shape[-1] - 1)
    raise ValueError(
        f"Could not infer a class count for model_type={model_type!r} from outputs "
        f"{list(zip(names, shapes, strict=True))!r}"
    )


def _synthetic_image(hw: tuple[int, int]) -> Any:
    """A deterministic pseudo-random BGR frame at the model's native size.

    Content doesn't matter for a latency benchmark (only shape does), and a
    fixed seed keeps the same frame across every runtime config for one model
    so cross-config comparisons aren't confounded by different inputs.
    """
    import numpy as np

    rng = np.random.default_rng(20260730)
    height, width = hw
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


# ───────────── ONNX cell ─────────────


def _onnx_providers_for(rc: RuntimeConfig, cache_dir: Path) -> list[Any]:
    if rc.onnx_format:
        opts = {"ModelFormat": rc.onnx_format, "ModelCacheDirectory": str(cache_dir)}
        return [(rc.onnx_provider, opts), "CPUExecutionProvider"]
    if rc.onnx_provider == "CPUExecutionProvider":
        return ["CPUExecutionProvider"]
    if rc.onnx_provider == "CUDAExecutionProvider":
        return [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
            "CPUExecutionProvider",
        ]
    if rc.onnx_provider == "TensorrtExecutionProvider":
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    raise ValueError(f"unhandled onnx runtime config {rc.name!r}")  # pragma: no cover - defensive


def _bench_onnx_cell(
    spec: ModelSpec,
    rc: RuntimeConfig,
    weights: Path,
    config: dict[str, Any],
    image_hw: tuple[int, int],
    warmup: int,
    iters: int,
) -> BenchRow:
    cache_used = bool(rc.onnx_format)
    cache_dir = Path(tempfile.mkdtemp(prefix="pictograph-bench-cache-"))
    try:
        t0 = time.perf_counter()
        model = _pinned_onnx_model(weights, config, spec, rc, cache_dir)
        build_cold_ms = (time.perf_counter() - t0) * 1000.0
        device = model.device
        model.close()

        t1 = time.perf_counter()
        model = _pinned_onnx_model(weights, config, spec, rc, cache_dir)
        build_warm_ms = (time.perf_counter() - t1) * 1000.0

        image = _synthetic_image(image_hw)
        for _ in range(warmup):
            model.predict(image)
        samples = [float(model.predict(image).inference_ms or 0.0) for _ in range(iters)]
        model.close()
    except Exception as exc:
        return _error_row(spec.label, rc.name, f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    stats = summarize(samples)
    return BenchRow(
        model=spec.label,
        config=rc.name,
        status="ok",
        device=device,
        cache_used=cache_used,
        build_cold_ms=build_cold_ms,
        build_warm_ms=build_warm_ms,
        p50_ms=stats["p50"],
        p95_ms=stats["p95"],
        mean_ms=stats["mean"],
        n=int(stats["count"]),
    )


# ───────────── torch cell ─────────────


def _model_record(spec: ModelSpec, n_classes: int, input_hw: tuple[int, int]) -> Any:
    """A minimal, LOCAL ``Model`` record - no API call, just the fields the
    torch rebuild recipe (:mod:`pictograph.inference._torch`) reads."""
    from pictograph.models.model import Model

    height, width = input_hw
    return Model.model_validate(
        {
            "id": spec.pth_stem,
            "organization_id": "inference-bench",
            "name": f"bench-{spec.label}",
            "model_type": spec.model_type,
            "architecture": spec.architecture,
            "visibility": "private",
            "status": "ready",
            "class_mapping": {"classes": [f"class_{i}" for i in range(n_classes)]},
            "training_config": {
                "image_height": height,
                "image_width": width,
                **spec.torch_config_extra,
            },
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
    )


def _bench_torch_cell(
    spec: ModelSpec,
    rc: RuntimeConfig,
    pth_path: Path,
    n_classes: int,
    image_hw: tuple[int, int],
    warmup: int,
    iters: int,
) -> BenchRow:
    from pictograph.inference._torch import _cache_stem, build_torch_engine
    from pictograph.inference.models import TASK_MODEL_TYPES

    record = _model_record(spec, n_classes, image_hw)
    cache_dir = Path(tempfile.mkdtemp(prefix="pictograph-bench-torch-"))
    try:
        staged = cache_dir / f"{_cache_stem(record)}.pth"
        shutil.copy(pth_path, staged)

        def _build() -> Any:
            # `models=None`: weights are pre-staged at the path build_torch_engine
            # computes, so `.download()` is never reached - no `Models` resource needed.
            return build_torch_engine(
                record,
                models=None,  # type: ignore[arg-type]
                cache_dir=cache_dir,
                # A named device, which is now also a guarantee: `build_torch_engine`
                # raises rather than quietly benchmarking a different one.
                device=rc.torch_device or "auto",
            )

        t0 = time.perf_counter()
        engine = _build()
        build_cold_ms = (time.perf_counter() - t0) * 1000.0
        device = engine.device
        engine.close()

        t1 = time.perf_counter()
        engine = _build()
        build_warm_ms = (time.perf_counter() - t1) * 1000.0

        wrapped = TASK_MODEL_TYPES[spec.model_type](
            engine=engine,
            model_id=record.id,
            name=record.name,
            architecture=record.architecture or "",
            confidence=0.5,
        )
        image = _synthetic_image(image_hw)
        for _ in range(warmup):
            wrapped.predict(image)
        samples = [float(wrapped.predict(image).inference_ms or 0.0) for _ in range(iters)]
        wrapped.close()
    except Exception as exc:
        return _error_row(spec.label, rc.name, f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    stats = summarize(samples)
    return BenchRow(
        model=spec.label,
        config=rc.name,
        status="ok",
        device=device,
        cache_used=False,
        build_cold_ms=build_cold_ms,
        build_warm_ms=build_warm_ms,
        p50_ms=stats["p50"],
        p95_ms=stats["p95"],
        mean_ms=stats["mean"],
        n=int(stats["count"]),
    )


# ───────────── auto-policy sanity check ─────────────


def _pinned_onnx_model(
    weights: Path,
    config: dict[str, Any],
    spec: ModelSpec,
    rc: RuntimeConfig,
    cache_dir: Path,
) -> Any:
    """One ONNX model on EXACTLY the provider configuration this row names.

    The benchmark compares provider configurations that `device=` deliberately does
    not expose separately - CoreML NeuralNetwork vs MLProgram is one device and two
    compilers, and the ORT TensorRT provider is excluded from `auto` on the strength
    of numbers measured HERE. So this drops to `build_onnx_engine`'s private
    `providers=` hatch rather than the public loader, which is the reason that hatch
    still exists. `_wrap` puts the same task class around it, so every row is the
    object a user would hold.
    """
    from pictograph.inference import _parse_model_config, _true_input_size, _wrap
    from pictograph.inference._onnx import build_onnx_engine

    model_type, architecture, classes, input_shape, name = _parse_model_config(config)
    engine = build_onnx_engine(
        weights=weights,
        model_type=model_type,
        architecture=architecture,
        classes=classes,
        input_shape=_true_input_size(weights, input_shape),
        confidence=0.5,
        cache_dir=cache_dir,
        providers=_onnx_providers_for(rc, cache_dir),
    )
    return _wrap(spec.model_type, engine, name or weights.stem, name, architecture, 0.5)


def _config_name_for_providers(providers: Sequence[Any]) -> str:
    """Map a resolved ORT provider ladder back to one of our config names."""
    for entry in providers:
        name = entry if isinstance(entry, str) else entry[0]
        if name == "CoreMLExecutionProvider":
            opts = entry[1] if isinstance(entry, tuple) else {}
            fmt = str(opts.get("ModelFormat", "NeuralNetwork")).lower()
            return f"onnx-coreml-{fmt}"
        if name == "CUDAExecutionProvider":
            return "onnx-cuda"
        if name == "TensorrtExecutionProvider":
            return "onnx-tensorrt"
    return "onnx-cpu"


def _auto_policy_check(spec: ModelSpec, rows: list[BenchRow], scratch: Path) -> str:
    """Does ``device='auto'`` actually pick the fastest measured config?

    This is the harness re-verifying the LOCKSTEP claim in
    ``pictograph.inference.runtime``'s module docstring against what was just
    measured, rather than trusting the table.
    """
    from pictograph.inference.runtime import resolve_providers

    ok_rows = {
        r.config: r
        for r in rows
        if r.model == spec.label and r.status == "ok" and r.config.startswith("onnx-")
    }
    if not ok_rows:
        return f"{spec.label}: no successful ONNX rows to check the auto policy against."

    chosen = resolve_providers(
        "auto",
        architecture=spec.architecture,
        model_type=spec.model_type,
        cache_dir=scratch,
    )
    chosen_name = _config_name_for_providers(chosen)
    chosen_row = ok_rows.get(chosen_name)
    fastest = min(
        ok_rows.values(), key=lambda r: r.p50_ms if r.p50_ms is not None else float("inf")
    )

    if chosen_row is None:
        return (
            f"{spec.label}: device='auto' would pick {chosen_name!r}, which wasn't "
            f"benchmarked/available here - cannot verify."
        )
    if chosen_row.config == fastest.config:
        return (
            f"{spec.label}: device='auto' -> {chosen_name} matches the fastest measured "
            f"config ({fastest.p50_ms:.2f} ms p50). OK."
        )
    c_p50 = chosen_row.p50_ms if chosen_row.p50_ms is not None else float("nan")
    f_p50 = fastest.p50_ms if fastest.p50_ms is not None else float("nan")
    return (
        f"{spec.label}: device='auto' -> {chosen_name} measured {c_p50:.2f} ms p50, but "
        f"{fastest.config} measured FASTER ({f_p50:.2f} ms p50, {c_p50 - f_p50:+.2f} ms). "
        f"runtime.py's policy table may need updating."
    )


# ───────────── reporting ─────────────

_COLUMNS = "{:<13} {:<26} {:<6} {:<9} {:>10} {:>10} {:>8} {:>8} {:>8} {:>4}"


def _fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def format_table(rows: list[BenchRow]) -> str:
    lines = [
        _COLUMNS.format(
            "model", "config", "status", "device", "cold(ms)", "warm(ms)", "p50", "p95", "mean", "n"
        ),
    ]
    lines.append("-" * len(lines[0]))
    for r in rows:
        lines.append(
            _COLUMNS.format(
                r.model,
                r.config,
                r.status.upper(),
                r.device or "-",
                _fmt_ms(r.build_cold_ms),
                _fmt_ms(r.build_warm_ms),
                _fmt_ms(r.p50_ms),
                _fmt_ms(r.p95_ms),
                _fmt_ms(r.mean_ms),
                r.n,
            )
        )
        if r.status != "ok":
            # A skip/error is never just another row - the reason is printed
            # right under it, so it cannot be mistaken for a fast measurement.
            lines.append(f"             -> {r.reason}")
    return "\n".join(lines)


def _environment_header() -> list[str]:
    lines = [f"platform:    {platform.platform()}"]
    try:
        import onnxruntime as ort

        lines.append(f"onnxruntime: {ort.__version__}  providers={ort.get_available_providers()}")
    except ImportError:
        lines.append("onnxruntime: NOT INSTALLED - pip install 'pictograph[inference]'")
    try:
        import torch

        mps_backend = getattr(torch.backends, "mps", None)
        mps = bool(
            mps_backend is not None and mps_backend.is_available() and mps_backend.is_built()
        )
        lines.append(
            f"torch:       {torch.__version__}  cuda={torch.cuda.is_available()}  mps={mps}"
        )
    except ImportError:
        lines.append("torch:       NOT INSTALLED - pip install torch")
    return lines


# ───────────── orchestration ─────────────


def _default_models_dir() -> Path:
    base = Path(os.environ.get("PICTOGRAPH_CACHE_DIR", Path.home() / ".pictograph"))
    return base / "models"


def run_matrix(
    *,
    models_dir: Path,
    specs: list[ModelSpec],
    configs: list[RuntimeConfig],
    warmup: int,
    iters: int,
    verbose: bool = True,
) -> list[BenchRow]:
    """Run every (model, config) cell, never letting one failure drop the rest."""
    rows: list[BenchRow] = []
    scratch = Path(tempfile.mkdtemp(prefix="pictograph-bench-scratch-"))
    try:
        for spec in specs:
            onnx_path = models_dir / f"{spec.onnx_stem}.onnx"
            if not onnx_path.exists():
                reason = f"{onnx_path.name} not found under {models_dir}"
                if verbose:
                    print(f"[{spec.label}] SKIP all configs - {reason}")
                rows.extend(_skip_row(spec.label, rc.name, reason) for rc in configs)
                continue

            try:
                probe = _probe_onnx_graph(onnx_path)
                n_classes = _infer_num_classes(spec.model_type, probe)
            except Exception as exc:
                reason = f"graph introspection failed: {type(exc).__name__}: {exc}"
                if verbose:
                    print(f"[{spec.label}] ERROR all configs - {reason}")
                rows.extend(_error_row(spec.label, rc.name, reason) for rc in configs)
                continue

            input_hw = probe["input_hw"]
            if verbose:
                print(
                    f"[{spec.label}] detected: input={input_hw[0]}x{input_hw[1]}  "
                    f"classes={n_classes}  outputs={probe['output_names']}"
                )
            onnx_config = {
                "model_type": spec.model_type,
                "architecture": spec.architecture,
                "class_mapping": {"classes": [f"class_{i}" for i in range(n_classes)]},
                "input_shape": [input_hw[0], input_hw[1]],
            }
            pth_path = models_dir / f"{spec.pth_stem}.pth" if spec.pth_stem else None

            for rc in configs:
                if rc.engine == "onnx":
                    ok, reason = _onnx_config_availability(rc)
                    if not ok:
                        if verbose:
                            print(f"[{spec.label}] SKIP {rc.name} - {reason}")
                        rows.append(_skip_row(spec.label, rc.name, reason))
                        continue
                    if verbose:
                        print(f"[{spec.label}] {rc.name}: building + benchmarking...")
                    row = _bench_onnx_cell(
                        spec, rc, onnx_path, onnx_config, input_hw, warmup, iters
                    )
                else:
                    if pth_path is None or not pth_path.exists():
                        reason = (
                            f"no matching .pth checkpoint for {spec.label} "
                            f"(only the .onnx export is on disk)"
                            if pth_path is None
                            else f"{pth_path.name} not found under {models_dir}"
                        )
                        if verbose:
                            print(f"[{spec.label}] SKIP {rc.name} - {reason}")
                        rows.append(_skip_row(spec.label, rc.name, reason))
                        continue
                    ok, reason = _torch_device_available(rc.torch_device or "")
                    if not ok:
                        if verbose:
                            print(f"[{spec.label}] SKIP {rc.name} - {reason}")
                        rows.append(_skip_row(spec.label, rc.name, reason))
                        continue
                    ok, reason = _torch_framework_available(spec.torch_family)
                    if not ok:
                        if verbose:
                            print(f"[{spec.label}] SKIP {rc.name} - {reason}")
                        rows.append(_skip_row(spec.label, rc.name, reason))
                        continue
                    if verbose:
                        print(f"[{spec.label}] {rc.name}: building + benchmarking...")
                    row = _bench_torch_cell(spec, rc, pth_path, n_classes, input_hw, warmup, iters)
                if verbose:
                    status = row.status.upper()
                    detail = (
                        f"cold={_fmt_ms(row.build_cold_ms)}ms p50={_fmt_ms(row.p50_ms)}ms"
                        if row.status == "ok"
                        else row.reason
                    )
                    print(f"[{spec.label}] {rc.name}: {status} - {detail}")
                rows.append(row)
        return rows
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _prewarm_imports() -> None:
    """Pay one-time provider/framework import cost BEFORE any timed cell.

    Otherwise whichever cell happens to run first (e.g. the classifier's
    onnx-cpu row) unfairly absorbs the C-extension load time for
    ``onnxruntime``/``torch`` and looks slower than later rows for no reason
    connected to the runtime config being measured.
    """
    for mod in ("onnxruntime", "onnx", "torch", "numpy", "cv2"):
        with contextlib.suppress(ImportError):
            __import__(mod)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Where the .onnx/.pth files live (default: $PICTOGRAPH_CACHE_DIR/models "
        "or ~/.pictograph/models).",
    )
    parser.add_argument(
        "--warmup", type=int, default=None, help="Warmup passes (default: 5, or 1 with --quick)."
    )
    parser.add_argument(
        "--iters", type=int, default=None, help="Timed iterations (default: 20, or 3 with --quick)."
    )
    parser.add_argument(
        "--quick", action="store_true", help="Fast smoke run: warmup=1, iters=3 unless overridden."
    )
    parser.add_argument(
        "--only-models",
        default=None,
        help=f"Comma-separated model labels to run (default: all - {[s.label for s in MODEL_SPECS]}).",
    )
    parser.add_argument(
        "--only-configs",
        default=None,
        help=f"Comma-separated runtime configs to run (default: all - {[c.name for c in RUNTIME_CONFIGS]}).",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Write machine-readable results here."
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress per-cell progress lines."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    specs = MODEL_SPECS
    if args.only_models:
        wanted = {s.strip() for s in args.only_models.split(",") if s.strip()}
        known = {s.label for s in MODEL_SPECS}
        if not wanted <= known:
            print(
                f"Unknown model label(s): {sorted(wanted - known)}. Known: {sorted(known)}",
                file=sys.stderr,
            )
            return 2
        specs = [s for s in MODEL_SPECS if s.label in wanted]

    configs = RUNTIME_CONFIGS
    if args.only_configs:
        wanted_c = {c.strip() for c in args.only_configs.split(",") if c.strip()}
        known_c = {c.name for c in RUNTIME_CONFIGS}
        if not wanted_c <= known_c:
            print(
                f"Unknown config(s): {sorted(wanted_c - known_c)}. Known: {sorted(known_c)}",
                file=sys.stderr,
            )
            return 2
        configs = [c for c in RUNTIME_CONFIGS if c.name in wanted_c]

    warmup = args.warmup if args.warmup is not None else (1 if args.quick else 5)
    iters = args.iters if args.iters is not None else (3 if args.quick else 20)
    models_dir = args.models_dir or _default_models_dir()

    print("Pictograph local-inference runtime-config benchmark")
    print("=" * 52)
    for line in _environment_header():
        print(line)
    print(f"models-dir:  {models_dir}")
    print(f"warmup={warmup}  iters={iters}")
    print()

    _prewarm_imports()
    rows = run_matrix(
        models_dir=models_dir,
        specs=specs,
        configs=configs,
        warmup=warmup,
        iters=iters,
        verbose=not args.quiet,
    )

    print()
    print(format_table(rows))

    print()
    print("Auto-policy sanity check (device='auto' vs the fastest measured config)")
    print("-" * 76)
    scratch = Path(tempfile.mkdtemp(prefix="pictograph-bench-policy-"))
    try:
        for spec in specs:
            print(_auto_policy_check(spec, rows, scratch))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if args.json:
        payload = [dataclasses.asdict(r) for r in rows]
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {len(payload)} rows to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
