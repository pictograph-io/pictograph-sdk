"""The ONNX Runtime inference engine.

A thin adapter over the vendored wrappers in :mod:`pictograph.inference._wrappers`,
giving them the same surface the torch engine exposes so the task model classes can
hold either without caring which.

Its one job beyond dispatch is honesty about where the model ran: it captures
``session.get_providers()`` - the providers ORT actually KEPT - so a CUDA request
that quietly fell back to CPU reports ``cpu``, and warns when that happens.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pictograph.inference._engine import WrapperEngine
from pictograph.inference.runtime import (
    Device,
    check_device_honoured,
    device_label,
    is_explicit,
    resolve_providers,
    session_options,
    warn_on_fallback,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence
    from pathlib import Path

__all__ = ["OnnxEngine", "build_onnx_engine"]

_LOG = logging.getLogger("pictograph.inference")


class OnnxEngine(WrapperEngine):
    """Runs an ONNX Runtime session and emits raw annotation dicts."""

    backend = "onnxruntime"

    def __init__(
        self,
        *,
        wrapper: Any,
        model_type: str,
        architecture: str,
        classes: list[str],
        providers: list[str],
    ) -> None:
        super().__init__(
            wrapper=wrapper,
            model_type=model_type,
            architecture=architecture,
            classes=classes,
            providers=providers,
            device=device_label(providers),
        )


def build_onnx_engine(
    *,
    weights: Path,
    model_type: str,
    architecture: str,
    classes: list[str],
    input_shape: tuple[int, int],
    confidence: float,
    device: Device = "auto",
    cache_dir: Path | None = None,
    keypoint_schema: dict[str, Any] | None = None,
    providers: Sequence[Any] | None = None,
) -> OnnxEngine:
    """Build an :class:`OnnxEngine` from an on-disk ONNX file + resolved config.

    The ONE place the ONNX wrapper is constructed, shared by every loader, so all
    paths produce a byte-identical model.

    ``providers`` is the private measurement escape hatch - see
    :func:`~pictograph.inference.runtime.resolve_providers`. It is not reachable from
    either public loader; ``device=`` is.
    """
    try:
        from ._wrappers import dispatch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Local inference needs the optional extra. Install it with:\n"
            '    pip install "pictograph[inference]"'
        ) from exc

    resolved = resolve_providers(
        device,
        architecture=architecture,
        model_type=model_type,
        cache_dir=cache_dir,
        requested=providers,
    )
    named = providers is None and is_explicit(device)
    wrapper, attempted = _build_with_fallback(
        dispatch,
        resolved,
        named_device=device if named else None,
        model_type=model_type,
        architecture=architecture,
        model_path=str(weights),
        classes=classes,
        input_shape=input_shape,
        confidence_threshold=confidence,
        keypoint_schema=keypoint_schema,
    )
    actual = _session_providers(wrapper)
    if named:
        check_device_honoured(device, actual)
    else:
        warn_on_fallback(attempted, actual)
    _check_class_count(wrapper, model_type, classes)
    return OnnxEngine(
        wrapper=wrapper,
        model_type=model_type,
        architecture=architecture,
        classes=classes,
        providers=actual,
    )


def _build_with_fallback(
    dispatch: Any, resolved: list[Any], *, named_device: str | None, **kwargs: Any
) -> tuple[Any, list[Any]]:
    """Build the wrapper, degrading down the provider ladder if a session build FAILS.

    ONNX Runtime handles the two provider failure modes differently, and only one of
    them is graceful:

    - a provider that fails to REGISTER (missing shared library, wrong driver) is
      dropped and the session still builds, which is what makes "CPU last" a real
      fallback;
    - a provider that registers and then fails to COMPILE THE MODEL **raises**, and
      the whole load dies even though CPU was sitting right there in the list.

    The second is not hypothetical. MEASURED on macOS 15.5 / onnxruntime 1.26: the
    RF-DETR keypoint export raises ``Failed to create MLModel ... error code: -7``
    from the CoreML MLProgram compiler - reproduced against a freshly-cleared
    compiled-model cache, so it is the model, not a stale artifact. Some graphs
    simply cannot be compiled by a given accelerator, and under ``device="auto"``
    the SDK's job is to run the model anyway.

    ``named_device`` is what makes that safe rather than sneaky. Under ``auto`` we
    retry CPU-only, because a ladder nobody named degrading is the ladder working.
    When the caller NAMED the device, the same retry would hand back a model running
    4-10x slower than the one they asked for, so it raises instead - with the
    accelerator's own compiler error attached, which is the actionable part.
    """
    try:
        wrapper = dispatch.build_wrapper(
            providers=resolved, sess_options=session_options(), **kwargs
        )
        return wrapper, resolved
    except Exception as exc:
        cpu_only = ["CPUExecutionProvider"]
        names = [p if isinstance(p, str) else p[0] for p in resolved]
        if names == cpu_only:
            raise
        first_line = str(exc).split("\n")[0][:200]
        if named_device is not None:
            raise RuntimeError(
                f"device={named_device!r} was requested, and ONNX Runtime could not "
                f"build a session for this model with {names}: {first_line}. This "
                f"model's graph is not compilable by that accelerator - it is not a "
                f"missing install. Pass device='cpu' (or 'auto', which falls back "
                f"automatically) to run it here."
            ) from exc
        _LOG.warning(
            "ONNX Runtime could not build a session with %s (%s) - retrying on CPU. "
            "This model's graph is not compilable by that accelerator; pass "
            "device='cpu' to skip the attempt.",
            names,
            first_line,
        )
        wrapper = dispatch.build_wrapper(
            providers=cpu_only, sess_options=session_options(), **kwargs
        )
        # Report CPU as what we ASKED for, so the caller's silent-fallback check does
        # not fire a second, misleading warning telling the user to install a runtime
        # that is in fact installed and simply could not compile this graph.
        return wrapper, cpu_only


def _check_class_count(wrapper: Any, model_type: str, classes: list[str]) -> None:
    """Fail loudly at LOAD time when the class list disagrees with the graph.

    The class count is silently load-bearing, and both directions of a mismatch are
    bad in ways that are hard to trace back from:

    - too MANY declared classes indexes past the model's output and dies deep in the
      emitter with a bare ``IndexError: index 81 is out of bounds for axis 2 with
      size 81`` - no mention of classes, the config, or the model;
    - too FEW silently DROPS every prediction whose class id lands past the end of
      the list, so the model appears to find nothing at all.

    Neither is discoverable from the symptom, so we say it here, in terms of the two
    numbers the caller can actually compare.

    Only the two tasks whose output-to-class relationship is unambiguous are checked.
    The detection families differ by architecture (YOLOX packs ``5 + C`` into its
    last axis, RF-DETR emits ``C + 1`` logits) and a wrong assumption here would
    reject a valid model, which is worse than the bug being fixed - so they are
    deliberately left alone.
    """
    session = getattr(wrapper, "session", None)
    if session is None:
        return
    try:
        shape = session.get_outputs()[0].shape
    except Exception:  # pragma: no cover - defensive
        return
    declared = len(classes)

    if model_type == "classification":
        width = shape[-1] if shape and isinstance(shape[-1], int) else None
        if width is None:
            return
        _report(declared, width, width, "output logits")
    elif model_type == "semantic_segmentation":
        # (batch, C, H, W) - C is channels, NOT the last axis.
        channels = shape[1] if len(shape) == 4 and isinstance(shape[1], int) else None
        if channels is None:
            return
        # Must stay in sync with the pipeline's create_model: a single-class model is ONE
        # sigmoid channel; multi-class is classes PLUS a background channel. So a
        # 1-channel graph supports 1 class, and a C-channel graph supports C - 1.
        supported = 1 if channels == 1 else channels - 1
        _report(declared, supported, channels, "output channels")


def _report(declared: int, supported: int, raw: int, unit: str) -> None:
    """Raise on an unusable class count, warn on a merely suspicious one."""
    if declared > supported:
        raise ValueError(
            f"This model's config declares {declared} classes, but its ONNX graph has "
            f"only {raw} {unit} - room for {supported}. Predictions would index past "
            f"the end of the graph's output. Check the class list in the config.json "
            f"you passed (or the model's class_mapping) against the model you loaded; "
            f"they are probably from different models."
        )
    if declared < supported:
        _LOG.warning(
            "This model's config declares %d classes but its ONNX graph has %d %s "
            "(room for %d). Any prediction for a class id at or above %d will be "
            "SILENTLY DROPPED - the model will appear to find less than it does.",
            declared,
            raw,
            unit,
            supported,
            declared,
        )


def _session_providers(wrapper: Any) -> list[str]:
    """What ORT actually resolved, read off whichever attribute holds the session."""
    session = getattr(wrapper, "session", None)
    if session is not None and hasattr(session, "get_providers"):
        try:
            return [str(p) for p in session.get_providers()]
        except Exception:  # pragma: no cover - defensive
            _LOG.debug("Could not read resolved providers off the ONNX session.")
    return []
