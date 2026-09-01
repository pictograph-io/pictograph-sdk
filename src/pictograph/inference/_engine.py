"""The shared engine base, and how a non-ONNX runtime reuses the ONNX wrappers.

Every runtime that executes a *graph* - ONNX Runtime, ExecuTorch, TensorRT - differs
in exactly one thing: the forward pass. The letterboxing, the normalization, the
RF-DETR query reduction, the semantic-seg channel upscale, the NMS, the polygon
extraction - all of that is the same arithmetic, and it already exists once, in the
vendored wrappers under :mod:`pictograph.inference._wrappers`.

So the new runtimes do not reimplement any of it. They substitute a **session** and
let the existing wrapper drive:

.. code-block:: text

    wrapper.preprocess(img)  ─┐
                              ├─→  session.run(...)  ←── THE ONLY PART THAT DIFFERS
    wrapper.postprocess(out) ─┘

This is parity BY CONSTRUCTION rather than parity by test: a difference between
ONNX Runtime and ExecuTorch on the same weights can only come from the numerics of
the forward pass itself, because there is no second copy of anything else to drift.
The alternative - a per-runtime reimplementation of six families' pre/postprocess -
would be four times the surface and would silently diverge the first time one copy
was fixed. (The wrappers are VENDORED from the backend's own inference wrappers
and gated by ``tests/unit/test_inference_wrappers_parity.py``, so forking them
here is not an option anyway; they must stay byte-identical to that source.)

The substitution mechanism
--------------------------
Each wrapper builds its own session in ``__init__``::

    self.session = ort.InferenceSession(self.model_path, providers=..., ...)

There is no injection hook, and adding one would edit a vendored file. So
:func:`build_wrapper_with_session` temporarily rebinds ``onnxruntime.InferenceSession``
for the duration of that one constructor call, under a module lock, and restores it
in a ``finally``. The wrapper then reads ``.get_inputs()`` / ``.get_outputs()`` off
the shim exactly as it would off a real session, and every later ``session.run``
call reaches the substituted runtime.

Two consequences worth stating plainly:

- ``onnxruntime`` is a hard requirement of ALL local graph inference, not just the
  ONNX path - the wrapper modules import it at module scope (``yolox_wrapper`` even
  evaluates ``ort.SessionOptions()`` as a default argument). It ships in the
  ``[inference]`` extra, which every runtime already needs for its preprocessing.
- The rebind is global for a few microseconds, so it is serialized by
  :data:`_BUILD_LOCK`. Nothing else in the SDK constructs an ORT session off-thread.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "NodeArg",
    "RuntimeSession",
    "WrapperEngine",
    "build_wrapper_with_session",
    "input_hw_from",
]

# Serializes the `onnxruntime.InferenceSession` rebind in `build_wrapper_with_session`.
_BUILD_LOCK = threading.Lock()


class NodeArg:
    """An input/output descriptor, shaped like ``onnxruntime.NodeArg``.

    The wrappers read ``.name`` (to key the feed dict) and, for the two RF-DETR
    families, ``.shape`` (to learn the graph's trained resolution). Nothing reads
    ``.type``, but it is carried so the shim is a faithful stand-in.
    """

    __slots__ = ("name", "shape", "type")

    def __init__(self, name: str, shape: Sequence[Any], type_: str = "tensor(float)") -> None:
        self.name = name
        self.shape = list(shape)
        self.type = type_

    def __repr__(self) -> str:
        return f"NodeArg(name={self.name!r}, shape={self.shape!r})"


class RuntimeSession:
    """The subset of ``onnxruntime.InferenceSession`` the vendored wrappers use.

    Subclasses implement :meth:`_forward`. Everything else - name lookup, the
    ``run(output_names, feed)`` contract, provider reporting - is shared, so an
    ExecuTorch and a TensorRT session cannot disagree about the calling convention.

    The input is deliberately named ``input``: :mod:`yolox_wrapper` hardcodes
    ``session.run(None, {"input": ...})`` rather than reading ``get_inputs()[0].name``,
    so a shim that named it anything else would work for five families and raise a
    ``KeyError`` for YOLOX only.
    """

    def __init__(self, *, inputs: list[NodeArg], outputs: list[NodeArg]) -> None:
        self._inputs = inputs
        self._outputs = outputs
        #: What ``.providers`` reports for a model on this session. Each subclass
        #: overwrites it after ``super().__init__()`` with its own execution
        #: targets - ORT providers, ExecuTorch delegates, or the engine's target.
        self.providers: list[str] = []

    def get_inputs(self) -> list[NodeArg]:
        return self._inputs

    def get_outputs(self) -> list[NodeArg]:
        return self._outputs

    def get_providers(self) -> list[str]:
        return list(self.providers)

    def run(self, output_names: Sequence[str] | None, feed: dict[str, Any]) -> list[Any]:
        """Run the graph. Mirrors ORT: ``None`` means "every output, in graph order".

        Args:
            output_names: The outputs to return, or ``None`` for all of them.
            feed: ``{input_name: ndarray}``. These graphs are single-input, so the
                one value is taken regardless of the key the caller used.
        """
        if not feed:
            raise ValueError("No input tensor was provided to run().")
        tensor = next(iter(feed.values()))
        outputs = self._forward(tensor)
        if output_names is None:
            return outputs
        index = {arg.name: i for i, arg in enumerate(self._outputs)}
        return [outputs[index[name]] for name in output_names]

    def _forward(self, tensor: Any) -> list[Any]:
        """One batched NCHW float32 array in, every output array out, in graph order."""
        raise NotImplementedError

    def close(self) -> None:
        """Release the runtime's handles. Idempotent."""


def input_hw_from(session: RuntimeSession, declared: tuple[int, int]) -> tuple[int, int]:
    """The (H, W) the compiled artifact actually takes, falling back to ``declared``.

    The ARTIFACT BEATS THE CONFIG, for the same reason
    ``pictograph.inference._true_input_size`` reads the ONNX graph rather than
    trusting ``training_config``: a model stored with a minimal or drifted config
    otherwise mis-sizes its input and the runtime rejects the tensor (RF-DETR nano
    is 384, not the 640 default). ``.pte`` and ``.engine`` are both AOT-compiled, so
    unlike ONNX their input shape is not merely declared - it is the only shape they
    accept, which makes reading it here strictly more correct than the config.

    Only a fully concrete NCHW shape is trusted; a dynamic axis (reported as ``-1``
    or ``0``) leaves the declared value in place.
    """
    inputs = session.get_inputs()
    if not inputs:
        return declared
    shape = inputs[0].shape
    if len(shape) != 4:
        return declared
    height, width = shape[2], shape[3]
    if not (isinstance(height, int) and isinstance(width, int)):
        return declared
    if height <= 0 or width <= 0:
        return declared
    return (height, width)


def build_wrapper_with_session(session: RuntimeSession, /, **kwargs: Any) -> Any:
    """Build a vendored ONNX wrapper whose forward pass is ``session``.

    See the module docstring for why this rebinds ``onnxruntime.InferenceSession``
    instead of injecting. ``kwargs`` is ``dispatch.build_wrapper``'s signature -
    ``model_type`` / ``architecture`` / ``model_path`` / ``classes`` / ``input_shape``
    / ``confidence_threshold`` / ``providers`` / ``sess_options`` / ``keypoint_schema``.

    ``model_path`` is still required by the wrapper's constructor and is still stored
    on it, but nothing reads the file: the rebind means the path is handed to a
    factory that ignores it. It is passed as the real artifact path anyway so a
    wrapper's ``repr`` and any error it raises name the file the user actually loaded.
    """
    import onnxruntime as ort

    from ._wrappers import dispatch

    def _factory(*_args: Any, **_kwargs: Any) -> RuntimeSession:
        return session

    with _BUILD_LOCK:
        original = ort.InferenceSession
        ort.InferenceSession = _factory
        try:
            return dispatch.build_wrapper(**kwargs)
        finally:
            ort.InferenceSession = original


class WrapperEngine:
    """Holds a vendored wrapper and emits raw annotation dicts.

    The shared body of every graph-executing engine. Subclasses set :attr:`backend`
    and supply the session; the inference path below is identical for all of them,
    which is what makes ``.predict()`` return the same typed result whichever runtime
    produced it.
    """

    #: The runtime that ran it - one of :data:`pictograph.inference.runtime.RUNTIMES`.
    backend = "onnxruntime"

    def __init__(
        self,
        *,
        wrapper: Any,
        model_type: str,
        architecture: str,
        classes: list[str],
        providers: list[str],
        device: str,
        session: RuntimeSession | None = None,
    ) -> None:
        self._wrapper = wrapper
        self._session = session
        self.model_type = model_type
        self.architecture = architecture
        self.classes = classes
        self.providers = providers
        self.device = device
        # The vendored wrappers carry per-call state (e.g. YOLOX's resize ratio),
        # so one engine serializes its own inference.
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(type={self.model_type!r}, device={self.device!r}, "
            f"classes={len(self.classes)})"
        )

    def infer(self, image_bgr: Any, *, confidence: float, top_k: int = 1) -> dict[str, Any]:
        if self._wrapper is None:
            raise RuntimeError("This model has been closed and can no longer predict.")
        from ._wrappers import dispatch

        with self._lock:
            return dispatch.infer_image(
                self._wrapper,
                image_bgr,
                model_type=self.model_type,
                architecture=self.architecture,
                classes=self.classes,
                confidence=confidence,
                top_k=top_k,
            )

    def infer_batch(
        self, images_bgr: list[Any], *, confidence: float, top_k: int = 1
    ) -> list[dict[str, Any]]:
        if self._wrapper is None:
            raise RuntimeError("This model has been closed and can no longer predict.")
        from ._wrappers import dispatch

        with self._lock:
            return dispatch.infer_batch(
                self._wrapper,
                images_bgr,
                model_type=self.model_type,
                architecture=self.architecture,
                classes=self.classes,
                confidence=confidence,
                top_k=top_k,
            )

    def close(self) -> None:
        """Release the wrapper and, when the runtime owns handles, the session."""
        self._wrapper = None
        if self._session is not None:
            self._session.close()
            self._session = None
