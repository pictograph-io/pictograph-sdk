# ------------------------------------------------------------------------
# Pictograph - original work, not vendored.
# ------------------------------------------------------------------------
"""The `transformers` surface the vendored DINOv2 backbone uses, reimplemented.

RF-DETR's backbone is a local copy of HuggingFace's DINOv2-with-registers, so it
inherits from four `transformers` base classes and calls a handful of its helpers.
That is the ONLY reason `pip install rfdetr` used to drag `transformers>=5.1,<6`
(and its own resolver constraints) into a user's environment just to rebuild an
architecture whose weights we already ship.

None of that machinery is reachable on a reload. `PreTrainedModel` exists to
serve `from_pretrained` - hub resolution, sharded safetensors, quantization,
device maps, dispatch hooks - and we never call it: the weights come from OUR
checkpoint, loaded by `models/weights.py`. What the backbone actually needs from
those base classes is small enough to state exactly, which is what this module is:

* `PretrainedConfig`  - an attribute bag with the four inference-relevant defaults.
* `BackboneConfigMixin` - `out_features`/`out_indices` kept in sync with `stage_names`.
* `PreTrainedModel`   - `nn.Module` + `config` + `post_init()`'s weight init.
* `BackboneMixin`     - the same alignment, applied to the module.
* `ACT2FN`, `torch_int`, `prune_linear_layer` - three small helpers.
* The `ModelOutput` dataclasses the forwards return.
* Four documentation decorators, which are no-ops at runtime.

The weight init in `post_init()` is retained rather than dropped even though every
initialised tensor is immediately overwritten by the checkpoint: a partial load
must land on the SAME distribution the real class would have produced, or a
missing tensor would differ between this path and rfdetr's.

`_attn_implementation` defaults to `"sdpa"`, matching what `transformers` selects
for a model declaring `_supports_sdpa` on any supported torch. Eager and SDPA are
the same attention over the same parameter names, so this changes neither the
state-dict contract nor the result beyond float accumulation order.
"""

from __future__ import annotations

import logging as _stdlib_logging
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterator, Sequence

__all__ = [
    "ACT2FN",
    "BackboneConfigMixin",
    "BackboneMixin",
    "BackboneOutput",
    "BaseModelOutput",
    "BaseModelOutputWithPooling",
    "ImageClassifierOutput",
    "PreTrainedModel",
    "PretrainedConfig",
    "add_start_docstrings",
    "add_start_docstrings_to_model_forward",
    "logging",
    "prune_linear_layer",
    "replace_return_docstrings",
    "torch_int",
]


# ───────────────────────── logging ─────────────────────────


class _Logger(_stdlib_logging.Logger):
    """A stdlib logger plus `warning_once`, which `transformers.utils.logging` adds.

    The vendored backbone calls `warning_once` for genuinely once-per-process
    conditions (a positional-encoding size that means DINOv2 hub weights are not
    being loaded, an SDPA→eager fallback). Logging those per forward would be noise.
    """

    _warned_once: set[tuple[str, tuple[Any, ...]]]

    def warning_once(self, msg: str, *args: Any, **kwargs: Any) -> None:
        key = (msg, args)
        if key in self._warned_once:
            return
        self._warned_once.add(key)
        self.warning(msg, *args, **kwargs)


class _LoggingModule:
    """Stands in for `transformers.utils.logging`."""

    @staticmethod
    def get_logger(name: str) -> _Logger:
        # `getLogger` hands back the process-wide singleton for this name, so the
        # `warning_once` state has to be grafted onto that instance rather than
        # created by a subclass constructor we never get to call.
        logger = _stdlib_logging.getLogger(name)
        if not hasattr(logger, "warning_once"):
            logger._warned_once = set()  # type: ignore[attr-defined]
            logger.warning_once = _Logger.warning_once.__get__(logger)  # type: ignore[attr-defined]
        return cast("_Logger", logger)


logging = _LoggingModule()


# ───────────────────────── activations ─────────────────────────

ACT2FN: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gelu": nn.functional.gelu,
    "gelu_new": lambda x: (
        0.5
        * x
        * (
            1.0
            + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3.0))
            )
        )
    ),
    "gelu_pytorch_tanh": lambda x: nn.functional.gelu(x, approximate="tanh"),
    "relu": nn.functional.relu,
    "relu6": nn.functional.relu6,
    "selu": nn.functional.selu,
    "silu": nn.functional.silu,
    "swish": nn.functional.silu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "quick_gelu": lambda x: x * torch.sigmoid(1.702 * x),
    "mish": nn.functional.mish,
    "linear": lambda x: x,
}


# ───────────────────────── helpers ─────────────────────────


def torch_int(value: Any) -> Any:
    """`int(value)`, except a tensor stays a tensor - `transformers.utils.torch_int`.

    The distinction matters when the graph is being traced for export: collapsing a
    symbolic dimension to a Python int would bake the current resolution into the
    exported model. Upstream reaches that conclusion via `torch.jit.is_tracing()`;
    keeping a tensor as an int64 tensor unconditionally is the same outcome for both
    call sites here (`interpolate_pos_encoding`, which passes either a plain int or a
    traced dimension) and does not depend on a private torch attribute.
    """
    if isinstance(value, torch.Tensor):
        return value.to(torch.int64)
    return int(value)


def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """A `nn.Linear` with only `index`'s rows (or columns) kept.

    Reachable only through `_prune_heads`, which is a training-time surgery the
    reload path never performs; kept so the vendored class stays complete.
    """
    keep = index.to(layer.weight.device)
    weight = layer.weight.index_select(dim, keep).clone().detach()
    bias = None
    if layer.bias is not None:
        bias = layer.bias.clone().detach() if dim == 1 else layer.bias[keep].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(keep)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(
        layer.weight.device
    )
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(weight.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None and bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(bias.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def _noop_decorator(*_args: Any, **_kwargs: Any) -> Callable[[Any], Any]:
    def wrap(obj: Any) -> Any:
        return obj

    return wrap


add_start_docstrings = _noop_decorator
add_start_docstrings_to_model_forward = _noop_decorator
replace_return_docstrings = _noop_decorator


# ───────────────────────── model outputs ─────────────────────────


class _ModelOutput:
    """The ordered-dict behaviour of `transformers.utils.ModelOutput`.

    The vendored forwards build these and RF-DETR's backbone then reads them
    BOTH by attribute (`outputs.last_hidden_state`) and by index (`outputs[0]`),
    with `None` fields skipped in the positional view - so indexing is not simply
    field order.
    """

    def _present(self) -> Iterator[tuple[str, Any]]:
        for field in fields(self):  # type: ignore[arg-type]
            value = getattr(self, field.name)
            if value is not None:
                yield field.name, value

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return getattr(self, key)
        return tuple(v for _, v in self._present())[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(name for name, _ in self._present())

    def __len__(self) -> int:
        return sum(1 for _ in self._present())

    def keys(self) -> list[str]:
        return [name for name, _ in self._present()]

    def values(self) -> list[Any]:
        return [value for _, value in self._present()]

    def items(self) -> list[tuple[str, Any]]:
        return list(self._present())

    def to_tuple(self) -> tuple[Any, ...]:
        return tuple(v for _, v in self._present())


@dataclass
class BaseModelOutput(_ModelOutput):
    last_hidden_state: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


@dataclass
class BaseModelOutputWithPooling(_ModelOutput):
    last_hidden_state: torch.Tensor | None = None
    pooler_output: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


@dataclass
class BackboneOutput(_ModelOutput):
    feature_maps: tuple[torch.Tensor, ...] | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


@dataclass
class ImageClassifierOutput(_ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


# ───────────────────────── config ─────────────────────────


class PretrainedConfig:
    """An attribute bag, which is all `transformers.PretrainedConfig` is here.

    The real class additionally handles serialisation, hub round-trips and a long
    tail of deprecated aliases. The vendored backbone assigns every field it reads
    in its own `__init__`; only the four below are read WITHOUT being assigned
    there, so only those need defaults.
    """

    model_type: str = ""

    def __init__(self, **kwargs: Any) -> None:
        self.return_dict = kwargs.pop("return_dict", True)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self._attn_implementation = kwargs.pop("attn_implementation", None) or kwargs.pop(
            "_attn_implementation", "sdpa"
        )
        self.num_labels = kwargs.pop("num_labels", 2)
        self.problem_type = kwargs.pop("problem_type", None)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def use_return_dict(self) -> bool:
        return bool(self.return_dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("__")}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.to_dict()!r})"


class BackboneConfigMixin:
    """`out_features` / `out_indices` kept aligned with `stage_names`.

    Verbatim in behaviour with `transformers.utils.BackboneConfigMixin`; the
    vendored config writes `_out_features` / `_out_indices` directly and reads
    them back through these properties.
    """

    stage_names: list[str]
    _out_features: list[str]
    _out_indices: list[int]

    @property
    def out_features(self) -> list[str]:
        return self._out_features

    @out_features.setter
    def out_features(self, out_features: list[str]) -> None:
        self._out_features, self._out_indices = _align(out_features, None, self.stage_names)

    @property
    def out_indices(self) -> list[int]:
        return self._out_indices

    @out_indices.setter
    def out_indices(self, out_indices: Sequence[int]) -> None:
        self._out_features, self._out_indices = _align(None, out_indices, self.stage_names)


def _align(
    out_features: list[str] | None,
    out_indices: Sequence[int] | None,
    stage_names: list[str],
) -> tuple[list[str], list[int]]:
    if out_indices is None and out_features is None:
        return [stage_names[-1]], [len(stage_names) - 1]
    if out_indices is None and out_features is not None:
        return list(out_features), [stage_names.index(layer) for layer in out_features]
    if out_features is None and out_indices is not None:
        return [stage_names[idx] for idx in out_indices], list(out_indices)
    return list(out_features or []), list(out_indices or [])


# ───────────────────────── models ─────────────────────────


class PreTrainedModel(nn.Module):
    """`nn.Module` + a `config` + `post_init()`, the whole of what the backbone uses.

    `transformers`' own class is ~5k lines, essentially all of it in service of
    `from_pretrained` / `save_pretrained` / hub + quantization plumbing that a
    checkpoint reload never reaches.
    """

    config_class: Any = None
    base_model_prefix: str = ""
    main_input_name: str = "pixel_values"
    supports_gradient_checkpointing: bool = False
    _no_split_modules: list[str] | None = None
    _supports_sdpa: bool = False

    def __init__(self, config: Any, *_args: Any, **_kwargs: Any) -> None:
        super().__init__()
        self.config = config

    def post_init(self) -> None:
        """Apply the subclass's `_init_weights` across the tree, as HF's does.

        Every tensor this touches is then overwritten by the checkpoint. It runs
        anyway so that a tensor the checkpoint does NOT carry is initialised from
        the same distribution `transformers` would have used.
        """
        self.init_weights()

    def init_weights(self) -> None:
        if hasattr(self, "_init_weights"):
            self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:  # noqa: ARG002 - overridden
        """Overridden by each vendored subclass; the base is a no-op, as HF's is."""
        return

    def _prune_heads(self, heads_to_prune: dict[int, list[int]]) -> None:  # pragma: no cover
        raise NotImplementedError

    def prune_heads(self, heads_to_prune: dict[int, list[int]]) -> None:  # pragma: no cover
        self._prune_heads(heads_to_prune)

    def get_input_embeddings(self) -> nn.Module:  # pragma: no cover - overridden
        raise NotImplementedError

    def gradient_checkpointing_enable(self, **_kwargs: Any) -> None:
        self.config.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.config.gradient_checkpointing = False

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:  # pragma: no cover - a parameterless module
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:  # pragma: no cover - a parameterless module
            return torch.get_default_dtype()


class BackboneMixin:
    """`_init_transformers_backbone`, the one method the vendored backbone calls."""

    config: Any
    stage_names: list[str]
    _out_features: list[str]
    _out_indices: list[int]
    num_features: list[int]

    def _init_transformers_backbone(self) -> None:
        self.stage_names = self.config.stage_names
        self.num_features = []
        self._out_features = self.config.out_features
        self._out_indices = self.config.out_indices

    @property
    def out_features(self) -> list[str]:
        return self._out_features

    @out_features.setter
    def out_features(self, out_features: list[str]) -> None:
        self._out_features, self._out_indices = _align(out_features, None, self.stage_names)

    @property
    def out_indices(self) -> list[int]:
        return self._out_indices

    @out_indices.setter
    def out_indices(self, out_indices: Sequence[int]) -> None:
        self._out_features, self._out_indices = _align(None, out_indices, self.stage_names)

    @property
    def out_feature_channels(self) -> dict[str, int]:
        return {stage: self.num_features[i] for i, stage in enumerate(self.stage_names)}

    @property
    def channels(self) -> list[int]:
        return [self.out_feature_channels[name] for name in self._out_features]
