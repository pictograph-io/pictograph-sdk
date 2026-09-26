"""Build augmentation ops from JSON-friendly specs.

Lets an agent, a CLI, or a stored config express an augmentation pipeline as
plain data - a list of ``{"op": "<name>", ...params}`` dicts - and get back the
typed op objects :class:`~pictograph.augment.Augmenter` consumes. This is the
serializable counterpart to constructing ops directly in Python.

    from pictograph.augment import Augmenter, build_ops

    ops = build_ops([
        {"op": "flip"},
        {"op": "rotate", "degrees": 15},
        {"op": "brightness", "factor": 0.2},
    ])
    aug = Augmenter(ops, seed=42)

A scalar "strength" param is interpreted the way augmentation tools conventionally
do: ``rotate degrees=15`` means ±15°, and ``brightness factor=0.2`` means ±20%
around 1.0 (i.e. ``(0.8, 1.2)``). Pass an explicit ``[low, high]`` list to set the
range directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pictograph.augment._ops import (
    Augmentation,
    Blur,
    Brightness,
    Contrast,
    Crop,
    CutOut,
    Grayscale,
    HorizontalFlip,
    HueShift,
    Noise,
    Resize,
    Rotate,
    Rotate90,
    Saturation,
    Shear,
    VerticalFlip,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: The op names accepted by :func:`build_ops` (also the ``"op"`` values).
OP_NAMES: tuple[str, ...] = (
    "flip",
    "vflip",
    "rotate90",
    "rotate",
    "resize",
    "crop",
    "brightness",
    "contrast",
    "saturation",
    "hue_shift",
    "grayscale",
    "blur",
    "noise",
    "cutout",
    "shear",
)

#: JSON-friendly catalog of each op and its strength param(s) - THE single
#: source any op picker derives its list from (the backend serves this to the
#: in-app augment dialog; a hand-maintained frontend copy drifted). Kept
#: adjacent to :func:`_one` so adding an op means adding its builder AND its
#: spec in one diff - the unit test pins the parity with :data:`OP_NAMES`.
#: ``params`` lists the UI-meaningful knobs only (the universal ``p``
#: probability and rotate90's ``k`` stay engine-level).
OP_SPECS: tuple[dict[str, Any], ...] = (
    {"op": "flip", "params": []},
    {"op": "vflip", "params": []},
    {"op": "rotate90", "params": []},
    {"op": "rotate", "params": [{"key": "degrees", "default": 15.0, "required": False}]},
    {
        "op": "resize",
        "params": [
            {"key": "width", "default": None, "required": True},
            {"key": "height", "default": None, "required": True},
        ],
    },
    {"op": "crop", "params": [{"key": "scale", "default": 0.8, "required": False}]},
    {"op": "brightness", "params": [{"key": "factor", "default": 0.2, "required": False}]},
    {"op": "contrast", "params": [{"key": "factor", "default": 0.2, "required": False}]},
    {"op": "saturation", "params": [{"key": "factor", "default": 0.2, "required": False}]},
    {"op": "hue_shift", "params": [{"key": "degrees", "default": 20.0, "required": False}]},
    {"op": "grayscale", "params": []},
    {"op": "blur", "params": [{"key": "radius", "default": 2.0, "required": False}]},
    {"op": "noise", "params": [{"key": "amount", "default": 0.08, "required": False}]},
    {"op": "cutout", "params": [{"key": "size", "default": 0.3, "required": False}]},
    {"op": "shear", "params": [{"key": "degrees", "default": 10.0, "required": False}]},
)


def _range(value: Any, *, symmetric: bool) -> float | tuple[float, float]:
    """Interpret a strength param: a scalar → a range, an explicit [lo, hi] → as-is.

    ``symmetric=True`` centres a scalar ``v`` on 1.0 → ``(1-v, 1+v)`` (brightness /
    contrast / saturation). ``symmetric=False`` centres on 0 → ``(-v, v)`` (rotate).
    """
    if isinstance(value, (list, tuple)):
        lo, hi = value
        return (float(lo), float(hi))
    v = float(value)
    return (1 - v, 1 + v) if symmetric else (-abs(v), abs(v))


def _one(spec: Mapping[str, Any]) -> Augmentation:
    name = spec.get("op")
    if name == "flip":
        return HorizontalFlip(p=float(spec.get("p", 0.5)))
    if name == "vflip":
        return VerticalFlip(p=float(spec.get("p", 0.5)))
    if name == "rotate90":
        k = spec.get("k")
        return Rotate90(k=None if k is None else int(k), p=float(spec.get("p", 1.0)))
    if name == "rotate":
        return Rotate(
            _range(spec.get("degrees", 15.0), symmetric=False), p=float(spec.get("p", 1.0))
        )
    if name == "resize":
        return Resize(int(spec["width"]), int(spec["height"]))
    if name == "crop":
        scale = spec.get("scale", 0.8)
        lo_hi = scale if isinstance(scale, (list, tuple)) else (float(scale), 1.0)
        return Crop(scale=(float(lo_hi[0]), float(lo_hi[1])), p=float(spec.get("p", 1.0)))
    if name == "brightness":
        return Brightness(
            _range(spec.get("factor", 0.2), symmetric=True), p=float(spec.get("p", 1.0))
        )
    if name == "contrast":
        return Contrast(
            _range(spec.get("factor", 0.2), symmetric=True), p=float(spec.get("p", 1.0))
        )
    if name == "saturation":
        return Saturation(
            _range(spec.get("factor", 0.2), symmetric=True), p=float(spec.get("p", 1.0))
        )
    if name == "hue_shift":
        return HueShift(
            _range(spec.get("degrees", 20.0), symmetric=False), p=float(spec.get("p", 1.0))
        )
    if name == "grayscale":
        return Grayscale(p=float(spec.get("p", 1.0)))
    if name == "blur":
        return Blur((0.0, float(spec.get("radius", 2.0))), p=float(spec.get("p", 1.0)))
    if name == "noise":
        return Noise((0.0, float(spec.get("amount", 0.08))), p=float(spec.get("p", 1.0)))
    if name == "cutout":
        size = spec.get("size", 0.3)
        size_range = (
            (float(size[0]), float(size[1]))
            if isinstance(size, (list, tuple))
            else (0.1, float(size))
        )
        return CutOut(size_range, count=int(spec.get("count", 1)), p=float(spec.get("p", 1.0)))
    if name == "shear":
        return Shear(
            _range(spec.get("degrees", 10.0), symmetric=False), p=float(spec.get("p", 1.0))
        )
    raise ValueError(f"Unknown augmentation op {name!r}. Valid ops: {', '.join(OP_NAMES)}.")


def build_ops(specs: Sequence[Mapping[str, Any]]) -> list[Augmentation]:
    """Turn a list of ``{"op": name, ...params}`` dicts into typed op objects.

    Args:
        specs: One dict per op. Every dict needs an ``"op"`` key naming one of
            :data:`OP_NAMES`; the remaining keys are that op's params (all
            optional, sensible defaults). ``rotate``/``brightness``/etc. accept a
            scalar strength (interpreted as a range) or an explicit ``[low, high]``.

    Returns:
        The ops in the order given, ready for :class:`~pictograph.augment.Augmenter`.

    Raises:
        ValueError: An unknown ``"op"`` name.
        KeyError: A required param is missing (e.g. ``resize`` without width/height).
    """
    return [_one(spec) for spec in specs]
