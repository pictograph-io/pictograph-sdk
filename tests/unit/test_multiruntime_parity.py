"""The four runtimes must agree, per task family, on the same weights.

This is the acceptance bar for the multi-runtime work: "interchangeable" is only
true if a caller can swap ``format=`` and get the same answer. So for every one of
the five task families, ONE torch module is exported to BOTH an ``.onnx`` graph and
an ExecuTorch ``.pte`` program, loaded through the SDK's own public
:func:`pictograph.load_model`, and run on the same image through the same shared
wrapper. Anything that differs is a difference in the forward pass, which is exactly
what the comparison is meant to detect.

**Every case runs at a NON-NATIVE source size as well as the native one.** This is
a hard-won lesson written into the harness: every pre-existing parity test happened to
feed an image whose dimensions already matched the model's input, so the entire
resize path - the one place a cv2/PIL disagreement lives - was never exercised and a
real divergence went unnoticed. A parity suite that only tests native size is
testing the easy half.

Tolerances, stated BEFORE the runs (the ``fp16_export.py`` house style):

=====================  ==========  ==========================================
comparison             tolerance   why
=====================  ==========  ==========================================
onnxruntime ↔ ptorch    1e-4       fp32 float noise across two independent
                                   kernel libraries, accumulated through a
                                   full backbone.
onnxruntime ↔ tensorrt  1e-3       same, plus TensorRT's kernel autotuning
                                   and permitted fusion reassociation.
predicted class set     exact      a different SET is never float noise; it
                                   is a bug in the decode or the threshold.
=====================  ==========  ==========================================

The measured numbers come in ~4 orders of magnitude under the fp32 bound; the bound
is set at the level where a REAL regression would trip it,
not at the level the current run happens to reach.

TensorRT is not exercised here because it is NVIDIA-only and CI/dev machines are
not. :mod:`tests.unit.test_tensorrt_engine` covers everything about the TRT path
that is testable without a GPU - and the part that matters most for support load,
the mismatch refusal, is fully covered there.
"""

from __future__ import annotations

import functools
import json
import warnings
from pathlib import Path
from typing import Any

import pytest

# The whole module needs a real graph runtime plus an exporter.
pytest.importorskip("torch", reason="the multi-runtime parity harness needs torch")
pytest.importorskip("onnxruntime", reason="needs the [inference] extra")
pytest.importorskip("executorch", reason="needs the [executorch] extra")
pytest.importorskip("cv2", reason="needs the [inference] extra")
pytest.importorskip("numpy", reason="needs the [inference] extra")

import numpy as np
import torch

# ExecuTorch's runtime extension declares itself experimental on import, and this
# suite runs with `filterwarnings = ["error"]`. The warning is ExecuTorch's own
# statement about its API stability and is not suppressed inside the SDK - a user is
# entitled to see it. It is filtered HERE, where it would otherwise fail the harness
# for saying something true.
pytestmark = pytest.mark.filterwarnings("ignore:This API is experimental")

# ── tolerances, stated before the runs ──────────────────────────────────────────
FP32_CONF_ATOL = 1e-4
"""Max |Δconfidence| allowed between two fp32 runtimes on the same weights."""

FP32_COORD_ATOL = 1e-1
"""Max |Δcoordinate| in PIXELS. Looser than the confidence bound on purpose: a box
edge is a decoded product of several tensors and is quoted in image pixels, so a
1e-7 logit difference lands as a sub-tenth-of-a-pixel shift that means nothing."""


# ── family-shaped modules ───────────────────────────────────────────────────────
#
# Each mirrors the OUTPUT CONTRACT its real pipeline exports, because that contract
# is what the shared wrapper decodes and is therefore what this harness has to
# exercise. Small random-init backbones, not trained weights: the question here is
# whether two runtimes compute the same function, which does not depend on the
# function being a good detector. The real trained weights are covered by the
# builders' own parity gate on the service side.


class _Classifier(torch.nn.Module):
    """``(B, num_classes)`` logits - the torchvision classification contract."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        import torchvision.models as tvm

        self.net = tvm.resnet18(weights=None)
        self.net.fc = torch.nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SemanticSeg(torch.nn.Module):
    """``(B, C+1, H, W)`` logits - channel 0 is background, per the smp contract."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, num_classes + 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


#: YOLOX's three FPN strides. The export emits one prediction per cell of each
#: level, so the anchor count is fully determined by the input size - and the
#: wrapper's `postprocess` rebuilds exactly these grids to decode it.
_YOLOX_STRIDES = (8, 16, 32)


def yolox_anchor_count(size: int) -> int:
    """Anchors a YOLOX graph emits for a square ``size`` input.

    NOT a free parameter. ``YOLOXDetector.postprocess`` reconstructs the grids for
    :data:`_YOLOX_STRIDES` and broadcasts them against the output, so a graph with a
    different anchor count fails to decode outright. Deriving it here is what keeps
    the synthetic head honest to the real export.
    """
    return sum((size // stride) ** 2 for stride in _YOLOX_STRIDES)


class _Yolox(torch.nn.Module):
    """``(B, anchors, 5 + C)`` - the RAW YOLOX head, exactly as the export emits it.

    Deliberately NOT pre-decoded: the real graph emits grid-relative centers and
    log-space extents, and the WRAPPER applies ``(raw + grid) * stride`` and
    ``exp(raw) * stride``. Emitting decoded boxes here would skip the wrapper's
    decode and quietly test less than it appears to.
    """

    def __init__(self, num_classes: int, size: int) -> None:
        super().__init__()
        self.anchors = yolox_anchor_count(size)
        self.num_classes = num_classes
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, stride=4, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        self.head = torch.nn.Linear(8, self.anchors * (5 + num_classes))
        # A monotone per-anchor score bias. WITHOUT it a random-init head saturates
        # every anchor's sigmoid to exactly 1.0, several hundred boxes tie, and NMS
        # tie-breaks on float noise - which showed up as a 299-vs-309 "divergence"
        # that was really the harness testing a degenerate model rather than the
        # runtimes disagreeing (their raw forwards matched to 8.3e-05). A real
        # detector has separated scores; this makes the synthetic one behave like it.
        self.register_buffer("_bias", torch.linspace(-9.0, 3.0, self.anchors).unsqueeze(-1))
        self.register_buffer("_class_bias", torch.linspace(-1.5, 1.5, num_classes).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOLOX's preprocessing feeds RAW 0-255 pixels - it does no /255 and no
        # ImageNet normalization (that is the letterbox-only path the wrapper
        # implements). A random-init head on inputs that large produces logits in the
        # hundreds, every sigmoid saturates to exactly 1.0, and the whole score
        # distribution collapses into one giant tie. Normalizing here is what a
        # trained YOLOX's learned first layer does; without it this harness would be
        # measuring NMS tie-breaking rather than runtime agreement.
        raw = self.head(self.stem(x / 255.0)).view(-1, self.anchors, 5 + self.num_classes)
        # Bounded so the wrapper's exp() on the extents stays in a sane range.
        offsets = torch.tanh(raw[..., :2])
        extents = torch.tanh(raw[..., 2:4]) * 1.5
        objectness = torch.sigmoid(raw[..., 4:5] + self._bias)
        classes = torch.sigmoid(raw[..., 5:] + self._class_bias + self._bias * 0.1)
        return torch.cat([offsets, extents, objectness, classes], dim=-1)


class _RFDetr(torch.nn.Module):
    """``(dets, labels)`` - ``(B, Q, 4)`` normalized cxcywh + ``(B, Q, C+1)`` logits."""

    def __init__(self, num_classes: int, queries: int = 30) -> None:
        super().__init__()
        self.queries = queries
        self.num_classes = num_classes
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, stride=4, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        self.box_head = torch.nn.Linear(8, queries * 4)
        self.cls_head = torch.nn.Linear(8, queries * (num_classes + 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.stem(x)
        boxes = torch.sigmoid(self.box_head(feat)).view(-1, self.queries, 4)
        # Keep w/h small so decoded boxes stay inside the frame.
        boxes = torch.cat([boxes[..., :2], boxes[..., 2:] * 0.2 + 0.02], dim=-1)
        logits = self.cls_head(feat).view(-1, self.queries, self.num_classes + 1)
        return boxes, logits


class _RFDetrSeg(torch.nn.Module):
    """``(boxes, logits, masks)`` - the instance-segmentation contract, positionally."""

    def __init__(self, num_classes: int, queries: int = 20, mask: int = 32) -> None:
        super().__init__()
        self.det = _RFDetr(num_classes, queries=queries)
        self.mask = mask
        self.mask_head = torch.nn.Linear(8, queries * mask * mask)
        self.queries = queries

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes, logits = self.det(x)
        feat = self.det.stem(x)
        masks = self.mask_head(feat).view(-1, self.queries, self.mask, self.mask)
        return boxes, logits, masks


class _RFDetrKeypoint(torch.nn.Module):
    """``(boxes, logits, keypoints)`` - ``(B, Q, S, D)`` keypoints."""

    def __init__(self, num_classes: int, queries: int = 20, slots: int = 4, depth: int = 3) -> None:
        super().__init__()
        self.det = _RFDetr(num_classes, queries=queries)
        self.queries, self.slots, self.depth = queries, slots, depth
        self.kp_head = torch.nn.Linear(8, queries * slots * depth)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes, logits = self.det(x)
        feat = self.det.stem(x)
        kps = torch.sigmoid(self.kp_head(feat)).view(-1, self.queries, self.slots, self.depth)
        return boxes, logits, kps


# ── export helpers ──────────────────────────────────────────────────────────────


def _export_onnx(module: torch.nn.Module, example: torch.Tensor, path: Path, n_out: int) -> None:
    """Export to ONNX the way the training pipelines do - the TorchScript exporter.

    ``dynamo=False`` is not laziness about a deprecation. torch 2.9 made the
    torch.export-based exporter the default, and it fails to decompose these graphs
    (``ConversionError: Failed to decompose the FX graph``) - while the pipelines
    that actually produce our ``.onnx`` artifacts use the TorchScript exporter. Using
    it here keeps the harness faithful to how the real ONNX side is built; comparing
    against a graph produced by an exporter we do not ship would be comparing the
    wrong thing.

    The warning filter is scoped to this call because the suite escalates
    ``DeprecationWarning`` to an error, and this one is torch's, not ours.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            module,
            (example,),
            str(path),
            input_names=["input"],
            output_names=[f"output_{i}" for i in range(n_out)],
            opset_version=17,
            dynamo=False,
        )


@functools.lru_cache(maxsize=1)
def _executorch_exporter() -> tuple[Any, Any]:
    """``(to_edge_transform_and_lower, XnnpackPartitioner)``, imported exactly once.

    Two things this guards, both learned the hard way:

    - The suite runs with ``filterwarnings = ["error"]``. ExecuTorch's package tree
      emits deprecation warnings while importing, so under that setting the import
      ABORTS partway - and because ``executorch.backends`` is a namespace package,
      the aborted attempt leaves its ``__path__`` unresolvable, so every later retry
      fails with a bare ``KeyError: 'executorch.backends.xnnpack'`` that says nothing
      about the real cause. The filter is scoped to the import for that reason.
    - Caching it keeps the (slow) import off every parametrized case.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower
    return to_edge_transform_and_lower, XnnpackPartitioner


def _export_pte(module: torch.nn.Module, example: torch.Tensor, path: Path) -> None:
    lower, partitioner_cls = _executorch_exporter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exported = torch.export.export(module, (example,))
        lowered = lower(exported, partitioner=[partitioner_cls()]).to_executorch()
    path.write_bytes(lowered.buffer)


def _config(model_type: str, architecture: str, classes: list[str], hw: tuple[int, int]) -> dict:
    return {
        "_pictograph": {
            "model_type": model_type,
            "architecture": architecture,
            "class_mapping": {"classes": classes},
            "input_shape": [hw[0], hw[1]],
            "export_name": f"Parity {model_type}",
        }
    }


def _image(height: int, width: int, seed: int = 11) -> np.ndarray:
    """A deterministic BGR image. Structured, not pure noise, so the resize path
    has real gradients to disagree about rather than uncorrelated pixels."""
    rng = np.random.RandomState(seed)
    ramp = np.linspace(0, 255, width, dtype=np.float32)[None, :, None]
    base = np.repeat(ramp, height, axis=0).repeat(3, axis=2)
    base += rng.rand(height, width, 3) * 40.0
    return np.clip(base, 0, 255).astype(np.uint8)


# ── the families under test ─────────────────────────────────────────────────────

_FAMILIES: list[dict[str, Any]] = [
    {
        "id": "classification",
        "confidence": 0.0,
        "task": "classification",
        "architecture": "resnet18",
        "classes": ["cat", "dog", "bird", "fish", "horse"],
        "hw": (224, 224),
        "outputs": 1,
        "build": lambda n: _Classifier(n),
    },
    {
        "id": "semantic_segmentation",
        "confidence": 0.0,
        "task": "semantic_segmentation",
        "architecture": "Unet",
        "classes": ["road", "sky", "tree"],
        "hw": (128, 128),
        "outputs": 1,
        "build": lambda n: _SemanticSeg(n),
    },
    {
        "id": "object_detection_yolox",
        "confidence": 0.5,
        "task": "object_detection",
        "architecture": "YOLOX-s",
        "classes": ["person", "car", "sign"],
        "hw": (128, 128),
        "outputs": 1,
        "build": lambda n: _Yolox(n, 128),
    },
    {
        "id": "object_detection_rfdetr",
        "confidence": 0.5,
        "task": "object_detection",
        "architecture": "RF-DETR",
        "classes": ["person", "car", "sign", "pole", "cone"],
        "hw": (128, 128),
        "outputs": 2,
        "build": lambda n: _RFDetr(n),
    },
    {
        "id": "instance_segmentation",
        "confidence": 0.5,
        "task": "instance_segmentation",
        "architecture": "RF-DETR Seg",
        "classes": ["person", "car", "sign", "pole", "cone"],
        "hw": (128, 128),
        "outputs": 3,
        "build": lambda n: _RFDetrSeg(n),
    },
    {
        "id": "keypoint_detection",
        "confidence": 0.5,
        "task": "keypoint_detection",
        "architecture": "RF-DETR Keypoint",
        "classes": ["joint_a", "joint_b", "joint_c", "joint_d"],
        "hw": (128, 128),
        "outputs": 3,
        "build": lambda n: _RFDetrKeypoint(n),
    },
]

# (label, height, width). "native" matches the model input exactly; the other two
# force the resize path, which no existing test was doing.
_SIZES = [("native", None), ("non-native-small", (97, 141)), ("non-native-large", (321, 259))]


def _build_pair(family: dict[str, Any], tmp: Path) -> tuple[Path, Path, dict]:
    torch.manual_seed(4242)
    module = family["build"](len(family["classes"])).eval()
    hw = family["hw"]
    example = torch.randn(1, 3, hw[0], hw[1])
    onnx_path = tmp / f"{family['id']}.onnx"
    pte_path = tmp / f"xnnpack-fp32-{family['id']}.pte"
    _export_onnx(module, example, onnx_path, family["outputs"])
    _export_pte(module, example, pte_path)
    cfg = _config(family["task"], family["architecture"], family["classes"], hw)
    return onnx_path, pte_path, cfg


def _summarize(result: Any) -> tuple[list[str], list[float], list[float]]:
    """(names, confidences, coordinates) - a comparable digest of any task result.

    Detections are canonically SORTED before comparison, rather than compared in
    emission order. Emission order is not part of any contract: the wrappers sort by
    score and NMS is order-sensitive at exact ties, so two runtimes that computed
    identical tensors can legitimately emit the same detections in a different
    sequence. Comparing positionally turns that into a spurious 187-pixel
    "divergence" between two boxes that both runtimes found - measured, and the
    reason this sorts.

    Classification is deliberately NOT sorted: its rank order IS the contract.
    """
    if hasattr(result, "classes"):
        scores = list(result.classes)
        return [c.name for c in scores], [c.confidence for c in scores], []

    rows: list[tuple[str, float, tuple[float, ...]]] = []
    for pred in result.predictions:
        coords: tuple[float, ...] = ()
        box = getattr(pred, "bounding_box", None)
        if box is not None:
            coords += (box.x, box.y, box.w, box.h)
        point = getattr(pred, "keypoint", None)
        if point is not None:
            coords += (point.x, point.y)
        rows.append((pred.name, float(pred.confidence), coords))

    rows.sort(key=lambda r: (r[0], -r[1], r[2]))
    names = [r[0] for r in rows]
    confs = [r[1] for r in rows]
    flat = [v for r in rows for v in r[2]]
    return names, confs, flat


@pytest.mark.parametrize("family", _FAMILIES, ids=[f["id"] for f in _FAMILIES])
def test_executorch_matches_onnxruntime_on_every_task(
    family: dict[str, Any], tmp_path: Path
) -> None:
    """Same weights, two runtimes, same predictions - at native AND non-native size."""
    from pictograph import load_model

    onnx_path, pte_path, cfg = _build_pair(family, tmp_path)
    task = family["task"]

    conf = family["confidence"]
    onnx_model = load_model(onnx_path, cfg, task=task, device="cpu", confidence=conf)
    pte_model = load_model(pte_path, cfg, task=task, confidence=conf)
    try:
        # The provenance must be honest before the numbers mean anything.
        assert onnx_model.backend == "onnxruntime"
        assert pte_model.backend == "executorch"
        assert pte_model.providers == ["XnnpackBackend"]
        assert pte_model.device == "cpu"
        assert type(onnx_model) is type(pte_model), "same task must give the same class"

        for label, size in _SIZES:
            height, width = size or family["hw"]
            image = _image(height, width)
            left = onnx_model.predict(image)
            right = pte_model.predict(image)
            assert type(left) is type(right)

            l_names, l_conf, l_xy = _summarize(left)
            r_names, r_conf, r_xy = _summarize(right)
            assert l_names == r_names, f"{family['id']} @ {label}: class set diverged"
            assert len(l_conf) == len(r_conf)
            if l_conf:
                worst = max(abs(a - b) for a, b in zip(l_conf, r_conf, strict=True))
                assert worst <= FP32_CONF_ATOL, (
                    f"{family['id']} @ {label}: max |dconf| {worst:.3e} > {FP32_CONF_ATOL:.1e}"
                )
            assert len(l_xy) == len(r_xy)
            if l_xy:
                worst_xy = max(abs(a - b) for a, b in zip(l_xy, r_xy, strict=True))
                assert worst_xy <= FP32_COORD_ATOL, (
                    f"{family['id']} @ {label}: max |dcoord| {worst_xy:.3e} px "
                    f"> {FP32_COORD_ATOL:.1e}"
                )
    finally:
        onnx_model.close()
        pte_model.close()


def test_non_native_sizes_actually_exercise_the_resize_path() -> None:
    """Guard the guard: the sizes above must genuinely differ from every model input.

    That divergence hid inside a suite whose inputs all happened to be native. If
    "tidies" the size list back to the model's own resolution, this fails and says
    why, rather than the suite quietly going back to testing the easy half.
    """
    non_native = [size for label, size in _SIZES if size is not None]
    assert non_native, "the parity suite must include at least one non-native size"
    for family in _FAMILIES:
        for size in non_native:
            assert size != family["hw"], (
                f"{family['id']}: {size} is this model's NATIVE input - it does not "
                f"exercise the resize path this suite exists to cover."
            )


if __name__ == "__main__":  # pragma: no cover - reporting mode for the handoff
    import tempfile

    from pictograph import load_model

    print(f"{'family':<26} {'size':<18} {'n':>4} {'max|dconf|':>12} {'max|dcoord| px':>15}")
    print("-" * 80)
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        for fam in _FAMILIES:
            onnx_p, pte_p, config = _build_pair(fam, tmp)
            a = load_model(
                onnx_p, config, task=fam["task"], device="cpu", confidence=fam["confidence"]
            )
            b = load_model(pte_p, config, task=fam["task"], confidence=fam["confidence"])
            for lbl, sz in _SIZES:
                h, w = sz or fam["hw"]
                img = _image(h, w)
                la, ca, xa = _summarize(a.predict(img))
                lb, cb, xb = _summarize(b.predict(img))
                dc = max((abs(x - y) for x, y in zip(ca, cb, strict=False)), default=0.0)
                dx = max((abs(x - y) for x, y in zip(xa, xb, strict=False)), default=0.0)
                flag = "" if la == lb else "  ** CLASS SET DIVERGED **"
                print(
                    f"{fam['id']:<26} {lbl + f' {h}x{w}':<18} {len(la):>4} "
                    f"{dc:>12.3e} {dx:>15.3e}{flag}"
                )
            a.close()
            b.close()
    print(
        "\nbackend values:", json.dumps({f["id"]: "onnxruntime vs executorch" for f in _FAMILIES})
    )
