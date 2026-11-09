"""The vendored YOLOX architecture must rebuild EXACTLY what upstream rebuilds.

`pictograph.inference._yolox` is a copy of `yolox/models/` at the commit the
training image pins, taken so that `pip install "pictograph[inference]"` is the
whole requirement for running a `.pth` we published - see that package's NOTICE
for why depending on PyPI's `yolox` is not an option.

A copy is only safe while it is faithful, and "faithful" here has a precise
meaning: a checkpoint is a mapping from layer NAME to tensor, so the vendored
module must present the SAME `state_dict()` key set with the SAME shapes, or a
`strict=True` load either fails outright or - far worse - succeeds against a
silently different block.

The strong form of that check needs the upstream source to compare against, which
is not in this repo, so it runs only when `PICTOGRAPH_YOLOX_UPSTREAM` points at an
extracted copy of upstream YOLOX at the pinned SHA:

    git clone https://github.com/Megvii-BaseDetection/YOLOX.git /tmp/yolox-upstream
    git -C /tmp/yolox-upstream checkout 6ddff48…
    PICTOGRAPH_YOLOX_UPSTREAM=/tmp/yolox-upstream pytest tests/unit/test_yolox_vendored.py

The always-on tests below need no upstream: they pin the rebuild recipe itself,
and prove the module builds and runs with `yolox` unimportable.
"""

from __future__ import annotations

import logging
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from pictograph.inference._yolox import (  # noqa: E402
    YOLOX_SIZES,
    YOLOX_VENDORED_SHA,
    build_yolox,
)
from tests.conftest import (  # noqa: E402
    ENV_TRAINING_SERVICE_SOURCE,
    companion_skip_reason,
    companion_source,
)

_UPSTREAM = os.environ.get("PICTOGRAPH_YOLOX_UPSTREAM")


def _accelerators() -> list[str]:
    """The non-CPU devices THIS machine can actually run a forward pass on."""
    found: list[str] = []
    if torch.backends.mps.is_available():
        found.append("mps")
    if torch.cuda.is_available():
        found.append("cuda")
    return found


def test_vendored_sha_matches_the_training_image() -> None:
    """LOCKSTEP: the SHA here and `YOLOX_SHA` in the training service are one value.

    The training service is not part of this repository, so the comparison is
    opt-in and skips without it (see ``tests/conftest.py``).
    """
    service = companion_source(ENV_TRAINING_SERVICE_SOURCE)
    if not service.exists():  # pragma: no cover - not configured
        pytest.skip(companion_skip_reason(ENV_TRAINING_SERVICE_SOURCE))
    assert YOLOX_VENDORED_SHA in service.read_text(encoding="utf-8"), (
        f"the vendored YOLOX is {YOLOX_VENDORED_SHA}, which the training image no "
        "longer pins - re-vendor from the SHA that trains the weights, or the "
        "checkpoints will load into the wrong architecture"
    )


def test_sizes_table_matches_the_pipeline() -> None:
    """LOCKSTEP with the YOLOX training pipeline's own size table."""
    assert YOLOX_SIZES == {
        "nano": (0.33, 0.25),
        "tiny": (0.33, 0.375),
        "s": (0.33, 0.50),
        "m": (0.67, 0.75),
        "l": (1.00, 1.00),
        "x": (1.33, 1.25),
    }


@pytest.mark.parametrize("size", sorted(YOLOX_SIZES))
def test_builds_and_runs_without_yolox_installed(
    size: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every size rebuilds and forwards with the upstream packages unimportable."""
    for absent in ("yolox", "yolox.models", "yolox.utils", "loguru", "thop", "tabulate"):
        monkeypatch.setitem(sys.modules, absent, None)

    depth, width = YOLOX_SIZES[size]
    model = build_yolox(depth, width, num_classes=3)
    model.eval()
    model.head.decode_in_inference = True
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 640, 640))
    # 640/8² + 640/16² + 640/32² = 6400 + 1600 + 400 anchors; 4 box + 1 obj + 3 cls.
    assert tuple(out.shape) == (1, 8400, 8)


def test_head_decodes_in_inference_by_default() -> None:
    """`decode_in_inference` must default True - the loader relies on it."""
    model = build_yolox(0.33, 0.50, num_classes=2)
    assert model.head.decode_in_inference is True


@pytest.mark.skipif(
    not _accelerators(),
    reason=(
        "no non-CPU device on this machine - the defect this pins is a device-string "
        "one and CANNOT reproduce on CPU"
    ),
)
@pytest.mark.parametrize("device", _accelerators())
def test_decoded_forward_runs_on_this_machines_accelerator(device: str) -> None:
    """The decoded forward pass must run on every device torch offers, not just CPU.

    Until 1.69.15 the LAST statement of an otherwise-complete forward pass read
    ``.type(xin[0].type())``. ``Tensor.type()`` manufactures the legacy string
    ``'torch.mps.FloatTensor'`` on Apple silicon, and ``Tensor.type(…)`` then
    refuses to parse it - legacy tensor types exist only for CPU and CUDA - so
    every native-container YOLOX cell the Use Model panel emits with the DEFAULT
    ``device="auto"`` raised ``ValueError: invalid type: 'torch.mps.FloatTensor'``
    on a Mac. The same weights ran on ``device="cpu"``, so no CPU test could see
    it, and no test that stops at ``build_yolox`` could either: the crash is in the
    decode, which only runs when the module is actually forwarded on that device.

    The output is compared against CPU rather than merely asserted non-crashing,
    because ``decode_outputs`` is where the anchor grid and the stride tensor are
    ADDED to the raw offsets - a device fix that quietly stopped placing them
    correctly would still "run".
    """
    model = build_yolox(0.33, 0.25, num_classes=4).eval()
    image = torch.randn(1, 3, 640, 640, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        on_cpu = model(image)
        moved = model.to(device)
        on_device = moved(image.to(device)).to("cpu")
        model.to("cpu")

    assert on_device.shape == on_cpu.shape == (1, 8400, 9)
    # Boxes come back as absolute pixels because the grid+stride were applied; a
    # decode that silently no-opped would leave raw offsets near zero.
    assert on_device[..., :2].max().item() > 100.0
    assert torch.allclose(on_device, on_cpu, atol=2e-3, rtol=2e-3)


# ── the strong form: identical to upstream at the pinned SHA ──


def _import_upstream() -> tuple[object, object, object]:
    """Import the real `yolox.models` from the mirrored source tree.

    Stubs the packages upstream's `yolox/utils/__init__` drags in but the model
    path never uses (psutil via metric.py, tabulate via logger.py, thop via
    model_utils.py, loguru in yolo_head). Having to do that is itself the reason
    `_upstream_utils.py` exists rather than a vendored `yolox/utils/` package.
    """

    def _noop_tabulate(*_args: object, **_kwargs: object) -> str:
        return ""

    def _noop_profile(*_args: object, **_kwargs: object) -> tuple[int, int]:
        return (0, 0)

    for name, attrs in (
        ("psutil", {}),
        ("tabulate", {"tabulate": _noop_tabulate}),
        ("thop", {"profile": _noop_profile}),
        ("loguru", {"logger": logging.getLogger("pictograph.test")}),
    ):
        try:
            __import__(name)
        except ImportError:
            module = types.ModuleType(name)
            for key, value in attrs.items():
                setattr(module, key, value)
            sys.modules[name] = module

    assert _UPSTREAM is not None
    sys.path.insert(0, _UPSTREAM)
    from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead  # type: ignore[import-not-found]

    return YOLOPAFPN, YOLOXHead, YOLOX


@pytest.mark.skipif(_UPSTREAM is None, reason="PICTOGRAPH_YOLOX_UPSTREAM not set")
@pytest.mark.parametrize("size", sorted(YOLOX_SIZES))
def test_state_dict_signature_is_identical_to_upstream(size: str) -> None:
    """Same keys, same shapes - and upstream weights strict-load into ours."""
    pafpn, head_cls, yolox_cls = _import_upstream()
    depth, width = YOLOX_SIZES[size]

    upstream = yolox_cls(  # type: ignore[operator]
        pafpn(depth, width, act="silu"),  # type: ignore[operator]
        head_cls(3, width, act="silu"),  # type: ignore[operator]
    ).eval()
    vendored = build_yolox(depth, width, num_classes=3).eval()

    assert {k: tuple(v.shape) for k, v in upstream.state_dict().items()} == {
        k: tuple(v.shape) for k, v in vendored.state_dict().items()
    }
    vendored.load_state_dict(upstream.state_dict(), strict=True)

    upstream.head.decode_in_inference = True
    vendored.head.decode_in_inference = True
    image = torch.randn(1, 3, 640, 640, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(upstream(image), vendored(image))


@pytest.mark.skipif(_UPSTREAM is None, reason="PICTOGRAPH_YOLOX_UPSTREAM not set")
def test_the_two_off_forward_type_casts_still_match_upstream_bit_for_bit() -> None:
    """The two edited lines OUTSIDE the forward pass are still upstream's arithmetic.

    ``bboxes_iou`` (label assignment) and ``IOUloss`` (the box loss) each carried
    ``(tl < br).type(tl.type())``, which is device-incomplete for the same reason
    the decode was, and each is now ``(tl < br).to(tl.dtype)``. Neither runs during
    inference, so the forward-equality test above cannot cover them - and the
    ``NOTICE`` asserts, under Apache-2.0 § 4b, that no edit here changes numerical
    behaviour. This is that assertion, executed.
    """
    _import_upstream()  # puts the upstream tree on sys.path and stubs its deps
    from yolox.models.losses import IOUloss as UpstreamIOUloss  # type: ignore[import-not-found]
    from yolox.utils.boxes import (
        bboxes_iou as upstream_bboxes_iou,  # type: ignore[import-not-found]
    )

    from pictograph.inference._yolox._upstream_utils import bboxes_iou
    from pictograph.inference._yolox.losses import IOUloss

    gen = torch.Generator().manual_seed(11)
    a = torch.rand(37, 4, generator=gen) * 100
    b = torch.rand(23, 4, generator=gen) * 100
    for xyxy in (True, False):
        assert torch.equal(bboxes_iou(a, b, xyxy), upstream_bboxes_iou(a, b, xyxy))

    pred = torch.rand(64, 4, generator=gen) * 100
    target = torch.rand(64, 4, generator=gen) * 100
    for loss_type in ("iou", "giou"):
        assert torch.equal(
            IOUloss(loss_type=loss_type)(pred, target),
            UpstreamIOUloss(loss_type=loss_type)(pred, target),
        )
