"""Three defects that only a REAL model, on REAL hardware, could have shown.

Each was found on 2026-07-31 by executing the code samples the web console emits
for a model, verbatim, against the live API - and none of them is reachable from
a payload fixture:

* the vendored YOLOX head raised on the machine's default device, and only there;
* two models loaded in ONE session collided in the weights cache, so the failure
  needed a SECOND model to exist at all;
* a real model's low-confidence tail contained a zero-extent box, and no
  hand-written payload would have thought to include one.

They run against a `fixture-*` reference set of models - one per training
pipeline, uniform build, full artifact sets - and skip cleanly wherever those are
not visible, so pointing this suite at an organization that does not have them
reports "not found" rather than failing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pictograph import Client

pytestmark = [
    pytest.mark.skipif(
        not (os.environ.get("PICTOGRAPH_TEST_KEY") or os.environ.get("PICTOGRAPH_API_KEY")),
        reason="needs a live key - these pull real published weights",
    ),
    pytest.mark.filterwarnings("ignore::ResourceWarning"),
]

_YOLOX = "fixture-yolox"
_DETECTION = "fixture-rfdetr_detection"
# The two RF-DETR models that publish `model.safetensors`, which is the artifact
# whose cache entry collided. `fixture-rfdetr_detection` publishes only a `.pth`,
# and a `.pth` needs no rebuilt container at all.
_SEGMENTATION = "fixture-rfdetr_segmentation"
_KEYPOINT = "fixture-rfdetr_keypoint"


def _fixture_model(client: Client, name: str) -> Any:
    for model in client.models.list(limit=100):
        if model.name == name:
            return model
    pytest.skip(f"{name} is not visible to this key - no `fixture-*` reference set here")


def _accelerator() -> str | None:
    """A non-CPU device this machine can forward on, if it has one."""
    torch = pytest.importorskip("torch")
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return None


def _boxes(result: Any) -> list[tuple[str, int, int, int, int]]:
    return [
        (
            p.name,
            round(p.bounding_box.x),
            round(p.bounding_box.y),
            round(p.bounding_box.w),
            round(p.bounding_box.h),
        )
        for p in result.predictions
    ]


def test_a_native_yolox_runs_on_the_default_device(client: Client, tmp_path: Path) -> None:
    """`device="auto"` - what the Use Model panel emits - must actually work.

    On Apple silicon `auto` resolves to MPS, and the vendored YOLOX head then hit
    `ValueError: invalid type: 'torch.mps.FloatTensor'` on the LAST statement of
    the forward pass, so every native-container YOLOX snippet the panel shipped
    was broken for any Mac user who pasted it unchanged. `device="cpu"` ran the
    same bytes fine, which is exactly why nothing in the suite saw it.

    The two devices are compared rather than `auto` merely being run, because
    the crash was in the decode that turns raw offsets into pixel coordinates -
    a "fix" that ran but misplaced the anchor grid would still return boxes.
    """
    pytest.importorskip("safetensors")
    device = _accelerator()
    if device is None:
        pytest.skip("CPU-only machine - `auto` resolves to cpu and the comparison is vacuous")

    from pictograph import get_model
    from pictograph.inference.runtime import resolve_torch_device

    assert resolve_torch_device("auto") == device

    model = _fixture_model(client, _YOLOX)
    image = _synthetic_image(tmp_path)
    runs = {
        asked: get_model(
            model.name,
            client=client,
            task="object_detection",
            format="safetensors",
            device=asked,
            confidence=0.005,
            cache_dir=tmp_path / "cache",
        ).predict(image)
        for asked in ("auto", "cpu")
    }

    assert runs["auto"].device == device
    assert runs["cpu"].device == "cpu"
    assert runs["auto"].predictions, "the fixture predicts at 0.005 - an empty run proves nothing"
    assert _boxes(runs["auto"]) == _boxes(runs["cpu"])


def test_two_models_load_offline_in_one_session_without_colliding(
    client: Client, tmp_path: Path
) -> None:
    """Loading model B must not depend on whether model A was loaded first.

    Every model publishes its native weights as `model.safetensors`, and the
    rebuilt RF-DETR container was named after that filename - so every model in an
    organization resolved to one `model.rfdetr.pth` in the shared cache. The
    second model loaded found it already there and rebuilt itself from the FIRST
    model's architecture. Measured before the fix on exactly this pair: the
    keypoint model loaded 480 of its 736 tensors out of the segmentation model's
    container.

    Both models are loaded, and the keypoint model is ALSO loaded into a clean
    cache, so the assertion is the property that was actually broken - order
    independence - rather than merely "it did not raise".
    """
    pytest.importorskip("safetensors")
    pytest.importorskip("torch")
    from pictograph import load_model

    image = _synthetic_image(tmp_path)
    bundles = {
        name: _download_bundle(client, _fixture_model(client, name), tmp_path / name)
        for name in (_SEGMENTATION, _KEYPOINT)
    }
    tasks = {_SEGMENTATION: "instance_segmentation", _KEYPOINT: "keypoint_detection"}

    def load(name: str, cache: Path) -> Any:
        weights, config = bundles[name]
        return load_model(
            weights, config, task=tasks[name], device="cpu", confidence=0.002, cache_dir=cache
        ).predict(image)

    shared = tmp_path / "shared-cache"
    load(_SEGMENTATION, shared)
    after = load(_KEYPOINT, shared)
    alone = load(_KEYPOINT, tmp_path / "clean-cache")

    assert [(p.name, round(p.keypoint.x, 2)) for p in after.predictions] == [
        (p.name, round(p.keypoint.x, 2)) for p in alone.predictions
    ]
    containers = sorted(p.name for p in shared.rglob("*.rfdetr.pth"))
    assert len(containers) == 2, f"two models must not share one container: {containers}"


def test_the_low_confidence_tail_does_not_take_down_the_call(
    client: Client, tmp_path: Path
) -> None:
    """A junk box at 0.001 must be dropped, not raised.

    `fixture-rfdetr_detection` returns a zero-height box in its tail, and
    `BoundingBox.h` is `gt=0`, so `.predict()` raised `ValidationError` and the
    caller got NOTHING - not the ~85 well-formed predictions alongside it.
    """
    pytest.importorskip("onnxruntime")
    from pictograph import get_model

    model = _fixture_model(client, _DETECTION)
    result = get_model(
        model.name,
        client=client,
        task="object_detection",
        format="onnx",
        device="cpu",
        confidence=0.001,
        cache_dir=tmp_path / "cache",
    ).predict(_synthetic_image(tmp_path))

    assert result.predictions, "0.001 must reach the tail - an empty result proves nothing"
    assert min(p.confidence for p in result.predictions) < 0.005
    assert all(p.bounding_box.w > 0 and p.bounding_box.h > 0 for p in result.predictions)


def _download_bundle(client: Client, model: Any, into: Path) -> tuple[Path, Path]:
    """This model's offline pair, under the names the API publishes them as.

    The filenames are the point: `model.safetensors` is what every model's native
    artifact is called, and renaming them here would hide the defect being pinned.
    """
    into.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in ("model.safetensors", "config.json"):
        target = into / filename
        if not target.exists():
            client.models.download_file(model_id=model.id, file_name=filename, output_path=target)
        paths.append(target)
    return paths[0], paths[1]


def _synthetic_image(tmp_path: Path) -> Path:
    """Hard-edged shapes on a gradient - never a real dataset image."""
    from PIL import Image, ImageDraw

    dest = tmp_path / "synthetic.jpg"
    if dest.exists():
        return dest
    image = Image.new("RGB", (640, 480))
    draw = ImageDraw.Draw(image)
    for y in range(480):
        draw.line([(0, y), (640, y)], fill=(30 + y // 4, 60, 140 - y // 6))
    draw.rectangle([120, 80, 330, 220], fill=(220, 200, 60), outline=(20, 20, 20), width=4)
    draw.ellipse([380, 150, 560, 330], fill=(200, 70, 70), outline=(255, 255, 255), width=3)
    draw.polygon([(60, 400), (200, 300), (340, 400)], fill=(80, 190, 120))
    image.save(dest, quality=92)
    return dest
