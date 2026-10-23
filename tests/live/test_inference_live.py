"""Live local-inference smoke test - real API + a real trained model.

Gated by ``PICTOGRAPH_TEST_KEY`` and by the [inference] extra being installed, so
it auto-skips in the base gate and in CI without a key. Exercises the full path:
resolve a model by name → download + cache its ONNX → build the right wrapper →
predict → typed, deployment-consistent result.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("PICTOGRAPH_TEST_KEY"),
        reason="live inference test needs PICTOGRAPH_TEST_KEY",
    ),
    # The large ONNX weights download can leave an SSL socket for the GC to close,
    # emitting a benign ResourceWarning; don't let the strict gate treat it as a
    # failure on a real-network test.
    pytest.mark.filterwarnings("ignore::ResourceWarning"),
]


def test_local_inference_end_to_end() -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    from pictograph import Client, InferenceResult

    client = Client(api_key=os.environ["PICTOGRAPH_TEST_KEY"])
    ready = [
        m
        for m in client.models.list(status="ready", limit=50)
        if m.model_type in ("object_detection", "instance_segmentation")
    ]
    if not ready:
        pytest.skip("no ready detection/segmentation model in the test org")

    target = ready[0]
    model = client.models.load(target.name)  # by name → get_by_name + download + build
    assert model.name == target.name
    assert model.classes  # class list resolved from class_mapping

    # A blank frame - asserts the pipeline runs + the envelope is typed/consistent
    # (detections may legitimately be empty on a blank image).
    result = model.predict(np.zeros((640, 640, 3), dtype=np.uint8), confidence=0.1)
    assert isinstance(result, InferenceResult)
    assert result.model_type == target.model_type
    assert isinstance(result.predictions, list)
    for pred in result.predictions:
        assert pred.name in model.classes
        assert 0.0 <= pred.confidence <= 1.0
        # round-trips to the deployment / storage dict shape
        dumped = pred.model_dump(mode="json", exclude_none=True)
        assert dumped["name"] and dumped["type"]


def test_get_by_name_resolves_the_right_model() -> None:
    from pictograph import Client

    client = Client(api_key=os.environ["PICTOGRAPH_TEST_KEY"])
    models = client.models.list(limit=5)
    if not models:
        pytest.skip("no models in the test org")
    fetched = client.models.get_by_name(models[0].name)
    assert fetched.name == models[0].name
    assert fetched.id == models[0].id
