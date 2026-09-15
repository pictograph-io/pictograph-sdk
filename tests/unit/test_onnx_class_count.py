"""The class count must be checked against the graph at LOAD time.

The class list is silently load-bearing and both directions of a mismatch fail in
ways you cannot trace back from the symptom:

- too MANY declared classes indexes past the model's output and dies deep inside the
  emitter with a bare ``IndexError: index 81 is out of bounds for axis 2 with size
  81`` - no mention of classes, the config, or the model. REPRODUCED against the
  shipped 1.68.0 wheel with a real 81-channel segmentation model.
- too FEW silently drops every prediction whose class id lands past the end of the
  list, so a working model appears to find nothing.

Both are now reported in terms of the two numbers a caller can actually compare.
"""

from __future__ import annotations

from typing import Any

import pytest

from pictograph.inference import _onnx


class _FakeOutput:
    def __init__(self, shape: list[Any]) -> None:
        self.shape = shape


class _FakeSession:
    def __init__(self, shape: list[Any]) -> None:
        self._shape = shape

    def get_outputs(self) -> list[_FakeOutput]:
        return [_FakeOutput(self._shape)]


class _FakeWrapper:
    def __init__(self, shape: list[Any] | None) -> None:
        self.session = _FakeSession(shape) if shape is not None else None


def _classes(n: int) -> list[str]:
    return [f"c{i}" for i in range(n)]


class TestSemanticSegmentation:
    """A (batch, C, H, W) graph supports C - 1 classes - one channel is background."""

    def test_too_many_classes_raises_naming_both_numbers(self) -> None:
        wrapper = _FakeWrapper([1, 81, 512, 512])
        with pytest.raises(ValueError) as exc:
            _onnx._check_class_count(wrapper, "semantic_segmentation", _classes(511))
        msg = str(exc.value)
        assert "511" in msg and "81" in msg and "80" in msg

    def test_the_exact_supported_count_loads(self) -> None:
        _onnx._check_class_count(
            _FakeWrapper([1, 81, 512, 512]), "semantic_segmentation", _classes(80)
        )

    def test_too_few_warns_about_silent_dropping(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="pictograph.inference"):
            _onnx._check_class_count(
                _FakeWrapper([1, 81, 512, 512]), "semantic_segmentation", _classes(10)
            )
        assert any("SILENTLY DROPPED" in r.getMessage() for r in caplog.records)

    def test_a_single_channel_graph_supports_one_class(self) -> None:
        """Single-class segmentation is ONE sigmoid channel, not 1 + background."""
        _onnx._check_class_count(
            _FakeWrapper([1, 1, 512, 512]), "semantic_segmentation", _classes(1)
        )
        with pytest.raises(ValueError):
            _onnx._check_class_count(
                _FakeWrapper([1, 1, 512, 512]), "semantic_segmentation", _classes(2)
            )

    def test_channels_are_read_from_axis_1_not_the_last_axis(self) -> None:
        """The bug that motivated this: (batch, C, H, W) - the last axis is WIDTH.

        Reading shape[-1] here yields 512 and the check silently passes anything up
        to 511 classes, which is exactly the crash it is supposed to prevent.
        """
        with pytest.raises(ValueError):
            _onnx._check_class_count(
                _FakeWrapper([1, 81, 512, 512]), "semantic_segmentation", _classes(100)
            )


class TestClassification:
    """Classification output width IS the class count - no background channel."""

    def test_too_many_classes_raises(self) -> None:
        with pytest.raises(ValueError, match="200"):
            _onnx._check_class_count(_FakeWrapper([1, 82]), "classification", _classes(200))

    def test_the_exact_count_loads(self) -> None:
        _onnx._check_class_count(_FakeWrapper([1, 82]), "classification", _classes(82))

    def test_too_few_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="pictograph.inference"):
            _onnx._check_class_count(_FakeWrapper([1, 82]), "classification", _classes(5))
        assert any("SILENTLY DROPPED" in r.getMessage() for r in caplog.records)


class TestDeliberatelyUnchecked:
    """Detection families are NOT checked, on purpose.

    YOLOX packs ``5 + C`` into its last axis and RF-DETR emits ``C + 1`` logits, so a
    single assumption would reject a valid model - worse than the bug being fixed.
    These must pass through untouched rather than guess.
    """

    @pytest.mark.parametrize(
        "model_type", ["object_detection", "instance_segmentation", "keypoint_detection"]
    )
    def test_detection_families_are_not_rejected(self, model_type: str) -> None:
        _onnx._check_class_count(_FakeWrapper([1, 300, 81]), model_type, _classes(500))


class TestDegradesQuietly:
    """A check that cannot run must never block a load that would have worked."""

    def test_no_session_is_a_no_op(self) -> None:
        _onnx._check_class_count(_FakeWrapper(None), "classification", _classes(999))

    def test_a_symbolic_dim_is_a_no_op(self) -> None:
        _onnx._check_class_count(_FakeWrapper([1, "num_classes"]), "classification", _classes(999))
        _onnx._check_class_count(
            _FakeWrapper([1, "C", 512, 512]), "semantic_segmentation", _classes(999)
        )

    def test_a_session_that_raises_is_a_no_op(self) -> None:
        class _Broken:
            @property
            def session(self) -> Any:
                class _S:
                    def get_outputs(self) -> None:
                        raise RuntimeError("no")

                return _S()

        _onnx._check_class_count(_Broken(), "classification", _classes(999))
