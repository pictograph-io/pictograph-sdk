"""Unit tests for `_verify_task` -- the honesty check behind `task=` -- and the
`TASK_MODEL_TYPES` / `TASK_RESULT_TYPES` registries every loader and result
class reads.

`task=` is a pure typing device: passing it lets `model: DetectionModel =
get_model(..., task="object_detection")` typecheck with no cast. That is only
sound if the loader actually VERIFIES the model's real task rather than
trusting the caller -- `_verify_task` is that check, and a mismatch must raise
rather than hand back a class whose `predict()` returns a different shape than
the annotation promised.
"""

from __future__ import annotations

import pytest

from pictograph.inference import _verify_task
from pictograph.inference.models import (
    TASK_MODEL_TYPES,
    ClassificationModel,
    DetectionModel,
    InstanceSegmentationModel,
    KeypointModel,
    SemanticSegmentationModel,
)
from pictograph.inference.results import TASK_RESULT_TYPES, ClassificationResult, ClassScore


class TestVerifyTask:
    def test_matching_task_is_accepted(self) -> None:
        assert (
            _verify_task("object_detection", "object_detection", "My Model") == "object_detection"
        )

    def test_no_declared_task_returns_the_actual_one(self) -> None:
        assert _verify_task(None, "classification", "My Model") == "classification"

    @pytest.mark.parametrize(
        ("declared", "actual"),
        [
            ("object_detection", "classification"),
            ("classification", "object_detection"),
            ("keypoint_detection", "instance_segmentation"),
        ],
    )
    def test_mismatched_task_raises(self, declared: str, actual: str) -> None:
        with pytest.raises(ValueError, match=f"is a {actual!r} model"):
            _verify_task(declared, actual, "My Model")

    def test_mismatch_error_names_the_correct_task_to_pass(self) -> None:
        with pytest.raises(ValueError, match=f"Pass task={'classification'!r}"):
            _verify_task("object_detection", "classification", "My Model")

    def test_unknown_actual_task_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot run locally"):
            _verify_task(None, "some_future_task", "My Model")

    def test_unknown_actual_task_raises_even_when_declared_agrees_with_the_junk(self) -> None:
        # The real-task check runs BEFORE the declared-vs-actual comparison, so
        # an unsupported model_type is always caught -- even if the caller's
        # guess happens to equal the same unsupported value.
        with pytest.raises(ValueError, match="cannot run locally"):
            _verify_task("some_future_task", "some_future_task", "My Model")

    def test_error_message_includes_the_model_name(self) -> None:
        with pytest.raises(ValueError, match="My Special Model"):
            _verify_task("object_detection", "classification", "My Special Model")


class TestTaskRegistries:
    def test_model_and_result_registries_share_the_same_keys(self) -> None:
        assert set(TASK_MODEL_TYPES) == set(TASK_RESULT_TYPES)

    def test_exactly_five_tasks(self) -> None:
        expected = {
            "object_detection",
            "instance_segmentation",
            "semantic_segmentation",
            "keypoint_detection",
            "classification",
        }
        assert set(TASK_MODEL_TYPES) == expected
        assert set(TASK_RESULT_TYPES) == expected

    def test_model_registry_maps_to_the_expected_classes(self) -> None:
        assert TASK_MODEL_TYPES == {
            "object_detection": DetectionModel,
            "instance_segmentation": InstanceSegmentationModel,
            "semantic_segmentation": SemanticSegmentationModel,
            "keypoint_detection": KeypointModel,
            "classification": ClassificationModel,
        }

    def test_every_model_class_model_type_matches_its_registry_key(self) -> None:
        for task, cls in TASK_MODEL_TYPES.items():
            assert cls.model_type == task

    def test_every_result_class_model_type_matches_its_registry_key(self) -> None:
        for task, cls in TASK_RESULT_TYPES.items():
            instance = (
                cls(classes=[ClassScore(name="x", confidence=1.0)])
                if cls is ClassificationResult
                else cls()
            )
            assert instance.model_type == task

    def test_verify_task_and_the_registries_agree_on_what_is_supported(self) -> None:
        """`_verify_task` accepts exactly the tasks the registries know about --
        it reads `TASK_MODEL_TYPES` directly, so this is the anti-drift check."""
        for task in TASK_MODEL_TYPES:
            assert _verify_task(None, task, "x") == task
        with pytest.raises(ValueError, match="cannot run locally"):
            _verify_task(None, "not_a_real_task", "x")
