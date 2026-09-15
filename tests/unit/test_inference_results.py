"""Unit tests for `pictograph.inference.results` -- the five task-typed result
classes local inference always returns.

Pure pydantic, no optional dependency (see the module's own docstring for why
each task gets ITS OWN `predictions` field rather than one shared
`InferenceResult` with optional `predictions`/`classes`/`tags`): the base
class deliberately does not declare `predictions` at all, so these tests also
pin that a geometry-shaped result carries no classifier-shaped fields and
vice versa.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pictograph.inference.results import (
    TASK_RESULT_TYPES,
    ClassificationResult,
    ClassScore,
    DetectionResult,
    InstanceSegmentationResult,
    KeypointResult,
    SemanticSegmentationResult,
    build_result,
)
from pictograph.models.annotation import (
    BBoxAnnotation,
    KeypointAnnotation,
    PolygonAnnotation,
)

_BOX = {"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0}


def _bbox(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "person",
        "type": "bbox",
        "bounding_box": {"x": 0, "y": 0, "w": 1, "h": 1},
    }
    base.update(over)
    return base


def _polygon(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "person",
        "type": "polygon",
        "polygon": {"paths": [[{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}]]},
    }
    base.update(over)
    return base


def _keypoint(**over: object) -> dict[str, object]:
    base: dict[str, object] = {"name": "nose", "type": "keypoint", "keypoint": {"x": 1, "y": 2}}
    base.update(over)
    return base


class TestDetectionResult:
    def test_predictions_narrow_to_bbox_annotations(self) -> None:
        result = DetectionResult.model_validate({"predictions": [_bbox()]})
        assert isinstance(result.predictions[0], BBoxAnnotation)

    def test_defaults_to_empty_predictions(self) -> None:
        assert DetectionResult.model_validate({}).predictions == []

    def test_model_type_is_fixed(self) -> None:
        assert DetectionResult().model_type == "object_detection"

    def test_has_no_classifier_shaped_fields(self) -> None:
        result = DetectionResult()
        assert not hasattr(result, "classes")
        assert not hasattr(result, "top")
        assert not hasattr(result, "tags")


class TestInstanceSegmentationResult:
    def test_predictions_accept_polygon_or_bbox(self) -> None:
        result = InstanceSegmentationResult.model_validate(
            {"predictions": [_polygon(), _bbox(name="degraded")]}
        )
        assert isinstance(result.predictions[0], PolygonAnnotation)
        assert isinstance(result.predictions[1], BBoxAnnotation)

    def test_polygons_property_filters_to_real_masks_only(self) -> None:
        """A class whose mask polygonized (PolygonAnnotation) mixed with one
        that degraded to its bbox -- `.polygons` keeps only the former."""
        result = InstanceSegmentationResult.model_validate(
            {"predictions": [_polygon(), _bbox(name="degraded")]}
        )
        assert len(result.polygons) == 1
        assert result.polygons[0].name == "person"
        assert all(isinstance(p, PolygonAnnotation) for p in result.polygons)

    def test_polygons_property_is_empty_when_all_degraded_to_bbox(self) -> None:
        result = InstanceSegmentationResult.model_validate({"predictions": [_bbox()]})
        assert result.polygons == []


class TestSemanticSegmentationResult:
    def test_predictions_are_polygon_only(self) -> None:
        result = SemanticSegmentationResult.model_validate({"predictions": [_polygon()]})
        assert isinstance(result.predictions[0], PolygonAnnotation)

    def test_model_type_is_fixed(self) -> None:
        assert SemanticSegmentationResult().model_type == "semantic_segmentation"


class TestKeypointResult:
    def test_predictions_are_keypoints_only(self) -> None:
        """The skeleton primitive is gone - every prediction is ONE point of one
        joint class, and `instance_id` is what says which object it belongs to."""
        result = KeypointResult.model_validate(
            {"predictions": [_keypoint(), _keypoint(name="left_eye")]}
        )
        assert len(result.predictions) == 2
        assert all(isinstance(p, KeypointAnnotation) for p in result.predictions)
        assert result.points == result.predictions

    def test_skeletons_view_is_gone(self) -> None:
        """Guard the removal: a stale `.skeletons` reader must break loudly rather
        than silently read an attribute that can never be populated again."""
        assert not hasattr(KeypointResult.model_validate({}), "skeletons")

    def test_instances_group_by_instance_id(self) -> None:
        result = KeypointResult.model_validate(
            {
                "predictions": [
                    _keypoint(name="nose", instance_id=2),
                    _keypoint(name="nose", instance_id=1),
                    _keypoint(name="left_eye", instance_id=1),
                ]
            }
        )
        instances = result.instances
        # Ordered by instance_id ascending, NOT emission order.
        assert [[p.name for p in inst] for inst in instances] == [
            ["nose", "left_eye"],
            ["nose"],
        ]

    def test_each_unassociated_point_is_its_own_instance(self) -> None:
        """`instance_id=None` means UNASSOCIATED - a lone landmark. Fusing them
        into one group would invent an object the model never predicted."""
        result = KeypointResult.model_validate(
            {"predictions": [_keypoint(), _keypoint(name="tip")]}
        )
        assert [len(inst) for inst in result.instances] == [1, 1]

    def test_unassociated_points_sort_last_in_emission_order(self) -> None:
        result = KeypointResult.model_validate(
            {
                "predictions": [
                    _keypoint(name="a"),
                    _keypoint(name="b", instance_id=3),
                    _keypoint(name="c"),
                ]
            }
        )
        assert [inst[0].name for inst in result.instances] == ["b", "a", "c"]

    def test_empty_predictions_give_empty_views(self) -> None:
        result = KeypointResult.model_validate({})
        assert result.predictions == []
        assert result.points == []
        assert result.instances == []


class TestClassificationResult:
    def test_top_is_the_first_ranked_class(self) -> None:
        result = ClassificationResult(
            classes=[
                ClassScore(name="cat", confidence=0.9),
                ClassScore(name="dog", confidence=0.1),
            ]
        )
        assert result.top == ClassScore(name="cat", confidence=0.9)

    def test_top_is_never_optional_even_with_one_class(self) -> None:
        result = ClassificationResult(classes=[ClassScore(name="only", confidence=0.5)])
        assert result.top.name == "only"

    def test_tags_is_just_the_names_in_rank_order(self) -> None:
        result = ClassificationResult(
            classes=[
                ClassScore(name="cat", confidence=0.9),
                ClassScore(name="dog", confidence=0.1),
            ]
        )
        assert result.tags == ["cat", "dog"]

    def test_empty_classes_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ClassificationResult(classes=[])

    def test_classes_field_is_required(self) -> None:
        with pytest.raises(ValidationError):
            ClassificationResult()

    def test_model_type_is_fixed(self) -> None:
        result = ClassificationResult(classes=[ClassScore(name="x", confidence=1.0)])
        assert result.model_type == "classification"

    def test_has_no_geometry_predictions_field(self) -> None:
        result = ClassificationResult(classes=[ClassScore(name="x", confidence=1.0)])
        assert not hasattr(result, "predictions")


class TestClassScore:
    def test_confidence_out_of_range_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ClassScore(name="x", confidence=1.5)
        with pytest.raises(ValidationError):
            ClassScore(name="x", confidence=-0.1)


class TestInferenceResultProvenance:
    """The fields every task result shares -- provenance for the run, not the
    prediction shape."""

    def test_defaults(self) -> None:
        result = DetectionResult()
        assert result.backend == "onnxruntime"
        assert result.device == "cpu"
        assert result.providers == []
        assert result.inference_ms is None

    def test_fields_are_settable(self) -> None:
        result = DetectionResult(backend="pytorch", device="mps", inference_ms=12.3)
        assert result.backend == "pytorch"
        assert result.device == "mps"
        assert result.inference_ms == 12.3


class TestTaskResultTypes:
    def test_maps_every_task_to_its_result_class(self) -> None:
        assert TASK_RESULT_TYPES == {
            "object_detection": DetectionResult,
            "instance_segmentation": InstanceSegmentationResult,
            "semantic_segmentation": SemanticSegmentationResult,
            "keypoint_detection": KeypointResult,
            "classification": ClassificationResult,
        }

    def test_every_class_model_type_default_matches_its_registry_key(self) -> None:
        for task, cls in TASK_RESULT_TYPES.items():
            instance = (
                cls(classes=[ClassScore(name="x", confidence=1.0)])
                if cls is ClassificationResult
                else cls()
            )
            assert instance.model_type == task


class TestBuildResult:
    """`build_result` is the ONE conversion - the local engines and the remote
    `DeploymentClient` both go through it, which is what makes "Edge and Remote
    agree" structural rather than a convention two builders are asked to keep."""

    def test_builds_every_task(self) -> None:
        for task in TASK_RESULT_TYPES:
            payload = (
                {"model_type": task, "predictions": [{"class": "dog", "confidence": 0.9}]}
                if task == "classification"
                else {"model_type": task, "predictions": []}
            )
            result = build_result(payload, task=task)  # type: ignore[arg-type]
            assert isinstance(result, TASK_RESULT_TYPES[task])
            assert result.model_type == task

    def test_backfills_a_missing_annotation_id(self) -> None:
        """Every real emitter assigns a uuid4 `id`; a hand-rolled payload must
        still validate rather than tripping the annotation model's required id."""
        result = build_result(
            {"predictions": [{"name": "dog", "type": "bbox", "bounding_box": _BOX}]},
            task="object_detection",
        )
        assert isinstance(result, DetectionResult)
        assert result.predictions[0].id

    def test_keeps_an_id_the_emitter_already_set(self) -> None:
        result = build_result(
            {"predictions": [{"id": "keep-me", "name": "d", "type": "bbox", "bounding_box": _BOX}]},
            task="object_detection",
        )
        assert isinstance(result, DetectionResult)
        assert result.predictions[0].id == "keep-me"

    def test_classification_reads_the_class_key(self) -> None:
        """The emitter keys a classifier's entries `class`, not `name` - the one
        task whose conversion is not a passthrough."""
        result = build_result(
            {"predictions": [{"class": "dog", "confidence": 0.9}, {"class": "cat"}]},
            task="classification",
        )
        assert isinstance(result, ClassificationResult)
        assert [c.name for c in result.classes] == ["dog", "cat"]
        assert result.classes[1].confidence == 0.0

    def test_classification_with_no_ranked_class_raises(self) -> None:
        with pytest.raises(ValueError, match="Whatever returned no classes"):
            build_result({"predictions": []}, task="classification", source="Whatever")

    def test_provenance_is_carried_through_verbatim(self) -> None:
        result = build_result(
            {"predictions": []},
            task="object_detection",
            backend="tensorrt",
            device="cuda:1",
            providers=["TensorrtExecutionProvider"],
            inference_ms=12.5,
        )
        assert (result.backend, result.device, result.inference_ms) == ("tensorrt", "cuda:1", 12.5)
        assert result.providers == ["TensorrtExecutionProvider"]

    def test_local_and_remote_agree_on_the_same_payload(self) -> None:
        """THE parity assertion.

        A deployment's serving container and the SDK's local ONNX engine call
        byte-identical vendored copies of `dispatch.infer_image`, and the
        inference gateway passes that dict through verbatim. So one payload run
        through both paths must differ only in provenance - never in predictions.
        """
        payload = {
            "model_type": "object_detection",
            "predictions": [
                {
                    "id": "a1",
                    "name": "dog",
                    "type": "bbox",
                    "bounding_box": _BOX,
                    "confidence": 0.9,
                    "attributes": ["auto-annotate"],
                }
            ],
        }
        local = build_result(
            payload, task="object_detection", device="cuda", inference_ms=4.2, providers=["CUDA"]
        )
        remote = build_result(payload, task="object_detection", device="remote")
        assert local.predictions == remote.predictions  # type: ignore[union-attr]
        assert (remote.device, remote.inference_ms, remote.providers) == ("remote", None, [])


class TestDegeneratePredictionsAreDroppedNotRaised:
    """A junk prediction in the low-confidence tail must not take down the call.

    Below roughly 0.005 confidence a detector returns boxes it has no real opinion
    about, and some of them have zero extent. `BoundingBox.w`/`.h` are `gt=0` and a
    polygon ring needs three points - correctly, because those are not annotations
    anyone can store or draw - so before 1.69.15 ONE such entry raised
    `ValidationError` out of `.predict()` and every good prediction beside it was
    lost with it. Measured on the published `fixture-rfdetr_detection` at
    `confidence=0.001`: entry 77 of 100 had `h == 0.0`, and the call returned
    nothing at all.

    The tests below are written around a SURVIVOR, because "did it raise" alone
    would also pass a fix that returned an empty list.
    """

    def test_a_zero_height_box_is_dropped_and_the_rest_survive(self) -> None:
        result = build_result(
            {
                "predictions": [
                    _bbox(name="keep-me", bounding_box={"x": 4, "y": 5, "w": 6, "h": 7}),
                    _bbox(name="zero-h", bounding_box={"x": 0, "y": 0, "w": 3, "h": 0.0}),
                    _bbox(name="zero-w", bounding_box={"x": 0, "y": 0, "w": 0.0, "h": 3}),
                    _bbox(name="negative", bounding_box={"x": 0, "y": 0, "w": -2, "h": 3}),
                    _bbox(name="keep-me-too"),
                ]
            },
            task="object_detection",
        )
        assert isinstance(result, DetectionResult)
        assert [p.name for p in result.predictions] == ["keep-me", "keep-me-too"]

    def test_a_sub_triangular_ring_is_dropped_and_the_rest_survive(self) -> None:
        """The segmentation twin: a mask that polygonizes to one or two vertices."""
        result = build_result(
            {
                "predictions": [
                    _polygon(
                        name="two-points", polygon={"paths": [[{"x": 0, "y": 0}, {"x": 1, "y": 1}]]}
                    ),
                    _polygon(name="keep-me"),
                    _polygon(
                        name="degenerate-hole",
                        polygon={
                            "paths": [
                                [{"x": 0, "y": 0}, {"x": 9, "y": 0}, {"x": 9, "y": 9}],
                                [{"x": 3, "y": 3}, {"x": 4, "y": 4}],
                            ]
                        },
                    ),
                ]
            },
            task="instance_segmentation",
        )
        assert isinstance(result, InstanceSegmentationResult)
        assert [p.name for p in result.predictions] == ["keep-me"]

    def test_a_zero_extent_oriented_box_is_dropped(self) -> None:
        result = build_result(
            {
                "predictions": [
                    _bbox(
                        name="flat-obb",
                        oriented_box={"cx": 5, "cy": 5, "w": 4, "h": 0.0, "angle": 30.0},
                    ),
                    _bbox(name="keep-me"),
                ]
            },
            task="object_detection",
        )
        assert isinstance(result, DetectionResult)
        assert [p.name for p in result.predictions] == ["keep-me"]

    def test_keypoints_are_never_dropped_because_a_point_has_no_extent(self) -> None:
        """The rule is "no spatial extent", not "small" - a point IS the geometry."""
        result = build_result(
            {"predictions": [_keypoint(name="nose"), _keypoint(name="tail")]},
            task="keypoint_detection",
        )
        assert isinstance(result, KeypointResult)
        assert [p.name for p in result.predictions] == ["nose", "tail"]

    def test_a_malformed_prediction_still_raises_rather_than_vanishing(self) -> None:
        """Deliberately narrow. Dropping is for geometry the model genuinely gave
        no extent - not for a producer that emitted nonsense, which is a bug that
        must stay loud."""
        for broken in (
            _bbox(bounding_box={"x": 0, "y": 0, "w": "wide", "h": 3}),
            _bbox(bounding_box={"x": 0, "y": 0, "h": 3}),
            _bbox(name=None),
        ):
            with pytest.raises(ValidationError):
                build_result({"predictions": [broken]}, task="object_detection")
