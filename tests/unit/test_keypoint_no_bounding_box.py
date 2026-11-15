"""THE CONTRACT GUARD: a ``type: "keypoint"`` annotation carries NO ``bounding_box``.

A single point has no extent. ``KeypointAnnotation`` declares no such field, and its
base is ``extra="forbid"`` in BOTH twins (``pictograph.models.annotation`` here, and
the API's own annotation models) - so an extra key is not ignored,
it **RAISES**.

That is easy to get wrong from the producer side: the arity-1 ("keypoint-as-class")
branch of the shared emitter attached a box for exactly this reason - "keeps
box-reading exporters happy" - which made every auto-annotated single-point model
unreadable through `KeypointModel.predict()` and through `client.annotations`.
Nothing caught it, because every existing keypoint proof ran at ENGINE level on raw
dicts and never re-validated them through the models the SDK actually hands users.

So this module deliberately asserts at both altitudes:

* the emitter's exact emitted key set (a producer regression is self-detecting), and
* a real ``KeypointModel.predict()`` over a fake engine (the integration the defect
  actually broke - the assertion that would have caught it).

**Updated for the instance model (SDK 1.68.1).** There is no longer a ``skeleton``
counter-case that DOES carry a box: a multi-joint pose is several ``keypoint``
annotations sharing an ``instance_id``, so the "no box" rule is now universal and the
object's extent is DERIVED by whoever needs it (``_keypoint.instance_bbox``). The
multi-joint tests below therefore assert the opposite of what they used to - that a
multi-joint detection emits N boxless points carrying ONE shared id - which is exactly
the regression lock that keeps the dead primitive from creeping back in through the
emitter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pictograph.inference.models import KeypointModel
from pictograph.inference.results import KeypointResult
from pictograph.models.annotation import KeypointAnnotation

pytest.importorskip("cv2")
pytest.importorskip("onnxruntime")


# The exact keys the emitter is allowed to emit. Spelled out rather than derived, so
# ADDING a key to the emitter fails here and forces a decision about whether the schema
# should carry it. `instance_id` is present on a multi-joint detection and absent on a
# lone point, so the two arities pin two key sets.
KEYPOINT_KEYS = {"id", "name", "type", "keypoint", "confidence", "attributes"}
KEYPOINT_INSTANCE_KEYS = KEYPOINT_KEYS | {"instance_id"}


def _engine(classes: list[str], npc: list[int]) -> Any:
    """A real ``TorchEngine`` standing in as the emitter's ``wrapper`` - it is what
    the keypoint path passes for itself, so the fixture cannot drift from reality."""
    from pictograph.inference._torch import TorchEngine

    return TorchEngine(
        module=object(),
        family="rfdetr",
        device="cpu",
        dtype=None,
        checkpoint_path=Path("unused.pth"),
        model_type="keypoint_detection",
        architecture="RF-DETR Keypoint Preview",
        classes=classes,
        input_size=(576, 576),
        num_keypoints_per_class=npc,
        keypoint_names={c: [f"j{i}" for i in range(npc[k])] for k, c in enumerate(classes)},
        skeleton_edges=None,
        keypoint_threshold=0.5,
    )


def _emit_all(
    npc: list[int],
    detections: list[list[list[float]]],
    boxes: list[list[float]] | None = None,
    *,
    classes: list[str] | None = None,
    class_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Every annotation the shared emitter produces for ONE image's ``detections``.

    ``instance_id`` is allocated per CALL (it is image-scoped), so a multi-object case
    has to go through a single call - stitching two calls' output together would hand
    both objects the id 1, which is a fixture bug, not a product one.
    """
    from pictograph.inference._wrappers import dispatch

    names = classes or ["thing"]
    wrapper = _engine(names, npc)
    n = len(detections)
    out = dispatch._keypoint_to_annotations(
        boxes or [[10.0, 10.0, 90.0, 90.0]] * n,
        [0.9] * n,
        class_ids or [0] * n,
        detections,
        wrapper,
        names,
        None,
        0.5,
    )
    assert isinstance(out, list)
    return out


def _emit(npc: list[int], joints: list[list[float]]) -> dict[str, Any]:
    """The single annotation the emitter produces for a single arity-1 detection."""
    anns = _emit_all(npc, [joints])
    assert len(anns) == 1
    return anns[0]


class TestEmitterContract:
    """The producer side: what the ONE shared emitter is allowed to put on the wire."""

    def test_arity_one_keypoint_has_no_bounding_box(self) -> None:
        ann = _emit([1], [[20.0, 15.0, 0.9]])

        assert ann["type"] == "keypoint"
        assert "bounding_box" not in ann, (
            "A keypoint must carry NO bounding_box - a point has no extent and "
            "KeypointAnnotation is extra='forbid', so this key makes the annotation "
            "unreadable through every typed client."
        )
        assert set(ann) <= KEYPOINT_INSTANCE_KEYS

    def test_arity_one_keypoint_validates_as_the_model_the_sdk_returns(self) -> None:
        """The assertion that would have caught the defect. ``extra='forbid'`` makes
        this self-detecting: any stray key raises here, naming itself."""
        parsed = KeypointAnnotation.model_validate(_emit([1], [[20.0, 15.0, 0.9]]))

        assert parsed.type == "keypoint"
        assert (parsed.keypoint.x, parsed.keypoint.y) == (20.0, 15.0)

    def test_keypoint_below_findability_still_has_no_box(self) -> None:
        """The fallback branch (joint under threshold → box centre) is where the box
        was most tempting to keep, since the detector box is right there in scope."""
        ann = _emit([1], [[20.0, 15.0, 0.01]])

        assert "bounding_box" not in ann
        assert KeypointAnnotation.model_validate(ann).keypoint.x == 50.0  # box centre

    def test_a_multi_joint_detection_emits_boxless_points_not_a_skeleton(self) -> None:
        """The regression lock on the dead primitive. A pose used to come back as ONE
        ``skeleton`` annotation carrying a derived box; it is now N ``keypoint``
        annotations, each boxless, and the object's extent is derived by whoever needs
        it. An emitter that reintroduces ``type: "skeleton"`` - or slips a box onto a
        joint - fails here rather than downstream in someone's training run."""
        anns = _emit_all([2], [[[20.0, 30.0, 0.9], [40.0, 50.0, 0.9]]])

        assert len(anns) == 2
        assert {a["type"] for a in anns} == {"keypoint"}
        for ann in anns:
            assert "bounding_box" not in ann
            assert "skeleton" not in ann
            assert set(ann) <= KEYPOINT_INSTANCE_KEYS
            KeypointAnnotation.model_validate(ann)  # extra='forbid' - self-detecting

    def test_the_joints_of_one_detection_share_an_instance_id(self) -> None:
        """The grouping IS the supervision signal for a top-down keypoint head. Without
        a shared id these are two unassociated landmarks, and multi-instance pose cannot
        be trained from them."""
        anns = _emit_all([2], [[[20.0, 30.0, 0.9], [40.0, 50.0, 0.9]]])

        ids = {a["instance_id"] for a in anns}
        assert ids == {1}, "1-based, per image, in detection order"

    def test_two_detections_get_distinct_instance_ids(self) -> None:
        """The case the whole change exists for: two objects on one image stay two
        objects, in detection order."""
        anns = _emit_all(
            [2],
            [
                [[20.0, 30.0, 0.9], [40.0, 50.0, 0.9]],
                [[120.0, 130.0, 0.9], [140.0, 150.0, 0.9]],
            ],
            boxes=[[10.0, 10.0, 90.0, 90.0], [110.0, 110.0, 190.0, 190.0]],
        )

        assert len(anns) == 4
        assert [a["instance_id"] for a in anns] == [1, 1, 2, 2]

    def test_the_joints_are_named_for_their_own_classes(self) -> None:
        """A joint is a CLASS. The emitter names each point for the joint it denotes,
        not for the object class - that is what lets the exporter slot it into the
        class template by NAME rather than by list position."""
        anns = _emit_all([2], [[[20.0, 30.0, 0.9], [40.0, 50.0, 0.9]]])

        assert [a["name"] for a in anns] == ["j0", "j1"]

    def test_a_multi_joint_detection_regroups_into_one_instance(self) -> None:
        """End to end through the canonical grouping rule: what the emitter puts on the
        wire is what ``group_instances`` reads back as ONE object."""
        from pictograph._keypoint import group_instances

        parsed = [
            KeypointAnnotation.model_validate(a)
            for a in _emit_all([2], [[[20.0, 30.0, 0.9], [40.0, 50.0, 0.9]]])
        ]
        instances = group_instances(parsed, ["j0", "j1"])

        assert len(instances) == 1
        assert instances[0].instance_id == 1
        assert [p.name for p in instances[0].points] == ["j0", "j1"]


class TestPredictPath:
    """The integration the defect actually broke.

    ``KeypointModel.predict()`` → ``KeypointResult.model_validate`` is the surface a
    user touches, and it RAISED for every arity-1 model while the engine-level proofs
    stayed green. Driving the real model class over a fake engine is what closes that
    gap without needing weights.
    """

    class _FakeEngine:
        """Minimal engine: returns whatever the shared emitter produced."""

        backend = "onnxruntime"
        device = "cpu"

        def __init__(self, predictions: list[dict[str, Any]]) -> None:
            self._predictions = predictions
            self.providers = ["CPUExecutionProvider"]
            self.classes = ["thing"]

        # The image and the confidence gate are irrelevant here on purpose: this
        # fixture replays a fixed emitter output so the assertions are about the
        # annotation CONTRACT, not about decoding.
        def infer(self, image: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
            return {"model_type": "keypoint_detection", "predictions": self._predictions}

        def infer_batch(self, images: list[Any], **kwargs: Any) -> list[dict[str, Any]]:  # noqa: ARG002
            return [self.infer(i) for i in images]

    def _model(self, predictions: list[dict[str, Any]]) -> KeypointModel:
        return KeypointModel(
            engine=self._FakeEngine(predictions),
            model_id="m-1",
            name="fake-keypoint",
            architecture="RF-DETR Keypoint Preview",
            confidence=0.5,
        )

    @staticmethod
    def _image() -> Any:
        import numpy as np

        return np.zeros((64, 64, 3), dtype=np.uint8)

    def test_predict_returns_a_populated_result_for_an_arity_one_model(self) -> None:
        emitted = _emit([1], [[20.0, 15.0, 0.9]])

        result = self._model([emitted]).predict(self._image())

        assert isinstance(result, KeypointResult)
        assert len(result.predictions) == 1
        assert len(result.points) == 1
        point = result.points[0]
        assert point.name == "thing"
        assert (point.keypoint.x, point.keypoint.y) == (20.0, 15.0)

    def test_predict_raises_if_a_producer_ever_reattaches_the_box(self) -> None:
        """Pins the failure mode itself, so the guard above cannot be weakened into a
        tautology: put the box back and `predict()` dies - it does not silently ignore
        the key. This is the behaviour that made the defect a hard break rather than a
        cosmetic one."""
        import pydantic

        poisoned = {
            **_emit([1], [[20.0, 15.0, 0.9]]),
            "bounding_box": {"x": 0, "y": 0, "w": 1, "h": 1},
        }

        with pytest.raises(pydantic.ValidationError, match="bounding_box"):
            self._model([poisoned]).predict(self._image())

    def test_predict_handles_a_mixed_arity_model(self) -> None:
        """A model may carry lone-landmark classes and multi-joint classes at once. Both
        arrive as ``keypoint`` annotations through the one result - what distinguishes
        them is ``instance_id``, not a second annotation type."""
        emitted = _emit_all(
            [1, 2],
            [[[20.0, 15.0, 0.9]], [[1.0, 2.0, 0.9], [3.0, 4.0, 0.9]]],
            boxes=[[10.0, 10.0, 90.0, 90.0], [0.0, 0.0, 20.0, 20.0]],
            classes=["dot", "pose"],
            class_ids=[0, 1],
        )
        result = self._model(emitted).predict(self._image())

        assert len(result.points) == 3
        assert len(result.instances) == 2, "the lone landmark is its own instance"
        assert sorted(len(group) for group in result.instances) == [1, 2]
