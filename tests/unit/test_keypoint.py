"""Keypoint INSTANCES - the model that replaced the ``skeleton`` primitive (SDK 1.68.1).

A joint is a CLASS; ``instance_id`` is the OBJECT. Three people at seventeen joints each
is fifty-one ``keypoint`` annotations whose ``name``s are the joint class names and whose
``instance_id``s are 1, 2, 3 - not three ``skeleton`` annotations.

This module replaces ``test_skeleton.py``. It keeps every invariant that file was
protecting, re-pointed at the new shape:

* the union member parses (``extra="forbid"`` means a missed field RAISES, it is not
  ignored - the reason `requirements.txt` pins a hard SDK floor),
* an absent joint serializes ``0, 0, 0`` (COCO readers key on ``v == 0`` and will plot a
  real coordinate left beside it),
* the derived box is a real box (``MIN_KEYPOINT_SIDE``), never zero-area,
* the augment remapper does not silently mangle a keypoint (the ``else:`` catch-all that
  hands an unhandled union member to the WRONG handler),
* and COCO round-trips the GROUPING, not a bag of loose points.

Plus the two new obligations: ``instance_id`` round-trips, and ``Skeleton*`` is gone from
the public API.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from pictograph._keypoint import (
    MIN_KEYPOINT_SIDE,
    OCCLUDED_ATTRIBUTE,
    VIS_OCCLUDED,
    VIS_UNLABELED,
    VIS_VISIBLE,
    annotation_attributes,
    coco_keypoints,
    group_instances,
    instance_bbox,
    match_template,
    num_labeled,
    point_visibility,
    slot_instance,
)
from pictograph.augment._geometry import remap_annotations
from pictograph.formats import from_coco, to_coco
from pictograph.models.annotation import Annotation, KeypointAnnotation
from pictograph.models.common import Point

_ADAPTER: TypeAdapter[Annotation] = TypeAdapter(Annotation)

# The COCO-ish template the fixtures below are drawn against: node names are JOINT CLASS
# names, edges are 0-INDEXED pairs into `nodes` (the +1 to COCO's 1-indexing happens
# exactly once, in the writer).
PERSON_TEMPLATE: dict[str, Any] = {
    "nodes": [
        {"name": "head", "x": 0.5, "y": 0.1},
        {"name": "left_hand", "x": 0.2, "y": 0.5},
        {"name": "right_hand", "x": 0.8, "y": 0.5},
        {"name": "left_foot", "x": 0.35, "y": 0.95},
    ],
    "edges": [[0, 1], [0, 2], [1, 3]],
}
JOINTS = ["head", "left_hand", "right_hand", "left_foot"]


def _kp(
    name: str,
    x: float,
    y: float,
    instance_id: int | None = None,
    attributes: dict[str, object] | list[object] | None = None,
) -> KeypointAnnotation:
    return KeypointAnnotation(
        id=f"{name}-{instance_id}",
        name=name,
        keypoint=Point(x=x, y=y),
        instance_id=instance_id,
        attributes=[] if attributes is None else attributes,
    )


def _visibilities(flat: list[float]) -> list[float]:
    """Just the ``v`` of each ``[x, y, v]`` triplet."""
    return flat[2::3]


def _one_person() -> list[KeypointAnnotation]:
    """One object, three of the four template joints placed (left_foot never was)."""
    return [
        _kp("head", 100, 40, 1),
        _kp("left_hand", 60, 120, 1),
        _kp("right_hand", 160, 120, 1),
    ]


def _two_people() -> list[KeypointAnnotation]:
    """TWO objects on one image, joints interleaved - which is exactly how the editor
    emits them, and exactly what a positional grouping rule would get wrong."""
    return [
        _kp("head", 100, 40, 1),
        _kp("head", 300, 50, 2),
        _kp("left_hand", 60, 120, 1),
        _kp("left_hand", 260, 130, 2),
        _kp("right_hand", 160, 120, 1),
        _kp("left_foot", 280, 200, 2),
    ]


# ───────────── the field itself ─────────────


class TestInstanceIdOnTheWire:
    def test_instance_id_round_trips_through_model_validate(self) -> None:
        """The whole point of the change. ``extra="forbid"`` makes this self-detecting:
        without the field the wire dict RAISES rather than being quietly ignored."""
        ann = KeypointAnnotation.model_validate(
            {
                "id": "a",
                "name": "nose",
                "type": "keypoint",
                "keypoint": {"x": 1.0, "y": 2.0},
                "confidence": 1.0,
                "created_by": None,
                "attributes": [],
                "instance_id": 3,
            }
        )
        assert ann.instance_id == 3
        assert ann.model_dump(mode="json", exclude_none=True)["instance_id"] == 3

    def test_it_parses_through_the_discriminated_union(self) -> None:
        ann = _ADAPTER.validate_python(
            {"name": "nose", "type": "keypoint", "keypoint": {"x": 1, "y": 2}, "instance_id": 1}
        )
        assert isinstance(ann, KeypointAnnotation)
        assert ann.instance_id == 1

    def test_it_is_last_in_the_canonical_wire_order(self) -> None:
        """The wire order is a contract shared with the backend and the editor; the new
        optional field goes LAST so nothing before it shifts."""
        keys = list(_kp("nose", 1, 2, 1).model_dump(mode="json").keys())
        assert keys == [
            "id",
            "name",
            "type",
            "keypoint",
            "confidence",
            "created_by",
            "attributes",
            "instance_id",
        ]

    def test_an_unassociated_point_stays_byte_identical(self) -> None:
        """The stored form dumps with ``exclude_none``, so an annotation that does not use
        the field is unchanged by the migration - which is why there is no data backfill."""
        dumped = _kp("nose", 1, 2).model_dump(mode="json", exclude_none=True)
        assert set(dumped) == {"id", "name", "type", "keypoint", "confidence", "attributes"}

    def test_instance_id_is_one_based(self) -> None:
        with pytest.raises(ValidationError):
            KeypointAnnotation(name="nose", keypoint=Point(x=0, y=0), instance_id=0)

    def test_a_keypoint_still_carries_no_bounding_box(self) -> None:
        """The other half of the contract survives the rewrite: a point has no extent."""
        with pytest.raises(ValidationError, match="bounding_box"):
            _ADAPTER.validate_python(
                {
                    "name": "nose",
                    "type": "keypoint",
                    "keypoint": {"x": 1, "y": 2},
                    "bounding_box": {"x": 0, "y": 0, "w": 1, "h": 1},
                }
            )


class TestSkeletonIsGone:
    """No back-compat alias: zero rows carried ``type: "skeleton"`` and there are no
    external SDK consumers, so a shim would only keep the dead paradigm alive."""

    def test_skeleton_names_are_not_importable(self) -> None:
        import pictograph
        import pictograph.models as models_pkg

        for name in ("SkeletonAnnotation", "SkeletonGeometry", "SkeletonNode"):
            assert name not in pictograph.__all__
            assert name not in models_pkg.__all__
            assert not hasattr(pictograph, name)
            assert not hasattr(models_pkg, name)

    def test_the_skeleton_module_is_gone(self) -> None:
        with pytest.raises(ImportError):
            import pictograph._skeleton  # noqa: F401

    def test_a_skeleton_shaped_payload_no_longer_validates(self) -> None:
        with pytest.raises(ValidationError):
            _ADAPTER.validate_python(
                {
                    "name": "person",
                    "type": "skeleton",
                    "skeleton": {"nodes": [{"name": "nose", "x": 1, "y": 2}], "edges": []},
                }
            )


# ───────────── grouping ─────────────


class TestGrouping:
    def test_interleaved_joints_group_by_instance_not_by_position(self) -> None:
        instances = group_instances(_two_people(), JOINTS)
        assert [i.instance_id for i in instances] == [1, 2]
        assert [p.name for p in instances[0].points] == ["head", "left_hand", "right_hand"]
        assert [p.name for p in instances[1].points] == ["head", "left_hand", "left_foot"]

    def test_ordering_is_deterministic_across_runs(self) -> None:
        """Two runs over the same image MUST produce byte-identical output - an exporter
        whose object order wanders re-shuffles a dataset on every re-export."""
        first = [
            (i.instance_id, [p.id for p in i.points])
            for i in group_instances(_two_people(), JOINTS)
        ]
        second = [
            (i.instance_id, [p.id for p in i.points])
            for i in group_instances(list(reversed(_two_people())), JOINTS)
        ]
        assert first == second

    def test_unassociated_points_are_singletons_and_come_last(self) -> None:
        anns = [_kp("corner", 5, 5), *_one_person(), _kp("corner", 9, 9)]
        instances = group_instances(anns, JOINTS)
        assert [i.instance_id for i in instances] == [1, None, None]
        assert [len(i.points) for i in instances] == [3, 1, 1]
        # …in annotation order, so the two lone corners never swap.
        assert [i.points[0].keypoint.x for i in instances[1:]] == [5.0, 9.0]

    def test_non_keypoint_annotations_are_ignored(self) -> None:
        from pictograph.models.annotation import BBoxAnnotation
        from pictograph.models.common import BoundingBox

        box = BBoxAnnotation(name="car", bounding_box=BoundingBox(x=0, y=0, w=1, h=1))
        assert len(group_instances([box, *_one_person()], JOINTS)) == 1

    def test_points_are_ordered_by_template_index(self) -> None:
        shuffled = [_kp("right_hand", 160, 120, 1), _kp("head", 100, 40, 1)]
        instance = group_instances(shuffled, JOINTS)[0]
        assert [p.name for p in instance.points] == ["head", "right_hand"]


class TestDerivedBox:
    def test_the_box_encloses_the_placed_points(self) -> None:
        box = instance_bbox(_one_person())
        assert (box.x, box.y, box.w, box.h) == (60.0, 40.0, 100.0, 80.0)

    def test_a_lone_point_gets_a_real_box_not_a_zero_area_one(self) -> None:
        """The box is NOT optional: a detection transformer matches queries to objects by
        box first, and a zero-area COCO bbox is silently dropped by most readers."""
        box = instance_bbox([_kp("corner", 10, 20)])
        assert (box.w, box.h) == (MIN_KEYPOINT_SIDE, MIN_KEYPOINT_SIDE)
        assert (box.x, box.y) == (9.5, 19.5)  # centred on the point

    def test_a_collinear_instance_is_floored_on_the_degenerate_axis_only(self) -> None:
        box = instance_bbox([_kp("a", 10, 20, 1), _kp("b", 10, 60, 1)])
        assert (box.x, box.w) == (9.5, MIN_KEYPOINT_SIDE)
        assert (box.y, box.h) == (20.0, 40.0)


class TestCocoFlattening:
    def test_an_absent_joint_serializes_as_zero_zero_zero(self) -> None:
        """COCO readers key on ``v == 0`` to mean absent, and will happily plot a real
        coordinate carrying v=0 if the position is left in."""
        flat = coco_keypoints(_one_person(), JOINTS)
        assert flat == [100.0, 40.0, 2.0, 60.0, 120.0, 2.0, 160.0, 120.0, 2.0, 0.0, 0.0, 0.0]
        assert num_labeled(flat) == 3

    def test_a_point_outside_the_template_is_not_dropped(self) -> None:
        """It falls back to its own arity-1 category - today's keypoint-as-class
        behaviour. Silence is what the original bug was."""
        slots, spilled = slot_instance([*_one_person(), _kp("antenna", 1, 1, 1)], JOINTS)
        assert [s.name if s else None for s in slots] == [
            "head",
            "left_hand",
            "right_hand",
            None,
        ]
        assert [p.name for p in spilled] == ["antenna"]

    def test_a_duplicate_joint_on_one_instance_spills_rather_than_overwrites(self) -> None:
        """The editor deliberately allows two annotations of one class on ONE instance
        (click the ID bubble). COCO has exactly one slot per joint name, so the first
        wins the slot and the second is exported as its own point - never silently lost.
        """
        dup = [_kp("head", 100, 40, 1), _kp("head", 105, 45, 1)]
        slots, spilled = slot_instance(dup, JOINTS)
        assert slots[0] is not None and slots[0].keypoint.x == 100.0
        assert [p.keypoint.x for p in spilled] == [105.0]

    def test_without_a_template_the_points_flatten_in_order(self) -> None:
        flat = coco_keypoints(_one_person(), [])
        assert flat == [100.0, 40.0, 2.0, 60.0, 120.0, 2.0, 160.0, 120.0, 2.0]


class TestTemplateMatching:
    def test_the_best_covering_template_wins(self) -> None:
        templates = {
            "person": PERSON_TEMPLATE,
            "robot": {"nodes": [{"name": "antenna"}, {"name": "head"}], "edges": []},
        }
        assert match_template(_one_person(), templates) == "person"

    def test_an_instance_matching_nothing_returns_none(self) -> None:
        assert match_template([_kp("corner", 1, 1)], {"person": PERSON_TEMPLATE}) is None

    def test_a_tie_breaks_on_the_given_order_not_alphabetically(self) -> None:
        """LOCKSTEP with the service's ``keypoint_instances._ordered_templates``.

        Two templates covering an instance equally must resolve the same way on the
        server and in the SDK, or the same instance exports under a DIFFERENT COCO
        category depending on which side wrote it.

        The server orders candidates by the dataset's ``class_names`` (ontology order)
        and falls back to dict order - never alphabetically. This used to be
        ``sorted(templates)``, so on the mapping below the server answered "zebra" and
        the SDK answered "antelope". Iterating as given is the half of that rule the
        SDK can apply: it is handed no ``class_names``, but a mapping built from
        ``project_config`` is already in ontology order.

        The two names are chosen so alphabetical and insertion order DISAGREE - with
        any pair where they coincide this test would pass either way.
        """
        shared = {"nodes": [{"name": "nose"}, {"name": "eye"}], "edges": []}
        instance = [_kp("nose", 1, 1), _kp("eye", 2, 2)]

        assert match_template(instance, {"zebra": shared, "antelope": shared}) == "zebra"
        # ...and it genuinely follows the mapping rather than any fixed rule:
        assert match_template(instance, {"antelope": shared, "zebra": shared}) == "antelope"


# ───────────── COCO ─────────────


class TestCocoTwoInstanceExport:
    """The fixture the contract calls for: TWO objects on one image."""

    def test_each_instance_becomes_one_coco_annotation(self) -> None:
        coco = to_coco(
            {"pose.png": _two_people()},
            image_sizes={"pose.png": (400, 300)},
            keypoint_templates={"person": PERSON_TEMPLATE},
        )
        entries = coco["annotations"]
        assert len(entries) == 2, "one COCO annotation per INSTANCE, not per joint"

        cat = next(c for c in coco["categories"] if c["name"] == "person")
        assert cat["keypoints"] == JOINTS
        # COCO's `skeleton` is 1-INDEXED; ours is 0-indexed. The +1 happens once, here.
        assert cat["skeleton"] == [[1, 2], [1, 3], [2, 4]]
        assert all(e["category_id"] == cat["id"] for e in entries)

    def test_the_per_instance_joint_arrays_are_correct(self) -> None:
        coco = to_coco({"pose.png": _two_people()}, keypoint_templates={"person": PERSON_TEMPLATE})
        first, second = coco["annotations"]

        # Person 1: head, left_hand, right_hand placed; left_foot never was.
        assert first["keypoints"] == [
            100.0, 40.0, 2.0,
            60.0, 120.0, 2.0,
            160.0, 120.0, 2.0,
            0.0, 0.0, 0.0,
        ]  # fmt: skip
        assert first["num_keypoints"] == 3
        assert first["bbox"] == [60.0, 40.0, 100.0, 80.0]

        # Person 2: head, left_hand, left_foot placed; right_hand never was - and its
        # slot is the THIRD triplet, not appended at the end.
        assert second["keypoints"] == [
            300.0, 50.0, 2.0,
            260.0, 130.0, 2.0,
            0.0, 0.0, 0.0,
            280.0, 200.0, 2.0,
        ]  # fmt: skip
        assert second["num_keypoints"] == 3
        assert second["bbox"] == [260.0, 50.0, 40.0, 150.0]

    def test_the_export_is_byte_stable(self) -> None:
        import json

        payload = {"pose.png": _two_people()}
        templates = {"person": PERSON_TEMPLATE}
        a = json.dumps(to_coco(payload, keypoint_templates=templates), sort_keys=False)
        b = json.dumps(
            to_coco({"pose.png": list(reversed(_two_people()))}, keypoint_templates=templates),
            sort_keys=False,
        )
        assert a == b


class TestCocoRoundTrip:
    def test_a_two_instance_pose_survives_to_coco_and_back(self) -> None:
        coco = to_coco({"pose.png": _two_people()}, keypoint_templates={"person": PERSON_TEMPLATE})
        back = from_coco(coco)

        anns = back.annotations["pose.png"]
        assert all(isinstance(a, KeypointAnnotation) for a in anns)
        # 6 placed joints across 2 objects - the absent ones do not come back as points.
        assert len(anns) == 6

        regrouped = group_instances(anns, JOINTS)
        assert [i.instance_id for i in regrouped] == [1, 2]
        assert [p.name for p in regrouped[0].points] == ["head", "left_hand", "right_hand"]
        assert [p.name for p in regrouped[1].points] == ["head", "left_hand", "left_foot"]
        assert regrouped[1].points[2].keypoint.x == 280.0

    def test_the_template_comes_back_so_the_class_can_be_recreated(self) -> None:
        """The joint names are what every ``[x, y, v]`` triplet is positionally aligned
        to. Import them and the destination project can declare the same class; drop
        them and the grouping is un-exportable on the way back out."""
        coco = to_coco({"pose.png": _two_people()}, keypoint_templates={"person": PERSON_TEMPLATE})
        back = from_coco(coco)

        template = back.keypoint_templates["person"]
        assert [n["name"] for n in template["nodes"]] == JOINTS
        assert template["edges"] == [[0, 1], [0, 2], [1, 3]], "edges come back 0-indexed"
        # The joint classes must reach the destination project, or every imported
        # annotation references a class that does not exist there.
        for joint in JOINTS:
            assert joint in back.class_names

    def test_a_single_name_keypoint_category_still_imports_as_a_lone_point(self) -> None:
        """Pictograph's own lone-keypoint export writes a one-name category. It must keep
        round-tripping as an UNASSOCIATED point, not acquire a spurious instance."""
        coco = {
            "images": [{"id": 1, "file_name": "a.png"}],
            "categories": [{"id": 1, "name": "corner", "keypoints": ["corner"], "skeleton": []}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "keypoints": [5, 6, 2],
                    "num_keypoints": 1,
                }
            ],
        }
        anns = from_coco(coco).annotations["a.png"]
        assert len(anns) == 1
        assert anns[0].type == "keypoint"
        assert isinstance(anns[0], KeypointAnnotation)
        assert anns[0].instance_id is None

    def test_a_template_less_multi_point_entry_still_groups(self) -> None:
        """A plain detection category carrying triplets has no joint names, so every point
        takes the category's own name - but they still came from ONE object, and that
        grouping is the one thing the old reader threw away."""
        coco = {
            "images": [{"id": 1, "file_name": "a.png"}],
            "categories": [{"id": 1, "name": "hand"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "keypoints": [3, 4, 2, 7, 8, 0, 9, 9, 1],  # middle one v=0 → skipped
                }
            ],
        }
        anns = from_coco(coco).annotations["a.png"]
        assert len(anns) == 2
        assert {a.instance_id for a in anns} == {1}


# ───────────── visibility: the third state a placed point can be in ─────────────


class TestPointVisibilityMatchesTheBackend:
    """``point_visibility`` is a port of the service's helper of the same name.

    The SDK used to hard-code v=2 for every placed point, so an imported OCCLUDED joint
    was silently promoted to "plainly visible" the moment it was exported. These pin the
    backend's rule - the attribute NAME, the truthy spellings, and the fallbacks.
    """

    def test_a_plain_placed_point_is_visible(self) -> None:
        assert point_visibility(_kp("head", 1, 2)) == VIS_VISIBLE == 2

    def test_the_occluded_attribute_downgrades_it_to_v1(self) -> None:
        assert point_visibility(_kp("head", 1, 2, attributes={"occluded": "true"})) == VIS_OCCLUDED
        assert VIS_OCCLUDED == 1

    @pytest.mark.parametrize("raw", ["true", "1", "yes", "y", "t", "occluded"])
    def test_every_backend_truthy_spelling_is_honoured(self, raw: str) -> None:
        """``_TRUTHY = {"true", "1", "yes", "y", "t", "occluded"}`` - copied verbatim.
        A spelling the SDK does not know is a joint the SDK exports as visible."""
        assert point_visibility(_kp("head", 1, 2, attributes={"occluded": raw})) == VIS_OCCLUDED

    @pytest.mark.parametrize("raw", ["TRUE", "  Yes  ", "Occluded", "T"])
    def test_the_match_is_case_and_whitespace_insensitive(self, raw: str) -> None:
        """The backend does ``str(raw).strip().lower()`` before the set test."""
        assert point_visibility(_kp("head", 1, 2, attributes={"occluded": raw})) == VIS_OCCLUDED

    @pytest.mark.parametrize("raw", ["false", "0", "no", "", "maybe", "2"])
    def test_anything_outside_the_truthy_set_stays_visible(self, raw: str) -> None:
        assert point_visibility(_kp("head", 1, 2, attributes={"occluded": raw})) == VIS_VISIBLE

    @pytest.mark.parametrize(("raw", "expected"), [(True, VIS_OCCLUDED), (False, VIS_VISIBLE)])
    def test_a_real_bool_is_read_as_a_bool_not_stringified(self, raw: bool, expected: int) -> None:
        """The backend checks ``isinstance(raw, bool)`` FIRST. Without that branch
        ``str(False).lower()`` is ``"false"`` (correctly falsy) but ``str(True).lower()``
        is ``"true"`` - right by luck; the explicit branch is what makes it a rule."""
        assert point_visibility(_kp("head", 1, 2, attributes={"occluded": raw})) == expected

    def test_an_unrelated_attribute_does_not_occlude(self) -> None:
        assert point_visibility(_kp("head", 1, 2, attributes={"pose": "standing"})) == VIS_VISIBLE

    def test_the_legacy_list_form_of_attributes_degrades_to_visible(self) -> None:
        """``attributes`` still accepts the pre-ontology LIST shape. It carries no names,
        so there is nothing to read - the backend's ``isinstance(attrs, dict)`` guard is
        what makes that a defined answer instead of an AttributeError."""
        assert point_visibility(_kp("head", 1, 2, attributes=["occluded"])) == VIS_VISIBLE

    def test_it_reads_a_non_keypoint_annotation_too(self) -> None:
        """The backend's twin takes any annotation dict. Keeping the SDK's signature on
        the union means the COCO writer can call it without narrowing gymnastics."""
        from pictograph.models.annotation import BBoxAnnotation
        from pictograph.models.common import BoundingBox

        box = BBoxAnnotation(
            name="car",
            bounding_box=BoundingBox(x=0, y=0, w=4, h=4),
            attributes={"occluded": "true"},
        )
        assert point_visibility(box) == VIS_OCCLUDED


class TestOccludedIsNotSaidTwice:
    """``annotation_attributes`` - the port of the backend's same-named helper."""

    def test_occluded_is_stripped_from_the_exported_attribute_set(self) -> None:
        point = _kp("head", 1, 2, attributes={"occluded": "true", "pose": "standing"})
        assert annotation_attributes(point) == {"pose": "standing"}
        assert OCCLUDED_ATTRIBUTE not in annotation_attributes(point)

    def test_an_attribute_set_that_was_only_occluded_comes_back_empty(self) -> None:
        assert annotation_attributes(_kp("head", 1, 2, attributes={"occluded": "true"})) == {}

    def test_the_legacy_list_form_yields_an_empty_map(self) -> None:
        assert annotation_attributes(_kp("head", 1, 2, attributes=["anything"])) == {}


class TestCocoFlatteningCarriesVisibility:
    def test_a_plain_placed_point_flattens_as_v2(self) -> None:
        flat = coco_keypoints(_one_person(), JOINTS)
        assert _visibilities(flat) == [2.0, 2.0, 2.0, 0.0]

    def test_an_occluded_point_flattens_as_v1_not_v2(self) -> None:
        """THE defect. The SDK emitted 2.0 here, so an occluded joint imported from
        V7 / Roboflow / COCO-pose left as "plainly visible"."""
        points = [
            _kp("head", 100, 40, 1),
            _kp("left_hand", 60, 120, 1, attributes={"occluded": "true"}),
            _kp("right_hand", 160, 120, 1),
        ]
        flat = coco_keypoints(points, JOINTS)
        assert flat == [100.0, 40.0, 2.0, 60.0, 120.0, 1.0, 160.0, 120.0, 2.0, 0.0, 0.0, 0.0]

    def test_an_absent_template_slot_is_still_zero_zero_zero(self) -> None:
        """Occlusion must not leak into the ABSENT encoding: v=1 is a placed point whose
        coordinate is real, v=0 is a slot with no point and a meaningless coordinate."""
        flat = coco_keypoints(_one_person(), JOINTS)
        assert flat[-3:] == [0.0, 0.0, 0.0]
        assert VIS_UNLABELED == 0

    def test_an_occluded_joint_still_counts_as_labelled(self) -> None:
        """``num_keypoints`` counts ``v > 0``. An occluded joint IS labelled - its
        position is known - so it counts; only the absent slot does not."""
        points = [
            _kp("head", 100, 40, 1, attributes={"occluded": "true"}),
            _kp("left_hand", 60, 120, 1),
        ]
        flat = coco_keypoints(points, JOINTS)
        assert _visibilities(flat) == [1.0, 2.0, 0.0, 0.0]
        assert num_labeled(flat) == 2

    def test_the_template_less_shape_carries_visibility_too(self) -> None:
        points = [_kp("corner", 5, 6, attributes={"occluded": True})]
        assert coco_keypoints(points, []) == [5.0, 6.0, 1.0]


class TestCocoVisibilityRoundTrip:
    """import → export → import, on real emitted triplets."""

    @staticmethod
    def _pose_coco(visibilities: tuple[int, int, int]) -> dict[str, Any]:
        v1, v2, v3 = visibilities
        return {
            "images": [{"id": 1, "file_name": "pose.png", "width": 400, "height": 300}],
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "keypoints": JOINTS,
                    "skeleton": [[1, 2], [1, 3], [2, 4]],
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "keypoints": [100, 40, v1, 60, 120, v2, 160, 120, v3, 0, 0, 0],
                    "num_keypoints": 3,
                    "bbox": [60, 40, 100, 80],
                }
            ],
        }

    def test_the_reader_maps_v1_onto_the_occluded_attribute(self) -> None:
        """The other direction of the same defect: drop the flag on IMPORT and the round
        trip is broken no matter how faithfully the writer encodes it."""
        anns = from_coco(self._pose_coco((2, 1, 2))).annotations["pose.png"]
        by_name = {a.name: a for a in anns}
        assert by_name["left_hand"].attributes == {"occluded": "true"}
        assert by_name["head"].attributes == []
        assert by_name["right_hand"].attributes == []

    def test_v1_survives_import_then_export_as_v1(self) -> None:
        imp = from_coco(self._pose_coco((2, 1, 2)))
        out = to_coco(
            imp.annotations,
            image_sizes={"pose.png": (400, 300)},
            keypoint_templates=imp.keypoint_templates,
        )
        entry = out["annotations"][0]
        assert entry["keypoints"] == [
            100.0, 40.0, 2.0,
            60.0, 120.0, 1.0,
            160.0, 120.0, 2.0,
            0.0, 0.0, 0.0,
        ]  # fmt: skip
        assert entry["num_keypoints"] == 3

    @pytest.mark.parametrize("visibilities", [(2, 2, 2), (1, 1, 1), (2, 1, 2), (1, 2, 1)])
    def test_every_visibility_combination_is_byte_stable_across_a_round_trip(
        self, visibilities: tuple[int, int, int]
    ) -> None:
        source = self._pose_coco(visibilities)
        first = from_coco(source)
        exported = to_coco(
            first.annotations,
            image_sizes={"pose.png": (400, 300)},
            keypoint_templates=first.keypoint_templates,
        )
        assert exported["annotations"][0]["keypoints"] == [
            float(v) for v in source["annotations"][0]["keypoints"]
        ]
        # And a SECOND lap changes nothing - the encoding is a fixed point, not a
        # one-way decay.
        second = from_coco(exported)
        again = to_coco(
            second.annotations,
            image_sizes={"pose.png": (400, 300)},
            keypoint_templates=second.keypoint_templates,
        )
        assert again["annotations"][0]["keypoints"] == exported["annotations"][0]["keypoints"]

    def test_the_instance_does_not_re_emit_occluded_as_an_object_attribute(self) -> None:
        """If it did, a re-import would hand the FIRST joint's occlusion to every joint of
        the object - the round trip would 'work' once and then quietly spread."""
        out = to_coco(
            {
                "pose.png": [
                    _kp("head", 100, 40, 1, attributes={"occluded": "true", "pose": "standing"}),
                    _kp("left_hand", 60, 120, 1),
                ]
            },
            keypoint_templates={"person": PERSON_TEMPLATE},
        )
        entry = out["annotations"][0]
        assert entry["attributes"] == {"pose": "standing"}
        assert _visibilities(entry["keypoints"]) == [1.0, 2.0, 0.0, 0.0]

        back = from_coco(out).annotations["pose.png"]
        by_name = {a.name: a.attributes for a in back}
        assert by_name["head"] == {"pose": "standing", "occluded": "true"}
        assert by_name["left_hand"] == {"pose": "standing"}, "left_hand must NOT be occluded"

    def test_an_object_whose_only_attribute_was_occluded_emits_no_attributes_key(self) -> None:
        out = to_coco(
            {"pose.png": [_kp("head", 100, 40, 1, attributes={"occluded": "true"})]},
            keypoint_templates={"person": PERSON_TEMPLATE},
        )
        assert "attributes" not in out["annotations"][0]
        assert _visibilities(out["annotations"][0]["keypoints"]) == [1.0, 0.0, 0.0, 0.0]

    def test_a_lone_landmark_round_trips_its_occlusion(self) -> None:
        coco = {
            "images": [{"id": 1, "file_name": "a.png"}],
            "categories": [{"id": 1, "name": "corner", "keypoints": ["corner"], "skeleton": []}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "keypoints": [5, 6, 1]}],
        }
        anns = from_coco(coco).annotations["a.png"]
        assert len(anns) == 1
        assert anns[0].instance_id is None, "still unassociated - occlusion is not grouping"
        assert anns[0].attributes == {"occluded": "true"}
        assert _visibilities(to_coco({"a.png": anns})["annotations"][0]["keypoints"]) == [1.0]

    def test_a_v0_slot_still_produces_no_annotation_at_all(self) -> None:
        anns = from_coco(self._pose_coco((2, 0, 2))).annotations["pose.png"]
        assert [a.name for a in anns] == ["head", "right_hand"]

    def test_an_out_of_range_flag_is_clamped_not_trusted(self) -> None:
        """COCO defines 0/1/2. A 7 is clamped to "visible" and a -1 to "absent", exactly
        as the backend importer does, rather than reaching the exporter as a 7."""
        anns = from_coco(self._pose_coco((7, -1, 2))).annotations["pose.png"]
        assert [a.name for a in anns] == ["head", "right_hand"]
        assert all(a.attributes == [] for a in anns)


# ───────────── the two places a pose used to be corrupted invisibly ─────────────


class TestAugmentDoesNotCorruptAnInstance:
    def test_instance_id_survives_a_geometry_remap_untouched(self) -> None:
        """``instance_id`` is METADATA, not geometry. A flip moves the points; it must not
        renumber, drop, or merge the objects they belong to."""
        flipped = remap_annotations(_two_people(), lambda x, y: (400.0 - x, y), 400.0, 300.0)
        assert [getattr(a, "instance_id", "missing") for a in flipped] == [1, 2, 1, 2, 1, 2]

    def test_a_flip_does_not_rename_left_right_joints(self) -> None:
        """The tempting bug: a mirrored person's left hand really is on the right of the
        frame. But the name is an INDEX into the class's node ordering, and permuting it
        here silently re-aligns every exported triplet against the wrong joint. Mirroring
        belongs in the training pipeline, via YOLO-pose's ``flip_idx``, where it applies
        to the model's output space and not to ground truth."""
        flipped = remap_annotations(_one_person(), lambda x, y: (200.0 - x, y), 200.0, 200.0)
        assert [a.name for a in flipped] == ["head", "left_hand", "right_hand"]
        assert [a.keypoint.x for a in flipped] == [100.0, 140.0, 40.0]  # type: ignore[union-attr]

    def test_a_point_carried_out_of_frame_is_dropped_and_the_rest_survive(self) -> None:
        """Each joint is now its own annotation, so an out-of-frame joint simply leaves -
        it cannot shift the positions of the joints that stayed, because the template
        alignment is by NAME at export time, not by list position."""
        shifted = remap_annotations(
            _one_person(), lambda x, y: (x - 90.0, y), 200.0, 200.0, clip=True
        )
        assert [a.name for a in shifted] == ["head", "right_hand"]
        assert coco_keypoints(shifted, JOINTS) == [  # type: ignore[arg-type]
            10.0, 40.0, 2.0,
            0.0, 0.0, 0.0,
            70.0, 120.0, 2.0,
            0.0, 0.0, 0.0,
        ]  # fmt: skip

    def test_an_instance_entirely_out_of_frame_is_dropped(self) -> None:
        assert (
            remap_annotations(_one_person(), lambda x, y: (x + 900.0, y), 200.0, 200.0, clip=True)
            == []
        )


class TestTilerUsesADerivedExtent:
    def test_a_point_on_a_tile_boundary_is_not_lost(self) -> None:
        """A point's extent is DERIVED (``MIN_KEYPOINT_SIDE``), never read from a field it
        does not have. Read as a zero-area rect it overlaps NOTHING under a strict
        overlap test, so a joint landing exactly on a tile seam vanished from the dataset.
        """
        from PIL import Image as _PILImage

        from pictograph.tile import tile_image

        img = _PILImage.new("RGB", (100, 100))
        tiles = tile_image(img, [_kp("seam", 50, 25, 1)], rows=1, cols=2)
        assert sum(len(t.annotations) for t in tiles) >= 1


def _lit(img: Any) -> int:
    """Count of non-black pixels. ``histogram()[0]`` is the black bucket in mode "L"."""
    grey = img.convert("L")
    return grey.width * grey.height - grey.histogram()[0]


class TestVizDrawsInstances:
    def test_a_lone_point_still_renders(self) -> None:
        from PIL import Image as _PILImage

        from pictograph.viz import draw_annotations

        img = _PILImage.new("RGB", (200, 200), "black")
        out = draw_annotations(img, [_kp("corner", 50, 50)])
        assert out.size == (200, 200)
        assert _lit(out) > 0, "something was drawn"

    def test_a_template_connects_the_joints_of_one_instance(self) -> None:
        """The limbs are the template's job now - group by ``instance_id``, connect via
        the per-class edges. Without the template the joints still render; only the
        cosmetic connectivity is missing."""
        from PIL import Image as _PILImage

        from pictograph.viz import draw_annotations

        base = _PILImage.new("RGB", (400, 300), "black")
        bare = draw_annotations(base, _one_person(), show_labels=False)
        limbed = draw_annotations(
            base,
            _one_person(),
            show_labels=False,
            keypoint_templates={"person": PERSON_TEMPLATE},
        )

        assert _lit(limbed) > _lit(bare), "the edges must add drawn pixels"

    def test_edges_are_not_drawn_between_two_different_instances(self) -> None:
        """The whole reason the grouping exists: two people's heads are not connected."""
        from PIL import Image as _PILImage

        from pictograph.viz import draw_annotations

        base = _PILImage.new("RGB", (400, 300), "black")
        together = draw_annotations(
            base, _two_people(), show_labels=False, keypoint_templates={"person": PERSON_TEMPLATE}
        )
        # Draw each object alone and sum the lit pixels; a renderer that joined the two
        # objects would light strictly MORE than the two drawn apart (the two are far
        # enough apart that their own marks never overlap).
        apart = _lit(
            draw_annotations(
                base,
                [p for p in _two_people() if p.instance_id == 1],
                show_labels=False,
                keypoint_templates={"person": PERSON_TEMPLATE},
            )
        ) + _lit(
            draw_annotations(
                base,
                [p for p in _two_people() if p.instance_id == 2],
                show_labels=False,
                keypoint_templates={"person": PERSON_TEMPLATE},
            )
        )

        assert _lit(together) == apart
