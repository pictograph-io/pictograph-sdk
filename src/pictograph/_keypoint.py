"""Keypoint INSTANCE grouping - the SDK's copy of the one grouping rule.

**A joint is a CLASS. ``instance_id`` is the OBJECT.** Three people at seventeen joints
each is fifty-one ``keypoint`` annotations whose ``name``s are the seventeen joint class
names (``nose``, ``left_eye``, …) and whose ``instance_id``s are 1, 2, 3.

This module replaces ``_skeleton.py`` (SDK 1.68.1). The ``skeleton`` primitive died
because its edge list was redundant - a per-class template identical for every instance -
while the only thing it uniquely carried was instance identity. RF-DETR keypoint is
query-based/top-down, so ground-truth instance grouping IS the supervision signal.
Skeletons are now POSTPROCESSING: group points by ``instance_id``, connect via the
per-class template that lives once on ``project_config``.

It is one of one shared definition, implemented once per language - the backend copy is the
REFERENCE the other three are ported from:

* backend  → the server's keypoint grouping (the reference)
* editor   → the annotation editor
* SDK      → this module
* trainer  → an inline twin in the training-data preprocessor

Two rules in here are a WIRE CONTRACT, not a convention, and are the reason the reasoning
is written down rather than assumed:

**Node ORDER.** The template's node ordering is what COCO's ``categories[].keypoints``
array holds and what every ``[x, y, v]`` triplet is positionally aligned to. If the SDK,
the server and the editor ever disagree on it, a nose lands on an ankle and nothing else
in the system notices.

**VISIBILITY.** Visibility is COCO's, verbatim - 0 = not labelled, 1 = labelled but
occluded, 2 = labelled and visible - chosen so the COCO writer is a straight copy with no
translation table to get backwards. An OCCLUDED joint IS labelled: it counts toward
``num_keypoints`` and toward the object's extent; only v=0 does not. A template slot the
instance LACKS MUST serialize ``0, 0, 0``, because COCO readers key on ``v == 0`` to mean
absent and will happily plot a real coordinate left beside a zero visibility.

v=1 is the one fact a point cannot carry in its geometry, and V7 / Roboflow / COCO-pose
imports all carry it, so it rides on the annotation's ontology ``attributes`` as
``{"occluded": "true"}`` - an existing, exportable, round-trippable field rather than a
new schema key. :func:`point_visibility` is the reader for that encoding and is a
character-for-character port of the backend's, ``_TRUTHY`` set included: an import that
says "occluded" therefore survives an export as v=1 instead of being silently promoted to
"plainly visible". :func:`annotation_attributes` is the other half - an instance does NOT
re-emit ``occluded`` alongside its triplets, or the same fact is said twice and, on a
multi-joint object, one joint's occlusion is attributed to all of them on the way back in.

Dependency-free (no numpy), like ``_obb``: the base SDK installs with nothing but
httpx + pydantic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .models.annotation import KeypointAnnotation
from .models.common import BoundingBox

if TYPE_CHECKING:
    from .models.annotation import Annotation

__all__ = [
    "MIN_KEYPOINT_SIDE",
    "OCCLUDED_ATTRIBUTE",
    "VIS_OCCLUDED",
    "VIS_UNLABELED",
    "VIS_VISIBLE",
    "KeypointInstance",
    "annotation_attributes",
    "coco_keypoints",
    "group_instances",
    "instance_bbox",
    "keypoint_annotations",
    "match_template",
    "num_labeled",
    "point_visibility",
    "slot_instance",
    "template_edge_pairs",
    "template_node_names",
]

VIS_UNLABELED = 0
VIS_OCCLUDED = 1
VIS_VISIBLE = 2

# The ontology attribute that carries COCO's v=1 ("labelled but not visible"), and the
# set of spellings read as TRUE. Both are copied verbatim from the backend twin
# (``keypoint_instances.OCCLUDED_ATTRIBUTE`` / ``_TRUTHY``); a divergence here is a
# silent promotion of an occluded joint to "plainly visible" on one side only.
OCCLUDED_ATTRIBUTE = "occluded"
_TRUTHY = frozenset({"true", "1", "yes", "y", "t", "occluded"})

# An instance whose placed points are collinear (or a single lone landmark) encloses a
# zero-area box, and every axis-aligned consumer downstream divides by or thresholds on
# that extent. Was ``MIN_SKELETON_SIDE``; same behaviour, new name.
MIN_KEYPOINT_SIDE = 1.0


@dataclass(frozen=True)
class KeypointInstance:
    """One OBJECT on one image: the keypoint annotations that share an ``instance_id``.

    ``instance_id`` is ``None`` for an UNASSOCIATED point - a lone landmark with no
    multi-joint object around it. Each such point is its own instance, so a project that
    only ever places single landmarks (the ``cook-medical`` 4x ``bag_corner`` shape) keeps
    working untouched without anyone assigning ids.
    """

    instance_id: int | None
    points: tuple[KeypointAnnotation, ...]


def keypoint_annotations(annotations: Iterable[Annotation]) -> list[KeypointAnnotation]:
    """Just the ``type == "keypoint"`` annotations, in the order given."""
    return [a for a in annotations if isinstance(a, KeypointAnnotation)]


def point_visibility(annotation: Annotation) -> int:
    """``VIS_OCCLUDED`` when the annotation's attributes say so, else ``VIS_VISIBLE``.

    The port of ``keypoint_instances.point_visibility`` in the backend - same attribute
    name, same ``_TRUTHY`` spellings, same precedence - so an occluded joint encodes v=1
    whichever side writes the COCO.

    A point that EXISTS was placed by someone, so it is never v=0: absence is expressed by
    a template slot with no point at all (see :func:`coco_keypoints`), never by a real
    coordinate carrying a zero visibility.

    The legacy LIST form of ``attributes`` (the pre-ontology shape, still accepted by the
    models) carries no names, so there is nothing to read and it degrades to
    ``VIS_VISIBLE`` - exactly as the backend's ``isinstance(attrs, dict)`` guard does.
    """
    attrs = annotation.attributes
    if isinstance(attrs, dict):
        raw = attrs.get(OCCLUDED_ATTRIBUTE)
        if isinstance(raw, bool):
            return VIS_OCCLUDED if raw else VIS_VISIBLE
        if raw is not None and str(raw).strip().lower() in _TRUTHY:
            return VIS_OCCLUDED
    return VIS_VISIBLE


def annotation_attributes(annotation: Annotation) -> dict[str, object]:
    """Exportable ontology attributes, MINUS the one consumed as visibility.

    The port of the backend's ``keypoint_instances.annotation_attributes``. ``occluded``
    is expressed in the ``[x, y, v]`` triplet, so re-emitting it as an INSTANCE attribute
    would say the same thing twice - and, on an instance built from many points, would
    attribute one joint's occlusion to the whole object the next time the file is read.

    (The backend stringifies every surviving value because its export rows are string
    typed; the SDK's ``attributes`` is ``dict[str, object]`` end to end and the COCO writer
    already emits non-keypoint attributes verbatim, so values are passed through unchanged
    here. The visibility rule itself - which key, and which spellings are true - is
    identical.)
    """
    attrs = annotation.attributes
    if not isinstance(attrs, dict) or not attrs:
        return {}
    return {k: v for k, v in attrs.items() if k != OCCLUDED_ATTRIBUTE}


def group_instances(
    annotations: Iterable[Annotation],
    template_names: Sequence[str] | None = None,
) -> list[KeypointInstance]:
    """Group one image's keypoint annotations into objects.

    Args:
        annotations: Every annotation on ONE image. Non-keypoint types are ignored.
        template_names: The class template's joint-name ordering, when known. Used only
            to order the points WITHIN an instance, so an export is stable no matter which
            order the editor emitted them in; off-template points keep annotation order
            and sort last. Grouping itself never depends on it.

    Returns:
        Instances sorted by ``instance_id`` ascending, with the ``None``-keyed singletons
        last in annotation order. **The ordering is deterministic**: two runs over the
        same image produce byte-identical output, so a re-export never re-shuffles a
        dataset.
    """
    keyed: dict[int, list[KeypointAnnotation]] = {}
    singletons: list[KeypointAnnotation] = []
    for point in keypoint_annotations(annotations):
        if point.instance_id is None:
            singletons.append(point)
        else:
            keyed.setdefault(point.instance_id, []).append(point)

    index_of = {name: i for i, name in enumerate(template_names or ())}
    spill_rank = len(index_of)

    out = [
        KeypointInstance(
            instance_id=iid,
            # `sorted` is stable, so two points sharing a name keep annotation order -
            # which is what makes the first-wins slot rule below deterministic.
            points=tuple(sorted(keyed[iid], key=lambda p: index_of.get(p.name, spill_rank))),
        )
        for iid in sorted(keyed)
    ]
    out.extend(KeypointInstance(instance_id=None, points=(p,)) for p in singletons)
    return out


def instance_bbox(points: Sequence[KeypointAnnotation]) -> BoundingBox:
    """The axis-aligned enclosure of an instance's points, floored to a real extent.

    **The box is not optional.** RF-DETR keypoint is a detection transformer: it matches
      queries to objects by BOX first, so an object without one is an object it cannot be
      taught. A zero-area COCO ``bbox`` is likewise dropped or NaN'd by most readers, which
      is why a lone landmark gets a ``MIN_KEYPOINT_SIDE`` square centred on itself rather
      than a ``0 x 0`` rectangle sitting on it.

      Raises:
          ValueError: ``points`` is empty. An instance always has at least one point, so an
              empty one is a programming error - and swallowing it is exactly the silence
              this codebase has been bitten by.
    """
    if not points:
        raise ValueError("instance_bbox() needs at least one point; an instance is never empty.")
    xs = [p.keypoint.x for p in points]
    ys = [p.keypoint.y for p in points]
    x, y = min(xs), min(ys)
    w, h = max(xs) - x, max(ys) - y
    if w < MIN_KEYPOINT_SIDE:
        x -= (MIN_KEYPOINT_SIDE - w) / 2.0
        w = MIN_KEYPOINT_SIDE
    if h < MIN_KEYPOINT_SIDE:
        y -= (MIN_KEYPOINT_SIDE - h) / 2.0
        h = MIN_KEYPOINT_SIDE
    return BoundingBox(x=x, y=y, w=w, h=h)


def slot_instance(
    points: Sequence[KeypointAnnotation],
    template_names: Sequence[str],
) -> tuple[list[KeypointAnnotation | None], list[KeypointAnnotation]]:
    """Place an instance's points into the template by NAME.

    Returns:
        ``(slots, spilled)``. ``slots`` is positionally aligned to ``template_names``,
        with ``None`` wherever the instance has no point for that joint. ``spilled`` is
        every point that could not take a slot.

    A point SPILLS for one of two reasons, and in neither case is it dropped - it falls
    back to its own arity-1 category, which is today's keypoint-as-class behaviour:

    * its ``name`` is not in the template at all, or
    * its slot is already taken. The editor deliberately allows two annotations of one
      class on ONE instance (that is what the ID bubble's cycle is FOR - two ``ear_l``
      auto-label 1 and 2, and clicking the second makes them one object). COCO has exactly
      one slot per joint name and cannot express that, so the first point in order wins
      the slot and the second spills. Silently overwriting would lose a human-placed
      annotation; silently dropping is the bug this codebase has been bitten by twice.
    """
    slots: list[KeypointAnnotation | None] = [None] * len(template_names)
    index_of = {name: i for i, name in enumerate(template_names)}
    spilled: list[KeypointAnnotation] = []
    for point in points:
        index = index_of.get(point.name)
        if index is None or slots[index] is not None:
            spilled.append(point)
            continue
        slots[index] = point
    return slots, spilled


def coco_keypoints(
    points: Sequence[KeypointAnnotation],
    template_names: Sequence[str],
) -> list[float]:
    """Flatten an instance to COCO's ``[x1, y1, v1, x2, y2, v2, ...]``.

    With a template the triplets are positionally aligned to ``template_names`` and an
    unfilled slot serializes ``0, 0, 0`` (see the module docstring - a real coordinate
    left beside ``v == 0`` gets plotted). Without one the points flatten in the order
    given, which is the arity-1 / template-less shape.

    A PLACED point's flag comes from :func:`point_visibility`: v=1 when its ontology
    ``attributes`` say ``occluded``, else v=2. Hard-coding v=2 here - which this did until
    the backend rule was ported - silently promotes every imported occluded joint to
    "plainly visible" on the way back out.
    """
    flat: list[float] = []
    if not template_names:
        for point in points:
            flat.extend(
                [float(point.keypoint.x), float(point.keypoint.y), float(point_visibility(point))]
            )
        return flat

    slots, _ = slot_instance(points, template_names)
    for slot in slots:
        if slot is None:
            flat.extend([0.0, 0.0, 0.0])
        else:
            flat.extend(
                [float(slot.keypoint.x), float(slot.keypoint.y), float(point_visibility(slot))]
            )
    return flat


def num_labeled(flat: Sequence[float]) -> int:
    """COCO's ``num_keypoints`` - the count of triplets with ``v > 0``."""
    return sum(1 for i in range(2, len(flat), 3) if flat[i] > VIS_UNLABELED)


def template_node_names(template: Mapping[str, Any] | None) -> list[str]:
    """The joint CLASS names a ``project_config`` keypoint-class template declares.

    The template shape is retained verbatim from the old skeleton class option::

        {"nodes": [{"name": "nose", "x": 0.5, "y": 0.06}, ...], "edges": [[15, 13], ...]}

    ``nodes[].x/y`` are normalized [0, 1] template positions - a starting pose for the
    editor, not a constraint - and are irrelevant here.
    """
    if not template:
        return []
    nodes = template.get("nodes")
    if not isinstance(nodes, list):
        return []
    out: list[str] = []
    for node in nodes:
        if isinstance(node, Mapping):
            name = node.get("name")
            if isinstance(name, str) and name:
                out.append(name)
    return out


def template_edge_pairs(template: Mapping[str, Any] | None) -> list[tuple[int, int]]:
    """A template's connectivity, as **0-indexed** pairs into its ``nodes``.

    COCO's own ``categories[].skeleton`` is 1-indexed; that ``+1`` happens exactly once,
    in the COCO writer, and its ``-1`` exactly once in the reader. Do not pre-shift.
    Out-of-range and self-loop pairs are dropped rather than emitted as a broken edge.
    """
    if not template:
        return []
    raw = template.get("edges")
    if not isinstance(raw, list):
        return []
    size = len(template_node_names(template))
    out: list[tuple[int, int]] = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            i, j = int(pair[0]), int(pair[1])
        except (TypeError, ValueError):
            continue
        if i == j or not (0 <= i < size) or not (0 <= j < size):
            continue
        out.append((i, j))
    return out


def match_template(
    points: Sequence[KeypointAnnotation],
    templates: Mapping[str, Mapping[str, Any]],
) -> str | None:
    """Which class's template an instance's joints belong to, or ``None``.

      The object class is no longer written on the annotation - a joint is named for its own
      class - so it is recovered by asking which template COVERS the most of the instance's
      joint names. An instance no template covers returns ``None``, and its points export as
      their own arity-1 categories rather than vanishing.

    **Ties break on the order ``templates`` is given in, first match wins.** This used to
      be ``sorted(templates)``, which is deterministic but disagrees with the other two
      members of this kept in sync set: the server's ``keypoint_instances`` and the
      training-pipeline twin in ``data_preprocessing`` both order candidates by the
      project's own ``class_names`` (its ontology order) and fall back to dict order, never
      alphabetically. Two templates covering an instance equally therefore chose DIFFERENT
      classes on the server and in the SDK - the same instance exported under a different
      COCO category depending on which side wrote it.

      The SDK is not handed ``class_names`` (``to_coco`` takes only
      ``keypoint_templates``), so it cannot apply the ontology-order half of that rule.
      Iterating the mapping as given implements the half it can: a caller who builds the
      mapping from ``project_config`` - the shape the docstring on ``to_coco`` describes -
      is in ontology order already, and then the two agree. `dict` preserves insertion
      order, so this is no less deterministic than sorting.
    """
    names = {p.name for p in points}
    best: str | None = None
    best_hits = 0
    for class_name in templates:
        hits = len(names & set(template_node_names(templates[class_name])))
        if hits > best_hits:
            best, best_hits = class_name, hits
    return best
