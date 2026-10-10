"""COCO ⇄ Pictograph annotation conversion (client-side, offline).

:func:`from_coco` parses a standard COCO detection/segmentation/keypoint JSON
into Pictograph :data:`~pictograph.models.annotation.Annotation` objects grouped
by image file name - ready to hand to ``client.annotations.save`` /
``client.annotations.bulk_save`` (sync or async). :func:`to_coco` does the
reverse. Both are pure functions on our own typed models, with zero third-party
dependencies (a deliberate contrast with pulling in a competitor's SDK).

Fidelity notes:

- COCO ``bbox`` is ``[x, y, w, h]`` (top-left + size) - identical to
  :class:`~pictograph.models.common.BoundingBox`, so boxes round-trip exactly.
- A COCO polygon ``segmentation`` is a list of flat coordinate rings, rendered
  as a **union** (there is no hole semantics in the polygon form). Each ring
  therefore maps to its own single-ring :class:`PolygonAnnotation`, and holes
  are not representable - for hole-accurate COCO use the server-side export
  (``client.exports.create(..., format="coco")``, which emits RLE).
- **RLE** (``segmentation`` is a dict) is skipped on import - decoding needs
  ``pycocotools``; those annotations are dropped with the rest preserved.
- Priority when an annotation carries several geometries: visible ``keypoints``
  → segmentation → ``bbox``.

**Keypoints are INSTANCE-grouped.** One COCO annotation is one OBJECT, so a pose
category's ``[x, y, v]`` triplets become one ``keypoint`` annotation per placed
joint - each named for the JOINT's own class, all sharing an ``instance_id``.
There is no ``skeleton`` annotation type; the connectivity lives once per class,
as a template, and travels on :attr:`CocoImport.keypoint_templates` /
:func:`to_coco`'s ``keypoint_templates`` argument.

**Keypoint visibility round-trips in BOTH directions.** COCO's flag is carried
verbatim (0 = not labelled, 1 = labelled but occluded, 2 = visible). On the way in,
``v == 1`` becomes the ontology attribute ``{"occluded": "true"}`` on that joint -
the one fact a bare point cannot hold in its geometry; on the way out,
:func:`pictograph._keypoint.point_visibility` reads it back to v=1. Both halves are
ports of the backend twins (``utils/keypoint_instances.py`` and
``utils/import_formats/_common.py``), so a pose survives
``from_coco → to_coco → from_coco`` unchanged rather than being flattened to
"plainly visible".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pictograph._keypoint import (
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
    slot_instance,
    template_edge_pairs,
    template_node_names,
)
from pictograph.models.annotation import (
    Annotation,
    BBoxAnnotation,
    KeypointAnnotation,
    PolygonAnnotation,
    PolygonGeometry,
)
from pictograph.models.common import BoundingBox, Point

from ._shared import annotation_bbox, flatten_ring, points_from_flat, polygon_area

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass
class CocoImport:
    """Result of :func:`from_coco`.

    Attributes:
        annotations: ``file_name`` → the list of :data:`Annotation` on that image.
            Feed a single entry to ``client.annotations.save`` (after resolving the
            image id) or the whole map - one list per image - to ``bulk_save``.
        class_names: The distinct class names seen, in COCO ``category`` id order -
            use them to create/extend the destination project's class list. A pose
            category contributes its own name AND its joint names, in template order:
            the imported annotations are named for the JOINTS, so a project created
            without them would have every keypoint referencing a class it does not
            declare.
        keypoint_templates: Class name → ``{"nodes": [{"name": ...}], "edges": [[i, j]]}``
            for every pose category, ready to put on that class in ``project_config``.
            ``edges`` come back **0-indexed** (COCO's are 1-indexed; the ``-1`` happens
            once, here). COCO carries no template LAYOUT, so ``nodes`` have only a
            ``name`` - add normalized ``x``/``y`` if you want an editor starting pose.
    """

    annotations: dict[str, list[Annotation]] = field(default_factory=dict)
    class_names: list[str] = field(default_factory=list)
    keypoint_templates: dict[str, dict[str, Any]] = field(default_factory=dict)


def _load(coco: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(coco, (str, Path)):
        with Path(coco).expanduser().open(encoding="utf-8") as fh:
            loaded: Any = json.load(fh)
        if not isinstance(loaded, dict):
            raise ValueError("COCO JSON must be an object at the top level.")
        return loaded
    return coco


def _point(
    name: str,
    x: float,
    y: float,
    confidence: float,
    kp_id: str | None,
    instance_id: int | None,
    attributes: dict[str, object] | None = None,
) -> KeypointAnnotation:
    kwargs: dict[str, Any] = {
        "name": name,
        "keypoint": Point(x=x, y=y),
        "confidence": confidence,
        "instance_id": instance_id,
    }
    if kp_id is not None:
        kwargs["id"] = kp_id
    # Only a non-empty map - an absent one keeps the model's legacy `[]` default, so a
    # point with nothing to say stays byte-identical to what this emitted before.
    if attributes:
        kwargs["attributes"] = attributes
    return KeypointAnnotation(**kwargs)


def _keypoint_annotations(
    kps: Sequence[float],
    joint_names: Sequence[str],
    name: str,
    confidence: float,
    base_id: str | None,
    instance_id: int,
    attributes: dict[str, object] | None = None,
) -> list[Annotation]:
    """Expand one COCO ``keypoints`` entry into per-joint annotations (v > 0).

    **One COCO annotation is one OBJECT**, so every point produced here shares an
      ``instance_id`` - that grouping is the entire supervision signal for a top-down
      keypoint head, and expanding into N *unassociated* points (what the reader used to do
      for a non-pose category) throws it away.

      ``joint_names`` is the category's ``keypoints`` array: the joint CLASS names the
      triplets are positionally aligned to. When it names more than one joint, each point
      takes its JOINT's name. When the category does not declare joint names at all, every
      point falls back to the category name - the arity-1 / keypoint-as-class shape.

      ``v == 0`` means never placed (the module docstring explains why the coordinate beside
      it is meaningless), so those triplets produce no annotation at all. Under the instance
      model that costs nothing: the exporter re-slots by NAME, not by list position, so an
      absent joint cannot shift the ones that are present.

      ``v == 1`` means labelled but OCCLUDED, and that is the one fact a bare point cannot
      carry in its geometry - so it rides on the ontology attribute ``{"occluded": "true"}``,
      the encoding :func:`pictograph._keypoint.point_visibility` reads back on export. Drop
      it here and the round trip is lossy in the other direction: the flag is thrown away on
      the way IN, and every occluded joint comes back out as v=2.

      ``attributes`` are the COCO annotation's own ontology attributes, inherited by each
      point of the instance (a COCO annotation is one object, so it carries one attribute
      set); the per-joint ``occluded`` is layered on top of them. This is the exact shape of
      the backend importer's ``import_formats._common.make_keypoint_instance``.

      A lone point (one placed joint, no joint-name array) stays UNASSOCIATED - it is a
      landmark, not an object, and Pictograph's own lone-keypoint export round-trips
      unchanged.
    """
    placed: list[tuple[int, float, float, int]] = []
    for slot, i in enumerate(range(0, len(kps) - 2, 3)):
        x, y, v = kps[i], kps[i + 1], kps[i + 2]
        try:
            # COCO defines exactly 0/1/2; clamp anything else rather than trusting it into
            # the pipeline, and degrade an unreadable flag to "not labelled" (the backend
            # importer does both).
            visibility = max(VIS_UNLABELED, min(VIS_VISIBLE, int(v)))
        except (TypeError, ValueError):
            continue
        if visibility == VIS_UNLABELED:
            continue
        placed.append((slot, float(x), float(y), visibility))
    if not placed:
        return []

    # A single point from a category with no joint-name array is a lone landmark.
    lone = len(placed) == 1 and len(joint_names) <= 1
    inherited: dict[str, object] = dict(attributes) if attributes else {}
    out: list[Annotation] = []
    for idx, (slot, x, y, visibility) in enumerate(placed):
        joint = joint_names[slot] if slot < len(joint_names) and len(joint_names) > 1 else name
        point_attributes = dict(inherited)
        if visibility == VIS_OCCLUDED:
            point_attributes[OCCLUDED_ATTRIBUTE] = "true"
        out.append(
            _point(
                joint,
                x,
                y,
                confidence,
                f"{base_id}-{idx}" if base_id else None,
                None if lone else instance_id,
                point_attributes or None,
            )
        )
    return out


def _polygon_annotations(
    seg: list[Any], name: str, confidence: float, base_id: str | None
) -> list[Annotation]:
    """Map a COCO polygon ``segmentation`` (list of flat rings) to annotations."""
    out: list[Annotation] = []
    idx = 0
    for ring_coords in seg:
        if not isinstance(ring_coords, list) or len(ring_coords) < 6:
            continue
        pts = points_from_flat([float(c) for c in ring_coords])
        if len(pts) < 3:
            continue
        poly_id = f"{base_id}-{idx}" if base_id and len(seg) > 1 else base_id
        kwargs: dict[str, Any] = {
            "name": name,
            "polygon": PolygonGeometry(paths=[pts]),
            "bounding_box": annotation_bbox(
                PolygonAnnotation(name=name, polygon=PolygonGeometry(paths=[pts]))
            ),
            "confidence": confidence,
        }
        if poly_id is not None:
            kwargs["id"] = poly_id
        out.append(PolygonAnnotation(**kwargs))
        idx += 1
    return out


def from_coco(coco: dict[str, Any] | str | Path) -> CocoImport:
    """Parse a COCO dataset into Pictograph annotations grouped by image file name.

    Args:
        coco: A parsed COCO dict, or a path / JSON string to one. Must have the
            standard ``images`` / ``annotations`` / ``categories`` arrays.

    Returns:
        A :class:`CocoImport` - ``annotations`` (``file_name`` → annotations), the
        ordered ``class_names``, and the ``keypoint_templates`` recovered from any
        pose categories.

    Raises:
        ValueError: The payload is not a COCO object.

    Example:
        >>> imp = from_coco("instances_val.json")  # doctest: +SKIP
        >>> project = client.datasets.create(  # doctest: +SKIP
        ...     "my-set", classes=[{"name": n, "type": "bbox"} for n in imp.class_names]
        ... )
        >>> ids = {img.filename: img.id for img in client.images.iter(project.id)}  # doctest: +SKIP
        >>> for fname, anns in imp.annotations.items():  # doctest: +SKIP
        ...     client.annotations.save(ids[fname], anns)
    """
    data = _load(coco)
    categories: dict[int, str] = {int(c["id"]): str(c["name"]) for c in data.get("categories", [])}
    # A pose category carries its joint NAMES (the array every triplet is positionally
    # aligned to) and its connectivity graph. Both are needed to rebuild the grouping
    # rather than a bag of points, and neither is present on a detection category.
    category_keypoints: dict[int, list[str]] = {
        int(c["id"]): [str(k) for k in c["keypoints"]]
        for c in data.get("categories", [])
        if isinstance(c.get("keypoints"), list) and c["keypoints"]
    }
    category_skeletons: dict[int, list[list[int]]] = {
        int(c["id"]): [[int(a), int(b)] for a, b in c["skeleton"] if True]
        for c in data.get("categories", [])
        if isinstance(c.get("skeleton"), list) and c["skeleton"]
    }
    images: dict[int, str] = {int(im["id"]): str(im["file_name"]) for im in data.get("images", [])}

    result = CocoImport()
    # Preserve COCO category-id order for a stable, reproducible class list. A pose
    # category also contributes its JOINT classes - the annotations it produces are named
    # for the joints, so a project created from `class_names` alone would be missing every
    # class its own keypoints reference.
    for cid in sorted(categories):
        result.class_names.append(categories[cid])
        joints = category_keypoints.get(cid, [])
        if len(joints) > 1:
            result.keypoint_templates[categories[cid]] = {
                # COCO carries no template LAYOUT - only the ordering - so nodes come back
                # with a name and nothing else.
                "nodes": [{"name": j} for j in joints],
                # COCO's `skeleton` is 1-INDEXED into `keypoints`; ours is 0-indexed. The
                # -1 happens here, exactly once, mirroring the +1 in the writer.
                "edges": [
                    [i - 1, j - 1]
                    for i, j in category_skeletons.get(cid, [])
                    if i != j and 1 <= i <= len(joints) and 1 <= j <= len(joints)
                ],
            }
            result.class_names.extend(j for j in joints if j not in result.class_names)

    # `instance_id` is 1-based and scoped to the IMAGE, so the counter restarts per image.
    instance_counter: dict[str, int] = {}

    for ann in data.get("annotations", []):
        image_id = ann.get("image_id")
        cat_id = ann.get("category_id")
        if image_id is None or int(image_id) not in images:
            continue
        name = categories.get(int(cat_id)) if cat_id is not None else None
        if name is None:
            continue
        confidence = float(ann.get("score", 1.0))
        base_id = str(ann["id"]) if "id" in ann else None
        filename = images[int(image_id)]
        # Preserve COCO per-annotation ontology attributes (a {name: value} object).
        coco_attrs = ann.get("attributes")
        ann_attrs = coco_attrs if isinstance(coco_attrs, dict) and coco_attrs else None

        built: list[Annotation] = []
        # Keypoints INHERIT the annotation's attributes point by point (so the per-joint
        # `occluded` from `v == 1` can layer on top); every other geometry takes them
        # wholesale below. Without this flag the wholesale copy would overwrite - and lose
        # - the visibility the reader just recovered.
        attributes_inherited = False
        kps = ann.get("keypoints")
        seg = ann.get("segmentation")
        bbox = ann.get("bbox")
        if isinstance(kps, list) and any(kps[i + 2] > 0 for i in range(0, len(kps) - 2, 3)):
            next_id = instance_counter.get(filename, 0) + 1
            built = _keypoint_annotations(
                kps,
                category_keypoints.get(int(cat_id), []) if cat_id is not None else [],
                name,
                confidence,
                base_id,
                next_id,
                ann_attrs,
            )
            attributes_inherited = True
            # A lone landmark stays UNASSOCIATED, so it must not consume an id - otherwise
            # the first real object on the image would come back numbered 2.
            if any(getattr(a, "instance_id", None) == next_id for a in built):
                instance_counter[filename] = next_id
        elif isinstance(seg, list) and seg:
            built = _polygon_annotations(seg, name, confidence, base_id)
        if not built and isinstance(bbox, list) and len(bbox) == 4:
            x, y, w, h = (float(v) for v in bbox)
            if w > 0 and h > 0:
                kwargs: dict[str, Any] = {
                    "name": name,
                    "bounding_box": BoundingBox(x=x, y=y, w=w, h=h),
                    "confidence": confidence,
                }
                if base_id is not None:
                    kwargs["id"] = base_id
                built = [BBoxAnnotation(**kwargs)]
        if built:
            if ann_attrs is not None and not attributes_inherited:
                built = [a.model_copy(update={"attributes": ann_attrs}) for a in built]
            result.annotations.setdefault(filename, []).extend(built)

    return result


@dataclass(frozen=True)
class _Emission:
    """One planned COCO ``annotations`` entry, before category ids are known.

    Either a plain geometry (``annotation``) or ONE keypoint instance (``points`` +
    the joint ordering its triplets align to).
    """

    category: str
    annotation: Annotation | None = None
    points: tuple[KeypointAnnotation, ...] = ()
    template: tuple[str, ...] = ()


def _plan_image(
    anns: Sequence[Annotation],
    templates: Mapping[str, Mapping[str, Any]],
) -> list[_Emission]:
    """Decide what one image emits - **one COCO annotation per keypoint INSTANCE**.

    An instance is matched to the class template that covers most of its joint names;
    that class becomes the COCO category, and its ``categories[].keypoints`` array is
    what the triplets align to.

    Points that cannot take a template slot (an off-template name, or a duplicate of a
    joint already slotted) fall back to their own arity-1 category. So does EVERY point
    of an instance no template covers - including all of them when no templates are
    supplied at all, which is the pre-instance behaviour, preserved exactly. Grouping
    genuinely needs the template: the object class is not written on the annotation, and
    inventing a category name that exists nowhere in the user's project would be worse
    than leaving the joints loose.

    Nothing is ever silently dropped except a polyline, which has no COCO form at all.
    """
    out: list[_Emission] = []
    for ann in anns:
        if ann.type in ("bbox", "polygon"):
            out.append(_Emission(category=ann.name, annotation=ann))
        # polyline: no COCO form; keypoints are handled per-instance below.

    for instance in group_instances(anns):
        class_name = match_template(instance.points, templates) if templates else None
        loose: Sequence[KeypointAnnotation] = instance.points
        if class_name is not None:
            joints = template_node_names(templates[class_name])
            slots, loose = slot_instance(instance.points, joints)
            placed = tuple(s for s in slots if s is not None)
            if placed:
                out.append(_Emission(category=class_name, points=placed, template=tuple(joints)))
        out.extend(_Emission(category=p.name, points=(p,)) for p in loose)
    return out


def to_coco(
    annotations_by_image: Mapping[str, Sequence[Annotation]],
    *,
    image_sizes: Mapping[str, tuple[int, int]] | None = None,
    keypoint_templates: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Serialize Pictograph annotations (grouped by image file name) to a COCO dict.

    Args:
        annotations_by_image: ``file_name`` → the annotations on that image.
        image_sizes: Optional ``file_name`` → ``(width, height)``. COCO ``images``
            entries carry ``width``/``height``; when a size is unknown it is
            written as ``0`` (tolerated by most tools, but supply sizes for full
            fidelity).
        keypoint_templates: Optional per-class keypoint templates (class name →
            ``{"nodes": [{"name": ...}], "edges": [[i, j]]}``, the shape
            ``project_config`` stores). Keypoints sharing an ``instance_id`` are emitted
            as ONE COCO annotation under the matching class, with its joint names on
            ``categories[].keypoints`` and its edges - **+1, since COCO's are
            1-indexed** - on ``categories[].skeleton``. Without it every point is
            emitted on its own; COCO cannot name an object class that appears nowhere
            in the annotations.

    Returns:
        A COCO dict with ``images`` / ``annotations`` / ``categories``. Boxes map
        to ``bbox``; polygons map to a polygon ``segmentation`` (all rings, union
        semantics - holes are not representable); each keypoint instance maps to one
        entry with ``[x, y, v]`` triplets, a real ``num_keypoints``, and a DERIVED
        ``bbox``. Polylines have no COCO equivalent and are skipped.
    """
    sizes = image_sizes or {}
    templates = keypoint_templates or {}

    # Pass 1: plan every image. The category set is not knowable up front any more - a
    # matched instance is filed under its TEMPLATE class, a name that need not appear on
    # any annotation, while the joint classes it absorbed must not become empty categories.
    plans = [
        (filename, _plan_image(anns, templates)) for filename, anns in annotations_by_image.items()
    ]

    # Deterministic 1-indexed category ids over the sorted class-name set.
    names = sorted({emission.category for _, emissions in plans for emission in emissions})
    cat_id_by_name = {name: i + 1 for i, name in enumerate(names)}

    images_out: list[dict[str, Any]] = []
    annotations_out: list[dict[str, Any]] = []
    ann_id = 1
    for image_id, (filename, emissions) in enumerate(plans, start=1):
        w, h = sizes.get(filename, (0, 0))
        images_out.append(
            {"id": image_id, "file_name": filename, "width": int(w), "height": int(h)}
        )
        for emission in emissions:
            entry = _coco_annotation(emission, ann_id, image_id, cat_id_by_name[emission.category])
            if entry is not None:
                annotations_out.append(entry)
                ann_id += 1

    # A pose class needs its joint NAMES and its connectivity graph on the CATEGORY - that
    # name array is what every `[x, y, v]` triplet is positionally aligned to. Emit the
    # triplets without it and a reader has a bag of numbers it cannot interpret.
    categories_out: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        cat: dict[str, Any] = {"id": i + 1, "name": name}
        joints = template_node_names(templates.get(name))
        if joints:
            cat["keypoints"] = joints
            # COCO's `skeleton` edges are 1-INDEXED into `keypoints`; ours are 0-indexed.
            # The +1 happens here and only here (and `from_coco` is its exact inverse).
            cat["skeleton"] = [[i0 + 1, j0 + 1] for i0, j0 in template_edge_pairs(templates[name])]
        categories_out.append(cat)

    return {
        "images": images_out,
        "annotations": annotations_out,
        "categories": categories_out,
    }


def _coco_annotation(
    emission: _Emission, ann_id: int, image_id: int, category_id: int
) -> dict[str, Any] | None:
    """Build one COCO ``annotations`` entry, or ``None`` when it has no valid COCO form."""
    source: Annotation | None = emission.annotation or (
        emission.points[0] if emission.points else None
    )
    if source is None:  # pragma: no cover - the planner never emits an empty emission
        return None

    entry: dict[str, Any] = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": 0,
    }
    # Per-annotation ontology attributes ({name: value}) → COCO `attributes` object
    # (CVAT/Datumaro-COCO convention). Only a non-empty dict; the legacy list form
    # (or absence) is not emitted, so existing exports are byte-unchanged. An instance
    # takes them from its first point - a COCO annotation is one object, so it carries
    # one attribute set - MINUS `occluded`, which the triplets already express per joint
    # (see `annotation_attributes`). Re-emitting it here would say the same thing twice
    # and hand the first joint's occlusion to every joint on the way back in.
    if isinstance(source.attributes, dict) and source.attributes:
        exported = annotation_attributes(source) if emission.points else dict(source.attributes)
        if exported:
            entry["attributes"] = exported

    if emission.points:
        entry["keypoints"] = coco_keypoints(emission.points, emission.template)
        entry["num_keypoints"] = num_labeled(entry["keypoints"])
        # The box is DERIVED and is NOT optional (see `instance_bbox`). It used to be
        # emitted as `[x, y, 0, 0]` for a lone point, which most COCO readers drop or
        # NaN out, and which a detection transformer cannot match a query against.
        derived = instance_bbox(emission.points)
        entry["bbox"] = [derived.x, derived.y, derived.w, derived.h]
        entry["area"] = derived.w * derived.h
        entry["segmentation"] = []  # a set of joints is not an area
        return entry

    ann = emission.annotation
    assert ann is not None  # noqa: S101 - narrowed by the `emission.points` branch above
    box = annotation_bbox(ann)
    if box is not None:
        entry["bbox"] = [box.x, box.y, box.w, box.h]
        entry["area"] = box.w * box.h

    if ann.type == "polygon":
        entry["segmentation"] = [flatten_ring(ring) for ring in ann.polygon.paths]
        entry["area"] = polygon_area(ann.polygon.paths[0])

    if "bbox" not in entry:
        # A geometry with no derivable positive-area box (shouldn't happen for
        # bbox/polygon) - drop rather than emit an invalid COCO entry.
        return None
    return entry
