# Annotations - the wire format

One JSON object per annotation. Field names are snake_case and there is no
shorthand: boxes are always `{x, y, w, h}` objects, polygons are always multi-ring
`paths`, coordinates are always absolute pixels.

The class-label field is **`name`**. Not `class`, not `label`.

The SDK's Pydantic models in `pictograph.models.annotation` are the source of truth
and forbid unknown fields - an extra key **raises** rather than being dropped. When
a payload is rejected, dump the model with
`.model_dump(mode="json", exclude_none=True)` and diff it against the message.

## Common fields

| Field | Required | Notes |
|---|---|---|
| `id` | yes | Unique within the image. UUIDs preferred. |
| `name` | yes | Class label. Must match a class on the dataset (case-sensitive). |
| `type` | yes | `bbox` / `polygon` / `polyline` / `keypoint`. Selects the geometry field. |
| `confidence` | no | `[0.0, 1.0]`, default `1.0`. Predictions carry a real score; hand-drawn annotations are 1.0. |
| `created_by` | no | Creator UUID. The backend fills this for SDK writes. |
| `attributes` | no | `{name: value}` map of ontology attributes, e.g. `{"occluded": "true"}`. Exported natively to COCO and Datumaro. |

## The four types

### `bbox` - axis-aligned box

```json
{"id": "ann-1", "name": "person", "type": "bbox",
 "bounding_box": {"x": 100, "y": 200, "w": 50, "h": 80}}
```

`x, y` is the top-left corner. Use this for anything a detector will train on -
`yolox`, `rfdetr_detection`.

**Rotated boxes** are a `bbox` carrying an extra `oriented_box`, not a separate
type:

```json
{"id": "ann-2", "name": "pallet", "type": "bbox",
 "bounding_box": {"x": 90, "y": 40, "w": 140, "h": 120},
 "oriented_box": {"cx": 160, "cy": 100, "w": 120, "h": 60, "angle": 30}}
```

`w`/`h` are measured along the box's **own** axes, so they do not change as it
rotates. `angle` is degrees, clockwise-positive in image space (y points down),
normalized to `[0, 360)`. `bounding_box` stays the axis-aligned enclosure, so a
box-only consumer keeps working. Exports as `yolo_obb` / `dota`.

### `polygon` - outline, with holes

```json
{"id": "ann-3", "name": "lake", "type": "polygon",
 "polygon": {"paths": [
   [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 100, "y": 100}, {"x": 0, "y": 100}],
   [{"x": 30, "y": 30}, {"x": 70, "y": 30}, {"x": 70, "y": 70}, {"x": 30, "y": 70}]
 ]}}
```

The first ring is the outer boundary; every later ring carves a hole (even-odd
fill). Each ring needs at least 3 points. Two disconnected regions of the same
object are **two annotations**, not two outer rings on one.

`bounding_box` is optional here - the backend computes the enclosing rectangle if
omitted.

Use for segmentation: `rfdetr_segmentation` (instance), `sm_pytorch` (semantic).

### `polyline` - open path

```json
{"id": "ann-4", "name": "lane_centerline", "type": "polyline",
 "polyline": {"path": [{"x": 0, "y": 100}, {"x": 50, "y": 100}, {"x": 100, "y": 100}]}}
```

At least 2 points; the ends are not joined. For lane markings, wires, road
centrelines. No training pipeline consumes polylines - they are for labelling and
export, not supervision.

### `keypoint` - one point

```json
{"id": "ann-5", "name": "left_eye", "type": "keypoint",
 "keypoint": {"x": 250, "y": 180}, "instance_id": 1}
```

A keypoint carries **no `bounding_box`** - a point has no extent, and sending one
raises. A consumer needing a box must derive it.

## Multi-joint pose = `keypoint` + `instance_id`

There is no `skeleton` annotation type, and none should be invented.

- **A joint is a class; `instance_id` is the object.** Each point's `name` is its
  joint class (`nose`, `left_eye`, …). Points sharing an `instance_id` are joints
  of one object.
- `instance_id` is a positive int, 1-based, scoped to the image. Three people at
  17 joints each is 51 annotations carrying ids 1, 2 and 3.
- `null` (or absent) means **unassociated** - a lone landmark, not "all one
  group". Do not fuse those.
- A keypoint class with no template is a standalone landmark class of arity 1.
  "Keypoint-as-class" is as common as pose and is the same task.
- Connectivity lives **once per class** on the dataset's class config as
  `skeleton: {nodes: [{name, x, y}], edges: [[i, j]]}`, with 0-indexed edges and
  node names in the canonical joint order. It is never on the annotation, because
  it would be an identical template repeated per instance.
- `rfdetr_keypoint` trains on exactly this grouping. It is query-based and
  top-down, so the grouping **is** the supervision signal - points with no
  `instance_id` cannot teach multi-instance pose.

Reading predictions back, `KeypointResult` gives you both views without writing
the grouping loop yourself:

```python
for point in result.points:  # flat, every joint
    print(point.name, point.keypoint.x, point.keypoint.y)

for obj in result.instances:  # one list per detected object
    print("object with", len(obj), "joints")
```

### Occlusion

Visibility is COCO's, verbatim: `0` = not labelled, `1` = labelled but occluded,
`2` = labelled and visible. A point cannot carry that in its geometry, so it rides
on `attributes`:

```json
{"id": "ann-6", "name": "left_wrist", "type": "keypoint",
 "keypoint": {"x": 310, "y": 402}, "instance_id": 1,
 "attributes": {"occluded": "true"}}
```

Set it and the joint exports as `v = 1`; omit it and a placed joint is `v = 2`. It
round-trips - a COCO import with `v = 1` comes back as `attributes.occluded`, so an
occluded joint is never quietly promoted to plainly visible. An occluded joint is
still labelled and counts toward `num_keypoints`.

## Storage shape

`client.annotations.get(dataset_name, image)` returns a **plain list** - no wrapper object.
An image is addressed by the pair you can read off the grid: its dataset and its
filename. An image id is accepted in place of the filename:

```json
[
  {"id": "ann-1", "name": "person", "type": "bbox", "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4}},
  {"id": "ann-2", "name": "car", "type": "polygon", "polygon": {"paths": [[{"x": 0, "y": 0}]]}}
]
```

## Reading and writing

```python
from pictograph import Client, BBoxAnnotation, BoundingBox, PolygonAnnotation
from pictograph.models.annotation import PolygonGeometry
from pictograph.models.common import Point

client = Client()

existing = client.annotations.get(
    dataset_name="my-photos",
    image="street-0421.jpg",
)  # list[Annotation]

client.annotations.save(
    dataset_name="my-photos",
    image="street-0421.jpg",
    annotations=[
        BBoxAnnotation(
            id="ann-1", name="person", bounding_box=BoundingBox(x=100, y=200, w=50, h=80)
        ),
        PolygonAnnotation(
            id="ann-2",
            name="car",
            polygon=PolygonGeometry(paths=[[Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=10)]]),
        ),
    ],
)
```

`save` replaces the image's annotations. For many images use
`client.annotations.bulk_save({image_id: [...], ...})` - one round trip instead of N.
This one is keyed by image ID rather than filename: the ids come straight from the
`client.images.list(...)` you just made, and turning N filenames back into N ids is
the round-trip a bulk call exists to avoid.

Class-level edits are their own calls, not a rewrite of every annotation:
`client.annotations.rename_class`, `merge_class`, `delete_class`.

## Converting to and from standard formats

`pictograph.formats` converts in memory, without touching the API:

```python
from pictograph.formats import to_coco, from_coco, to_yolo, from_yolo

coco = to_coco({"img1.jpg": annotations}, image_sizes={"img1.jpg": (1920, 1080)})
imported = from_coco("./instances.json")

text = to_yolo(annotations, ["person", "car"], image_width=1920, image_height=1080)
back = from_yolo(text, ["person", "car"], image_width=1920, image_height=1080)
```

`to_pascal_voc` / `from_pascal_voc` are the same shape. To push a whole annotation
file onto an uploaded dataset, use the importers instead -
`client.annotations.import_coco` and friends, see
`references/bulk-operations.md`.

For a full dataset export as a downloadable ZIP (12 formats, optionally with
images), use `client.exports.create(...)`.

## Drawing annotations

```python
from pictograph import draw_annotations

image = draw_annotations("photo.jpg", result.predictions, show_confidence=True)
image.save("annotated.png")
```

Returns a PIL image. Pass `keypoint_templates=` to connect pose joints.

## Mistakes that get rejected

- `"class": "person"` - the field is `"name"`.
- `"polygon": [[10, 20, 30, 40]]` - flat array. Must be `[{"x": ..., "y": ...}]`, wrapped in `paths`.
- `"bbox": [x, y, w, h]` - array. Must be `"bounding_box": {x, y, w, h}`.
- A `bounding_box` on a `keypoint` - raises, does not get ignored.
- A polygon ring with fewer than 3 points, or a polyline with fewer than 2.
- A `name` that is not a class on the dataset.
