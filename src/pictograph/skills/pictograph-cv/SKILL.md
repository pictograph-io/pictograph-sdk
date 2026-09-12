---
name: pictograph-cv
description: Use when working with computer vision on Pictograph (https://pictograph.io) - uploading and labelling image datasets, auto-annotating with SAM3 or a trained model, training detection / segmentation / keypoint / classification models, running a trained model locally or against a deployed endpoint, and exporting to COCO, YOLO, Pascal VOC, CVAT and other formats.
allowed-tools: Bash, Read, Edit, Write
---

# Pictograph CV

Pictograph is a computer-vision dataset and model platform. This skill covers the
Python SDK (`pictograph`) and its CLI (`pictograph`), which expose the same
capabilities: get images in, label them, train a model, run it, get data out.

Use this skill when the user mentions a directory of images to label, auto-annotation,
training or fine-tuning a CV model, running a trained model on an image, importing
from V7 / Roboflow / a COCO or YOLO file, or exporting a dataset.

Do not use it for classical CV that never touches Pictograph, or for bounding-box
maths you can do yourself.

## Setup

```bash
pip install pictograph                  # SDK + CLI
pip install "pictograph[inference]"     # + run trained models locally
export PICTOGRAPH_API_KEY=pk_live_...   # app.pictograph.io → Settings → API Keys
```

`Client()` reads `PICTOGRAPH_API_KEY`, then `~/.pictograph/config.toml` (written by
`pictograph login`). Pass `Client(api_key=...)` to override.

## Pick the entry point

| You want to… | Use |
|---|---|
| One shell command | the `pictograph` CLI, or `scripts/*.py` for the chains it lacks |
| A multi-step chain in Python | `client.<resource>.<method>` - the bulk ones live on the resource that owns the noun |
| One specific API call | `client.<resource>.<method>` |
| Run a trained model | `pictograph.get_model` / `load_model` / `DeploymentClient` |

`pictograph --help` lists every command group; most read commands take `--json`.
The `scripts/` wrappers always print JSON. The SDK resource modules are the
canonical contract - read them when a signature matters.

## How things are addressed: by NAME

Every user-facing method takes the name a user actually knows. A uuid is a
database detail, and needing one means an extra call just to get a value you
cannot read back later.

| thing | you pass | example |
|---|---|---|
| dataset, model, workflow, deployment | its **name** | `"road-signs"` |
| image | its **dataset + filename** | `("road-signs", "street-0421.jpg")` |
| directory | its **dataset + path** | `("road-signs", "/train")` |
| export | its **dataset + export name** | `("road-signs", "for-training")` |
| webhook endpoint | its **URL** (it has no name) | `"https://hooks.example.com/pg"` |

```python
client.images.get(
    dataset_name="road-signs",
    image="street-0421.jpg",
)
client.deployments.pause(
    deployment="prod-detector",
)
```

An **id is still accepted anywhere a name is**, detected by shape, so a caller
holding one from a previous response pays no extra lookup:

```python
for image in client.images.list(dataset_name="road-signs"):
    client.annotations.get(
        dataset_name="road-signs",
        image=image.id,
    )  # no lookup
```

Two consequences worth knowing:

- **A filename is unique per (dataset, DIRECTORY), not per dataset.** The same name
  in two directories raises and names them rather than picking one - pass
  `directory_path=` to disambiguate.
- **Ephemeral handles stay ids**, because nobody names them: run ids, job ids,
  version ids, notification ids.

## Recipes

### 1. Upload a directory and label it

```bash
python scripts/upload_and_annotate.py \
  --directory ./road_signs --dataset road-signs \
  --classes "stop_sign:bbox,yield:bbox,speed_limit:bbox"
```

Equivalent Python:

```python
from pictograph import Client

client = Client()
uploaded = client.images.upload_from_directory(
    "road-signs",
    directory="./road_signs",
)
labelled = client.auto_annotate.dataset(
    "road-signs",
    classes=[("stop_sign", "bbox"), ("yield", "bbox")],
)
print(uploaded.images_uploaded, labelled.annotations_added)
```

`images.upload_from_directory` creates the dataset if missing, walks subdirectories,
and skips filenames already present. `auto_annotate.dataset` runs a batch
SAM3 job over the dataset. Both return a report with a `.failures` list - inspect
it rather than assuming a clean run. See `references/auto-annotation.md`.

### 2. Train a model

```bash
python scripts/train.py --dataset road-signs --pipeline yolox --epochs 50
```

```python
# Training runs off an EXPORT, never off a dataset. Create the export first,
# check it, then train it.
client.exports.create(
    dataset_name="road-signs",
    name="road-signs-2026-07",
    format="pictograph",
    include_images=True,
    wait=True,
)
run = client.training.create(
    dataset_name="road-signs",
    export_name="road-signs-2026-07",
    pipeline_type="yolox",
    name="road-signs-detector",
    gpu_type="a10g",
    config={"epochs": 50},
)
print(run.status)
```

Choose the pipeline by what the user wants to predict, not by name recognition -
`references/training.md` has the mapping, the config keys, and how versions and
precision work.

### 3. Run a trained model

Locally, on your own hardware:

```python
from pictograph import get_model

model = get_model("Road Sign Detector", task="object_detection")
result = model.predict(
    "photo.jpg",
)  # path, URL, bytes, PIL, or BGR array

for p in result.predictions:
    print(p.name, round(p.confidence, 2), p.bounding_box)

model.close()
```

Against a deployed endpoint, with no local runtime:

```python
from pictograph import DeploymentClient

with DeploymentClient(endpoint_url, "pk_deploy_...", task="object_detection") as dc:
    result = dc.infer("photo.jpg")
    for p in result.predictions:
        print(p.name, round(p.confidence, 2), p.bounding_box)
```

Both return the **same five typed result classes**, so downstream code does not
branch on where inference ran. `task=` narrows the return type and is verified
against the model - it cannot silently lie. Full detail, including the four local
runtimes and their artifacts, is in `references/inference.md`.

```bash
pictograph models predict "Road Sign Detector" photo.jpg --json   # local
pictograph models predict "Road Sign Detector" photo.jpg --remote # server-side
```

### 4. Export a dataset

```bash
python scripts/export.py --dataset road-signs --name for-training \
  --format coco --output ./road-signs.zip
```

```python
export = client.exports.create(
    "road-signs",
    name="for-training",
    format="coco",
    include_images=True,
    status_filter="complete",
    wait=True,
)
client.exports.download(
    "road-signs",
    export_name="for-training",
    output_path="./road-signs.zip",
)
```

`status_filter="complete"` restricts the export to images marked finished.
`organize_by_split=True` writes train/val/test subdirectories.

### 5. Bring an existing dataset in

From a connected provider:

```bash
python scripts/import_connector.py --provider v7 --api-key $V7_KEY --list
python scripts/import_connector.py --provider v7 --api-key $V7_KEY --dataset-ids ds_abc
```

From annotation files you already have, onto a dataset whose images are uploaded:

```python
report = client.annotations.import_coco(
    "road-signs",
    coco="./instances.json",
)
print(report.annotations_saved, len(report.failures))
```

`client.annotations.import_yolo` and `.import_pascal_voc` are the same shape.
Matching is by image file name.

### 6. Grow or reshape a dataset

```python
client.images.tile(
    "road-signs",
    rows=2,
    cols=2,
    into="road-signs-tiled",
)
client.images.augment(
    "road-signs",
    ops=ops,
    multiplier=3,
    into="road-signs-aug",
)
```

Tiling slices each image into a grid and re-cuts the annotations - the standard fix
for small objects in large images. Augmentation writes variants into a separate
dataset so the original stays clean. See `references/bulk-operations.md`.

## Annotation format

Four geometry types. The class-label field is **`name`**, never `class`.

```json
{"id": "ann-1", "name": "person", "type": "bbox",
 "bounding_box": {"x": 100, "y": 200, "w": 50, "h": 80}}
```

| `type` | Geometry field |
|---|---|
| `bbox` | `bounding_box: {x, y, w, h}` - plus optional `oriented_box` for a rotated box |
| `polygon` | `polygon: {paths: [[{x, y}, ...], ...]}` - first ring outer, rest are holes |
| `polyline` | `polyline: {path: [{x, y}, ...]}` |
| `keypoint` | `keypoint: {x, y}` - plus optional `instance_id` |

**There is no `skeleton` type.** A multi-joint pose is several `keypoint`
annotations that share an `instance_id`; each one's `name` is the joint's class.
Three people at 17 joints is 51 annotations with `instance_id` 1, 2 and 3. The
connectivity that draws the pose lives once per class on the dataset's class
config, not on the annotation.

The SDK's Pydantic models reject unknown fields, so a stray key raises rather than
being ignored. Build annotations with the models and let them validate:

```python
from pictograph import BBoxAnnotation, BoundingBox

client.annotations.save(
    dataset_name="my-photos",
    image="street-0421.jpg",
    annotations=[
        BBoxAnnotation(
            id="ann-1", name="person", bounding_box=BoundingBox(x=100, y=200, w=50, h=80)
        ),
    ],
)
```

Full field list, attributes, occlusion and the round-trip rules:
`references/annotations.md`.

## Before you spend

Auto-annotation, training, deployments and image generation consume the
organization's compute credit, which is **USD-denominated** and reported in
micro-USD (µUSD, 1 USD = 1,000,000 µUSD). Uploading, exporting and reading do not
consume it.

Price a batch labelling job before running it:

```python
quote = client.auto_annotate.quote(
    dataset_name="road-signs",
    image_filenames=[...],
    classes=[{"name": "stop_sign", "output_type": "bbox"}],
)
print(quote.total_images, quote.estimated_credits, quote.sufficient)
```

For anything else, estimate by operation, or read the balance:

```python
est = client.credits.estimate(
    "training_a10g",
    quantity=30,
)
print(est.total_micro_usd, est.remaining_micro_usd, est.sufficient)

balance = client.credits.balance()
print(balance.included_remaining_micro_usd, balance.budget_remaining_micro_usd)
```

`client.credits.usage_by_operation()` lists the operation slugs the org has
actually spent on - use it to discover a slug rather than guessing one.

`sufficient=True` is a snapshot, not a reservation. The authoritative answer is the
operation itself raising `PaymentRequiredError`, which carries `credit_cost`,
`credits_remaining`, `unit` and `upgrade_url`.

## When calls fail

Every error subclasses `pictograph.exceptions.PictographError`.

| Exception | Usual cause |
|---|---|
| `NotFoundError` | Dataset / model / image name wrong. Names are case-sensitive. |
| `ConflictError` | Filename already exists in that directory. Uploads skip by default. |
| `ValidationError` | Payload shape wrong - `class` instead of `name`, a flat polygon array, a `bounding_box` on a keypoint. |
| `PaymentRequiredError` | Out of compute credit. Read `credit_cost` / `upgrade_url`. |
| `RateLimitError` | Retried automatically when `Retry-After` is short; otherwise back off. |
| `PollTimeoutError` | A long job outlived the wait. Re-poll: `client.training.get(run_id=...)`. |

Long-running calls (`exports.create`, `training.create`, `auto_annotate.batch`,
`connectors.import_`) accept `wait=False` and return a handle you poll later. Use
that for anything an agent should not block on.

## References

| File | Covers |
|---|---|
| `references/annotations.md` | The annotation wire format, every type, attributes, keypoint instances |
| `references/auto-annotation.md` | SAM3 prompts, batch labelling, model-assisted labelling, SAHI |
| `references/training.md` | Pipelines, hyperparameters, GPU selection, versions, precision, artifacts |
| `references/inference.md` | Local runtimes, the five result types, deployed endpoints |
| `references/bulk-operations.md` | Every whole-dataset call and the report it returns |

## Scripts

Bash-callable wrappers that print JSON to stdout. Run any of them with `--help`.
They read `PICTOGRAPH_API_KEY` from the environment.

| Script | Does |
|---|---|
| `scripts/upload_and_annotate.py` | Upload a directory, optionally auto-annotate it. |
| `scripts/train.py` | Export a dataset and train a model from it. |
| `scripts/export.py` | Create and download an export in any format. |
| `scripts/import_connector.py` | List and import datasets from V7 or Roboflow. |
