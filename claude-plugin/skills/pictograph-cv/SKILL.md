---
name: pictograph-cv
description: Use when working with computer vision on Pictograph (https://pictograph.io) - uploading and labelling image datasets, auto-annotating with SAM3 or a trained model, training detection / segmentation / keypoint / classification models, running a trained model locally or against a deployed endpoint, and exporting to COCO, YOLO, Pascal VOC, CVAT and other formats.
allowed-tools: Bash(pictograph *), Read
---

# Pictograph CV

Pictograph is a computer-vision dataset and model platform. This skill covers the
Python SDK (`pictograph`) and its CLI (`pictograph`), which expose the same
capabilities: get images in, label them, train a model, run it, get data out.

Do not use it for classical CV that never touches Pictograph, or for bounding-box
maths you can do yourself.

## Setup

```bash
pip install "pictograph[cli]"           # SDK + the pictograph CLI
pip install "pictograph[inference]"     # + run trained models locally
```

The API key (app.pictograph.io → Settings → API Keys) is `PICTOGRAPH_API_KEY`: the Claude Code
plugin sets it from the key entered when it was enabled; otherwise the user sets it. `Client()`
reads it (or `api_key=`); the CLI also reads `pictograph login`'s profiles, one per org's key:
`pictograph --profile <name> ...`.

## Pick the entry point

| You want to… | Use |
|---|---|
| One shell command | the `pictograph` CLI, or `scripts/*.py` for the chains it lacks |
| Any call or chain in Python | `client.<resource>.<method>` - the bulk ones live on the resource that owns the noun |
| Run a trained model | `pictograph.get_model` / `load_model` / `DeploymentClient` |
| Learn how to call a model, SAM3, a deployment or a workflow | `pictograph tools list`, then `pictograph tools describe <tool>` |
| Segment objects in an image you have (no dataset, no training) | `pictograph sam3 predict photo.jpg --text "red car"` |

`pictograph --help` lists every command group; every command takes `--json`, so add it
everywhere.
The `scripts/` wrappers always print JSON. The SDK resource modules are the
canonical contract - read them when a signature matters.

## How things are addressed: by NAME

Every user-facing method takes the name a user actually knows - never a uuid you
would need an extra call to find.

| thing | you give | example |
|---|---|---|
| dataset, model, workflow, deployment | its **name** | `"road-signs"` |
| image | its **dataset + filename** | `("road-signs", "street-0421.jpg")` |
| directory | its **dataset + path** | `("road-signs", "/train")` |
| export | its **dataset + export name** | `("road-signs", "for-training")` |
| webhook endpoint | its **URL** (it has no name) | `"https://hooks.example.com/pg"` |
| a public dataset or model | **org/name** | `"pictograph-research/coco-5k"` |

```python
client.images.get(
    dataset="road-signs",
    image="street-0421.jpg",
)
client.deployments.pause(
    deployment="prod-detector",
)
```

An **id is still accepted anywhere a name is**, detected by shape, so a caller
holding one from a previous response pays no extra lookup:

```python
for image in client.images.list(dataset="road-signs"):
    client.annotations.get(
        dataset="road-signs",
        image=image.id,
    )  # no lookup
```

Two consequences:

- **A filename is unique per (dataset, DIRECTORY), not per dataset.** The same name
  in two directories raises and names them rather than picking one - give
  `directory_path=` to disambiguate.
- **Ephemeral handles stay ids**, because nobody names them: run ids, job ids,
  version ids, notification ids.

## Recipes

Auto-annotations are **drafts**. Have a person review them before you export and
train (recipe 2), and never approve your own drafts unless the user says so.

### 1. Upload a directory and draft labels

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
    source_path="./road_signs",
)
labelled = client.auto_annotate.dataset(
    "road-signs",
    classes=[("stop_sign", "bbox"), ("yield", "bbox")],
)
print(uploaded.images_uploaded, labelled.annotations_added)
```

`images.upload_from_directory` creates the dataset if missing, walks subdirectories,
and skips filenames already present. `auto_annotate.dataset` runs a batch
SAM3 job over the dataset; what it writes are drafts. Both return a report with a
`.failures` list - inspect it rather than assuming a clean run. See
`references/auto-annotation.md`. From the CLI:
`pictograph auto-annotate dataset road-signs --classes "stop_sign:bbox,yield:bbox"`.

A drafting job adds the classes it finds that the dataset lacks. To set up the classes
yourself (type, colour, attributes, a keypoint class's joint template):

```bash
pictograph classes list road-signs
pictograph classes add road-signs stop_sign --type box
pictograph classes add people person --type keypoint --skeleton person.json
pictograph classes update road-signs stop_sign --attributes attributes.json
```

A class is its name AND its type. Each change is merged onto the list as it is now,
so a class someone adds in the app at the same time is kept.

### 2. Have a person review the drafts

Before you export and train, hand the drafts to the user to review in the app.
Start them on the ones the model was least sure of:

```python
shaky = client.images.list(
    "road-signs",
    min_confidence_lt=0.5,
    limit=500,
)
print([image.filename for image in shaky])
```

```bash
pictograph images list road-signs --min-confidence-lt 0.5
```

Wait for the user to finish the review, then export and train (recipe 3).
`client.images.review` (CLI `pictograph images review`) records a review decision,
but the decision is the person's: never approve your own drafts unless the user
says so.

### 3. Train a model

Training runs off an EXPORT, never off a dataset. Create the export, quote the run, then
start it:

```bash
pictograph exports create road-signs --name road-signs-2026-07 -f pictograph --include-images
pictograph train quote road-signs road-signs-2026-07 --pipeline yolox --gpu auto --config '{"epochs": 50}'
pictograph train start road-signs road-signs-2026-07 --pipeline yolox --gpu auto --config '{"epochs": 50}' --name road-signs-detector
```

`python scripts/train.py --dataset road-signs --pipeline yolox --epochs 50` exports and trains in one step, without the quote.
The Python calls are in `references/training.md`.

Show the user the estimate before a paid run; when `can_start` is false,
`blocking` says why.

Choose the pipeline by what the user wants to predict, not by name recognition -
`references/training.md` has the mapping, the config keys, and how versions and
precision work.

### 4. Run a trained model

**List, describe, then run.** SAM3, every ready model, every active deployment and
every workflow is a tool, and the platform tells you how to call each one:

```bash
pictograph tools list
pictograph tools describe "model/Road Sign Detector" --json
```

The card is generated from what the platform stores, so trust it over memory: its
`modes` say which inputs each way of running the tool takes (file, URL, bytes, video,
dataset), whether it writes anything, what it costs, and the exact CLI, Python and
HTTP call; `parameters` carry the bounds the API enforces; `output` shows one
prediction. `client.tools.describe("sam3")` is the same call in Python.

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

Against a deployment (on a GPU; `pictograph deployments create <model> --wait` returns
once it is `active`):

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
pictograph models predict "Road Sign Detector" photo.jpg --remote # on Pictograph; free, one image per call
pictograph models predict "Road Sign Detector" 'frames/*.jpg' https://example.com/a.jpg  # several: one JSON line each
cat photo.jpg | pictograph models predict "Road Sign Detector" - --remote --json   # stdin
pictograph models predict "Road Sign Detector" --video dashcam.mp4 --sample-fps 2   # a video's frames
pictograph models predict "Road Sign Detector" --dataset road-signs --directory /val --limit 50   # read-only
```

SAM3 runs on any image the same way - no dataset, no training, nothing saved. Give it
ONE prompt: a phrase, a box, or points on one object:

```bash
pictograph sam3 predict photo.jpg --text "price tag" --json
pictograph sam3 predict photo.jpg --box 120,80,200,160 --name car --output bbox
pictograph sam3 predict photo.jpg --point 320,240 --point 300,400,neg
pictograph sam3 predict --video dock.mp4 --sample-fps 1 --text forklift
```

`client.sam3.predict("photo.jpg", text="price tag")` is the Python call. Each image is
charged the SAM3 session fee once (repeat prompts on it are free); coordinates are
pixels of the upright image, `result.width` x `result.height`.

An image is a path, a glob, an `http(s)` URL or `-` (stdin). With more than one (or
`--jsonl`, a video or a dataset) each line is `{"input", "result"}` or
`{"input", "error", "error_code"}` and the exit code is 1 if any failed. A video frame's line adds
`frame` and `t` (seconds) - the frames a workflow samples from the same file; a dataset
image's adds `image_id`. `--dataset` writes nothing: to save predictions as draft labels
use `pictograph auto-annotate dataset`, then review them. Local runs have no platform
limit; `--remote` is one hosted call per image or frame from a shared daily capacity.
`pictograph deployments predict` and `pictograph sam3 predict` read the same inputs.

A workflow reads a whole video on Pictograph (tracking, counting, dwell); write one from
a template (`references/inference.md`). `--file` takes a path, a URL, or `-`:

```bash
pictograph workflows invoke "Door counter" --file entrance.mp4
pictograph workflows invoke "Door counter" --file https://example.com/entrance.mp4
```

### 5. Export a dataset

```bash
pictograph exports create road-signs --name for-training -f coco --include-images --status-filter complete
pictograph exports download road-signs for-training -o ./road-signs.zip
```

`--status-filter complete` keeps only the images marked finished.
`--organize-by-split` writes train/val/test subdirectories.

### 6. Bring an existing dataset in

From a public dataset (you get a private copy):

```bash
pictograph explore datasets --search food    # REF = org/dataset
pictograph datasets import-public pictograph-research/foodseg103 --name food
```

From a connected provider:

```bash
python scripts/import_connector.py --provider v7 --api-key "<V7 API key>" --list
python scripts/import_connector.py --provider v7 --api-key - --dataset-ids ds_abc < v7.key
```

From annotation files you already have, onto a dataset whose images are uploaded:

```python
report = client.annotations.import_coco(
    "road-signs",
    coco="./instances.json",
)
print(report.annotations_saved, len(report.failures))
```

`.import_yolo` and `.import_pascal_voc` work alike (CLI: `annotations import-coco|yolo|voc`).
A file lands on its image by name or stem (a.txt labels a.jpg); a shared name is never guessed.

### 7. Grow or reshape a dataset

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

Bulk CLI changes select images by ids, a file of ids, filters (`--directory`,
`--status`, `--split`) or `--all`; check the count with `--dry-run` first:

```bash
pictograph images move road-signs --to /val --directory /incoming --dry-run
pictograph images move road-signs --to /val --directory /incoming
pictograph images copy road-signs --to /backup --split val --with-annotations
pictograph images copy road-signs --to-dataset signs-night --directory /night
pictograph images set-status road-signs --to annotate --directory /redo
pictograph images archive road-signs --status new --yes     # restore brings them back
```

## Annotation format

Four geometry types. The class-label field is **`name`**, never `class`.

```json
{"id": "ann-1", "name": "person", "type": "bbox",
 "bounding_box": {"x": 100, "y": 200, "w": 50, "h": 80}}
```

| `type` | Geometry field |
|---|---|
| `bbox` | `bounding_box: {x, y, w, h}` - plus optional `oriented_box` for a rotated box |
| `polygon` | `polygon: {paths: [[{x, y}, ...], ...]}` - rings filled even-odd: a ring inside another is a hole, a ring outside the others is another piece of the same object; main outline first |
| `polyline` | `polyline: {path: [{x, y}, ...]}` |
| `keypoint` | `keypoint: {x, y}` - plus optional `instance_id` |

**There is no `skeleton` type.** A multi-joint pose is several `keypoint`
annotations that share an `instance_id`; each one's `name` is the joint's class.
Three people at 17 joints is 51 annotations with `instance_id` 1, 2 and 3. The
connectivity that draws the pose lives once per class on the dataset's class
config, not on the annotation: the class's `skeleton`, `{"nodes": [{"name", "x", "y"},
...], "edges": [[i, j], ...]}` with 0-indexed edges (`pictograph classes add ... --type
keypoint --skeleton person.json`).

The SDK's Pydantic models reject unknown fields, so a stray key raises rather than
being ignored. Build annotations with the models and let them validate:

```python
from pictograph import BBoxAnnotation, BoundingBox

client.annotations.save(
    dataset="my-photos",
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
    dataset="road-signs",
    images=[...],
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

`client.credits.usage_by_operation()` lists the slugs the org has spent on.

`sufficient=True` is a snapshot, not a reservation. The authoritative answer is the
operation itself raising `PaymentRequiredError`, which carries `credit_cost`,
`credits_remaining`, `unit` and `upgrade_url`.

## When calls fail

From the CLI, branch on the exit code, never on the message. Set
`PICTOGRAPH_OUTPUT=json` once and every failure is also one line of JSON on stderr,
`{"error": {"code", "message", "exit_code", "status", "retry_after", "details", "fix", ...}}`
(stdout stays empty):

| Exit | Meaning | Do |
|---|---|---|
| 0 | success | |
| 1 | failed: a job ended `failed` / `cancelled`, inputs of a batch failed, a confirmation was declined | read `message`; add `--yes` to commands that ask |
| 2 | no API key, or the key was refused | stop and ask the user for a key |
| 3 | the role or the plan does not allow it | stop and tell the user |
| 4 | not found | check the name (`pictograph datasets list`); names are case-sensitive |
| 5 | conflicts with the current state (exists, dataset in cold storage, job not finished) | read `code`; fetch it, restore it, or wait |
| 6 | invalid input: a flag, a value, a file, or a request the API refused | fix the arguments (`--help`, `details`) |
| 7 | not enough compute credit | stop and ask the user to top up |
| 8 | rate limited | wait `retry_after` seconds, then retry |
| 9 | server error or network | retry later |
| 10 | a `--wait` ran out; the job keeps running | `pictograph jobs wait <id>`; never start it twice |

In Python, every error subclasses `pictograph.exceptions.PictographError`.

| Exception | Usual cause |
|---|---|
| `NotFoundError` | Dataset / model / image name wrong. Names are case-sensitive. |
| `ConflictError` | Filename already exists in that directory. Uploads skip by default. |
| `ValidationError` | Payload shape wrong - `class` instead of `name`, a flat polygon array, a `bounding_box` on a keypoint. |
| `PaymentRequiredError` | Out of compute credit. Read `credit_cost` / `upgrade_url`. |
| `RateLimitError` | Retried automatically when `Retry-After` is short; otherwise back off. |
| `PollTimeoutError` | A long job outlived the wait. Re-poll: `client.training.get(run_id=...)`. |

Long-running calls (`exports.create`, `training.create`, `auto_annotate.batch`,
`connectors.import_`, `datasets.import_public`) take `wait=False` and return a handle
you poll later.

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
