# Bulk operations - the calls that act on a whole dataset

These are ordinary resource methods, on the resource that owns the noun they act
on. Each returns a **report dataclass**, so a partial failure is visible instead
of swallowed.

Every report has `.success` and a `.failures` list. Check both. `.failures` names
the exact items to retry - never assume a clean run because no exception was
raised.

> There is no `pictograph.pipelines` module. "Pipeline" in this SDK means a
> *training* pipeline (`yolox`, `rfdetr_detection`, …), and "workflow" means the
> node-graph resource `client.workflows`. Nothing else.

## Getting images in

```python
report = client.images.upload_from_directory(
    "road-signs",
    source_path="./photos",
    organize_by_class=True,
    preserve_structure=False,
    parallel=True,
    max_workers=8,
    skip_existing=True,
    create_if_missing=True,
    progress=lambda done, total, name: None,
)
print(report.images_uploaded, report.images_skipped, report.failures)
```

Report: `dataset_name`, `images_attempted`, `images_uploaded`, `images_skipped`,
`failures`.

### Starting from a public dataset

Datasets other organizations published to Explore can be copied into yours. A public
dataset is named `org/dataset` - the `ref` of a row from `client.explore`:

```python
for ds in client.explore.datasets(
    search="food",
    sort="popular",
):
    print(ds.ref, ds.image_count, ds.license_id, ds.class_count)

job = client.datasets.import_public(
    "pictograph-research/foodseg103",
    name="food",
)
print(job.status, job.dataset_name, job.image_count)
```

The copy is private, keeps the source's classes, license and attribution, and stays
yours if the source is unpublished. Its images are referenced rather than copied, so
even a large dataset is ready in minutes. `name` defaults to the source's; a name you
already use gets `(imported)` - read `dataset_name`. A copy past the plan's image limit
is refused (`PaymentRequiredError`, `image_limit_exceeded`) before anything is made.
`wait=False` returns at once; follow it with `datasets.wait_for_public_import(job.import_id)`.
CLI: `pictograph explore datasets|dataset|models`, `pictograph datasets import-public`.
A public MODEL is copied with `client.models.fork("org", "model")`.

## Labelling

```python
report = client.auto_annotate.dataset(
    "road-signs",
    classes=[("stop_sign", "bbox"), ("yield", "bbox")],
    mode="batch",
    confidence_threshold=0.5,
    overwrite=False,
    max_images=None,
    timeout=1800.0,
)
```

The class list accepts tuples, `{"name": ..., "output_type": ...}` dicts, or
`BatchClass` instances. Report: `images_processed`, `images_skipped`,
`images_capped`, `annotations_added`, `job_id`, `failures`. Details in
`references/auto-annotation.md`.

## Training

```python
run = client.training.create(
    dataset="road-signs",
    export_name="road-signs-2026-07",
    pipeline_type="yolox",
    name="road-signs-detector",
    gpu_type="a10g",
)
```

The one call here that returns a `TrainingRun` rather than a report. It trains
from an export you already created - it never makes one for you - and waits for a
terminal status by default; `wait=False` hands back the pending run to poll with
`client.training.get(run_id=...)`. The produced model is
`client.models.get(model_id=run.model_id)`. See `references/training.md`.

## Running them in sequence

The three above compose - but you write the composition, so you can see and
control each stage:

```python
uploaded = client.images.upload_from_directory(
    "road-signs",
    source_path="./photos",
)
if not uploaded.success:
    raise SystemExit(uploaded.failures)

# Paid from here. Pre-flight the wallet rather than discovering it mid-run.
balance = client.credits.balance()
spendable = balance.included_remaining_micro_usd + balance.budget_remaining_micro_usd
if spendable < 1_000_000:  # µUSD - 1_000_000 == $1.00
    raise SystemExit(f"only ${spendable / 1_000_000:,.2f} of compute credit")

labelled = client.auto_annotate.dataset(
    "road-signs",
    classes=[{"name": "stop_sign", "output_type": "bbox"}],
)
# → review the annotations before they become training data

client.exports.create(
    dataset="road-signs",
    name="road-signs-2026-07",
    format="pictograph",
    include_images=True,
    wait=True,
)
run = client.training.create(
    dataset="road-signs",
    export_name="road-signs-2026-07",
    pipeline_type="yolox",
    name="road-signs-detector",
    gpu_type="a10g",
    config={"epochs": 50},
)
```

There is deliberately **no single call that does all three**. Auto-annotation
output should be reviewed before it becomes training data, and training takes
explicit config (pipeline, GPU, class order, hyperparameters) that a blind chain
cannot choose for you. Check each report's `.success` and `.failures` before
running the next step, and do not write a helper that hides these three behind
one function.

## Importing annotations you already have

For a dataset whose images are already uploaded. A file lands on the image with its exact
name, else the one image with its stem (`a.txt` and `a.xml` label `a.jpg`); a name two images
share is reported in `failures` and never guessed (an import replaces the image's annotations).
From a shell: `pictograph annotations import-coco <dataset> instances.json`,
`import-yolo <dataset> ./labels [--classes data.yaml]`, `import-voc <dataset> ./Annotations`.

```python
report = client.annotations.import_coco(
    "road-signs",
    coco="./instances.json",
)

report = client.annotations.import_yolo(
    "road-signs",
    labels={"img-1.jpg": "0 0.5 0.5 0.2 0.3\n"},
    class_names=["stop_sign", "yield"],
)

report = client.annotations.import_pascal_voc(
    "road-signs",
    xml_by_filename={"img-1.jpg": "<annotation>...</annotation>"},
)
```

All three take `create_missing_classes=True` (add classes the file references) and
`save_chunk=200` (images per bulk write). YOLO coordinates are normalized, so the
importer reads each image's pixel size from the dataset to denormalize.

Report: `images_matched`, `images_saved`, `annotations_saved`, `unmatched_files`,
`failures`. **`unmatched_files` is the one to check** - a file whose name matches no
image in the dataset is skipped, not an error.

To import a whole dataset (images included) from V7 or Roboflow, use
`client.connectors` instead - see `scripts/import_connector.py`.

## Reshaping a dataset

### Images into another dataset

```python
client.datasets.create(name="signs-night")
night = client.images.iter(
    dataset="road-signs",
    directory="/night",
)
ids = [img.id for img in night]
for start in range(0, len(ids), 10_000):  # up to 10,000 ids per call
    result = client.batch.copy_to_dataset(
        dataset="road-signs",
        images=ids[start : start + 10_000],
        target_dataset_name="signs-night",
    )
    print(result.processed, result.classes_added, result.failed)
```

The copies share the stored files (nothing is uploaded, and the image quota does not
count them twice) and keep their annotations, tags, split and stage; the classes they
use are added to the target dataset as the source defines them. The source stays as it
was. A name already in the target folder gets `-1`, `-2`, ... (`renamed`). CLI, chunked
for you: `pictograph images copy road-signs --to-dataset signs-night --directory /night`.

### Tiling - small objects in large images

```python
report = client.images.tile(
    "aerial",
    rows=2,
    cols=2,
    overlap=0.1,
    min_visibility=0.1,
    include_empty=True,
    into="aerial-tiled",
    max_source_images=None,
)
```

Slices each image into a grid and re-cuts the annotations to each tile, so objects
occupy more pixels at the model's input size. Report: `tiles_created`,
`empty_tiles`, `annotations_written`, `failures`.

### Augmentation

```python
from pictograph.augment import Brightness, HorizontalFlip, Rotate

report = client.images.augment(
    "road-signs",
    ops=[HorizontalFlip(p=0.5), Rotate(degrees=(-15, 15)), Brightness(factor=(0.8, 1.2))],
    multiplier=3,
    into="road-signs-aug",
    include_original=True,
    seed=42,
    skip_empty=False,
    drop_classes=None,
)
```

Geometry-changing ops transform the annotations with the pixels. Available:
`flip`, `vflip`, `rotate90`, `rotate`, `resize`, `crop`, `brightness`, `contrast`,
`saturation`, `hue_shift`, `grayscale`, `blur`, `noise`, `cutout`, `shear` - as
classes in `pictograph.augment`, or built from dicts with `build_ops([...])`.

Augment into a **separate** dataset and train on that. Keeping the original clean is
what lets you change the recipe later.

Report: `originals_copied`, `variants_created`, `annotations_written`,
`skipped_empty`, `failures`.

## Offline metrics

`pictograph.metrics` scores predictions against ground truth with no API call:

```python
from pictograph.metrics import evaluate_detections

metrics = evaluate_detections(predictions_by_image, truth_by_image, iou_threshold=0.5)
```

For a server-side evaluation against an export, use `client.model_evaluations`.

## Async

Every resource has an async twin under `pictograph.aio`, driven by `AsyncClient`.
Same names, same arguments, awaited. The bulk calls that HAVE an async twin are
the I/O-bound ones - `images.upload_from_directory` and the three
`annotations.import_*` methods. The dataset-wide `auto_annotate.dataset` and the
Pillow-bound `images.augment` / `images.tile` are sync-only, because concurrency
would not shorten them; call them from the sync `Client`.

```python
from pictograph import AsyncClient

async with AsyncClient() as client:
    dataset = await client.datasets.get(
        "road-signs",
    )
    report = await client.images.upload_from_directory(
        "road-signs",
        source_path="./photos",
    )
```
