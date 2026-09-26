# Training

Training reads a **completed export** of a dataset and produces a model you can
download, deploy, or run locally. Pick the pipeline by what the user wants to
predict.

## Pipelines

| `pipeline` | Predicts | Trains on | Use when |
|---|---|---|---|
| `yolox` | boxes | `bbox` | fast detection, tight edge budget |
| `rfdetr_detection` | boxes | `bbox` | higher accuracy than YOLOX, heavier |
| `rfdetr_segmentation` | per-instance masks | `polygon` | you need each object outlined separately |
| `sm_pytorch` | per-pixel class map | `polygon` | regions, not objects - no instance identity |
| `classification` | one label per image | image tags | whole-image labels, no geometry |
| `rfdetr_keypoint` | points grouped per object | `keypoint` + `instance_id` | pose, landmarks |

`rfdetr_keypoint` is query-based and top-down: the `instance_id` grouping **is** the
supervision signal. Keypoints with no `instance_id` cannot teach multi-instance
pose. See `references/annotations.md`.

Each pipeline maps to one inference task, which is what `get_model(task=...)`
expects: `yolox` / `rfdetr_detection` → `object_detection`; `rfdetr_segmentation` →
`instance_segmentation`; `sm_pytorch` → `semantic_segmentation`;
`rfdetr_keypoint` → `keypoint_detection`; `classification` → `classification`.

## End to end

```python
# A training run is always ON AN EXPORT. Restricting classes is a property of
# the export, so it is chosen here rather than at train time.
client.exports.create(
    dataset_name="road-signs",
    name="road-signs-2026-07",
    format="pictograph",
    include_images=True,
    class_filter=["stop_sign", "yield"],
    wait=True,
)

run = client.training.create(
    dataset_name="road-signs",
    export_name="road-signs-2026-07",
    pipeline_type="yolox",
    name="road-signs-detector",
    gpu_type="a10g",
    config={"epochs": 50, "batch_size": 16},
    wait=True,
    timeout=7200.0,
)
print(run.status, run.metrics)
if run.model_id:
    client.models.download(
        model_id=run.model_id,
        output_path="./yolox.onnx",
    )
```

The export is its own explicit step. `client.training.create` never mints one for
you, so an export that came out empty is caught before a GPU is billed. Pass
`status_filter="complete"` on the export when only images marked finished should
be trained on.

Kick off and walk away:

```python
run = client.training.create(
    "road-signs",
    export_name="road-signs-2026-07",
    pipeline_type="yolox",
    name="road-signs-detector",
    wait=False,
)

# later, or in another process
run = client.training.get(
    run_id=run.id,
)
print(run.status, run.progress, run.current_epoch, "/", run.total_epochs)
if run.status == "completed":
    model = client.models.get(
        model_id=run.model_id,
    )
```

`client.training.cancel(run_id=run.id)` stops an in-flight run.

## Starting from an export you already have

```python
run = client.training.create(
    "road-signs",
    export_name="my-export",
    pipeline_type="yolox",
    name="road-signs-v2",
    config={"epochs": 50},
    gpu_type="a10g",
)
```

The export must already be in `completed` status.

## GPU selection

`gpu_type` is `a10g` (default), `a100`, `h100`, or `auto`. `auto` picks the
cheapest tier whose memory fits the configuration's predicted peak, and records the
decision on the run's `config`. Prefer it unless you have a reason not to.

`gpu_count` (1-4, RF-DETR pipelines only) trains with DDP and bills that many
times the per-second rate.

Training is charged on success, for the GPU time actually used. Estimate first
with `client.credits.estimate("training_a10g", quantity=<minutes>)`.

## Configuration

`config` is a dict of pipeline-specific hyperparameters. Common keys:

| Key | Notes |
|---|---|
| `epochs` | Number of passes over the data. |
| `batch_size` | Auto-tuned per GPU; override when memory is tight. |
| `learning_rate` | Override for fine-tuning. |
| `image_size` | Resize target. |
| `class_overrides` | `{source: target}` - train `bus` **as** `truck`, collapsing the model's class set. Stored annotations are never modified; every target must be a selected class. |

Unrecognized keys are ignored by the pipeline.

`config` also accepts a **path** to a JSON file - most usefully a `config.json`
downloaded from an earlier model with `client.models.download_file(...)`. That
closes the reproducibility loop: retrain with the exact settings that produced a
model.

### Class order is the class index

For every pipeline, a class's index in the configured class list is its index in
the trained model. `models.class_mapping` records the effective list. Do not
re-derive indices from a directory listing or an alphabetical sort - they will not
match.

### Precision

Models carry a `precision` of `fp32` (default) or `fp16`. fp16 halves the weights
while keeping fp32 inputs and outputs, so it drops into the same serving path.
Most pipelines support it; `rfdetr_keypoint` is fp32 only.

Precision affects which artifacts exist to download - see `references/inference.md`.

## Versions

Train into an **existing** model instead of creating a new one:

```python
run = client.training.create(
    "road-signs",
    export_name="export-v2",
    pipeline_type="yolox",
    name="road-signs-v2",
    version_of_model_id="<model-uuid>",
)
```

The target must be in your organization and share the task type. The new version
goes live unless the model is pinned.

```python
payload = client.models.versions(
    "Road Sign Detector",
)
for v in payload.versions:
    print(v.version_number, v.version_label, v.status, v.is_current, v.metrics)

client.models.set_current_version(
    "Road Sign Detector",
    version_id="<version-uuid>",
)
```

Pinning a version is how you roll back without retraining. Local inference caches
by model **and** version, so a rollback is picked up rather than served stale.

## Dataset size

There is no fixed image-count floor. Below three annotated images the split falls
back to training on everything; at three or more it guarantees at least one image
each in train and val. The real constraint is having enough annotated images for
the model to learn anything, plus enough credit to start.

Check what will actually train:

```python
ds = client.datasets.get(
    "road-signs",
)
print(ds.image_count, ds.completed_image_count)
```

## When a run fails

```python
run = client.training.get(
    run_id=run.id,
)
print(run.status, run.error_message)
```

| Symptom | Cause |
|---|---|
| `num_samples=0` | No annotated images survived the export - usually a `status_filter` that excluded everything, or annotations whose `name` is not a class on the dataset. |
| out-of-memory | Lower `batch_size` or `image_size`, or move up a GPU tier (or use `gpu_type="auto"`). |
| geometry mismatch | Polygon annotations sent to a box-only pipeline, or keypoints with no `instance_id` to `rfdetr_keypoint`. |
| `PaymentRequiredError` | Below the minimum spendable balance. Read `credit_cost` / `upgrade_url`. |
| `PollTimeoutError` | The wait elapsed; the run is still going. Re-poll with `client.training.get(run_id=...)`. |

## Evaluating a trained model

Score a model against a held-out export:

```python
evaluation = client.model_evaluations.evaluate(
    model.name,
    dataset_name="road-signs",
    export_name=holdout_export.name,
    iou_threshold=0.5,
    confidence_threshold=0.5,
)
```

`evaluate` blocks until the run finishes; `create` returns immediately and you poll
with `get` / `wait_for_completion`. The result carries per-class metrics, a
confusion matrix, confidence buckets and the worst-performing images - the input to
deciding whether to relabel or retrain.

## CLI

`train start` takes the dataset AND the completed export to train on, in that
order - the same two-step the SDK enforces.

```bash
pictograph exports create road-signs --name road-signs-2026-07 --include-images
pictograph train start road-signs road-signs-2026-07 --pipeline yolox --gpu a10g
pictograph train start road-signs road-signs-2026-07 -p rfdetr_detection \
  --config '{"epochs": 50}' --no-wait
pictograph train status <run-id>
pictograph train wait <run-id>
pictograph train cancel <run-id>
pictograph models list --json
```
