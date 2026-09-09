# Auto-annotation

Two labelling engines behind one resource, `client.auto_annotate`:

- **SAM3** - open-vocabulary segmentation. Prompt it with a point, a box, or a
  text phrase. No training required; works on classes you invent on the spot.
- **A trained model of yours** - pass `model=` to the batch endpoint and label a
  dataset with a model you already trained.

All of it consumes compute credit. Price a batch job before running it (see
"Cost", below).

## Single-image prompts

Synchronous, one image, one call. Each returns a `PromptResult`:

| Field | Notes |
|---|---|
| `status` | `success` / `no_detection` / `below_threshold` |
| `annotations` | The predicted annotations. Empty unless `status == "success"`. |
| `score` | Best candidate's confidence. |
| `inference_time` | Seconds. |

Prompt results are **not saved**. Persist them with `client.annotations.save`.

### Point - "click here, segment that"

```python
result = client.auto_annotate.point(
    "my-photos",
    image_filename="img-1.jpg",
    x=320,
    y=240,
    name="car",
    positive_points=[(310, 250)],
    negative_points=[(100, 100)],
    score_threshold=0.75,
)
if result.status == "success":
    client.annotations.save(
        dataset_name="my-photos",
        image="img-1.jpg",
        annotations=result.annotations,
    )
```

Best when the user knows where the object is.

### Box - "segment what's in this box"

```python
result = client.auto_annotate.box(
    "my-photos",
    image_filename="img-1.jpg",
    box={"x": 100, "y": 200, "w": 200, "h": 150},
    name="car",
    return_polygon=True,
    confidence_threshold=0.5,
    negative_boxes=[{"x": 50, "y": 50, "w": 30, "h": 30}],
)
```

Best when the user has drawn a rough box and wants a precise outline.
`return_polygon=False` returns only the refined box - right for detection work,
where a polygon is noise.

### Text - "find every X"

```python
result = client.auto_annotate.text(
    "my-photos",
    image_filename="img-1.jpg",
    text_prompt="red cars",
    output_type="polygon",
    confidence_threshold=0.3,
    max_detections=50,
)
```

Open-vocabulary - the phrase does not have to be a class that exists yet. Best when
nobody is going to click every object.

## Batch - many images

Use this above roughly ten images. One job, polled to completion.

```python
job = client.auto_annotate.batch(
    "my-photos",
    image_filenames=["img-1.jpg", "img-2.jpg"],
    classes=[
        {"name": "car", "output_type": "polygon"},
        {"name": "person", "output_type": "bbox"},
    ],
    confidence_threshold=0.5,
    wait=True,
)
print(job.status, job.processed_images, job.total_annotations_added, job.failed_images)
```

Unlike the single-image prompts, a batch job **writes the annotations** onto the
dataset itself.

`wait=False` returns the kick-off snapshot immediately; poll with
`client.auto_annotate.get_batch(job.job_id)`, or block later with
`wait_for_batch(job.job_id)`. `cancel_batch(job.job_id)` stops it.

### Labelling with your own trained model

```python
job = client.auto_annotate.batch(
    "my-photos",
    image_filenames=filenames,
    classes=[],
    model="Road Sign Detector",
    top_k=3,
)
```

`model=None` (default) is the SAM3 path and requires at least one class. This is
the loop that makes a dataset grow: train on what is labelled, label the rest with
the model, correct, retrain.

### SAHI - small objects in large images

```python
job = client.auto_annotate.batch(
    "aerial",
    image_filenames=filenames,
    classes=[{"name": "vehicle", "output_type": "bbox"}],
    sahi=True,
    sahi_slice_size=640,
)
```

Slices each image into overlapping tiles plus one full-image pass, so small objects
are seen at near-native resolution; fragments are merged back into whole instances.
Smaller tiles find smaller objects and cost more passes. Range 256-1024.

**SAM3 only** - `sahi=True` together with `model=` is rejected. For a
non-inference alternative that also fixes small objects, see `client.images.tile`
in `references/bulk-operations.md`.

## `output_type`

Each class in a batch job declares what to emit:

| `output_type` | Writes | Use for |
|---|---|---|
| `polygon` (default) | a `polygon` annotation | segmentation, precise coverage |
| `bbox` | a `bbox` annotation | detection training |
| `tag` | an image-level tag, **not** an annotation | classification training |

`tag` writes no geometry. Do not use it when a detector will train on the result.

## Cost

Quote a batch job before you run it. Quoting spends nothing.

```python
quote = client.auto_annotate.quote(
    dataset_name="my-photos",
    image_filenames=filenames,
    classes=[{"name": "car", "output_type": "polygon"}],
    sahi=False,
)
print(quote.total_images, quote.estimated_credits, quote.remaining_credits)
print(quote.sufficient, quote.max_images, quote.exceeds_max_images)
```

To quote images that are not uploaded yet, pass `projected=` - a list of
`{"count": n, "width": w, "height": h}` - instead of `image_filenames`.

A job submitted without enough credit raises `PaymentRequiredError`, carrying
`credit_cost`, `credits_remaining` and `upgrade_url`.

## What SAM3 returns, and what it does not

Returns polygons (holes preserved as extra rings in `polygon.paths`), boxes (the
enclosing rectangle of the mask), and image-level tags.

Does **not** return polylines or keypoints - there is no segmentation analogue.
Label those by hand, or predict them with a trained `rfdetr_keypoint` model.

## Choosing a mode

| Situation | Use |
|---|---|
| The user clicks one spot | `point` |
| The user drags a rough box | `box` |
| One image, "find all the X" | `text` |
| More than ~10 images | `batch` |
| You already have a model for these classes | `batch(model=...)` |
| Small objects in high-resolution images | `batch(sahi=True)` |

## Over a whole dataset

`client.auto_annotate.dataset` wraps the batch endpoint with image pagination and
skip-already-annotated logic:

```python
report = client.auto_annotate.dataset(
    "my-photos",
    classes=[("car", "polygon"), ("person", "bbox")],
    confidence_threshold=0.5,
    overwrite=False,
    max_images=500,
)
print(report.images_processed, report.annotations_added, report.failures)
```

## CLI

```bash
pictograph auto-annotate text my-photos img-1.jpg --prompt "red cars"
pictograph auto-annotate quote --dataset my-photos --classes car:polygon
pictograph auto-annotate quote --frames 900 --width 1920 --height 1080 --classes car:bbox
pictograph auto-annotate batch my-photos --images img-1.jpg,img-2.jpg --classes car:polygon
pictograph auto-annotate get <job-id>
```
