# Inference - running a trained model

Three places a Pictograph model runs. All three return the **same five typed result
classes**, so code that consumes predictions never branches on where they came from.

| Where | Call | Needs |
|---|---|---|
| Edge - your machine | `get_model` / `load_model` | `pip install "pictograph[inference]"` |
| Remote - a deployed endpoint | `DeploymentClient(...).infer()` | the endpoint URL + its `pk_deploy_` token |
| Remote - one-off, no deployment | `client.models.predict(...)` | an API key; spends compute credit |

## The five tasks

One model class and one result class per task. `task=` narrows the return type at
the call site, and it is **verified** against the model's real task - a mismatch
raises rather than handing back a class whose `predict` returns another shape.

| `task=` | Model class | Result class | `predictions` are |
|---|---|---|---|
| `object_detection` | `DetectionModel` | `DetectionResult` | `BBoxAnnotation` |
| `instance_segmentation` | `InstanceSegmentationModel` | `InstanceSegmentationResult` | `PolygonAnnotation \| BBoxAnnotation` |
| `semantic_segmentation` | `SemanticSegmentationModel` | `SemanticSegmentationResult` | `PolygonAnnotation` |
| `keypoint_detection` | `KeypointModel` | `KeypointResult` | `KeypointAnnotation` |
| `classification` | `ClassificationModel` | `ClassificationResult` | - see `.classes` |

Omit `task=` and you get the `AnyModel` / `AnyResult` union, which narrows with
`isinstance` or by matching on `result.model_type`.

Per-task extras:

- `InstanceSegmentationResult.polygons` - only the instances whose mask
  polygonized. The union exists because a mask that is empty or below the area
  floor degrades to its box; both carry `bounding_box`, so a box-only consumer can
  ignore the distinction.
- `KeypointResult.points` - every predicted point, flat.
  `KeypointResult.instances` - the same points grouped into objects by
  `instance_id`, ordered ascending, with unassociated points as their own
  single-element groups sorted last.
- `ClassificationResult.classes` - ranked, highest first, never empty.
  `.top` is therefore non-optional. `.tags` is just the names.

Every result also carries provenance: `backend`, `device`, `providers`,
`inference_ms`.

## Edge - by name

```python
from pictograph import get_model, DetectionModel, DetectionResult

model: DetectionModel = get_model("Shelf Detector", task="object_detection")
result: DetectionResult = model.predict(
    "photo.jpg",
)

for p in result.predictions:
    print(p.name, round(p.confidence, 2), p.bounding_box)

model.close()
```

Weights download and cache on first use (`~/.pictograph/models`, or
`$PICTOGRAPH_CACHE_DIR`). The cache key includes the model's current version, so a
retrained model is re-fetched rather than served stale.

`predict` accepts a file path, an `http(s)` URL, raw bytes, a PIL image, or a
decoded **BGR** numpy array (`cv2.imread` order). `predict_batch(list)` returns one
result per image, in order.

Models are context managers, and should be closed - an ONNX session and a CUDA/MPS
allocator both hold memory the garbage collector will not promptly return:

```python
with get_model("Shelf Detector", task="object_detection") as model:
    results = model.predict_batch(
        ["a.jpg", "b.jpg"],
        confidence=0.4,
    )
```

## Edge - fully offline

`load_model` takes local files and makes no API call. Point it at the weights and
the `config.json` from the model's file bundle (`client.models.files(...)` lists
them; `client.models.download_file(...)` fetches one).

```python
from pictograph import load_model

model = load_model("shelf.onnx", "config.json", task="object_detection")
```

**`format=` defaults to the weights' own suffix**, so the call shape never changes:

```python
load_model("model.onnx", cfg, task="classification")  # ONNX Runtime
load_model("xnnpack-fp32.pte", cfg, task="classification")  # ExecuTorch
load_model("sm75-trt10.13.3.9-fp16.engine", cfg, task="classification")  # TensorRT
```

A `.pth` / `.safetensors` checkpoint is **not** an executable graph - it is rebuilt
from its training pipeline's own model definition, which needs the model record. So
`load_model` refuses it and points you at `get_model(name, format="safetensors")`.

## The five formats

The same trained weights are published in every executable form. You name the
**weights file**; the runtime that executes it follows from that and is never asked
for separately. Pick by what the hardware is; the model class and the result type do
not change.

| `format=` | File | Runtime | For |
|---|---|---|---|
| `pytorch` | `.pth` | PyTorch | the training checkpoint - fine-tuning, research |
| `safetensors` | `.safetensors` | PyTorch | the same `nn.Module` from the parity-gated container |
| `pytorch_engine` | `.pte` | ExecuTorch | portable edge - phones, ARM boards, embedded CPU |
| `onnx` | `.onnx` | ONNX Runtime | the default; runs anywhere |
| `tensorrt_engine` | `.engine` | TensorRT | NVIDIA, lowest latency |

```python
get_model("Detector", task="object_detection")  # onnx
get_model("Detector", task="object_detection", format="pytorch_engine")
get_model("Detector", task="object_detection", format="tensorrt_engine")
get_model("Detector", task="object_detection", format="safetensors")
```

**A format the model does not publish is refused, never substituted** - the error
names the formats it does have. Not every pipeline writes both native containers:
keypoint models publish `safetensors` but no `.pth`.

Install the one you want:

```bash
pip install "pictograph[inference]"              # ONNX Runtime
pip install "pictograph[inference,torch]"        # + PyTorch
pip install "pictograph[inference,executorch]"   # + ExecuTorch
pip install "pictograph[inference,tensorrt]"     # + TensorRT (NVIDIA only)
```

`model.backend` reports which runtime actually ran it.

### A TensorRT `.engine` is not portable (important)

A plan is compiled for **one GPU architecture, one TensorRT version and one
precision**. Copying it to another machine fails at load. The loader checks before
deserializing and refuses with a message naming both the built-for target and your
device, rather than surfacing a raw crash.

Because of that, `format="tensorrt_engine"` defaults `target=` to **this machine's**
SM - fetching any other engine would be downloading a file that provably cannot run
here. Pass it explicitly (`target="sm80"`, …) only when building for elsewhere.

A `.pte` **is** portable across devices for its lowering backend, which defaults to
`xnnpack` (portable CPU).

### `precision=` and `target=`

- `precision` is `"fp32"` or `"fp16"`. `pytorch_engine` and `tensorrt_engine`
  artifacts are published per precision, so this selects which one to fetch. For
  `onnx` a precision differing from the model's own fetches a derived graph.
- The two native formats serve the version's checkpoint as-is. A checkpoint has no
  derived form, so asking for a precision it was not trained at raises rather than
  quietly handing back the other file.
- `target` is the GPU architecture for `tensorrt_engine`, and the lowering backend
  for `pytorch_engine`. It does not apply to the other three.

### `device=` - the one hardware argument

`device=` picks the HARDWARE, on **both** loaders, with the same values and the same
meaning on every format:

| `device=` | means |
|---|---|
| `"auto"` (default) | the best hardware available for these weights |
| `"cpu"` | the CPU only - reproducible, instant to load, right for CI and parity work |
| `"cuda"` / `"cuda:1"` | an NVIDIA GPU; the index picks one on a multi-GPU box |
| `"mps"` | Apple's accelerator |

```python
get_model("Detector", task="object_detection", device="cuda")
load_model("weights.pth", cfg, task="classification", device="cuda:1")
load_model("model.onnx", cfg, task="classification", device="cpu")
```

**You never name an execution provider.** It is derived from `(device, format)`:
`mps` reaches torch's MPS backend for a `.pth` and CoreML for an `.onnx`; `cuda`
reaches TensorRT for an `.engine` and CUDA for an `.onnx`. A pairing that does not
exist - a `.engine` on the CPU, a `.pte` on CUDA - raises and names what that format
CAN run on.

**A device that cannot be honoured raises**, naming what IS available; it is never
silently downgraded to the CPU. `"auto"` is the exception and the only one: it never
promised particular hardware, so it degrades down the ladder and warns.

`model.device` reports what **ran**, which can be more specific than what you asked
for - `device="mps"` on an `.onnx` reports `"coreml"`, because that is the mechanism
that got you there. That is what tells you a CUDA request really landed on CUDA.

`PICTOGRAPH_INFERENCE_THREADS` caps the ONNX Runtime thread pool - set it in a
container, where the default sizes from host cores rather than the cgroup limit.

## Remote - a deployed endpoint

```python
from pictograph import DeploymentClient, DetectionResult

with DeploymentClient(
    "https://<your-deployment-endpoint>/predict",
    "pk_deploy_...",
    task="object_detection",
) as dc:
    result: DetectionResult = dc.infer("photo.jpg")  # path | URL | bytes
    for p in result.predictions:
        print(p.name, round(p.confidence, 2), p.bounding_box)
```

`infer` returns the same five typed result classes as a local model - it is not a
dict, and `result["predictions"]` is wrong. Use `infer_raw()` if you specifically
want the endpoint's untouched JSON body.

Per-call overrides: `confidence=`, `class_filter=[...]`, `top_k=`.

`task=` is checked against the `model_type` the endpoint reports on every response,
so pointing a detector client at a classifier endpoint raises instead of silently
producing an empty result.

What a remote result cannot tell you: `providers` is empty and `inference_ms` is
`None`, because the wire carries no forward-pass timing and substituting the
client's round-trip would be a different measurement wearing the same name.
`device` is what the endpoint reports, or `"remote"` when it reports none - never a
fabricated `"cpu"`.

Errors map to the same typed hierarchy as `Client`. Inference is side-effect-free,
so transient `429`s, cold-start `5xx`s and network blips are retried with backoff;
pass `max_retries=0` for a latency-strict caller.

Manage deployments with `client.deployments` (`create`, `quote`, `pause`, `resume`,
`list`). Deployments bill by uptime, not per call.

## Remote - one-off, no deployment

```python
result = client.models.predict(
    "Shelf Detector",
    image="photo.jpg",
    confidence=0.5,
)
print(result.model_type, result.annotations, result.tags)
```

This returns `ModelPredictResult` (raw annotation dicts plus `tags`), not one of
the five typed classes - it is the convenience path for a single image when you
have neither a local runtime nor a deployment. It spends compute credit.

## Getting the files yourself

```python
manifest = client.models.files(
    "Shelf Detector",
)
for f in manifest.files:
    print(f.name, f.runtime, f.precision, f.target_key, f.size_bytes, f.stale)

client.models.download(
    "Shelf Detector",
    output_path="./model.onnx",
    format="onnx",
)
client.models.download_file(
    "Shelf Detector",
    file_name="config.json",
    output_path="./config.json",
)
```

`format=` is `onnx` / `pytorch` / `safetensors` / `pte` / `engine`, with
`precision=` and `target=` as above. A file marked `stale` was built against an
older pinned toolchain: advisory for `.onnx` and `.pte` (they still load), but
**blocking** for `.engine`, which TensorRT refuses to deserialize across versions.

## CLI

```bash
pictograph models predict "Shelf Detector" photo.jpg --json    # local
pictograph models predict "Shelf Detector" photo.jpg --remote  # server-side
pictograph models download "Shelf Detector" --output ./m.onnx
pictograph deployments predict <deployment-id> photo.jpg \
  --token pk_deploy_... --endpoint https://…/predict
```
