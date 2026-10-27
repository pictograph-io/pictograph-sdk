<div align="center">

<img src="https://storage.googleapis.com/pictograph-assets/logos/FullLogo_Transparent_NoBuffer.png" alt="Pictograph" height="64">

<h3>Pictograph</h3>

<p><strong>Real-time vision tools for agents and developers.</strong></p>

<p>Label images with SAM3, train small task-specific models, and run them in real time:<br>GPU endpoints, live video streams, and tracking workflows.</p>

[![PyPI](https://img.shields.io/pypi/v/pictograph?color=3e62b0&label=pypi)](https://pypi.org/project/pictograph/)
[![Python](https://img.shields.io/pypi/pyversions/pictograph?color=3e62b0)](https://pypi.org/project/pictograph/)
[![License](https://img.shields.io/badge/license-MIT-3e62b0)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-pictograph.io-3e62b0)](https://pictograph.io/docs)

<p>
<a href="https://pictograph.io/docs">Docs</a> ·
<a href="https://pictograph.io/docs/quick-start">Quick start</a> ·
<a href="https://pictograph.io/docs/api-reference">API reference</a> ·
<a href="https://app.pictograph.io">App</a>
</p>

</div>

---

Four ways in. Same API underneath, so you can move between them.

| | |
|---|---|
| **Python** | `pip install pictograph`, this repo |
| **CLI** | `pip install "pictograph[cli]"` · [reference](https://pictograph.io/docs/cli) |
| **REST** | Any language, `X-API-Key` header · [reference](https://pictograph.io/docs/api-reference) |
| **Claude Code** | `/plugin install pictograph --marketplace pictograph-io/pictograph-sdk` · [guide](https://pictograph.io/docs/agents) |

## Quick start

```bash
pip install pictograph
```

Upload a directory, draft labels from a text prompt, review them, then export and train.

```python
from pictograph import AnnotateReport, Client, TrainingRun, UploadReport

client = Client()  # reads PICTOGRAPH_API_KEY

uploaded: UploadReport = client.images.upload_from_directory(
    dataset_name="road-signs",
    directory="./road_signs",
)
print(f"{uploaded.images_uploaded} images uploaded")

labelled: AnnotateReport = client.auto_annotate.dataset(
    dataset_name="road-signs",
    classes=[("stop_sign", "bbox"), ("yield", "bbox")],
)
print(f"{labelled.annotations_added} annotations added")

# They are drafts: a person reviews them before you train. Start with the least
# confident (the app's review stage approves an image or sends it back).
for image in client.images.list(
    dataset_name="road-signs",
    min_confidence_lt=0.5,
):
    print("review first:", image.filename)

client.exports.create(
    dataset_name="road-signs",
    name="road-signs-v1",
    format="pictograph",
    include_images=True,
    wait=True,
)

run: TrainingRun = client.training.create(
    dataset_name="road-signs",
    export_name="road-signs-v1",
    pipeline_type="yolox",
    name="road-signs-detector",
)
print("Trained model:", run.model_id or run.status)
```

A class is a `(name, output_type)` pair, where the output type is `bbox`,
`polygon` or `tag`, because auto-annotation has to know what shape to produce.
Training runs on an **export**, never on a dataset directly: you create the
export, see what went into it, then train that.

The same thing from the shell:

```bash
pictograph login
pictograph images upload-directory road-signs ./road_signs
pictograph auto-annotate batch road-signs --images 001.jpg,002.jpg --classes "stop_sign:bbox"
pictograph images list road-signs --min-confidence-lt 0.5    # review the drafts, least confident first
pictograph exports create road-signs --name road-signs-v1 -f pictograph --include-images
pictograph train start road-signs road-signs-v1 --pipeline yolox --gpu a10g
pictograph models download "road-signs-detector" -o ./yolox.onnx
```

Or over REST:

```bash
curl -s https://api.pictograph.io/api/v1/developer/datasets/ \
  -H "X-API-Key: $PICTOGRAPH_API_KEY"
```

Resources are addressed by **name** everywhere. There are no IDs to look up first.

## Typed

Responses are Pydantic models. Failures are typed exceptions.

```python
from pictograph.exceptions import NotFoundError, PaymentRequiredError, RateLimitError

try:
    client.datasets.get(name="does-not-exist")
except NotFoundError:
    ...
```

Transient failures retry with backoff, and writes carry an idempotency key so a
retry cannot apply twice.

## Async

Every method has an async twin with the same signature.

```python
import asyncio
from pictograph import AsyncClient

async def main() -> None:
    async with AsyncClient() as client:
        datasets = await client.datasets.list(limit=5)
        print([d.name for d in datasets])

asyncio.run(main())
```

## Deploy, then stream live video

Deploy a trained model on a GPU, then send it single images or a live stream. An RTSP
camera, a browser webcam (WebRTC) or a video file comes back as one result per frame.

```python
from pictograph import Client, CreatedDeployment, DetectionResult

client = Client()
created: CreatedDeployment = client.deployments.create(
    model="road-signs-detector",
    gpu_type="t4",
    precision="fp16",
)
deployment = client.deployments.wait_until_active(
    deployment=created.deployment,
)
infer = client.deployments.connect(
    deployment=deployment,
    api_key=created.auth_token,
    task="object_detection",
)

with infer.open_stream(source="rtsp://camera.example.com:554/live") as stream:
    for frame in stream.frames():
        result: DetectionResult = frame.result
        print(frame.index, [p.name for p in result.predictions])
```

```bash
pictograph deployments create "road-signs-detector" --gpu t4 --precision fp16 --wait
pictograph deployments stream road-signs-detector rtsp://camera.example.com:554/live --token pk_deploy_...
```

See [deployments](https://pictograph.io/docs/deployments) and [streams](https://pictograph.io/docs/streams).

## Run models on your own hardware

```python
from pictograph import get_model

model  = get_model(name="signs-detector", task="object_detection")
result = model.predict(image="test.jpg")
```

Weights download and cache on first use. ONNX, PyTorch, ExecuTorch and TensorRT
are supported targets. See [local inference](https://pictograph.io/docs/local-inference).

## Agents

```bash
pip install "pictograph[agents]"
pictograph agents install-skill --target claude-code
```

Or install the skill as a Claude Code plugin from this repository:

```text
/plugin install pictograph --marketplace pictograph-io/pictograph-sdk
```

The plugin asks for your API key when you enable it, keeps it in your system's credential
store, and sets it as `PICTOGRAPH_API_KEY` for the commands Claude runs (a key already in
your environment wins). Its skill may run `pictograph` commands and read files without
asking; anything else still asks.

```python
from pictograph.agents import Toolkit

tools = Toolkit(client).as_anthropic_tools()   # or .as_openai_tools()
```

## Offline utilities

No network needed, so an existing local dataset can be brought into typed models
before uploading anything.

| | |
|---|---|
| `pictograph.formats` | Read and write COCO, YOLO, Pascal VOC |
| `pictograph.metrics` | Score detections against ground truth by IoU |
| `pictograph.augment` | Transform images, remapping annotation geometry with them |
| `pictograph.tile` | Slice images into a grid, clipping annotations per tile |

## Documentation

Full reference at [pictograph.io/docs](https://pictograph.io/docs).

[Quick start](https://pictograph.io/docs/quick-start) ·
[Installation](https://pictograph.io/docs/installation) ·
[Authentication](https://pictograph.io/docs/authentication) ·
[API reference](https://pictograph.io/docs/api-reference) ·
[CLI](https://pictograph.io/docs/cli) ·
[Agents](https://pictograph.io/docs/agents) ·
[Annotation format](https://pictograph.io/docs/annotation-format) ·
[Auto-annotation](https://pictograph.io/docs/sam3-auto-annotation) ·
[Local inference](https://pictograph.io/docs/local-inference) ·
[Deployments](https://pictograph.io/docs/deployments) ·
[Streams](https://pictograph.io/docs/streams) ·
[Export conversion](https://pictograph.io/docs/export-conversion) ·
[Async client](https://pictograph.io/docs/async-client) ·
[Error handling](https://pictograph.io/docs/error-handling) ·
[Rate limits](https://pictograph.io/docs/rate-limits)

Requires Python 3.10 or newer. Extras: `[cli]`, `[agents]`, `[inference]`,
`[torch]`, `[all]`.

## Contributing, security, license

[CONTRIBUTING.md](CONTRIBUTING.md) has the development setup and the checks a
change must pass. Report vulnerabilities per [SECURITY.md](SECURITY.md), not as a
public issue.

The SDK is MIT licensed, see [LICENSE](LICENSE). Two model architectures are
vendored so local inference works without pulling their full training stacks;
both are Apache-2.0 and keep their own licence and attribution alongside the
code:

| Component | Upstream | Licence |
|---|---|---|
| `pictograph.inference._rfdetr` | [RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow) | [Apache-2.0](src/pictograph/inference/_rfdetr/LICENSE) · [NOTICE](src/pictograph/inference/_rfdetr/NOTICE) |
| `pictograph.inference._yolox` | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Megvii) | [Apache-2.0](src/pictograph/inference/_yolox/LICENSE) · [NOTICE](src/pictograph/inference/_yolox/NOTICE) |
