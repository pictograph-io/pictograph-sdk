# Changelog

All notable changes to the Pictograph Python SDK are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses conservative [Semantic Versioning](https://semver.org/):
ordinary changes bump the **patch** version; a genuinely substantial new
capability bumps the **minor** version.

## [Unreleased]

## [1.69.67]

### Changed
- **The rest of the prototype residue is out of the inference wrappers.** The
  same shape found in the classifier last release was in three more files:
  `RFDETRConfig` and `RFDETRSegConfig` (the latter defaulting its class map to a
  customer's vehicle/windshield classes, both pointing `model_path` at a local
  file that never existed here), and a dead `Detection`, `Metrics` and
  `SemanticSegmentationModel` in the YOLOX wrapper - the last shadowing the real
  semantic-segmentation class in a different module. With them went
  `process_detections`, `process_instance_segmentations` and
  `InstanceSegmentation`, and the module docstrings that told you to
  `from data_objects import ...`, a module that only exists in prototype
  workspaces. Nothing referenced any of it. No behaviour change.
- A new gate keeps it out: every public symbol in a wrapper must be exported,
  referenced in the SDK, or used by another wrapper, and no wrapper may hard-code
  a local model path or import from the prototype module. References are resolved
  through the AST rather than by text search, because `Detection` and `Metrics`
  both look used when you grep - those words appear in ordinary prose, which is
  how they survived an earlier cleanup.

## [1.69.66]

### Changed
- **The classification wrapper is construction and preprocessing only.** It
  carried a `PytorchClassifierConfig` whose defaults were one developer's local
  model path and a nine-class candy dataset, plus a `from_config`, a
  `Classification` result model, `predict`, `predict_batch` and `postprocess` -
  none of which anything called. Both real callers drive the ONNX session
  directly so they can batch across images and apply their own top-k, threshold
  and class-filter policy, which means that dead code was a second decode path
  waiting to diverge from the one actually in use. Removed; the class went from
  253 lines to 107. `Classification` is no longer exported from the private
  `_wrappers` package. No behaviour change - nothing referenced any of it.
- The classifier's execution-provider default is a tuple rather than a shared
  mutable list, and now includes CUDA. The config object it used to live on
  omitted CUDA, so anything constructed from it would silently have run on CPU.

### Fixed
- README examples now match the documented flow exactly, and are bound by a test
  that resolves every call against the real client.

## [1.69.65]

### Fixed
- **The bundled Claude skill ships its four wrapper scripts again.** The published
  1.69.64 wheel and sdist contained `SKILL.md` and the reference pages but an empty
  `scripts/` directory, so `pictograph agents install-skill` installed a skill whose
  own instructions pointed at `scripts/train.py`, `scripts/export.py`,
  `scripts/upload_and_annotate.py` and `scripts/import_connector.py` - none of which
  were there. Reinstall on this version to repair an existing install.
- **`from pictograph import AsyncClient` is typed again.** The lazy-import shim that
  keeps `import pictograph` fast had no type-checker counterpart, so every lazily
  exported name resolved to `object`: under mypy, `AsyncClient()` failed with
  "object not callable" and its attributes were untyped. Affected the documented
  async entry point in a package that ships `py.typed`. Runtime behaviour, including
  the laziness, is unchanged.
- `deployments.create(...)` example now reads the bearer token from `auth_token`.
  There is no `token` attribute; the example crashed on the line that prints a token
  the API only ever returns once.

### Security
- **`datasets.download()` cannot be steered outside `output_dir`.** The per-image
  filename in the server's listing was joined into the local path unmodified, so an
  absolute value discarded `output_dir` entirely and `../` escaped it. Both the sync
  and async paths now reduce it to a single component. Ordinary filenames, including
  ones with spaces or parentheses, are byte-identical to before.
- **`pictograph agents install-skill` validates `--skill`.** The value was joined to
  `~/.claude/skills` and the result passed to a recursive delete, so `--skill ..`
  removed the user's entire `~/.claude` directory. Only a plain name is accepted.
- The local weights cache reduces the server-supplied model id and version to safe
  path components, matching the guard the PyTorch loader already applied.

### Changed
- **Live tests need `PICTOGRAPH_TEST_KEY` and are deselected by default.** They were
  collected by a bare `pytest` and unlocked by `PICTOGRAPH_API_KEY` - the variable
  every user exports for ordinary use - so running the suite in a clone could spend
  real credits against production. They now require a dedicated test-organization key
  and an explicit `-m live`.
- Removed the `cache` and `telemetry` extras. Nothing in the package imported
  `aiosqlite` or `opentelemetry`, so `pip install "pictograph[all]"` was pulling three
  packages that did nothing. `[all]` is now `agents,cli,inference`.

## [1.69.64]

### Fixed
- **`pictograph directories list` / `tree` / `create` name their first argument
  `DATASET`, and its help says "Dataset name."** All three previously advertised
  a dataset UUID, which was simply wrong - the commands have always taken a
  name, and their sibling commands (`stats`, `delete`, `rename`) already said so.
  Positional, so how you invoke them is unchanged; only `--help` was wrong.

## [1.69.63]

### Added
- **`images.download_bundle(dataset, image, path)`** - fetch one image's data
  bundle as a single zip: the original bytes, its depth map when one exists, its
  annotations as Pictograph JSON, and a `manifest.json` naming exactly what is
  and is not inside. It is byte for byte the archive the annotation editor's
  "Image data" button hands over. A missing depth map is not an error - the zip
  still arrives and the manifest records the omission and why. Writes atomically
  like `download()`, so a failed transfer leaves no partial zip and retrying is
  safe. CLI: `pictograph images download-bundle`; async twin included.

## [1.69.62]

### Changed
- **Response models expose `dataset_id` (and a Task's dataset name as
  `dataset`), not `project_id`/`project`.** The developer-facing primitive is a
  *dataset*; `project` was an internal name that had leaked into the surface.
  Renamed on `Task`, `Export`, `ModelEvaluation`, `TaggedImage`, `Directory` and
  `DatasetImportProgress`. **Zero-break on the wire:** each field accepts either
  key, so this build parses today's API unchanged, and `model_dump()` emits the
  canonical `dataset_id`. Code still reading `.project_id` now fails loudly
  instead of silently diverging. Request-side query params are unchanged.

## [1.69.61]

### Changed
- **`project` is gone from user-facing text across the developer surface.**
  Docstrings, CLI `--help`, Pydantic field descriptions, the agent tool-arg
  schemas and the bundled Claude skill all say *dataset* now. Prose only; no
  field, wire or behaviour change.

## [1.69.60]

### Changed
- **Internal codenames removed from two public model docstrings.**
  `ModelVersionsPayload` and `WorkflowRunCreated` carried internal tracker
  references in text that surfaces under `help()` and in the online API
  reference. Meaning unchanged; no API, field or behaviour change.

## [1.69.59]

### Added
- **`client.tasks`** - read your organization's annotation tasks. `list()` and
  `iter()` page them newest first; `contributions(task_id)` returns the
  per-annotator breakdown (active time, images worked, images completed,
  annotations added) alongside the task's totals. CLI: `pictograph tasks list`
  and `pictograph tasks contributions`; async twin included.

## [1.69.58]

### Changed
- **A cold-storage restore is charged on success, not up front.**
  `datasets.restore()` used to bill the restore fee the moment the transition
  was accepted, so a restore that failed part-way or was abandoned still cost
  you. The fee now settles only after every object is warm; a failed or
  cancelled restore costs nothing. `DatasetStorageTransition.charged_micro_usd`
  is accordingly now `quoted_micro_usd` - the amount that *will* be charged on
  success. Your balance is still checked before the restore starts, so an
  underfunded restore is refused rather than left half-done, and the charge is
  still idempotent per frozen generation.

## [1.69.57]

### Added
- **Custom auth headers on a webhook endpoint.** `webhooks.create()` and
  `.update()` take `auth_headers={"Authorization": "Bearer …"}` - headers sent
  on every delivery, so your endpoint can sit behind your own gateway. They
  complement the delivery signature rather than replacing it: the header gates
  who can *reach* the endpoint, the signature proves who *sent* the payload.
  Values are stored encrypted and are never returned by `list()` / `get()` -
  only their names, as `auth_header_names`. Pictograph's own headers are layered
  on top at send time, so a custom header can never clobber the signature.
  CLI: `--auth-header 'Name: Value'`, repeatable.

## [1.69.56]

### Changed
- **The five synchronous-only resource methods now say so in their docstrings.**
  `datasets.as_pytorch`, `models.load`, `images.augment`, `images.tile` and
  `auto_annotate.dataset` exist on `Client` but not on `AsyncClient`, so an
  async caller used to meet the asymmetry as a surprise `AttributeError`. Four
  are CPU- or GPU-bound (a torch `Dataset`, a local weight load, sequential
  deterministic Pillow work) where `asyncio` would add nothing. No behaviour
  change.

## [1.69.54]

### Security
- **The API key is pinned to the configured API host and can never be sent to a
  server-supplied one.** The authenticated client attaches `X-API-Key` as a
  default header, so every request it makes carries your credentials - including
  requests to absolute URLs that came *from* the API, such as the
  `annotation_url` in a dataset `download()` listing. A malicious or compromised
  API could have pointed one at another host and had the key delivered there.
  The transport now reduces any absolute URL to its path and query and
  re-resolves it against `base_url`, so a server-supplied scheme or host is
  ignored; a foreign host is logged at WARNING. Enforced once in the transport,
  not per call site. Normal operation is unchanged. Signed-URL image and model
  transfers were already safe - they use separate unauthenticated clients.

## [1.69.52]

### Changed
- **No emoji in the shipped source, enforced by a test rather than a one-time
  sweep.** The published package is read by anyone, so its source is now checked
  on every `pytest` run. Typography stays allowed (the ASCII arrow, middle dot,
  `->`, `!=`). Two user-visible consequences: the CLI renders boolean cells as
  `yes`/`no` instead of dingbats, and a skill-reference heading dropped a warning
  emoji. Internal references were also removed from the vendored YOLOX `NOTICE`;
  its Apache-2.0 provenance and re-diffable-mirror statement are unchanged.

## [1.69.51]

### Added
- **Runnable `examples/`.** Seven self-contained scripts - quickstart, every
  annotation geometry type, directory upload, search, predict, train + deploy
  (GPU work gated behind an env flag), and an async twin. Each generates its own
  demo images, so it runs with only `PICTOGRAPH_API_KEY` set, and each is
  idempotent - re-running reuses the dataset it made. A gate parses every example
  and binds each `client.<resource>.<method>` call and each
  `from pictograph import …` name against a real client, so a rename breaks the
  build here rather than in your first script.

## [1.69.43]

### Changed
- **`folders` is now `directories`, everywhere.** The resource, its async twin,
  the models, the CLI command group and every argument. `client.folders` is
  gone; `client.directories` replaces it. The server renamed at the same time,
  so 1.69.42 and earlier call an endpoint that no longer exists.
- **Models are addressed by `name=`.** `get_model(name_or_id=…)`,
  `models.predict(model=…)` and `models.load(model=…)` all take `name=` now,
  matching the seven `models.*` methods that already did. `name_or_id` was the
  worst of the three: it advertised that an id would work, which is exactly what
  the name-addressed API removes.

### Fixed
- **A checkpoint can no longer execute code when it is loaded.** `torch.load`
  fell back to `weights_only=False` whenever the safe unpickler refused, logging
  it at `debug` (off by default). That runs whatever the file's author embedded,
  and models can be FORKED between organizations, so a `.pth` is not always
  yours. The inert types our own checkpoints carry are allowlisted so the safe
  path succeeds, and anything beyond it raises `UnsafeCheckpointError` instead of
  executing.
- **Package metadata pointed at a GitHub organization that does not exist**
  (`pictograph-labs`), so Repository, Changelog and Issues were dead links on
  every published version. Corrected to `pictograph-io`.
- **The directories resource called routes that no longer exist** - the
  id-addressed list and tree paths, retired when the API moved to names.
- **Cache paths built from server-supplied values are sanitised.** A model id or
  filename from the API was interpolated straight into a filename, letting a
  malicious or compromised server steer a write outside the cache directory.
- **Source-provider credentials are no longer required on `argv`.** The CLI's
  `connectors --key` and the bundled skill's `import_connector.py` now prefer
  `PICTOGRAPH_SOURCE_KEY`; argv persists in shell history and is readable by any
  local user via `ps`.

### Removed
- **Customer cloud-storage (`client.storage`).** Untested and never exercised by
  a customer, and it held the highest-consequence secret in the product: a live
  credential to somebody else's cloud account.
- 3,668 lines of dead code from the vendored inference wrappers; `yolox_wrapper`
  alone was 76% unreferenced.

## [1.69.16]

### Fixed
- **`build_result`'s `source` no longer calls every model a "Classifier".** The
  one local call site hard-coded `Classifier {name}` for all five tasks, which
  was invisible while the only message that read it was the classifier's own
  refusal. 1.69.15 added a second reader and it started telling detector users
  their detector was a classifier.

## [1.69.15]

Three local-inference defects, all found by EXECUTING the Use Model panel's real
emitted snippets against the live API rather than asserting on the emitted
string. None was reachable from a payload fixture, and the first was shipped and
user-visible.

### Fixed
- **A native-container YOLOX raised on Apple silicon under the default
  `device="auto"`.** `Tensor.type()` manufactures the legacy string
  `'torch.mps.FloatTensor'` on MPS and `Tensor.type(…)` then refuses to parse it
  - legacy tensor types exist only for CPU and CUDA - so `decode_outputs` died
  with `ValueError: invalid type: 'torch.mps.FloatTensor'` on the last statement
  of an otherwise-complete forward pass. Every `format="pytorch"` /
  `format="safetensors"` YOLOX snippet a Mac user copied failed on paste; the
  same weights ran on `device="cpu"`. The six legacy-type-string casts in the
  vendored YOLOX tree now carry a dtype and an explicit device, which is the form
  upstream's own sibling call already uses. Numerically identical on CPU and CUDA
  (upstream weights strict-load into this tree and the forward pass is
  `torch.equal`), and it newly permits a device the model could not run on at
  all. Recorded in `_yolox/NOTICE` under Apache-2.0 § 4b.
- **Two models loaded in one session collided in the weights cache.** The
  rebuilt RF-DETR container was named `{weights.stem}.rfdetr.pth`, and every
  Pictograph model publishes its native weights as `model.safetensors` - so
  offline, every model in an organization resolved to one `model.rfdetr.pth`.
  The second model loaded found it already present and rebuilt itself from the
  FIRST model's architecture, class list and resolution. The container is now
  keyed on the model: the source file's `(path, size, mtime_ns)` plus the
  variant, class list and resolution that shape the payload.
- **A degenerate prediction no longer takes down `.predict()`.** A zero-extent
  box in a model's low-confidence tail raised
  `ValidationError: bounding_box.h Input should be greater than 0` and the caller
  got nothing - not the ~85 well-formed predictions beside it. `build_result` now
  drops predictions with no spatial extent (a zero or negative box or oriented
  box, a polygon ring under three points). Deliberately narrow: a prediction
  malformed in any other way still raises, because that is a producer bug rather
  than a weak detection.

## [1.69.14]

### Changed
- **`pictograph.pipelines` is dissolved. Every orchestrator moved onto the
  resource that owns its noun.** A module of wrappers around other calls is not a
  primitive: `upload_dataset_from_folder(client, …)` did nothing but call
  `client.datasets` and `client.images`, so it now IS an images method. The
  mapping, in full (shown with each method's CURRENT name):

  | Was | Is |
  |---|---|
  | `pipelines.upload_dataset_from_folder(client, ds, folder)` | `client.images.upload_from_directory(ds, folder)` |
  | `pipelines.augment_dataset(client, src, ops=[...])` | `client.images.augment(src, [...])` |
  | `pipelines.tile_dataset(client, src, rows=2)` | `client.images.tile(src, rows=2)` |
  | `pipelines.auto_annotate_dataset(client, ds, classes=[...])` | `client.auto_annotate.dataset(ds, [...])` |
  | `pipelines.import_coco_annotations(client, ds, coco)` | `client.annotations.import_coco(ds, coco)` |
  | `pipelines.import_pascal_voc_annotations(client, ds, xml)` | `client.annotations.import_pascal_voc(ds, xml)` |
  | `pipelines.import_yolo_annotations(client, ds, labels, names)` | `client.annotations.import_yolo(ds, labels, names)` |

  Behaviour is identical - this is a relocation, not a redesign. The `client`
  first argument is gone (it is `self`), and `ops` / `classes` are now positional
  on `images.augment` and `auto_annotate.dataset`, matching how every other
  resource method takes its subject.

  `pipelines.train_pipeline` moved too, but the method it moved to
  (`training.from_dataset`) has since been removed: a training run is started
  from a completed export, so use
  `client.training.create(dataset, export, pipeline_type=…, name=…)`.

- **`pictograph.aio.pipelines` moved the same way**: `AsyncImages
  .upload_from_directory` and `AsyncAnnotations.import_coco` /
  `.import_pascal_voc` / `.import_yolo`. The poll-bound and Pillow-bound flows
  stay sync-only, as before - concurrency would not shorten them.

- **Report and failure types travelled with their methods**, and are now
  top-level exports so they have a stable home: `UploadReport` / `UploadFailure`
  / `AugmentReport` / `AugmentFailure` / `TileReport` / `TileFailure` (in
  `pictograph.resources.images`), `AnnotateReport` / `AnnotationFailure` /
  `AnnotateMode` (`…auto_annotate`), `AnnotationImportReport` /
  `AnnotationImportFailure` (`…annotations`). `from pictograph import
  UploadReport` now works.

### Removed
- **`pictograph.workflows` is gone.** It existed only as the deprecated 1.7.0
  alias of `pipelines`; with `pipelines` gone it had nothing to alias.
  **"Workflow" now means exactly one thing in this SDK: `client.workflows`, the
  node-graph DAG resource.**

### Notes
- Agent tool NAMES are unchanged (37 tools, same set). They are a flat
  agent-facing vocabulary, not a Python call path, and renaming them would
  invalidate every agent prompt for no gain. Only the handlers were re-pointed.
- The bundled skill's `references/pipelines.md` is now
  `references/bulk-operations.md`.

## [1.69.13]

### Changed
- **Three overlapping hardware arguments collapsed into ONE `device=`, present
  and identical on BOTH loaders.** `get_model` used to take `device` +
  `accelerate` + `providers`; `load_model` took two of the three and **not**
  `device`, so a local `.pth` could not be pinned to a GPU at all. They are now
  twins:

  ```python
  get_model(name_or_id, *, task, format, precision, target, api_key, client,
            confidence, device="auto", cache_dir)
  load_model(weights, config, *, task, format, confidence, device="auto", cache_dir)
  ```

  `device=` names the **hardware** - `"auto"` (default), `"cpu"`, `"cuda"` /
  `"cuda:1"`, `"mps"` - and `format=` names the **weights**. The execution
  provider is derived from the pair and is never named by the caller: `mps`
  reaches torch's MPS backend for a `.pth` and CoreML for an `.onnx`, `cuda`
  reaches TensorRT for an `.engine` and CUDA for an `.onnx`.
  `DEVICES_BY_RUNTIME` in `pictograph.inference.runtime` is that mapping as one
  readable table, and a pair that does not exist (a `.engine` on the CPU) raises
  naming what the format CAN run on, before any download or runtime import.

- **A named device is honoured or RAISES, naming what is available.** Never a
  silent fall back to CPU, which is the difference between "cuda was ignored" and
  "you have no CUDA". Three gates enforce it: the format/device pair must exist,
  the runtime must actually have the hardware, and the built ONNX session must
  have KEPT the provider it registered. `device="auto"` is the sole exception and
  always was - it never promised particular hardware, so it degrades down the
  measured ladder and warns.

- `model.device` continues to report what **actually ran**, measured not
  requested, and may be more specific than the request: `device="mps"` on an
  `.onnx` reports `"coreml"`.

### Added
- **`load_model` now loads all five formats**, including the two native
  containers. It previously refused `pytorch` and `safetensors` on the grounds
  that rebuilding a checkpoint needs the model record - it does not, it needs the
  task, architecture and training config, and a pipeline writes all three into
  the `config.json` that ships beside the weights. The offline rebuild shares
  every line with the online one, so a `.pth` cannot behave differently depending
  on which loader it came through.
- **`device="cuda:N"` is honoured everywhere**, not just by torch. ONNX Runtime
  gets `device_id`; the TensorRT session deserializes the plan, creates its
  context, allocates its stream and every buffer inside one `torch.cuda.device(N)`
  scope rather than trusting the ambient current device. Without this a multi-GPU
  box silently ran everything on GPU 0.

### Fixed
- A checkpoint nested under `model_state_dict` (or `state_dict`) is now unwrapped
  like one nested under `model`. The classification pipeline writes the published
  `checkpoint_best_*.pth` bare and its resumable checkpoint nested, and the
  filename does not distinguish them - so handing `load_model` the wrong one
  failed the strict load with a wall of "Missing key(s)" naming every layer in
  the backbone, which says nothing about the container.

### Removed
- **`accelerate=` is gone**, and no behaviour went with it. `"off"` is now
  `device="cpu"` (same CPU-only pin, same instant load), and `"max"`'s entire
  measured effect - paying CoreML's slow session build for a slow-to-build
  architecture, 31.7 s cold for RF-DETR - is what naming `device="mps"` does.
  `device="auto"` still skips that build, because a 31 s pause nobody asked for
  reads as a hang. Steady-state throughput is unchanged in every configuration.
- **`providers=` is gone from both public loaders.** Everything it was needed for
  is now expressible as a device, including per-GPU selection. It survives as a
  private argument used by `benchmarks/inference_bench.py` to pin one exact
  provider configuration per row, which is how the measured tables in
  `runtime.py` are produced. `model.providers` is untouched: it reports what ran.
- `Accelerate` and `ProviderSpec` are no longer exported. `Device` and `DEVICES`
  are, from both `pictograph` and `pictograph.inference`.

## [1.69.11]

### Removed
- **`full_pipeline` is gone - entirely, not deprecated.** The one call that
  chained upload, auto-annotate and train has been deleted, along with
  `PipelineReport`, its agent-registry tool and its bundled-skill wrapper. The
  agent registry goes **38 -> 37 tools**.

  It was the wrong shape: it fed unreviewed auto-annotations straight into a
  training run with no config attached, so a caller could neither see nor correct
  any stage. **The primitives stay** - upload, auto-annotate and train are called
  explicitly, in order, with each report inspected before the next step. Do not
  re-add a function (or an agent tool) that composes them for you.

- **Active-learning uncertainty ranking is gone - entirely.** Removed
  `rank_dataset_by_uncertainty` and its async twin, the `ActiveLearningReport` /
  `UncertaintyFailure` types, the `pictograph.metrics` uncertainty scorer
  (`rank_by_uncertainty`, `UncertaintyScore`, `score_image`, `image_uncertainty`,
  `UncertaintyMethod`), and the `pictograph metrics rank` CLI subcommand. It was
  never an agent tool, so the registry is unaffected by this half.

  **`evaluate_detections` is a separate feature and stays**, as does
  `pictograph metrics evaluate` (offline P/R/F1/mAP, useful in CI). Unrelated:
  the active-learning *filter* on the images and datasets API
  (`min_confidence_lt`, `model_confidence`) is an API capability and is untouched.

### Changed
- `pictograph.workflows` remains a deprecated alias of `pictograph.pipelines`; it
  simply no longer re-exports the removed names.

## [1.69.10]

### Changed
- **`pip install "pictograph[inference]"` is now the WHOLE install for every
  weights format it covers.** `format="onnx"`, `"pytorch"` and `"safetensors"`
  all run, for all six model families, with no second `pip` command - the extra
  gained `torch`, `torchvision`, `safetensors` and a pinned
  `segmentation-models-pytorch==0.5.0` alongside the ONNX stack.

  It replaces a real defect, not a rough edge: the app's "Install Pictograph"
  block printed our install line and then, beneath it,
  `pip install torch segmentation-models-pytorch` - or, for a YOLOX model, a
  `pip install git+…/YOLOX.git --no-deps` plus five transitive packages. Those
  are our own undeclared dependencies, handed to the reader as homework. The rule
  now, enforced by a test: anything a published model needs in order to load is
  **declared by an extra or vendored into the wheel**, and the only install
  command this SDK prints - in a docstring, an `ImportError`, the docs or the app
  - names `pictograph`.

- **`segmentation-models-pytorch` is PINNED, not floored**, to the same `0.5.0`
  the training image pins. It DEFINES AN ARCHITECTURE: a `.pth` is a map from
  layer NAME to tensor, strict-loaded into `smp.Segformer(…)` / `smp.Unet(…)`
  rebuilt from that code. A floor would let pip install a release that renames or
  re-nests one submodule, and the load then fails - or succeeds against a
  silently different block. Bump only in lockstep with the training image.

- `[executorch]` and `[tensorrt]` stay separate extras, and their hints now read
  `pictograph[inference,executorch]` / `pictograph[inference,tensorrt]`. Neither
  can be folded in: ExecuTorch pins one exact `torch` minor (`>=2.12,<2.13`),
  which would impose that torch on every ONNX-only user, and `tensorrt` on PyPI
  is a CUDA-only meta package with no macOS or CPU distribution at all - folding
  it in would make `pip install "pictograph[inference]"` fail outright on any
  machine without a CUDA stack.

- The `[torch]` extra is unchanged and stays, for `Datasets.as_pytorch()` - a
  training-loop user should not have to install an ONNX runtime, OpenCV and
  scikit-image to feed a DataLoader. `[inference]` is a strict superset of it.

### Added
- **The YOLOX architecture is vendored** into `pictograph.inference._yolox`
  (Apache-2.0, `LICENSE` + `NOTICE` shipped in the wheel), from commit
  `6ddff4824372906469a7fae2dc3206c7aa4bbaee` - the one the training service pins,
  i.e. the architecture our published weights encode. Same treatment RF-DETR got
  in 1.69.8, and for a harder reason: `yolox` could not have been declared as a
  dependency even if we wanted to. Measured, in a clean venv:

  * PyPI's only release is **0.3.0 (2022-04-22), sdist only**, and its `setup.py`
    compiles a C++ extension behind
    `assert TORCH_AVAILABLE, "torch is required for pre-compiling ops"`. pip
    generates a dependency's metadata *before* it installs torch, so
    `pip install yolox` aborts with exactly that AssertionError.
  * 0.3.0 is a **different architecture** from the pinned commit - they differ
    inside `yolo_head.py`, including `decode_outputs`.
  * Its `install_requires` would add `onnx-simplifier==0.4.10`, `pycocotools`,
    `tensorboard`, `thop`, `ninja` and - worst - `opencv_python`, a SECOND
    provider of the top-level `cv2` package beside our `opencv-python-headless`.
  * A `git+https://…@<sha>` direct reference is rejected by PyPI at upload
    (PEP 440), so a published wheel cannot carry one.

  1,251 of the upstream package's 7,691 Python lines are vendored (16.3%) plus a
  127-line `_upstream_utils.py` gathering the four `yolox.utils` helpers
  `yolo_head.py` imports; `loguru` is replaced by the stdlib logger so it is not
  a runtime dependency for two unreachable log calls. Verified against the real
  upstream source: identical `state_dict()` keys and shapes for all six sizes,
  upstream weights strict-load into the vendored module, and outputs are exactly
  equal (`max|Δ| = 0.0`). See `tests/unit/test_yolox_vendored.py`.

## [1.69.8]

### Removed
- **`rfdetr` is no longer needed to run RF-DETR weights - the architecture is
  vendored.** Loading a `format="pytorch"` / `"safetensors"` RF-DETR model used
  to require `pip install rfdetr`, which also pulls **`transformers>=5.1,<6`**
  and **`supervision`** into the environment - a hard, narrow version range on
  one of the most conflict-prone packages in the ecosystem, imposed on a user who
  only wanted to run weights we had already given them. All three are gone;
  `pictograph` alone now rebuilds and runs the model.

  The dependency existed to do exactly one thing: reconstruct the `nn.Module` so
  a state dict can load into it. That subset of RF-DETR **1.8.3** - backbone,
  transformer, heads, weight loader, postprocessor - now lives in
  `pictograph.inference._rfdetr`, under Apache-2.0 with attribution in its
  `NOTICE`. Training, export, LoRA and the platform integrations are not
  vendored; a checkpoint reload never reaches them. The `transformers` base
  classes the DINOv2 backbone inherited from are reimplemented in `_compat.py`.

  **1.8.3 is deliberate and pinned**: it is the version the training image
  installs, so the shipped weights and the architecture that rebuilds them come
  from the same release. A test fails if the two ever drift.

  Verified against real trained RF-DETR detection, segmentation and keypoint
  models rebuilt from their published artifacts: the vendored architecture
  reproduces stock `rfdetr` 1.8.3 **to float noise** (identical detection counts
  and classes, ≤0.004 px geometry, 0.000 confidence, byte-identical masks), and
  its delta against each model's ONNX twin is unchanged from what the `rfdetr`
  package itself produced.

### Fixed
- `pip install "pictograph[torch]"` did not actually install everything a native
  PyTorch model needs: **`torchvision`** was a hard requirement of the
  classification path but went undeclared. It is now part of the extra.

## [1.69.7]

### Changed
- **The model-loading API is unified on ONE argument: `format=`.** `get_model`
  and `load_model` now both take the same `format`, typed as a `Literal` so a
  wrong value is a type error rather than a runtime surprise. It names the
  **weights file**, and the runtime that executes it is derived from that instead
  of being asked for separately:

  | `format=` | file | runtime |
  |---|---|---|
  | `pytorch` | `.pth` | `pytorch` |
  | `safetensors` | `.safetensors` | `pytorch` |
  | `pytorch_engine` | `.pte` | `executorch` |
  | `onnx` (default) | `.onnx` | `onnxruntime` |
  | `tensorrt_engine` | `.engine` | `tensorrt` |

  Two selectors are two sources of truth that can disagree; one cannot. A
  `.engine` is executed by TensorRT and by nothing else, so `runtime=` was always
  a restatement of the artifact choice. `Runtime` remains the vocabulary of what
  RAN (`model.backend`), which is a report, not a request.

  `load_model` keeps dispatching on the weights SUFFIX; `format=` is now the
  explicit override for a renamed file, replacing `runtime=` in that role.

- **`get_model(format=…)` distinguishes the two native containers**, which
  `load_pytorch` conflated behind a preference. `format="safetensors"` fetches
  the parity-gated artifact (published only after a publish-blocking comparison
  against that version's ONNX - the one to reach for, and the only native form
  the keypoint family publishes); `format="pytorch"` fetches the raw training
  checkpoint. **Neither is ever substituted for the other**: a model that does not
  publish the requested container is refused with a `ConflictError` naming the
  formats it DOES publish, read off its own files manifest.

- `client.models.load(…)` gained the same `format=` (plus `precision`, `target`
  and `device`), so the client-bound loader is a true twin of `get_model` rather
  than an ONNX-only subset. Both now run through one shared implementation.

- `pictograph models download --format` accepts all five of the download route's
  formats (`onnx`, `pytorch`, `safetensors`, `pte`, `engine`). It previously
  refused `safetensors`/`pte`/`engine` client-side on models that publish them.

### Removed
- **`load_pytorch` - top-level and `client.models.load_pytorch`.** Replaced by
  `get_model(name, format="pytorch")` / `format="safetensors"` and the
  client-bound `client.models.load(name, format=…)`. A breaking change, shipped
  in place: the SDK is early and has no external consumers.
- **`runtime=` on `get_model` and `load_model`.** Superseded by `format=`; pass
  the format and the runtime follows.

### Notes
- `Models.download(format=…)` is unchanged and still speaks the DOWNLOAD ROUTE's
  vocabulary (`pte` / `engine`), because it is a raw byte fetch that mirrors
  `GET /models/{id}/download?format=` one-for-one. The correspondence between the
  two vocabularies is documented on the method and translated in exactly one
  place (`inference.runtime.wire_format`).
- The wheel's own docs (README + the bundled `pictograph-cv` skill) are now
  hard-gated against the loader signature and the `format=` vocabulary.

## [1.69.6]

### Changed
- **Internal infrastructure names removed from user-facing documentation.**
  Docstrings, Pydantic `Field(description=…)` (which becomes the JSON Schema
  agent tool-callers read), tool descriptions, CLI `--help` text and two
  exception messages named vendors and platforms a caller cannot act on. They now
  describe the observable behaviour instead ("a signed storage URL", "object
  storage", "the training service").

  Behaviour a caller must reason about was **kept and, in places, sharpened**:
  - `Exports.download` / `Models.download` / `Datasets.download` still state that
    the transfer is a **second request to a different host** carrying **no SDK
    credentials** (the authorization is the signature in the URL) - the fact that
    matters when a download fails while API calls succeed.
  - `Images.upload` still states that bytes go **straight to object storage,
    never relayed through the Pictograph API**, which is why the 10 MB
    request-body limit does not apply.
  - `Datasets.freeze` / `restore` still document the cold-storage trade-off
    (discounted quota, paused byte-heavy operations, 90-day minimum-duration
    charge on early restore) without naming the storage tier's product name.
  - Wire field names the API actually sends are unchanged - renaming them would
    break the contract and make the docs wrong.

  A build-failing guard keeps this true: it scans every docstring, `description=`
  and `help=` string in the package on each run.

### Fixed
- The `cancel_training` agent tool description claimed it "refunds remaining GPU
  minutes". Training is **charge-on-success**: a run cancelled before completion
  is never charged, so there is nothing to refund. The description now matches
  `Training.cancel`.

## [1.69.5]

### Changed
- **Bundled `pictograph-cv` Claude Skill rewritten against the shipped SDK.** The
  skill had drifted well behind the package it documents. Corrected, not merely
  reworded:
  - **Local inference was entirely missing.** `get_model` / `load_model` /
    `load_pytorch`, the five task-typed model and result classes, the four
    runtimes and their artifacts, and the suffix dispatch on `.onnx` / `.pte` /
    `.engine` now have their own reference - including that a TensorRT plan is
    bound to one GPU architecture and is not portable.
  - **`DeploymentClient.infer()` returns the five typed result classes**, not a
    dict. Nothing now suggests `result["predictions"]`.
  - **Compute credit is USD-denominated.** The old integer-"credits" cost table
    was fabricated; estimation now reads `total_micro_usd` / `remaining_micro_usd`,
    and `PaymentRequiredError` is documented with its real attributes
    (`credit_cost` / `credits_remaining` / `unit` / `upgrade_url`) rather than the
    non-existent `required` / `remaining`.
  - **Removed the "training needs at least 5 images" rule**, which does not exist.
  - Added the previously undocumented bulk helpers (tiling, augmentation,
    COCO/YOLO/Pascal-VOC annotation import), model-assisted batch labelling
    (`model_id=`), SAHI, `quote()`, model versions and precision, and the `obb`
    (`oriented_box`) geometry.
  - The five wrapper scripts now report a failure as JSON on stderr instead of a
    traceback, cover all twelve export formats, price the images they would
    actually annotate, and use the real terminal-status values.

### Added
- Skill tests resolve the documentation against the installed SDK: every wrapper
  is executed, every documented SDK call is bound against its real signature, and
  every `pictograph <group> <command>` is checked against the CLI registry.

## [1.69.1]

### Fixed
- **`load_pytorch` now prefers the GATED native artifact (safetensors) over the
  `.pth` checkpoint.** Safetensors is published only after a publish-blocking
  parity gate has compared it against that version's ONNX graph; the `.pth` has
  never been gated. Until this release `rfdetr_detection` glob-picked its `.pth`
  out of the training output directory (the best/EMA epoch) while exporting its
  ONNX from the live post-train model, so the two published artifacts of one
  model version encoded **different models** - and `load_pytorch(m)` returned
  materially different predictions than `get_model(m)`, breaking the documented
  interchangeability promise.

  Measured on real published pairs (three real dataset images each, detection
  rule, non-vacuous survivor counts):

  | model | `.pth` vs its own ONNX | safetensors vs the same ONNX |
  |---|---|---|
  | RF-DETR Nano | 5.4e-03 / **0.630** (3 / 13 survivors) | 3.3e-04 / 2.4e-04 - PASS |
  | RF-DETR Medium | **0.630 / 0.822 / 0.945** (24 / 37 / 25 survivors) | no safetensors published |
  | RF-DETR Large | **3.682 / 2.991 / 0.633** (2 / 1 / 15 survivors) | 6.6e-04 / 8.5e-04 - PASS |

  The training pipeline is fixed separately so new runs publish a `.pth` written
  from the export target itself, but every model trained before that fix still has
  the wrong `.pth` in storage - this preference repairs all of them with no change
  to a single stored artifact. A version that publishes no safetensors (its gate
  failed) still falls back to the `.pth`, and a `.pth` already in the local cache
  is still honoured, so upgrading forces no re-download.

## [1.69.0]

### Added
- **Two new inference runtimes - ExecuTorch (`.pte`) and TensorRT (`.engine`)**,
  bringing local inference to **four**: `pytorch | executorch | onnxruntime |
  tensorrt`. All four return the SAME five task model classes and the SAME five
  result classes; there is no `ExecuTorchDetectionModel`.
- **`load_model` dispatches on the weights suffix** (`.onnx` / `.pte` /
  `.engine` / `.plan`), so the offline call stays
  `load_model(weights, config, task=…)` for every runtime and the artifact
  remains self-describing.
- **`get_model(runtime=…, precision=…, target=…)`** selects which artifact to
  fetch. `runtime="tensorrt"` defaults `target` to **this machine's** GPU
  architecture, since any other engine is a file that provably cannot load here.
- **`pictograph.Runtime`** and **`pictograph.RUNTIMES`** - the runtime type and
  the ordered vocabulary, both exported top-level for callers that render or
  validate it.
- New optional extras `[executorch]` and `[tensorrt]`. The `executorch` extra
  declares `torch>=2.9` because ExecuTorch's prebuilt runtime is compiled against
  a specific torch ABI, and a mismatch otherwise surfaces as a missing C++ symbol
  at `dlopen` time rather than as a resolver conflict.
- The files manifest (`ModelFileEntry`) gained five additive artifact fields -
  `runtime`, `precision`, `target_key`, `toolchain_version`, `stale`,
  `artifact_id`. Older API versions that omit them still parse.

### Changed
- **`InferenceResult.backend` now names the RUNTIME**: `"onnx"` ->
  `"onnxruntime"` and `"torch"` -> `"pytorch"`, joined by `"executorch"` and
  `"tensorrt"`. The value is the one vocabulary the API, the SDK and the model
  card all match on; three spellings of one concept is how three surfaces stop
  agreeing.
- `.providers` is documented per runtime - ORT providers, a `.pte`'s delegate
  backends, a plan's TensorRT version and GPU target, empty on `pytorch`.
- ExecuTorch and TensorRT read their input shape off the loaded ARTIFACT rather
  than the config, which is strictly more authoritative for an AOT-compiled
  program.
- README's local-inference section rewritten; it still documented the
  `LocalModel` / `PyTorchModel` classes that 1.68 replaced with the task-typed
  ones.

### Notes
- **A mismatched `.engine` is refused BEFORE deserialization**, with a message
  naming the built-for target and the detected device, matching the API's own
  wording verbatim.
- **TensorRT remains OUT of the automatic `accelerate` ladder** (the measured
  do-not-undo in `inference/runtime.py`). An ahead-of-time `.engine` removes the
  first-inference build cost that made ORT's `TensorrtExecutionProvider` a bad
  default; it does not make that provider a good automatic choice.
- Parity is by CONSTRUCTION: the three graph runtimes share one preprocessing /
  decode / postprocess implementation and substitute only the forward pass.
  Measured ONNX Runtime and ExecuTorch agreement across all five task families at
  native and non-native input sizes: max |Δconfidence| 3.6e-07, max |Δcoordinate|
  8.1e-05 px, identical class sets.

## [1.67.14]

### Added
- **`load_model(weights, config)`** - load a trained model from its local files
  (ONNX weights + the `config.json` a training pipeline writes) fully offline, no
  API call. The offline twin of `get_model()`; returns the same `LocalModel`.
- **`InferenceModel`** protocol - the common, swappable interface of `LocalModel`
  (ONNX) and `PyTorchModel` (native), so inference code can target either backend.
- Top-level exports: `PyTorchModel`, `load_model`, `InferenceModel`.
- README "Run your trained models" section covering every path (hosted, local
  ONNX, offline config.json, native PyTorch) and their equivalence.

### Changed
- **`LocalModel` is now thread-safe** - `predict()` / `predict_batch()` serialize
  internally (the ONNX wrappers keep per-call state such as the resize ratio),
  fixing a race when sharing one model across threads.
- **`LocalModel.predict_batch()` now batches** YOLOX detection into a single ONNX
  run (a real throughput win) instead of looping `predict()`.
- `LocalModel`, `PyTorchModel` and `DeploymentClient` gained `close()` plus
  context-manager support to release native sessions and connections promptly.
- `DeploymentClient` reuses one pooled, keep-alive HTTP client instead of opening
  a fresh connection per `infer()` call.

## [1.67.13]

### Changed
- Clarified `get_model()`'s authentication docstring: the API key comes from the
  `api_key=` argument or the `PICTOGRAPH_API_KEY` environment variable. The
  `pictograph login` config file is read by the CLI only, not by `Client`.
- Inference input-shape introspection now logs through the `pictograph.inference`
  logger instead of printing to stdout.

### Added
- `NearDuplicatesResult` is now exported at the top level
  (`from pictograph import NearDuplicatesResult`) so `datasets.near_duplicates()`
  return values can be annotated directly.
- `CHANGELOG.md`, `CONTRIBUTING.md` and `SECURITY.md`.

## [1.67.12]

- Baseline for this changelog. For releases prior to this version, see the
  commit history and the [PyPI release history](https://pypi.org/project/pictograph/#history).
