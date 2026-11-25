# Benchmarks

Operator-run performance harnesses for the Pictograph REST API / SDK. `load_bench.py` and
`async_bench.py` need a real API key and data and hit the live backend; `inference_bench.py` runs
models on your machine. None of them is part of the CI/test gate; their pure logic is unit-tested
(`tests/unit/test_load_bench.py`, `test_inference_bench.py`).

> **Warning:** Always benchmark against a **disposable, synthetic `e2e-`-prefixed
> dataset**, never a production dataset.

## `load_bench.py` - bulk-vs-interactive load

Measures interactive request latency (1) with the backend idle and (2) under a
storm of concurrent bulk image downloads, then reports both distributions and
the interactive **p95 delta**. This is the load-isolation acceptance test made runnable:
*a bulk-download storm should keep interactive p95 within a small delta of
baseline.*

```bash
PICTOGRAPH_API_KEY=pk_live_... python benchmarks/load_bench.py \
    --dataset e2e-bench --storm 200 --interactive 50
```

`--base-url` (or `PICTOGRAPH_BASE_URL`) points it at another API host; the default is
`https://api.pictograph.io`.

Run it **before** and **after** a load-isolation change (a bulk-download thread
pool, a worker-count change, or a server-side concurrency/scaling change) and
compare the p95 delta to confirm the improvement.

## `async_bench.py` - async-vs-sync fan-out

Fires N independent `datasets.list()` calls three ways - sync serial, sync
threads, and `asyncio.gather` over `AsyncClient` - and reports the wall-clock +
req/s for each, plus the async speed-up. Quantifies the headline benefit of the
async client (concurrent fan-out on one HTTP/2 pool).

```bash
PICTOGRAPH_API_KEY=pk_live_... python benchmarks/async_bench.py --calls 40 --workers 8
```

The deterministic "are the requests actually concurrent?" property is unit-tested
(no key, no network) in `tests/unit/aio/test_async_concurrency.py`.

## `inference_bench.py` - local inference runtimes

Measures the SDK's local inference stack (`pictograph.inference`) across model families
(classifier, semantic segmentation, YOLOX, RF-DETR) and runtime configs (ONNX Runtime on CPU,
CoreML, CUDA and TensorRT; PyTorch on CPU, MPS and CUDA). For each cell it reports the cold and warm
model build time and the steady-state p50 / p95 per image. A config the machine cannot run, or a
model that is not on disk, is skipped with a printed reason. These are the numbers in
`pictograph.inference.runtime`'s docstring. No API key: it reads ONNX exports (and `.pth`
checkpoints) you downloaded to `~/.pictograph/models/`, naming each family's file stem in
`$PICTOGRAPH_BENCH_<FAMILY>`.

```bash
python benchmarks/inference_bench.py --quick                       # warmup 1, iters 3
python benchmarks/inference_bench.py --only-configs onnx-cpu,torch-cpu --json out.json
```

Its pure logic is unit-tested in `tests/unit/test_inference_bench.py`.
