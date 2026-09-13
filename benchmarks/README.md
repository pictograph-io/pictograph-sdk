# Benchmarks

Operator-run performance harnesses for the Pictograph REST API / SDK. They need
a real API key + data and hit the live backend, so they are **not** part of the
CI/test gate (the pure logic is unit-tested in `tests/unit/test_load_bench.py`).

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

# Point at a staging / alternate deployment:
python benchmarks/load_bench.py --dataset e2e-bench \
    --base-url https://staging.example.com
```

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
