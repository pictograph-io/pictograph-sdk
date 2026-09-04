#!/usr/bin/env python3
"""Async-vs-sync SDK fan-out benchmark.

Quantifies the headline benefit of :class:`pictograph.AsyncClient`: issuing a
fan-out of independent API calls concurrently on one HTTP/2 connection pool is
much faster than the sync client's one-at-a-time loop, and competitive with the
sync client's thread-pool. It fires N independent ``datasets.list()`` calls three
ways and reports the wall-clock for each:

1. **sync serial** - a plain ``for`` loop on :class:`pictograph.Client`.
2. **sync threads** - a ``ThreadPoolExecutor`` over the sync client.
3. **async gather** - ``asyncio.gather`` over :class:`pictograph.AsyncClient`.

Operator-run (needs a real key; hits the live backend - never runs in CI):

    PICTOGRAPH_API_KEY=pk_live_... python benchmarks/async_bench.py --calls 40 --workers 8

Point at staging / a specific revision with ``--base-url``. The deterministic
"is it actually concurrent" property is unit-tested in
``tests/unit/aio/test_async_concurrency.py`` (no key, no network).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor


def _summary(label: str, elapsed: float, calls: int) -> str:
    rps = calls / elapsed if elapsed > 0 else float("inf")
    return f"{label:<14} {elapsed * 1000:8.1f} ms   {rps:7.1f} req/s"


def _bench_sync_serial(api_key: str, base_url: str | None, calls: int) -> float:
    from pictograph import Client

    with Client(api_key=api_key, base_url=base_url) as client:
        start = time.monotonic()
        for _ in range(calls):
            client.datasets.list(limit=1)
        return time.monotonic() - start


def _bench_sync_threads(api_key: str, base_url: str | None, calls: int, workers: int) -> float:
    from pictograph import Client

    with Client(api_key=api_key, base_url=base_url) as client:
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda _: client.datasets.list(limit=1), range(calls)))
        return time.monotonic() - start


async def _bench_async(api_key: str, base_url: str | None, calls: int) -> float:
    from pictograph import AsyncClient

    async with AsyncClient(api_key=api_key, base_url=base_url) as client:
        start = time.monotonic()
        await asyncio.gather(*(client.datasets.list(limit=1) for _ in range(calls)))
        return time.monotonic() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calls", type=int, default=40, help="Independent list() calls to fan out."
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Thread-pool size for the sync-threads run."
    )
    parser.add_argument(
        "--base-url", default=None, help="Override the API base URL (staging / a revision)."
    )
    args = parser.parse_args()

    api_key = os.environ.get("PICTOGRAPH_API_KEY")
    if not api_key:
        parser.error("Set PICTOGRAPH_API_KEY (use a disposable/test key, not production).")

    print(f"Fanning out {args.calls} datasets.list() calls (threads={args.workers})\n")
    serial = _bench_sync_serial(api_key, args.base_url, args.calls)
    print(_summary("sync serial", serial, args.calls))
    threads = _bench_sync_threads(api_key, args.base_url, args.calls, args.workers)
    print(_summary("sync threads", threads, args.calls))
    gathered = asyncio.run(_bench_async(api_key, args.base_url, args.calls))
    print(_summary("async gather", gathered, args.calls))

    if gathered > 0:
        print(f"\nasync speed-up vs serial: {serial / gathered:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
