#!/usr/bin/env python3
"""Bulk-vs-interactive load benchmark.

Quantifies the load-isolation acceptance criterion directly: *"a benchmarked
bulk-download storm keeps interactive p95 (dataset list / grid) within a small
delta of baseline."* It measures interactive request latency (1) with the API idle
and (2) while a configurable storm of concurrent bulk image downloads runs, then
reports both distributions and the p95 delta.

Operator-run (needs a real key + a dataset with images - never runs in CI):

    PICTOGRAPH_API_KEY=pk_live_... python benchmarks/load_bench.py \
        --dataset road-signs --storm 200 --interactive 50

Point at staging / a specific revision with ``--base-url``. Use a disposable,
synthetic ``e2e-`` dataset - NEVER a production dataset.
"""

from __future__ import annotations

import argparse
import os
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def _percentile(sorted_ms: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted list (pct in [0, 100])."""
    if not sorted_ms:
        return 0.0
    if len(sorted_ms) == 1:
        return sorted_ms[0]
    rank = max(0, min(len(sorted_ms) - 1, round(pct / 100 * (len(sorted_ms) - 1))))
    return sorted_ms[rank]


def summarize(samples_ms: list[float]) -> dict[str, float]:
    """Latency summary (milliseconds) for a list of per-call durations."""
    if not samples_ms:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    s = sorted(samples_ms)
    return {
        "count": len(s),
        "min": s[0],
        "p50": _percentile(s, 50),
        "p95": _percentile(s, 95),
        "p99": _percentile(s, 99),
        "max": s[-1],
        "mean": statistics.fmean(s),
    }


@dataclass
class BenchResult:
    baseline: dict[str, float]
    under_load: dict[str, float]
    storm_count: int
    storm_errors: int
    interactive_errors: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def p95_delta_ms(self) -> float:
        return self.under_load["p95"] - self.baseline["p95"]

    @property
    def p95_delta_pct(self) -> float:
        base = self.baseline["p95"] or 1.0
        return 100.0 * self.p95_delta_ms / base


def _timed(fn: Callable[[], Any]) -> tuple[float, bool]:
    """Run ``fn``; return (elapsed_ms, ok)."""
    start = time.perf_counter()
    try:
        fn()
        ok = True
    except Exception:
        ok = False
    return (time.perf_counter() - start) * 1000.0, ok


def run_bench(
    *,
    interactive_call: Callable[[], Any],
    bulk_call: Callable[[], Any],
    interactive_n: int = 50,
    storm_n: int = 200,
    storm_concurrency: int = 50,
    interactive_concurrency: int = 4,
) -> BenchResult:
    """Measure interactive latency at idle, then under a bulk-download storm.

    ``interactive_call`` and ``bulk_call`` are zero-arg callables (closures over
    a Client). Kept injectable so the runner is unit-testable without a network.
    """
    # Phase 1 - baseline interactive latency, backend otherwise idle.
    baseline_ms: list[float] = []
    interactive_errors = 0
    with ThreadPoolExecutor(max_workers=interactive_concurrency) as pool:
        for ms, ok in pool.map(lambda _: _timed(interactive_call), range(interactive_n)):
            baseline_ms.append(ms)
            interactive_errors += 0 if ok else 1

    # Phase 2 - fire the bulk storm in the background while re-measuring
    # interactive latency. The storm and the interactive probes run on separate
    # pools so the client-side isn't the bottleneck being measured.
    under_load_ms: list[float] = []
    storm_errors = 0
    with (
        ThreadPoolExecutor(max_workers=storm_concurrency) as storm_pool,
        ThreadPoolExecutor(max_workers=interactive_concurrency) as probe_pool,
    ):
        storm_futures = [storm_pool.submit(lambda: _timed(bulk_call)) for _ in range(storm_n)]
        probe_futures = [
            probe_pool.submit(lambda: _timed(interactive_call)) for _ in range(interactive_n)
        ]
        for fut in as_completed(probe_futures):
            ms, ok = fut.result()
            under_load_ms.append(ms)
            interactive_errors += 0 if ok else 1
        for fut in as_completed(storm_futures):
            _, ok = fut.result()
            storm_errors += 0 if ok else 1

    return BenchResult(
        baseline=summarize(baseline_ms),
        under_load=summarize(under_load_ms),
        storm_count=storm_n,
        storm_errors=storm_errors,
        interactive_errors=interactive_errors,
    )


def format_report(result: BenchResult) -> str:
    b, u = result.baseline, result.under_load
    lines = [
        "Bulk-vs-interactive load benchmark",
        "=" * 42,
        f"storm downloads : {result.storm_count} ({result.storm_errors} errors)",
        f"interactive errs: {result.interactive_errors}",
        "",
        f"{'metric':<8}{'baseline':>12}{'under load':>14}",
        f"{'p50':<8}{b['p50']:>11.1f}m{u['p50']:>13.1f}m",
        f"{'p95':<8}{b['p95']:>11.1f}m{u['p95']:>13.1f}m",
        f"{'p99':<8}{b['p99']:>11.1f}m{u['p99']:>13.1f}m",
        f"{'max':<8}{b['max']:>11.1f}m{u['max']:>13.1f}m",
        "",
        f"interactive p95 delta: {result.p95_delta_ms:+.1f} ms ({result.p95_delta_pct:+.0f}%)",
    ]
    return "\n".join(lines)


def _build_calls(args: argparse.Namespace) -> tuple[Callable[[], Any], Callable[[], Any]]:
    from pictograph import Client  # deferred so --help works without the dep

    client = Client(api_key=args.api_key, base_url=args.base_url)
    dataset = client.datasets.get(args.dataset, include_images=True)
    image_ids = [img.id for img in (dataset.images or [])][: args.storm]
    if not image_ids:
        raise SystemExit(f"Dataset {args.dataset!r} has no images to download.")

    tmp = Path(tempfile.mkdtemp(prefix="pictograph-bench-"))
    counter = {"i": 0}

    def interactive_call() -> None:
        client.datasets.list(limit=20)

    def bulk_call() -> None:
        idx = counter["i"] % len(image_ids)
        counter["i"] += 1
        client.images.download(image_ids[idx], output_path=tmp / f"{idx}.bin")

    return interactive_call, bulk_call


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Disposable e2e- dataset with images.")
    parser.add_argument("--storm", type=int, default=200, help="Concurrent bulk downloads to fire.")
    parser.add_argument("--interactive", type=int, default=50, help="Interactive probe calls.")
    parser.add_argument("--storm-concurrency", type=int, default=50)
    parser.add_argument("--api-key", default=os.getenv("PICTOGRAPH_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("PICTOGRAPH_BASE_URL"))
    args = parser.parse_args()

    interactive_call, bulk_call = _build_calls(args)
    result = run_bench(
        interactive_call=interactive_call,
        bulk_call=bulk_call,
        interactive_n=args.interactive,
        storm_n=args.storm,
        storm_concurrency=args.storm_concurrency,
    )
    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
