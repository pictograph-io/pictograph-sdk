"""Unit tests for the bulk-vs-interactive load benchmark logic.

The harness lives under ``benchmarks/`` (operator tooling, not shipped in the
wheel), so it's loaded by file path - the same way the service-side openapi drift
test loads its generator. Only the pure, network-free logic is exercised here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_HARNESS = Path(__file__).resolve().parents[2] / "benchmarks" / "load_bench.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("load_bench", _HARNESS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass annotation resolution (which looks up
    # sys.modules[cls.__module__]) works under `from __future__ import annotations`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summarize_empty() -> None:
    bench = _load()
    s = bench.summarize([])
    assert s["count"] == 0
    assert s["p95"] == 0.0


def test_summarize_percentiles() -> None:
    bench = _load()
    s = bench.summarize([float(x) for x in range(1, 101)])  # 1..100 ms
    assert s["count"] == 100
    assert s["min"] == 1.0
    assert s["max"] == 100.0
    assert s["p50"] == 50.0 or s["p50"] == 51.0  # nearest-rank lands mid-range
    assert s["p95"] >= 95.0
    assert s["p99"] >= 99.0
    assert 50.0 <= s["mean"] <= 51.0


def test_run_bench_with_mock_calls() -> None:
    bench = _load()
    calls = {"interactive": 0, "bulk": 0}

    def interactive() -> None:
        calls["interactive"] += 1

    def bulk() -> None:
        calls["bulk"] += 1

    result = bench.run_bench(
        interactive_call=interactive,
        bulk_call=bulk,
        interactive_n=10,
        storm_n=20,
        storm_concurrency=5,
        interactive_concurrency=2,
    )
    # interactive runs in both phases (baseline + under-load).
    assert result.baseline["count"] == 10
    assert result.under_load["count"] == 10
    assert calls["interactive"] == 20
    assert calls["bulk"] == 20
    assert result.storm_count == 20
    assert result.storm_errors == 0
    assert result.interactive_errors == 0


def test_run_bench_records_errors() -> None:
    bench = _load()

    def boom() -> None:
        raise RuntimeError("server overloaded")

    def ok() -> None:
        return None

    result = bench.run_bench(
        interactive_call=ok,
        bulk_call=boom,
        interactive_n=5,
        storm_n=8,
        storm_concurrency=4,
    )
    assert result.storm_errors == 8  # every bulk call failed, recorded not raised
    assert result.interactive_errors == 0


def test_format_report_contains_delta() -> None:
    bench = _load()
    result = bench.run_bench(
        interactive_call=lambda: None,
        bulk_call=lambda: None,
        interactive_n=4,
        storm_n=4,
        storm_concurrency=2,
    )
    report = bench.format_report(result)
    assert "interactive p95 delta" in report
    assert "baseline" in report
