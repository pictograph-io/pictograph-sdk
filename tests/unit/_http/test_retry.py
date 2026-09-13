"""Tests for ``pictograph._http.retry``.

Coverage targets:
- Method-safety classification (idempotent vs needs-key).
- Status-code retry classification, including 429's special unconditional case.
- Backoff math (exponential, jitter range, cap).
- ``RetryPolicy.execute`` decision tree across {success, transient status,
  network error} × {idempotent method, POST without key, POST with key} ×
  {within-budget, exhausted-budget}.
- ``Retry-After`` honoured below threshold; surfaced above threshold.

A controlled ``request_fn`` (canned-response queue) and injected ``sleep`` /
``rng`` make every code path deterministic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import httpx
import pytest

from pictograph._http.retry import (
    JITTER_HIGH,
    JITTER_LOW,
    RETRYABLE_STATUS_CODES,
    SAFE_METHODS,
    RetryPolicy,
    is_method_safe_to_retry,
    is_retryable_status,
)
from pictograph.exceptions import NetworkError, RequestTimeoutError

# ───────────── helpers ─────────────


def _resp(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    """Construct a minimal httpx.Response with a status code and headers."""
    return httpx.Response(
        status_code=status,
        headers=headers or {},
        request=httpx.Request("GET", "https://example.test/"),
    )


class _Fixed:
    """Callable that returns a queue of responses or raises queued exceptions.

    Use to deterministically drive ``RetryPolicy.execute``.
    """

    def __init__(self, items: Iterable[httpx.Response | Exception]) -> None:
        self._iter: Iterator[httpx.Response | Exception] = iter(items)
        self.calls = 0

    def __call__(self) -> httpx.Response:
        self.calls += 1
        try:
            item = next(self._iter)
        except StopIteration as e:  # pragma: no cover - would indicate a test bug
            raise AssertionError("request_fn called more times than canned items") from e
        if isinstance(item, Exception):
            raise item
        return item


def _capture_sleep() -> tuple[Callable[[float], None], list[float]]:
    """Return (sleep, calls) - collected sleep durations for assertions."""
    calls: list[float] = []

    def fake(d: float) -> None:
        calls.append(d)

    return fake, calls


def _fixed_rng(value: float = 1.0) -> Callable[[float, float], float]:
    """Return a deterministic rng that emits ``value`` regardless of inputs."""

    def rng(_low: float, _high: float) -> float:
        return value

    return rng


# ───────────── method classification ─────────────


@pytest.mark.parametrize(
    ("method", "has_key", "expected"),
    [
        # Idempotent methods always retryable, key irrelevant.
        ("GET", False, True),
        ("GET", True, True),
        ("HEAD", False, True),
        ("OPTIONS", False, True),
        ("PUT", False, True),
        ("DELETE", False, True),
        # Non-idempotent methods only retryable with a key.
        ("POST", False, False),
        ("POST", True, True),
        ("PATCH", False, False),
        ("PATCH", True, True),
        # Case-insensitive.
        ("post", True, True),
        ("Post", False, False),
        # Unknown methods conservative: treat as POST/PATCH.
        ("PROPFIND", False, False),
        ("PROPFIND", True, True),
    ],
)
def test_is_method_safe_to_retry(method: str, has_key: bool, expected: bool) -> None:
    assert is_method_safe_to_retry(method, has_idempotency_key=has_key) is expected


def test_safe_methods_set_is_pinned() -> None:
    # Pin the exact set; any change here is intentional and surfaces here first.
    assert SAFE_METHODS == frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})


# ───────────── status classification ─────────────


@pytest.mark.parametrize(
    "status",
    [408, 500, 502, 503, 504],
)
def test_safe_method_retries_5xx_and_408(status: int) -> None:
    assert is_retryable_status(status, safe_to_retry=True) is True


@pytest.mark.parametrize(
    "status",
    [408, 500, 502, 503, 504],
)
def test_unsafe_method_does_not_retry_5xx_or_408(status: int) -> None:
    assert is_retryable_status(status, safe_to_retry=False) is False


def test_status_429_retries_even_for_unsafe_methods() -> None:
    # Server explicitly rejected the request; retry is always safe.
    assert is_retryable_status(429, safe_to_retry=False) is True
    assert is_retryable_status(429, safe_to_retry=True) is True


@pytest.mark.parametrize("status", [200, 201, 204, 301, 302, 400, 401, 403, 404, 409, 422])
def test_non_retryable_statuses(status: int) -> None:
    assert is_retryable_status(status, safe_to_retry=True) is False
    assert is_retryable_status(status, safe_to_retry=False) is False


def test_retryable_status_set_is_pinned() -> None:
    # Pin the exact set; new entries (e.g. 599) require deliberate addition.
    assert RETRYABLE_STATUS_CODES == frozenset({408, 429, 500, 502, 503, 504})


# ───────────── RetryPolicy construction ─────────────


@pytest.mark.parametrize("bad", [-1, -10])
def test_negative_max_retries_rejected(bad: int) -> None:
    with pytest.raises(ValueError, match="max_retries"):
        RetryPolicy(max_retries=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_backoff_base_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="backoff_base"):
        RetryPolicy(backoff_base=bad)


def test_backoff_cap_must_be_at_least_backoff_base() -> None:
    with pytest.raises(ValueError, match="backoff_cap"):
        RetryPolicy(backoff_base=10.0, backoff_cap=5.0)


def test_zero_max_retries_disables_retries() -> None:
    fn = _Fixed([_resp(503)])
    sleep, calls = _capture_sleep()
    policy = RetryPolicy(max_retries=0, sleep=sleep, rng=_fixed_rng())
    response = policy.execute(fn, method="GET", has_idempotency_key=False)
    assert response.status_code == 503
    assert fn.calls == 1
    assert calls == []


# ───────────── compute_backoff ─────────────


def test_compute_backoff_doubles_per_attempt_with_jitter_constant() -> None:
    # rng=1.0 makes the calculation deterministic.
    policy = RetryPolicy(backoff_base=1.0, backoff_cap=100.0, rng=_fixed_rng(1.0))
    assert policy.compute_backoff(0) == 1.0
    assert policy.compute_backoff(1) == 2.0
    assert policy.compute_backoff(2) == 4.0
    assert policy.compute_backoff(3) == 8.0


def test_compute_backoff_caps_at_backoff_cap() -> None:
    policy = RetryPolicy(backoff_base=1.0, backoff_cap=5.0, rng=_fixed_rng(1.0))
    assert policy.compute_backoff(2) == 4.0
    assert policy.compute_backoff(3) == 5.0  # would be 8.0 uncapped
    assert policy.compute_backoff(10) == 5.0


def test_compute_backoff_applies_jitter_multiplicatively() -> None:
    # jitter=0.5 halves; jitter=2.0 doubles.
    policy_half = RetryPolicy(backoff_base=2.0, backoff_cap=100.0, rng=_fixed_rng(0.5))
    policy_double = RetryPolicy(backoff_base=2.0, backoff_cap=100.0, rng=_fixed_rng(2.0))
    assert policy_half.compute_backoff(0) == 1.0
    assert policy_double.compute_backoff(0) == 4.0


def test_compute_backoff_jitter_within_documented_range() -> None:
    # Real rng - assert the jitter multiplier always lands in [JITTER_LOW, JITTER_HIGH]
    # by sampling many times and checking the implied multiplier.
    policy = RetryPolicy(backoff_base=10.0, backoff_cap=10.0)  # capped, so no doubling
    samples = [policy.compute_backoff(0) for _ in range(200)]
    multipliers = [s / 10.0 for s in samples]
    assert all(JITTER_LOW <= m <= JITTER_HIGH for m in multipliers)


def test_compute_backoff_rejects_negative_attempt() -> None:
    policy = RetryPolicy()
    with pytest.raises(ValueError, match="attempt"):
        policy.compute_backoff(-1)


def test_compute_backoff_does_not_overflow_for_huge_attempt() -> None:
    # ``base * (2**attempt)`` coerced the int to float before the cap, so an
    # absurdly large configured max_retries (>= 1024 attempts) raised
    # OverflowError. Clamping the shift keeps it pinned at the cap instead.
    policy = RetryPolicy(backoff_base=1.0, backoff_cap=30.0, rng=_fixed_rng(1.0))
    assert policy.compute_backoff(5000) == 30.0  # raised OverflowError pre-fix


# ───────────── execute: success paths ─────────────


def test_execute_success_first_try_does_not_sleep() -> None:
    fn = _Fixed([_resp(200)])
    sleep, calls = _capture_sleep()
    policy = RetryPolicy(sleep=sleep, rng=_fixed_rng())
    response = policy.execute(fn, method="GET", has_idempotency_key=False)
    assert response.status_code == 200
    assert fn.calls == 1
    assert calls == []


def test_execute_2xx_after_one_5xx_returns_2xx_with_one_sleep() -> None:
    fn = _Fixed([_resp(503), _resp(200)])
    sleep, calls = _capture_sleep()
    policy = RetryPolicy(max_retries=3, sleep=sleep, rng=_fixed_rng(1.0))
    response = policy.execute(fn, method="GET", has_idempotency_key=False)
    assert response.status_code == 200
    assert fn.calls == 2
    assert len(calls) == 1


def test_execute_2xx_after_two_transient_failures() -> None:
    fn = _Fixed([_resp(503), _resp(502), _resp(200)])
    sleep, calls = _capture_sleep()
    policy = RetryPolicy(max_retries=3, backoff_base=1.0, sleep=sleep, rng=_fixed_rng(1.0))
    response = policy.execute(fn, method="GET", has_idempotency_key=False)
    assert response.status_code == 200
    assert fn.calls == 3
    assert calls == [1.0, 2.0]  # exponential


# ───────────── execute: budget exhaustion ─────────────


def test_execute_returns_final_response_when_budget_exhausted() -> None:
    fn = _Fixed([_resp(503), _resp(503), _resp(503), _resp(503)])
    sleep, calls = _capture_sleep()
    policy = RetryPolicy(max_retries=3, sleep=sleep, rng=_fixed_rng())
    response = policy.execute(fn, method="GET", has_idempotency_key=False)
    assert response.status_code == 503
    assert fn.calls == 4  # original + 3 retries
    assert len(calls) == 3  # one sleep per retry, none after the last


# ───────────── execute: network errors ─────────────


def test_execute_retries_network_error_for_idempotent_method() -> None:
    fn = _Fixed([NetworkError("dns"), NetworkError("conn"), _resp(200)])
    sleep, calls = _capture_sleep()
    policy = RetryPolicy(max_retries=3, sleep=sleep, rng=_fixed_rng(1.0))
    response = policy.execute(fn, method="GET", has_idempotency_key=False)
    assert response.status_code == 200
    assert fn.calls == 3
    assert len(calls) == 2


def test_execute_does_not_retry_network_error_on_post_without_key() -> None:
    err = NetworkError("conn")
    fn = _Fixed([err])
    sleep, calls = _capture_sleep()
    policy = RetryPolicy(max_retries=3, sleep=sleep, rng=_fixed_rng())
    with pytest.raises(NetworkError):
        policy.execute(fn, method="POST", has_idempotency_key=False)
    assert fn.calls == 1
    assert calls == []


def test_execute_retries_network_error_on_post_with_idempotency_key() -> None:
    fn = _Fixed([NetworkError("conn"), _resp(200)])
    sleep, calls = _capture_sleep()
    policy = RetryPolicy(max_retries=3, sleep=sleep, rng=_fixed_rng(1.0))
    response = policy.execute(fn, method="POST", has_idempotency_key=True)
    assert response.status_code == 200
    assert fn.calls == 2
    assert len(calls) == 1


def test_execute_retries_request_timeout_subclass_of_network_error() -> None:
    fn = _Fixed([RequestTimeoutError("read timeout"), _resp(200)])
    sleep, calls = _capture_sleep()
    policy = RetryPolicy(max_retries=3, sleep=sleep, rng=_fixed_rng(1.0))
    response = policy.execute(fn, method="GET", has_idempotency_key=False)
    assert response.status_code == 200
    assert fn.calls == 2
    assert len(calls) == 1


def test_execute_re_raises_network_error_after_exhausting_attempts() -> None:
    fn = _Fixed([NetworkError("e1"), NetworkError("e2"), NetworkError("final")])
    sleep, _ = _capture_sleep()
    policy = RetryPolicy(max_retries=2, sleep=sleep, rng=_fixed_rng())
    with pytest.raises(NetworkError) as exc:
        policy.execute(fn, method="GET", has_idempotency_key=False)
    assert "final" in str(exc.value)
    assert fn.calls == 3


# ───────────── execute: 429 special handling ─────────────


def test_execute_retries_429_for_idempotent_method() -> None:
    fn = _Fixed([_resp(429, {"Retry-After": "1"}), _resp(200)])
    sleep, calls = _capture_sleep()
    policy = RetryPolicy(max_retries=3, sleep=sleep, rng=_fixed_rng())
    response = policy.execute(fn, method="GET", has_idempotency_key=False)
    assert response.status_code == 200
    assert calls == [1.0]


def test_execute_retries_429_for_post_without_key() -> None:
    # 429 means "I rejected the request, didn't process it" - always safe to retry.
    # ``Retry-After: 0`` is floored to the exponential backoff (not an immediate,
    # jitter-less retry burst) - see test_execute_429_floors_zero_retry_after.
    fn = _Fixed([_resp(429, {"Retry-After": "0"}), _resp(200)])
    sleep, calls = _capture_sleep()
    policy = RetryPolicy(max_retries=3, backoff_base=1.0, sleep=sleep, rng=_fixed_rng())
    response = policy.execute(fn, method="POST", has_idempotency_key=False)
    assert response.status_code == 200
    assert calls == [1.0]  # floored to compute_backoff(0), not 0.0


def test_execute_429_floors_zero_retry_after() -> None:
    """Regression: a misconfigured server's ``Retry-After: 0`` on a 429 must NOT
    cause a zero-delay retry burst - the honoured delay is floored to the
    exponential backoff so retries stay spaced."""
    fn = _Fixed([_resp(429, {"Retry-After": "0"}), _resp(200)])
    sleep, calls = _capture_sleep()
    policy = RetryPolicy(
        max_retries=3,
        backoff_base=2.0,
        retry_after_autowait_threshold=120.0,
        sleep=sleep,
        rng=_fixed_rng(1.0),
    )
    response = policy.execute(fn, method="GET", has_idempotency_key=False)
    assert response.status_code == 200
    assert calls == [2.0]  # max(0.0, backoff(0)=2.0) - never 0.0

    # A Retry-After ABOVE the backoff is still honoured exactly.
    fn2 = _Fixed([_resp(429, {"Retry-After": "9"}), _resp(200)])
    sleep2, calls2 = _capture_sleep()
    policy2 = RetryPolicy(
        max_retries=3,
        backoff_base=2.0,
        retry_after_autowait_threshold=120.0,
        sleep=sleep2,
        rng=_fixed_rng(1.0),
    )
    policy2.execute(fn2, method="GET", has_idempotency_key=False)
    assert calls2 == [9.0]  # max(9.0, backoff(0)=2.0) == 9.0


def test_execute_429_honours_retry_after_value_below_threshold() -> None:
    fn = _Fixed([_resp(429, {"Retry-After": "5"}), _resp(200)])
    sleep, calls = _capture_sleep()
    policy = RetryPolicy(
        max_retries=3,
        retry_after_autowait_threshold=120.0,
        sleep=sleep,
        rng=_fixed_rng(),
    )
    response = policy.execute(fn, method="GET", has_idempotency_key=False)
    assert response.status_code == 200
    assert calls == [5.0]  # not the exponential backoff


def test_execute_429_retry_after_exactly_at_threshold_is_auto_waited() -> None:
    # Boundary contract: the SDK auto-waits when retry_after <= threshold (it
    # bails only when strictly greater). 120 is a very common server value, and
    # RateLimitError's docstring/.fix now document "<= 120s" to match this.
    fn = _Fixed([_resp(429, {"Retry-After": "120"}), _resp(200)])
    sleep, calls = _capture_sleep()
    policy = RetryPolicy(
        max_retries=3,
        retry_after_autowait_threshold=120.0,
        sleep=sleep,
        rng=_fixed_rng(1.0),
    )
    response = policy.execute(fn, method="GET", has_idempotency_key=False)
    assert response.status_code == 200
    assert calls == [120.0]  # honoured at the exact boundary, not bailed


def test_execute_429_returns_response_when_retry_after_exceeds_threshold() -> None:
    fn = _Fixed([_resp(429, {"Retry-After": "300"})])
    sleep, calls = _capture_sleep()
    policy = RetryPolicy(
        max_retries=3,
        retry_after_autowait_threshold=120.0,
        sleep=sleep,
        rng=_fixed_rng(),
    )
    response = policy.execute(fn, method="GET", has_idempotency_key=False)
    assert response.status_code == 429
    assert fn.calls == 1  # no retry attempted
    assert calls == []


def test_execute_429_without_retry_after_uses_exponential_backoff() -> None:
    fn = _Fixed([_resp(429), _resp(200)])
    sleep, calls = _capture_sleep()
    policy = RetryPolicy(max_retries=3, backoff_base=2.0, sleep=sleep, rng=_fixed_rng(1.0))
    response = policy.execute(fn, method="GET", has_idempotency_key=False)
    assert response.status_code == 200
    assert calls == [2.0]


# ───────────── execute: non-retryable response returns immediately ─────────────


@pytest.mark.parametrize("status", [200, 201, 400, 401, 404, 422])
def test_execute_non_retryable_status_returns_first_response(status: int) -> None:
    fn = _Fixed([_resp(status)])
    sleep, calls = _capture_sleep()
    policy = RetryPolicy(max_retries=5, sleep=sleep, rng=_fixed_rng())
    response = policy.execute(fn, method="GET", has_idempotency_key=False)
    assert response.status_code == status
    assert fn.calls == 1
    assert calls == []


# ───────────── execute: unsafe method skips status retry ─────────────


def test_execute_unsafe_method_does_not_retry_5xx() -> None:
    # POST without idempotency key + 503 → return immediately, do not retry.
    fn = _Fixed([_resp(503)])
    sleep, calls = _capture_sleep()
    policy = RetryPolicy(max_retries=3, sleep=sleep, rng=_fixed_rng())
    response = policy.execute(fn, method="POST", has_idempotency_key=False)
    assert response.status_code == 503
    assert fn.calls == 1
    assert calls == []


# ───────────── injected sleep / rng default behaviour ─────────────


def test_default_sleep_is_real(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real time.sleep is the default; we patch it so the test stays fast.
    sleeps: list[float] = []

    def fast_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("pictograph._http.retry.time.sleep", fast_sleep)
    fn = _Fixed([_resp(503), _resp(200)])
    policy = RetryPolicy(max_retries=1, rng=_fixed_rng(1.0))
    policy.execute(fn, method="GET", has_idempotency_key=False)
    assert sleeps == [1.0]


def test_default_rng_is_real_random_uniform(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default rng is random.uniform; patch to verify it's actually called.
    seen: list[tuple[float, float]] = []

    def fake_uniform(a: float, b: float) -> float:
        seen.append((a, b))
        return 1.0

    monkeypatch.setattr("pictograph._http.retry.random.uniform", fake_uniform)
    sleep, _ = _capture_sleep()
    policy = RetryPolicy(max_retries=1, sleep=sleep)
    fn = _Fixed([_resp(503), _resp(200)])
    policy.execute(fn, method="GET", has_idempotency_key=False)
    assert seen == [(JITTER_LOW, JITTER_HIGH)]


# ───────────── _delay_for_response edge ─────────────


def test_delay_for_non_429_uses_exponential_backoff() -> None:
    policy = RetryPolicy(backoff_base=2.0, rng=_fixed_rng(1.0))
    # Indirect via execute: 500 + then 200 → first delay should be 2.0
    fn = _Fixed([_resp(500), _resp(200)])
    sleep, calls = _capture_sleep()
    policy = RetryPolicy(max_retries=3, backoff_base=2.0, sleep=sleep, rng=_fixed_rng(1.0))
    policy.execute(fn, method="GET", has_idempotency_key=False)
    assert calls == [2.0]


# ───────────── stress: queue empty → AssertionError surfaces test bugs ─────────────


def test_request_fn_overrun_surfaces_assertion_error_in_test_helper() -> None:
    # If we accidentally let execute call request_fn more than provided, the
    # test helper raises AssertionError - protects against silent infinite loops.
    fn = _Fixed([_resp(503)])
    policy = RetryPolicy(max_retries=2, sleep=lambda _: None, rng=_fixed_rng())
    # Force the helper to overrun by setting only one canned response but
    # asking for retries; we expect the helper to assert.
    with pytest.raises(AssertionError, match="more times"):
        policy.execute(fn, method="GET", has_idempotency_key=False)


# ───────────── doc-style invariants used elsewhere ─────────────


def test_jitter_constants_form_a_symmetric_window_around_one() -> None:
    # Documented guarantee: jitter spans ±25% of the base delay.
    assert JITTER_LOW < 1.0 < JITTER_HIGH
    assert (1.0 - JITTER_LOW) == pytest.approx(JITTER_HIGH - 1.0)


# ───────────── unused-import safety ─────────────


def test_module_exports_match_documentation() -> None:
    """Adding OR removing a public symbol must show up here.

    This asserted ``expected.issubset(actual)``, which by definition only sees
    REMOVALS - so a new public name that the SDK's ``__init__`` does not re-export
    passed silently, which is half of what the comment claimed it did. The two
    hand-maintained ``actual -= {...}`` scrub lines were the tell: they exist to
    make the sets EQUAL.

    Flipping to ``==`` immediately found ``asyncio``, an import added after those
    scrub lines were written and never added to them - the exact drift the weaker
    operator could not report. So the filter no longer hand-lists imports: module
    objects and anything whose ``__module__`` is elsewhere are dropped
    structurally, and only genuinely-local names remain.
    """
    import inspect as _inspect

    from pictograph._http import retry as retry_module

    expected = {
        "DEFAULT_BACKOFF_BASE",
        "DEFAULT_BACKOFF_CAP",
        "DEFAULT_MAX_RETRIES",
        "DEFAULT_RETRY_AFTER_THRESHOLD",
        "JITTER_HIGH",
        "JITTER_LOW",
        "RETRYABLE_STATUS_CODES",
        "RetryPolicy",
        "SAFE_METHODS",
        "is_method_safe_to_retry",
        "is_retryable_status",
    }
    actual = {
        name
        for name, obj in vars(retry_module).items()
        if not name.startswith("_")
        # imported modules (logging, random, time, httpx, asyncio, ...)
        and not _inspect.ismodule(obj)
        # names that live somewhere else (NetworkError, typing's Any/Callable/Final,
        # __future__.annotations). A plain constant has no __module__, so it
        # defaults to this module's own name and is kept.
        and getattr(obj, "__module__", retry_module.__name__) == retry_module.__name__
    }
    # `TYPE_CHECKING` is the bool False at runtime, so it carries no __module__ and
    # survives the filter above. It is typing's, not ours.
    actual -= {"TYPE_CHECKING", "logger"}
    assert expected == actual, (
        f"the module's public surface no longer matches this list.\n"
        f"  missing: {sorted(expected - actual)}\n"
        f"  extra:   {sorted(actual - expected)}"
    )


# satisfy type hint usage at runtime
_ = (Any, Callable, Iterable, Iterator)
