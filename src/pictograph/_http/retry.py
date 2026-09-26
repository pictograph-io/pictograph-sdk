"""Retry policy for HTTP requests.

The SDK auto-retries when:

- The server returns a transient status (``500``, ``502``, ``503``, ``504``,
  ``408`` Request Timeout, or ``429`` Too Many Requests).
- The request fails before reaching the server (DNS / TCP / TLS / read failure)
  - represented in the SDK as :class:`pictograph.exceptions.NetworkError`.

Whether a retry actually happens depends on **method safety**:

- Idempotent methods (``GET``, ``HEAD``, ``OPTIONS``, ``PUT``, ``DELETE``)
  always retry transient failures.
- Non-idempotent methods (``POST``, ``PATCH``) retry **only** when an
  ``Idempotency-Key`` is set. The SDK auto-attaches one for every write
  through :mod:`pictograph._http.idempotency`, so in practice every SDK
  request is retry-safe - but the policy is conservative against direct
  callers that strip the header.

``429`` is the one exception: the server explicitly stated the request was
*rejected* (not processed), so we retry regardless of method.

Backoff is exponential with jitter: ``base * 2**attempt`` seconds, multiplied
by a uniform jitter in ``[0.75, 1.25]``, capped at ``backoff_cap``. ``Retry-After``
on 429 overrides this - we honour it as long as it's under
``retry_after_autowait_threshold`` seconds (default 120s); larger values are
surfaced to the caller so they can decide whether to wait.

The :class:`RetryPolicy` accepts injected ``sleep`` and ``rng`` callables to
make tests fully deterministic.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import TYPE_CHECKING, Final

from pictograph.exceptions import NetworkError, _parse_retry_after

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import httpx

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES: Final = 3
DEFAULT_BACKOFF_BASE: Final = 1.0
DEFAULT_BACKOFF_CAP: Final = 30.0
DEFAULT_RETRY_AFTER_THRESHOLD: Final = 120.0
JITTER_LOW: Final = 0.75
JITTER_HIGH: Final = 1.25

SAFE_METHODS: Final[frozenset[str]] = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})
"""HTTP methods that are idempotent at the protocol level.

These are always retry-safe even without an explicit idempotency key.
``POST``/``PATCH`` are retried only when an ``Idempotency-Key`` header is
present (which the SDK auto-attaches for writes).
"""

RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504})
"""Status codes that trigger a retry attempt."""


def is_method_safe_to_retry(method: str, *, has_idempotency_key: bool) -> bool:
    """Decide whether a method may be retried.

    Idempotent methods are always safe. ``POST``/``PATCH`` require an
    idempotency key so the backend can deduplicate at the application level.
    """
    if method.upper() in SAFE_METHODS:
        return True
    return has_idempotency_key


def is_retryable_status(status_code: int, *, safe_to_retry: bool) -> bool:
    """Decide whether a response status warrants a retry.

    ``429`` is unconditionally retryable - the server explicitly told us the
    request was *not processed*, so even non-idempotent calls are safe.
    Other transient statuses (5xx, 408) gate on ``safe_to_retry`` so a flaky
    ``502`` from a non-idempotent ``POST`` doesn't risk double execution.
    """
    if status_code == 429:
        return True
    if not safe_to_retry:
        return False
    return status_code in RETRYABLE_STATUS_CODES


class RetryPolicy:
    """Stateless retry policy applied around request callables.

    Args:
        max_retries: Maximum *additional* attempts after the initial one.
            ``0`` disables retries entirely.
        backoff_base: First-attempt delay in seconds.
        backoff_cap: Upper bound on a single delay (post-jitter ``min``).
        retry_after_autowait_threshold: ``Retry-After`` values up to this many
            seconds are honoured automatically. Larger values cause the call
            to return the 429 response so the caller can decide.
        sleep: Injected for tests. Defaults to :func:`time.sleep`.
        rng: Injected for tests. Defaults to :func:`random.uniform`.
    """

    _sleep: Callable[[float], None]
    _async_sleep: Callable[[float], Awaitable[None]]
    _rng: Callable[[float, float], float]

    def __init__(
        self,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        backoff_cap: float = DEFAULT_BACKOFF_CAP,
        retry_after_autowait_threshold: float = DEFAULT_RETRY_AFTER_THRESHOLD,
        sleep: Callable[[float], None] | None = None,
        async_sleep: Callable[[float], Awaitable[None]] | None = None,
        rng: Callable[[float, float], float] | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if backoff_base <= 0:
            raise ValueError(f"backoff_base must be > 0, got {backoff_base}")
        if backoff_cap < backoff_base:
            raise ValueError(
                f"backoff_cap ({backoff_cap}) must be >= backoff_base ({backoff_base})"
            )
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.retry_after_autowait_threshold = retry_after_autowait_threshold
        self._sleep = sleep if sleep is not None else time.sleep
        self._async_sleep = async_sleep if async_sleep is not None else asyncio.sleep
        self._rng = rng if rng is not None else random.uniform

    def compute_backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter for the given 0-indexed attempt."""
        if attempt < 0:
            raise ValueError(f"attempt must be >= 0, got {attempt}")
        # Cap the exponent before exponentiation: backoff_base is a float, so
        # ``base * (2**attempt)`` coerces the int 2**attempt to float *before*
        # min() caps it, and ``2**1024`` overflows float (OverflowError). The
        # cap is applied right after anyway, and 2**30 already vastly exceeds
        # any sane backoff_cap, so clamping the shift is exactly equivalent.
        shift = min(attempt, 30)
        raw = min(self.backoff_base * (2**shift), self.backoff_cap)
        jitter: float = self._rng(JITTER_LOW, JITTER_HIGH)
        return float(raw * jitter)

    def execute(
        self,
        request_fn: Callable[[], httpx.Response],
        *,
        method: str,
        has_idempotency_key: bool,
    ) -> httpx.Response:
        """Run ``request_fn`` with retry semantics; return the final response.

        Raises whatever :class:`NetworkError` ``request_fn`` last raised once
        retries are exhausted (or immediately if the method isn't safe to
        retry). 4xx/5xx responses are returned to the caller - error mapping
        happens in the transport layer, not here.
        """
        safe = is_method_safe_to_retry(method, has_idempotency_key=has_idempotency_key)

        for attempt in range(self.max_retries + 1):
            is_last = attempt == self.max_retries
            try:
                response = request_fn()
            except NetworkError as exc:
                if not safe or is_last:
                    raise
                delay = self.compute_backoff(attempt)
                logger.debug(
                    "Retrying after network error (attempt %d/%d, sleeping %.2fs): %s",
                    attempt + 1,
                    self.max_retries,
                    delay,
                    exc,
                )
                self._sleep(delay)
                continue

            if not is_retryable_status(response.status_code, safe_to_retry=safe):
                return response
            if is_last:
                return response

            wait = self._delay_for_response(response, attempt)
            if wait is None:
                # Retry-After exceeded the auto-wait threshold; surface response.
                return response
            logger.debug(
                "Retrying after status %d (attempt %d/%d, sleeping %.2fs)",
                response.status_code,
                attempt + 1,
                self.max_retries,
                wait,
            )
            self._sleep(wait)

        # Loop guarantees a return for max_retries >= 0 (validated). Unreachable.
        raise AssertionError(
            "RetryPolicy.execute fell through the retry loop without returning; "
            "this indicates a logic bug."
        )

    async def execute_async(
        self,
        request_fn: Callable[[], Awaitable[httpx.Response]],
        *,
        method: str,
        has_idempotency_key: bool,
    ) -> httpx.Response:
        """Async twin of :meth:`execute` - identical policy, ``await``\\ ed I/O.

        Shares every decision helper (:func:`is_method_safe_to_retry`,
        :func:`is_retryable_status`, :meth:`compute_backoff`,
        :meth:`_delay_for_response`) with the sync path so the two can never
        drift; only the request call and the backoff sleep are awaited.
        """
        safe = is_method_safe_to_retry(method, has_idempotency_key=has_idempotency_key)

        for attempt in range(self.max_retries + 1):
            is_last = attempt == self.max_retries
            try:
                response = await request_fn()
            except NetworkError as exc:
                if not safe or is_last:
                    raise
                delay = self.compute_backoff(attempt)
                logger.debug(
                    "Retrying after network error (attempt %d/%d, sleeping %.2fs): %s",
                    attempt + 1,
                    self.max_retries,
                    delay,
                    exc,
                )
                await self._async_sleep(delay)
                continue

            if not is_retryable_status(response.status_code, safe_to_retry=safe):
                return response
            if is_last:
                return response

            wait = self._delay_for_response(response, attempt)
            if wait is None:
                return response
            logger.debug(
                "Retrying after status %d (attempt %d/%d, sleeping %.2fs)",
                response.status_code,
                attempt + 1,
                self.max_retries,
                wait,
            )
            await self._async_sleep(wait)

        raise AssertionError(
            "RetryPolicy.execute_async fell through the retry loop without returning; "
            "this indicates a logic bug."
        )

    def _delay_for_response(self, response: httpx.Response, attempt: int) -> float | None:
        """Pick the sleep duration before retrying a 4xx/5xx response.

        For 429 with a ``Retry-After`` header within the auto-wait threshold
        we honour the server's number. For 429 with a ``Retry-After`` greater
        than the threshold we return ``None`` to signal the caller should
        return immediately. Everything else uses exponential backoff.
        """
        if response.status_code == 429:
            retry_after = _parse_retry_after(response.headers)
            if retry_after is not None:
                if retry_after > self.retry_after_autowait_threshold:
                    return None
                # Honour the server's hint, but never below our own backoff:
                # a misconfigured/overloaded gateway can return ``Retry-After: 0``
                # (or a past HTTP-date, which _parse_retry_after clamps to 0),
                # which would otherwise fire an immediate, jitter-less retry
                # burst and amplify load on an already-throttled server. The
                # floor keeps the burst spaced by the normal exponential backoff.
                return max(retry_after, self.compute_backoff(attempt))
        return self.compute_backoff(attempt)
