"""Idempotency keys for safe retries of write requests.

Every write request (``POST``, ``PUT``, ``PATCH``) sent through the SDK is
auto-tagged with an ``Idempotency-Key`` header. The Pictograph backend
deduplicates within a 24-hour window so a retried request returns the same
response without re-executing side effects (avoiding double-charged credits,
duplicate annotation rows, etc.).

Callers may supply their own key via the ``idempotency_key`` request argument
when a stable, business-meaningful key is preferable (e.g. derived from a
user-facing job ID). Otherwise the SDK generates a fresh ``uuid4``.

``GET``, ``HEAD``, ``OPTIONS`` and ``DELETE`` are idempotent at the HTTP-method
level and do not get keys (DELETE's idempotence is the protocol guarantee that
``DELETE x; DELETE x`` leaves the resource in the same state - gone).
"""

from __future__ import annotations

import uuid
from typing import Final

IDEMPOTENCY_HEADER: Final = "Idempotency-Key"
"""Header name the SDK uses for keys (Stripe-compatible spelling)."""

WRITE_METHODS: Final[frozenset[str]] = frozenset({"POST", "PUT", "PATCH"})
"""HTTP methods the SDK auto-keys.

DELETE is intentionally absent: it is naturally idempotent and benefits no more
from a server-side dedup window.
"""


def generate_key() -> str:
    """Return a fresh, opaque idempotency key.

    Uses ``uuid4().hex`` (32-char lowercase hex, no dashes) - universally unique,
    URL-safe, and compact enough not to inflate request headers.
    """
    return uuid.uuid4().hex


def needs_idempotency(method: str) -> bool:
    """``True`` if the SDK should auto-attach an idempotency key.

    Method comparison is case-insensitive - callers can pass ``"post"`` or
    ``"POST"`` interchangeably.
    """
    return method.upper() in WRITE_METHODS
