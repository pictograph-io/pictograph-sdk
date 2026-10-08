"""Tests for ``pictograph._http.idempotency``."""

from __future__ import annotations

import re
import uuid

import pytest

from pictograph._http.idempotency import (
    IDEMPOTENCY_HEADER,
    WRITE_METHODS,
    generate_key,
    needs_idempotency,
)


def test_idempotency_header_name_is_stable() -> None:
    # Stripe-compatible spelling. If this changes, downstream backend dedup
    # middleware and any user-supplied keys break - pin the name explicitly.
    assert IDEMPOTENCY_HEADER == "Idempotency-Key"


def test_write_methods_set_is_exactly_post_put_patch() -> None:
    # DELETE is excluded by design (naturally idempotent); GET/HEAD/OPTIONS too.
    assert WRITE_METHODS == frozenset({"POST", "PUT", "PATCH"})


def test_generate_key_returns_uuid4_hex() -> None:
    key = generate_key()
    # 32-char lowercase hex, no dashes.
    assert re.fullmatch(r"[0-9a-f]{32}", key) is not None
    # Round-trip through uuid.UUID confirms it's a real uuid hex value.
    parsed = uuid.UUID(hex=key)
    assert parsed.version == 4


def test_generate_key_returns_unique_values() -> None:
    # Birthday-paradox safety: 1000 keys should never collide for uuid4.
    keys = {generate_key() for _ in range(1000)}
    assert len(keys) == 1000


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("POST", True),
        ("PUT", True),
        ("PATCH", True),
        ("post", True),  # case-insensitive
        ("Post", True),
        ("GET", False),
        ("HEAD", False),
        ("OPTIONS", False),
        ("DELETE", False),
        ("get", False),
    ],
)
def test_needs_idempotency_classifies_methods(method: str, expected: bool) -> None:
    assert needs_idempotency(method) is expected


def test_needs_idempotency_unknown_method_is_treated_as_safe_no_key() -> None:
    # Custom methods (e.g., WebDAV's PROPFIND) shouldn't accidentally get keyed.
    assert needs_idempotency("PROPFIND") is False
    assert needs_idempotency("CUSTOM") is False
