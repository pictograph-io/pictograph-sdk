"""Shared fixtures for the async (``pictograph.aio``) test suite.

Async tests run on the ``anyio`` pytest plugin (ships with ``httpx``'s ``anyio``
dependency - no extra dev dep). The ``anyio_backend`` fixture pins the backend to
``asyncio``; each async test module sets ``pytestmark = pytest.mark.anyio`` so
every ``async def test_*`` is collected without a per-test decorator.

``pytest-httpx``'s ``httpx_mock`` intercepts ``httpx.AsyncClient`` exactly as it
does the sync client, so the async transport/resources are exercised end-to-end
against canned responses (no network).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from pictograph._http.async_transport import AsyncTransport
from pictograph._http.retry import RetryPolicy
from pictograph._internal.config import ClientConfig

BASE_URL = "https://api.test.local"
API_KEY = "pk_live_test"


@pytest.fixture
def anyio_backend() -> str:
    """Run every ``@pytest.mark.anyio`` test on the asyncio backend only."""
    return "asyncio"


@pytest.fixture
def config() -> ClientConfig:
    return ClientConfig(
        api_key=API_KEY,  # type: ignore[arg-type]
        base_url=BASE_URL,
        timeout=10.0,
        max_retries=0,
    )


@pytest.fixture
async def transport(config: ClientConfig) -> AsyncIterator[AsyncTransport]:
    """Async transport with retries disabled for fast, deterministic tests."""
    policy = RetryPolicy(
        max_retries=0,
        sleep=lambda _: None,
        rng=lambda _a, _b: 1.0,
    )
    t = AsyncTransport(config, api_key=API_KEY, retry_policy=policy)
    yield t
    await t.aclose()
