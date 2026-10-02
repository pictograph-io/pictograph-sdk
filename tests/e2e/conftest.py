"""E2E test config - gated on ``PICTOGRAPH_TEST_KEY``.

E2E tests hit the real production API (``https://api.pictograph.io``)
with the test API key. They are excluded from the default suite - run
them via::

    PICTOGRAPH_TEST_KEY=pk_live_… pytest -m e2e

In CI they are gated to ``workflow_dispatch`` only (manual trigger).
"""

from __future__ import annotations

import os

import pytest

E2E_KEY_ENV_VAR = "PICTOGRAPH_TEST_KEY"


@pytest.fixture(scope="session")
def api_key() -> str:
    key = os.environ.get(E2E_KEY_ENV_VAR)
    if not key:
        pytest.skip(
            f"E2E tests require {E2E_KEY_ENV_VAR} env var. "
            "Set it to a valid Pictograph API key to enable."
        )
    return key


@pytest.fixture(scope="session")
def base_url() -> str | None:
    """Allow overriding the API base URL for E2E (default: production)."""
    return os.environ.get("PICTOGRAPH_TEST_BASE_URL")


@pytest.fixture(scope="session")
def e2e_client(api_key: str, base_url: str | None):  # type: ignore[no-untyped-def]
    """Real Client instance pointed at the configured environment."""
    from pictograph import Client

    return Client(api_key=api_key, base_url=base_url)
