"""Integration test config - vcrpy cassettes for cross-resource flows.

While unit tests use ``pytest-httpx`` to mock individual responses,
integration tests use ``vcrpy`` cassettes recorded against staging.
This catches drift between the SDK's expectations and real backend
behavior (status codes, header names, response field shapes).

Cassettes live in ``tests/integration/cassettes/`` and are committed
to the repo. To re-record::

    PICTOGRAPH_TEST_KEY=… pytest tests/integration/ --record-mode=once

Without the key, cassettes are replayed and the suite stays offline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

CASSETTE_DIR = Path(__file__).parent / "cassettes"


@pytest.fixture(scope="session")
def vcr_config() -> dict[str, Any]:
    """vcrpy configuration applied to every cassette in this directory."""
    return {
        "filter_headers": [
            "authorization",
            "x-api-key",
            "cookie",
            ("user-agent", "pictograph-sdk-test/1.0"),
        ],
        "filter_query_parameters": ["api_key"],
        "decode_compressed_response": True,
        "record_mode": "none",  # CI never records - cassettes must exist.
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
    }


@pytest.fixture
def vcr_cassette_dir() -> str:
    return str(CASSETTE_DIR)
