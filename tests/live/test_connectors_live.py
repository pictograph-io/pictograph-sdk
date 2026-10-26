"""Live: connectors - validate with an invalid key surfaces cleanly."""

from __future__ import annotations

import pytest

from pictograph import Client
from pictograph.models.connector import LimitCheckResult, ValidationResult

pytestmark = pytest.mark.live


def test_validate_invalid_key_returns_failure(client: Client) -> None:
    """A bogus provider key returns ``valid=False`` rather than a 500."""
    result = client.connectors.validate("v7", "this-is-not-a-real-v7-key")
    assert isinstance(result, ValidationResult)
    assert result.valid is False


def test_check_limits(client: Client) -> None:
    """Preflight the import check with a small counts."""
    result = client.connectors.check_limits(total_images=10, estimated_size_bytes=1_000_000)
    assert isinstance(result, LimitCheckResult)
    # Org has caps on images/storage - expected to be allowed for 10 tiny imgs.
    assert result.allowed is True
