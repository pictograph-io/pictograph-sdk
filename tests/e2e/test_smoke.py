"""E2E smoke tests against the live Pictograph API.

These verify the SDK's basic plumbing against the real backend - they
don't try to be exhaustive. They are gated by ``PICTOGRAPH_TEST_KEY``
and skipped in the default test run.

What's intentionally NOT here:
- Heavy operations (training, batch SAM3) - too slow + too expensive.
- Mutating ops on shared state - relies on the test org being a
  scratch space, not a customer's prod org.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pictograph import Client


pytestmark = pytest.mark.e2e


def test_credit_balance_returns_real_data(e2e_client: Client) -> None:
    """Balance endpoint round-trips against the real backend (USD/µUSD)."""
    balance = e2e_client.credits.balance()
    assert balance.included_remaining_micro_usd >= 0
    assert balance.included_allowance_micro_usd >= 0
    assert balance.remaining_usd >= 0


def test_list_datasets_returns_array(e2e_client: Client) -> None:
    """List endpoint round-trips. Empty org returns []."""
    datasets = e2e_client.datasets.list(limit=5)
    assert isinstance(datasets, list)


def test_estimate_credit_cost_returns_estimate(e2e_client: Client) -> None:
    """Cost estimation endpoint round-trips (USD/µUSD)."""
    estimate = e2e_client.credits.estimate("training_a10g", quantity=1)
    assert estimate.operation == "training_a10g"
    assert estimate.total_micro_usd > 0
    assert estimate.total_usd > 0
    assert estimate.unit == "per_minute"


def test_create_and_delete_scratch_dataset(e2e_client: Client) -> None:
    """Round-trip a dataset create + delete, with a unique name."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    name = f"sdk-e2e-{timestamp}"
    project = e2e_client.datasets.create(name)
    assert project.name == name
    try:
        # Verify it's listable.
        fetched = e2e_client.datasets.get(name)
        assert fetched.name == name
    finally:
        e2e_client.datasets.delete(name)


def test_unauthorized_request_raises_authentication_error(
    e2e_client: Client,
    base_url: str | None,
) -> None:
    """Bad API key surfaces as an AuthError, not a generic ApiError."""
    from pictograph import Client
    from pictograph.exceptions import AuthError

    bad_client = Client(api_key="pk_live_invalid_key_definitely_wrong", base_url=base_url)
    with pytest.raises(AuthError):
        bad_client.datasets.list(limit=1)
