"""Live: credits balance / history / estimate."""

from __future__ import annotations

import pytest

from pictograph import Client
from pictograph.models.credit import CreditBalance, CreditEstimate, CreditLedgerEntry

pytestmark = pytest.mark.live


def test_balance_shape(client: Client) -> None:
    balance = client.credits.balance()
    assert isinstance(balance, CreditBalance)
    # Compute credit is USD-denominated (µUSD on the wire).
    assert balance.included_remaining_micro_usd >= 0
    assert balance.included_allowance_micro_usd > 0
    assert balance.remaining_usd >= 0
    assert balance.allowance_usd > 0
    assert balance.credits_reset_at is not None


def test_history_default_page(client: Client) -> None:
    entries = client.credits.history(limit=5)
    assert isinstance(entries, list)
    for e in entries:
        assert isinstance(e, CreditLedgerEntry)
        assert e.id
        assert e.created_at is not None


def test_history_iter_auto_pages(client: Client) -> None:
    # Pull up to 10 entries via the auto-pager.
    pager = client.credits.iter(page_size=5, max_total=10)
    items = pager.all()
    assert isinstance(items, list)
    assert len(items) <= 10


@pytest.mark.parametrize(
    "operation,expected_unit",
    [
        # The unit vocabulary is "minute" / "call" - utils/pricing.py emits
        # exactly those strings. This table said "per_minute"/"per_call" and had
        # been red on every one of these rows.
        ("training_a10g", "minute"),
        ("training_a100", "minute"),
        ("training_h100", "minute"),
        ("sam3_auto_annotation", "minute"),
        ("inference_t4", "minute"),
        ("image_edit_gemini_flash", "call"),
        ("image_generate_imagen_fast", "call"),
    ],
)
def test_estimate_operations(client: Client, operation: str, expected_unit: str) -> None:
    est = client.credits.estimate(operation, quantity=1)
    assert isinstance(est, CreditEstimate)
    assert est.operation == operation
    assert est.micro_usd_per_unit > 0
    assert est.total_micro_usd > 0
    assert est.total_usd > 0
    assert est.unit == expected_unit


def test_estimate_quantity_scales_linearly(client: Client) -> None:
    one = client.credits.estimate("training_a10g", quantity=1)
    ten = client.credits.estimate("training_a10g", quantity=10)
    assert ten.quantity == 10
    assert one.total_micro_usd == one.micro_usd_per_unit

    # NOT `total == per_unit * 10`. For a GPU-minute operation the two figures
    # are rounded INDEPENDENTLY - utils/pricing.py computes
    # per_unit = gpu_minutes_charge(key, 1) and total = gpu_minutes_charge(key, qty),
    # so the total is taken at full precision and rounded once instead of
    # multiplying an already-rounded rate. That is the more accurate of the two,
    # and it makes them differ by up to one µUSD per unit: this assertion used to
    # demand equality and was red on 36667*10 = 366670 vs an actual 366667.
    assert abs(ten.total_micro_usd - ten.micro_usd_per_unit * 10) <= 10
    assert ten.total_micro_usd > one.total_micro_usd * 9
