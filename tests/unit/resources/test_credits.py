"""Tests for ``pictograph.resources.credits.Credits``.

Coverage targets:
- ``balance`` returns typed CreditBalance with µUSD fields + USD properties.
- ``history`` paginates correctly + preserves signed µUSD amounts.
- ``iter`` auto-pages.
- ``estimate`` parses the µUSD response shape, exposes ``total_usd``, and
  surfaces sufficient/cost fields for agent decision-making.
- Deprecated whole-dollar fields stay readable for back-compat.
- 400 on unknown operation propagates as ValidationError.

All credit amounts are micro-USD (µUSD): ``1 USD = 1_000_000 µUSD``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import ValidationError
from pictograph.models.credit import CreditBalance, CreditEstimate, CreditLedgerEntry
from pictograph.resources.credits import Credits

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"


@pytest.fixture
def transport() -> Transport:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def credits(transport: Transport) -> Credits:
    return Credits(transport)


def _ledger_entry(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "ledger-uuid-1",
        "operation": "training_a10g_per_minute",
        "amount": -150_000,  # µUSD ($0.15 debit)
        "balance_after": 850_000_000,  # µUSD ($850.00)
        "description": "Training pre-charge",
        "metadata": {"run_id": "r1"},
        "created_at": "2026-04-01T00:00:00Z",
    }
    base.update(overrides)
    return base


def _balance_body(**overrides: Any) -> dict[str, Any]:
    """The new µUSD balance wire shape (deprecated whole-dollar fields still echoed)."""
    base: dict[str, Any] = {
        "included_allowance_micro_usd": 1_000_000_000,  # $1,000.00
        "included_remaining_micro_usd": 850_000_000,  # $850.00
        "budget_micro_usd": 50_000_000,  # $50.00
        "period_spend_micro_usd": 150_000_000,  # $150.00
        "period_overage_micro_usd": 0,
        "budget_remaining_micro_usd": 50_000_000,  # $50.00
        "period_start": "2026-04-01T00:00:00Z",
        "credits_reset_at": "2026-05-01T00:00:00Z",
        # Deprecated whole-dollar fields the backend still emits.
        "credits_remaining": 850,
        "credits_monthly_allowance": 1000,
        "recent_history": [_ledger_entry()],
    }
    base.update(overrides)
    return base


def _estimate_body(**overrides: Any) -> dict[str, Any]:
    """The new µUSD estimate wire shape (deprecated whole-dollar fields still echoed)."""
    base: dict[str, Any] = {
        "operation": "training_a10g",
        "micro_usd_per_unit": 22_917,  # µUSD/minute
        "unit": "minute",
        "quantity": 20,
        "total_micro_usd": 458_340,  # µUSD ($0.45834)
        "sufficient": True,
        "remaining_micro_usd": 850_000_000,  # $850.00
        # Deprecated whole-dollar fields.
        "credits_per_unit": 0,
        "total_credits": 0,
        "minimum": 0,
        "credits_remaining": 850,
    }
    base.update(overrides)
    return base


# ───────────── balance ─────────────


def test_balance_returns_typed_credit_balance(httpx_mock: HTTPXMock, credits: Credits) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/credits/balance",
        json=_balance_body(
            recent_history=[
                _ledger_entry(),
                _ledger_entry(id="l2", amount=-3_000, operation="sam3_per_minute"),
            ],
        ),
    )
    balance = credits.balance()
    assert isinstance(balance, CreditBalance)
    # µUSD wire fields parse as ints.
    assert balance.included_allowance_micro_usd == 1_000_000_000
    assert balance.included_remaining_micro_usd == 850_000_000
    assert balance.budget_micro_usd == 50_000_000
    assert len(balance.recent_history) == 2
    assert all(isinstance(e, CreditLedgerEntry) for e in balance.recent_history)


def test_usage_by_operation_returns_typed_report(httpx_mock: HTTPXMock, credits: Credits) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/credits/usage-by-operation?range=week",
        json={
            "range": "week",
            "since": "2026-06-13T00:00:00Z",
            "operations": [
                {"operation": "training_a10g", "total_micro_usd": 2_500_000, "event_count": 3},
                {"operation": "sam3_per_minute", "total_micro_usd": 500_000, "event_count": 12},
            ],
            "total_micro_usd": 3_000_000,
            "total_events": 15,
        },
    )
    report = credits.usage_by_operation(range="week")
    assert report.range == "week"
    assert report.total_micro_usd == 3_000_000
    assert report.total_usd == 3.0
    assert report.total_events == 15
    assert [op.operation for op in report.operations] == ["training_a10g", "sam3_per_minute"]
    assert report.operations[0].total_usd == 2.5
    assert report.operations[1].event_count == 12


def test_usage_by_operation_defaults_to_month(httpx_mock: HTTPXMock, credits: Credits) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/credits/usage-by-operation?range=month",
        json={"range": "month", "operations": [], "total_micro_usd": 0, "total_events": 0},
    )
    report = credits.usage_by_operation()
    assert report.range == "month"
    assert report.operations == []
    # The default range is sent on the wire.
    assert "range=month" in str(httpx_mock.get_requests()[-1].url)


def test_balance_usd_properties(httpx_mock: HTTPXMock, credits: Credits) -> None:
    """The USD convenience properties divide µUSD by 1,000,000."""
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/credits/balance",
        json=_balance_body(period_overage_micro_usd=1_250_000),
    )
    balance = credits.balance()
    assert balance.remaining_usd == 850.0
    assert balance.allowance_usd == 1000.0
    assert balance.budget_usd == 50.0
    assert balance.overage_usd == 1.25
    assert balance.spend_usd == 150.0
    assert balance.budget_remaining_usd == 50.0


def test_balance_parses_period_start(httpx_mock: HTTPXMock, credits: Credits) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/credits/balance",
        json=_balance_body(),
    )
    balance = credits.balance()
    assert balance.period_start is not None
    assert balance.period_start.year == 2026


def test_balance_deprecated_fields_still_present(httpx_mock: HTTPXMock, credits: Credits) -> None:
    """Legacy whole-dollar fields remain readable for back-compat."""
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/credits/balance",
        json=_balance_body(),
    )
    balance = credits.balance()
    assert balance.credits_remaining == 850
    assert balance.credits_monthly_allowance == 1000


def test_balance_handles_missing_recent_history(httpx_mock: HTTPXMock, credits: Credits) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/credits/balance",
        json={
            "included_allowance_micro_usd": 200_000_000,
            "included_remaining_micro_usd": 0,
            "period_start": None,
            "credits_reset_at": None,
        },
    )
    balance = credits.balance()
    assert balance.recent_history == []
    assert balance.credits_reset_at is None
    assert balance.period_start is None
    # Fields absent from the body fall back to their int=0 defaults.
    assert balance.budget_micro_usd == 0
    assert balance.remaining_usd == 0.0


# ───────────── history ─────────────


def test_history_returns_typed_entries(httpx_mock: HTTPXMock, credits: Credits) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/credits/history?limit=50&offset=0",
        json={
            "entries": [_ledger_entry(), _ledger_entry(id="l2")],
            "limit": 50,
            "offset": 0,
        },
    )
    entries = credits.history()
    assert len(entries) == 2
    assert all(isinstance(e, CreditLedgerEntry) for e in entries)


def test_history_passes_pagination_params(httpx_mock: HTTPXMock, credits: Credits) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/credits/history?limit=10&offset=20",
        json={"entries": [], "limit": 10, "offset": 20},
    )
    credits.history(limit=10, offset=20)
    sent = httpx_mock.get_request()
    assert sent is not None
    assert "limit=10" in str(sent.url)
    assert "offset=20" in str(sent.url)


def test_history_empty_result(httpx_mock: HTTPXMock, credits: Credits) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/credits/history?limit=50&offset=0",
        json={"entries": [], "limit": 50, "offset": 0},
    )
    assert credits.history() == []


def test_history_amount_sign_preserved(httpx_mock: HTTPXMock, credits: Credits) -> None:
    """Debits are negative, credits/refunds are positive (µUSD) - sign must survive."""
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/credits/history?limit=50&offset=0",
        json={
            "entries": [
                _ledger_entry(id="debit", amount=-100_000, operation="sam3_per_minute"),
                _ledger_entry(id="refund", amount=80_000, operation="training_refund"),
                _ledger_entry(id="topup", amount=200_000_000, operation="monthly_reset"),
            ],
            "limit": 50,
            "offset": 0,
        },
    )
    entries = credits.history()
    by_id = {e.id: e for e in entries}
    assert by_id["debit"].amount == -100_000
    assert by_id["refund"].amount == 80_000
    assert by_id["topup"].amount == 200_000_000


# ───────────── iter ─────────────


def test_iter_paginates(httpx_mock: HTTPXMock, credits: Credits) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/credits/history?offset=0&limit=2",
        json={
            "entries": [_ledger_entry(id="l1"), _ledger_entry(id="l2")],
            "limit": 2,
            "offset": 0,
        },
    )
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/credits/history?offset=2&limit=2",
        json={"entries": [_ledger_entry(id="l3")], "limit": 2, "offset": 2},
    )
    entries = list(credits.iter(page_size=2))
    assert [e.id for e in entries] == ["l1", "l2", "l3"]


# ───────────── estimate ─────────────


def test_estimate_returns_typed_credit_estimate(httpx_mock: HTTPXMock, credits: Credits) -> None:
    httpx_mock.add_response(
        method="GET",
        url=(f"{BASE}/api/v1/developer/credits/estimate?operation=training_a10g&quantity=20"),
        json=_estimate_body(),
    )
    estimate = credits.estimate("training_a10g", quantity=20)
    assert isinstance(estimate, CreditEstimate)
    # µUSD wire fields.
    assert estimate.micro_usd_per_unit == 22_917
    assert estimate.total_micro_usd == 458_340
    assert estimate.remaining_micro_usd == 850_000_000
    assert estimate.sufficient is True


def test_estimate_total_usd_property(httpx_mock: HTTPXMock, credits: Credits) -> None:
    """total_usd / per_unit_usd / remaining_usd divide µUSD by 1,000,000."""
    httpx_mock.add_response(
        method="GET",
        url=(f"{BASE}/api/v1/developer/credits/estimate?operation=training_a10g&quantity=20"),
        json=_estimate_body(total_micro_usd=2_500_000, remaining_micro_usd=10_000_000),
    )
    estimate = credits.estimate("training_a10g", quantity=20)
    assert estimate.total_usd == 2.5
    assert estimate.remaining_usd == 10.0
    assert estimate.per_unit_usd == pytest.approx(0.022917)


def test_estimate_default_quantity_is_one(httpx_mock: HTTPXMock, credits: Credits) -> None:
    httpx_mock.add_response(
        method="GET",
        url=(
            f"{BASE}/api/v1/developer/credits/estimate"
            "?operation=image_generate_imagen_fast&quantity=1"
        ),
        json=_estimate_body(
            operation="image_generate_imagen_fast",
            unit="image",
            quantity=1,
            micro_usd_per_unit=25_000,  # µUSD/image
            total_micro_usd=25_000,
            remaining_micro_usd=100_000_000,
        ),
    )
    credits.estimate("image_generate_imagen_fast")
    sent = httpx_mock.get_request()
    assert sent is not None
    assert "quantity=1" in str(sent.url)


def test_estimate_deprecated_fields_default_zero(httpx_mock: HTTPXMock, credits: Credits) -> None:
    """Legacy whole-dollar fields are absent from the wire → default to 0."""
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/credits/estimate?operation=sam3_auto_annotation&quantity=1",
        json={
            "operation": "sam3_auto_annotation",
            "micro_usd_per_unit": 2_500,
            "unit": "minute",
            "quantity": 1,
            "total_micro_usd": 2_500,
            "sufficient": True,
            "remaining_micro_usd": 100_000_000,
        },
    )
    estimate = credits.estimate("sam3_auto_annotation", quantity=1)
    assert estimate.total_micro_usd == 2_500
    assert estimate.credits_per_unit == 0
    assert estimate.total_credits == 0
    assert estimate.minimum == 0
    assert estimate.credits_remaining == 0


def test_estimate_insufficient_balance(httpx_mock: HTTPXMock, credits: Credits) -> None:
    httpx_mock.add_response(
        method="GET",
        url=(f"{BASE}/api/v1/developer/credits/estimate?operation=training_h100&quantity=60"),
        json=_estimate_body(
            operation="training_h100",
            micro_usd_per_unit=82_292,  # µUSD/minute
            quantity=60,
            total_micro_usd=4_937_520,  # ~$4.94
            sufficient=False,
            remaining_micro_usd=1_000_000,  # only $1.00 left
        ),
    )
    estimate = credits.estimate("training_h100", quantity=60)
    assert estimate.sufficient is False
    assert estimate.remaining_micro_usd < estimate.total_micro_usd


def test_estimate_400_unknown_operation(httpx_mock: HTTPXMock, credits: Credits) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/credits/estimate?operation=bogus&quantity=1",
        status_code=400,
        json={"detail": "Unknown operation: bogus"},
    )
    with pytest.raises(ValidationError):
        credits.estimate("bogus")
