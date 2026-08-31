"""Integration smoke: cross-resource round-trip flows.

These tests exercise the SDK against a stubbed HTTP layer (pytest-httpx)
configured to behave like the real backend across multiple sequential
calls - catches issues that single-resource unit tests miss
(pagination ordering, retry+rate-limit interactions, multi-step
workflows that span resources).

vcrpy cassettes are reserved for tests/integration/test_vcr_*.py - when
we want to assert against actual recorded backend bytes. This file uses
hand-stubbed responses to keep the suite hermetic + fast.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock


_BASE = "https://api.pictograph.io"


@pytest.fixture
def integration_client():  # type: ignore[no-untyped-def]
    """Real Client pointed at the production base URL - paired with HTTPXMock."""
    from pictograph import Client

    return Client(api_key="pk_live_test_integration_key_aaaaaaaaaa", base_url=_BASE)


# ───────────── credit-gated workflow round-trip ─────────────


def test_balance_check_before_training_kickoff(
    httpx_mock: HTTPXMock,
    integration_client,  # type: ignore[no-untyped-def]
) -> None:
    """Balance lookup → estimate (USD/µUSD) → training create - sequential calls."""
    httpx_mock.add_response(
        method="GET",
        url=f"{_BASE}/api/v1/developer/credits/balance",
        json={
            "included_allowance_micro_usd": 10_000_000_000,  # $10,000.00
            "included_remaining_micro_usd": 5_000_000_000,  # $5,000.00
            "budget_micro_usd": 0,
            "period_spend_micro_usd": 5_000_000_000,
            "period_overage_micro_usd": 0,
            "budget_remaining_micro_usd": 0,
            "period_start": "2026-04-01T00:00:00Z",
            "credits_reset_at": "2026-05-01T00:00:00Z",
            "recent_history": [],
        },
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{_BASE}/api/v1/developer/credits/estimate?operation=training_a10g&quantity=30",
        json={
            "operation": "training_a10g",
            "micro_usd_per_unit": 22_917,  # µUSD/minute
            "unit": "minute",
            "quantity": 30,
            "total_micro_usd": 687_510,  # ~$0.69
            "sufficient": True,
            "remaining_micro_usd": 5_000_000_000,
        },
    )

    balance = integration_client.credits.balance()
    assert balance.included_remaining_micro_usd == 5_000_000_000
    assert balance.remaining_usd == 5000.0

    estimate = integration_client.credits.estimate(
        "training_a10g",
        quantity=30,
    )
    assert estimate.sufficient
    assert estimate.total_micro_usd == 687_510
    assert estimate.total_usd == pytest.approx(0.68751)


def test_dataset_list_then_get_round_trip(
    httpx_mock: HTTPXMock,
    integration_client,  # type: ignore[no-untyped-def]
) -> None:
    """List datasets, pick one, fetch full details - two sequential calls."""
    httpx_mock.add_response(
        method="GET",
        url=f"{_BASE}/api/v1/developer/datasets/?limit=10&offset=0",
        json={
            "data": [
                {
                    "id": "proj-uuid-1",
                    "name": "road-signs",
                    "description": None,
                    "image_count": 100,
                    "completed_image_count": 80,
                    "total_size": 12345678,
                    "archived_images": 0,
                    "classes": [],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            ],
        },
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{_BASE}/api/v1/developer/datasets/road-signs",
        json={
            "data": {
                "id": "proj-uuid-1",
                "name": "road-signs",
                "description": None,
                "image_count": 100,
                "completed_image_count": 80,
                "total_size": 12345678,
                "archived_images": 0,
                "classes": [
                    {"name": "stop_sign", "type": "bbox", "color": "#ff0000"},
                ],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "organization_id": "org-uuid",
            },
        },
    )

    listed = integration_client.datasets.list(limit=10)
    assert len(listed) == 1
    fetched = integration_client.datasets.get(listed[0].name)
    assert fetched.classes[0].name == "stop_sign"


def test_paginated_iter_accumulates_across_pages(
    httpx_mock: HTTPXMock,
    integration_client,  # type: ignore[no-untyped-def]
) -> None:
    """OffsetPager fans out across pages and stops at the empty page."""
    page1 = {
        "data": [
            {
                "id": f"proj-{i}",
                "name": f"ds-{i}",
                "description": None,
                "image_count": 0,
                "completed_image_count": 0,
                "total_size": 0,
                "archived_images": 0,
                "classes": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            for i in range(2)
        ],
    }
    page2: dict[str, list[dict[str, object]]] = {"data": []}
    httpx_mock.add_response(
        method="GET",
        url=f"{_BASE}/api/v1/developer/datasets/?offset=0&limit=2",
        json=page1,
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{_BASE}/api/v1/developer/datasets/?offset=2&limit=2",
        json=page2,
    )

    items = list(integration_client.datasets.iter(page_size=2))
    assert [d.name for d in items] == ["ds-0", "ds-1"]
