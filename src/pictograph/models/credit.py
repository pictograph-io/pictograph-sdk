"""Credit models - balance, ledger entries, cost estimates.

Returned by the :class:`pictograph.resources.credits.Credits` resource.
Response-side schema (``extra="ignore"``) so backend column additions
don't break callers.

**Units.** Compute credits are USD-denominated and stored on the wire as
integer **micro-USD (µUSD)**: ``1 USD = 1_000_000 µUSD``. Every ``*_micro_usd``
field is an ``int`` count of µUSD. Each model also exposes ``*_usd`` convenience
properties that divide by 1,000,000 to give a ``float`` dollar amount for
display.

Ledger ``amount`` (µUSD) is a signed integer: **negative** for debits
(operations that consume credit) and **positive** for credits/refunds (top-ups,
training-overcharge refunds, monthly resets).

The legacy whole-dollar integer fields (``credits_remaining``,
``credits_monthly_allowance``, ``credits_per_unit``, ``total_credits``,
``minimum``) are **deprecated** - they're retained for backward compatibility
and default to ``0``. Prefer the ``*_micro_usd`` fields / ``*_usd`` properties.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_MICRO_USD_PER_USD = 1_000_000


class CreditLedgerEntry(BaseModel):
    """A single entry in the organization's credit ledger.

    ``amount`` and ``balance_after`` are **micro-USD (µUSD)** integers
    (``1 USD = 1_000_000 µUSD``).

    ``amount`` sign carries the direction:

    - ``amount < 0`` - debit (operation consumed credit)
    - ``amount > 0`` - credit or refund (top-up, training overcharge refund)

    ``balance_after`` is the org's remaining balance (µUSD) immediately after
    this entry posted - useful for time-travel charts without recomputing from
    the ledger head.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    operation: str = Field(
        description="Operation slug, such as 'training_a10g' or 'sam3_auto_annotation'.",
    )
    amount: int = Field(
        description="Micro-USD: negative for a charge, positive for a top-up, reset or refund."
    )
    balance_after: int | None = Field(
        default=None,
        description="What could be spent right after this entry, in micro-USD.",
    )
    description: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime


class CreditBalance(BaseModel):
    """An organization's compute wallet, in micro-USD (µUSD).

    The plan grants a monthly ``included_allowance_micro_usd`` of compute credit;
    ``included_remaining_micro_usd`` is what's left of it this period. Beyond the
    allowance, spend draws against an optional ``budget_micro_usd`` overage cap
    (``period_overage_micro_usd`` is the overage incurred so far,
    ``budget_remaining_micro_usd`` what's left under the cap).
    """

    model_config = ConfigDict(extra="ignore")

    # ── micro-USD wire fields (authoritative) ──
    included_allowance_micro_usd: int = Field(
        default=0,
        description="The plan's monthly allowance.",
    )
    included_remaining_micro_usd: int = Field(
        default=0,
        description="Allowance and prepaid credit left to spend this period.",
    )
    budget_micro_usd: int = Field(
        default=0,
        description="Monthly spend cap, allowance included; equals the allowance with pay-as-you-go off.",
    )
    period_spend_micro_usd: int = Field(
        default=0,
        description="Spent this period.",
    )
    period_overage_micro_usd: int = Field(
        default=0,
        description="Spent beyond the allowance this period.",
    )
    budget_remaining_micro_usd: int = Field(
        default=0,
        description="Left under the spend cap this period.",
    )
    period_start: datetime | None = Field(
        default=None,
        description="Start of the current billing period.",
    )
    credits_reset_at: datetime | None = Field(
        default=None,
        description="When the monthly allowance resets next.",
    )

    # ── deprecated whole-dollar integer fields (back-compat) ──
    credits_remaining: int = Field(
        default=0,
        description="DEPRECATED: whole-dollar remaining credit. Use included_remaining_micro_usd.",
    )
    credits_monthly_allowance: int = Field(
        default=0,
        description=(
            "DEPRECATED: whole-dollar monthly allowance. Use included_allowance_micro_usd."
        ),
    )

    recent_history: list[CreditLedgerEntry] = Field(
        default_factory=list,
        description="The last 20 ledger entries, newest first.",
    )

    # ── USD convenience properties ──
    @property
    def remaining_usd(self) -> float:
        """Remaining included allowance in USD (``included_remaining_micro_usd / 1e6``)."""
        return self.included_remaining_micro_usd / _MICRO_USD_PER_USD

    @property
    def allowance_usd(self) -> float:
        """Monthly included allowance in USD (``included_allowance_micro_usd / 1e6``)."""
        return self.included_allowance_micro_usd / _MICRO_USD_PER_USD

    @property
    def budget_usd(self) -> float:
        """Overage budget cap in USD (``budget_micro_usd / 1e6``)."""
        return self.budget_micro_usd / _MICRO_USD_PER_USD

    @property
    def overage_usd(self) -> float:
        """Overage incurred this period in USD (``period_overage_micro_usd / 1e6``)."""
        return self.period_overage_micro_usd / _MICRO_USD_PER_USD

    @property
    def spend_usd(self) -> float:
        """Total spend this period in USD (``period_spend_micro_usd / 1e6``)."""
        return self.period_spend_micro_usd / _MICRO_USD_PER_USD

    @property
    def budget_remaining_usd(self) -> float:
        """Remaining overage budget in USD (``budget_remaining_micro_usd / 1e6``)."""
        return self.budget_remaining_micro_usd / _MICRO_USD_PER_USD


class CreditEstimate(BaseModel):
    """What a planned operation would cost, in micro-USD (µUSD), and whether the balance covers it."""

    model_config = ConfigDict(extra="ignore")

    operation: str
    unit: str = Field(description="The unit priced, such as 'minute', 'image' or 'run'.")
    quantity: int

    # ── micro-USD wire fields (authoritative) ──
    micro_usd_per_unit: int = Field(
        default=0,
        description="Price per unit.",
    )
    total_micro_usd: int = Field(
        default=0,
        description="``micro_usd_per_unit * quantity``.",
    )
    remaining_micro_usd: int = Field(
        default=0,
        description="What the organization could spend when this was estimated.",
    )
    sufficient: bool = Field(
        default=False,
        description="Whether the balance covers ``total_micro_usd``.",
    )

    # ── deprecated whole-dollar integer fields (back-compat) ──
    credits_per_unit: int = Field(
        default=0,
        description="DEPRECATED: whole-dollar per-unit cost. Use micro_usd_per_unit.",
    )
    total_credits: int = Field(
        default=0,
        description="DEPRECATED: whole-dollar total cost. Use total_micro_usd.",
    )
    minimum: int = Field(
        default=0,
        description="DEPRECATED: whole-dollar floor on total. No longer used (always 0).",
    )
    credits_remaining: int = Field(
        default=0,
        description="DEPRECATED: whole-dollar remaining balance. Use remaining_micro_usd.",
    )

    # ── USD convenience property ──
    @property
    def total_usd(self) -> float:
        """Total estimated cost in USD (``total_micro_usd / 1e6``)."""
        return self.total_micro_usd / _MICRO_USD_PER_USD

    @property
    def per_unit_usd(self) -> float:
        """Per-unit cost in USD (``micro_usd_per_unit / 1e6``)."""
        return self.micro_usd_per_unit / _MICRO_USD_PER_USD

    @property
    def remaining_usd(self) -> float:
        """Spendable compute credit in USD (``remaining_micro_usd / 1e6``)."""
        return self.remaining_micro_usd / _MICRO_USD_PER_USD


class OperationUsage(BaseModel):
    """Aggregated compute spend for a single operation slug over a window.

    ``total_micro_usd`` is the summed debit (µUSD) across ``event_count`` events.
    """

    model_config = ConfigDict(extra="ignore")

    operation: str = Field(description="Operation slug, such as 'training_a10g'.")
    total_micro_usd: int = Field(default=0, description="Total spend on this operation (µUSD).")
    event_count: int = Field(default=0, description="Charges for this operation.")

    @property
    def total_usd(self) -> float:
        """Total spend on this operation in USD (``total_micro_usd / 1e6``)."""
        return self.total_micro_usd / _MICRO_USD_PER_USD


class UsageByOperation(BaseModel):
    """Per-operation compute-spend breakdown over a rolling window.

    Returned by :meth:`pictograph.resources.credits.Credits.usage_by_operation`.
    ``operations`` is ordered by spend descending; ``total_micro_usd`` /
    ``total_events`` are the window totals across every operation.
    """

    model_config = ConfigDict(extra="ignore")

    range: str = Field(description="Window: 'day' | 'week' | 'month' | 'all'.")
    since: str | None = Field(default=None, description="ISO timestamp the window starts at.")
    operations: list[OperationUsage] = Field(default_factory=list)
    total_micro_usd: int = Field(default=0, description="Total spend across the window (µUSD).")
    total_events: int = Field(default=0, description="Charges across the window.")

    @property
    def total_usd(self) -> float:
        """Total spend across the window in USD (``total_micro_usd / 1e6``)."""
        return self.total_micro_usd / _MICRO_USD_PER_USD
