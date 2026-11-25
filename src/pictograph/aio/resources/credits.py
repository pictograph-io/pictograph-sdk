"""Async Credits resource - balance, ledger history, pre-operation cost estimate.

Async twin of :class:`pictograph.resources.credits.Credits`. Compute credits are
USD-denominated micro-USD (µUSD) on the wire (``1 USD = 1_000_000 µUSD``). Use
:meth:`AsyncCredits.estimate` to gate a planned operation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from pictograph._http.pagination import AsyncOffsetPager
from pictograph.models.credit import (
    CreditBalance,
    CreditEstimate,
    CreditLedgerEntry,
    UsageByOperation,
)
from pictograph.resources._base import AsyncResource

if TYPE_CHECKING:
    from collections.abc import Mapping

UsageRange = Literal["day", "week", "month", "all"]

_API_PATH = "/api/v1/developer/credits"


class AsyncCredits(AsyncResource):
    """Read credit balance and ledger; estimate operation costs (async)."""

    async def balance(self) -> CreditBalance:
        """Current compute-credit state + last 20 ledger entries (µUSD fields)."""
        response = await self._transport.request("GET", f"{_API_PATH}/balance")
        return self._parse(CreditBalance, response)

    async def history(self, *, limit: int = 50, offset: int = 0) -> list[CreditLedgerEntry]:
        """Single page of credit-ledger entries, newest first.

        Each entry's ``amount`` / ``balance_after`` are signed micro-USD (µUSD)
        integers (``1 USD = 1_000_000 µUSD``).

        Args:
            limit: Page size (backend cap: 100).
            offset: Page offset for paginating manually.
        """
        response = await self._transport.request(
            "GET",
            f"{_API_PATH}/history",
            params={"limit": limit, "offset": offset},
        )
        return self._parse_list(CreditLedgerEntry, response.get("entries", []))

    def iter(
        self,
        *,
        page_size: int = 100,
        max_total: int | None = None,
    ) -> AsyncOffsetPager[CreditLedgerEntry]:
        """Auto-paging async iterator across the entire credit ledger."""

        async def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            return cast(
                "Mapping[str, Any]",
                await self._transport.request(
                    "GET",
                    f"{_API_PATH}/history",
                    params={"offset": offset, "limit": limit},
                ),
            )

        return AsyncOffsetPager(
            fetch,
            items_key="entries",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(CreditLedgerEntry, raw),
        )

    async def usage_by_operation(self, *, range: UsageRange = "month") -> UsageByOperation:
        """Per-operation compute spend (µUSD, debits only) over a rolling window.

        Args:
            range: Window - ``"day"``, ``"week"``, ``"month"`` (default), or ``"all"``.

        Returns:
            :class:`UsageByOperation` - ``operations`` (each with
            ``total_micro_usd`` / ``total_usd`` + ``event_count``) plus window
            totals ``total_micro_usd`` / ``total_events``.
        """
        response = await self._transport.request(
            "GET",
            f"{_API_PATH}/usage-by-operation",
            params={"range": range},
        )
        return self._parse(UsageByOperation, response)

    async def estimate(self, operation: str, *, quantity: int = 1) -> CreditEstimate:
        """Estimate the compute-credit cost of a planned operation (USD / µUSD).

        Args:
            operation: Operation slug (e.g. ``"training_a10g"``,
                ``"sam3_auto_annotation"``, ``"inference_t4"``).
            quantity: Number of units (minutes for per-minute ops, calls for
                per-call ops).

        Returns:
            :class:`CreditEstimate` - inspect ``.sufficient`` to gate, and
            ``.total_micro_usd`` / ``.total_usd`` for the cost. The operation may
            still raise :class:`PaymentRequiredError` if the balance drains
            between the estimate and the call.

        Raises:
            ValidationError: ``operation`` is not a known credit operation.
        """
        response = await self._transport.request(
            "GET",
            f"{_API_PATH}/estimate",
            params={"operation": operation, "quantity": quantity},
        )
        return self._parse(CreditEstimate, response)
