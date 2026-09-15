"""Credits resource - balance, ledger history, pre-operation cost estimation.

Compute credits are USD-denominated and carried on the wire as integer
**micro-USD (µUSD)**: ``1 USD = 1_000_000 µUSD``. Balance/estimate models
expose both the raw ``*_micro_usd`` integers and ``*_usd`` float properties.

This is the gating surface for agents: call :meth:`Credits.estimate` before
any operation that consumes credit (training, batch SAM3, image
generation/edit) to decide whether to proceed. The estimate's
``sufficient`` flag tells you whether the org has enough compute credit **at
the moment of the call** - there's no race-free guarantee against another
caller draining the balance between the estimate and the actual operation, so
the SDK and backend both treat the operation's own
:class:`PaymentRequiredError` as the authoritative answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from pictograph._http.pagination import OffsetPager
from pictograph.models.credit import (
    CreditBalance,
    CreditEstimate,
    CreditLedgerEntry,
    UsageByOperation,
)
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Mapping

UsageRange = Literal["day", "week", "month", "all"]

_API_PATH = "/api/v1/developer/credits"


class Credits(Resource):
    """Read credit balance and ledger; estimate operation costs."""

    def balance(self) -> CreditBalance:
        """Current compute-credit state + last 20 ledger entries.

        Returns a :class:`CreditBalance` with USD-denominated micro-USD (µUSD)
        fields (``included_remaining_micro_usd``, ``budget_micro_usd``, etc.)
        and ``*_usd`` float convenience properties (``.remaining_usd``,
        ``.allowance_usd``, ``.budget_usd``, ``.overage_usd``).
        """
        response = self._transport.request("GET", f"{_API_PATH}/balance")
        return self._parse(CreditBalance, response)

    def history(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[CreditLedgerEntry]:
        """Single page of credit-ledger entries, newest first.

        Each entry's ``amount`` / ``balance_after`` are signed micro-USD (µUSD)
        integers (``1 USD = 1_000_000 µUSD``).

        Args:
            limit: Page size (backend cap: 100).
            offset: Page offset for paginating manually.
        """
        response = self._transport.request(
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
    ) -> OffsetPager[CreditLedgerEntry]:
        """Auto-paging iterator across the entire credit ledger."""

        def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            return cast(
                "Mapping[str, Any]",
                self._transport.request(
                    "GET",
                    f"{_API_PATH}/history",
                    params={"offset": offset, "limit": limit},
                ),
            )

        return OffsetPager(
            fetch,
            items_key="entries",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(CreditLedgerEntry, raw),
        )

    def usage_by_operation(self, *, range: UsageRange = "month") -> UsageByOperation:
        """Per-operation compute spend (µUSD, debits only) over a rolling window.

        DB-side aggregation - exact totals regardless of ledger size, ordered by
        spend descending. Use it to see where compute credit is going (training
        vs. SAM3 vs. image generation) without paging the full ledger.

        Args:
            range: Window to aggregate over - ``"day"``, ``"week"``,
                ``"month"`` (default), or ``"all"``.

        Returns:
            :class:`UsageByOperation` - ``operations`` (each with
            ``total_micro_usd`` / ``total_usd`` + ``event_count``) plus the
            window totals ``total_micro_usd`` / ``total_events``.
        """
        response = self._transport.request(
            "GET",
            f"{_API_PATH}/usage-by-operation",
            params={"range": range},
        )
        return self._parse(UsageByOperation, response)

    def estimate(self, operation: str, *, quantity: int = 1) -> CreditEstimate:
        """Estimate the compute-credit cost of a planned operation (USD / µUSD).

        Args:
            operation: Operation slug. Common values:
              ``"training_a10g"``, ``"training_a100"``, ``"training_h100"``,
              ``"sam3_auto_annotation"``, ``"inference_t4"``,
              ``"image_generate_imagen_fast"``, ``"image_edit_gemini_flash"``.
              The full slug list is defined server-side.
            quantity: Number of units (minutes for ``per_minute`` ops, calls
                for ``per_call`` ops).

        Returns:
            :class:`CreditEstimate` - inspect ``.sufficient`` to gate the
            operation, and ``.total_micro_usd`` / ``.total_usd`` for the cost.
            Note that the underlying operation may still raise
            :class:`PaymentRequiredError` if the balance drains between the
            estimate and the call (race window is small but real).

        Raises:
            ValidationError: ``operation`` is not a known credit operation.
        """
        response = self._transport.request(
            "GET",
            f"{_API_PATH}/estimate",
            params={"operation": operation, "quantity": quantity},
        )
        return self._parse(CreditEstimate, response)
