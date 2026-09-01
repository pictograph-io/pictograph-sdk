"""Async Connectors resource - V7 (Darwin) + Roboflow dataset import.

Async twin of :class:`pictograph.resources.connectors.Connectors`. Three stages:
:meth:`validate` → :meth:`check_limits` → :meth:`import_` (poll until terminal).
The 3rd-party API key only lives in memory during the call.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from pictograph.exceptions import ApiError, PollTimeoutError
from pictograph.models.connector import (
    ConnectorProvider,
    ImportJob,
    ImportStatus,
    LimitCheckResult,
    RemoteDataset,
    ValidationResult,
)
from pictograph.resources._base import AsyncResource

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

_API_PATH = "/api/v1/developer/connectors"
_DEFAULT_POLL_INTERVAL = 3.0
_DEFAULT_TIMEOUT = 3600.0  # 1h - large imports can take this long
_TERMINAL_STATUSES: frozenset[ImportStatus] = frozenset({"completed", "error", "cancelled"})


class AsyncConnectors(AsyncResource):
    """Validate source providers + import datasets from V7 / Roboflow (async)."""

    # ───────────── validate / check ─────────────

    async def validate(self, provider: ConnectorProvider, api_key: str) -> ValidationResult:
        """Verify a source-provider API key and list available datasets (no quota).

        Args:
            provider: ``"v7"`` or ``"roboflow"``.
            api_key: The source provider's API key (V7 token, Roboflow key).

        Returns:
            :class:`ValidationResult`. Inspect ``valid`` first; ``datasets`` is
            populated only on success.
        """
        body = {"provider": provider, "api_key": api_key}
        response = await self._transport.request("POST", f"{_API_PATH}/validate", json=body)
        return self._parse(ValidationResult, response)

    async def check_limits(
        self, *, total_images: int, estimated_size_bytes: int
    ) -> LimitCheckResult:
        """Preflight tier-cap check before kicking off an import.

        Returns ``allowed=True`` when the import fits under the org's current
        ``max_images`` + ``max_storage_bytes`` caps; else inspect ``exceeded``.
        """
        body = {
            "total_images": total_images,
            "estimated_size_bytes": estimated_size_bytes,
        }
        response = await self._transport.request("POST", f"{_API_PATH}/check-limits", json=body)
        return self._parse(LimitCheckResult, response)

    # ───────────── import ─────────────

    async def import_(
        self,
        provider: ConnectorProvider,
        api_key: str,
        datasets: Sequence[RemoteDataset | dict[str, Any]],
        *,
        wait: bool = True,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> ImportJob:
        """Import one or more datasets from the source provider.

        Args:
            provider: Source provider.
            api_key: Provider API key (used only on this call).
            datasets: Datasets to import - :class:`RemoteDataset` instances
                (from :meth:`validate`) or raw dicts.
            wait: Poll until terminal status (default ``True``). ``False`` returns
                immediately for caller-side polling.
            poll_interval: Seconds between status checks (default 3s).
            timeout: Max seconds to wait (default 3600 = 1h).

        Raises:
            PaymentRequiredError: Tier cap exceeded.
            ValidationError: ``provider`` invalid, ``api_key`` rejected, or empty.
            ApiError: Job ended in ``error`` (when ``wait=True``).
            PollTimeoutError: Deadline elapsed (when ``wait=True``).
        """
        body: dict[str, Any] = {
            "provider": provider,
            "api_key": api_key,
            "datasets": [
                d.model_dump(exclude_none=True) if isinstance(d, RemoteDataset) else dict(d)
                for d in datasets
            ],
        }
        response = await self._transport.request("POST", f"{_API_PATH}/import/start", json=body)
        kicker = ImportJob(
            import_id=response["import_id"],
            status="processing",
            progress=0.0,
        )
        if not wait:
            return kicker
        return await self.wait_for_import(
            kicker.import_id, poll_interval=poll_interval, timeout=timeout
        )

    async def get_import(self, import_id: str) -> ImportJob:
        """Fetch the current state of an import."""
        response = await self._transport.request("GET", f"{_API_PATH}/import/status/{import_id}")
        return self._parse(ImportJob, response)

    async def cancel_import(self, import_id: str) -> ImportJob:
        """Soft-cancel an in-flight import (already-imported images are kept)."""
        response = await self._transport.request("POST", f"{_API_PATH}/import/cancel/{import_id}")
        if "status" in response and "import_id" in response:
            return await self.get_import(response["import_id"])
        return self._parse(ImportJob, response)

    async def wait_for_import(
        self,
        import_id: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> ImportJob:
        """Poll an import until terminal status."""
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else asyncio.sleep
        deadline = time.monotonic() + timeout
        while True:
            job = await self.get_import(import_id)
            if job.status == "completed":
                return job
            if job.status == "cancelled":
                return job
            if job.status == "error":
                raise ApiError(
                    f"Import {import_id} failed (imported={job.imported_images}, "
                    f"failed={job.failed_images})",
                    response=job.model_dump(mode="json"),
                )
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Import {import_id} did not complete within {timeout:.0f}s "
                    f"(status: {job.status}, progress: {job.progress:.0f}%). "
                    f"Fetch later via client.connectors.get_import(...)."
                )
            await sleep_fn(poll_interval)
