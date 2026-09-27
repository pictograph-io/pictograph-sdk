"""Connectors resource - V7 (Darwin) + Roboflow dataset import.

Three-stage workflow:

1. :meth:`Connectors.validate` - verify the source provider's API key
   and list importable datasets.
2. :meth:`Connectors.check_limits` - preflight the org's tier capacity.
3. :meth:`Connectors.import_` - kick off the import. Pass ``wait=True``
   (default) to poll until terminal status; ``wait=False`` returns the
   :class:`ImportJob` immediately for caller-side polling.

Polling beats SSE here for two reasons: (1) the serving infrastructure cuts
long-lived connections, and (2) imports are minutes-long but the polling cost is
trivial (one cheap GET every few seconds). Same pattern as
:class:`pictograph.resources.training.Training`.

The 3rd-party API key only lives in memory during the call - it's never
persisted and not echoed back in the SDK or backend logs.
"""

from __future__ import annotations

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
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_API_PATH = "/api/v1/developer/connectors"
_DEFAULT_POLL_INTERVAL = 3.0
_DEFAULT_TIMEOUT = 3600.0  # 1h - large imports can take this long
_TERMINAL_STATUSES: frozenset[ImportStatus] = frozenset({"completed", "error", "cancelled"})


class Connectors(Resource):
    """Validate source providers + import datasets from V7 / Roboflow."""

    # ───────────── validate / check ─────────────

    def validate(self, provider: ConnectorProvider, api_key: str) -> ValidationResult:
        """Verify a source-provider API key and list available datasets.

        No quota consumed. The ``api_key`` is sent only on this call.

        Args:
            provider: ``"v7"`` or ``"roboflow"``.
            api_key: The source provider's API key (V7 token, Roboflow key).

        Returns:
            :class:`ValidationResult`. Inspect ``valid`` first;
            ``datasets`` is populated only on success.
        """
        body = {"provider": provider, "api_key": api_key}
        response = self._transport.request("POST", f"{_API_PATH}/validate", json=body)
        return self._parse(ValidationResult, response)

    def check_limits(self, *, total_images: int, estimated_size_bytes: int) -> LimitCheckResult:
        """Preflight tier-cap check before kicking off an import.

        Returns ``allowed=True`` when the import fits under the org's
        current ``max_images`` + ``max_storage_bytes`` caps. Otherwise
        inspect ``exceeded`` (``"images"`` / ``"storage"`` / ``"both"``).
        """
        body = {
            "total_images": total_images,
            "estimated_size_bytes": estimated_size_bytes,
        }
        response = self._transport.request("POST", f"{_API_PATH}/check-limits", json=body)
        return self._parse(LimitCheckResult, response)

    # ───────────── import ─────────────

    def import_(
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

        Trailing underscore on the method name avoids shadowing the
        Python ``import`` keyword while keeping the verb-first naming
        consistent with the rest of the SDK.

        Args:
            provider: Source provider.
            api_key: Provider API key (used only on this call).
            datasets: Datasets to import. Pass :class:`RemoteDataset`
                instances (typically obtained from :meth:`validate`) or
                raw dicts with ``id``, ``name``, ``slug``, optional
                ``image_count`` / ``version``.
            wait: Poll until terminal status (default ``True``). Set
                ``False`` to fire-and-forget; poll later via
                :meth:`get_import` or :meth:`wait_for_import`.
            poll_interval: Seconds between status checks (default 3s).
            timeout: Max seconds to wait (default 3600 = 1h). Large V7
                exports can take 30+ minutes.

        Returns:
            :class:`ImportJob`. Terminal-state when ``wait=True``.

        Raises:
            PaymentRequiredError: Tier cap exceeded.
            ValidationError: ``provider`` invalid, ``api_key`` rejected
                upstream, or ``datasets`` empty.
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
        response = self._transport.request("POST", f"{_API_PATH}/import/start", json=body)
        # Backend kicker returns ``{import_id, status: "started", datasets: [...]}``.
        # Normalise into the polling shape so the caller always gets ImportJob.
        kicker = ImportJob(
            import_id=response["import_id"],
            status="processing",
            progress=0.0,
        )
        if not wait:
            return kicker
        return self.wait_for_import(kicker.import_id, poll_interval=poll_interval, timeout=timeout)

    def get_import(self, import_id: str) -> ImportJob:
        """Fetch the current state of an import."""
        response = self._transport.request("GET", f"{_API_PATH}/import/status/{import_id}")
        return self._parse(ImportJob, response)

    def cancel_import(self, import_id: str) -> ImportJob:
        """Soft-cancel an in-flight import.

        Already-imported images are kept in the destination dataset; the
        worker stops downloading new ones and the status transitions to
        ``cancelled``. The 3rd-party API key is not re-used.
        """
        response = self._transport.request("POST", f"{_API_PATH}/import/cancel/{import_id}")
        # Backend returns ``{status: "cancelled", import_id}`` - re-fetch
        # for full state.
        if "status" in response and "import_id" in response:
            return self.get_import(response["import_id"])
        return self._parse(ImportJob, response)

    def wait_for_import(
        self,
        import_id: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        sleep: Callable[[float], None] | None = None,
    ) -> ImportJob:
        """Poll an import until terminal status."""
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else time.sleep
        deadline = time.monotonic() + timeout
        while True:
            job = self.get_import(import_id)
            if job.status == "completed":
                return job
            if job.status == "cancelled":
                # Cancelled is terminal but not an error - return the snapshot.
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
            sleep_fn(poll_interval)
