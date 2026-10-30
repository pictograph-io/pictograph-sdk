"""Async Training resource - create, list, get, cancel, wait_for_completion.

Async twin of :class:`pictograph.resources.training.Training`. Training is
asynchronous server-side; ``wait=True`` polls until a terminal status.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast

from pictograph._http.pagination import AsyncOffsetPager
from pictograph.exceptions import ApiError, PollTimeoutError, job_end_code
from pictograph.models.common import BulkActionResult
from pictograph.models.training import (
    GpuType,
    PipelineType,
    TrainingQuote,
    TrainingRun,
    TrainingStatus,
)
from pictograph.resources._base import AsyncResource
from pictograph.resources.training import _plan_body, _single_path

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from pathlib import Path

_API_PATH = "/api/v1/developer/training/"
_DEFAULT_POLL_INTERVAL = 5.0
_DEFAULT_TIMEOUT = 7200.0  # 2h - the training service's own hard cap
_TERMINAL_STATUSES: frozenset[TrainingStatus] = frozenset({"completed", "failed", "cancelled"})


class AsyncTraining(AsyncResource):
    """Operations on training runs (async)."""

    async def create(
        self,
        dataset: str,
        export_name: str,
        *,
        pipeline_type: PipelineType,
        name: str,
        config: dict[str, Any] | str | Path | None = None,
        gpu_type: GpuType = "a10g",
        version_of: str | None = None,
        wait: bool = True,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> TrainingRun:
        """Spawn a new training run.

        The backend resolves dataset + export by name, validates the export is
        ``completed``, gates on a minimum spendable balance, and submits the GPU
        job (charged once, on success, for the GPU time it used - see the sync
        twin). Returns the :class:`TrainingRun` - pending if ``wait=False``,
        terminal otherwise.

        Args:
            dataset: Project name within your organization.
            export_name: Completed export's name (within the dataset).
            pipeline_type: ``"yolox"``, ``"sm_pytorch"``, ``"classification"``,
                ``"rfdetr_detection"``, ``"rfdetr_segmentation"``, or ``"rfdetr_keypoint"``.
            name: Human-readable label (1-100 chars).
            config: Pipeline hyperparameters (``epochs``, ``batch_size``,
                ``model_size``, ``image_width`` / ``image_height``, …; a key left out
                takes the web app's default - see the sync twin),
                or a path to a JSON file - e.g. a downloaded
                ``config.json`` (the round-trip; see the sync twin).
            gpu_type: ``"a10g"`` (default), ``"a100"``, ``"h100"``, or ``"auto"``
                (see the sync twin). Training runs on ONE GPU, always.
            version_of: Append the result to this EXISTING model as
                a new version instead of minting a new model - see the
                sync twin.
            wait: Poll until terminal status (default ``True``).
            poll_interval: Seconds between checks (default 5s).
            timeout: Max seconds to wait (default 7200 = 2h).

        Raises:
            NotFoundError: ``dataset`` or ``export_name`` doesn't exist.
            ValidationError: Export not ``completed`` or bad gpu/pipeline type.
            PaymentRequiredError: Below the minimum spendable balance to start.
            ApiError: Run spawned but reached ``failed`` (when ``wait=True``).
            PollTimeoutError: ``timeout`` elapsed (the run is still executing).
        """
        body = _plan_body(dataset, export_name, pipeline_type, config, gpu_type, version_of)
        body["name"] = name
        response = await self._transport.request("POST", _API_PATH, json=body)
        run = self._parse(TrainingRun, response["data"])
        if not wait:
            return run
        return await self.wait_for_completion(run.id, poll_interval=poll_interval, timeout=timeout)

    async def quote(
        self,
        dataset: str,
        export_name: str,
        *,
        pipeline_type: PipelineType,
        config: dict[str, Any] | str | Path | None = None,
        gpu_type: GpuType = "a10g",
        version_of: str | None = None,
    ) -> TrainingQuote:
        """Check and price a training run without starting it (see the sync twin).

        Runs every check :meth:`create` runs on the same arguments and raises what it
        would raise; reports the GPU, the per-class training support, the estimate,
        the balance, ``can_start`` and ``blocking``. Writes nothing, charges nothing.
        """
        body = _plan_body(dataset, export_name, pipeline_type, config, gpu_type, version_of)
        response = await self._transport.request("POST", f"{_API_PATH}quote", json=body)
        return self._parse(TrainingQuote, response["data"])

    async def list(
        self,
        *,
        dataset: str | None = None,
        status: TrainingStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TrainingRun]:
        """Single-page list of training runs in your organization."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if dataset is not None:
            params["dataset_name"] = dataset
        if status is not None:
            params["status"] = status
        response = await self._transport.request("GET", _API_PATH, params=params)
        return self._parse_list(TrainingRun, response.get("data", []))

    def iter(
        self,
        *,
        dataset: str | None = None,
        status: TrainingStatus | None = None,
        page_size: int = 50,
        max_total: int | None = None,
    ) -> AsyncOffsetPager[TrainingRun]:
        """Auto-paging async iterator across every training run in your org."""
        base: dict[str, Any] = {}
        if dataset is not None:
            base["dataset_name"] = dataset
        if status is not None:
            base["status"] = status

        async def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            params = {**base, "offset": offset, "limit": limit}
            return cast(
                "Mapping[str, Any]",
                await self._transport.request("GET", _API_PATH, params=params),
            )

        return AsyncOffsetPager(
            fetch,
            items_key="data",
            page_size=page_size,
            max_total=max_total,
            parse_item=lambda raw: self._parse(TrainingRun, raw),
        )

    async def get(
        self,
        run: str,
    ) -> TrainingRun:
        """Fetch a training run by name.

        Run names are not unique - a by-name lookup returns the most recent run
        of that name.
        """
        response = await self._transport.request("GET", _single_path(run))
        return self._parse(TrainingRun, response["data"])

    async def cancel(
        self,
        run: str,
    ) -> TrainingRun:
        """Cancel a running training job.

        The backend stops the GPU job in-flight and marks the run
        ``cancelled``; a run cancelled before completion is never charged (a
        legacy pre-charged run is refunded once). By-name cancels the most
        recent run of that name. Member+ API key.
        """
        response = await self._transport.request("POST", _single_path(run, "/cancel"))
        return self._parse(TrainingRun, response["data"])

    async def bulk_cancel(self, runs: Sequence[str]) -> BulkActionResult:
        """Cancel many training runs in one org-scoped server-side call.

        Idempotent: duplicate ids are collapsed; ids that don't resolve in your
        org OR are already terminal land in
        :attr:`~pictograph.models.common.BulkActionResult.not_found` rather than
        raising. Requires member+ role.
        """
        response = await self._transport.request(
            "POST", f"{_API_PATH}bulk-cancel", json={"run_ids": list(runs)}
        )
        return self._parse(BulkActionResult, response.get("data", response))

    async def wait_for_completion(
        self,
        run: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> TrainingRun:
        """Poll a training current until it reaches a terminal status.

        Args:
            poll_interval: Seconds between checks (default 5s).
            timeout: Max seconds to wait (default 7200 = 2h).
            sleep: Override the async sleep function (testing hook).

        Raises:
            ApiError: The current reached ``failed`` or ``cancelled`` status.
            PollTimeoutError: ``timeout`` elapsed (the current is still executing).
        """
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else asyncio.sleep
        deadline = time.monotonic() + timeout
        while True:
            current = await self.get(run)
            if current.status == "completed":
                return current
            if current.status in ("failed", "cancelled"):
                raise ApiError(
                    f"Training run '{current.name}' ended with status "
                    f"'{current.status}': {current.error_message or 'no error message provided'}",
                    response=current.model_dump(mode="json"),
                    code=job_end_code(current.status),
                )
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Training run '{current.name}' did not complete within "
                    f"{timeout:.0f}s (last status: {current.status}). The run is "
                    f"still executing - fetch later via client.training.get(...)."
                )
            await sleep_fn(poll_interval)
