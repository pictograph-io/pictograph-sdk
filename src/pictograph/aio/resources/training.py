"""Async Training resource - create, list, get, cancel, wait_for_completion.

Async twin of :class:`pictograph.resources.training.Training`. Training is
asynchronous server-side; ``wait=True`` polls until a terminal status.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast

from pictograph._http.pagination import AsyncOffsetPager
from pictograph.exceptions import ApiError, PollTimeoutError
from pictograph.models.common import BulkActionResult
from pictograph.models.training import GpuType, PipelineType, TrainingRun, TrainingStatus
from pictograph.resources._base import AsyncResource
from pictograph.resources.training import _load_config, _single_path

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
        dataset_name: str,
        export_name: str,
        *,
        pipeline_type: PipelineType,
        name: str,
        config: dict[str, Any] | str | Path | None = None,
        gpu_type: GpuType = "a10g",
        gpu_count: int = 1,
        version_of_model_id: str | None = None,
        wait: bool = True,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> TrainingRun:
        """Spawn a new training run.

        The backend resolves dataset + export by name, validates the export is
        ``completed``, deducts the estimated credit cost, and submits the GPU
        job. Returns the :class:`TrainingRun` - pending if ``wait=False``,
        terminal otherwise.

        Args:
            dataset_name: Project name within your organization.
            export_name: Completed export's name (within the dataset).
            pipeline_type: ``"yolox"``, ``"sm_pytorch"``, ``"classification"``,
                ``"rfdetr_detection"``, ``"rfdetr_segmentation"``, or ``"rfdetr_keypoint"``.
            name: Human-readable label (1-100 chars).
            config: Pipeline hyperparameters (``epochs``, ``batch_size``, …),
                or a path to a JSON file - e.g. a downloaded
                ``config.json`` (the round-trip; see the sync twin).
            gpu_type: ``"a10g"`` (default), ``"a100"``, or ``"h100"``.
            gpu_count: GPUs in the training container (1-4, RF-DETR pipelines
                only). ``>1`` trains with DDP and bills ``gpu_count x`` the
                per-second GPU rate. It existed only on the SYNC twin until
                2026-07-31, so the async client could not request multi-GPU
                training at all.
            version_of_model_id: Append the result to this EXISTING model as
                a new version instead of minting a new model - see the
                sync twin.
            wait: Poll until terminal status (default ``True``).
            poll_interval: Seconds between checks (default 5s).
            timeout: Max seconds to wait (default 7200 = 2h).

        Raises:
            NotFoundError: ``dataset_name`` or ``export_name`` doesn't exist.
            ValidationError: Export not ``completed`` or bad gpu/pipeline type.
            PaymentRequiredError: Insufficient credits for the estimated cost.
            ApiError: Run spawned but reached ``failed`` (when ``wait=True``).
            PollTimeoutError: ``timeout`` elapsed (the run is still executing).
        """
        body: dict[str, Any] = {
            "dataset_name": dataset_name,
            "export_name": export_name,
            "pipeline_type": pipeline_type,
            "name": name,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "config": _load_config(config),
        }
        if version_of_model_id is not None:
            body["version_of_model_id"] = version_of_model_id
        response = await self._transport.request("POST", _API_PATH, json=body)
        run = self._parse(TrainingRun, response["data"])
        if not wait:
            return run
        return await self.wait_for_completion(run.id, poll_interval=poll_interval, timeout=timeout)

    async def list(
        self,
        *,
        dataset_name: str | None = None,
        status: TrainingStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TrainingRun]:
        """Single-page list of training runs in your organization."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if dataset_name is not None:
            params["dataset_name"] = dataset_name
        if status is not None:
            params["status"] = status
        response = await self._transport.request("GET", _API_PATH, params=params)
        return self._parse_list(TrainingRun, response.get("data", []))

    def iter(
        self,
        *,
        dataset_name: str | None = None,
        status: TrainingStatus | None = None,
        page_size: int = 50,
        max_total: int | None = None,
    ) -> AsyncOffsetPager[TrainingRun]:
        """Auto-paging async iterator across every training run in your org."""
        base: dict[str, Any] = {}
        if dataset_name is not None:
            base["dataset_name"] = dataset_name
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

    async def get(self, name: str | None = None, *, run_id: str | None = None) -> TrainingRun:
        """Fetch a training run by name (or ``run_id=`` UUID).

        Run names are not unique - a by-name lookup returns the most recent run
        of that name.
        """
        response = await self._transport.request("GET", _single_path(name, run_id))
        return self._parse(TrainingRun, response["data"])

    async def cancel(self, name: str | None = None, *, run_id: str | None = None) -> TrainingRun:
        """Cancel a running training job (by name or ``run_id=`` UUID).

        The backend stops the GPU job in-flight and marks the run
        ``cancelled``; a run cancelled before completion is never charged (a
        legacy pre-charged run is refunded once). By-name cancels the most
        recent run of that name. Member+ API key.
        """
        response = await self._transport.request("POST", _single_path(name, run_id, "/cancel"))
        return self._parse(TrainingRun, response["data"])

    async def bulk_cancel(self, run_ids: Sequence[str]) -> BulkActionResult:
        """Cancel many training runs in one org-scoped server-side call.

        Idempotent: duplicate ids are collapsed; ids that don't resolve in your
        org OR are already terminal land in
        :attr:`~pictograph.models.common.BulkActionResult.not_found` rather than
        raising. Requires member+ role.
        """
        response = await self._transport.request(
            "POST", f"{_API_PATH}bulk-cancel", json={"run_ids": list(run_ids)}
        )
        return self._parse(BulkActionResult, response.get("data", response))

    async def wait_for_completion(
        self,
        name: str | None = None,
        *,
        run_id: str | None = None,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> TrainingRun:
        """Poll a training run until it reaches a terminal status.

        Args:
            run_id: Training run UUID.
            poll_interval: Seconds between checks (default 5s).
            timeout: Max seconds to wait (default 7200 = 2h).
            sleep: Override the async sleep function (testing hook).

        Raises:
            ApiError: The run reached ``failed`` or ``cancelled`` status.
            PollTimeoutError: ``timeout`` elapsed (the run is still executing).
        """
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else asyncio.sleep
        deadline = time.monotonic() + timeout
        while True:
            run = await self.get(name, run_id=run_id)
            if run.status == "completed":
                return run
            if run.status in ("failed", "cancelled"):
                raise ApiError(
                    f"Training run '{run.name}' ended with status "
                    f"'{run.status}': {run.error_message or 'no error message provided'}",
                    response=run.model_dump(mode="json"),
                )
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Training run '{run.name}' did not complete within "
                    f"{timeout:.0f}s (last status: {run.status}). The run is "
                    f"still executing - fetch later via client.training.get(...)."
                )
            await sleep_fn(poll_interval)
