"""Async model-evaluation resource - the async mirror of
:mod:`pictograph.resources.model_evaluations`."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast

from pictograph._http.pagination import AsyncOffsetPager
from pictograph.aio.resources import _resolve
from pictograph.exceptions import ApiError, PollTimeoutError
from pictograph.models.evaluation import ModelEvaluation
from pictograph.resources._base import AsyncResource

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

_API_PATH = "/api/v1/developer/model-evaluations"
_DEFAULT_POLL_INTERVAL = 3.0
_DEFAULT_TIMEOUT = 1800.0


class AsyncModelEvaluations(AsyncResource):
    """Create, poll, list, and cancel server-side model evaluations (async)."""

    async def create(
        self,
        model: str,
        dataset_name: str,
        export_name: str,
        *,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
    ) -> ModelEvaluation:
        """Start evaluating a model against one of a dataset's exports.

        The export's recorded image set + class filter define exactly which
        images and ground-truth annotations are scored - aligning
        evaluation with training, which also runs off exports."""
        body = {
            "model_id": await _resolve.model_id(self._transport, model),
            "export_id": await _resolve.export_id(self._transport, dataset_name, export_name),
            "iou_threshold": iou_threshold,
            "confidence_threshold": confidence_threshold,
        }
        response = await self._transport.request("POST", _API_PATH, json=body)
        return self._parse(ModelEvaluation, response["evaluation"])

    async def get(self, evaluation_id: str) -> ModelEvaluation:
        """Fetch one evaluation by id."""
        response = await self._transport.request("GET", f"{_API_PATH}/{evaluation_id}")
        return self._parse(ModelEvaluation, response["evaluation"])

    async def list(
        self,
        *,
        model: str | None = None,
        dataset_name: str | None = None,
        limit: int = 50,
    ) -> list[ModelEvaluation]:
        """List the org's evaluations (newest first), optionally by model / dataset."""
        params = _params(
            model_id=None if model is None else await _resolve.model_id(self._transport, model),
            project_id=(
                None
                if dataset_name is None
                else await _resolve.dataset_id(self._transport, dataset_name)
            ),
            limit=limit,
        )
        response = await self._transport.request("GET", _API_PATH, params=params)
        return self._parse_list(ModelEvaluation, response.get("evaluations", []))

    def iter(
        self,
        *,
        model: str | None = None,
        dataset_name: str | None = None,
        page_size: int = 50,
    ) -> AsyncOffsetPager[ModelEvaluation]:
        """Auto-paging async iterator over the org's evaluations.

        The filters resolve on the FIRST page, as :meth:`AsyncImages.iter` does -
        this is a plain ``def`` returning a pager, so there is nothing to await
        in until iteration starts.
        """
        model_uuid: str | None = None
        dataset_uuid: str | None = None

        async def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            nonlocal model_uuid, dataset_uuid
            if model is not None and model_uuid is None:
                model_uuid = await _resolve.model_id(self._transport, model)
            if dataset_name is not None and dataset_uuid is None:
                dataset_uuid = await _resolve.dataset_id(self._transport, dataset_name)
            params = _params(
                model_id=model_uuid, project_id=dataset_uuid, limit=limit, offset=offset
            )
            return cast(
                "Mapping[str, Any]",
                await self._transport.request("GET", _API_PATH, params=params),
            )

        return AsyncOffsetPager(
            fetch,
            items_key="evaluations",
            page_size=page_size,
            parse_item=lambda raw: self._parse(ModelEvaluation, raw),
        )

    async def cancel(self, evaluation_id: str) -> ModelEvaluation:
        """Cancel a running / pending evaluation."""
        response = await self._transport.request("POST", f"{_API_PATH}/{evaluation_id}/cancel")
        return self._parse(ModelEvaluation, response["evaluation"])

    async def report(self, evaluation_id: str) -> dict[str, Any]:
        """Download a completed evaluation's full metrics as a self-contained report.

        The async mirror of :meth:`pictograph.resources.model_evaluations.ModelEvaluations.report`:
        returns the same canonical JSON-serializable dict (schema-versioned model /
        dataset / metrics / best+worst images / warnings). The evaluation must be
        ``completed`` or a :class:`~pictograph.exceptions.ConflictError` is raised.
        """
        response = await self._transport.request("GET", f"{_API_PATH}/{evaluation_id}/export")
        return cast("dict[str, Any]", response["report"])

    async def wait_for_completion(
        self,
        evaluation_id: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> ModelEvaluation:
        """Poll an evaluation until it reaches a terminal status.

        Raises:
            ApiError: the evaluation ``failed`` or was ``cancelled``.
            PollTimeoutError: ``timeout`` elapsed before completion.
        """
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else asyncio.sleep
        deadline = time.monotonic() + timeout
        while True:
            ev = await self.get(evaluation_id)
            if ev.status == "completed":
                return ev
            if ev.status in ("failed", "cancelled"):
                raise ApiError(
                    f"Model evaluation '{ev.id}' ended with status '{ev.status}': "
                    f"{ev.error_message or 'no error message provided'}",
                    response=ev.model_dump(mode="json"),
                )
            if time.monotonic() >= deadline:
                raise PollTimeoutError(
                    f"Model evaluation '{ev.id}' did not complete within {timeout:.0f}s "
                    f"(last status: {ev.status}). Fetch later via "
                    f"client.model_evaluations.get(...)."
                )
            await sleep_fn(poll_interval)

    async def evaluate(
        self,
        model: str,
        dataset_name: str,
        export_name: str,
        *,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> ModelEvaluation:
        """Create an evaluation and await its completion (the one-call path)."""
        created = await self.create(
            model,
            dataset_name,
            export_name,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
        )
        return await self.wait_for_completion(
            created.id, poll_interval=poll_interval, timeout=timeout
        )


def _params(**kw: Any) -> dict[str, Any]:
    """Drop None-valued query params."""
    return {k: v for k, v in kw.items() if v is not None}
