"""Model-evaluation resource - run + read server-side model diagnostics.

Evaluate a trained detection / instance-segmentation model against a labeled
dataset and read per-class + overall precision / recall / F1 and a confusion
matrix, all computed server-side without ever mutating your ground truth::

    ev = client.model_evaluations.evaluate("Swift Falcon", "road-signs", "v1", iou_threshold=0.5)
    print(ev.overall_metrics.f1)
    for c in ev.per_class_metrics or []:
        print(c.class_name, c.precision, c.recall)

This is the server-side companion to the offline :mod:`pictograph.metrics`: the
metric math is identical, but the server runs the inference for you (billed as a
trained-model batch inference job).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

from pictograph._http.pagination import OffsetPager
from pictograph.exceptions import ApiError, PollTimeoutError
from pictograph.models.evaluation import ModelEvaluation
from pictograph.resources import _resolve
from pictograph.resources._base import Resource

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

_API_PATH = "/api/v1/developer/model-evaluations"
_DEFAULT_POLL_INTERVAL = 3.0
_DEFAULT_TIMEOUT = 1800.0


class ModelEvaluations(Resource):
    """Create, poll, list, and cancel server-side model evaluations."""

    def create(
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
        evaluation with training, which also runs off exports.

        Args:
            model: Trained detection / instance-seg model to evaluate, by NAME.
            dataset_name: The dataset the export belongs to.
            export_name: Completed export whose images + class filter are
                the eval set. An export name is unique within its dataset,
                not globally, so the pair is what identifies it.
            iou_threshold: Min IoU for a prediction to match a ground-truth box.
            confidence_threshold: Min detection confidence to keep a prediction.

        Returns:
            The created :class:`ModelEvaluation` (status ``pending``). Poll it with
            :meth:`wait_for_completion`, or use :meth:`evaluate` to do both.
        """
        body = {
            "model_id": _resolve.model_id(self._transport, model),
            "export_id": _resolve.export_id(self._transport, dataset_name, export_name),
            "iou_threshold": iou_threshold,
            "confidence_threshold": confidence_threshold,
        }
        response = self._transport.request("POST", _API_PATH, json=body)
        return self._parse(ModelEvaluation, response["evaluation"])

    def get(self, evaluation_id: str) -> ModelEvaluation:
        """Fetch one evaluation by id."""
        response = self._transport.request("GET", f"{_API_PATH}/{evaluation_id}")
        return self._parse(ModelEvaluation, response["evaluation"])

    def list(
        self,
        *,
        model: str | None = None,
        dataset_name: str | None = None,
        limit: int = 50,
    ) -> list[ModelEvaluation]:
        """List the org's evaluations (newest first), optionally by model / dataset."""
        params = _params(
            model_id=None if model is None else _resolve.model_id(self._transport, model),
            project_id=(
                None if dataset_name is None else _resolve.dataset_id(self._transport, dataset_name)
            ),
            limit=limit,
        )
        response = self._transport.request("GET", _API_PATH, params=params)
        return self._parse_list(ModelEvaluation, response.get("evaluations", []))

    def iter(
        self,
        *,
        model: str | None = None,
        dataset_name: str | None = None,
        page_size: int = 50,
    ) -> OffsetPager[ModelEvaluation]:
        """Auto-paging iterator over the org's evaluations."""
        model_uuid = None if model is None else _resolve.model_id(self._transport, model)
        dataset_uuid = (
            None if dataset_name is None else _resolve.dataset_id(self._transport, dataset_name)
        )

        def fetch(offset: int, limit: int) -> Mapping[str, Any]:
            params = _params(
                model_id=model_uuid, project_id=dataset_uuid, limit=limit, offset=offset
            )
            return cast(
                "Mapping[str, Any]", self._transport.request("GET", _API_PATH, params=params)
            )

        return OffsetPager(
            fetch,
            items_key="evaluations",
            page_size=page_size,
            parse_item=lambda raw: self._parse(ModelEvaluation, raw),
        )

    def cancel(self, evaluation_id: str) -> ModelEvaluation:
        """Cancel a running / pending evaluation."""
        response = self._transport.request("POST", f"{_API_PATH}/{evaluation_id}/cancel")
        return self._parse(ModelEvaluation, response["evaluation"])

    def report(self, evaluation_id: str) -> dict[str, Any]:
        """Download a completed evaluation's full metrics as a self-contained report.

        Returns a plain JSON-serializable dict - the same canonical shape the
        in-app "Download JSON" button produces - bundling the model / dataset /
        export labels, the overall + per-class + COCO mAP + PR-curve + confusion
        metrics, the best/worst-performing images, and any run warnings, under a
        ``schema_version``. Persist it, diff two runs, or feed it to a report
        generator. The evaluation must be ``completed`` (a running / failed one
        raises :class:`~pictograph.exceptions.ConflictError`).

        Example::

            report = client.model_evaluations.report(ev.id)
            import json, pathlib

            pathlib.Path("eval.json").write_text(json.dumps(report, indent=2))
        """
        response = self._transport.request("GET", f"{_API_PATH}/{evaluation_id}/export")
        return cast("dict[str, Any]", response["report"])

    def wait_for_completion(
        self,
        evaluation_id: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        sleep: Callable[[float], None] | None = None,
    ) -> ModelEvaluation:
        """Poll an evaluation until it reaches a terminal status.

        Returns:
            The :class:`ModelEvaluation` in ``completed`` status (with metrics).

        Raises:
            ApiError: the evaluation ``failed`` or was ``cancelled``.
            PollTimeoutError: ``timeout`` elapsed before completion.
        """
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        sleep_fn = sleep if sleep is not None else time.sleep
        deadline = time.monotonic() + timeout
        while True:
            ev = self.get(evaluation_id)
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
            sleep_fn(poll_interval)

    def evaluate(
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
        """Create an evaluation and block until it completes (the one-call path)."""
        created = self.create(
            model,
            dataset_name,
            export_name,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
        )
        return self.wait_for_completion(created.id, poll_interval=poll_interval, timeout=timeout)


def _params(**kw: Any) -> dict[str, Any]:
    """Drop None-valued query params."""
    return {k: v for k, v in kw.items() if v is not None}
