"""Tests for ``pictograph.resources.model_evaluations.ModelEvaluations``.

Coverage:
- ``create`` sends the right body + parses the pending row.
- ``get`` parses a completed evaluation with full metrics + confusion matrix.
- ``list`` / ``cancel`` round-trips.
- ``wait_for_completion`` polls to completed; raises ``ApiError`` on failed /
  cancelled; raises ``PollTimeoutError`` on deadline; validates its args.
- ``evaluate`` (create + wait) one-call path.
"""

from __future__ import annotations

import json as _json
from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import ApiError, PollTimeoutError
from pictograph.models.evaluation import ModelEvaluation
from pictograph.resources.model_evaluations import ModelEvaluations

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
EVAL_ID = "eval-uuid-1"


@pytest.fixture
def transport() -> Transport:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def evals(transport: Transport) -> ModelEvaluations:
    return ModelEvaluations(transport)


def _payload(status: str = "pending", **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": EVAL_ID,
        "organization_id": "org-1",
        "model_id": "model-1",
        "project_id": "proj-1",
        "export_id": "exp-1",
        "status": status,
        "progress": 100 if status == "completed" else 0,
        "iou_threshold": 0.5,
        "confidence_threshold": 0.5,
        "total_images": 10,
        "evaluated_images": 10 if status == "completed" else 0,
        "failed_images": 0,
        "created_at": "2026-01-01T00:00:00Z",
    }
    if status == "completed":
        base.update(
            overall_metrics={
                "tp": 8,
                "fp": 1,
                "fn": 2,
                "precision": 0.888,
                "recall": 0.8,
                "f1": 0.842,
                "macro_f1": 0.75,
            },
            per_class_metrics=[
                {
                    "class_name": "cat",
                    "tp": 8,
                    "fp": 1,
                    "fn": 2,
                    "support": 10,
                    "precision": 0.888,
                    "recall": 0.8,
                    "f1": 0.842,
                },
            ],
            confusion_matrix={
                "iou_threshold": 0.5,
                "labels": ["cat", "__background__"],
                "grid": [[8, 2], [1, 0]],
            },
            worst_images=[
                {
                    "image_id": "img-1",
                    "filename": "a.jpg",
                    "virtual_directory_path": "/",
                    "tp": 1,
                    "fp": 1,
                    "fn": 1,
                    "gt_count": 2,
                    "pred_count": 2,
                },
            ],
            best_images=[
                {
                    "image_id": "img-2",
                    "filename": "b.jpg",
                    "virtual_directory_path": "/",
                    "tp": 5,
                    "fp": 0,
                    "fn": 0,
                    "gt_count": 5,
                    "pred_count": 5,
                    "predictions": [
                        {
                            "name": "cat",
                            "type": "bbox",
                            "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4},
                            "confidence": 0.97,
                        },
                    ],
                    "ground_truth": [
                        {
                            "name": "cat",
                            "type": "bbox",
                            "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4},
                        },
                    ],
                },
            ],
        )
    base.update(overrides)
    return base


# ── create ──


def test_create_sends_body_and_returns_pending(
    httpx_mock: HTTPXMock, evals: ModelEvaluations
) -> None:
    httpx_mock.add_response(
        method="POST", json={"success": True, "evaluation": _payload("pending")}
    )
    result = evals.create(
        "abcdef01-2345-6789-abcd-ef0123456789",
        "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "ee111111-2222-3333-4444-555555555555",
        iou_threshold=0.4,
        confidence_threshold=0.25,
    )
    assert isinstance(result, ModelEvaluation)
    assert result.status == "pending"
    sent = httpx_mock.get_request()
    assert sent is not None
    body = _json.loads(sent.read())
    assert body == {
        "model_id": "abcdef01-2345-6789-abcd-ef0123456789",
        "export_id": "ee111111-2222-3333-4444-555555555555",
        "iou_threshold": 0.4,
        "confidence_threshold": 0.25,
    }


# ── get ──


def test_get_parses_full_metrics(httpx_mock: HTTPXMock, evals: ModelEvaluations) -> None:
    httpx_mock.add_response(
        method="GET", json={"success": True, "evaluation": _payload("completed")}
    )
    ev = evals.get(EVAL_ID)
    assert ev.status == "completed"
    assert ev.overall_metrics is not None and ev.overall_metrics.f1 == 0.842
    assert ev.per_class_metrics is not None and ev.per_class_metrics[0].class_name == "cat"
    assert ev.confusion_matrix is not None and ev.confusion_matrix.labels[-1] == "__background__"
    assert ev.confusion_matrix.grid == [[8, 2], [1, 0]]
    assert ev.worst_images is not None and ev.worst_images[0].fp == 1
    # B453 S4: best_images parses (perfect image - 0 fp/fn, most tp) + carries
    # the model's predictions + ground-truth overlay sets.
    assert ev.best_images is not None and ev.best_images[0].fp == 0
    assert ev.best_images[0].tp == 5 and ev.best_images[0].filename == "b.jpg"
    assert (
        ev.best_images[0].predictions is not None
        and ev.best_images[0].predictions[0]["name"] == "cat"
    )
    assert ev.best_images[0].ground_truth is not None and len(ev.best_images[0].ground_truth) == 1


# ── report (export JSON) ──


def test_report_returns_curated_dict(httpx_mock: HTTPXMock, evals: ModelEvaluations) -> None:
    report = {
        "schema_version": "1.0",
        "generated_at": "2026-01-02T00:00:00Z",
        "model": {"id": "model-1", "name": "Swift Falcon", "type": "object_detection"},
        "metrics": {"overall": {"f1": 0.842}},
        "images": {"best": [], "worst": []},
        "warnings": [],
    }
    httpx_mock.add_response(method="GET", json={"success": True, "report": report})
    out = evals.report(EVAL_ID)
    assert out["schema_version"] == "1.0"
    assert out["model"]["name"] == "Swift Falcon"
    sent = httpx_mock.get_request()
    assert sent is not None and str(sent.url).endswith(f"/{EVAL_ID}/export")


# ── list / cancel ──


def test_list_round_trip(httpx_mock: HTTPXMock, evals: ModelEvaluations) -> None:
    httpx_mock.add_response(
        method="GET",
        json={"success": True, "evaluations": [_payload("completed"), _payload("running")]},
    )
    rows = evals.list(model="abcdef01-2345-6789-abcd-ef0123456789", limit=25)
    assert len(rows) == 2 and all(isinstance(r, ModelEvaluation) for r in rows)
    sent = httpx_mock.get_request()
    assert sent is not None and "model_id=abcdef01-2345-6789-abcd-ef0123456789" in str(sent.url)


def test_cancel_round_trip(httpx_mock: HTTPXMock, evals: ModelEvaluations) -> None:
    httpx_mock.add_response(
        method="POST", json={"success": True, "evaluation": _payload("cancelled")}
    )
    ev = evals.cancel(EVAL_ID)
    assert ev.status == "cancelled"


# ── wait_for_completion ──


def test_wait_polls_until_completed(httpx_mock: HTTPXMock, evals: ModelEvaluations) -> None:
    httpx_mock.add_response(method="GET", json={"evaluation": _payload("running")})
    httpx_mock.add_response(method="GET", json={"evaluation": _payload("completed")})
    ev = evals.wait_for_completion(EVAL_ID, poll_interval=0.01, sleep=lambda _s: None)
    assert ev.status == "completed" and ev.overall_metrics is not None


def test_wait_raises_on_failed(httpx_mock: HTTPXMock, evals: ModelEvaluations) -> None:
    httpx_mock.add_response(
        method="GET", json={"evaluation": _payload("failed", error_message="boom")}
    )
    with pytest.raises(ApiError, match="boom"):
        evals.wait_for_completion(EVAL_ID, sleep=lambda _s: None)


def test_wait_raises_polltimeout(httpx_mock: HTTPXMock, evals: ModelEvaluations) -> None:
    # Never terminal + an immediately-elapsed deadline → PollTimeoutError.
    httpx_mock.add_response(method="GET", json={"evaluation": _payload("running")})
    with pytest.raises(PollTimeoutError):
        evals.wait_for_completion(EVAL_ID, poll_interval=10, timeout=0.0001, sleep=lambda _s: None)


@pytest.mark.parametrize("kwargs", [{"poll_interval": 0}, {"timeout": 0}, {"poll_interval": -1}])
def test_wait_validates_args(evals: ModelEvaluations, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        evals.wait_for_completion(EVAL_ID, **kwargs)


# ── evaluate (create + wait) ──


def test_evaluate_one_call(httpx_mock: HTTPXMock, evals: ModelEvaluations) -> None:
    httpx_mock.add_response(method="POST", json={"evaluation": _payload("pending")})
    httpx_mock.add_response(method="GET", json={"evaluation": _payload("completed")})
    ev = evals.evaluate(
        "abcdef01-2345-6789-abcd-ef0123456789",
        "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "ee111111-2222-3333-4444-555555555555",
        poll_interval=0.01,
    )
    assert ev.status == "completed" and ev.overall_metrics is not None
