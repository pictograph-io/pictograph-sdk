"""Tests for ``pictograph.resources.training.Training``.

Coverage targets:
- ``create`` (wait=False) - pending TrainingRun returned immediately, body
  serialised correctly (names, pipeline, gpu_type, config).
- ``create`` (wait=True) - polls until terminal; raises on ``failed`` /
  ``cancelled`` with backend error message; raises ``PollTimeoutError``
  when deadline elapses.
- ``create`` - propagates 402 (PaymentRequiredError) with credit context,
  404 (dataset/export missing), 400 (export not completed).
- ``list`` and ``iter`` - pagination + filter forwarding.
- ``get`` - happy + 404.
- ``cancel`` - happy + terminal-status 400.
- ``wait_for_completion`` - argument validation, immediate-return when
  already terminal.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import (
    ApiError,
    NotFoundError,
    PaymentRequiredError,
    PollTimeoutError,
    ValidationError,
)
from pictograph.models.training import TrainingRun
from pictograph.resources.training import Training

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"


@pytest.fixture
def transport() -> Transport:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def training(transport: Transport) -> Training:
    return Training(transport)


def _run_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "run-uuid-1",
        "organization_id": "org-uuid",
        "name": "Swift Falcon",
        "dataset_id": "ds-uuid",
        "export_id": "exp-uuid",
        "model_id": None,
        "pipeline_type": "yolox",
        "gpu_type": "a10g",
        "status": "queued",
        "progress": 0,
        "current_epoch": 0,
        "total_epochs": 100,
        "metrics": {},
        "config": {"epochs": 100, "batch_size": 16},
        "eta_seconds": None,
        "training_time_seconds": None,
        "error_message": None,
        "started_at": None,
        "completed_at": None,
        "created_at": "2026-04-19T00:00:00Z",
        "created_by": "user-uuid",
    }
    base.update(overrides)
    return base


# ───────────── create - body serialisation ─────────────


def test_create_with_wait_false_returns_pending_run(
    httpx_mock: HTTPXMock, training: Training
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/training/",
        json={"data": _run_payload(status="queued")},
    )
    run = training.create(
        "road-signs",
        "v1",
        pipeline_type="yolox",
        name="Swift Falcon",
        wait=False,
    )
    assert isinstance(run, TrainingRun)
    assert run.status == "queued"


def test_create_serialises_required_body_fields(httpx_mock: HTTPXMock, training: Training) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/training/",
        json={"data": _run_payload()},
    )
    training.create(
        "road-signs",
        "v1",
        pipeline_type="yolox",
        name="Swift Falcon",
        wait=False,
    )
    sent = httpx_mock.get_request()
    assert sent is not None
    body = json.loads(sent.read())
    assert body["dataset_name"] == "road-signs"
    assert body["export_name"] == "v1"
    assert body["pipeline_type"] == "yolox"
    assert body["name"] == "Swift Falcon"
    assert body["gpu_type"] == "a10g"
    assert body["config"] == {}


def test_create_serialises_explicit_config_and_gpu(
    httpx_mock: HTTPXMock, training: Training
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/training/",
        json={"data": _run_payload(gpu_type="h100")},
    )
    cfg = {"epochs": 200, "batch_size": 8, "learning_rate": 1e-4}
    training.create(
        "road-signs",
        "v1",
        pipeline_type="rfdetr_segmentation",
        name="Big Model",
        config=cfg,
        gpu_type="h100",
        wait=False,
    )
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body["pipeline_type"] == "rfdetr_segmentation"
    assert body["gpu_type"] == "h100"
    assert body["config"] == cfg


# ───────────── create - wait + polling ─────────────


def test_create_with_wait_polls_until_completed(
    httpx_mock: HTTPXMock, training: Training, monkeypatch: pytest.MonkeyPatch
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/training/",
        json={"data": _run_payload(status="queued")},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/training/run-uuid-1",
        json={"data": _run_payload(status="running", progress=50)},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/training/run-uuid-1",
        json={
            "data": _run_payload(
                status="completed",
                progress=100,
                model_id="model-uuid",
                metrics={"mAP": 0.85},
            ),
        },
    )
    sleeps: list[float] = []
    monkeypatch.setattr("pictograph.resources.training.time.sleep", lambda d: sleeps.append(d))
    run = training.create(
        "road-signs",
        "v1",
        pipeline_type="yolox",
        name="X",
        wait=True,
        poll_interval=2.0,
        timeout=600.0,
    )
    assert run.status == "completed"
    assert run.model_id == "model-uuid"
    assert run.metrics == {"mAP": 0.85}
    assert sleeps == [2.0]  # one sleep between running and completed


def test_create_with_wait_raises_apierror_on_failed_status(
    httpx_mock: HTTPXMock, training: Training, monkeypatch: pytest.MonkeyPatch
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/training/",
        json={"data": _run_payload(status="queued")},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/training/run-uuid-1",
        json={
            "data": _run_payload(
                status="failed",
                error_message="CUDA out of memory at epoch 47",
            ),
        },
    )
    monkeypatch.setattr("pictograph.resources.training.time.sleep", lambda _: None)
    with pytest.raises(ApiError, match="CUDA out of memory"):
        training.create(
            "road-signs",
            "v1",
            pipeline_type="yolox",
            name="X",
            wait=True,
            poll_interval=0.1,
            timeout=10.0,
        )


def test_create_with_wait_raises_apierror_on_cancelled_status(
    httpx_mock: HTTPXMock, training: Training, monkeypatch: pytest.MonkeyPatch
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/training/",
        json={"data": _run_payload(status="queued")},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/training/run-uuid-1",
        json={"data": _run_payload(status="cancelled")},
    )
    monkeypatch.setattr("pictograph.resources.training.time.sleep", lambda _: None)
    with pytest.raises(ApiError, match="cancelled"):
        training.create(
            "road-signs",
            "v1",
            pipeline_type="yolox",
            name="X",
            wait=True,
            poll_interval=0.1,
            timeout=10.0,
        )


def test_create_with_wait_polltimeout_when_deadline_elapses(
    httpx_mock: HTTPXMock, training: Training, monkeypatch: pytest.MonkeyPatch
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/training/",
        json={"data": _run_payload(status="queued")},
    )
    for _ in range(3):
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE}/api/v1/developer/training/run-uuid-1",
            json={"data": _run_payload(status="running", progress=10)},
        )
    times = iter([100.0, 100.0, 105.0, 110.0])  # deadline = 100 + 10 = 110
    monkeypatch.setattr("pictograph.resources.training.time.monotonic", lambda: next(times))
    monkeypatch.setattr("pictograph.resources.training.time.sleep", lambda _: None)
    with pytest.raises(PollTimeoutError, match="did not complete"):
        training.create(
            "road-signs",
            "v1",
            pipeline_type="yolox",
            name="X",
            wait=True,
            poll_interval=0.1,
            timeout=10.0,
        )


# ───────────── create - error propagation ─────────────


def test_create_404_dataset_not_found(httpx_mock: HTTPXMock, training: Training) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/training/",
        status_code=404,
        json={"detail": "Dataset 'missing' not found"},
    )
    with pytest.raises(NotFoundError, match="Dataset"):
        training.create("missing", "v1", pipeline_type="yolox", name="X", wait=False)


def test_create_400_export_not_completed(httpx_mock: HTTPXMock, training: Training) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/training/",
        status_code=400,
        json={"detail": "Export 'v1' must be completed before training (current status: pending)"},
    )
    with pytest.raises(ValidationError):
        training.create("road-signs", "v1", pipeline_type="yolox", name="X", wait=False)


def test_create_402_insufficient_credits_carries_credit_context(
    httpx_mock: HTTPXMock, training: Training
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/training/",
        status_code=402,
        json={
            "detail": "Insufficient credits for training",
            "required": 2_000_000,  # µUSD ($2.00)
            "remaining": 500_000,  # µUSD ($0.50)
            "unit": "micro_usd",
            "block_reason": "insufficient_credits",
            "upgrade_url": "/settings?tab=billing",
        },
    )
    with pytest.raises(PaymentRequiredError) as exc:
        training.create("road-signs", "v1", pipeline_type="yolox", name="X", wait=False)
    assert exc.value.credit_cost == 2_000_000
    assert exc.value.credits_remaining == 500_000
    assert exc.value.unit == "micro_usd"
    assert exc.value.upgrade_url == "/settings?tab=billing"


# ───────────── list / iter ─────────────


def test_list_passes_filter_params(httpx_mock: HTTPXMock, training: Training) -> None:
    httpx_mock.add_response(
        method="GET",
        url=(
            f"{BASE}/api/v1/developer/training/"
            "?limit=20&offset=5&dataset_name=road-signs&status=running"
        ),
        json={"data": [_run_payload(status="running")]},
    )
    runs = training.list(dataset_name="road-signs", status="running", limit=20, offset=5)
    assert len(runs) == 1
    assert runs[0].status == "running"


def test_list_omits_optional_params_when_none(httpx_mock: HTTPXMock, training: Training) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/training/?limit=50&offset=0",
        json={"data": []},
    )
    assert training.list() == []


def test_iter_paginates(httpx_mock: HTTPXMock, training: Training) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/training/?offset=0&limit=2",
        json={"data": [_run_payload(id="r1"), _run_payload(id="r2")]},
    )
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/training/?offset=2&limit=2",
        json={"data": [_run_payload(id="r3")]},
    )
    runs = list(training.iter(page_size=2))
    assert [r.id for r in runs] == ["r1", "r2", "r3"]


# ───────────── get / cancel ─────────────


def test_get_returns_typed_run(httpx_mock: HTTPXMock, training: Training) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/training/run-uuid-1",
        json={"data": _run_payload(progress=42)},
    )
    run = training.get(run_id="run-uuid-1")
    assert isinstance(run, TrainingRun)
    assert run.progress == 42


def test_get_404(httpx_mock: HTTPXMock, training: Training) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/training/missing",
        status_code=404,
        json={"detail": "Training run not found"},
    )
    with pytest.raises(NotFoundError):
        training.get(run_id="missing")


def test_cancel_returns_updated_run(httpx_mock: HTTPXMock, training: Training) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/training/run-uuid-1/cancel",
        json={"data": _run_payload(status="cancelled")},
    )
    run = training.cancel(run_id="run-uuid-1")
    assert run.status == "cancelled"


def test_get_by_name_hits_by_name_endpoint(httpx_mock: HTTPXMock, training: Training) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/training/Swift%20Falcon",
        json={"data": _run_payload(progress=7)},
    )
    assert training.get("Swift Falcon").progress == 7


def test_cancel_by_name_hits_by_name_endpoint(httpx_mock: HTTPXMock, training: Training) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/training/Swift%20Falcon/cancel",
        json={"data": _run_payload(status="cancelled")},
    )
    assert training.cancel("Swift Falcon").status == "cancelled"


def test_get_requires_exactly_one_addressing_arg(training: Training) -> None:
    with pytest.raises(ValueError):
        training.get()
    with pytest.raises(ValueError):
        training.get("a", run_id="b")


def test_cancel_400_on_terminal_status(httpx_mock: HTTPXMock, training: Training) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/training/run-uuid-1/cancel",
        status_code=400,
        json={"detail": "Cannot cancel training run with status 'completed'"},
    )
    with pytest.raises(ValidationError):
        training.cancel(run_id="run-uuid-1")


def test_bulk_cancel_returns_typed(httpx_mock: HTTPXMock, training: Training) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/training/bulk-cancel",
        json={"data": {"succeeded": ["r1"], "not_found": ["r2"], "count": 1}},
    )
    result = training.bulk_cancel(["r1", "r2"])
    assert result.succeeded == ["r1"]
    assert result.not_found == ["r2"]
    assert result.count == 1
    body = httpx_mock.get_requests()[-1].read().decode()
    assert '"run_ids"' in body and "r1" in body and "r2" in body


# ───────────── wait_for_completion ─────────────


def test_wait_for_completion_argument_validation(training: Training) -> None:
    with pytest.raises(ValueError, match="poll_interval"):
        training.wait_for_completion("r", poll_interval=0.0)
    with pytest.raises(ValueError, match="poll_interval"):
        training.wait_for_completion("r", poll_interval=-1.0)
    with pytest.raises(ValueError, match="timeout"):
        training.wait_for_completion("r", timeout=0.0)


def test_wait_for_completion_returns_immediately_when_already_completed(
    httpx_mock: HTTPXMock, training: Training
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/training/r",
        json={"data": _run_payload(id="r", status="completed")},
    )
    sleeps: list[float] = []
    run = training.wait_for_completion("r", sleep=sleeps.append)
    assert run.status == "completed"
    assert sleeps == []


# ───────────── config accepts a config.json path ─────────────


def test_create_accepts_config_file_path_and_posts_it_whole(
    httpx_mock: HTTPXMock, training: Training, tmp_path: Any
) -> None:
    """A downloaded config.json is read, parsed, and sent verbatim as
    `config` - envelope and all. Unwrapping is the BACKEND's job (that is
    where the identity check lives), so the SDK must not pre-strip it."""
    doc = {
        "_pictograph": {"schema_version": 1, "pipeline": "yolox"},
        "config": {"epochs": 3, "batch_size": 2},
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(doc), encoding="utf-8")

    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/training/",
        json={"data": _run_payload(status="queued")},
    )
    run = training.create(
        "road-signs",
        "v1",
        pipeline_type="yolox",
        name="Round Trip",
        config=cfg_path,
        wait=False,
    )
    assert run.status == "queued"
    body = json.loads(httpx_mock.get_requests()[0].content)
    assert body["config"] == doc


def test_create_accepts_config_path_as_str(
    httpx_mock: HTTPXMock, training: Training, tmp_path: Any
) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"epochs": 7}), encoding="utf-8")
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/training/",
        json={"data": _run_payload(status="queued")},
    )
    training.create(
        "road-signs", "v1", pipeline_type="yolox", name="R", config=str(cfg_path), wait=False
    )
    body = json.loads(httpx_mock.get_requests()[0].content)
    assert body["config"] == {"epochs": 7}


def test_create_config_file_must_hold_an_object(training: Training, tmp_path: Any) -> None:
    bad = tmp_path / "list.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        training.create("road-signs", "v1", pipeline_type="yolox", name="R", config=bad, wait=False)
