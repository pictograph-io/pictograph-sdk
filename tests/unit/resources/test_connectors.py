"""Tests for ``pictograph.resources.connectors.Connectors``.

Coverage targets:
- ``validate``: success + failure paths, body shape.
- ``check_limits``: allowed + exceeded paths.
- ``import_``: kicker body shape, RemoteDataset vs dict accepted, wait
  semantics (poll until completed; raise on error; cancelled is terminal).
- ``cancel_import``: body + re-fetch behaviour.
- ``wait_for_import``: argument validation + immediate-return on terminal.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import (
    ApiError,
    PaymentRequiredError,
    PollTimeoutError,
)
from pictograph.models.connector import (
    ImportJob,
    LimitCheckResult,
    RemoteDataset,
    ValidationResult,
)
from pictograph.resources.connectors import Connectors

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
def connectors(transport: Transport) -> Connectors:
    return Connectors(transport)


def _import_job(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "import_id": "imp-uuid-1",
        "status": "processing",
        "progress": 0.0,
        "total_images": 100,
        "imported_images": 0,
        "failed_images": 0,
        "current_dataset": "road-signs",
        "datasets": [
            {
                "name": "road-signs",
                "project_id": "proj-1",
                "status": "pending",
                "imported": 0,
                "failed": 0,
            }
        ],
    }
    base.update(overrides)
    return base


# ───────────── validate ─────────────


def test_validate_v7_success(httpx_mock: HTTPXMock, connectors: Connectors) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/connectors/validate",
        json={
            "valid": True,
            "workspace": "team-name",
            "datasets": [
                {"id": "v7-1", "name": "Road Signs", "slug": "road-signs", "image_count": 250},
            ],
        },
    )
    result = connectors.validate("v7", "v7-api-key-here")
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {"provider": "v7", "api_key": "v7-api-key-here"}
    assert isinstance(result, ValidationResult)
    assert result.valid
    assert result.workspace == "team-name"
    assert len(result.datasets) == 1
    assert isinstance(result.datasets[0], RemoteDataset)


def test_validate_invalid_key(httpx_mock: HTTPXMock, connectors: Connectors) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/connectors/validate",
        json={"valid": False, "error": "Invalid API key", "datasets": []},
    )
    result = connectors.validate("roboflow", "bogus")
    assert not result.valid
    assert result.error == "Invalid API key"
    assert result.datasets == []


# ───────────── check_limits ─────────────


def test_check_limits_allowed(httpx_mock: HTTPXMock, connectors: Connectors) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/connectors/check-limits",
        json={
            "allowed": True,
            "current_images": 1000,
            "image_limit": 50000,
            "images_after_import": 1500,
            "current_storage_bytes": 500000000,
            "storage_limit_bytes": 53687091200,
            "storage_after_import_bytes": 600000000,
            "exceeded": None,
        },
    )
    result = connectors.check_limits(total_images=500, estimated_size_bytes=100000000)
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {"total_images": 500, "estimated_size_bytes": 100000000}
    assert isinstance(result, LimitCheckResult)
    assert result.allowed
    assert result.exceeded is None


def test_check_limits_exceeded(httpx_mock: HTTPXMock, connectors: Connectors) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/connectors/check-limits",
        json={
            "allowed": False,
            "current_images": 4900,
            "image_limit": 5000,
            "images_after_import": 5500,
            "current_storage_bytes": 100,
            "storage_limit_bytes": 1000,
            "storage_after_import_bytes": 200,
            "exceeded": "images",
        },
    )
    result = connectors.check_limits(total_images=600, estimated_size_bytes=100)
    assert not result.allowed
    assert result.exceeded == "images"


# ───────────── import_ ─────────────


def test_import_kicker_body(httpx_mock: HTTPXMock, connectors: Connectors) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/connectors/import/start",
        json={
            "import_id": "imp-uuid-1",
            "status": "started",
            "datasets": [{"name": "Road Signs", "project_id": "proj-1"}],
        },
    )
    datasets = [
        RemoteDataset(id="v7-1", name="Road Signs", slug="road-signs", image_count=250),
    ]
    job = connectors.import_("v7", "key-here", datasets, wait=False)
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {
        "provider": "v7",
        "api_key": "key-here",
        "datasets": [
            {
                "id": "v7-1",
                "name": "Road Signs",
                "slug": "road-signs",
                "image_count": 250,
            }
        ],
    }
    assert isinstance(job, ImportJob)
    assert job.import_id == "imp-uuid-1"
    assert job.status == "processing"


def test_import_accepts_dict_datasets(httpx_mock: HTTPXMock, connectors: Connectors) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/connectors/import/start",
        json={"import_id": "x", "status": "started", "datasets": []},
    )
    connectors.import_(
        "roboflow",
        "k",
        [{"id": "rf-1", "name": "X", "slug": "x", "image_count": 0, "version": 1}],
        wait=False,
    )
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body["datasets"][0]["version"] == 1


def test_import_wait_polls_until_completed(
    httpx_mock: HTTPXMock, connectors: Connectors, monkeypatch: pytest.MonkeyPatch
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/connectors/import/start",
        json={"import_id": "imp-uuid-1", "status": "started", "datasets": []},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/connectors/import/status/imp-uuid-1",
        json=_import_job(status="processing", progress=50.0, imported_images=50),
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/connectors/import/status/imp-uuid-1",
        json=_import_job(
            status="completed",
            progress=100.0,
            imported_images=100,
            datasets=[
                {
                    "name": "road-signs",
                    "project_id": "proj-1",
                    "status": "completed",
                    "imported": 100,
                    "failed": 0,
                }
            ],
        ),
    )
    sleeps: list[float] = []
    monkeypatch.setattr("pictograph.resources.connectors.time.sleep", lambda d: sleeps.append(d))
    job = connectors.import_(
        "v7",
        "k",
        [RemoteDataset(id="1", name="rs", slug="rs", image_count=100)],
        wait=True,
        poll_interval=2.0,
        timeout=120.0,
    )
    assert job.status == "completed"
    assert job.imported_images == 100
    assert sleeps == [2.0]


def test_import_wait_raises_on_error(
    httpx_mock: HTTPXMock, connectors: Connectors, monkeypatch: pytest.MonkeyPatch
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/connectors/import/start",
        json={"import_id": "imp-uuid-1", "status": "started", "datasets": []},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/connectors/import/status/imp-uuid-1",
        json=_import_job(
            status="error",
            failed_images=100,
            imported_images=0,
        ),
    )
    monkeypatch.setattr("pictograph.resources.connectors.time.sleep", lambda _: None)
    with pytest.raises(ApiError, match="failed"):
        connectors.import_(
            "v7",
            "k",
            [RemoteDataset(id="1", name="x", slug="x", image_count=100)],
            wait=True,
            poll_interval=0.1,
            timeout=10.0,
        )


def test_import_wait_cancelled_returns_snapshot(
    httpx_mock: HTTPXMock, connectors: Connectors, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelled is terminal but not an error - return snapshot rather than raise."""
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/connectors/import/start",
        json={"import_id": "imp-uuid-1", "status": "started", "datasets": []},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/connectors/import/status/imp-uuid-1",
        json=_import_job(status="cancelled", progress=30.0),
    )
    monkeypatch.setattr("pictograph.resources.connectors.time.sleep", lambda _: None)
    job = connectors.import_(
        "v7",
        "k",
        [RemoteDataset(id="1", name="x", slug="x", image_count=100)],
        wait=True,
        poll_interval=0.1,
        timeout=10.0,
    )
    assert job.status == "cancelled"
    assert job.progress == 30.0


def test_import_wait_polltimeout(
    httpx_mock: HTTPXMock, connectors: Connectors, monkeypatch: pytest.MonkeyPatch
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/connectors/import/start",
        json={"import_id": "imp-uuid-1", "status": "started", "datasets": []},
    )
    for _ in range(3):
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE}/api/v1/developer/connectors/import/status/imp-uuid-1",
            json=_import_job(status="processing", progress=20.0),
        )
    times = iter([100.0, 100.0, 105.0, 110.0])
    monkeypatch.setattr("pictograph.resources.connectors.time.monotonic", lambda: next(times))
    monkeypatch.setattr("pictograph.resources.connectors.time.sleep", lambda _: None)
    with pytest.raises(PollTimeoutError, match="did not complete"):
        connectors.import_(
            "v7",
            "k",
            [RemoteDataset(id="1", name="x", slug="x", image_count=10)],
            wait=True,
            poll_interval=0.1,
            timeout=10.0,
        )


def test_import_402_tier_cap(httpx_mock: HTTPXMock, connectors: Connectors) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/connectors/import/start",
        status_code=402,
        json={
            "detail": {
                "error": "limit_exceeded",
                "limit_type": "images",
                "current": 5000,
                "limit": 5000,
                "upgrade_url": "/settings?tab=billing",
            }
        },
    )
    with pytest.raises(PaymentRequiredError) as exc:
        connectors.import_(
            "v7",
            "k",
            [RemoteDataset(id="1", name="x", slug="x", image_count=100)],
            wait=False,
        )
    assert exc.value.upgrade_url == "/settings?tab=billing"


# ───────────── cancel_import / get_import ─────────────


def test_get_import_returns_typed(httpx_mock: HTTPXMock, connectors: Connectors) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/connectors/import/status/imp-1",
        json=_import_job(status="processing", progress=42.5),
    )
    job = connectors.get_import("imp-1")
    assert isinstance(job, ImportJob)
    assert job.progress == 42.5


def test_get_import_tolerates_project_name_and_null_name(
    httpx_mock: HTTPXMock, connectors: Connectors
) -> None:
    # The live status payload labels the per-dataset row ``project_name`` (the
    # worker's dataset dicts carry no ``name``) and it is null until the row
    # resolves. A poll during a chunked import must still parse, not 500 on a
    # str-type validation error (regression: v1.67.10 raised ServerError here).
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/connectors/import/status/imp-null",
        json={
            "import_id": "imp-null",
            "status": "processing",
            "progress": 81.7,
            "datasets": [
                {
                    "project_name": "pipes-taps",
                    "project_id": "p1",
                    "status": "processing",
                    "imported": 0,
                    "failed": 0,
                },
                {"name": None, "project_id": None, "status": None, "imported": 5, "failed": 1},
            ],
        },
    )
    job = connectors.get_import("imp-null")
    assert isinstance(job, ImportJob)
    assert job.progress == 81.7
    assert job.datasets[0].name == "pipes-taps"  # project_name alias populates name
    assert job.datasets[1].name is None  # null tolerated


def test_cancel_import_re_fetches_state(httpx_mock: HTTPXMock, connectors: Connectors) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/connectors/import/cancel/imp-1",
        json={"status": "cancelled", "import_id": "imp-1"},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/connectors/import/status/imp-1",
        json=_import_job(status="cancelled", progress=30.0),
    )
    job = connectors.cancel_import("imp-1")
    assert job.status == "cancelled"
    assert job.progress == 30.0


# ───────────── wait_for_import ─────────────


def test_wait_for_import_argument_validation(connectors: Connectors) -> None:
    with pytest.raises(ValueError, match="poll_interval"):
        connectors.wait_for_import("imp", poll_interval=0.0)
    with pytest.raises(ValueError, match="timeout"):
        connectors.wait_for_import("imp", timeout=0.0)


def test_wait_for_import_returns_immediately_on_completed(
    httpx_mock: HTTPXMock, connectors: Connectors
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/connectors/import/status/imp",
        json=_import_job(status="completed", progress=100.0),
    )
    sleeps: list[float] = []
    job = connectors.wait_for_import("imp", sleep=sleeps.append)
    assert job.status == "completed"
    assert sleeps == []
