"""Tests for ``pictograph.resources.exports.Exports``.

Coverage targets:
- ``create`` (wait=False) returns the pending Export immediately.
- ``create`` (wait=True) polls until completed; raises on ``failed``;
  raises ``PollTimeoutError`` when the deadline elapses.
- ``list`` and ``iter`` against the developer exports endpoint.
- ``get`` (by dataset+export name) - happy + 404.
- ``download`` streams the ZIP via signed GCS URL with progress; atomic
  rename so partial files don't land at the destination.
- ``delete`` round-trip.
- ``wait_for_completion`` argument validation (poll_interval, timeout > 0).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import ApiError, ForbiddenError, NotFoundError, PollTimeoutError
from pictograph.models.export import Export
from pictograph.resources.exports import Exports

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
def exports(transport: Transport) -> Exports:
    return Exports(transport)


def _export_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "exp-uuid-1",
        "project_id": "ds-uuid-1",
        "dataset_name": "road-signs",
        "name": "v1",
        "format": "pictograph",
        "include_images": False,
        "class_filter": None,
        "status_filter": None,
        "status": "pending",
        "error_message": None,
        "file_size": None,
        "image_count": None,
        "annotation_count": None,
        "created_at": "2026-01-01T00:00:00Z",
        "expires_at": None,
        "download_url": None,
    }
    base.update(overrides)
    return base


# ───────────── create ─────────────


def test_create_with_wait_false_returns_pending_immediately(
    httpx_mock: HTTPXMock, exports: Exports
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/exports/",
        json={"data": _export_payload(status="pending")},
    )
    result = exports.create("road-signs", "v1", wait=False)
    assert isinstance(result, Export)
    assert result.status == "pending"


def test_create_serialises_optional_filter_fields_only_when_set(
    httpx_mock: HTTPXMock, exports: Exports
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/exports/",
        json={"data": _export_payload(status="pending")},
    )
    exports.create("road-signs", "v1", wait=False)
    sent = httpx_mock.get_request()
    assert sent is not None
    import json as _json

    body = _json.loads(sent.read())
    assert "class_filter" not in body
    assert "status_filter" not in body
    # organize_by_split defaults False and is omitted (not sent as False).
    assert "organize_by_split" not in body


def test_create_sends_organize_by_split_when_true(httpx_mock: HTTPXMock, exports: Exports) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/exports/",
        json={"data": _export_payload(status="pending")},
    )
    exports.create("road-signs", "v1", format="yolo", organize_by_split=True, wait=False)
    sent = httpx_mock.get_request()
    assert sent is not None
    import json as _json

    body = _json.loads(sent.read())
    assert body["organize_by_split"] is True


def test_create_serialises_filters_when_provided(httpx_mock: HTTPXMock, exports: Exports) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/exports/",
        json={"data": _export_payload(status="pending")},
    )
    exports.create(
        "road-signs",
        "v1",
        format="coco",
        include_images=True,
        class_filter=["stop_sign"],
        status_filter="complete",
        wait=False,
    )
    sent = httpx_mock.get_request()
    assert sent is not None
    import json as _json

    body = _json.loads(sent.read())
    assert body["format"] == "coco"
    assert body["include_images"] is True
    assert body["class_filter"] == ["stop_sign"]
    assert body["status_filter"] == "complete"


def test_create_with_wait_true_polls_until_completed(
    httpx_mock: HTTPXMock, exports: Exports, monkeypatch: pytest.MonkeyPatch
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/exports/",
        json={"data": _export_payload(status="pending")},
    )
    # Two polls: pending then completed.
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/exports/v1?dataset=road-signs",
        json={"data": _export_payload(status="processing")},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/exports/v1?dataset=road-signs",
        json={"data": _export_payload(status="completed", file_size=1024)},
    )

    sleeps: list[float] = []
    monkeypatch.setattr(
        "pictograph.resources.exports.time.sleep",
        lambda d: sleeps.append(d),
    )

    result = exports.create("road-signs", "v1", wait=True, poll_interval=0.5, timeout=10.0)
    assert result.status == "completed"
    assert result.file_size == 1024
    assert sleeps == [0.5]  # one sleep between processing and completed


def test_create_with_wait_raises_apierror_on_failed_status(
    httpx_mock: HTTPXMock, exports: Exports, monkeypatch: pytest.MonkeyPatch
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/exports/",
        json={"data": _export_payload(status="pending")},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/exports/v1?dataset=road-signs",
        json={
            "data": _export_payload(
                status="failed",
                error_message="Out of disk space",
            ),
        },
    )
    monkeypatch.setattr("pictograph.resources.exports.time.sleep", lambda _: None)
    with pytest.raises(ApiError, match="Out of disk space"):
        exports.create("road-signs", "v1", wait=True, poll_interval=0.1, timeout=5.0)


def test_create_with_wait_raises_polltimeout_when_deadline_elapses(
    httpx_mock: HTTPXMock, exports: Exports, monkeypatch: pytest.MonkeyPatch
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/exports/",
        json={"data": _export_payload(status="pending")},
    )
    # Three poll cycles before the fake clock crosses the deadline:
    #   call 1: deadline = 100 + 10 = 110
    #   iter 1: time=100 (<110) → sleep
    #   iter 2: time=105 (<110) → sleep
    #   iter 3: time=110 (>=110) → raise
    for _ in range(3):
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE}/api/v1/developer/exports/v1?dataset=road-signs",
            json={"data": _export_payload(status="processing")},
        )

    times = iter([100.0, 100.0, 105.0, 110.0])
    monkeypatch.setattr("pictograph.resources.exports.time.monotonic", lambda: next(times))
    monkeypatch.setattr("pictograph.resources.exports.time.sleep", lambda _: None)

    with pytest.raises(PollTimeoutError, match="did not complete"):
        exports.create("road-signs", "v1", wait=True, poll_interval=0.1, timeout=10.0)


# ───────────── list / iter ─────────────


def test_list_passes_optional_filter_params(httpx_mock: HTTPXMock, exports: Exports) -> None:
    httpx_mock.add_response(
        method="GET",
        url=(
            f"{BASE}/api/v1/developer/exports/"
            "?limit=50&offset=10&dataset_name=road-signs&status=completed"
        ),
        json={"data": [_export_payload(status="completed")]},
    )
    result = exports.list(dataset_name="road-signs", status="completed", limit=50, offset=10)
    assert len(result) == 1


def test_list_omits_optional_params_when_none(httpx_mock: HTTPXMock, exports: Exports) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/exports/?limit=100&offset=0",
        json={"data": []},
    )
    assert exports.list() == []


def test_iter_paginates_across_pages(httpx_mock: HTTPXMock, exports: Exports) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/exports/?offset=0&limit=2",
        json={
            "data": [
                _export_payload(id="e1", name="n1"),
                _export_payload(id="e2", name="n2"),
            ],
        },
    )
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/exports/?offset=2&limit=2",
        json={"data": [_export_payload(id="e3", name="n3")]},
    )
    items = list(exports.iter(page_size=2))
    assert [e.name for e in items] == ["n1", "n2", "n3"]


# ───────────── get ─────────────


def test_get_returns_typed_export(httpx_mock: HTTPXMock, exports: Exports) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/exports/v1?dataset=road-signs",
        json={"data": _export_payload(status="completed")},
    )
    result = exports.get("road-signs", "v1")
    assert isinstance(result, Export)
    assert result.status == "completed"


def test_get_404_raises_not_found(httpx_mock: HTTPXMock, exports: Exports) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/exports/missing?dataset=road-signs",
        status_code=404,
        json={"detail": "Export not found"},
    )
    with pytest.raises(NotFoundError):
        exports.get("road-signs", "missing")


def test_get_by_id_returns_typed_export(httpx_mock: HTTPXMock, exports: Exports) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/exports/exp-123",
        json={"data": _export_payload(status="completed")},
    )
    result = exports.get_by_id("exp-123")
    assert isinstance(result, Export)
    assert result.status == "completed"


def test_get_by_id_404_raises_not_found(httpx_mock: HTTPXMock, exports: Exports) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/exports/missing-id",
        status_code=404,
        json={"detail": "Export not found"},
    )
    with pytest.raises(NotFoundError):
        exports.get_by_id("missing-id")


def test_get_percent_encodes_name_path_segments(httpx_mock: HTTPXMock, exports: Exports) -> None:
    """A dataset/export name containing path-structural characters (``/ ? #``)
    must be encoded so it round-trips through the backend's ``/{export}``
    route - the export name as a single path segment, the dataset as a query
    value. Without encoding, ``a/b`` injects an extra segment (404) and ``q?x``
    bleeds into the query string."""
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/exports/v%3F1?dataset=road%2Fsigns",
        json={"data": _export_payload(status="completed")},
    )
    result = exports.get("road/signs", "v?1")
    assert result.status == "completed"
    sent = httpx_mock.get_requests()[0]
    # The export name stays ONE encoded path segment; the dataset travels as an
    # encoded query value, so a `/` in either cannot invent a route.
    assert sent.url.raw_path == b"/api/v1/developer/exports/v%3F1?dataset=road%2Fsigns"
    assert sent.url.query == b"dataset=road%2Fsigns"


def test_delete_percent_encodes_name_path_segments(httpx_mock: HTTPXMock, exports: Exports) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/exports/v%231?dataset=road%2Fsigns",
        status_code=204,
    )
    exports.delete("road/signs", "v#1")
    sent = httpx_mock.get_requests()[0]
    assert sent.url.raw_path == b"/api/v1/developer/exports/v%231?dataset=road%2Fsigns"


def test_download_by_id_streams_zip_to_file(
    httpx_mock: HTTPXMock, exports: Exports, tmp_path: Path
) -> None:
    download_url = "https://storage.test/export-token-by-id"
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/exports/exp-123/download",
        json={"data": {"download_url": download_url}},
    )
    httpx_mock.add_response(
        method="GET",
        url=download_url,
        content=b"PK\x03\x04FAKE_ZIP_BYTES",
        headers={"Content-Length": "16"},
    )
    out = tmp_path / "export.zip"
    result = exports.download_by_id("exp-123", out)
    assert result == out
    assert out.read_bytes() == b"PK\x03\x04FAKE_ZIP_BYTES"
    assert not (tmp_path / "export.zip.part").exists()


# ───────────── wait_for_completion ─────────────


def test_wait_for_completion_argument_validation() -> None:
    config = ClientConfig(api_key=KEY, base_url=BASE)  # type: ignore[arg-type]
    transport = Transport(config, api_key=KEY)
    try:
        ex = Exports(transport)
        with pytest.raises(ValueError, match="poll_interval"):
            ex.wait_for_completion("d", "e", poll_interval=0.0)
        with pytest.raises(ValueError, match="poll_interval"):
            ex.wait_for_completion("d", "e", poll_interval=-1.0)
        with pytest.raises(ValueError, match="timeout"):
            ex.wait_for_completion("d", "e", timeout=0.0)
        with pytest.raises(ValueError, match="timeout"):
            ex.wait_for_completion("d", "e", timeout=-5.0)
    finally:
        transport.close()


def test_wait_for_completion_returns_immediately_if_already_completed(
    httpx_mock: HTTPXMock, exports: Exports
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/exports/v1?dataset=road-signs",
        json={"data": _export_payload(status="completed", file_size=999)},
    )
    sleeps: list[float] = []
    result = exports.wait_for_completion("road-signs", "v1", sleep=sleeps.append)
    assert result.status == "completed"
    assert sleeps == []


# ───────────── download ─────────────


def test_download_streams_zip_to_file_atomically(
    httpx_mock: HTTPXMock, exports: Exports, tmp_path: Path
) -> None:
    download_url = "https://storage.test/export-token"
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/exports/v1/download?dataset=road-signs",
        json={"data": {"download_url": download_url}},
    )
    httpx_mock.add_response(
        method="GET",
        url=download_url,
        content=b"PK\x03\x04FAKE_ZIP_BYTES",
        headers={"Content-Length": "16"},
    )
    out = tmp_path / "export.zip"
    result = exports.download("road-signs", "v1", out)
    assert result == out
    assert out.read_bytes() == b"PK\x03\x04FAKE_ZIP_BYTES"
    # No leftover .part file
    assert not (tmp_path / "export.zip.part").exists()


def test_download_progress_callback_receives_running_totals(
    httpx_mock: HTTPXMock, exports: Exports, tmp_path: Path
) -> None:
    download_url = "https://storage.test/u"
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/exports/v1/download?dataset=road-signs",
        json={"data": {"download_url": download_url}},
    )
    payload = b"X" * 100
    httpx_mock.add_response(
        method="GET",
        url=download_url,
        content=payload,
        headers={"Content-Length": "100"},
    )
    seen: list[tuple[int, int]] = []

    def cb(sent: int, total: int) -> None:
        seen.append((sent, total))

    exports.download("road-signs", "v1", tmp_path / "out.zip", progress=cb)
    # At least one call; final sent == total.
    assert len(seen) >= 1
    assert seen[-1][0] == 100
    assert seen[-1][1] == 100


def test_download_5xx_from_gcs_raises_apierror_no_partial_file(
    httpx_mock: HTTPXMock, exports: Exports, tmp_path: Path
) -> None:
    download_url = "https://storage.test/u"
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/exports/v1/download?dataset=road-signs",
        json={"data": {"download_url": download_url}},
    )
    httpx_mock.add_response(method="GET", url=download_url, status_code=500, content=b"oops")
    out = tmp_path / "out.zip"
    with pytest.raises(ApiError):
        exports.download("road-signs", "v1", out)
    assert not out.exists()
    assert not (tmp_path / "out.zip.part").exists()


def test_download_creates_parent_directories(
    httpx_mock: HTTPXMock, exports: Exports, tmp_path: Path
) -> None:
    download_url = "https://storage.test/u"
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/exports/v1/download?dataset=road-signs",
        json={"data": {"download_url": download_url}},
    )
    httpx_mock.add_response(method="GET", url=download_url, content=b"x")
    out = tmp_path / "deep" / "nested" / "out.zip"
    exports.download("road-signs", "v1", out)
    assert out.exists()


# ───────────── delete ─────────────


def test_delete_round_trip(httpx_mock: HTTPXMock, exports: Exports) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/exports/v1?dataset=road-signs",
        json={"success": True, "message": "Export deleted"},
    )
    exports.delete("road-signs", "v1")
    # `delete` returns None - name the route, so this cannot pass on the wrong one.
    req = httpx_mock.get_requests()[0]
    assert req.method == "DELETE"
    assert req.url.path == "/api/v1/developer/exports/v1"
    assert req.url.query == b"dataset=road-signs"


def test_bulk_delete_round_trip(httpx_mock: HTTPXMock, exports: Exports) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/exports/bulk-delete",
        json={"data": {"succeeded": ["e1", "e2"], "not_found": ["e3"], "count": 2}},
    )
    result = exports.bulk_delete(["e1", "e2", "e3"])
    assert result.succeeded == ["e1", "e2"]
    assert result.not_found == ["e3"]
    assert result.count == 2
    import json as _json

    assert _json.loads(httpx_mock.get_requests()[0].content) == {"export_ids": ["e1", "e2", "e3"]}


def test_bulk_delete_403_propagates(httpx_mock: HTTPXMock, exports: Exports) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/exports/bulk-delete",
        status_code=403,
        json={"detail": "Requires admin or owner role to delete exports"},
    )
    with pytest.raises(ForbiddenError):
        exports.bulk_delete(["e1"])


def test_delete_403_propagates(httpx_mock: HTTPXMock, exports: Exports) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/exports/v1?dataset=road-signs",
        status_code=403,
        json={"detail": "Insufficient permissions"},
    )
    with pytest.raises(ForbiddenError):
        exports.delete("road-signs", "v1")
