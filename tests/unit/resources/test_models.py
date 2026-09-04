"""Tests for ``pictograph.resources.models.Models`` (house-rules contract).

Coverage targets:
- ``list`` / ``iter`` on the ``{"data": [...], "pagination": {...}}`` envelope.
- ``get`` / ``update`` / ``delete`` / ``download`` addressed by NAME and by
  ``model_id=`` UUID (both hit the same serializer → ``{"data": {...}}``).
- ``download`` - signed URL, streamed ONNX bytes, atomic rename, failure leaves
  no partial file.
- ``bulk_delete`` - canonical ``succeeded`` key (+ the ``.deleted`` back-compat
  property).
- role / status errors propagate to the typed exception hierarchy.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import (
    ApiError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ValidationError,
)
from pictograph.models.model import Model
from pictograph.resources.models import Models

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
def models(transport: Transport) -> Models:
    return Models(transport)


def _model_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "abcdef01-2345-6789-abcd-ef0123456789",
        "organization_id": "org-uuid",
        "name": "Stop Sign Detector",
        "description": "YOLOX-S trained on road-signs v1",
        "model_type": "object_detection",
        "architecture": "yolox-s",
        "visibility": "private",
        "status": "ready",
        "metrics": {"mAP": 0.87, "precision": 0.91, "recall": 0.85},
        "class_mapping": {"0": "stop_sign", "1": "yield"},
        "version": "1.0.0",
        "parent_model_id": None,
        "created_at": "2026-04-01T00:00:00Z",
        "updated_at": "2026-04-01T01:00:00Z",
    }
    base.update(overrides)
    return base


def _item(payload: dict[str, Any]) -> dict[str, Any]:
    """The single-resource envelope."""
    return {"data": payload}


def _collection(
    items: list[dict[str, Any]], *, total: int | None = None, offset: int = 0
) -> dict[str, Any]:
    """The collection envelope with server-computed pagination."""
    n = total if total is not None else len(items)
    return {
        "data": items,
        "pagination": {
            "limit": 50,
            "offset": offset,
            "total": n,
            "has_more": offset + len(items) < n,
        },
    }


# ───────────── list / iter ─────────────


def test_list_returns_typed_models(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/?limit=50&offset=0",
        json=_collection([_model_payload(), _model_payload(id="m2", name="Other")]),
    )
    result = models.list()
    assert len(result) == 2
    assert all(isinstance(m, Model) for m in result)


def test_list_passes_all_filter_params(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="GET",
        url=(
            f"{BASE}/api/v1/developer/models/"
            "?limit=20&offset=0&dataset_name=road-signs&status=ready"
            "&model_type=object_detection"
        ),
        json=_collection([_model_payload()]),
    )
    result = models.list(
        dataset_name="road-signs",
        status="ready",
        model_type="object_detection",
        limit=20,
    )
    assert len(result) == 1


def test_list_empty_result(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/?limit=50&offset=0",
        json=_collection([]),
    )
    assert models.list() == []


def test_iter_stops_on_has_more_false(httpx_mock: HTTPXMock, models: Models) -> None:
    # Page 1 is full (2 items) but the server says has_more=False → no 2nd fetch.
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/models/?offset=0&limit=2",
        json=_collection([_model_payload(id="m1"), _model_payload(id="m2")], total=2),
    )
    result = list(models.iter(page_size=2))
    assert [m.id for m in result] == ["m1", "m2"]
    assert len(httpx_mock.get_requests()) == 1


def test_iter_paginates_across_pages(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/models/?offset=0&limit=2",
        json=_collection([_model_payload(id="m1"), _model_payload(id="m2")], total=3),
    )
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/models/?offset=2&limit=2",
        json=_collection([_model_payload(id="m3")], total=3, offset=2),
    )
    result = list(models.iter(page_size=2))
    assert [m.id for m in result] == ["m1", "m2", "m3"]


# ───────────── get (by name / by id) ─────────────


def test_get_by_name(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/road-signs",
        json=_item(_model_payload(name="road-signs")),
    )
    m = models.get("road-signs")
    assert isinstance(m, Model)
    assert m.name == "road-signs"


def test_get_by_id(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789",
        json=_item(_model_payload()),
    )
    m = models.get(model_id="abcdef01-2345-6789-abcd-ef0123456789")
    assert isinstance(m, Model)
    assert m.id == "abcdef01-2345-6789-abcd-ef0123456789"
    assert m.metrics == {"mAP": 0.87, "precision": 0.91, "recall": 0.85}


def test_get_requires_exactly_one_addressing_arg(models: Models) -> None:
    with pytest.raises(ValueError):
        models.get()
    with pytest.raises(ValueError):
        models.get("a", model_id="b")


def test_get_by_name_routes_an_id_straight_to_the_by_id_form(
    httpx_mock: HTTPXMock, models: Models
) -> None:
    """No speculative by-name request first.

    This used to try the name, catch the 404, and retry as an id - which cost a
    whole extra round-trip for EVERY id and logged a spurious 404 server-side on
    each one. Detection is by shape now, so exactly one request goes out and the
    only mock registered is the by-id form.
    """
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789",
        json=_item(_model_payload()),
    )
    assert (
        models.get_by_name("abcdef01-2345-6789-abcd-ef0123456789").id
        == "abcdef01-2345-6789-abcd-ef0123456789"
    )
    assert len(httpx_mock.get_requests()) == 1


def test_get_by_name_routes_a_name_straight_to_the_by_name_form(
    httpx_mock: HTTPXMock, models: Models
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/Swift%20Falcon",
        json=_item(_model_payload()),
    )
    assert models.get_by_name("Swift Falcon").id == "abcdef01-2345-6789-abcd-ef0123456789"
    assert len(httpx_mock.get_requests()) == 1


def test_update_by_id_patches_editable_fields(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789",
        json=_item(_model_payload(name="Renamed", description="d")),
    )
    m = models.update(
        model_id="abcdef01-2345-6789-abcd-ef0123456789",
        new_name="Renamed",
        description="d",
        visibility="public",
    )
    assert isinstance(m, Model) and m.name == "Renamed"
    body = httpx_mock.get_requests()[-1].read().decode().replace(" ", "")
    assert '"name":"Renamed"' in body and '"visibility":"public"' in body


def test_update_by_name(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE}/api/v1/developer/models/road-signs",
        json=_item(_model_payload(name="road-signs", description="d")),
    )
    m = models.update("road-signs", description="d")
    assert m.description == "d"


def test_update_no_fields_raises(models: Models) -> None:
    with pytest.raises(ValueError):
        models.update(model_id="abcdef01-2345-6789-abcd-ef0123456789")


def test_get_404(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/missing",
        status_code=404,
        json={"detail": "Model 'missing' not found"},
    )
    with pytest.raises(NotFoundError):
        models.get("missing")


# ───────────── download ─────────────


def test_download_streams_onnx_to_file_atomically(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    download_url = "https://storage.test/model-token"
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789/download?format=onnx",
        json=_item(
            {
                "download_url": download_url,
                "model_name": "Stop Sign Detector",
                "filename": "model.onnx",
            }
        ),
    )
    httpx_mock.add_response(
        method="GET",
        url=download_url,
        content=b"\x08\x01\x12FAKE_ONNX_BYTES",
        headers={"Content-Length": "17"},
    )
    out = tmp_path / "model.onnx"
    result = models.download(model_id="abcdef01-2345-6789-abcd-ef0123456789", output_path=out)
    assert result == out
    assert out.read_bytes() == b"\x08\x01\x12FAKE_ONNX_BYTES"
    assert not (tmp_path / "model.onnx.part").exists()


def test_download_by_name(httpx_mock: HTTPXMock, models: Models, tmp_path: Path) -> None:
    download_url = "https://storage.test/by-name-token"
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/road-signs/download?format=onnx",
        json=_item({"download_url": download_url}),
    )
    httpx_mock.add_response(method="GET", url=download_url, content=b"ONNX")
    out = models.download("road-signs", output_path=tmp_path / "model.onnx")
    assert out.read_bytes() == b"ONNX"


def test_download_defaults_to_onnx_format_query_param(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    download_url = "https://storage.test/onnx-token"
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789/download?format=onnx",
        json=_item({"download_url": download_url}),
    )
    httpx_mock.add_response(method="GET", url=download_url, content=b"ONNX")
    out = models.download(
        model_id="abcdef01-2345-6789-abcd-ef0123456789", output_path=tmp_path / "model.onnx"
    )
    assert out.read_bytes() == b"ONNX"


def test_download_pytorch_format_threads_query_param(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    download_url = "https://storage.test/pth-token"
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789/download?format=pytorch",
        json=_item({"download_url": download_url, "filename": "model.pth"}),
    )
    httpx_mock.add_response(method="GET", url=download_url, content=b"PYTORCH_PTH_BYTES")
    out = models.download(
        model_id="abcdef01-2345-6789-abcd-ef0123456789",
        output_path=tmp_path / "model.pth",
        format="pytorch",
    )
    assert out.read_bytes() == b"PYTORCH_PTH_BYTES"
    assert not (tmp_path / "model.pth.part").exists()


def test_download_pytorch_409_raises_conflict_for_legacy_model(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789/download?format=pytorch",
        status_code=409,
        json={"detail": "PyTorch weights unavailable: model trained before dual-format export."},
    )
    with pytest.raises(ConflictError):
        models.download(
            model_id="abcdef01-2345-6789-abcd-ef0123456789",
            output_path=tmp_path / "model.pth",
            format="pytorch",
        )


def test_download_progress_callback(httpx_mock: HTTPXMock, models: Models, tmp_path: Path) -> None:
    url = "https://storage.test/u"
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789/download?format=onnx",
        json=_item({"download_url": url}),
    )
    httpx_mock.add_response(
        method="GET",
        url=url,
        content=b"X" * 256,
        headers={"Content-Length": "256"},
    )
    seen: list[tuple[int, int]] = []

    def cb(sent: int, total: int) -> None:
        seen.append((sent, total))

    models.download(
        model_id="abcdef01-2345-6789-abcd-ef0123456789",
        output_path=tmp_path / "m.onnx",
        progress=cb,
    )
    assert seen[-1] == (256, 256)


def test_download_400_when_not_ready(httpx_mock: HTTPXMock, models: Models, tmp_path: Path) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789/download?format=onnx",
        status_code=400,
        json={"detail": "Model status is 'training'; only 'ready' models can be downloaded."},
    )
    with pytest.raises(ValidationError):
        models.download(
            model_id="abcdef01-2345-6789-abcd-ef0123456789", output_path=tmp_path / "m.onnx"
        )


def test_download_404_when_weights_missing(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789/download?format=onnx",
        status_code=404,
        json={"detail": "Model weights file not found in storage"},
    )
    with pytest.raises(NotFoundError):
        models.download(
            model_id="abcdef01-2345-6789-abcd-ef0123456789", output_path=tmp_path / "m.onnx"
        )


def test_download_5xx_from_gcs_no_partial_file(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    url = "https://storage.test/u"
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789/download?format=onnx",
        json=_item({"download_url": url}),
    )
    httpx_mock.add_response(method="GET", url=url, status_code=500, content=b"oops")
    out = tmp_path / "m.onnx"
    with pytest.raises(ApiError):
        models.download(model_id="abcdef01-2345-6789-abcd-ef0123456789", output_path=out)
    assert not out.exists()
    assert not (tmp_path / "m.onnx.part").exists()


def test_download_creates_parent_directories(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    url = "https://storage.test/u"
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789/download?format=onnx",
        json=_item({"download_url": url}),
    )
    httpx_mock.add_response(method="GET", url=url, content=b"x")
    out = tmp_path / "deep" / "nested" / "m.onnx"
    models.download(model_id="abcdef01-2345-6789-abcd-ef0123456789", output_path=out)
    assert out.exists()


# ───────────── delete ─────────────


def test_delete_by_id_round_trip(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789",
        json=_item(
            {
                "id": "abcdef01-2345-6789-abcd-ef0123456789",
                "name": "Stop Sign Detector",
                "deleted": True,
            }
        ),
    )
    models.delete(model_id="abcdef01-2345-6789-abcd-ef0123456789")
    # by-ID addressing must hit the bare path, NOT `/by-name/` - the pair below is the
    # only thing distinguishing these two tests from each other.
    req = httpx_mock.get_requests()[0]
    assert req.method == "DELETE"
    assert req.url.path == "/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789"


def test_delete_by_name_round_trip(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/models/road-signs",
        json=_item({"id": "m1", "name": "road-signs", "deleted": True}),
    )
    models.delete("road-signs")
    # ...and a NAME must resolve through `/by-name/`, server-side.
    req = httpx_mock.get_requests()[0]
    assert req.method == "DELETE"
    assert req.url.path == "/api/v1/developer/models/road-signs"


def test_delete_403_propagates(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789",
        status_code=403,
        json={"detail": "An admin or owner API key is required to delete models."},
    )
    with pytest.raises(ForbiddenError):
        models.delete(model_id="abcdef01-2345-6789-abcd-ef0123456789")


# ───────────── bulk_delete ─────────────


def test_bulk_delete_round_trip(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/models/bulk-delete",
        json={"data": {"succeeded": ["m1", "m2"], "not_found": ["m3"], "count": 2}},
    )
    result = models.bulk_delete(["m1", "m2", "m3"])
    assert result.succeeded == ["m1", "m2"]
    assert result.deleted == ["m1", "m2"]  # back-compat property
    assert result.not_found == ["m3"]
    assert result.count == 2
    # The request body carries the id list under model_ids.
    sent = httpx_mock.get_requests()[0]
    import json as _json

    assert _json.loads(sent.content) == {"model_ids": ["m1", "m2", "m3"]}


def test_bulk_delete_403_propagates(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/models/bulk-delete",
        status_code=403,
        json={"detail": "An admin or owner API key is required to delete models."},
    )
    with pytest.raises(ForbiddenError):
        models.bulk_delete(["m1", "m2"])


def test_delete_404_propagates(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/models/missing",
        status_code=404,
        json={"detail": "Model 'missing' not found"},
    )
    with pytest.raises(NotFoundError):
        models.delete("missing")


# ───────────── fork ─────────────


def test_fork_returns_new_model(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/models/acme-vision/src-model/fork",
        json=_item(
            _model_payload(
                id="fork-1",
                name="Stop Sign Detector",
                forked_from_model_id="src-model",
            )
        ),
    )
    m = models.fork("acme-vision", "src-model")
    assert isinstance(m, Model)
    assert m.id == "fork-1"
    assert m.forked_from_model_id == "src-model"
    assert m.visibility == "private"


def test_fork_404_propagates(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/models/acme-vision/missing/fork",
        status_code=404,
        json={"detail": "Model not found"},
    )
    with pytest.raises(NotFoundError):
        models.fork("acme-vision", "missing")


def test_fork_403_propagates(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/models/acme-vision/src-model/fork",
        status_code=403,
        json={"detail": "A member, admin, or owner API key is required to import models."},
    )
    with pytest.raises(ForbiddenError):
        models.fork("acme-vision", "src-model")


def test_fork_400_on_not_ready(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/models/acme-vision/src-model/fork",
        status_code=400,
        json={"detail": "Only ready models can be imported"},
    )
    with pytest.raises(ValidationError):
        models.fork("acme-vision", "src-model")


# ───────────── predict (remote test inference) ─────────────


def test_predict_posts_multipart_and_parses_result(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Any
) -> None:
    # name resolution first (get_by_name → by-name GET), then the predict POST
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/models/Stop%20Sign%20Detector",
        json=_item(_model_payload()),
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789/predict?confidence_threshold=0.4&top_k=2",
        json=_item(
            {
                "success": True,
                "annotations": [
                    {
                        "name": "stop_sign",
                        "type": "bbox",
                        "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4},
                        "confidence": 0.91,
                    }
                ],
                "tags": [],
                "model_type": "object_detection",
                "inference_seconds": 1.42,
            }
        ),
    )
    img = tmp_path / "t.jpg"
    img.write_bytes(b"\xff\xd8\xffjpeg")

    result = models.predict("Stop Sign Detector", image=img, confidence=0.4, top_k=2)

    assert result.success is True
    assert result.model_type == "object_detection"
    assert result.inference_seconds == 1.42
    assert result.annotations[0]["name"] == "stop_sign"
    request = httpx_mock.get_requests()[-1]
    assert b'filename="t.jpg"' in request.content
    assert b"\xff\xd8\xffjpeg" in request.content


def test_predict_accepts_raw_bytes(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/models/m",
        json=_item(_model_payload(name="m")),
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789/predict?confidence_threshold=0.5&top_k=3",
        json=_item(
            {
                "success": True,
                "annotations": [],
                "tags": ["has_person"],
                "model_type": "classification",
                "inference_seconds": 0.3,
            }
        ),
    )
    result = models.predict("m", image=b"rawbytes")
    assert result.tags == ["has_person"]
    request = httpx_mock.get_requests()[-1]
    assert b'filename="upload.jpg"' in request.content


# ───────────── files manifest + per-file download ─────────────

V_LATEST = "11111111-1111-1111-1111-111111111111"
V_OLD = "22222222-2222-2222-2222-222222222222"


def _manifest_payload() -> dict[str, object]:
    return {
        "versions": [
            {
                "version_id": V_LATEST,
                "version_number": 2,
                "version_label": "2.0.0",
                "created_at": "2026-07-21T00:00:00Z",
                "status": "ready",
                "is_latest": True,
                "precision": "fp16",
            },
            {
                "version_id": V_OLD,
                "version_number": 1,
                "version_label": "1.0.0",
                "created_at": "2026-07-20T00:00:00Z",
                "status": "ready",
                "is_latest": False,
                "precision": "fp32",
            },
        ],
        "files": [
            {
                "version_id": V_LATEST,
                "name": "yolox-abc.onnx",
                "kind": "weights",
                "format": "onnx",
                "size_bytes": 123,
                "content_type": "application/octet-stream",
                "updated_at": "2026-07-21T00:00:00Z",
            },
            {
                "version_id": V_LATEST,
                "name": "config.json",
                "kind": "config",
                "format": "json",
                "size_bytes": 512,
                "content_type": "application/json",
                "updated_at": "2026-07-21T00:00:00Z",
            },
            {
                "version_id": V_LATEST,
                "name": "LICENSE.md",
                "kind": "license",
                "format": "markdown",
                "size_bytes": 64,
                "content_type": "text/markdown",
                "updated_at": "2026-07-21T01:00:00Z",
            },
        ],
    }


def test_files_returns_typed_manifest(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789/files",
        json=_item(_manifest_payload()),
    )
    manifest = models.files(model_id="abcdef01-2345-6789-abcd-ef0123456789")
    assert [v.version_label for v in manifest.versions] == ["2.0.0", "1.0.0"]
    assert manifest.versions[0].is_latest is True
    assert {f.kind for f in manifest.files} == {"weights", "config", "license"}
    # Every file row joins a version - the keying the Files tab relies on.
    version_ids = {v.version_id for v in manifest.versions}
    assert all(f.version_id in version_ids for f in manifest.files)


def test_files_by_name(httpx_mock: HTTPXMock, models: Models) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/road-signs-v1/files",
        json=_item(_manifest_payload()),
    )
    manifest = models.files("road-signs-v1")
    assert len(manifest.files) == 3


def test_download_file_with_version_id_skips_manifest_fetch(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    """A UUID-shaped version is used as-is - exactly TWO requests (URL mint +
    GCS), no manifest round-trip."""
    signed = "https://storage.test/config-token"
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789/files/download"
        f"?version_id={V_LATEST}&name=config.json",
        json=_item(
            {
                "download_url": signed,
                "filename": "config.json",
                "content_type": "application/json",
            }
        ),
    )
    body = b'{\n  "_pictograph": {}\n}'
    httpx_mock.add_response(
        method="GET", url=signed, content=body, headers={"Content-Length": str(len(body))}
    )
    out = tmp_path / "config.json"
    result = models.download_file(
        model_id="abcdef01-2345-6789-abcd-ef0123456789",
        file_name="config.json",
        version=V_LATEST,
        output_path=out,
    )
    assert result == out
    assert out.read_bytes() == body
    assert not (tmp_path / "config.json.part").exists()
    assert len(httpx_mock.get_requests()) == 2


def test_download_file_resolves_version_label_via_manifest(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789/files",
        json=_item(_manifest_payload()),
    )
    signed = "https://storage.test/old-onnx"
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789/files/download"
        f"?version_id={V_OLD}&name=yolox-abc.onnx",
        json=_item(
            {
                "download_url": signed,
                "filename": "yolox-abc.onnx",
                "content_type": "application/octet-stream",
            }
        ),
    )
    httpx_mock.add_response(method="GET", url=signed, content=b"OLD")
    out = models.download_file(
        model_id="abcdef01-2345-6789-abcd-ef0123456789",
        file_name="yolox-abc.onnx",
        version="1.0.0",
        output_path=tmp_path / "old.onnx",
    )
    assert out.read_bytes() == b"OLD"


def test_download_file_defaults_to_latest_version(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789/files",
        json=_item(_manifest_payload()),
    )
    signed = "https://storage.test/latest"
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789/files/download"
        f"?version_id={V_LATEST}&name=config.json",
        json=_item(
            {
                "download_url": signed,
                "filename": "config.json",
                "content_type": "application/json",
            }
        ),
    )
    httpx_mock.add_response(method="GET", url=signed, content=b"{}")
    models.download_file(
        model_id="abcdef01-2345-6789-abcd-ef0123456789",
        file_name="config.json",
        output_path=tmp_path / "cfg.json",
    )
    # DEFAULTING TO LATEST is the claim in the name. Assert the version actually sent -
    # otherwise a regression that defaults to the oldest version still satisfies the mock
    # only through pytest-httpx's implicit "every registered response was requested".
    download = [r for r in httpx_mock.get_requests() if r.url.path.endswith("/files/download")]
    assert len(download) == 1
    assert download[0].url.params["version_id"] == V_LATEST


def test_download_file_generated_data_url_writes_body_directly(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    """LICENSE.md / README.md are generated at request time and arrive as a
    data: URL - the payload IS the file; no GCS request happens."""
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789/files/download"
        f"?version_id={V_LATEST}&name=LICENSE.md",
        json=_item(
            {
                "download_url": "data:text/markdown;charset=utf-8,MIT%20License%0A%0ACopyright",
                "filename": "LICENSE.md",
                "content_type": "text/markdown",
            }
        ),
    )
    out = models.download_file(
        model_id="abcdef01-2345-6789-abcd-ef0123456789",
        file_name="LICENSE.md",
        version=V_LATEST,
        output_path=tmp_path / "LICENSE.md",
    )
    assert out.read_text(encoding="utf-8") == "MIT License\n\nCopyright"
    assert not (tmp_path / "LICENSE.md.part").exists()
    assert len(httpx_mock.get_requests()) == 1


def test_download_file_unknown_version_raises_value_error(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/abcdef01-2345-6789-abcd-ef0123456789/files",
        json=_item(_manifest_payload()),
    )
    with pytest.raises(ValueError, match=r"No version '9\.9\.9'"):
        models.download_file(
            model_id="abcdef01-2345-6789-abcd-ef0123456789",
            file_name="config.json",
            version="9.9.9",
            output_path=tmp_path / "x.json",
        )
