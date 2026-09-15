"""Tests for ``pictograph.resources.images.Images``.

Coverage targets:
- ``get`` happy path + 404 propagation.
- ``download`` streams bytes to disk; creates parent dirs.
- ``upload`` three-step flow (signed URL → chunked PUT → register), with:
    - filename / directory / content_type overrides
    - progress callback wiring
    - PIL dimension extraction for valid images
    - graceful degradation for files PIL can't decode
    - missing file raises FileNotFoundError
    - unrecognised extension raises ValueError
- ``delete`` archive (default) vs permanent (query param).
- Helpers: ``_infer_content_type`` and ``_safe_image_dimensions`` boundary cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PIL import Image as PILImage

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import ForbiddenError, NotFoundError
from pictograph.models.image import Image
from pictograph.resources.images import (
    AugmentFailure,
    BulkUploadResult,
    Images,
    TileFailure,
    _infer_content_type,
    _safe_image_dimensions,
)

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"

# ───────────── fixtures ─────────────


@pytest.fixture
def transport() -> Transport:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def images(transport: Transport) -> Images:
    return Images(transport)


def _image_metadata_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "11aa22bb-33cc-44dd-55ee-66ff77008899",
        "filename": "img_001.jpg",
        "status": "complete",
        "annotation_count": 3,
        "file_size": 24576,
        "width": 1920,
        "height": 1080,
        "image_url": "https://api.pictograph.io/api/v1/developer/images/road-signs/img.jpg",
        "thumbnail_url": "https://api.pictograph.io/api/cdn/.../img_001.jpg?size=md",
        "annotation_url": "https://api.pictograph.io/api/v1/developer/annotations/11aa22bb-33cc-44dd-55ee-66ff77008899/file",
        "created_at": "2026-01-01T00:00:00Z",
    }
    base.update(overrides)
    return base


def _make_jpeg(path: Path, width: int = 32, height: int = 16) -> Path:
    """Write a real, decodable JPEG to ``path`` of the given dimensions."""
    PILImage.new("RGB", (width, height), color=(128, 64, 32)).save(path, format="JPEG")
    return path


# ───────────── list / iter ─────────────


def test_list_returns_typed_images(httpx_mock: HTTPXMock, images: Images) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/images/?dataset=11111111-2222-3333-4444-555555555555&limit=100&offset=0",
        json={
            "data": [
                _image_metadata_payload(id="i1", filename="a.jpg"),
                _image_metadata_payload(id="i2", filename="b.jpg"),
            ],
        },
    )
    result = images.list("11111111-2222-3333-4444-555555555555")
    assert len(result) == 2
    assert all(isinstance(i, Image) for i in result)
    assert {i.filename for i in result} == {"a.jpg", "b.jpg"}


def test_list_empty_result(httpx_mock: HTTPXMock, images: Images) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/images/?dataset=11111111-2222-3333-4444-555555555555&limit=100&offset=0",
        json={"data": []},
    )
    assert images.list("11111111-2222-3333-4444-555555555555") == []


def test_list_missing_data_key_returns_empty(httpx_mock: HTTPXMock, images: Images) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/images/?dataset=11111111-2222-3333-4444-555555555555&limit=100&offset=0",
        json={},
    )
    assert images.list("11111111-2222-3333-4444-555555555555") == []


def test_list_passes_all_filters(httpx_mock: HTTPXMock, images: Images) -> None:
    httpx_mock.add_response(json={"data": []})
    images.list(
        "11111111-2222-3333-4444-555555555555",
        directory_path="/train",
        status="complete",
        include_archived=True,
        limit=25,
        offset=5,
    )
    sent = httpx_mock.get_request()
    assert sent is not None
    url = str(sent.url)
    assert "dataset=11111111-2222-3333-4444-555555555555" in url
    assert "limit=25" in url
    assert "offset=5" in url
    assert ("directory_path=%2Ftrain" in url) or ("directory_path=/train" in url)
    assert "status=complete" in url
    assert "include_archived=true" in url


def test_list_omits_unset_filters(httpx_mock: HTTPXMock, images: Images) -> None:
    httpx_mock.add_response(json={"data": []})
    images.list("11111111-2222-3333-4444-555555555555")
    sent = httpx_mock.get_request()
    assert sent is not None
    url = str(sent.url)
    assert "directory_path" not in url
    assert "status" not in url
    assert "include_archived" not in url
    assert "min_confidence_lt" not in url


def test_list_passes_min_confidence_filter(httpx_mock: HTTPXMock, images: Images) -> None:
    # Active learning: the confidence threshold rides into the query string
    # and each parsed Image carries the min_confidence the backend returned.
    httpx_mock.add_response(
        json={"data": [_image_metadata_payload(id="i1", filename="u.jpg", min_confidence=0.42)]}
    )
    result = images.list("11111111-2222-3333-4444-555555555555", min_confidence_lt=0.9)
    sent = httpx_mock.get_request()
    assert sent is not None
    assert "min_confidence_lt=0.9" in str(sent.url)
    assert result[0].min_confidence == 0.42


def test_iter_threads_min_confidence(httpx_mock: HTTPXMock, images: Images) -> None:
    httpx_mock.add_response(json={"data": []})
    list(images.iter("11111111-2222-3333-4444-555555555555", min_confidence_lt=0.7))
    sent = httpx_mock.get_request()
    assert sent is not None
    assert "min_confidence_lt=0.7" in str(sent.url)


def test_iter_paginates_across_pages(httpx_mock: HTTPXMock, images: Images) -> None:
    p1 = [
        _image_metadata_payload(id="i1", filename="1.jpg"),
        _image_metadata_payload(id="i2", filename="2.jpg"),
    ]
    p2 = [
        _image_metadata_payload(id="i3", filename="3.jpg"),
        _image_metadata_payload(id="i4", filename="4.jpg"),
    ]
    p3 = [_image_metadata_payload(id="i5", filename="5.jpg")]
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/images/?dataset=11111111-2222-3333-4444-555555555555&limit=2&offset=0",
        json={"data": p1},
    )
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/images/?dataset=11111111-2222-3333-4444-555555555555&limit=2&offset=2",
        json={"data": p2},
    )
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/images/?dataset=11111111-2222-3333-4444-555555555555&limit=2&offset=4",
        json={"data": p3},
    )
    all_items = list(images.iter("11111111-2222-3333-4444-555555555555", page_size=2))
    assert [i.filename for i in all_items] == ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"]


def test_iter_max_total_caps(httpx_mock: HTTPXMock, images: Images) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/images/?dataset=11111111-2222-3333-4444-555555555555&limit=2&offset=0",
        json={
            "data": [
                _image_metadata_payload(id="i1", filename="1.jpg"),
                _image_metadata_payload(id="i2", filename="2.jpg"),
            ]
        },
    )
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/images/?dataset=11111111-2222-3333-4444-555555555555&limit=2&offset=2",
        json={
            "data": [
                _image_metadata_payload(id="i3", filename="3.jpg"),
                _image_metadata_payload(id="i4", filename="4.jpg"),
            ]
        },
    )
    items = list(images.iter("11111111-2222-3333-4444-555555555555", page_size=2, max_total=3))
    assert [i.filename for i in items] == ["1.jpg", "2.jpg", "3.jpg"]


def test_iter_short_page_terminates(httpx_mock: HTTPXMock, images: Images) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/images/?dataset=11111111-2222-3333-4444-555555555555&limit=10&offset=0",
        json={"data": [_image_metadata_payload(id="only", filename="only.jpg")]},
    )
    assert [
        i.filename for i in images.iter("11111111-2222-3333-4444-555555555555", page_size=10)
    ] == ["only.jpg"]


def test_iter_directory_filter_threaded(httpx_mock: HTTPXMock, images: Images) -> None:
    httpx_mock.add_response(json={"data": [_image_metadata_payload(id="v", filename="v.jpg")]})
    result = images.iter("11111111-2222-3333-4444-555555555555", directory_path="/val").all()
    assert [i.filename for i in result] == ["v.jpg"]
    sent = httpx_mock.get_request()
    assert sent is not None
    assert ("directory_path=%2Fval" in str(sent.url)) or ("directory_path=/val" in str(sent.url))


# ───────────── get ─────────────


def test_get_returns_typed_image(httpx_mock: HTTPXMock, images: Images) -> None:
    # A FILENAME is answered by the list route in ONE request - it already
    # carries the whole row, so there is no id-then-refetch round trip.
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/images/?dataset=road-signs&limit=2&offset=0&filename=img.jpg",
        json={"data": [_image_metadata_payload()]},
    )
    img = images.get("road-signs", "img.jpg")
    assert isinstance(img, Image)
    assert img.id == "11aa22bb-33cc-44dd-55ee-66ff77008899"
    assert img.width == 1920
    assert img.height == 1080
    assert img.annotation_count == 3


def test_get_404_raises_not_found(httpx_mock: HTTPXMock, images: Images) -> None:
    # No row for that filename - the SDK raises NotFoundError itself rather than
    # returning an empty result the caller has to check.
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/images/?dataset=road-signs&limit=2&offset=0&filename=img.jpg",
        json={"data": []},
    )
    with pytest.raises(NotFoundError):
        images.get("road-signs", "img.jpg")


# ───────────── download ─────────────


def test_download_streams_bytes_to_file(
    httpx_mock: HTTPXMock, images: Images, tmp_path: Path
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/images/road-signs/img.jpg",
        content=b"\xff\xd8\xff\xe0FAKEJPEG_PAYLOAD",
        headers={"Content-Type": "image/jpeg"},
    )
    out = tmp_path / "downloaded.jpg"
    result = images.download("road-signs", "img.jpg", out)
    assert result == out
    assert out.read_bytes() == b"\xff\xd8\xff\xe0FAKEJPEG_PAYLOAD"


def test_download_bundle_hits_the_data_bundle_route_and_writes_the_zip(
    httpx_mock: HTTPXMock, images: Images, tmp_path: Path
) -> None:
    """The bundle is a DIFFERENT route from the raw image bytes. Pinned by URL,
    because the whole point is that `download_bundle` reproduces the editor's
    "Image data" button - hitting the plain image route would silently hand back
    a JPEG named .zip."""
    zip_bytes = b"PK\x03\x04FAKEZIP"
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/images/road-signs/img.jpg/data-bundle",
        content=zip_bytes,
        headers={"Content-Type": "application/zip"},
    )
    out = tmp_path / "img.zip"
    assert images.download_bundle("road-signs", "img.jpg", out) == out
    assert out.read_bytes() == zip_bytes


def test_download_bundle_leaves_no_partial_file_on_failure(
    httpx_mock: HTTPXMock, images: Images, tmp_path: Path
) -> None:
    """Same atomic-write guarantee as download(): a failed transfer must not
    leave a truncated zip sitting at the destination, or a retry silently
    "succeeds" against a corrupt archive."""
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/images/road-signs/img.jpg/data-bundle",
        status_code=404,
        json={"detail": "not found"},
    )
    out = tmp_path / "img.zip"
    with pytest.raises(Exception):
        images.download_bundle("road-signs", "img.jpg", out)
    assert not out.exists()
    assert not out.with_name(out.name + ".part").exists()


def test_download_creates_parent_directories(
    httpx_mock: HTTPXMock, images: Images, tmp_path: Path
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/images/road-signs/img.jpg",
        content=b"x",
    )
    deep = tmp_path / "nested" / "deeper" / "img.jpg"
    images.download("road-signs", "img.jpg", deep)
    assert deep.exists()


def test_download_404_raises_before_writing_file(
    httpx_mock: HTTPXMock, images: Images, tmp_path: Path
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/images/road-signs/img.jpg",
        status_code=404,
        json={"detail": "Image not found"},
    )
    out = tmp_path / "should-not-exist.jpg"
    with pytest.raises(NotFoundError):
        images.download("road-signs", "img.jpg", out)
    # File was opened for write *only after* the response status was checked.
    # An empty file at the destination would indicate a bug - assert it isn't.
    assert not out.exists()


# ───────────── upload - happy path & overrides ─────────────


def _add_upload_mocks(
    httpx_mock: HTTPXMock,
    *,
    final_image_id: str = "new-img-uuid",
    upload_url: str = "https://storage.test/u-token",
) -> None:
    """Register the three canned responses an upload exercises end to end.

    Upload-url carries no storage internals, and register returns the
    FULL canonical image - so there is no follow-up metadata GET.
    """
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/images/upload-url",
        json={
            "data": {
                "filename": "img.jpg",
                "directory_path": "/",
                "upload_url": upload_url,
                "expires_in_minutes": 15,
            }
        },
    )
    httpx_mock.add_response(method="PUT", url=upload_url, status_code=200)
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/images/register",
        json={"data": _image_metadata_payload(id=final_image_id, filename="img.jpg")},
    )


def test_upload_happy_path_returns_typed_image(
    httpx_mock: HTTPXMock, images: Images, tmp_path: Path
) -> None:
    src = _make_jpeg(tmp_path / "img.jpg", width=64, height=48)
    _add_upload_mocks(httpx_mock)
    result = images.upload("11111111-2222-3333-4444-555555555555", src)
    assert isinstance(result, Image)
    assert result.filename == "img.jpg"


def test_upload_calls_endpoints_in_correct_order(
    httpx_mock: HTTPXMock, images: Images, tmp_path: Path
) -> None:
    src = _make_jpeg(tmp_path / "img.jpg")
    _add_upload_mocks(httpx_mock)
    images.upload("11111111-2222-3333-4444-555555555555", src)
    # Order: upload-url POST → GCS PUT → register POST. Register returns the
    # full image, so there is NO trailing metadata GET (a round trip saved).
    paths = [str(r.url) for r in httpx_mock.get_requests()]
    assert len(paths) == 3
    assert paths[0].endswith("/upload-url")
    assert "storage.test" in paths[1]
    assert paths[2].endswith("/register")


def test_upload_missing_file_raises(images: Images, tmp_path: Path) -> None:
    bogus = tmp_path / "does-not-exist.jpg"
    with pytest.raises(FileNotFoundError, match=r"does-not-exist\.jpg"):
        images.upload("11111111-2222-3333-4444-555555555555", bogus)


def _add_bulk_upload_mocks(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/images/bulk-upload-url",
        json={
            "data": {
                "upload_urls": [
                    {
                        "filename": "a.jpg",
                        "directory_path": "/",
                        "upload_url": "https://storage.test/u-a",
                    },
                    {
                        "filename": "b.jpg",
                        "directory_path": "/",
                        "upload_url": "https://storage.test/u-b",
                    },
                ],
                "expires_in_minutes": 15,
            }
        },
    )
    httpx_mock.add_response(method="PUT", url="https://storage.test/u-a", status_code=200)
    httpx_mock.add_response(method="PUT", url="https://storage.test/u-b", status_code=200)
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/images/bulk-register",
        json={
            "data": {
                "succeeded": [_image_metadata_payload(id="id-a", filename="a.jpg")],
                "failed": [{"filename": "b.jpg", "directory_path": "/", "error": "duplicate"}],
                "count": 1,
            }
        },
    )


def test_bulk_upload_two_round_trips_and_typed_result(
    httpx_mock: HTTPXMock, images: Images, tmp_path: Path
) -> None:
    a = _make_jpeg(tmp_path / "a.jpg")
    b = _make_jpeg(tmp_path / "b.jpg")
    _add_bulk_upload_mocks(httpx_mock)

    result = images.bulk_upload("11111111-2222-3333-4444-555555555555", [a, b])
    assert isinstance(result, BulkUploadResult)
    assert result.count == 1
    assert isinstance(result.succeeded[0], Image)
    assert result.succeeded[0].id == "id-a"
    assert result.failed[0].filename == "b.jpg"
    assert result.failed[0].error == "duplicate"
    assert result.failed[0].directory_path == "/"

    # Exactly: bulk-upload-url POST → 2 PUTs → bulk-register POST. No per-image
    # GET (the whole point - N+2 round trips become 2 + N PUTs, not 3N).
    paths = [str(r.url) for r in httpx_mock.get_requests()]
    assert paths[0].endswith("/bulk-upload-url")
    assert "storage.test" in paths[1] and "storage.test" in paths[2]
    assert paths[3].endswith("/bulk-register")
    assert not any(p.endswith("/metadata") for p in paths)


def test_bulk_upload_empty_raises(images: Images) -> None:
    with pytest.raises(ValueError, match="at least one file"):
        images.bulk_upload("11111111-2222-3333-4444-555555555555", [])


def test_bulk_upload_missing_file_raises(images: Images, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"nope\.jpg"):
        images.bulk_upload("11111111-2222-3333-4444-555555555555", [tmp_path / "nope.jpg"])


def test_upload_custom_filename_overrides_basename(
    httpx_mock: HTTPXMock, images: Images, tmp_path: Path
) -> None:
    src = _make_jpeg(tmp_path / "local-name.jpg")
    _add_upload_mocks(httpx_mock)
    images.upload("11111111-2222-3333-4444-555555555555", src, filename="server-side-name.jpg")
    upload_url_request = httpx_mock.get_requests()[0]
    body = upload_url_request.read().decode()
    assert "server-side-name.jpg" in body
    assert "local-name.jpg" not in body


def test_upload_custom_directory_path_propagates(
    httpx_mock: HTTPXMock, images: Images, tmp_path: Path
) -> None:
    src = _make_jpeg(tmp_path / "img.jpg")
    _add_upload_mocks(httpx_mock)
    images.upload("11111111-2222-3333-4444-555555555555", src, directory_path="/train/positive")
    body = httpx_mock.get_requests()[0].read().decode()
    assert "/train/positive" in body


def test_upload_custom_content_type_overrides_inference(
    httpx_mock: HTTPXMock, images: Images, tmp_path: Path
) -> None:
    # File has unknown extension; explicit content_type unblocks the upload.
    src = tmp_path / "data.bin"
    src.write_bytes(b"\x00" * 32)
    _add_upload_mocks(httpx_mock)
    images.upload("11111111-2222-3333-4444-555555555555", src, content_type="image/png")
    body = httpx_mock.get_requests()[0].read().decode()
    assert "image/png" in body


def test_upload_unknown_extension_without_explicit_content_type_raises(
    images: Images, tmp_path: Path
) -> None:
    src = tmp_path / "data.bin"
    src.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="MIME type"):
        images.upload("11111111-2222-3333-4444-555555555555", src)


def test_upload_progress_callback_is_invoked(
    httpx_mock: HTTPXMock, images: Images, tmp_path: Path
) -> None:
    src = _make_jpeg(tmp_path / "img.jpg")
    _add_upload_mocks(httpx_mock)
    received: list[tuple[int, int]] = []

    def cb(sent: int, total: int) -> None:
        received.append((sent, total))

    images.upload("11111111-2222-3333-4444-555555555555", src, progress=cb)
    # At least one progress event for the chunked PUT phase.
    assert len(received) >= 1
    # Final reported value matches the file size on disk.
    assert received[-1][0] == src.stat().st_size
    assert received[-1][1] == src.stat().st_size


def test_upload_extracts_image_dimensions_via_pil(
    httpx_mock: HTTPXMock, images: Images, tmp_path: Path
) -> None:
    import json

    src = _make_jpeg(tmp_path / "img.jpg", width=100, height=50)
    _add_upload_mocks(httpx_mock)
    images.upload("11111111-2222-3333-4444-555555555555", src)
    register_request = httpx_mock.get_requests()[2]
    body = json.loads(register_request.read())
    # Register is a JSON body with the canonical width/height names.
    assert body["width"] == 100
    assert body["height"] == 50
    assert body["content_type"] == "image/jpeg"
    assert not any(k.startswith("gcs") for k in body)


def test_upload_non_image_falls_through_with_null_dimensions(
    httpx_mock: HTTPXMock, images: Images, tmp_path: Path
) -> None:
    import json

    # Garbage bytes with .png suffix → PIL fails to decode → dimensions null.
    src = tmp_path / "bad.png"
    src.write_bytes(b"this-is-not-a-valid-png")
    _add_upload_mocks(httpx_mock)
    images.upload("11111111-2222-3333-4444-555555555555", src)
    register_request = httpx_mock.get_requests()[2]
    body = json.loads(register_request.read())
    assert body["width"] is None and body["height"] is None


# ───────────── delete ─────────────


def test_delete_archives_by_default(httpx_mock: HTTPXMock, images: Images) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/images/road-signs/img.jpg",
        json={"success": True, "message": "Image archived"},
    )
    images.delete("road-signs", "img.jpg")
    sent = httpx_mock.get_request()
    assert sent is not None
    assert "permanent" not in str(sent.url)


def test_delete_permanent_passes_query_param(httpx_mock: HTTPXMock, images: Images) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/images/road-signs/img.jpg?permanent=true",
        json={"success": True, "message": "Image permanently deleted"},
    )
    images.delete(
        "road-signs",
        "img.jpg",
        permanent=True,
    )
    sent = httpx_mock.get_request()
    assert sent is not None
    assert "permanent=true" in str(sent.url)


def test_delete_forbidden_propagates_typed_error(httpx_mock: HTTPXMock, images: Images) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/images/road-signs/img.jpg",
        status_code=403,
        json={"detail": "Insufficient permissions to delete images"},
    )
    with pytest.raises(ForbiddenError):
        images.delete("road-signs", "img.jpg")


def test_delete_404_propagates_typed_error(httpx_mock: HTTPXMock, images: Images) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE}/api/v1/developer/images/road-signs/img.jpg",
        status_code=404,
        json={"detail": "Image not found"},
    )
    with pytest.raises(NotFoundError):
        images.delete("road-signs", "img.jpg")


# ───────────── helpers ─────────────


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("photo.jpg", "image/jpeg"),
        ("photo.JPG", "image/jpeg"),  # case-insensitive
        ("photo.jpeg", "image/jpeg"),
        ("photo.png", "image/png"),
        ("photo.gif", "image/gif"),
        ("photo.bmp", "image/bmp"),
        ("photo.webp", "image/webp"),
        ("photo.tiff", "image/tiff"),
        ("photo.tif", "image/tiff"),
        ("photo.heic", "image/heic"),
        ("photo.heif", "image/heif"),
        ("data.bin", "application/octet-stream"),
        ("README", "application/octet-stream"),
        ("something.unknown", "application/octet-stream"),
    ],
)
def test_infer_content_type(filename: str, expected: str) -> None:
    assert _infer_content_type(filename) == expected


def test_safe_image_dimensions_for_real_jpeg(tmp_path: Path) -> None:
    src = _make_jpeg(tmp_path / "img.jpg", width=200, height=100)
    assert _safe_image_dimensions(src) == (200, 100)


def test_safe_image_dimensions_for_garbage_returns_none(tmp_path: Path) -> None:
    src = tmp_path / "bad.png"
    src.write_bytes(b"not an image")
    assert _safe_image_dimensions(src) == (None, None)


def test_safe_image_dimensions_for_missing_file_returns_none(tmp_path: Path) -> None:
    src = tmp_path / "missing.png"
    assert _safe_image_dimensions(src) == (None, None)


# ───────────── bulk_tag ─────────────


def test_bulk_tag_posts_payload_and_returns_processed(
    httpx_mock: HTTPXMock, images: Images
) -> None:
    import json

    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/images/bulk-tag",
        json={"data": {"processed": 3, "tags": ["wet"], "added": True}},
    )
    n = images.bulk_tag("11111111-2222-3333-4444-555555555555", ["i1", "i2", "i3"], ["wet"])
    assert n == 3
    sent = httpx_mock.get_request()
    assert sent is not None
    body = json.loads(sent.read())
    assert body == {
        "dataset": "11111111-2222-3333-4444-555555555555",
        "image_ids": ["i1", "i2", "i3"],
        "tags": ["wet"],
        "add": True,
    }


def test_bulk_tag_remove_sets_add_false(httpx_mock: HTTPXMock, images: Images) -> None:
    import json

    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/images/bulk-tag",
        json={"processed": 1},
    )
    images.bulk_tag("11111111-2222-3333-4444-555555555555", ["i1"], ["wet"], add=False)
    sent = httpx_mock.get_request()
    assert sent is not None
    assert json.loads(sent.read())["add"] is False


# ───────────── review ─────────────


def test_review_approve_posts_action_and_returns_status(
    httpx_mock: HTTPXMock, images: Images
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/images/road-signs/img.jpg/review",
        json={
            "data": {
                "id": "img.jpg",
                "status": "complete",
                "review_note": None,
                "processed": 1,
            }
        },
    )
    status = images.review("road-signs", "img.jpg", "approve")
    assert status == "complete"
    sent = httpx_mock.get_request()
    assert sent is not None
    import json

    body = json.loads(sent.content)
    assert body == {"action": "approve"}  # no note key on approve


def test_review_request_changes_sends_note(httpx_mock: HTTPXMock, images: Images) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/images/road-signs/img2.jpg/review",
        json={
            "data": {
                "id": "img2.jpg",
                "status": "annotate",
                "review_note": "fix the roof",
                "processed": 1,
            }
        },
    )
    status = images.review(
        "road-signs",
        "img2.jpg",
        "request_changes",
        note="fix the roof",
    )
    assert status == "annotate"
    import json

    body = json.loads(httpx_mock.get_request().content)  # type: ignore[union-attr]
    assert body == {"action": "request_changes", "note": "fix the roof"}


def test_review_falls_back_to_expected_status_when_absent(
    httpx_mock: HTTPXMock, images: Images
) -> None:
    # If the response omits status, the client infers it from the action.
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/images/road-signs/img3.jpg/review",
        json={"processed": 1},
    )
    assert (
        images.review(
            "road-signs",
            "img3.jpg",
            "request_changes",
        )
        == "annotate"
    )


# ───────────── split ─────────────


def test_list_sends_split_filter(httpx_mock: HTTPXMock, images: Images) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/images/?dataset=11111111-2222-3333-4444-555555555555&limit=100&offset=0&split=train",
        json={
            "data": [
                {
                    "id": "i1",
                    "filename": "a.jpg",
                    "status": "complete",
                    "split": "train",
                    "image_url": "https://x/i1",
                    "created_at": "2026-07-01T00:00:00Z",
                }
            ]
        },
    )
    imgs = images.list("11111111-2222-3333-4444-555555555555", split="train")
    assert imgs[0].split == "train"


def test_set_split_posts_and_returns(httpx_mock: HTTPXMock, images: Images) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/images/road-signs/img.jpg/split",
        json={"data": {"id": "img.jpg", "split": "val"}},
    )
    assert images.set_split("road-signs", "img.jpg", "val") == "val"
    import json

    body = json.loads(httpx_mock.get_request().content)  # type: ignore[union-attr]
    assert body == {"split": "val"}


def test_set_split_none_clears(httpx_mock: HTTPXMock, images: Images) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/images/road-signs/img2.jpg/split",
        json={"data": {"id": "img2.jpg", "split": None}},
    )
    assert images.set_split("road-signs", "img2.jpg", None) is None


def test_assign_splits_posts_ratio_and_returns_counts(
    httpx_mock: HTTPXMock, images: Images
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/images/assign-splits",
        json={
            "data": {
                "dataset": "11111111-2222-3333-4444-555555555555",
                "processed": 100,
                "train": 70,
                "val": 20,
                "test": 10,
            }
        },
    )
    result = images.assign_splits(
        "11111111-2222-3333-4444-555555555555", train=70, val=20, test=10, seed=7
    )
    assert result == {"processed": 100, "train": 70, "val": 20, "test": 10}
    import json

    body = json.loads(httpx_mock.get_request().content)  # type: ignore[union-attr]
    assert body == {
        "dataset": "11111111-2222-3333-4444-555555555555",
        "train": 70,
        "val": 20,
        "test": 10,
        "seed": 7,
        "split_mode": "random",
    }


def test_assign_splits_embedding_mode_sends_the_mode_and_returns_cluster_counts(
    httpx_mock: HTTPXMock, images: Images
) -> None:
    """The same ratio, taken out of EACH embedding cluster.

    `clusters`/`unclustered` are only reported in embedding mode, so they are
    passed through when present rather than defaulted - a caller can tell "not
    reported" from "zero".
    """
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/images/assign-splits",
        json={
            "data": {
                "dataset": "11111111-2222-3333-4444-555555555555",
                "processed": 412,
                "train": 288,
                "val": 82,
                "test": 42,
                "clusters": 9,
                "unclustered": 3,
            }
        },
    )
    result = images.assign_splits("11111111-2222-3333-4444-555555555555", mode="embedding")
    assert result == {
        "processed": 412,
        "train": 288,
        "val": 82,
        "test": 42,
        "clusters": 9,
        "unclustered": 3,
    }
    import json

    body = json.loads(httpx_mock.get_request().content)  # type: ignore[union-attr]
    assert body["split_mode"] == "embedding"


def test_assign_splits_random_mode_omits_the_cluster_keys(
    httpx_mock: HTTPXMock, images: Images
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/images/assign-splits",
        json={"data": {"processed": 10, "train": 7, "val": 2, "test": 1}},
    )
    result = images.assign_splits("11111111-2222-3333-4444-555555555555")
    assert "clusters" not in result and "unclustered" not in result


# ── The conflict branch of augment()/tile() constructed its failure model
# positionally. Both are Pydantic BaseModels, which are KEYWORD-ONLY, so the
# branch raised `TypeError: BaseModel.__init__() takes 1 positional argument`
# instead of recording the failure - turning "one filename already exists" into a
# crash that loses the whole run AND the original ConflictError. mypy reported it
# as `Too many positional arguments` on both lines; nothing executed it, because
# no test drove a conflict.


@pytest.mark.parametrize("model", [AugmentFailure, TileFailure])
def test_failure_models_reject_positional_construction(model: type) -> None:
    """Pin WHY the call sites must pass keywords.

    If these models ever gain positional construction this test fails loudly,
    which is the signal to simplify the call sites - not a reason to go back to
    positional args at a distance.
    """
    with pytest.raises(TypeError):
        model("img-1", "a.jpg", "already exists")
    # The keyword form is the one the call sites must use.
    assert model(image_id="img-1", filename="a.jpg", reason="already exists").filename == "a.jpg"
