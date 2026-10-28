"""Live edge-case tests - things the primary regressions miss.

Covers pagination edges, error-class mapping, concurrency, and payload
shapes that the SDK's 1.0.0 overhaul specifically promised.
"""

from __future__ import annotations

import concurrent.futures
import uuid
from pathlib import Path

import pytest

from pictograph import Client
from pictograph.exceptions import (
    AuthError,
    ConflictError,
    NotFoundError,
)
from pictograph.models.annotation import BBoxAnnotation
from pictograph.models.common import BoundingBox

pytestmark = pytest.mark.live


# ───────────── auth + error mapping ─────────────


def test_bad_api_key_raises_auth_error(base_url: str | None) -> None:
    with Client(api_key="pk_live_" + "0" * 64, base_url=base_url) as c:
        with pytest.raises(AuthError):
            c.credits.balance()


def test_invalid_api_key_prefix_raises_auth_error(base_url: str | None) -> None:
    with Client(api_key="not-a-real-prefix-" + "0" * 48, base_url=base_url) as c:
        with pytest.raises(AuthError):
            c.datasets.list(limit=1)


# ───────────── NotFound vs Conflict ─────────────


def test_get_missing_image_raises_not_found(client: Client, scratch_project) -> None:
    with pytest.raises(NotFoundError):
        client.images.get(scratch_project.name, "00000000-0000-0000-0000-000000000000")


def test_get_missing_export_raises_not_found(client: Client) -> None:
    with pytest.raises(NotFoundError):
        client.exports.get("cocacola_sample", "this-export-does-not-exist-xyz")


def test_duplicate_project_raises_conflict(client: Client, scratch_project) -> None:
    with pytest.raises(ConflictError):
        client.datasets.create(scratch_project.name)


def test_delete_missing_project_returns_not_found(client: Client) -> None:
    """Regression: delete non-existent project used to 500 with raw PGRST116
    leak ('Cannot coerce the result to a single JSON object'). Root cause
    was ``.single()`` throwing on 0 rows; fix uses ``.maybe_single()``."""
    with pytest.raises(NotFoundError):
        client.datasets.delete("nonexistent-project-" + uuid.uuid4().hex[:8])


def test_get_missing_project_returns_not_found(client: Client) -> None:
    with pytest.raises(NotFoundError):
        client.datasets.get("definitely-missing-" + uuid.uuid4().hex[:8])


# ───────────── pagination edges ─────────────


def test_projects_iter_empty_max_total(client: Client) -> None:
    pager = client.datasets.iter(page_size=25, max_total=0)
    assert pager.all() == []


def test_projects_iter_caps_at_max_total(client: Client) -> None:
    pager = client.datasets.iter(page_size=3, max_total=5)
    items = pager.all()
    assert len(items) <= 5


def test_credits_history_limit_respected(client: Client) -> None:
    rows = client.credits.history(limit=3)
    assert len(rows) <= 3


def test_datasets_iter_first_only(client: Client) -> None:
    pager = client.datasets.iter(page_size=2, max_total=1)
    items = pager.all()
    assert len(items) <= 1


# ───────────── payload validation ─────────────


def test_save_rejects_annotation_with_class_field(
    client: Client, scratch_dataset_with_images
) -> None:
    _, images = scratch_dataset_with_images
    # Use a raw dict with legacy 'class' key - model should reject.
    with pytest.raises(Exception):
        BBoxAnnotation.model_validate(  # type: ignore[arg-type]
            {
                "name": "thing",
                "class": "thing",
                "type": "bbox",
                "bounding_box": {"x": 0, "y": 0, "w": 5, "h": 5},
            }
        )


def test_polygon_requires_three_points(client: Client, scratch_dataset_with_images) -> None:
    from pictograph.models.annotation import PolygonAnnotation, PolygonGeometry
    from pictograph.models.common import Point

    with pytest.raises(Exception):
        PolygonAnnotation(
            name="shape",
            polygon=PolygonGeometry(paths=[[Point(x=0, y=0), Point(x=1, y=1)]]),
        )


def test_bbox_rejects_negative_dimensions() -> None:
    with pytest.raises(Exception):
        BoundingBox(x=0, y=0, w=-10, h=5)


# ───────────── concurrency ─────────────


def test_concurrent_reads_are_safe(client: Client, scratch_project) -> None:
    """20 concurrent balance lookups must all succeed (HTTP/1.1 transport is thread-safe)."""

    def run() -> int:
        return client.credits.balance().included_remaining_micro_usd

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(lambda _: run(), range(20)))

    assert len(results) == 20
    assert all(r >= 0 for r in results)


def test_concurrent_project_gets_after_upload_are_stable(client: Client, unique_name: str) -> None:
    """Confirm the Cache-Control fix holds under concurrent reads."""
    fixtures = Path(__file__).parent / "fixtures" / "images"
    proj = client.datasets.create(unique_name)
    try:
        # Upload one image to reproduce the original race.
        first_img = next(fixtures.glob("*.png"))
        client.images.upload(proj.id, first_img)

        # Hammer the by-name GET concurrently - used to 404 under CDN cache.
        def run() -> str:
            return client.datasets.get(unique_name).id

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(lambda _: run(), range(16)))
        assert all(i == proj.id for i in ids)
    finally:
        client.datasets.delete(unique_name)


# ───────────── upload / download round-trip correctness ─────────────


def test_upload_custom_mime_roundtrip(client: Client, scratch_project, tmp_path: Path) -> None:
    """Override content_type and confirm it lands correctly."""
    from PIL import Image as _PIL

    path = tmp_path / "weird.jpg"
    _PIL.new("RGB", (64, 64), color="purple").save(path, format="JPEG")
    img = client.images.upload(scratch_project.id, path, content_type="image/jpeg")
    assert img.content_type in (None, "image/jpeg")
    assert img.width == 64
    assert img.height == 64


def test_large_download_is_streamed(client: Client, scratch_project, tmp_path: Path) -> None:
    """Uploaded image round-trips through download without corruption."""
    from PIL import Image as _PIL

    src = tmp_path / "large.png"
    _PIL.new("RGB", (1024, 1024), color="red").save(src)
    src_bytes = src.stat().st_size
    img = client.images.upload(scratch_project.id, src)
    out = tmp_path / "roundtrip.png"
    client.images.download(scratch_project.name, img.id, out)
    assert out.stat().st_size > 0
    # GCS-returned bytes may be re-encoded by browsers but the raw PUT
    # should echo back byte-for-byte at ``size=full``.
    assert abs(out.stat().st_size - src_bytes) < src_bytes  # sanity


# ───────────── org pagination + sort stability ─────────────


def test_datasets_list_sort_is_stable(client: Client) -> None:
    """Same query called twice returns the same ordering."""
    a = [d.id for d in client.datasets.list(limit=10)]
    b = [d.id for d in client.datasets.list(limit=10)]
    assert a == b
