"""Second-pass coverage for the async resources - the mirror methods the
happy-path smoke suite (``test_async_resources.py``) doesn't reach: ``iter``
pagers, poll-loop success/terminal branches, bulk ops, updates, cancels, and the
download/upload edge paths. Keeps every async method exercised at least once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pictograph import AsyncClient
from pictograph.exceptions import ApiError

from .conftest import API_KEY, BASE_URL
from .test_async_resources import (
    _deployment,
    _export,
    _image,
    _model,
    _project,
    _run,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from pytest_httpx import HTTPXMock

pytestmark = pytest.mark.anyio

DEV = f"{BASE_URL}/api/v1/developer"


async def _no_sleep(_: float) -> None:
    return None


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    c = AsyncClient(api_key=API_KEY, base_url=BASE_URL, max_retries=0)
    yield c
    await c.aclose()


# ───────────── iter pagers (one per paginated resource) ─────────────


async def test_images_iter(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(method="GET", json={"data": [_image(id="a"), _image(id="b")]})
    httpx_mock.add_response(method="GET", json={"data": []})
    ids = [
        i.id async for i in client.images.iter("11111111-1111-1111-1111-111111111111", page_size=2)
    ]
    assert ids == ["a", "b"]


async def test_datasets_iter_single_page(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(method="GET", json={"data": [_project()]})
    assert len(await client.datasets.iter(page_size=50).all()) == 1


async def test_exports_iter(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(method="GET", json={"data": [_export()]})
    assert (await client.exports.iter().first()) is not None


async def test_training_iter(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(method="GET", json={"data": [_run()]})
    assert len(await client.training.iter(status="completed").all()) == 1


async def test_models_iter(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(method="GET", json={"data": [_model()]})
    assert len(await client.models.iter(dataset_name="x").all()) == 1


async def test_deployments_iter(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(method="GET", json={"deployments": [_deployment()]})
    assert (
        len(await client.deployments.iter(model="abcdef01-2345-6789-abcd-ef0123456789").all()) == 1
    )


async def test_credits_iter(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    entry = {
        "id": "l1",
        "amount": -1,
        "operation": "sam3_auto_annotation",
        "created_at": "2026-01-01T00:00:00Z",
    }
    httpx_mock.add_response(method="GET", json={"entries": [entry]})
    assert len(await client.credits.iter().all()) == 1


# ───────────── datasets: remaining ─────────────


async def test_datasets_get_by_id_and_include_images(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    from .test_async_resources import _dataset

    httpx_mock.add_response(method="GET", url=f"{DEV}/datasets/ds1", json={"data": _dataset()})
    assert (await client.datasets.get(dataset_id="ds1")).id == "ds1"
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/datasets/road-signs?include_images=true&images_limit=1000&images_offset=0",
        json={"data": _dataset()},
    )
    assert (await client.datasets.get("road-signs", include_images=True)).id == "ds1"


async def test_datasets_download_full_with_failure_and_progress(
    httpx_mock: HTTPXMock, client: AsyncClient, tmp_path: Path
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/datasets/road-signs/download?mode=full&limit=10000",
        json={
            "data": {
                "id": "ds1",
                "items": [
                    {
                        "filename": "a.jpg",
                        "image_url": "https://gcs/a",
                        "annotation_url": f"{DEV}/ann/a",
                    }
                ],
            }
        },
    )
    httpx_mock.add_response(method="GET", url="https://gcs/a", status_code=404)  # image fails
    httpx_mock.add_response(method="GET", url=f"{DEV}/ann/a", json=[{"id": "x"}])  # annotation ok
    seen: list[tuple[int, int, str | None]] = []
    report = await client.datasets.download(
        "road-signs", tmp_path, mode="full", progress=lambda c, t, f: seen.append((c, t, f))
    )
    assert report.images_downloaded == 0
    assert report.annotations_downloaded == 1
    assert len(report.failures) == 1 and report.failures[0].kind == "image"
    assert seen and seen[-1][0] == 2  # progress fired for both tasks


async def test_datasets_restore_and_wait_storage_success(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/datasets/ds1/storage/restore",
        json={"data": {"job_id": "j", "storage_state": "restoring"}},
    )
    assert (await client.datasets.restore(dataset_id="ds1")).storage_state == "restoring"
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/datasets/ds1/storage",
        json={"data": {"storage_state": "restoring"}},
    )
    httpx_mock.add_response(
        method="GET", url=f"{DEV}/datasets/ds1/storage", json={"data": {"storage_state": "idle"}}
    )
    done = await client.datasets.wait_for_storage(
        dataset_id="ds1", poll_interval=0.01, sleep=_no_sleep
    )
    assert done.storage_state == "idle"


# ───────────── images: remaining ─────────────


async def test_images_upload_bad_content_type(client: AsyncClient, tmp_path: Path) -> None:
    f = tmp_path / "notes.txt"
    f.write_text("hi")
    with pytest.raises(ValueError, match="MIME type"):
        await client.images.upload("ds1", f)


async def test_images_bulk_upload(
    httpx_mock: HTTPXMock, client: AsyncClient, tmp_path: Path
) -> None:
    a, b = tmp_path / "a.jpg", tmp_path / "b.png"
    a.write_bytes(b"\xff\xd8\xff")
    b.write_bytes(b"\x89PNG\r\n")
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/images/bulk-upload-url",
        json={
            "data": {
                "upload_urls": [
                    {"upload_url": "https://gcs/a"},
                    {"upload_url": "https://gcs/b"},
                ],
                "expires_in_minutes": 15,
            }
        },
    )
    httpx_mock.add_response(method="PUT", url="https://gcs/a", status_code=200)
    httpx_mock.add_response(method="PUT", url="https://gcs/b", status_code=200)
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/images/bulk-register",
        json={
            "data": {
                "succeeded": [_image(id="i1"), _image(id="i2")],
                "failed": [],
                "count": 2,
            }
        },
    )
    res = await client.images.bulk_upload("11111111-1111-1111-1111-111111111111", [a, b])
    assert res.count == 2
    assert [i.id for i in res.succeeded] == ["i1", "i2"]


# ───────────── annotations: bulk_save ─────────────


async def test_annotations_bulk_save(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    from pictograph.models.annotation import BBoxAnnotation, BoundingBox

    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/annotations/bulk",
        json={
            "saved": [
                {"image_id": "i1", "previous_count": 0, "new_count": 1, "status": "in_progress"}
            ],
            "failed": [{"image_id": "i2", "error": "gone"}],
        },
    )
    box = BBoxAnnotation(id="a", name="car", bounding_box=BoundingBox(x=1, y=2, w=3, h=4))
    res = await client.annotations.bulk_save({"i1": [box], "i2": [box]})
    assert res.saved_count == 1 and res.failed[0].error == "gone"


# ───────────── projects/batch/api_keys: update-arg validation ─────────────


async def test_update_no_args_raises(client: AsyncClient) -> None:
    with pytest.raises(ValueError, match="Nothing to update"):
        await client.datasets.update("p")
    with pytest.raises(ValueError, match="At least one"):
        await client.batch.update("d", ["i"])
    with pytest.raises(ValueError, match="At least one"):
        await client.api_keys.update("k")


async def test_batch_copy_and_update(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/api/v1/developer/batch/images/copy",
        json={"success": True, "processed": 1},
    )
    assert (await client.batch.copy("d", ["i"])).processed == 1
    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE_URL}/api/v1/developer/batch/images/update",
        json={"success": True, "processed": 1},
    )
    assert (await client.batch.update("d", ["i"], status="complete")).processed == 1


# ───────────── training/exports: wait success + create wait=True ─────────────


async def test_training_create_nowait_then_wait(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(method="POST", url=f"{DEV}/training/", json={"data": _run()})
    run = await client.training.create("ds", "exp", pipeline_type="yolox", name="r", wait=False)
    assert run.id == "r1"
    httpx_mock.add_response(
        method="GET", url=f"{DEV}/training/r1", json={"data": _run(status="running")}
    )
    httpx_mock.add_response(
        method="GET", url=f"{DEV}/training/r1", json={"data": _run(status="completed")}
    )
    done = await client.training.wait_for_completion("r1", poll_interval=0.01, sleep=_no_sleep)
    assert done.status == "completed"


async def test_training_wait_failed_raises(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/training/r1",
        json={"data": _run(status="failed", error_message="boom")},
    )
    with pytest.raises(ApiError, match="boom"):
        await client.training.wait_for_completion("r1", sleep=_no_sleep)


async def test_exports_get_by_id_and_download(
    httpx_mock: HTTPXMock, client: AsyncClient, tmp_path: Path
) -> None:
    httpx_mock.add_response(method="GET", url=f"{DEV}/exports/e1", json={"data": _export()})
    assert (await client.exports.get_by_id("e1")).id == "e1"
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/exports/e1/download",
        json={"data": {"download_url": "https://gcs/z"}},
    )
    httpx_mock.add_response(method="GET", url="https://gcs/z", content=b"zipbytes")
    out = await client.exports.download_by_id("e1", tmp_path / "e.zip")
    assert out.read_bytes() == b"zipbytes"


# ───────────── models: get_by_name (both branches) + bulk_delete ─────────────


async def test_models_get_by_name_routes_by_shape_not_by_404(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    """An id goes straight to the by-id form - one request, no speculative 404."""
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/models/abcdef01-2345-6789-abcd-ef0123456789",
        json={"data": _model()},
    )
    assert (await client.models.get_by_name("abcdef01-2345-6789-abcd-ef0123456789")).id == "m1"
    assert len(httpx_mock.get_requests()) == 1


async def test_models_bulk_delete(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/models/bulk-delete",
        json={"data": {"succeeded": ["m1"], "not_found": ["x"], "count": 1}},
    )
    res = await client.models.bulk_delete(["m1", "x"])
    assert res.succeeded == ["m1"]
    assert res.count == 1


# ───────────── deployments: quote/compute_options/resume/delete/bulk ─────────────


async def test_deployments_quote_and_options(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/deployments/quote?compute_type=gpu&min_containers=0&gpu_type=t4",
        json={
            "quote": {
                "rate_per_min_micro_usd": 100,
                "cost_per_hour_micro_usd": 6000,
                "cost_per_day_micro_usd": 144000,
                "scale_to_zero": True,
                "billing_note": "billed while serving",
            }
        },
    )
    assert (await client.deployments.quote(gpu_type="t4")).scale_to_zero
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/deployments/compute-options",
        json={
            "options": [
                {
                    "key": "t4",
                    "label": "T4",
                    "compute_type": "gpu",
                    "is_gpu": True,
                    "rate_per_min_micro_usd": 100,
                }
            ]
        },
    )
    assert (await client.deployments.compute_options())[0].key == "t4"


async def test_deployments_resume_delete_bulk(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/deployments/d1111111-1111-1111-1111-111111111111/resume",
        json={"deployment": _deployment()},
    )
    assert (
        await client.deployments.resume("d1111111-1111-1111-1111-111111111111")
    ).id == "d1111111-1111-1111-1111-111111111111"
    httpx_mock.add_response(
        method="DELETE", url=f"{DEV}/deployments/d1111111-1111-1111-1111-111111111111"
    )
    assert await client.deployments.delete("d1111111-1111-1111-1111-111111111111") is None
    for verb in ("pause", "resume", "delete"):
        httpx_mock.add_response(
            method="POST",
            url=f"{DEV}/deployments/bulk-{verb}",
            json={"succeeded": ["d1"], "not_found": [], "count": 1},
        )
    assert (
        await client.deployments.bulk_pause(["d1111111-1111-1111-1111-111111111111"])
    ).count == 1
    assert (
        await client.deployments.bulk_resume(["d1111111-1111-1111-1111-111111111111"])
    ).count == 1
    assert (
        await client.deployments.bulk_delete(["d1111111-1111-1111-1111-111111111111"])
    ).count == 1


# ───────────── credits: usage_by_operation ─────────────


async def test_credits_usage_by_operation(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/credits/usage-by-operation?range=week",
        json={"range": "week", "operations": [], "total_micro_usd": 0, "total_events": 0},
    )
    assert (await client.credits.usage_by_operation(range="week")).range == "week"


# ───────────── organizations: members + invites remaining ─────────────


async def test_organizations_member_and_invite_ops(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/organizations/members",
        json={
            "members": [
                {"id": "m", "user_id": "u", "role": "member", "joined_at": "2026-01-01T00:00:00Z"}
            ]
        },
    )
    assert (await client.organizations.list_members())[0].id == "m"
    httpx_mock.add_response(
        method="PATCH",
        url=f"{DEV}/organizations/members/m",
        json={"member": {"id": "m", "role": "admin"}},
    )
    assert (await client.organizations.update_member_role("m", role="admin"))["role"] == "admin"
    httpx_mock.add_response(method="DELETE", url=f"{DEV}/organizations/members/m")
    assert await client.organizations.remove_member("m") is None
    httpx_mock.add_response(
        method="GET", url=f"{DEV}/organizations/invites?status=pending", json={"invites": []}
    )
    assert await client.organizations.list_invites(status="pending") == []
    httpx_mock.add_response(method="DELETE", url=f"{DEV}/organizations/invites/i1")
    assert await client.organizations.revoke_invite("i1") is None
    httpx_mock.add_response(
        method="PATCH",
        url=f"{DEV}/organizations/me",
        json={
            "organization": {
                "id": "o",
                "name": "Renamed",
                "slug": "org",
                "is_public": True,
                "subscription_tier": "pro",
                "credits_remaining": 0,
                "credits_monthly_allowance": 0,
                "max_users": 1,
                "max_images": 0,
                "max_storage_bytes": 0,
                "member_count": 0,
                "pending_invite_count": 0,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        },
    )
    updated = await client.organizations.update(name="Renamed", is_public=True)
    assert updated.name == "Renamed" and updated.is_public is True


# ───────────── video: extract + waits ─────────────


async def test_video_extract_nowait_and_wait_success(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/video/extract-frames",
        json={"job_id": "v1", "status": "processing"},
    )
    job = await client.video.extract_frames("ds", "p", directory_name="frames", wait=False)
    assert job.job_id == "v1"
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/video/extract-frames/v1",
        json={"job_id": "v1", "status": "complete"},
    )
    done = await client.video.wait_for_extraction("v1", sleep=_no_sleep)
    assert done.status == "complete"


async def test_video_wait_failed_raises(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/video/extract-frames/v1",
        json={"job_id": "v1", "status": "failed", "error": "bad"},
    )
    with pytest.raises(ApiError, match="bad"):
        await client.video.wait_for_extraction("v1", sleep=_no_sleep)


# ───────────── connectors: check_limits + import waits ─────────────


async def test_connectors_check_limits_and_import_wait(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/connectors/check-limits",
        json={
            "allowed": True,
            "current_images": 0,
            "current_storage_bytes": 0,
            "image_limit": 100,
            "images_after_import": 10,
            "storage_after_import_bytes": 5,
            "storage_limit_bytes": 1000,
        },
    )
    assert (await client.connectors.check_limits(total_images=10, estimated_size_bytes=5)).allowed
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/connectors/import/status/imp1",
        json={"import_id": "imp1", "status": "processing"},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/connectors/import/status/imp1",
        json={"import_id": "imp1", "status": "completed"},
    )
    done = await client.connectors.wait_for_import("imp1", poll_interval=0.01, sleep=_no_sleep)
    assert done.status == "completed"


async def test_connectors_import_error_raises(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/connectors/import/status/imp1",
        json={"import_id": "imp1", "status": "error"},
    )
    with pytest.raises(ApiError):
        await client.connectors.wait_for_import("imp1", sleep=_no_sleep)


async def test_connectors_cancel_import(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/connectors/import/cancel/imp1",
        json={"status": "cancelled", "import_id": "imp1"},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/connectors/import/status/imp1",
        json={"import_id": "imp1", "status": "cancelled"},
    )
    assert (await client.connectors.cancel_import("imp1")).status == "cancelled"


# ───────────── webhooks: get/delete/test/deliveries ─────────────


async def test_webhooks_get_delete_test_deliveries(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    ep = {
        "id": "beefcafe-0000-1111-2222-333344445555",
        "url": "https://hook",
        "organization_id": "0a111111-2222-3333-4444-555566667777",
    }
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/webhooks/endpoints/beefcafe-0000-1111-2222-333344445555",
        json={"endpoint": ep},
    )
    assert (
        await client.webhooks.get("beefcafe-0000-1111-2222-333344445555")
    ).id == "beefcafe-0000-1111-2222-333344445555"
    httpx_mock.add_response(
        method="DELETE", url=f"{DEV}/webhooks/endpoints/beefcafe-0000-1111-2222-333344445555"
    )
    assert await client.webhooks.delete("beefcafe-0000-1111-2222-333344445555") is None
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/webhooks/endpoints/beefcafe-0000-1111-2222-333344445555/test",
        json={"ok": True},
    )
    assert (await client.webhooks.test("beefcafe-0000-1111-2222-333344445555"))["ok"] is True
    delivery = {
        "id": "d",
        "delivery_id": "del1",
        "endpoint_id": "w1",
        "event_type": "run.completed",
        "organization_id": "0a111111-2222-3333-4444-555566667777",
        "status": "delivered",
    }
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/webhooks/deliveries?limit=50&offset=0",
        json={"deliveries": [delivery]},
    )
    assert (await client.webhooks.deliveries())[0].delivery_id == "del1"
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/webhooks/event-types",
        json={"event_types": ["workflow_run.completed", "workflow_run.failed"]},
    )
    assert (await client.webhooks.event_types()) == [
        "workflow_run.completed",
        "workflow_run.failed",
    ]
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/webhooks/endpoints/beefcafe-0000-1111-2222-333344445555/rotate-secret",
        json={"endpoint": {**ep, "secret_version": 2}, "secret": "whsec_new"},
    )
    rotated = await client.webhooks.rotate_secret("beefcafe-0000-1111-2222-333344445555")
    assert rotated.secret == "whsec_new"  # noqa: S105 - test fixture


# ───────────── workflows: list/get/update/delete/cancel/bulk/wait ─────────────


async def test_workflows_full_lifecycle(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    wf = {
        "id": "fedcba98-1111-2222-3333-444455556666",
        "name": "w",
        "organization_id": "0a111111-2222-3333-4444-555566667777",
    }
    httpx_mock.add_response(method="GET", url=f"{DEV}/workflows/", json={"workflows": [wf]})
    assert (await client.workflows.list())[0].id == "fedcba98-1111-2222-3333-444455556666"
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/workflows/fedcba98-1111-2222-3333-444455556666",
        json={"workflow": wf},
    )
    assert (
        await client.workflows.get("fedcba98-1111-2222-3333-444455556666")
    ).id == "fedcba98-1111-2222-3333-444455556666"
    httpx_mock.add_response(
        method="PATCH",
        url=f"{DEV}/workflows/fedcba98-1111-2222-3333-444455556666",
        json={"workflow": {**wf, "name": "w2"}},
    )
    assert (
        await client.workflows.update("fedcba98-1111-2222-3333-444455556666", name="w2")
    ).name == "w2"
    httpx_mock.add_response(
        method="DELETE", url=f"{DEV}/workflows/fedcba98-1111-2222-3333-444455556666"
    )
    assert await client.workflows.delete("fedcba98-1111-2222-3333-444455556666") is None
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/workflows/bulk-delete",
        json={"deleted": ["fedcba98-1111-2222-3333-444455556666"], "not_found": [], "count": 1},
    )
    assert (await client.workflows.bulk_delete(["fedcba98-1111-2222-3333-444455556666"])).count == 1
    httpx_mock.add_response(method="POST", url=f"{DEV}/workflows/runs/run1/cancel")
    assert await client.workflows.cancel_run("run1") is None
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/workflows/runs/bulk-cancel",
        json={"succeeded": ["run1"], "not_found": [], "count": 1},
    )
    assert (await client.workflows.bulk_cancel_runs(["run1"])).count == 1


async def test_workflows_wait_for_run(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    run = {
        "id": "run1",
        "workflow_id": "fedcba98-1111-2222-3333-444455556666",
        "organization_id": "0a111111-2222-3333-4444-555566667777",
        "status": "processing",
    }
    httpx_mock.add_response(method="GET", url=f"{DEV}/workflows/runs/run1", json={"run": run})
    httpx_mock.add_response(
        method="GET", url=f"{DEV}/workflows/runs/run1", json={"run": {**run, "status": "completed"}}
    )
    done = await client.workflows.wait_for_run("run1", poll_interval=0.01, sleep=_no_sleep)
    assert done.status == "completed"


async def test_workflows_wait_error_raises(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    run = {
        "id": "run1",
        "workflow_id": "fedcba98-1111-2222-3333-444455556666",
        "organization_id": "0a111111-2222-3333-4444-555566667777",
        "status": "error",
        "error": "kaput",
    }
    httpx_mock.add_response(method="GET", url=f"{DEV}/workflows/runs/run1", json={"run": run})
    with pytest.raises(ApiError, match="kaput"):
        await client.workflows.wait_for_run("run1", sleep=_no_sleep)


# ───────────── api_keys: get/update ─────────────


async def test_api_keys_get_update(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    key = {
        "id": "k1",
        "name": "key",
        "key_prefix": "pk_live_abcd",
        "role": "member",
        "organization_id": "0a111111-2222-3333-4444-555566667777",
        "is_active": True,
        "rate_limit": 5000,
        "created_at": "2026-01-01T00:00:00Z",
    }
    httpx_mock.add_response(
        method="GET", url=f"{BASE_URL}/api/v1/api-keys/k1", json={"api_key": key}
    )
    assert (await client.api_keys.get("k1")).id == "k1"
    httpx_mock.add_response(
        method="PATCH",
        url=f"{BASE_URL}/api/v1/api-keys/k1",
        json={"api_key": {**key, "name": "renamed"}},
    )
    assert (await client.api_keys.update("k1", name="renamed")).name == "renamed"


# ───────────── auto_annotate: box/text/get/cancel + wait success ─────────────


async def test_auto_annotate_box_text_get_cancel(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    ann = {
        "id": "a",
        "name": "car",
        "type": "bbox",
        "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4},
    }
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/auto-annotate/sam3/box",
        json={"status": "success", "annotations": [ann]},
    )
    assert (
        await client.auto_annotate.box(
            "d", "a.jpg", box={"x": 1, "y": 2, "w": 3, "h": 4}, name="car"
        )
    ).status == "success"
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/auto-annotate/sam3/text",
        json={"status": "success", "annotations": [ann]},
    )
    assert len((await client.auto_annotate.text("d", "a.jpg", text_prompt="car")).annotations) == 1
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/auto-annotate/batch/j1",
        json={"job_id": "j1", "status": "running"},
    )
    assert (await client.auto_annotate.get_batch("j1")).status == "running"
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/auto-annotate/batch/j1/cancel",
        json={"job_id": "j1", "status": "cancelled"},
    )
    assert (await client.auto_annotate.cancel_batch("j1")).status == "cancelled"


async def test_auto_annotate_wait_success(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/auto-annotate/batch/j1",
        json={"job_id": "j1", "status": "running"},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/auto-annotate/batch/j1",
        json={"job_id": "j1", "status": "completed"},
    )
    done = await client.auto_annotate.wait_for_batch("j1", poll_interval=0.01, sleep=_no_sleep)
    assert done.status == "completed"


# ───────────── directories / search extra params + api_keys create with opts ─────────────


async def test_search_by_similarity_directory_and_by_tag_full(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    sim = {
        "id": "i2",
        "filename": "b.jpg",
        "similarity": 0.9,
        "annotation_count": 0,
        "status": "new",
    }
    httpx_mock.add_response(method="GET", json={"results": [sim]})
    assert (
        await client.search.by_similarity(
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "11aa22bb-33cc-44dd-55ee-66ff77008899",
            directory_path="/train",
        )
    )[0].id == "i2"
    tagged = {
        "id": "i3",
        "filename": "c.jpg",
        "project_id": "p1",
        "annotation_count": 1,
        "status": "complete",
    }
    httpx_mock.add_response(method="GET", json={"results": [tagged]})
    got = await client.search.by_tag(
        objects=["car"], scenes=["road"], attributes=["red"], dataset_name="ds"
    )
    assert got[0].id == "i3"


# ───────────── files manifest + per-file download (async twins) ─────────────

_V194 = "11111111-1111-1111-1111-111111111111"


def _manifest_194() -> dict[str, object]:
    return {
        "versions": [
            {
                "version_id": _V194,
                "version_number": 1,
                "version_label": "1.0.0",
                "status": "ready",
                "is_latest": True,
                "precision": "fp32",
            }
        ],
        "files": [
            {
                "version_id": _V194,
                "name": "config.json",
                "kind": "config",
                "format": "json",
                "size_bytes": 256,
                "content_type": "application/json",
            },
            {
                "version_id": _V194,
                "name": "README.md",
                "kind": "readme",
                "format": "markdown",
                "size_bytes": 20,
                "content_type": "text/markdown",
            },
        ],
    }


async def test_models_files_manifest(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{DEV}/models/m1/files", json={"data": _manifest_194()}
    )
    manifest = await client.models.files(model_id="m1")
    assert manifest.versions[0].is_latest is True
    assert {f.kind for f in manifest.files} == {"config", "readme"}


async def test_models_download_file_streams_and_data_url(
    httpx_mock: HTTPXMock, client: AsyncClient, tmp_path: Path
) -> None:
    # Stored artifact: UUID version → straight to the mint + GCS stream.
    signed = "https://gcs.test/cfg"
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/models/m1/files/download?version_id={_V194}&name=config.json",
        json={"data": {"download_url": signed, "filename": "config.json"}},
    )
    httpx_mock.add_response(method="GET", url=signed, content=b"{}")
    out = await client.models.download_file(
        model_id="m1",
        file_name="config.json",
        version=_V194,
        output_path=tmp_path / "cfg.json",
    )
    assert out.read_bytes() == b"{}"

    # Generated artifact: data: URL payload written directly, no GCS hop.
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/models/m1/files/download?version_id={_V194}&name=README.md",
        json={
            "data": {
                "download_url": "data:text/markdown;charset=utf-8,hello%20world",
                "filename": "README.md",
            }
        },
    )
    out2 = await client.models.download_file(
        model_id="m1",
        file_name="README.md",
        version=_V194,
        output_path=tmp_path / "README.md",
    )
    assert out2.read_text(encoding="utf-8") == "hello world"


async def test_training_create_accepts_config_path(
    httpx_mock: HTTPXMock, client: AsyncClient, tmp_path: Path
) -> None:
    import json as _json

    doc = {"_pictograph": {"pipeline": "yolox"}, "config": {"epochs": 2}}
    cfg = tmp_path / "config.json"
    cfg.write_text(_json.dumps(doc), encoding="utf-8")
    httpx_mock.add_response(
        method="POST", url=f"{DEV}/training/", json={"data": _run(status="queued")}
    )
    run = await client.training.create(
        "ds1", "v1", pipeline_type="yolox", name="rt", config=cfg, wait=False
    )
    assert run.status == "queued"
    body = _json.loads(httpx_mock.get_requests()[0].content)
    assert body["config"] == doc
