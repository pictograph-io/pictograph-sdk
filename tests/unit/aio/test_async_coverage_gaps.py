"""Targeted tests for the request-body-building branches (optional kwargs) and
download error paths the first two async suites don't reach - bringing the async
resources to parity with the sync suite's branch coverage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

from pictograph import AsyncClient
from pictograph.exceptions import ApiError
from pictograph.models.connector import RemoteDataset
from pictograph.models.dataset import DatasetClass

from .conftest import API_KEY, BASE_URL
from .test_async_resources import _export, _project

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from pytest_httpx import HTTPXMock

pytestmark = pytest.mark.anyio

DEV = f"{BASE_URL}/api/v1/developer"


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    c = AsyncClient(api_key=API_KEY, base_url=BASE_URL, max_retries=0)
    yield c
    await c.aclose()


# ───────────── download error path (_download.py) ─────────────


async def test_download_non_2xx_raises_and_cleans_up(
    httpx_mock: HTTPXMock, client: AsyncClient, tmp_path: Path
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/models/m1/download?format=onnx",
        json={"data": {"download_url": "https://gcs/w"}},
    )
    httpx_mock.add_response(method="GET", url="https://gcs/w", status_code=500, content=b"err")
    dest = tmp_path / "m.onnx"
    with pytest.raises(ApiError, match="Model download"):
        await client.models.download(model_id="m1", output_path=dest)
    assert not dest.exists()
    assert not dest.with_name("m.onnx.part").exists()


async def test_models_download_pytorch_format(
    httpx_mock: HTTPXMock, client: AsyncClient, tmp_path: Path
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/models/m1/download?format=pytorch",
        json={"data": {"download_url": "https://gcs/w"}},
    )
    httpx_mock.add_response(method="GET", url="https://gcs/w", content=b"pth")
    out = await client.models.download(
        model_id="m1", output_path=tmp_path / "m.pth", format="pytorch"
    )
    assert out.read_bytes() == b"pth"


# ───────────── projects: full create/update payloads ─────────────


async def test_datasets_create_full(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(method="POST", url=f"{DEV}/datasets/", json={"data": _project()})
    await client.datasets.create(
        "proj",
        description="d",
        annotation_types=["bbox"],
        classes=[
            DatasetClass(name="car", type="bbox", color="#fff"),
            {"name": "x", "type": "polygon"},
        ],
    )
    body = httpx_mock.get_request().read()
    assert b"annotation_types" in body and b"classes" in body


async def test_datasets_update_full(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(method="PATCH", url=f"{DEV}/datasets/proj", json={"data": _project()})
    await client.datasets.update(
        "proj",
        description="d",
        annotation_types=["polygon"],
        classes=[{"name": "x", "type": "bbox"}],
    )
    assert httpx_mock.get_request() is not None


# ───────────── exports: filters + wait=True poll-to-complete ─────────────


async def test_exports_create_with_filters_and_wait(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    httpx_mock.add_response(method="POST", url=f"{DEV}/exports/", json={"data": _export()})
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/exports/exp?dataset=road-signs",
        json={"data": _export(status="completed")},
    )
    exp = await client.exports.create(
        "road-signs",
        "exp",
        format="coco",
        include_images=True,
        class_filter=["car"],
        status_filter="complete",
        poll_interval=0.01,
    )
    assert exp.status == "completed"
    body = httpx_mock.get_requests()[0].read()
    assert b"class_filter" in body and b"status_filter" in body


# ───────────── images: list filters + permanent delete ─────────────


async def test_images_list_filters_and_permanent_delete(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    httpx_mock.add_response(method="GET", json={"data": []})
    await client.images.list(
        "11111111-1111-1111-1111-111111111111",
        directory_path="/train",
        status="complete",
        include_archived=True,
    )
    req = httpx_mock.get_request()
    assert b"include_archived" in req.url.query and b"directory_path" in req.url.query
    httpx_mock.add_response(method="DELETE", url=f"{DEV}/images/road-signs/img.jpg?permanent=true")
    await client.images.delete("road-signs", "img.jpg", permanent=True)


# ───────────── deployments: create with name/config + quote no gpu ─────────────


async def test_deployments_create_full_and_quote_no_gpu(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    from .test_async_resources import _deployment

    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/deployments/",
        json={"deployment": _deployment(), "auth_token": "pk_deploy_x"},
    )
    await client.deployments.create(
        "abcdef01-2345-6789-abcd-ef0123456789",
        name="dep",
        compute_type="cpu",
        gpu_type=None,
        inference_config={"conf": 0.5},
    )
    body = httpx_mock.get_request().read()
    assert b'"name":"dep"' in body.replace(b" ", b"")
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/deployments/quote?compute_type=cpu&min_containers=1",
        json={
            "quote": {
                "rate_per_min_micro_usd": 1,
                "cost_per_hour_micro_usd": 60,
                "cost_per_day_micro_usd": 1440,
                "scale_to_zero": False,
                "billing_note": "warm",
            }
        },
    )
    assert not (await client.deployments.quote(compute_type="cpu", min_containers=1)).scale_to_zero


# ───────────── connectors: RemoteDataset obj + import wait=True ─────────────


async def test_connectors_import_remote_dataset_obj_and_wait(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/connectors/import/start",
        json={"import_id": "imp1", "status": "started"},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/connectors/import/status/imp1",
        json={"import_id": "imp1", "status": "completed"},
    )
    ds = RemoteDataset(id="1", name="d", slug="d")
    job = await client.connectors.import_("v7", "tok", [ds], poll_interval=0.01)
    assert job.status == "completed"


# ───────────── webhooks: create with opts + deliveries filters ─────────────


async def test_webhooks_create_opts_and_delivery_filters(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    ep = {
        "id": "w1",
        "url": "https://hook",
        "organization_id": "0a111111-2222-3333-4444-555566667777",
    }
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/webhooks/endpoints",
        json={"endpoint": ep, "secret": "whsec_x"},
    )
    await client.webhooks.create("https://hook", description="d", event_types=["run.completed"])
    body = httpx_mock.get_request().read()
    assert b"event_types" in body and b"description" in body
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/webhooks/deliveries?limit=50&offset=0&endpoint_id=beefcafe-0000-1111-2222-333344445555&status=failed",
        json={"deliveries": []},
    )
    assert (
        await client.webhooks.deliveries(
            endpoint="beefcafe-0000-1111-2222-333344445555", status="failed"
        )
        == []
    )


# ───────────── api_keys: create with opts (datetime + str expiry) ─────────────


async def test_api_keys_create_with_options(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    created = {
        "key_id": "k2",
        "api_key": "pk_live_secret",
        "name": "key2",
        "key_prefix": "pk_live_wxyz",
        "role": "admin",
        "rate_limit": 100,
        "created_at": "2026-01-01T00:00:00Z",
    }
    httpx_mock.add_response(method="POST", url=f"{BASE_URL}/api/v1/api-keys/", json=created)
    await client.api_keys.create(
        "key2",
        organization="0a111111-2222-3333-4444-555566667777",
        role="admin",
        rate_limit=100,
        expires_at=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )
    assert b"expires_at" in httpx_mock.get_request().read()
    httpx_mock.add_response(method="POST", url=f"{BASE_URL}/api/v1/api-keys/", json=created)
    await client.api_keys.create(
        "key2",
        organization="0a111111-2222-3333-4444-555566667777",
        expires_at="2027-01-01T00:00:00Z",
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/api/v1/api-keys/?organization_id=0a111111-2222-3333-4444-555566667777",
        json={"api_keys": []},
    )
    assert await client.api_keys.list(organization="0a111111-2222-3333-4444-555566667777") == []


# ───────────── auto_annotate: point/box/batch with all options ─────────────


async def test_auto_annotate_full_options(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{DEV}/auto-annotate/sam3/point", json={"status": "no_detection"}
    )
    res = await client.auto_annotate.point(
        "d", "a.jpg", x=1, y=2, positive_points=[(3, 4)], negative_points=[(5, 6)]
    )
    assert res.status == "no_detection" and res.annotations == []
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/auto-annotate/sam3/box",
        json={"status": "success", "annotations": []},
    )
    await client.auto_annotate.box(
        "d",
        "a.jpg",
        box={"x": 1, "y": 2, "w": 3, "h": 4},
        name="c",
        negative_boxes=[{"x": 0, "y": 0, "w": 1, "h": 1}],
    )
    httpx_mock.add_response(
        method="POST", url=f"{DEV}/auto-annotate/batch", json={"job_id": "j1", "status": "pending"}
    )
    job = await client.auto_annotate.batch(
        "d",
        ["a.jpg"],
        [{"name": "c", "output_type": "bbox"}],
        model="abcdef01-2345-6789-abcd-ef0123456789",
        sahi=True,
        wait=False,
    )
    assert job.job_id == "j1"
    body = httpx_mock.get_requests()[-1].read()
    assert b"sahi_enabled" in body and b"model_id" in body


# ───────────── batch: copy dup handling + permanent delete ─────────────


async def test_batch_copy_and_permanent_delete(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/api/v1/developer/batch/images/copy",
        json={"success": True, "processed": 1},
    )
    await client.batch.copy("d", ["i"], duplicate_handling="skip", copy_annotations=True)
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/api/v1/developer/batch/images/delete",
        json={"success": True, "processed": 1},
    )
    await client.batch.delete("d", ["i"], permanent=True)
    assert b'"permanent":true' in httpx_mock.get_requests()[-1].read().replace(b" ", b"")


# ───────────── training: list filters + create wait=True ─────────────


async def test_training_list_filters(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    from .test_async_resources import _run

    httpx_mock.add_response(method="GET", json={"training_runs": [_run()]})
    await client.training.list(dataset_name="ds", status="running")
    req = httpx_mock.get_request()
    assert b"dataset_name" in req.url.query and b"status" in req.url.query
