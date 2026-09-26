"""Happy-path + key-error smoke tests for every ``AsyncClient`` resource.

Each test drives a real ``AsyncClient`` (with retries off) against ``pytest-httpx``
canned responses, verifying the async resource issues the right request and parses
the response into the same typed model the sync resource returns. Poll loops run
with an injected no-op async sleep so nothing actually waits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from pictograph import AsyncClient
from pictograph.exceptions import ApiError, NotFoundError, PollTimeoutError
from pictograph.models.annotation import BBoxAnnotation, BoundingBox
from pictograph.resources._deployment_client import DeploymentClient

from .conftest import API_KEY, BASE_URL

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


# ───────────── payload builders ─────────────


def _dataset(**o: Any) -> dict[str, Any]:
    return {"id": "ds1", "name": "road-signs", "created_at": "2026-01-01T00:00:00Z", **o}


def _image(**o: Any) -> dict[str, Any]:
    return {
        "id": "11aa22bb-33cc-44dd-55ee-66ff77008899",
        "filename": "a.jpg",
        "image_url": "https://cdn/x",
        "created_at": "2026-01-01T00:00:00Z",
        **o,
    }


def _project(**o: Any) -> dict[str, Any]:
    return {
        "id": "p1",
        "name": "proj",
        "organization_id": "0a111111-2222-3333-4444-555566667777",
        "created_at": "2026-01-01T00:00:00Z",
        **o,
    }


def _export(**o: Any) -> dict[str, Any]:
    return {
        "id": "e1",
        "name": "exp",
        "dataset_name": "road-signs",
        "project_id": "p1",
        "format": "pictograph",
        "status": "pending",
        "created_at": "2026-01-01T00:00:00Z",
        **o,
    }


def _run(**o: Any) -> dict[str, Any]:
    return {
        "id": "r1",
        "name": "run",
        "organization_id": "0a111111-2222-3333-4444-555566667777",
        "pipeline_type": "yolox",
        "status": "pending",
        "created_at": "2026-01-01T00:00:00Z",
        **o,
    }


def _model(**o: Any) -> dict[str, Any]:
    return {
        "id": "m1",
        "name": "mdl",
        "organization_id": "0a111111-2222-3333-4444-555566667777",
        "model_type": "object_detection",
        "status": "ready",
        "visibility": "private",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        **o,
    }


# ───────────── datasets ─────────────


async def test_datasets_list_get_insights(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{DEV}/datasets/?limit=100&offset=0", json={"data": [_dataset()]}
    )
    assert (await client.datasets.list())[0].name == "road-signs"

    httpx_mock.add_response(
        method="GET", url=f"{DEV}/datasets/road-signs", json={"data": _dataset()}
    )
    assert (await client.datasets.get("road-signs")).id == "ds1"

    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/datasets/road-signs/insights",
        json={
            "data": {
                "total_images": 1,
                "total_annotations": 2,
            }
        },
    )
    assert (await client.datasets.insights("road-signs")).total_annotations == 2


async def test_datasets_iter_pages(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/datasets/?offset=0&limit=2",
        json={"data": [_dataset(id="a"), _dataset(id="b")]},
    )
    httpx_mock.add_response(
        method="GET", url=f"{DEV}/datasets/?offset=2&limit=2", json={"data": [_dataset(id="c")]}
    )
    ids = [d.id async for d in client.datasets.iter(page_size=2)]
    assert ids == ["a", "b", "c"]


async def test_datasets_storage_and_freeze(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/datasets/ds1/storage",
        json={"data": {"storage_class": "standard", "storage_state": "idle"}},
    )
    assert (await client.datasets.storage_status(dataset_id="ds1")).storage_state == "idle"
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/datasets/ds1/storage/freeze",
        json={"data": {"job_id": "job1", "storage_state": "freezing"}},
    )
    assert (await client.datasets.freeze(dataset_id="ds1")).storage_state == "freezing"


async def test_datasets_download_images_only(
    httpx_mock: HTTPXMock, client: AsyncClient, tmp_path: Path
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/datasets/road-signs/download?mode=images_only&limit=10000",
        json={
            "data": {"id": "ds1", "items": [{"filename": "a.jpg", "image_url": "https://gcs/a"}]}
        },
    )
    httpx_mock.add_response(method="GET", url="https://gcs/a", content=b"jpegbytes")
    report = await client.datasets.download("road-signs", tmp_path, mode="images_only")
    assert report.images_downloaded == 1
    assert report.success
    assert (tmp_path / "a.jpg").read_bytes() == b"jpegbytes"


# ───────────── images ─────────────


async def test_images_list_get_delete(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(method="GET", json={"data": [_image()]})
    assert (await client.images.list("11111111-1111-1111-1111-111111111111"))[0].filename == "a.jpg"
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/images/11aa22bb-33cc-44dd-55ee-66ff77008899/metadata",
        json={"data": _image()},
    )
    assert (
        await client.images.get(
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "11aa22bb-33cc-44dd-55ee-66ff77008899"
        )
    ).id == "11aa22bb-33cc-44dd-55ee-66ff77008899"
    httpx_mock.add_response(method="DELETE", url=f"{DEV}/images/road-signs/img.jpg")
    assert await client.images.delete("road-signs", "img.jpg") is None


async def test_images_upload_three_step(
    httpx_mock: HTTPXMock, client: AsyncClient, tmp_path: Path
) -> None:
    f = tmp_path / "a.jpg"
    f.write_bytes(b"\xff\xd8\xff")  # jpeg-ish
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/images/upload-url",
        json={"data": {"upload_url": "https://gcs/put", "expires_in_minutes": 15}},
    )
    httpx_mock.add_response(method="PUT", url="https://gcs/put", status_code=200)
    # Register returns the full canonical image - no metadata re-fetch.
    httpx_mock.add_response(method="POST", url=f"{DEV}/images/register", json={"data": _image()})
    img = await client.images.upload("11111111-1111-1111-1111-111111111111", f)
    assert img.id == "11aa22bb-33cc-44dd-55ee-66ff77008899"


async def test_images_bulk_tag(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(method="POST", url=f"{DEV}/images/bulk-tag", json={"processed": 3})
    assert (
        await client.images.bulk_tag(
            "11111111-1111-1111-1111-111111111111", ["i1", "i2", "i3"], ["car"]
        )
        == 3
    )


async def test_images_download(httpx_mock: HTTPXMock, client: AsyncClient, tmp_path: Path) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{DEV}/images/11aa22bb-33cc-44dd-55ee-66ff77008899", content=b"rawbytes"
    )
    out = await client.images.download(
        "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "11aa22bb-33cc-44dd-55ee-66ff77008899",
        tmp_path / "out.jpg",
    )
    assert out.read_bytes() == b"rawbytes"


# ───────────── annotations ─────────────


async def test_annotations_get_save_delete(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    ann = {
        "id": "a",
        "name": "car",
        "type": "bbox",
        "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4},
    }
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/annotations/road-signs/img.jpg",
        json={"annotations": [ann]},
    )
    got = await client.annotations.get("road-signs", "img.jpg")
    assert got[0].name == "car"

    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/annotations/road-signs/img.jpg",
        json={
            "image_id": "22222222-2222-2222-2222-222222222222",
            "previous_count": 0,
            "new_count": 1,
            "status": "in_progress",
        },
    )
    box = BBoxAnnotation(id="a", name="car", bounding_box=BoundingBox(x=1, y=2, w=3, h=4))
    res = await client.annotations.save("road-signs", "img.jpg", [box])
    assert res.new_count == 1

    httpx_mock.add_response(
        method="DELETE",
        url=f"{DEV}/annotations/road-signs/img.jpg",
        json={"image_id": "22222222-2222-2222-2222-222222222222", "deleted_count": 1},
    )
    assert (await client.annotations.delete("road-signs", "img.jpg")).deleted_count == 1


# ───────────── projects ─────────────


async def test_datasets_crud(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{DEV}/datasets/?limit=100&offset=0", json={"data": [_project()]}
    )
    assert (await client.datasets.list())[0].name == "proj"
    httpx_mock.add_response(method="POST", url=f"{DEV}/datasets/", json={"data": _project()})
    assert (await client.datasets.create("proj")).id == "p1"
    httpx_mock.add_response(
        method="PATCH", url=f"{DEV}/datasets/proj", json={"data": _project(name="new")}
    )
    assert (await client.datasets.update("proj", new_name="new")).name == "new"
    httpx_mock.add_response(
        method="DELETE",
        url=f"{DEV}/datasets/proj",
        json={"data": {"id": "p1", "name": "proj", "deleted": True, "images_deleted": 5}},
    )
    assert (await client.datasets.delete("proj"))["images_deleted"] == 5
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/datasets/proj/archive",
        json={"data": _project(is_archived=True)},
    )
    assert (await client.datasets.archive("proj")).is_archived is True
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/datasets/proj/unarchive",
        json={"data": _project()},
    )
    assert (await client.datasets.unarchive("proj")).is_archived is False


# ───────────── exports ─────────────


async def test_exports_create_nowait_and_bulk_delete(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    httpx_mock.add_response(method="POST", url=f"{DEV}/exports/", json={"data": _export()})
    exp = await client.exports.create("road-signs", "exp", wait=False)
    assert exp.status == "pending"
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/exports/bulk-delete",
        json={"data": {"succeeded": ["e1"], "not_found": [], "count": 1}},
    )
    assert (await client.exports.bulk_delete(["e1"])).count == 1


async def test_exports_wait_for_completion_polls(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/exports/exp?dataset=road-signs",
        json={"data": _export(status="processing")},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/exports/exp?dataset=road-signs",
        json={"data": _export(status="completed")},
    )
    done = await client.exports.wait_for_completion(
        "road-signs", "exp", poll_interval=0.01, sleep=_no_sleep
    )
    assert done.status == "completed"


async def test_exports_wait_raises_on_failed(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/exports/exp?dataset=road-signs",
        json={"data": _export(status="failed", error_message="boom")},
    )
    with pytest.raises(ApiError, match="boom"):
        await client.exports.wait_for_completion("road-signs", "exp", sleep=_no_sleep)


# ───────────── training ─────────────


async def test_training_create_list_cancel(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(method="POST", url=f"{DEV}/training/", json={"data": _run()})
    run = await client.training.create(
        "road-signs", "exp", pipeline_type="yolox", name="run", wait=False
    )
    assert run.status == "pending"
    httpx_mock.add_response(method="GET", json={"data": [_run()]})
    assert len(await client.training.list()) == 1
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/training/r1/cancel",
        json={"data": _run(status="cancelled")},
    )
    assert (await client.training.cancel(run_id="r1")).status == "cancelled"


async def test_training_bulk_cancel(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/training/bulk-cancel",
        json={"data": {"succeeded": ["r1"], "not_found": [], "count": 1}},
    )
    assert (await client.training.bulk_cancel(["r1"])).count == 1


# ───────────── models ─────────────


async def test_models_list_get_fork_delete(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(method="GET", json={"data": [_model()]})
    assert (await client.models.list())[0].name == "mdl"
    httpx_mock.add_response(method="GET", url=f"{DEV}/models/m1", json={"data": _model()})
    assert (await client.models.get(model_id="m1")).id == "m1"
    httpx_mock.add_response(
        method="POST", url=f"{DEV}/models/acme/m1/fork", json={"data": _model(id="m2")}
    )
    assert (await client.models.fork("acme", "m1")).id == "m2"
    httpx_mock.add_response(method="DELETE", url=f"{DEV}/models/m1")
    assert await client.models.delete(model_id="m1") is None


async def test_models_download(httpx_mock: HTTPXMock, client: AsyncClient, tmp_path: Path) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/models/m1/download?format=onnx",
        json={"data": {"download_url": "https://gcs/w"}},
    )
    httpx_mock.add_response(method="GET", url="https://gcs/w", content=b"onnxbytes")
    out = await client.models.download(model_id="m1", output_path=tmp_path / "m.onnx")
    assert out.read_bytes() == b"onnxbytes"


async def test_models_predict_remote(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    # get_by_name resolves the name first, then the multipart predict POST.
    httpx_mock.add_response(
        method="GET", url=f"{DEV}/models/mdl", json={"data": _model(name="mdl")}
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/models/m1/predict?confidence_threshold=0.5&top_k=3",
        json={
            "data": {
                "success": True,
                "annotations": [
                    {
                        "name": "car",
                        "type": "bbox",
                        "confidence": 0.8,
                        "bounding_box": {"x": 1, "y": 1, "w": 2, "h": 2},
                    }
                ],
                "tags": [],
                "model_type": "object_detection",
                "inference_seconds": 0.9,
            }
        },
    )
    result = await client.models.predict("mdl", image=b"jpegbytes")
    assert result.annotations[0]["name"] == "car"
    assert result.inference_seconds == 0.9
    request = httpx_mock.get_requests()[-1]
    assert b"jpegbytes" in request.content


# ───────────── deployments ─────────────


def _deployment(**o: Any) -> dict[str, Any]:
    return {
        "id": "d1111111-1111-1111-1111-111111111111",
        "name": "dep",
        "organization_id": "0a111111-2222-3333-4444-555566667777",
        "model_id": "abcdef01-2345-6789-abcd-ef0123456789",
        "compute_type": "gpu",
        "status": "active",
        "min_containers": 0,
        "max_containers": 1,
        "scaledown_window": 60,
        "endpoint_url": "https://gw/slug/predict",
        **o,
    }


async def test_deployments_list_create_pause(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/deployments/?limit=50&offset=0",
        json={"deployments": [_deployment()]},
    )
    assert (await client.deployments.list())[0].id == "d1111111-1111-1111-1111-111111111111"
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/deployments/",
        json={"deployment": _deployment(status="provisioning"), "auth_token": "pk_deploy_x"},
    )
    created = await client.deployments.create("abcdef01-2345-6789-abcd-ef0123456789")
    assert created.auth_token == "pk_deploy_x"  # noqa: S105 - test fixture, not a real secret
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/deployments/d1111111-1111-1111-1111-111111111111/pause",
        json={"deployment": _deployment(status="paused")},
    )
    assert (
        await client.deployments.pause("d1111111-1111-1111-1111-111111111111")
    ).status == "paused"


async def test_deployments_connect_builds_client(client: AsyncClient) -> None:
    from pictograph.models.deployment import Deployment

    dep = Deployment.model_validate(_deployment())
    dc = client.deployments.connect(dep, "pk_deploy_x")
    assert isinstance(dc, DeploymentClient)


# ───────────── credits ─────────────


async def test_credits_balance_history_estimate(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(method="GET", url=f"{DEV}/credits/balance", json={})
    assert (await client.credits.balance()) is not None
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/credits/history?limit=50&offset=0",
        json={
            "entries": [
                {
                    "id": "l1",
                    "amount": -100,
                    "operation": "training_a10g",
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ]
        },
    )
    assert (await client.credits.history())[0].operation == "training_a10g"
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/credits/estimate?operation=training_a10g&quantity=1",
        json={"operation": "training_a10g", "quantity": 1, "unit": "per_minute"},
    )
    assert (await client.credits.estimate("training_a10g")).operation == "training_a10g"


# ───────────── organizations ─────────────


async def test_organizations_me_and_invite(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    org = {
        "id": "org1",
        "name": "Org",
        "slug": "org",
        "subscription_tier": "core",
        "credits_monthly_allowance": 0,
        "credits_remaining": 0,
        "max_images": 100,
        "max_storage_bytes": 1,
        "max_users": 5,
        "member_count": 1,
        "pending_invite_count": 0,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    httpx_mock.add_response(method="GET", url=f"{DEV}/organizations/me", json={"organization": org})
    assert (await client.organizations.me()).slug == "org"
    inv = {
        "id": "i1",
        "email": "a@b.co",
        "role": "member",
        "status": "pending",
        "organization_id": "0a111111-2222-3333-4444-555566667777",
        "created_at": "2026-01-01T00:00:00Z",
        "expires_at": "2026-02-01T00:00:00Z",
    }
    httpx_mock.add_response(method="POST", url=f"{DEV}/organizations/invites", json={"invite": inv})
    assert (await client.organizations.invite("a@b.co")).email == "a@b.co"


# ───────────── directories ─────────────


async def test_directories_list_tree_stats(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    directory = {
        "id": "f1",
        "name": "train",
        "full_path": "/train",
        "organization_id": "0a111111-2222-3333-4444-555566667777",
        "project_id": "p1",
    }
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/directories/33333333-3333-3333-3333-333333333333",
        json=[directory],
    )
    assert (await client.directories.list("33333333-3333-3333-3333-333333333333"))[
        0
    ].name == "train"
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/directories/33333333-3333-3333-3333-333333333333/tree",
        json=[{**directory, "children": []}],
    )
    assert len(await client.directories.tree("33333333-3333-3333-3333-333333333333")) == 1
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/directories/33333333-3333-3333-3333-333333333333/train/stats?include_subdirectories=true",
        json={"total_directories": 1, "total_images": 10, "total_size_bytes": 999},
    )
    assert (
        await client.directories.stats("33333333-3333-3333-3333-333333333333", "/train")
    ).total_images == 10


# ───────────── batch ─────────────


async def test_batch_move_delete(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/api/v1/developer/batch/images/move",
        json={"success": True, "processed": 2},
    )
    assert (
        await client.batch.move("road-signs", ["i1", "i2"], target_directory_path="/x")
    ).processed == 2
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/api/v1/developer/batch/images/delete",
        json={"success": True, "processed": 1},
    )
    assert (await client.batch.delete("road-signs", ["i1"])).processed == 1


# ───────────── search ─────────────


async def test_search_similarity_and_tag(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
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
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "11aa22bb-33cc-44dd-55ee-66ff77008899"
        )
    )[0].similarity == 0.9
    tagged = {
        "id": "i3",
        "filename": "c.jpg",
        "project_id": "p1",
        "annotation_count": 1,
        "status": "complete",
    }
    httpx_mock.add_response(method="GET", json={"results": [tagged]})
    assert (await client.search.by_tag(objects=["car"]))[0].id == "i3"


async def test_search_by_tag_requires_a_tag(client: AsyncClient) -> None:
    with pytest.raises(ValueError, match="objects"):
        await client.search.by_tag()


# ───────────── video ─────────────


async def test_video_upload_probe(
    httpx_mock: HTTPXMock, client: AsyncClient, tmp_path: Path
) -> None:
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"video")
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/video/upload-url",
        json={"upload_url": "https://gcs/put", "gcs_path": "p", "gcs_uri": "gs://b/p"},
    )
    httpx_mock.add_response(method="PUT", url="https://gcs/put", status_code=200)
    info = await client.video.upload(v)
    assert info.gcs_path == "p"
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/video/probe",
        json={
            "duration_seconds": 5.0,
            "frame_count": 150,
            "native_fps": 30.0,
            "width": 1920,
            "height": 1080,
        },
    )
    assert (await client.video.probe("p")).frame_count == 150


# ───────────── connectors ─────────────


async def test_connectors_validate_and_import_nowait(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{DEV}/connectors/validate", json={"valid": True, "datasets": []}
    )
    assert (await client.connectors.validate("v7", "tok")).valid
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/connectors/import/start",
        json={"import_id": "imp1", "status": "started"},
    )
    job = await client.connectors.import_(
        "v7", "tok", [{"id": "1", "name": "d", "slug": "d"}], wait=False
    )
    assert job.import_id == "imp1"


# ───────────── webhooks ─────────────


async def test_webhooks_create_list_replay(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    ep = {
        "id": "w1",
        "url": "https://hook",
        "organization_id": "0a111111-2222-3333-4444-555566667777",
    }
    httpx_mock.add_response(
        method="POST", url=f"{DEV}/webhooks/endpoints", json={"endpoint": ep, "secret": "whsec_x"}
    )
    created = await client.webhooks.create("https://hook")
    assert created.secret == "whsec_x"  # noqa: S105 - test fixture, not a real secret
    httpx_mock.add_response(method="GET", url=f"{DEV}/webhooks/endpoints", json={"endpoints": [ep]})
    assert (await client.webhooks.list())[0].id == "w1"
    httpx_mock.add_response(method="POST", url=f"{DEV}/webhooks/deliveries/del1/replay")
    assert await client.webhooks.replay("del1") is None


# ───────────── workflows ─────────────


async def test_workflows_create_run(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    wf = {
        "id": "fedcba98-1111-2222-3333-444455556666",
        "name": "w",
        "organization_id": "0a111111-2222-3333-4444-555566667777",
    }
    httpx_mock.add_response(method="POST", url=f"{DEV}/workflows/", json={"workflow": wf})
    assert (
        await client.workflows.create("w", {"version": 1, "nodes": [], "edges": []})
    ).id == "fedcba98-1111-2222-3333-444455556666"
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/workflows/fedcba98-1111-2222-3333-444455556666/run",
        json={"run_id": "run1"},
    )
    assert (await client.workflows.run("fedcba98-1111-2222-3333-444455556666")).run_id == "run1"
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/workflows/runs/run1",
        json={
            "run": {
                "id": "run1",
                "workflow_id": "fedcba98-1111-2222-3333-444455556666",
                "organization_id": "0a111111-2222-3333-4444-555566667777",
                "status": "completed",
            }
        },
    )
    assert (await client.workflows.get_run("run1")).status == "completed"


# ───────────── api_keys ─────────────


async def test_api_keys_list_create_delete(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
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
        method="GET", url=f"{BASE_URL}/api/v1/api-keys/", json={"api_keys": [key]}
    )
    assert (await client.api_keys.list())[0].id == "k1"
    created = {
        "key_id": "k2",
        "api_key": "pk_live_secret",
        "name": "key2",
        "key_prefix": "pk_live_wxyz",
        "role": "member",
        "rate_limit": 5000,
        "created_at": "2026-01-01T00:00:00Z",
    }
    httpx_mock.add_response(method="POST", url=f"{BASE_URL}/api/v1/api-keys/", json=created)
    assert (
        await client.api_keys.create("key2", organization="0a111111-2222-3333-4444-555566667777")
    ).api_key == "pk_live_secret"
    httpx_mock.add_response(method="DELETE", url=f"{BASE_URL}/api/v1/api-keys/k1")
    assert await client.api_keys.delete("k1") is None


# ───────────── auto_annotate ─────────────


async def test_auto_annotate_point_and_batch_nowait(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    ann = {
        "id": "a",
        "name": "car",
        "type": "polygon",
        "polygon": {"paths": [[{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5, "y": 6}]]},
    }
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/auto-annotate/sam3/point",
        json={"status": "success", "annotation": ann, "score": 0.9},
    )
    res = await client.auto_annotate.point("road-signs", "a.jpg", x=10, y=20)
    assert res.status == "success"
    assert len(res.annotations) == 1

    httpx_mock.add_response(
        method="POST", url=f"{DEV}/auto-annotate/batch", json={"job_id": "j1", "status": "pending"}
    )
    job = await client.auto_annotate.batch(
        "road-signs", ["a.jpg"], [{"name": "car", "output_type": "polygon"}], wait=False
    )
    assert job.job_id == "j1"


async def test_auto_annotate_wait_for_batch_error(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/auto-annotate/batch/j1",
        json={"job_id": "j1", "status": "failed", "error_message": "kaput"},
    )
    with pytest.raises(ApiError, match="kaput"):
        await client.auto_annotate.wait_for_batch("j1", sleep=_no_sleep)


# ───────────── error propagation ─────────────


async def test_get_404_raises_not_found(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/datasets/11aa22bb-33cc-44dd-55ee-66ff77008899",
        status_code=404,
        json={"detail": "no"},
    )
    with pytest.raises(NotFoundError):
        await client.datasets.get("11aa22bb-33cc-44dd-55ee-66ff77008899")


async def test_wait_for_storage_times_out(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/datasets/ds1/storage",
        json={"data": {"storage_class": "coldline", "storage_state": "restoring"}},
        is_reusable=True,
    )
    with pytest.raises(PollTimeoutError):
        await client.datasets.wait_for_storage(
            dataset_id="ds1", timeout=0.05, poll_interval=0.01, sleep=_no_sleep
        )


async def test_images_review_approve_and_request_changes(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/images/road-signs/img.jpg/review",
        json={
            "id": "11aa22bb-33cc-44dd-55ee-66ff77008899",
            "status": "complete",
            "review_note": None,
            "processed": 1,
        },
    )
    assert await client.images.review("road-signs", "img.jpg", "approve") == "complete"

    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/images/road-signs/other.jpg/review",
        json={
            "id": "22aa22bb-33cc-44dd-55ee-66ff77008899",
            "status": "annotate",
            "review_note": "fix it",
            "processed": 1,
        },
    )
    assert (
        await client.images.review("road-signs", "other.jpg", "request_changes", note="fix it")
        == "annotate"
    )


async def test_images_split_filter_and_set(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/images/?dataset=11111111-1111-1111-1111-111111111111&limit=100&offset=0&split=test",
        json={"data": [_image(id="i1", status="new", split="test")]},
    )
    imgs = await client.images.list("11111111-1111-1111-1111-111111111111", split="test")
    assert imgs[0].split == "test"

    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/images/road-signs/img.jpg/split",
        json={"id": "i1", "split": "train"},
    )
    assert await client.images.set_split("road-signs", "img.jpg", "train") == "train"


async def test_annotations_rename_class(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/annotations/rename-class",
        json={
            "data": {
                "dataset_id": "ds1",
                "old_name": "car",
                "new_name": "vehicle",
                "images_updated": 2,
                "annotations_updated": 5,
                "config_updated": True,
            }
        },
    )
    result = await client.annotations.rename_class(
        "11111111-1111-1111-1111-111111111111", "car", "vehicle"
    )
    assert result.annotations_updated == 5 and result.config_updated is True


async def test_annotations_merge_class(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/annotations/merge-class",
        json={
            "data": {
                "dataset_id": "ds1",
                "source_name": "auto",
                "target_name": "vehicle",
                "images_updated": 2,
                "annotations_updated": 5,
                "config_updated": True,
            }
        },
    )
    result = await client.annotations.merge_class(
        "11111111-1111-1111-1111-111111111111", "auto", "vehicle"
    )
    assert result.annotations_updated == 5 and result.target_name == "vehicle"


async def test_annotations_delete_class(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/annotations/delete-class",
        json={
            "data": {
                "dataset_id": "ds1",
                "name": "obsolete",
                "config_updated": True,
                "images_updated": 1,
                "annotations_removed": 3,
            }
        },
    )
    result = await client.annotations.delete_class(
        "11111111-1111-1111-1111-111111111111", "obsolete", delete_annotations=True
    )
    assert result.annotations_removed == 3 and result.config_updated is True
