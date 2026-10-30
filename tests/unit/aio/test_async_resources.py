"""Happy-path + key-error smoke tests for every ``AsyncClient`` resource.

Each test drives a real ``AsyncClient`` (with retries off) against ``pytest-httpx``
canned responses, verifying the async resource issues the right request and parses
the response into the same typed model the sync resource returns. Poll loops run
with an injected no-op async sleep so nothing actually waits.
"""

from __future__ import annotations

import json
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
    assert (await client.datasets.storage_status(dataset="ds1")).storage_state == "idle"
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/datasets/ds1/storage/freeze",
        json={"data": {"job_id": "job1", "storage_state": "freezing"}},
    )
    assert (await client.datasets.freeze(dataset="ds1")).storage_state == "freezing"


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
    report = await client.datasets.download("road-signs", output_path=tmp_path, mode="images_only")
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
        # The image must be in the dataset the call names (B524).
        json={"data": _image(dataset_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")},
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
    img = await client.images.upload("11111111-1111-1111-1111-111111111111", file_path=f)
    assert img.id == "11aa22bb-33cc-44dd-55ee-66ff77008899"


async def test_images_bulk_tag(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    httpx_mock.add_response(method="POST", url=f"{DEV}/images/bulk-tag", json={"processed": 3})
    assert (
        await client.images.bulk_tag(
            "11111111-1111-1111-1111-111111111111", ["i1", "i2", "i3"], tags=["car"]
        )
        == 3
    )


async def test_images_download(httpx_mock: HTTPXMock, client: AsyncClient, tmp_path: Path) -> None:
    """Path-addressed, like the sync twin (B524): a filename needs no lookup, and a
    folder's image is addressed by its folder path - the same route either way."""
    httpx_mock.add_response(method="GET", url=f"{DEV}/images/road-signs/img.jpg", content=b"root")
    httpx_mock.add_response(
        method="GET", url=f"{DEV}/images/road-signs/val/img.jpg", content=b"val"
    )
    out = await client.images.download("road-signs", "img.jpg", output_path=tmp_path / "root.jpg")
    assert out.read_bytes() == b"root"
    out = await client.images.download(
        "road-signs", "val/img.jpg", output_path=tmp_path / "val.jpg"
    )
    assert out.read_bytes() == b"val"


async def test_images_download_by_uuid_checks_the_dataset_then_streams_its_path(
    httpx_mock: HTTPXMock, client: AsyncClient, tmp_path: Path
) -> None:
    # A UUID is looked up once, checked to be in the named dataset, and re-addressed by
    # its folder + filename (the route cannot take an id).
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/images/11aa22bb-33cc-44dd-55ee-66ff77008899/metadata",
        json={
            "data": _image(dataset_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", directory_path="/val")
        },
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/images/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/val/a.jpg",
        content=b"rawbytes",
    )
    out = await client.images.download(
        "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "11aa22bb-33cc-44dd-55ee-66ff77008899",
        output_path=tmp_path / "out.jpg",
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
    res = await client.annotations.save("road-signs", "img.jpg", annotations=[box])
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
    assert (await client.datasets.update("proj", name="new")).name == "new"
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
    exp = await client.exports.create("road-signs", name="exp", wait=False)
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
    assert (await client.training.cancel(run="r1")).status == "cancelled"


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
    assert (await client.models.get(model="m1")).id == "m1"
    httpx_mock.add_response(
        method="POST", url=f"{DEV}/models/acme/m1/fork", json={"data": _model(id="m2")}
    )
    assert (await client.models.fork("acme", "m1")).id == "m2"
    httpx_mock.add_response(
        method="POST", url=f"{DEV}/models/acme/m1/fork", json={"data": _model(id="m3")}
    )
    assert (await client.models.fork("acme", "m1", name="Ours")).id == "m3"
    assert json.loads(httpx_mock.get_requests()[-1].content) == {"name": "Ours"}
    httpx_mock.add_response(method="DELETE", url=f"{DEV}/models/m1")
    assert await client.models.delete(model="m1") is None


async def test_models_download(httpx_mock: HTTPXMock, client: AsyncClient, tmp_path: Path) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/models/m1/download?format=onnx",
        json={"data": {"download_url": "https://gcs/w"}},
    )
    httpx_mock.add_response(method="GET", url="https://gcs/w", content=b"onnxbytes")
    out = await client.models.download(model="m1", output_path=tmp_path / "m.onnx")
    assert out.read_bytes() == b"onnxbytes"


async def test_models_predict_remote(httpx_mock: HTTPXMock, client: AsyncClient) -> None:
    # get resolves the name first, then the multipart predict POST.
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
    dc = client.deployments.connect(dep, api_key="pk_deploy_x")
    assert isinstance(dc, DeploymentClient)


_DEP = "d1111111-1111-1111-1111-111111111111"
_BUILDING = {
    "state": "building",
    "models": [
        {
            "model_id": "abcdef01-2345-6789-abcd-ef0123456789",
            "precision": "fp32",
            "state": "building",
            "estimate_s": 85.8,
        }
    ],
}


class _AsyncSleeps:
    """Recording and bounded: a wait that never stops fails instead of hanging."""

    def __init__(self, limit: int = 50) -> None:
        self.calls: list[float] = []
        self._limit = limit

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        assert len(self.calls) <= self._limit, f"polled {len(self.calls)} times: it never stopped"


async def test_deployments_wait_until_active_polls_until_active(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    httpx_mock.add_response(
        url=f"{DEV}/deployments/?limit=100&offset=0&name=dep",
        json={"deployments": [_deployment()]},
    )
    for status in ("provisioning", "provisioning", "active"):
        httpx_mock.add_response(
            method="GET",
            url=f"{DEV}/deployments/{_DEP}",
            json={"deployment": _deployment(status=status, engine_build=_BUILDING)},
        )
    sleeps = _AsyncSleeps()
    dep = await client.deployments.wait_until_active("dep", poll_interval=2.0, sleep=sleeps)
    assert dep.status == "active" and sleeps.calls == [2.0, 2.0]
    assert [str(r.url) for r in httpx_mock.get_requests()][1:] == [f"{DEV}/deployments/{_DEP}"] * 3


async def test_deployments_wait_until_active_takes_a_deployment(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    from pictograph.models.deployment import Deployment

    httpx_mock.add_response(
        method="GET", url=f"{DEV}/deployments/{_DEP}", json={"deployment": _deployment()}
    )
    created = Deployment.model_validate(_deployment(status="provisioning"))
    assert (await client.deployments.wait_until_active(created)).status == "active"
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.parametrize(
    ("status", "code"), [("failed", "job_failed"), ("paused", "job_cancelled")]
)
async def test_deployments_wait_until_active_raises_when_it_cannot_turn_active(
    httpx_mock: HTTPXMock, client: AsyncClient, status: str, code: str
) -> None:
    from pictograph.exceptions import ApiError

    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/deployments/{_DEP}",
        json={"deployment": _deployment(status=status, error_message="Provisioning failed: x")},
    )
    with pytest.raises(ApiError) as exc:
        await client.deployments.wait_until_active(_DEP, sleep=_AsyncSleeps())
    assert exc.value.code == code and exc.value.response["status"] == status


async def test_deployments_wait_until_active_times_out(
    httpx_mock: HTTPXMock, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pictograph.exceptions import PollTimeoutError

    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/deployments/{_DEP}",
        json={"deployment": _deployment(status="provisioning", engine_build=_BUILDING)},
        is_reusable=True,
    )

    class _Clock:
        def __init__(self, *ticks: float) -> None:
            self._ticks = list(ticks)

        def monotonic(self) -> float:
            return self._ticks.pop(0) if len(self._ticks) > 1 else self._ticks[0]

    monkeypatch.setattr("pictograph.aio.resources.deployments.time", _Clock(0.0, 0.0, 61.0))
    sleeps = _AsyncSleeps()
    with pytest.raises(PollTimeoutError, match="building its TensorRT engine"):
        await client.deployments.wait_until_active(_DEP, timeout=60.0, sleep=sleeps)
    assert sleeps.calls == [3.0]


async def test_deployments_wait_until_active_refuses_a_bad_window(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    with pytest.raises(ValueError):
        await client.deployments.wait_until_active(_DEP, poll_interval=0)
    assert httpx_mock.get_requests() == []


async def test_deployments_update_patches_only_the_scaling_given(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    httpx_mock.add_response(
        url=f"{DEV}/deployments/?limit=100&offset=0&name=dep",
        json={"deployments": [_deployment()]},
    )
    httpx_mock.add_response(
        method="PATCH",
        url=f"{DEV}/deployments/{_DEP}",
        json={"deployment": _deployment(min_containers=0, max_containers=3)},
    )
    dep = await client.deployments.update("dep", min_containers=0, max_containers=3)
    assert (dep.min_containers, dep.max_containers, dep.scaledown_window) == (0, 3, 60)
    lookup, patch = httpx_mock.get_requests()
    assert lookup.method == "GET" and (patch.method, str(patch.url)) == (
        "PATCH",
        f"{DEV}/deployments/{_DEP}",
    )
    # Only what was given, and the 0 with it: scaledown_window is not sent at all.
    assert json.loads(patch.content) == {"min_containers": 0, "max_containers": 3}


async def test_deployments_update_renames_and_writes_the_readme(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    httpx_mock.add_response(
        method="PATCH",
        url=f"{DEV}/deployments/{_DEP}",
        json={"deployment": _deployment(name="dock-gate", readme="Gate cameras.")},
    )
    dep = await client.deployments.update(_DEP, name="dock-gate", readme="Gate cameras.")
    assert (dep.name, dep.readme) == ("dock-gate", "Gate cameras.")
    patch = httpx_mock.get_requests()[-1]
    assert json.loads(patch.content) == {"name": "dock-gate", "readme": "Gate cameras."}


async def test_deployments_take_a_request_ceiling_like_the_sync_twin(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    """B584: set at create, changed (0 removes it) at update - the bodies the sync twin sends."""
    httpx_mock.add_response(
        method="PATCH",
        url=f"{DEV}/deployments/{_DEP}",
        json={"deployment": _deployment(rate_limit_rpm=None)},
    )
    dep = await client.deployments.update(_DEP, rate_limit_rpm=0)
    assert dep.rate_limit_rpm is None
    assert json.loads(httpx_mock.get_requests()[-1].content) == {"rate_limit_rpm": 0}
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/deployments/",
        json={"deployment": _deployment(rate_limit_rpm=600), "auth_token": "pk_deploy_x"},
    )
    created = await client.deployments.create(
        "abcdef01-2345-6789-abcd-ef0123456789", rate_limit_rpm=600
    )
    assert json.loads(httpx_mock.get_requests()[-1].content)["rate_limit_rpm"] == 600
    assert created.deployment.rate_limit_rpm == 600


async def test_deployments_update_with_nothing_to_change_raises(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    with pytest.raises(ValueError, match="Nothing to change"):
        await client.deployments.update("dep")
    assert httpx_mock.get_requests() == []


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
    assert (await client.batch.move("road-signs", ["i1", "i2"], directory="/x")).processed == 2
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
    # The source UUID is checked to be in the named dataset first (B524).
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/images/11aa22bb-33cc-44dd-55ee-66ff77008899/metadata",
        json={"data": _image(dataset_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")},
    )
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
    assert info.video == "p"
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
    assert (await client.connectors.validate("v7", api_key="tok")).valid
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/connectors/import/start",
        json={"import_id": "imp1", "status": "started"},
    )
    job = await client.connectors.import_(
        "v7", api_key="tok", datasets=[{"id": "1", "name": "d", "slug": "d"}], wait=False
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
        await client.workflows.create("w", graph={"version": 1, "nodes": [], "edges": []})
    ).id == "fedcba98-1111-2222-3333-444455556666"
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/workflows/fedcba98-1111-2222-3333-444455556666/run",
        json={"run_id": "run1"},
    )
    created = await client.workflows.run(
        "fedcba98-1111-2222-3333-444455556666", source={"kind": "dataset", "dataset": "d"}
    )
    assert created.run_id == "run1"
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


async def test_workflows_run_log_pages_and_counts(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    rec = {
        "ts": "2026-10-18T07:00:01Z",
        "engine_run": "e",
        "seq": 1,
        "record_index": 0,
        "type": "event",
        "trigger": "Gate",
        "classes": [],
        "track_ids": [],
    }
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/workflows/runs/run1/log?stream=events&limit=1&order=asc",
        json={"stream": "events", "records": [rec], "next_cursor": "c"},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/workflows/runs/run1/log?stream=events&limit=1&order=asc&cursor=c",
        json={"stream": "events", "records": [dict(rec, record_index=1)]},
    )
    got = [r.record_index async for r in client.workflows.iter_run_log("run1", page_size=1)]
    assert got == [0, 1]
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/workflows/runs/run1/log/summary",
        json={"log_frames": True, "streams": {"events": {"count": 2}}},
    )
    assert (await client.workflows.run_log_summary("run1")).streams["events"].count == 2


async def test_workflows_logs_filter_by_a_condition_branch(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    """B549, the async twins: a run's log, its iterator, the live log and ITS iterator all take ``branch=``."""
    wid = "7d4e1c2a-9f3b-4c5d-8e6f-0a1b2c3d4e5f"
    rec = {
        "ts": "2026-10-18T07:00:01Z",
        "engine_run": "e",
        "seq": 1,
        "record_index": 0,
        "condition": False,
    }
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/workflows/runs/run1/log?stream=frames&limit=50&order=asc&branch=false",
        json={"stream": "frames", "records": [rec]},
    )
    page = await client.workflows.run_log("run1", stream="frames", branch=False)
    assert page.records[0].condition is False
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/workflows/runs/run1/log?stream=frames&limit=200&order=asc&branch=true",
        json={"stream": "frames", "records": [dict(rec, condition=True)]},
    )
    got = [
        r.condition
        async for r in client.workflows.iter_run_log("run1", stream="frames", branch=True)
    ]
    assert got == [True]
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/workflows/{wid}/endpoint/log?stream=frames&limit=50&order=desc&branch=true",
        json={"stream": "frames", "records": []},
    )
    assert (await client.workflows.live_log(wid, branch=True)).records == []
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/workflows/{wid}/endpoint/log?stream=frames&limit=200&order=desc&branch=false",
        json={"stream": "frames", "records": [rec]},
    )
    assert [r.record_index async for r in client.workflows.iter_live_log(wid, branch=False)] == [0]


async def test_workflows_create_makes_its_endpoint_with_it(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    """B579 slice 5 (async twin): where its endpoint runs and how it scales are sent with its create, and its token
    comes back on the workflow once."""
    import json

    wf = {"id": "w1", "organization_id": "o", "name": "Gate counter"}
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/workflows/",
        json={"workflow": wf, "endpoint_token": "pk_deploy_once"},
    )
    made = await client.workflows.create(
        "Gate counter", region="us-west", max_containers=3, scaledown_window=120
    )
    assert json.loads(httpx_mock.get_requests()[-1].content) == {
        "name": "Gate counter",
        "region": "us-west",
        "max_containers": 3,
        "scaledown_window": 120,
    }
    assert made.endpoint_token == "pk_deploy_once"  # noqa: S105 - test fixture, not a real secret


async def test_workflows_compute_options_in_a_region_and_a_run_on_its_endpoint(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    """B579 slice 5: the GPUs priced in a region (a workflow's endpoint runs there), and a run on its
    workflow's endpoint - it names no GPU of its own."""
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/workflows/compute-options?region=us-west",
        json={
            "gpus": [
                {
                    "gpu": "l4",
                    "label": "L4 GPU",
                    "rate_per_min_micro_usd": 26667,
                    "cost_per_hour_micro_usd": 1600000,
                }
            ],
            "precisions": [{"precision": "fp16", "label": "FP16"}],
            "default": {"gpu": "t4"},
            "region": "us-west",
            "regions": [{"id": "us-west", "label": "US West", "location": "Oregon"}],
        },
    )
    options = await client.workflows.compute_options("us-west")
    assert options.gpus[0].cost_per_hour_micro_usd == 1600000 and options.region == "us-west"
    assert [r.id for r in options.regions] == ["us-west"]
    wid = "fedcba98-1111-2222-3333-444455556666"
    httpx_mock.add_response(method="POST", url=f"{DEV}/workflows/{wid}/run", json={"run_id": "r"})
    assert (await client.workflows.run(wid, source={"kind": "export", "export": "e"})).run_id == "r"
    sent = json.loads(httpx_mock.get_requests()[-1].content)
    assert sent == {"source": {"kind": "export", "export": "e"}}, "its input, and no GPU of its own"


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
        "road-signs", ["a.jpg"], classes=[{"name": "car", "output_type": "polygon"}], wait=False
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
            dataset="ds1", timeout=0.05, poll_interval=0.01, sleep=_no_sleep
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
    assert await client.images.review("road-signs", "img.jpg", action="approve") == "complete"

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
        await client.images.review(
            "road-signs", "other.jpg", action="request_changes", note="fix it"
        )
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
    assert await client.images.set_split("road-signs", "img.jpg", split="train") == "train"


async def test_images_assign_splits_async_twin_sends_only_the_weights_given(
    httpx_mock: HTTPXMock, client: AsyncClient
) -> None:
    """The async twin builds the same body as the sync one (B604): an omitted weight is the
    dataset's saved split, so it is not sent; the mode is."""
    import json

    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/images/assign-splits",
        json={
            "data": {
                "processed": 3,
                "train": 2,
                "val": 1,
                "test": 0,
                "clusters": 2,
                "unclustered": 0,
            }
        },
    )
    out = await client.images.assign_splits("road-signs", mode="embedding")
    assert out == {"processed": 3, "train": 2, "val": 1, "test": 0, "clusters": 2, "unclustered": 0}
    body = json.loads(httpx_mock.get_request().content)  # type: ignore[union-attr]
    assert body == {"dataset": "road-signs", "seed": 42, "mode": "embedding"}


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
        "11111111-1111-1111-1111-111111111111", "car", new_name="vehicle"
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
        "11111111-1111-1111-1111-111111111111", "auto", into="vehicle"
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


@pytest.mark.parametrize(
    ("runtime", "state"),
    [
        (
            {"state": "live", "containers": 1, "creating": 0, "checked_at": "2026-10-23T12:00:00Z"},
            "live",
        ),
        (
            {
                "state": "asleep",
                "containers": 0,
                "creating": 0,
                "checked_at": "2026-10-23T12:00:00Z",
            },
            "asleep",
        ),
        (
            {
                "state": "waiting",
                "containers": 0,
                "creating": 0,
                "waiting_since": "2026-10-23T11:59:00Z",
                "checked_at": "2026-10-23T12:00:00Z",
            },
            "waiting",
        ),
        (
            {
                "state": "dozing",
                "containers": 0,
                "creating": 0,
                "checked_at": "2026-10-23T12:00:00Z",
            },
            "dozing",
        ),
        (None, None),
    ],
    ids=["live", "asleep", "waiting", "a-later-state", "none"],
)
async def test_deployments_get_and_list_read_the_runtime(
    httpx_mock: HTTPXMock, client: AsyncClient, runtime: dict[str, Any] | None, state: str | None
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/deployments/{_DEP}",
        json={"deployment": _deployment(runtime=runtime)},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{DEV}/deployments/?limit=50&offset=0",
        json={"deployments": [_deployment(runtime=runtime)]},
    )
    got = await client.deployments.get(_DEP)
    listed = await client.deployments.list()
    for dep in (got, listed[0]):
        assert (dep.runtime.state if dep.runtime else None) == state
