"""The SDK parses the job-shaped responses the API has actually sent.

From 2026-08-01 to 2026-10-02 the API's auto-annotate batch, video extraction and
connector import routes answered with a bare ``job`` key where these models required
``job_id`` / ``import_id`` (a server-side vocabulary rename). Every call below raised -
``batch()`` AFTER the job was created and its deposit charged. The API now sends both
keys; the SDK accepts either, so it works against every server build.

The bodies here are copied from the routes: ``BARE`` is what the API sent in that window,
``BOTH`` is what it sends now. Before this change every ``BARE`` case raised
(ServerError, or KeyError on the connector kicker), and a V7 import cancelled mid-way
failed to parse at all: the import worker records a cancelled dataset as ``"cancelled"``,
which ``DatasetImportStatus`` did not allow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from pictograph import AsyncClient, Client
from pictograph.exceptions import ServerError
from pictograph.models.connector import kicker_import_id

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
DEV = f"{BASE}/api/v1/developer"
KEY = "pk_live_test"
JOB = "44444444-3333-2222-1111-000000000000"
IMPORT = "9f1c2d3e4b5a69788796a5b4c3d2e1f0"

BATCH_KICK_BARE = {
    "success": True,
    "job": JOB,
    "status": "pending",
    "total_images": 4,
    "estimated_credits": 394,
}
BATCH_KICK_BOTH = {**BATCH_KICK_BARE, "job_id": JOB}
BATCH_STATUS_BARE = {
    "job": JOB,
    "status": "completed",
    "progress": 100,
    "total_images": 4,
    "processed_images": 4,
    "total_annotations_added": 9,
    "failed_images": 0,
    "error_message": None,
    "completed_at": "2026-10-02T09:00:00+00:00",
}
BATCH_CANCEL_BARE = {"job": JOB, "status": "cancelled"}
VIDEO_KICK_BARE = {"job": JOB, "status": "processing"}
VIDEO_STATUS_BARE = {
    "job": JOB,
    "status": "complete",
    "progress": 100,
    "frames_extracted": 12,
    "total_frames": 12,
    "error": None,
    "directory_path": "/frames",
    "warning": None,
    "image_ids": ["a", "b"],
}
IMPORT_KICK_BARE = {
    "job": IMPORT,
    "status": "started",
    "datasets": [{"name": "birds", "project_id": "p-1", "dataset_id": "p-1"}],
}
IMPORT_STATUS_CANCELLED = {
    "job": IMPORT,
    "status": "cancelled",
    "progress": 40.0,
    "total_images": 10,
    "imported_images": 4,
    "failed_images": 0,
    "current_dataset": "birds",
    # What pictograph_connector_import_service writes for a dataset cut short by a cancel.
    "datasets": [
        {
            "name": "birds",
            "project_id": "p-1",
            "dataset_id": "p-1",
            "status": "cancelled",
            "imported": 4,
            "failed": 0,
        }
    ],
}


@pytest.fixture
def client() -> Any:
    c = Client(api_key=KEY, base_url=BASE, max_retries=0)
    yield c
    c.close()


# ── auto-annotate batch ────────────────────────────────────────────────────────


@pytest.mark.parametrize("kicker", [BATCH_KICK_BARE, BATCH_KICK_BOTH], ids=["bare", "both"])
def test_batch_kicker_parses(httpx_mock: HTTPXMock, client: Client, kicker: dict) -> None:
    httpx_mock.add_response(method="POST", url=f"{DEV}/auto-annotate/batch", json=kicker)
    job = client.auto_annotate.batch(
        "demo", ["a.png"], [{"name": "car", "output_type": "bbox"}], wait=False
    )
    assert (job.job_id, job.status, job.estimated_credits) == (JOB, "pending", 394)


def test_batch_waits_through_bare_status_polls(httpx_mock: HTTPXMock, client: Client) -> None:
    httpx_mock.add_response(method="POST", url=f"{DEV}/auto-annotate/batch", json=BATCH_KICK_BARE)
    httpx_mock.add_response(
        method="GET", url=f"{DEV}/auto-annotate/batch/{JOB}", json=BATCH_STATUS_BARE
    )
    job = client.auto_annotate.batch(
        "demo", ["a.png"], [{"name": "car", "output_type": "bbox"}], poll_interval=0.01
    )
    assert (job.job_id, job.status, job.total_annotations_added) == (JOB, "completed", 9)


def test_batch_cancel_parses_bare(httpx_mock: HTTPXMock, client: Client) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{DEV}/auto-annotate/batch/{JOB}/cancel", json=BATCH_CANCEL_BARE
    )
    job = client.auto_annotate.cancel_batch(JOB)
    assert (job.job_id, job.status) == (JOB, "cancelled")


# ── video frame extraction ────────────────────────────────────────────────────


def test_extract_frames_parses_bare(httpx_mock: HTTPXMock, client: Client) -> None:
    httpx_mock.add_response(method="POST", url=f"{DEV}/video/extract-frames", json=VIDEO_KICK_BARE)
    httpx_mock.add_response(
        method="GET", url=f"{DEV}/video/extract-frames/{JOB}", json=VIDEO_STATUS_BARE
    )
    job = client.video.extract_frames(
        "demo", "gs://b/v.mp4", directory_name="frames", poll_interval=0.01
    )
    assert (job.job_id, job.status, job.image_ids) == (JOB, "complete", ["a", "b"])


# ── connector import ──────────────────────────────────────────────────────────


def test_import_kicker_reads_either_key(httpx_mock: HTTPXMock, client: Client) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{DEV}/connectors/import/start", json=IMPORT_KICK_BARE
    )
    job = client.connectors.import_(
        "v7", "k", [{"id": "d1", "name": "birds", "slug": "birds", "image_count": 10}], wait=False
    )
    assert job.import_id == IMPORT


def test_a_cancelled_import_parses(httpx_mock: HTTPXMock, client: Client) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/connectors/import/cancel/{IMPORT}",
        json={"status": "cancelled", "job": IMPORT},
    )
    httpx_mock.add_response(
        method="GET", url=f"{DEV}/connectors/import/status/{IMPORT}", json=IMPORT_STATUS_CANCELLED
    )
    job = client.connectors.cancel_import(IMPORT)
    assert (job.import_id, job.status) == (IMPORT, "cancelled")
    assert [(d.name, d.status) for d in job.datasets] == [("birds", "cancelled")]


def test_kicker_without_any_id_is_a_typed_error() -> None:
    with pytest.raises(ServerError):
        kicker_import_id({"status": "started"})


# ── the async twins ───────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_async_twins_parse_bare(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=f"{DEV}/auto-annotate/batch", json=BATCH_KICK_BARE)
    httpx_mock.add_response(
        method="GET", url=f"{DEV}/auto-annotate/batch/{JOB}", json=BATCH_STATUS_BARE
    )
    httpx_mock.add_response(method="POST", url=f"{DEV}/video/extract-frames", json=VIDEO_KICK_BARE)
    httpx_mock.add_response(
        method="POST", url=f"{DEV}/connectors/import/start", json=IMPORT_KICK_BARE
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{DEV}/connectors/import/cancel/{IMPORT}",
        json={"status": "cancelled", "job": IMPORT},
    )
    httpx_mock.add_response(
        method="GET", url=f"{DEV}/connectors/import/status/{IMPORT}", json=IMPORT_STATUS_CANCELLED
    )
    async with AsyncClient(api_key=KEY, base_url=BASE, max_retries=0) as c:
        kicked = await c.auto_annotate.batch(
            "demo", ["a.png"], [{"name": "car", "output_type": "bbox"}], wait=False
        )
        assert kicked.job_id == JOB
        done = await c.auto_annotate.get_batch(JOB)
        assert (done.job_id, done.status) == (JOB, "completed")
        video = await c.video.extract_frames(
            "demo", "gs://b/v.mp4", directory_name="frames", wait=False
        )
        assert video.job_id == JOB
        imp = await c.connectors.import_(
            "v7",
            "k",
            [{"id": "d1", "name": "birds", "slug": "birds", "image_count": 10}],
            wait=False,
        )
        assert imp.import_id == IMPORT
        cancelled = await c.connectors.cancel_import(IMPORT)
        assert [d.status for d in cancelled.datasets] == ["cancelled"]
