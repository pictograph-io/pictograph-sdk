"""``pictograph auto-annotate dataset`` - draft labels for a whole dataset or folder (B533 P1).

Before it, drafting a dataset from the CLI meant ``auto-annotate batch --images`` with
every filename on one command line: 5,000 images, 5,000 names. These drive the REAL
app and the REAL client over a mocked HTTP transport, so each test covers the command,
the SDK's ``auto_annotate.dataset`` and the requests on the wire.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from pictograph.cli._app import app
from pictograph.cli._errors import ExitCode

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
IMAGES = re.compile(rf"{re.escape(BASE)}/api/v1/developer/images/\?.*")
BATCH = f"{BASE}/api/v1/developer/auto-annotate/batch"
MODEL_ID = "abcdef01-2345-6789-abcd-ef0123456789"
NOW = "2026-04-01T00:00:00Z"


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_test")
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    monkeypatch.setenv("PICTOGRAPH_MAX_RETRIES", "0")
    monkeypatch.delenv("PICTOGRAPH_OUTPUT", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    def invoke(*args: str, env: dict[str, str] | None = None) -> Any:
        return runner.invoke(app, ["auto-annotate", "dataset", *args], env=env)

    return invoke


def _image(image_id: str, filename: str, annotations: int = 0) -> dict[str, Any]:
    return {
        "id": image_id,
        "filename": filename,
        "annotation_count": annotations,
        "directory_path": "/val",
        "created_at": NOW,
    }


def _listing(httpx_mock: HTTPXMock, rows: list[dict[str, Any]]) -> None:
    httpx_mock.add_response(url=IMAGES, json={"data": rows})


def _job(status: str, **counts: Any) -> dict[str, Any]:
    return {"job_id": "job-1", "status": status, "progress": 100, **counts}


def _batch(httpx_mock: HTTPXMock, *, status_code: int = 200, body: Any = None) -> None:
    httpx_mock.add_response(
        method="POST", url=BATCH, status_code=status_code, json=body or _job("pending")
    )


def _poll(httpx_mock: HTTPXMock, body: dict[str, Any]) -> None:
    httpx_mock.add_response(method="GET", url=f"{BATCH}/job-1", json=body)


def _batch_body(httpx_mock: HTTPXMock) -> dict[str, Any]:
    (request,) = [r for r in httpx_mock.get_requests() if r.method == "POST"]
    body: dict[str, Any] = json.loads(request.content)
    return body


def _last_json_line(stderr: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(stderr.strip().splitlines()[-1])
    return payload["error"]


def test_drafts_every_unannotated_image_and_says_to_review_them(
    run: Any, httpx_mock: HTTPXMock
) -> None:
    _listing(httpx_mock, [_image("i1", "a.jpg"), _image("i2", "b.jpg", annotations=4)])
    _batch(httpx_mock)
    _poll(
        httpx_mock,
        _job("completed", processed_images=1, total_annotations_added=3, failed_images=0),
    )

    res = run("shelf", "--classes", "cereal box:bbox,shelf:polygon")

    assert res.exit_code == ExitCode.OK, res.output
    body = _batch_body(httpx_mock)
    assert body["images"] == ["i1"], "the annotated image is skipped, by id"
    assert body["classes"] == [
        {"name": "cereal box", "output_type": "bbox"},
        {"name": "shelf", "output_type": "polygon"},
    ]
    assert "Drafted 1 of 1 image(s) in shelf: 3 annotation(s) added." in res.stdout
    assert "Skipped 1 image(s)" in res.stdout
    # The draft -> review -> export -> train loop, in the command's own words.
    assert "review them before you train" in res.stderr
    assert "--min-confidence-lt 0.5" in res.stderr


def test_json_prints_the_report(run: Any, httpx_mock: HTTPXMock) -> None:
    _listing(httpx_mock, [_image("i1", "a.jpg")])
    _batch(httpx_mock)
    _poll(
        httpx_mock,
        _job("completed", processed_images=1, total_annotations_added=2, failed_images=0),
    )

    res = run("shelf", "--classes", "car:bbox", "--json")

    assert res.exit_code == ExitCode.OK
    report = json.loads(res.stdout)
    assert report["success"] is True
    assert report["job_id"] == "job-1" and report["job_status"] == "completed"
    assert report["images_attempted"] == 1 and report["annotations_added"] == 2


def test_a_folder_a_limit_and_a_threshold_reach_the_wire(run: Any, httpx_mock: HTTPXMock) -> None:
    _listing(httpx_mock, [_image(f"i{i}", f"{i}.jpg") for i in range(5)])
    _batch(httpx_mock, body=_job("pending"))

    res = run(
        "shelf",
        "--classes",
        "car",
        "--directory",
        "/val",
        "--limit",
        "2",
        "--confidence",
        "0.3",
        "--no-wait",
    )

    assert res.exit_code == ExitCode.OK, res.output
    (listing,) = [r for r in httpx_mock.get_requests() if r.method == "GET"]
    assert listing.url.params["directory"] == "/val"
    assert listing.url.params["dataset"] == "shelf"
    body = _batch_body(httpx_mock)
    assert body["images"] == ["i0", "i1"]
    assert body["classes"] == [{"name": "car", "output_type": "polygon"}]
    assert body["confidence_threshold"] == 0.3
    assert "Submitted 2 image(s) of shelf as job job-1 (pending)." in res.stdout
    assert "keeps running: pictograph auto-annotate get job-1" in res.stderr


def test_a_trained_model_drafts_without_classes(run: Any, httpx_mock: HTTPXMock) -> None:
    _listing(httpx_mock, [_image("i1", "a.jpg")])
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/models/Shelf%20Classifier",
        json={
            "data": {
                "id": MODEL_ID,
                "organization_id": "org",
                "name": "Shelf Classifier",
                "model_type": "classification",
                "status": "ready",
                "visibility": "private",
                "created_at": NOW,
                "updated_at": NOW,
            }
        },
    )
    _batch(httpx_mock, body=_job("pending"))

    res = run("shelf", "--model", "Shelf Classifier", "--no-wait")

    assert res.exit_code == ExitCode.OK, res.output
    body = _batch_body(httpx_mock)
    assert body["model"] == MODEL_ID and body["classes"] == []


@pytest.mark.parametrize(
    ("args", "why"),
    [
        ((), "name what SAM3 should find"),
        (("--model", "m", "--sahi"), "--sahi is for SAM3"),
        (("--classes", "car", "--confidence", "1.5"), "--confidence"),
        (("--classes", "car", "--limit", "0"), "--limit"),
    ],
)
def test_a_wrong_call_is_refused_before_any_request(
    run: Any, httpx_mock: HTTPXMock, args: tuple[str, ...], why: str
) -> None:
    res = run("shelf", *args)
    assert res.exit_code == ExitCode.INVALID
    assert why in " ".join(res.output.replace("│", " ").split())
    assert httpx_mock.get_requests() == []


def test_nothing_to_draft_is_not_a_failure(run: Any, httpx_mock: HTTPXMock) -> None:
    _listing(httpx_mock, [_image("i1", "a.jpg", annotations=2)])

    res = run("shelf", "--classes", "car", "--json")

    assert res.exit_code == ExitCode.OK
    report = json.loads(res.stdout)
    assert report["images_attempted"] == 0 and report["images_skipped"] == 1
    assert not [r for r in httpx_mock.get_requests() if r.method == "POST"]


def test_out_of_credit_exits_7_with_the_report_and_the_error(
    run: Any, httpx_mock: HTTPXMock
) -> None:
    _listing(httpx_mock, [_image("i1", "a.jpg")])
    _batch(
        httpx_mock,
        status_code=402,
        body={
            "error": {
                "code": "insufficient_credits",
                "message": "Not enough compute credit for this job.",
                "details": {"required_micro_usd": 90000, "remaining_micro_usd": 0},
            }
        },
    )

    res = run("shelf", "--classes", "car", "--json")

    assert res.exit_code == ExitCode.PAYMENT
    report = json.loads(res.stdout)
    assert report["success"] is False
    (failure,) = report["failures"]
    assert failure["error_type"] == "PaymentRequiredError"
    error = _last_json_line(res.stderr)
    assert error["code"] == "insufficient_credits" and error["exit_code"] == 7
    assert "Not enough compute credit" in error["message"]


def test_a_job_that_ends_failed_exits_1_with_job_failed(run: Any, httpx_mock: HTTPXMock) -> None:
    _listing(httpx_mock, [_image("i1", "a.jpg")])
    _batch(httpx_mock)
    _poll(httpx_mock, {**_job("failed"), "error_message": "SAM3 is unavailable"})

    res = run("shelf", "--classes", "car", env={"PICTOGRAPH_OUTPUT": "json"})

    assert res.exit_code == ExitCode.FAILED
    error = _last_json_line(res.stderr)
    assert error["code"] == "job_failed"
    assert "SAM3 is unavailable" in error["message"]
    assert error["details"]["job_id"] == "job-1", "the failed job is named, to look at"


def test_an_unknown_dataset_is_not_found(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=IMAGES,
        status_code=404,
        json={"error": {"code": "not_found", "message": "Dataset 'nope' not found"}},
    )
    res = run("nope", "--classes", "car")
    assert res.exit_code == ExitCode.NOT_FOUND
    assert "Dataset 'nope' not found" in res.stderr


def test_a_folder_with_no_images_is_refused_not_an_empty_success(
    run: Any, httpx_mock: HTTPXMock
) -> None:
    """A typo in --directory lists nothing: drafting nothing must not read as done."""
    _listing(httpx_mock, [])
    res = run("shelf", "--classes", "car", "--directory", "/vall", "--json")
    assert res.exit_code == ExitCode.INVALID
    error = _last_json_line(res.stderr)
    assert error["code"] == "no_images" and "/vall" in error["message"]
    assert res.stdout == ""
