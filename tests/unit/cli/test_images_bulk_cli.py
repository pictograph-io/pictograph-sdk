"""Bulk image changes from the CLI: ``images move / copy / set-status / archive / restore``.

Before them an agent could not move images into a folder, mark a reviewed set
``complete``, or archive and restore in bulk without the app (B533 P1 #6). These drive
the REAL app and the REAL client over a mocked HTTP transport and read every request.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from pictograph.cli._app import app
from pictograph.cli._errors import ExitCode

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
BATCH = f"{BASE}/api/v1/developer/batch/images"
LISTING = re.compile(rf"{re.escape(BASE)}/api/v1/developer/images/\?.*")
NOW = "2026-04-01T00:00:00Z"
A, B, C = (str(uuid.UUID(int=i)) for i in (1, 2, 3))


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_test")
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    monkeypatch.setenv("PICTOGRAPH_MAX_RETRIES", "0")
    monkeypatch.delenv("PICTOGRAPH_OUTPUT", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    def invoke(*args: str, stdin: str | None = None) -> Any:
        return runner.invoke(app, ["images", *args], input=stdin)

    return invoke


def _result(processed: int, **extra: Any) -> dict[str, Any]:
    return {"success": True, "processed": processed, "failed": [], **extra}


def _image(image_id: str, archived: bool = False) -> dict[str, Any]:
    return {
        "id": image_id,
        "filename": f"{image_id[-4:]}.jpg",
        "annotation_count": 0,
        "is_archived": archived,
        "created_at": NOW,
    }


def _posts(httpx_mock: HTTPXMock, method: str = "POST") -> list[dict[str, Any]]:
    return [json.loads(r.content) for r in httpx_mock.get_requests() if r.method == method]


def test_move_sends_the_ids_and_the_folder(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BATCH}/move",
        json=_result(2, renamed=1, affected_directories=["/", "/val"]),
    )
    res = run("move", "shelf", "--to", "/val", "--images", f"{A},{B}")
    assert res.exit_code == ExitCode.OK, res.output
    (body,) = _posts(httpx_mock)
    assert body == {"dataset_name": "shelf", "image_ids": [A, B], "target_directory_path": "/val"}
    report = json.loads(res.stdout)
    assert report["processed"] == 2 and report["renamed"] == 1 and report["selected"] == 2


def test_a_filter_selects_the_live_images_that_match(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=LISTING, json={"data": [_image(A), _image(B)]})
    httpx_mock.add_response(method="PATCH", url=f"{BATCH}/update", json=_result(2))

    res = run(
        "set-status", "shelf", "--to", "complete", "--directory", "/val", "--status", "review"
    )

    assert res.exit_code == ExitCode.OK, res.output
    (listing,) = [r for r in httpx_mock.get_requests() if r.method == "GET"]
    assert listing.url.params["directory_path"] == "/val"
    assert listing.url.params["status"] == "review"
    assert listing.url.params["limit"] == "1000"
    assert "include_archived" not in listing.url.params
    (body,) = _posts(httpx_mock, "PATCH")
    assert body == {"dataset_name": "shelf", "image_ids": [A, B], "updates": {"status": "complete"}}
    assert json.loads(res.stdout)["selected_by"] == "directory=/val, status=review"


def test_a_big_selection_runs_in_chunks_of_10000(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    ids = [str(uuid.UUID(int=i + 10)) for i in range(10_001)]
    (tmp_path / "ids.txt").write_text("# reviewed\n" + "\n".join(ids) + "\n\n")
    httpx_mock.add_response(method="POST", url=f"{BATCH}/copy", json=_result(10_000))
    httpx_mock.add_response(method="POST", url=f"{BATCH}/copy", json=_result(1))

    res = run("copy", "shelf", "--to", "/backup", "--from-file", "ids.txt", "--with-annotations")

    assert res.exit_code == ExitCode.OK, res.output
    first, second = _posts(httpx_mock)
    assert len(first["image_ids"]) == 10_000 and second["image_ids"] == ids[-1:]
    assert first["copy_annotations"] is True and first["duplicate_handling"] == "rename"
    assert json.loads(res.stdout)["processed"] == 10_001


def test_archive_asks_first_and_says_how_to_skip_the_question(
    run: Any, httpx_mock: HTTPXMock
) -> None:
    res = run("archive", "shelf", "--images", A, stdin="")
    assert res.exit_code == ExitCode.FAILED
    assert "--yes" in res.output
    assert httpx_mock.get_requests() == []


def test_archive_with_yes_soft_deletes(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{BATCH}/delete", json=_result(1, operation="archived")
    )
    res = run("archive", "shelf", "--images", A, "--yes")
    assert res.exit_code == ExitCode.OK, res.output
    (body,) = _posts(httpx_mock)
    assert body["permanent"] is False


def test_restore_selects_among_the_archived_images(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=LISTING, json={"data": [_image(A, archived=True), _image(B), _image(C, archived=True)]}
    )
    httpx_mock.add_response(method="PATCH", url=f"{BATCH}/update", json=_result(2))

    res = run("restore", "shelf", "--all")

    assert res.exit_code == ExitCode.OK, res.output
    (listing,) = [r for r in httpx_mock.get_requests() if r.method == "GET"]
    assert listing.url.params["include_archived"] == "true"
    (body,) = _posts(httpx_mock, "PATCH")
    assert body["image_ids"] == [A, C] and body["updates"] == {"is_archived": False}


def test_some_images_failing_exits_1_and_names_each(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BATCH}/move",
        json={"success": False, "processed": 1, "failed": [{"id": B, "reason": "not found"}]},
    )
    res = run("move", "shelf", "--to", "/val", "--images", f"{A},{B}")
    assert res.exit_code == ExitCode.FAILED
    report = json.loads(res.stdout)
    assert report["failed"] == [{"image_id": B, "reason": "not found"}]
    assert "1 of 2 image(s) were not moved" in res.stderr


@pytest.mark.parametrize(
    ("args", "why"),
    [
        (("move", "shelf", "--to", "/val"), "select the images"),
        (("move", "shelf", "--to", "/val", "--images", A, "--all"), "give one selection"),
        (("move", "shelf", "--to", "/val", "--images", "a.jpg"), "is not an image UUID"),
        (("set-status", "shelf", "--to", "approved", "--all"), "--to"),
        (("move", "shelf", "--to", "/x", "--status", "done"), "use one of new"),
        (
            ("copy", "shelf", "--to", "/x", "--all", "--duplicates", "keep"),
            "rename, skip or overwrite",
        ),
    ],
)
def test_a_wrong_selection_is_refused_before_any_request(
    run: Any, httpx_mock: HTTPXMock, args: tuple[str, ...], why: str
) -> None:
    res = run(*args)
    assert res.exit_code == ExitCode.INVALID
    assert why in " ".join(res.output.replace("│", " ").split())
    assert httpx_mock.get_requests() == []


def test_a_filter_that_matches_nothing_is_not_found(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=LISTING, json={"data": []})
    res = run("archive", "shelf", "--directory", "/vall", "--yes")
    assert res.exit_code == ExitCode.NOT_FOUND
    assert "No images of 'shelf' match directory=/vall" in res.output


def test_dry_run_counts_and_changes_nothing(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=LISTING, json={"data": [_image(A), _image(B)]})
    res = run("set-status", "shelf", "--to", "complete", "--split", "val", "--dry-run")
    assert res.exit_code == ExitCode.OK, res.output
    report = json.loads(res.stdout)
    assert report == {
        "operation": "set-status",
        "dataset": "shelf",
        "selected_by": "split=val",
        "selected": 2,
        "image_ids": [A, B],
        "dry_run": True,
    }
    assert [r.method for r in httpx_mock.get_requests()] == ["GET"]


def test_ids_from_stdin(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=f"{BATCH}/move", json=_result(2))
    res = run("move", "shelf", "--to", "/val", "--from-file", "-", stdin=f"{A}\n{B}\n{A}\n")
    assert res.exit_code == ExitCode.OK, res.output
    assert _posts(httpx_mock)[0]["image_ids"] == [A, B], "repeats dropped, order kept"


# ── copy --to-dataset (copy into another dataset) ──────────────────────────────


def _to_dataset(processed: int, **extra: Any) -> dict[str, Any]:
    return {
        "success": True,
        "processed": processed,
        "failed": [],
        "affected_directories": ["/"],
        "renamed": 0,
        "operation": "copied",
        "target_dataset": "curated",
        "classes_added": [],
        **extra,
    }


def test_copy_to_dataset_sends_the_route_body_and_reports_the_classes_added(
    run: Any, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BATCH}/copy-to-dataset",
        json=_to_dataset(
            2, renamed=1, affected_directories=["/val"], classes_added=["car", "night"]
        ),
    )
    res = run("copy", "shelf", "--to-dataset", "curated", "--to", "/val", "--images", f"{A},{B}")
    assert res.exit_code == ExitCode.OK, res.output
    (body,) = _posts(httpx_mock)
    assert body == {
        "dataset_name": "shelf",
        "image_ids": [A, B],
        "target_dataset_name": "curated",
        "target_directory_path": "/val",
    }
    report = json.loads(res.stdout)
    assert {
        k: report[k] for k in ("to_dataset", "to", "processed", "renamed", "classes_added")
    } == {
        "to_dataset": "curated",
        "to": "/val",
        "processed": 2,
        "renamed": 1,
        "classes_added": ["car", "night"],
    }


def test_copy_to_dataset_defaults_to_its_root_and_adds_up_the_chunks(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    ids = [str(uuid.UUID(int=i + 10)) for i in range(10_001)]
    (tmp_path / "ids.txt").write_text("\n".join(ids))
    url = f"{BATCH}/copy-to-dataset"
    httpx_mock.add_response(method="POST", url=url, json=_to_dataset(10_000, classes_added=["car"]))
    httpx_mock.add_response(
        method="POST", url=url, json=_to_dataset(1, renamed=1, classes_added=["car", "bus"])
    )
    res = run("copy", "shelf", "--to-dataset", "curated", "--from-file", "ids.txt")
    assert res.exit_code == ExitCode.OK, res.output
    first, second = _posts(httpx_mock)
    assert first["target_directory_path"] == "/" and len(first["image_ids"]) == 10_000
    assert second["image_ids"] == ids[-1:]
    report = json.loads(res.stdout)
    assert (report["processed"], report["renamed"], report["classes_added"]) == (
        10_001,
        1,
        ["car", "bus"],
    )


def test_copy_to_dataset_names_each_image_that_made_no_copy(
    run: Any, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BATCH}/copy-to-dataset",
        json=_to_dataset(1, failed=[{"id": B, "reason": "Not an image of dataset 'shelf'"}]),
    )
    res = run("copy", "shelf", "--to-dataset", "curated", "--images", f"{A},{B}")
    assert res.exit_code == ExitCode.FAILED
    assert json.loads(res.stdout)["failed"] == [
        {"image_id": B, "reason": "Not an image of dataset 'shelf'"}
    ]
    assert "1 of 2 image(s) were not copied" in res.stderr


def test_copy_to_dataset_dry_run_says_where_and_changes_nothing(
    run: Any, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=LISTING, json={"data": [_image(A), _image(B)]})
    res = run("copy", "shelf", "--to-dataset", "curated", "--all", "--dry-run")
    assert res.exit_code == ExitCode.OK, res.output
    report = json.loads(res.stdout)
    assert (report["to_dataset"], report["to"], report["selected"], report["dry_run"]) == (
        "curated",
        "/",
        2,
        True,
    )
    assert [r.method for r in httpx_mock.get_requests()] == ["GET"]


@pytest.mark.parametrize(
    ("args", "why"),
    [
        (("copy", "shelf", "--images", A), "or a dataset with --to-dataset"),
        (
            ("copy", "shelf", "--to-dataset", "curated", "--duplicates", "skip", "--images", A),
            "cannot skip or overwrite",
        ),
    ],
)
def test_copy_refusals_before_any_request(
    run: Any, httpx_mock: HTTPXMock, args: tuple[str, ...], why: str
) -> None:
    res = run(*args)
    assert res.exit_code == ExitCode.INVALID
    assert why in " ".join(res.output.replace("│", " ").split())
    assert httpx_mock.get_requests() == []
