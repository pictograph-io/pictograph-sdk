"""What ``pictograph workflows run`` runs on (B533 V4, B606).

A workflow's Source block holds no input: each run names its own. ``--file`` takes a file on disk, an ``http(s)``
link Pictograph downloads, or ``-`` for a video or image piped in; ``--video`` / ``--image`` take a stored file's
handle under your organization or a link; ``--dataset`` (with ``--directory``) and ``--export`` name images already
there; ``--sample-fps`` is a video's rate. ``invoke`` is ``run``'s old name. Until 1.69.108 ``--file`` read a link as a
local path (and Typer's ``Path`` folded its ``//``), and nothing read stdin.

These drive the REAL client over a mocked HTTP transport, so each test covers the
command, the SDK call and the request on the wire.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from pictograph.cli._app import app
from pictograph.cli._errors import ExitCode

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
API = f"{BASE}/api/v1/developer/workflows"
WF = "fedcba98-1111-2222-3333-444455556666"  # a uuid: no name lookup
CREATED = {
    "success": True,
    "run_id": "run-9",
    "deposit_micro_usd": 1000,
    "status_url": "/api/v1/developer/workflows/runs/run-9",
}
PNG_HEAD = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x08\x00\x00\x00\x06"


def _plain(output: str) -> str:
    """An error panel's text on one line (the box wraps long messages)."""
    return " ".join(output.replace("│", " ").split())


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_test")
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    def invoke(*args: str, stdin: bytes | None = None, command: str = "run") -> Any:
        return runner.invoke(app, ["workflows", command, WF, *args], input=stdin)

    return invoke


def _run_body(httpx_mock: HTTPXMock) -> dict[str, Any]:
    (request,) = [r for r in httpx_mock.get_requests() if r.url.path.endswith("/run")]
    return json.loads(request.content)


def test_a_link_given_to_file_is_downloaded_by_pictograph(httpx_mock: HTTPXMock, run: Any) -> None:
    httpx_mock.add_response(method="POST", url=f"{API}/{WF}/run", json=CREATED)
    res = run("--file", "https://media.test/clips/dock%20cam.mp4", "--no-wait")
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["run_id"] == "run-9"
    assert _run_body(httpx_mock) == {
        "source": {"kind": "video", "url": "https://media.test/clips/dock%20cam.mp4"}
    }
    assert len(httpx_mock.get_requests()) == 1, "a link is never uploaded from here"


def test_a_link_given_to_video_carries_the_frame_rate(httpx_mock: HTTPXMock, run: Any) -> None:
    httpx_mock.add_response(method="POST", url=f"{API}/{WF}/run", json=CREATED)
    res = run("--video", "https://media.test/live", "--sample-fps", "2", "--no-wait")
    assert res.exit_code == 0, res.output
    assert _run_body(httpx_mock) == {
        "source": {"kind": "video", "url": "https://media.test/live", "sample_fps": 2.0}
    }


def test_a_storage_uri_given_to_image_is_used_in_place(httpx_mock: HTTPXMock, run: Any) -> None:
    httpx_mock.add_response(method="POST", url=f"{API}/{WF}/run", json=CREATED)
    res = run("--image", "gs://bucket/org/workflows/uploads/a.png", "--no-wait")
    assert res.exit_code == 0, res.output
    assert _run_body(httpx_mock) == {
        "source": {"kind": "image", "image": "gs://bucket/org/workflows/uploads/a.png"}
    }


def test_a_local_path_given_to_video_points_at_file(httpx_mock: HTTPXMock, run: Any) -> None:
    res = run("--video", "clip.mp4", "--no-wait")
    assert res.exit_code == ExitCode.INVALID
    assert "neither the handle an upload returned nor an http(s) link" in _plain(res.output)
    assert "use --file" in _plain(res.output)
    assert httpx_mock.get_requests() == []


def test_stdin_is_uploaded_then_run(httpx_mock: HTTPXMock, run: Any) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/upload-url",
        json={
            "success": True,
            "upload_url": "https://signed.test/put",
            "gcs_uri": "gs://b/o/u.png",
        },
    )
    httpx_mock.add_response(method="PUT", url="https://signed.test/put", status_code=200)
    httpx_mock.add_response(method="POST", url=f"{API}/{WF}/run", json=CREATED)
    res = run("--file", "-", "--no-wait", stdin=PNG_HEAD)
    assert res.exit_code == 0, res.output
    upload, put, _invoke = httpx_mock.get_requests()
    assert json.loads(upload.content) == {"filename": "upload.png", "content_type": "image/png"}
    assert put.content == PNG_HEAD
    assert _run_body(httpx_mock) == {"source": {"kind": "image", "image": "gs://b/o/u.png"}}


def test_empty_stdin_says_how_to_pipe_into_this_command(httpx_mock: HTTPXMock, run: Any) -> None:
    res = run("--file", "-", "--no-wait", stdin=b"")
    assert res.exit_code == ExitCode.INVALID
    text = _plain(res.output)
    assert "nothing arrived on stdin" in text and "workflows run" in text
    assert httpx_mock.get_requests() == []


def test_stdin_that_is_not_a_video_or_an_image_is_refused(httpx_mock: HTTPXMock, run: Any) -> None:
    res = run("--file", "-", "--no-wait", stdin=b"just some text")
    assert res.exit_code == ExitCode.INVALID
    assert "not a video or an image" in _plain(res.output)
    assert httpx_mock.get_requests() == []


def test_a_link_without_a_kind_asks_for_one(httpx_mock: HTTPXMock, run: Any) -> None:
    res = run("--file", "https://media.test/stream?id=4", "--no-wait")
    assert res.exit_code == ExitCode.INVALID
    assert "Say which" in _plain(res.output)
    assert httpx_mock.get_requests() == []


def test_a_file_on_disk_is_still_uploaded(httpx_mock: HTTPXMock, run: Any, tmp_path: Path) -> None:
    (tmp_path / "clip.mp4").write_bytes(b"\x00\x00\x00\x1cftypisom")
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/upload-url",
        json={
            "success": True,
            "upload_url": "https://signed.test/put",
            "gcs_uri": "gs://b/o/c.mp4",
        },
    )
    httpx_mock.add_response(method="PUT", url="https://signed.test/put", status_code=200)
    httpx_mock.add_response(method="POST", url=f"{API}/{WF}/run", json=CREATED)
    res = run("--file", "clip.mp4", "--sample-fps", "10", "--no-wait")
    assert res.exit_code == 0, res.output
    upload = httpx_mock.get_requests()[0]
    assert json.loads(upload.content) == {"filename": "clip.mp4", "content_type": "video/mp4"}
    assert _run_body(httpx_mock) == {
        "source": {"kind": "video", "video": "gs://b/o/c.mp4", "sample_fps": 10.0}
    }


def test_a_dataset_and_its_folder_and_an_export(httpx_mock: HTTPXMock, run: Any) -> None:
    httpx_mock.add_response(method="POST", url=f"{API}/{WF}/run", json=CREATED)
    res = run("--dataset", "Street scenes", "--directory", "/night", "--no-wait")
    assert res.exit_code == 0, res.output
    assert _run_body(httpx_mock) == {
        "source": {"kind": "dataset", "dataset": "Street scenes", "directory": "/night"}
    }
    httpx_mock.reset()
    httpx_mock.add_response(method="POST", url=f"{API}/{WF}/run", json=CREATED)
    res = run("--export", "street-v1", "--no-wait")
    assert res.exit_code == 0, res.output
    assert _run_body(httpx_mock) == {"source": {"kind": "export", "export": "street-v1"}}


@pytest.mark.parametrize(
    ("args", "says"),
    [
        ((), "Name what it runs on"),
        (("--dataset", "d", "--export", "e"), "Name what it runs on"),
        (("--export", "e", "--directory", "/a"), "--directory is a folder of --dataset"),
        (("--dataset", "d", "--duration", "9"), "--duration goes with --video"),
        (
            ("--dataset", "d", "--sample-fps", "5"),
            "sample_fps is a video's: this source is a dataset",
        ),
    ],
    ids=["none", "two", "folder-without-dataset", "duration-without-video", "rate-for-a-dataset"],
)
def test_what_cannot_run_is_refused_before_any_request(
    httpx_mock: HTTPXMock, run: Any, args: tuple[str, ...], says: str
) -> None:
    res = run(*args, "--no-wait")
    assert res.exit_code == ExitCode.INVALID
    assert says in _plain(res.output)
    assert httpx_mock.get_requests() == []


def test_invoke_is_runs_old_name_kept_out_of_the_help(httpx_mock: HTTPXMock, run: Any) -> None:
    httpx_mock.add_response(method="POST", url=f"{API}/{WF}/run", json=CREATED)
    res = run("--export", "e", "--no-wait", command="invoke")
    assert res.exit_code == 0, res.output
    assert _run_body(httpx_mock) == {"source": {"kind": "export", "export": "e"}}
    listing = CliRunner().invoke(app, ["workflows", "--help"]).output
    assert " run " in listing and " invoke " not in listing


# ---------------------------------------------------------------- B607: --outputs and --write-video


def test_outputs_go_with_the_run_and_write_video_is_its_shortcut(
    httpx_mock: HTTPXMock, run: Any, tmp_path: Path
) -> None:
    """B607: a run's outputs are given at the call - --outputs as JSON, or @a file of it - and --write-video sets
    annotated_video."""
    httpx_mock.add_response(method="POST", url=f"{API}/{WF}/run", json=CREATED)
    given = {"annotations": {"dataset": "Street scenes", "directory": "/drafts"}}
    res = run("--dataset", "d", "--outputs", json.dumps(given), "--write-video", "--no-wait")
    assert res.exit_code == 0, res.output
    assert _run_body(httpx_mock)["outputs"] == {
        "annotated_video": True,
        "annotations": {
            "dataset": "Street scenes",
            "directory": "/drafts",
            "new_folder_each_run": False,
            "only_with_detections": True,
        },
    }
    httpx_mock.reset()
    httpx_mock.add_response(method="POST", url=f"{API}/{WF}/run", json=CREATED)
    path = tmp_path / "outputs.json"
    path.write_text(json.dumps({"webhook": {"endpoint": "wh-1"}}))
    res = run("--dataset", "d", "--outputs", f"@{path}", "--no-wait")
    assert res.exit_code == 0, res.output
    assert _run_body(httpx_mock)["outputs"] == {
        "annotated_video": False,
        "webhook": {"endpoint": "wh-1", "events": []},
    }


@pytest.mark.parametrize(
    ("given", "says"),
    [
        ("not json", "not JSON"),
        ("[1, 2]", "a JSON object"),
        ('{"annotation": {}}', "annotation"),
        ("@/no/such/outputs.json", "could not read"),
    ],
)
def test_outputs_a_run_cannot_take_are_a_usage_error_before_any_request(
    httpx_mock: HTTPXMock, run: Any, given: str, says: str
) -> None:
    res = run("--dataset", "d", "--outputs", given, "--no-wait")
    assert res.exit_code == ExitCode.INVALID, res.output
    assert says in _plain(res.output)
    assert httpx_mock.get_requests() == []
