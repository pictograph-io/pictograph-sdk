"""``models predict`` / ``deployments predict`` on a video's frames and a dataset's images.

Until 1.69.102 the two commands took images only: a video had to be cut into files
first, and a dataset's images downloaded one by one. B533 V2b adds ``--video`` (with
``--sample-fps``) and ``--dataset`` (with ``--directory``), both read-only, both
printed as JSON Lines.

These drive the REAL client over a mocked HTTP transport and REAL video files written
with OpenCV, so each test covers the command, the SDK calls and the bytes on the wire.
The frame-sampling rule itself (``pictograph._video``) is held to the workflow runner by
Pictograph's platform tests, which run both on the same video files.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pictograph.cli._app import app
from pictograph.cli._errors import ExitCode

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

# Video decoding is the `[inference]` extra's OpenCV, installed by CI's SDK job;
# without it these tests cannot write or read a video at all.
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

BASE = "https://api.test.local"
KEY = "pk_live_test"
MODEL_ID = "abcdef01-2345-6789-abcd-ef0123456789"
VIDEO_URL = "https://vid.test/clips/dock%20cam.avi"
ENDPOINT = "https://infer.test/dep-1/predict"
DATASET = "Shelf Photos"
DATASET_PATH = "Shelf%20Photos"


def _plain(output: str) -> str:
    """An error panel's text on one line (the box wraps long messages)."""
    return " ".join(output.replace("│", " ").split())


def _value(index: int) -> int:
    """The flat grey level of frame ``index``: far enough apart to survive MJPEG."""
    return (index * 11 + 20) % 250


def _write_video(path: Path, frames: int, fps: float) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    assert writer.isOpened(), "OpenCV could not open an MJPG writer"
    for index in range(frames):
        writer.write(np.full((48, 64, 3), _value(index), dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", KEY)
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    def invoke(*args: str, stdin: bytes | None = None) -> Any:
        return runner.invoke(app, list(args), input=stdin)

    return invoke


class _Result:
    def __init__(self, label: str) -> None:
        self.label = label

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {"model_type": "object_detection", "predictions": [{"name": self.label}]}


def _local_model() -> MagicMock:
    local = MagicMock()
    local.predict.side_effect = lambda image, **_: _Result(type(image).__name__)
    return local


def _lines(output: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def _model(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/models/Shelf%20Detector",
        json={
            "data": {
                "id": MODEL_ID,
                "organization_id": "org",
                "name": "Shelf Detector",
                "model_type": "object_detection",
                "status": "ready",
                "visibility": "private",
                "created_at": "2026-04-01T00:00:00Z",
                "updated_at": "2026-04-01T00:00:00Z",
            }
        },
    )


def _hosted(
    httpx_mock: HTTPXMock, *, status: int = 200, headers: dict[str, str] | None = None
) -> None:
    body: dict[str, Any] = {
        "data": {
            "success": True,
            "annotations": [{"name": "box", "type": "bbox", "confidence": 0.9}],
            "tags": [],
            "model_type": "object_detection",
            "inference_seconds": 0.1,
        }
    }
    if status != 200:
        body = {"detail": "Test inference is at daily capacity - please retry tomorrow."}
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/models/{MODEL_ID}/predict?confidence_threshold=0.5&top_k=3",
        status_code=status,
        json=body,
        headers=headers,
    )


def _uploads(httpx_mock: HTTPXMock) -> list[bytes]:
    return [r.content for r in httpx_mock.get_requests() if r.url.path.endswith("/predict")]


# ───────────────────────── --video ─────────────────────────


def test_video_runs_the_sampled_frames_with_their_index_and_time(run: Any, tmp_path: Path) -> None:
    _write_video(tmp_path / "clip.avi", frames=20, fps=10)
    local = _local_model()
    with patch("pictograph.resources.models.Models.load", return_value=local) as load:
        res = run("models", "predict", "Shelf Detector", "--video", "clip.avi", "--sample-fps", "2")
    assert res.exit_code == 0, res.output
    load.assert_called_once()
    lines = _lines(res.stdout)
    # 10 fps sampled at 2 fps: every 5th frame - the workflow runner's stride.
    assert [(line["input"], line["frame"], line["t"]) for line in lines] == [
        ("clip.avi", 0, 0.0),
        ("clip.avi", 5, 0.5),
        ("clip.avi", 10, 1.0),
        ("clip.avi", 15, 1.5),
    ]
    # A local model gets the decoded frame itself (BGR array), not a re-encoded file,
    # and it is the RIGHT frame: each one's grey level is its index's.
    frames = [call.args[0] for call in local.predict.call_args_list]
    assert all(isinstance(frame, np.ndarray) and frame.shape == (48, 64, 3) for frame in frames)
    assert [round(float(frame.mean())) for frame in frames] == pytest.approx(
        [_value(i) for i in (0, 5, 10, 15)], abs=3
    )
    assert all(line["result"]["predictions"] == [{"name": "ndarray"}] for line in lines)


@pytest.mark.parametrize(
    ("sample_fps", "frames"),
    [
        # 10 / 6 = 1.67: the stride ROUNDS to 2, as the runner's does (truncating
        # would run every frame).
        ("6", [0, 2, 4, 6, 8]),
        # 10 / 4 = 2.5: Python's round-half-to-even gives 2, the runner's too.
        ("4", [0, 2, 4, 6, 8]),
        # Faster than the video itself: every frame, never a duplicate.
        ("25", list(range(10))),
        # Slower than the clip is long: the first frame only.
        ("0.05", [0]),
    ],
)
def test_the_stride_rounds_like_the_runners(
    run: Any, tmp_path: Path, sample_fps: str, frames: list[int]
) -> None:
    _write_video(tmp_path / "clip.avi", frames=10, fps=10)
    with patch("pictograph.resources.models.Models.load", return_value=_local_model()):
        res = run("models", "predict", "m", "--video", "clip.avi", "--sample-fps", sample_fps)
    assert res.exit_code == 0, res.output
    assert [line["frame"] for line in _lines(res.stdout)] == frames


def test_video_defaults_to_one_frame_per_second(run: Any, tmp_path: Path) -> None:
    _write_video(tmp_path / "clip.avi", frames=25, fps=10)
    with patch("pictograph.resources.models.Models.load", return_value=_local_model()):
        res = run("models", "predict", "Shelf Detector", "--video", "clip.avi")
    assert res.exit_code == 0, res.output
    assert [line["frame"] for line in _lines(res.stdout)] == [0, 10, 20]


def test_limit_stops_after_that_many_frames(run: Any, tmp_path: Path) -> None:
    _write_video(tmp_path / "clip.avi", frames=30, fps=10)
    local = _local_model()
    with patch("pictograph.resources.models.Models.load", return_value=local):
        res = run("models", "predict", "Shelf Detector", "--video", "clip.avi",
                  "--sample-fps", "10", "--limit", "2")  # fmt: skip
    assert res.exit_code == 0, res.output
    assert [line["frame"] for line in _lines(res.stdout)] == [0, 1]
    assert local.predict.call_count == 2


def test_several_videos_run_in_order(run: Any, tmp_path: Path) -> None:
    _write_video(tmp_path / "a.avi", frames=10, fps=10)
    _write_video(tmp_path / "b.avi", frames=20, fps=10)
    with patch("pictograph.resources.models.Models.load", return_value=_local_model()):
        res = run("models", "predict", "Shelf Detector", "--video", "a.avi", "--video", "b.avi")
    assert res.exit_code == 0, res.output
    assert [(line["input"], line["frame"]) for line in _lines(res.stdout)] == [
        ("a.avi", 0),
        ("b.avi", 0),
        ("b.avi", 10),
    ]


def test_video_frames_reach_a_hosted_model_as_jpeg(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    _write_video(tmp_path / "clip.avi", frames=20, fps=10)
    _model(httpx_mock)
    for _ in range(2):
        _hosted(httpx_mock)
    res = run("models", "predict", "Shelf Detector", "--video", "clip.avi", "--remote")
    assert res.exit_code == 0, res.output
    assert [line["frame"] for line in _lines(res.stdout)] == [0, 10]
    uploads = _uploads(httpx_mock)
    assert len(uploads) == 2
    # An uploaded image, not a numpy dump: each body carries a JPEG.
    assert all(b"\xff\xd8\xff" in body for body in uploads)


def test_a_video_on_stdin_and_at_a_url(run: Any, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    data = _write_video(tmp_path / "clip.avi", frames=10, fps=10).read_bytes()
    httpx_mock.add_response(url=VIDEO_URL, content=data)
    with patch("pictograph.resources.models.Models.load", return_value=_local_model()):
        res = run("models", "predict", "Shelf Detector", "--video", "-", "--video", VIDEO_URL,
                  stdin=data)  # fmt: skip
    assert res.exit_code == 0, res.output
    assert [(line["input"], line["frame"]) for line in _lines(res.stdout)] == [
        ("-", 0),
        (VIDEO_URL, 0),
    ]


def test_a_video_url_that_fails_is_that_videos_line_and_the_next_still_runs(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    _write_video(tmp_path / "b.avi", frames=10, fps=10)
    httpx_mock.add_response(url=VIDEO_URL, status_code=404)
    with patch("pictograph.resources.models.Models.load", return_value=_local_model()):
        res = run("models", "predict", "Shelf Detector", "--video", VIDEO_URL, "--video", "b.avi")
    assert res.exit_code == 1, "a run with a failure exits non-zero"
    first, second = _lines(res.stdout)
    assert first["input"] == VIDEO_URL and "HTTP 404" in first["error"]
    assert first["error_type"] == "ValueError"
    assert second == {"input": "b.avi", "frame": 0, "t": 0.0, "result": second["result"]}


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda tmp: tmp.joinpath("bad.avi").write_bytes(b"not a video " * 64),
         "Could not open bad.avi as a video"),
        (lambda _tmp: None, "No such file: bad.avi"),
    ],
)  # fmt: skip
def test_a_video_that_cannot_be_read_is_refused_before_the_model_loads(
    run: Any, tmp_path: Path, make: Any, message: str
) -> None:
    make(tmp_path)
    with patch("pictograph.resources.models.Models.load") as load:
        res = run("models", "predict", "Shelf Detector", "--video", "bad.avi")
    assert res.exit_code == ExitCode.INVALID
    assert message in _plain(res.output)
    load.assert_not_called()


def test_a_video_needs_opencv_and_says_how_to_get_it(
    run: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_video(tmp_path / "clip.avi", frames=5, fps=10)
    monkeypatch.setitem(__import__("sys").modules, "cv2", None)  # `import cv2` -> ImportError
    res = run("models", "predict", "Shelf Detector", "--video", "clip.avi", "--remote")
    assert res.exit_code == ExitCode.INVALID
    assert "pip install 'pictograph[inference]'" in _plain(res.output)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["a.jpg", "--video", "clip.avi"], "one kind per run"),
        (["--video", "clip.avi", "--dataset", DATASET], "one kind per run"),
        (["--directory", "/val", "a.jpg"], "--directory needs --dataset"),
        (["--sample-fps", "2", "a.jpg"], "--sample-fps needs --video"),
        ([], "Give at least one image (or --video / --dataset)."),
    ],
)
def test_one_kind_of_input_per_run(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path, args: list[str], message: str
) -> None:
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8 jpeg")
    _write_video(tmp_path / "clip.avi", frames=5, fps=10)
    res = run("models", "predict", "Shelf Detector", *args, "--remote")
    assert res.exit_code == ExitCode.INVALID
    assert message in _plain(res.output)
    assert httpx_mock.get_requests() == []


# ───────────────────────── --dataset ─────────────────────────


def _row(image_id: str, filename: str, directory: str = "/") -> dict[str, Any]:
    return {
        "id": image_id,
        "filename": filename,
        "directory_path": directory,
        "status": "new",
        "annotation_count": 0,
        "file_size": 10,
        "created_at": "2026-04-01T00:00:00Z",
    }


ROWS = [
    _row("11111111-1111-1111-1111-111111111111", "a.jpg"),
    _row("22222222-2222-2222-2222-222222222222", "b 2.jpg", "/val"),
]


def _storage(httpx_mock: HTTPXMock, storage_class: str = "standard", state: str = "idle") -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/{DATASET_PATH}/storage",
        json={"data": {"storage_class": storage_class, "storage_state": state}},
    )


def _listing(
    httpx_mock: HTTPXMock,
    rows: list[dict[str, Any]],
    *,
    limit: int = 200,
    directory: str | None = None,
) -> None:
    query = f"dataset={DATASET_PATH}&limit={limit}&offset=0"
    if directory is not None:
        query += f"&directory_path={directory.replace('/', '%2F')}"
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/images/?{query}",
        json={"data": rows, "pagination": {"has_more": False}},
    )


def _image_bytes(httpx_mock: HTTPXMock, address: str, body: bytes, status: int = 200) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/images/{DATASET_PATH}/{address.replace(' ', '%20')}",
        content=body,
        status_code=status,
    )


def test_dataset_runs_every_image_and_writes_nothing(run: Any, httpx_mock: HTTPXMock) -> None:
    _storage(httpx_mock)
    _listing(httpx_mock, ROWS)
    _image_bytes(httpx_mock, "a.jpg", b"bytes of a")
    _image_bytes(httpx_mock, "val/b 2.jpg", b"bytes of b")
    local = _local_model()
    with patch("pictograph.resources.models.Models.load", return_value=local):
        res = run("models", "predict", "Shelf Detector", "--dataset", DATASET)
    assert res.exit_code == 0, res.output
    lines = _lines(res.stdout)
    # Each line names the image by its address inside the dataset, plus its id.
    assert [(line["input"], line["image_id"]) for line in lines] == [
        ("a.jpg", ROWS[0]["id"]),
        ("val/b 2.jpg", ROWS[1]["id"]),
    ]
    assert [call.args[0] for call in local.predict.call_args_list] == [b"bytes of a", b"bytes of b"]
    # Read-only: nothing but reads ever reached the API.
    assert {request.method for request in httpx_mock.get_requests()} == {"GET"}


def test_dataset_directory_is_one_folder_in_the_apis_spelling(
    run: Any, httpx_mock: HTTPXMock
) -> None:
    _storage(httpx_mock)
    _listing(httpx_mock, [ROWS[1]], directory="/val")
    _image_bytes(httpx_mock, "val/b 2.jpg", b"bytes of b")
    with patch("pictograph.resources.models.Models.load", return_value=_local_model()):
        res = run(
            "models", "predict", "Shelf Detector", "--dataset", DATASET, "--directory", "val/"
        )
    assert res.exit_code == 0, res.output
    assert [line["input"] for line in _lines(res.stdout)] == ["val/b 2.jpg"]


def test_dataset_limit_asks_the_api_for_only_that_many(run: Any, httpx_mock: HTTPXMock) -> None:
    _storage(httpx_mock)
    _listing(httpx_mock, [ROWS[0]], limit=1)
    _image_bytes(httpx_mock, "a.jpg", b"bytes of a")
    with patch("pictograph.resources.models.Models.load", return_value=_local_model()):
        res = run("models", "predict", "Shelf Detector", "--dataset", DATASET, "--limit", "1")
    assert res.exit_code == 0, res.output
    assert len(_lines(res.stdout)) == 1


@pytest.mark.parametrize(
    ("storage_class", "state", "message"),
    [
        ("coldline", "idle", "is in cold storage"),
        ("coldline", "restoring", "is being restored"),
        ("standard", "freezing", "is moving to cold storage"),
    ],
)
def test_a_cold_dataset_is_refused_before_any_image_is_read(
    run: Any, httpx_mock: HTTPXMock, storage_class: str, state: str, message: str
) -> None:
    _storage(httpx_mock, storage_class, state)
    with patch("pictograph.resources.models.Models.load") as load:
        res = run("models", "predict", "Shelf Detector", "--dataset", DATASET)
    assert res.exit_code == ExitCode.INVALID
    text = _plain(res.output)
    assert message in text
    assert "pictograph datasets" in text
    load.assert_not_called()
    # Only the storage check: no listing, no image bytes.
    assert [r.url.path for r in httpx_mock.get_requests()] == [
        f"/api/v1/developer/datasets/{DATASET}/storage"
    ]


def test_an_empty_folder_is_refused_with_its_name(run: Any, httpx_mock: HTTPXMock) -> None:
    _storage(httpx_mock)
    _listing(httpx_mock, [], directory="/vall")
    with patch("pictograph.resources.models.Models.load") as load:
        res = run(
            "models", "predict", "Shelf Detector", "--dataset", DATASET, "--directory", "vall"
        )
    assert res.exit_code == ExitCode.INVALID
    assert "has no images in /vall (only that folder, not its subfolders)" in _plain(res.output)
    load.assert_not_called()


def test_one_unreadable_image_is_its_own_line(run: Any, httpx_mock: HTTPXMock) -> None:
    _storage(httpx_mock)
    _listing(httpx_mock, ROWS)
    _image_bytes(httpx_mock, "a.jpg", b"", status=404)
    _image_bytes(httpx_mock, "val/b 2.jpg", b"bytes of b")
    with patch("pictograph.resources.models.Models.load", return_value=_local_model()):
        res = run("models", "predict", "Shelf Detector", "--dataset", DATASET)
    assert res.exit_code == 1
    first, second = _lines(res.stdout)
    assert first["input"] == "a.jpg" and first["error_type"] == "NotFoundError"
    assert second["input"] == "val/b 2.jpg" and "result" in second


def test_hosted_capacity_spent_stops_the_run_instead_of_failing_every_image(
    run: Any, httpx_mock: HTTPXMock
) -> None:
    """A 429 with a day-long Retry-After would fail every remaining image the same way."""
    _storage(httpx_mock)
    _listing(httpx_mock, ROWS)
    _image_bytes(httpx_mock, "a.jpg", b"bytes of a")
    _model(httpx_mock)
    _hosted(httpx_mock, status=429, headers={"Retry-After": "3600"})
    res = run("models", "predict", "Shelf Detector", "--dataset", DATASET, "--remote")
    assert res.exit_code == ExitCode.RATE_LIMITED
    (line,) = _lines(res.stdout)
    assert line["input"] == "a.jpg" and line["error_type"] == "RateLimitError"
    assert line["error_code"] == "rate_limited"
    note = _plain(res.stderr)
    assert "Stopped after 1 input(s)" in note and "drop --remote" in note
    # The second image was never even downloaded.
    assert not any(r.url.path.endswith("b%202.jpg") for r in httpx_mock.get_requests())


# ───────────────────────── deployments predict ─────────────────────────


def _endpoint(httpx_mock: HTTPXMock, times: int) -> None:
    for _ in range(times):
        httpx_mock.add_response(
            method="POST",
            url=ENDPOINT,
            json={"model_type": "object_detection", "predictions": [{"name": "box"}]},
        )


def test_deployment_predict_reads_a_dataset(run: Any, httpx_mock: HTTPXMock) -> None:
    _storage(httpx_mock)
    _listing(httpx_mock, ROWS)
    _image_bytes(httpx_mock, "a.jpg", b"bytes of a")
    _image_bytes(httpx_mock, "val/b 2.jpg", b"bytes of b")
    _endpoint(httpx_mock, times=2)
    res = run("deployments", "predict", "dep", "--dataset", DATASET, "--token", "pk_deploy_t",
              "--endpoint", ENDPOINT)  # fmt: skip
    assert res.exit_code == 0, res.output
    assert [line["input"] for line in _lines(res.stdout)] == ["a.jpg", "val/b 2.jpg"]
    posted = [json.loads(r.content) for r in httpx_mock.get_requests() if r.url == ENDPOINT]
    assert [base64.b64decode(body["image"]["value"]) for body in posted] == [
        b"bytes of a",
        b"bytes of b",
    ]


def test_deployment_predict_on_a_video_needs_only_the_token(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PICTOGRAPH_API_KEY")  # the deployment token alone authorizes it
    _write_video(tmp_path / "clip.avi", frames=20, fps=10)
    _endpoint(httpx_mock, times=2)
    res = run("deployments", "predict", "dep", "--video", "clip.avi", "--token", "pk_deploy_t",
              "--endpoint", ENDPOINT)  # fmt: skip
    assert res.exit_code == 0, res.output
    assert [line["frame"] for line in _lines(res.stdout)] == [0, 10]
    for request in httpx_mock.get_requests():
        assert request.headers["authorization"] == "Bearer pk_deploy_t"
        frame = base64.b64decode(json.loads(request.content)["image"]["value"])
        assert frame.startswith(b"\xff\xd8\xff"), "a frame travels as a JPEG"
