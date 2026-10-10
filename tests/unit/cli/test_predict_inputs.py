"""What ``models predict`` / ``deployments predict`` accept as an image.

Until 1.69.99 the two commands took exactly one image, and ``models predict
--remote`` read it as a file path even though its help said "path or URL": a URL
raised ``FileNotFoundError``. The CLI is how an agent calls a model, and an agent's
image is as often a link, a pipe or a folder as it is one file.

These drive the REAL client over a mocked HTTP transport, so each test covers the
whole path - the command, the SDK call and the request on the wire - not a mock of
the SDK standing in for it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import typer
from typer.testing import CliRunner

from pictograph import AsyncClient, Client
from pictograph._image_input import read_image, read_image_async
from pictograph.cli import _inputs
from pictograph.cli._app import app
from pictograph.exceptions import NetworkError, NotFoundError

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
MODEL_ID = "abcdef01-2345-6789-abcd-ef0123456789"
IMAGE_URL = "https://img.test/shelf/aisle%204.jpg"
JPEG = b"\xff\xd8\xff\xe0 not really a jpeg \xff\xd9"


def _plain(output: str) -> str:
    """An error panel's text on one line (the box wraps long messages)."""
    return " ".join(output.replace("│", " ").split())


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """Invoke the CLI as a signed-in user pointed at the mocked API."""
    monkeypatch.setenv("PICTOGRAPH_API_KEY", KEY)
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    def invoke(*args: str, stdin: bytes | None = None) -> Any:
        return runner.invoke(app, list(args), input=stdin)

    return invoke


def _model(httpx_mock: HTTPXMock, name: str = "Shelf Detector", **overrides: Any) -> None:
    payload = {
        "id": MODEL_ID,
        "organization_id": "org",
        "name": name,
        "model_type": "object_detection",
        "status": "ready",
        "visibility": "private",
        "created_at": "2026-04-01T00:00:00Z",
        "updated_at": "2026-04-01T00:00:00Z",
        **overrides,
    }
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/models/{name.replace(' ', '%20')}", json={"data": payload}
    )


def _hosted(
    httpx_mock: HTTPXMock, *, confidence: float = 0.5, top_k: int = 3, **result: Any
) -> None:
    body = {
        "success": True,
        "annotations": [
            {
                "name": "bottle",
                "type": "bbox",
                "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4},
                "confidence": 0.91,
            }
        ],
        "tags": [],
        "model_type": "object_detection",
        "inference_seconds": 0.4,
        **result,
    }
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/models/{MODEL_ID}/predict"
        f"?confidence_threshold={confidence}&top_k={top_k}",
        json={"data": body},
    )


def _uploads(httpx_mock: HTTPXMock) -> list[bytes]:
    """The multipart bodies of every hosted predict call, in order."""
    return [r.content for r in httpx_mock.get_requests() if r.url.path.endswith("/predict")]


# ───────────── the SDK call: a URL is fetched, not opened as a file ─────────────


def test_read_image_takes_a_path_bytes_and_a_url(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(JPEG)
    httpx_mock.add_response(url=IMAGE_URL, content=JPEG)

    assert read_image(photo) == (JPEG, "photo.jpg")
    assert read_image(str(photo)) == (JPEG, "photo.jpg")
    assert read_image(JPEG) == (JPEG, "upload.jpg")
    # The upload is named for the file the URL points at, decoded.
    assert read_image(IMAGE_URL) == (JPEG, "aisle 4.jpg")


def test_a_url_that_does_not_serve_an_image_says_so(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=IMAGE_URL, status_code=404)
    with pytest.raises(ValueError, match=r"answered HTTP 404"):
        read_image(IMAGE_URL)
    httpx_mock.add_response(url=IMAGE_URL, content=b"")
    with pytest.raises(ValueError, match=r"response was empty"):
        read_image(IMAGE_URL)
    httpx_mock.add_exception(httpx.ConnectError("no route"), url=IMAGE_URL)
    with pytest.raises(NetworkError, match=r"Could not fetch the image"):
        read_image(IMAGE_URL)


def test_a_missing_path_is_still_a_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "nope.jpg")


@pytest.mark.anyio
async def test_async_read_image_matches_the_sync_one(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(JPEG)
    httpx_mock.add_response(url=IMAGE_URL, content=JPEG)
    assert await read_image_async(photo) == (JPEG, "photo.jpg")
    assert await read_image_async(JPEG) == (JPEG, "upload.jpg")
    assert await read_image_async(IMAGE_URL) == (JPEG, "aisle 4.jpg")
    httpx_mock.add_response(url=IMAGE_URL, status_code=403)
    with pytest.raises(ValueError, match=r"answered HTTP 403"):
        await read_image_async(IMAGE_URL)


def test_hosted_predict_uploads_the_bytes_behind_a_url(httpx_mock: HTTPXMock) -> None:
    """`client.models.predict(name, image=<url>)` raised FileNotFoundError before."""
    httpx_mock.add_response(url=IMAGE_URL, content=JPEG)
    _model(httpx_mock)
    _hosted(httpx_mock)
    with Client(api_key=KEY, base_url=BASE) as client:
        result = client.models.predict("Shelf Detector", image=IMAGE_URL)
    assert result.annotations[0]["name"] == "bottle"
    (upload,) = _uploads(httpx_mock)
    assert JPEG in upload and b'filename="aisle 4.jpg"' in upload


@pytest.mark.anyio
async def test_async_hosted_predict_uploads_the_bytes_behind_a_url(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(url=IMAGE_URL, content=JPEG)
    _model(httpx_mock)
    _hosted(httpx_mock)
    async with AsyncClient(api_key=KEY, base_url=BASE) as client:
        result = await client.models.predict("Shelf Detector", image=IMAGE_URL)
    assert result.annotations[0]["name"] == "bottle"
    (upload,) = _uploads(httpx_mock)
    assert JPEG in upload


# ───────────── what the commands accept ─────────────


def test_inputs_expand_globs_in_order_and_leave_the_rest_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "frames" / "deep").mkdir(parents=True)
    for name in ("frames/b.jpg", "frames/a.jpg", "frames/deep/c.jpg", "frames/notes.txt"):
        (tmp_path / name).write_bytes(JPEG)
    (tmp_path / "one.png").write_bytes(JPEG)

    assert _inputs.expand_inputs(["one.png", "frames/*.jpg", IMAGE_URL, "-"]) == [
        "one.png",
        "frames/a.jpg",
        "frames/b.jpg",
        IMAGE_URL,
        "-",
    ]
    # ** reaches into folders; a directory a pattern matches is not an image.
    assert _inputs.expand_inputs(["frames/**/*.jpg"]) == [
        "frames/a.jpg",
        "frames/b.jpg",
        "frames/deep/c.jpg",
    ]
    assert _inputs.expand_inputs(["frames/*"]) == [
        "frames/a.jpg",
        "frames/b.jpg",
        "frames/notes.txt",
    ]


def test_a_real_file_with_glob_characters_in_its_name_is_that_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`frame[01].jpg` is a file name, and worked as one before globs existed.

    1.69.99 read any value containing `[`, `*` or `?` as a pattern, and `[01]` is a
    character class that does not match the literal brackets - so the file that was
    typed, and exists, was refused with "No file matches".
    """
    monkeypatch.chdir(tmp_path)
    for name in ("frame[01].jpg", "frame0.jpg", "what?.jpg"):
        (tmp_path / name).write_bytes(JPEG)
    assert _inputs.expand_inputs(["frame[01].jpg"]) == ["frame[01].jpg"]
    assert _inputs.expand_inputs(["what?.jpg"]) == ["what?.jpg"]
    # ...and a pattern that is NOT also a file name still expands.
    assert _inputs.expand_inputs(["frame[0-9].jpg"]) == ["frame0.jpg"]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "Give at least one image."),
        (["-", "-"], "can only be given once"),
        (["frames/*.jpg"], "No file matches 'frames/*.jpg'."),
        (["missing.jpg"], "No such file: missing.jpg"),
    ],
)
def test_bad_inputs_are_refused_before_anything_runs(
    values: list[str], message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(typer.BadParameter) as caught:
        _inputs.expand_inputs(values)
    assert message in str(caught.value.message)


# ───────────── models predict --remote ─────────────


def test_remote_predict_takes_a_url(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=IMAGE_URL, content=JPEG)
    _model(httpx_mock)
    _hosted(httpx_mock)
    res = run("models", "predict", "Shelf Detector", IMAGE_URL, "--remote", "--json")
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["annotations"][0]["name"] == "bottle"
    (upload,) = _uploads(httpx_mock)
    assert JPEG in upload


def test_remote_predict_reads_the_image_from_stdin(run: Any, httpx_mock: HTTPXMock) -> None:
    _model(httpx_mock)
    _hosted(httpx_mock)
    res = run("models", "predict", "Shelf Detector", "-", "--remote", "--json", stdin=JPEG)
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["model_type"] == "object_detection"
    (upload,) = _uploads(httpx_mock)
    assert JPEG in upload


def test_stdin_with_nothing_on_it_is_refused(run: Any, httpx_mock: HTTPXMock) -> None:
    res = run("models", "predict", "Shelf Detector", "-", "--remote", stdin=b"")
    assert res.exit_code == 2
    assert "nothing arrived on stdin" in _plain(res.output)
    assert httpx_mock.get_requests() == []


def test_several_images_print_one_json_line_each_and_look_the_model_up_once(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    for name in ("b.jpg", "a.jpg"):
        (tmp_path / name).write_bytes(JPEG + name.encode())
    httpx_mock.add_response(url=IMAGE_URL, content=JPEG + b"url")
    _model(httpx_mock)
    for _ in range(3):
        _hosted(httpx_mock)

    res = run("models", "predict", "Shelf Detector", "*.jpg", IMAGE_URL, "--remote")
    assert res.exit_code == 0, res.output
    lines = [json.loads(line) for line in res.stdout.splitlines()]
    assert [line["input"] for line in lines] == ["a.jpg", "b.jpg", IMAGE_URL]
    assert all(line["result"]["annotations"][0]["name"] == "bottle" for line in lines)
    # Each image's own bytes, in the order printed.
    uploads = _uploads(httpx_mock)
    assert len(uploads) == 3
    for suffix, body in zip([b"a.jpg", b"b.jpg", b"url"], uploads, strict=True):
        assert JPEG + suffix in body
    # ONE name lookup for the whole batch, not one per image.
    lookups = [r for r in httpx_mock.get_requests() if r.url.path.endswith("/Shelf Detector")]
    assert len(lookups) == 1


def test_one_bad_image_does_not_stop_the_batch(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    for name in ("a.jpg", "c.jpg"):
        (tmp_path / name).write_bytes(JPEG)
    httpx_mock.add_response(url=IMAGE_URL, status_code=404)
    _model(httpx_mock)
    _hosted(httpx_mock)
    _hosted(httpx_mock)

    res = run("models", "predict", "Shelf Detector", "a.jpg", IMAGE_URL, "c.jpg", "--remote")
    assert res.exit_code == 1, "a batch with a failure exits non-zero"
    lines = [json.loads(line) for line in res.stdout.splitlines()]
    assert [sorted(line) for line in lines] == [
        ["input", "result"],
        ["error", "error_type", "input"],
        ["input", "result"],
    ]
    assert lines[1]["input"] == IMAGE_URL
    assert "HTTP 404" in lines[1]["error"] and lines[1]["error_type"] == "ValueError"
    assert len(_uploads(httpx_mock)) == 2


def test_an_unknown_model_stops_before_any_image_is_sent(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    for name in ("a.jpg", "b.jpg"):
        (tmp_path / name).write_bytes(JPEG)
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/models/nope",
        status_code=404,
        json={"detail": "Model 'nope' not found"},
    )
    res = run("models", "predict", "nope", "a.jpg", "b.jpg", "--remote")
    assert res.exit_code != 0
    assert res.stdout.strip() == "", "no per-image lines for a model that does not exist"
    assert _uploads(httpx_mock) == []
    assert isinstance(res.exception, (SystemExit, NotFoundError))


def test_jsonl_for_one_image_and_top_k(run: Any, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(JPEG)
    _model(httpx_mock, model_type="classification")
    _hosted(
        httpx_mock,
        confidence=0.3,
        top_k=5,
        annotations=[],
        tags=["full", "empty"],
        model_type="classification",
    )
    res = run(
        "models", "predict", "Shelf Detector", "a.jpg", "--remote", "--jsonl",
        "--confidence", "0.3", "--top-k", "5",
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    (line,) = [json.loads(text) for text in res.stdout.splitlines()]
    assert line == {
        "input": "a.jpg",
        "result": {
            "success": True,
            "annotations": [],
            "tags": ["full", "empty"],
            "tag_scores": [],
            "model_type": "classification",
            "inference_seconds": 0.4,
        },
    }


def test_a_missing_file_is_refused_before_the_model_is_looked_up(
    run: Any, httpx_mock: HTTPXMock
) -> None:
    res = run("models", "predict", "Shelf Detector", "missing.jpg", "--remote")
    assert res.exit_code == 2
    assert "No such file: missing.jpg" in _plain(res.output)
    assert httpx_mock.get_requests() == []


# ───────────── models predict (local) ─────────────


class _Result:
    model_type = "object_detection"
    predictions: tuple[Any, ...] = ()

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {"model_type": self.model_type, "predictions": []}


def test_local_predict_loads_once_and_hands_over_each_input(run: Any, tmp_path: Path) -> None:
    for name in ("a.jpg", "b.jpg"):
        (tmp_path / name).write_bytes(JPEG)
    local = MagicMock()
    local.predict.return_value = _Result()
    with patch("pictograph.resources.models.Models.load", return_value=local) as load:
        res = run("models", "predict", "Shelf Detector", "a.jpg", IMAGE_URL, "-", "b.jpg",
                  "--confidence", "0.4", stdin=JPEG)  # fmt: skip
    assert res.exit_code == 0, res.output
    load.assert_called_once_with("Shelf Detector", confidence=0.4)
    # A path, a URL (the local loader fetches it), the piped bytes, a path.
    assert [call.args[0] for call in local.predict.call_args_list] == [
        Path("a.jpg"),
        IMAGE_URL,
        JPEG,
        Path("b.jpg"),
    ]
    assert [json.loads(line)["input"] for line in res.stdout.splitlines()] == [
        "a.jpg",
        IMAGE_URL,
        "-",
        "b.jpg",
    ]


def test_local_top_k_reaches_a_classifier_and_only_a_classifier(run: Any, tmp_path: Path) -> None:
    from pictograph.inference import ClassificationModel, DetectionModel

    (tmp_path / "a.jpg").write_bytes(JPEG)
    for model_class, expected in ((ClassificationModel, {"top_k": 4}), (DetectionModel, {})):
        local = MagicMock(spec=model_class)
        local.predict.return_value = _Result()
        with patch("pictograph.resources.models.Models.load", return_value=local):
            res = run("models", "predict", "m", "a.jpg", "--top-k", "4", "--jsonl")
        assert res.exit_code == 0, res.output
        assert local.predict.call_args.kwargs == expected, model_class.__name__


# ───────────── deployments predict ─────────────

ENDPOINT = "https://infer.test/dep-1/predict"


def _endpoint(httpx_mock: HTTPXMock, times: int = 1) -> None:
    for _ in range(times):
        httpx_mock.add_response(
            method="POST",
            url=ENDPOINT,
            json={"model_type": "object_detection", "predictions": [{"name": "bottle"}]},
        )


def test_deployment_predict_reads_stdin(run: Any, httpx_mock: HTTPXMock) -> None:
    _endpoint(httpx_mock)
    res = run("deployments", "predict", "dep", "-", "--token", "pk_deploy_t",
              "--endpoint", ENDPOINT, stdin=JPEG)  # fmt: skip
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["predictions"] == [{"name": "bottle"}]
    (request,) = httpx_mock.get_requests()
    assert request.headers["authorization"] == "Bearer pk_deploy_t"


def test_deployment_predict_runs_several_inputs_as_json_lines(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    for name in ("a.jpg", "b.jpg"):
        (tmp_path / name).write_bytes(JPEG)
    _endpoint(httpx_mock, times=3)
    res = run("deployments", "predict", "dep", "*.jpg", IMAGE_URL, "--token", "pk_deploy_t",
              "--endpoint", ENDPOINT)  # fmt: skip
    assert res.exit_code == 0, res.output
    lines = [json.loads(line) for line in res.stdout.splitlines()]
    assert [line["input"] for line in lines] == ["a.jpg", "b.jpg", IMAGE_URL]
    assert all(line["result"]["predictions"] == [{"name": "bottle"}] for line in lines)
    assert len(httpx_mock.get_requests()) == 3
