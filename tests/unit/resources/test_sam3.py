"""``client.sam3.predict`` and ``pictograph sam3 predict``: SAM3 on any image (B533 V3).

Until 1.69.107 SAM3 ran only on an image already in a dataset. These drive the REAL
client over a mocked HTTP transport, and decode the multipart body each call sends, so
every test covers the arguments, the bytes on the wire and the parsed result.

Coverage:
- The prompt: exactly one of ``text`` / ``box`` / ``points``, each in the wire syntax the
  route reads (``box=x,y,w,h``, ``point=x,y`` / ``x,y,neg`` repeated); refused before any
  request when it is missing, doubled or malformed.
- The image: a path, bytes, or a URL fetched on this machine and then uploaded.
- The result: a typed :class:`Sam3Prediction` - annotations, the upright size, the
  charge identity - and the async twin making the same request.
- The CLI: one image as JSON or a table, several as JSON Lines, stdin, a dataset's
  images (read only), clear refusals, and a batch that stops at a 402.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from pictograph import AsyncClient, Client, Sam3Prediction
from pictograph.cli._app import app
from pictograph.cli._errors import ExitCode
from pictograph.exceptions import PaymentRequiredError, ValidationError
from pictograph.models.annotation import BBoxAnnotation, PolygonAnnotation
from pictograph.resources.sam3 import PREDICT_TIMEOUT_SECONDS, prompt_form

if TYPE_CHECKING:
    from pathlib import Path

    import httpx
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
URL = f"{BASE}/api/v1/developer/sam3/predict"
SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
RING = [{"x": 10, "y": 10}, {"x": 90, "y": 10}, {"x": 90, "y": 70}]


def _result(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "status": "success",
        "prompt": "text",
        "annotations": [
            {
                "id": "a1",
                "name": "red car",
                "type": "polygon",
                "polygon": {"paths": [RING]},
                "bounding_box": {"x": 10, "y": 10, "w": 80, "h": 60},
                "confidence": 0.88,
            }
        ],
        "score": None,
        "inference_time": 0.4,
        "image_sha256": SHA,
        "confidence_threshold": 0.3,
        "degraded": False,
        "width": 640,
        "height": 480,
    }
    body.update(over)
    return {"data": body}


def _parts(request: httpx.Request) -> tuple[dict[str, list[str]], dict[str, tuple[str, bytes]]]:
    """``(form fields, files)`` of a multipart request, decoded from the wire bytes."""
    boundary = request.headers["content-type"].split("boundary=", 1)[1].encode()
    fields: dict[str, list[str]] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for chunk in request.content.split(b"--" + boundary):
        chunk = chunk.strip(b"\r\n")
        if not chunk or chunk == b"--":
            continue
        head, _, body = chunk.partition(b"\r\n\r\n")
        disposition = next(
            line
            for line in head.decode().split("\r\n")
            if line.lower().startswith("content-disposition")
        )
        name = disposition.split('name="', 1)[1].split('"', 1)[0]
        if 'filename="' in disposition:
            files[name] = (disposition.split('filename="', 1)[1].split('"', 1)[0], body)
        else:
            fields.setdefault(name, []).append(body.decode())
    return fields, files


@pytest.fixture
def client() -> Client:
    return Client(api_key=KEY, base_url=BASE, max_retries=0)


# ───────────────────────── the prompt ─────────────────────────


def _form(**kw: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "text": None, "box": None, "points": None, "negative_points": None,
        "negative_boxes": None, "name": None, "output": "polygon", "confidence": None,
        "max_detections": 50,
    }  # fmt: skip
    args.update(kw)
    return prompt_form(**args)


def test_a_text_prompt_sends_the_phrase_and_its_match_cap() -> None:
    assert _form(text="  red car ") == {
        "output": "polygon",
        "text": "red car",
        "max_detections": "50",
    }


@pytest.mark.parametrize("box", [{"x": 120, "y": 80.5, "w": 200, "h": 160}, (120, 80.5, 200, 160)])
def test_a_box_is_one_field_in_the_routes_syntax(box: Any) -> None:
    form = _form(box=box, negative_boxes=[(0, 0, 10, 10)], name="car", confidence=0.4)
    assert form == {
        "output": "polygon",
        "box": "120,80.5,200,160",
        "negative_box": ["0,0,10,10"],
        "name": "car",
        "confidence_threshold": "0.4",
    }


def test_points_repeat_and_a_negative_point_says_neg() -> None:
    form = _form(points=[(320, 240), (330.6, 250)], negative_points=[(10, 20)], output="bbox")
    assert form == {"output": "bbox", "point": ["320,240", "331,250", "10,20,neg"]}


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({}, "Got none"),
        ({"points": []}, "Got none"),
        ({"text": "car", "box": (1, 2, 3, 4)}, "Got text and box"),
        ({"negative_points": [(1, 2)], "text": "car"}, "negative_points go with points="),
        ({"negative_boxes": [(1, 2, 3, 4)], "text": "car"}, "negative_boxes go with box="),
        ({"box": (1, 2, 3)}, "four numbers, got 3"),
        ({"box": {"x": 1, "y": 2, "w": 3}}, "missing 'h'"),
        ({"points": [(1, 2, 3)]}, "two numbers, got 3"),
        ({"text": "car", "output": "mask"}, "'polygon' or 'bbox'"),
        ({"text": "   "}, "must not be blank"),
    ],
)
def test_a_wrong_prompt_is_refused_before_any_request(kw: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _form(**kw)


# ───────────────────────── the call ─────────────────────────


def test_predict_uploads_the_file_with_the_prompt_and_parses_the_result(
    httpx_mock: HTTPXMock, client: Client, tmp_path: Path
) -> None:
    httpx_mock.add_response(method="POST", url=URL, json=_result())
    image = tmp_path / "shelf.png"
    image.write_bytes(b"\x89PNG fake")

    result = client.sam3.predict(image, text="red car", confidence=0.25)

    assert isinstance(result, Sam3Prediction)
    assert (result.status, result.prompt, result.width, result.height) == (
        "success",
        "text",
        640,
        480,
    )
    assert result.image_sha256 == SHA and result.degraded is False
    (annotation,) = result.annotations
    assert isinstance(annotation, PolygonAnnotation) and annotation.confidence == 0.88
    (request,) = httpx_mock.get_requests()
    fields, files = _parts(request)
    assert fields == {
        "output": ["polygon"], "text": ["red car"], "max_detections": ["50"],
        "confidence_threshold": ["0.25"],
    }  # fmt: skip
    assert files == {"file": ("shelf.png", b"\x89PNG fake")}
    assert b"Content-Type: image/png" in request.content
    assert request.headers["x-api-key"] == KEY


def test_predict_waits_for_a_gpu_that_is_starting(httpx_mock: HTTPXMock, client: Client) -> None:
    """The first prompt after a quiet spell starts a GPU: longer than the default 30 s."""
    httpx_mock.add_response(method="POST", url=URL, json=_result())
    client.sam3.predict(b"raw", text="car")
    (request,) = httpx_mock.get_requests()
    assert request.extensions["timeout"]["read"] == PREDICT_TIMEOUT_SECONDS


def test_predict_takes_raw_bytes_and_a_box(httpx_mock: HTTPXMock, client: Client) -> None:
    httpx_mock.add_response(
        method="POST",
        url=URL,
        json=_result(
            prompt="box",
            score=0.93,
            annotations=[
                {
                    "id": "b1",
                    "name": "car",
                    "type": "bbox",
                    "confidence": 0.93,
                    "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4},
                },
            ],
        ),
    )
    result = client.sam3.predict(b"raw jpeg", box=(1, 2, 3, 4), name="car", output="bbox")
    assert isinstance(result.annotations[0], BBoxAnnotation)
    assert result.score == 0.93
    fields, files = _parts(httpx_mock.get_requests()[0])
    assert fields == {"output": ["bbox"], "box": ["1,2,3,4"], "name": ["car"]}
    assert files == {"file": ("upload.jpg", b"raw jpeg")}


def test_predict_fetches_a_url_on_this_machine_then_uploads_it(
    httpx_mock: HTTPXMock, client: Client
) -> None:
    httpx_mock.add_response(url="https://img.test/street%20cam/frame.jpg", content=b"jpeg bytes")
    httpx_mock.add_response(method="POST", url=URL, json=_result(prompt="point", annotations=[]))
    client.sam3.predict("https://img.test/street%20cam/frame.jpg", points=[(5, 6)])
    fetch, upload = httpx_mock.get_requests()
    assert fetch.method == "GET" and "x-api-key" not in fetch.headers, "the key stays with the API"
    fields, files = _parts(upload)
    assert fields == {"output": ["polygon"], "point": ["5,6"]}
    assert files == {"file": ("frame.jpg", b"jpeg bytes")}


def test_no_detection_is_a_status_not_an_error(httpx_mock: HTTPXMock, client: Client) -> None:
    httpx_mock.add_response(
        method="POST", url=URL, json=_result(status="no_detection", annotations=[])
    )
    result = client.sam3.predict(b"raw", text="unicorn")
    assert result.status == "no_detection" and result.annotations == []


def test_a_bad_prompt_never_reaches_the_api(httpx_mock: HTTPXMock, client: Client) -> None:
    with pytest.raises(ValueError, match="exactly one prompt"):
        client.sam3.predict(b"raw")
    assert httpx_mock.get_requests() == []


@pytest.mark.parametrize(("status", "error"), [(402, PaymentRequiredError), (400, ValidationError)])
def test_refusals_surface_as_their_typed_errors(
    httpx_mock: HTTPXMock, client: Client, status: int, error: type[Exception]
) -> None:
    httpx_mock.add_response(method="POST", url=URL, status_code=status, json={"detail": "no"})
    with pytest.raises(error):
        client.sam3.predict(b"raw", text="car")


@pytest.mark.anyio
async def test_the_async_twin_sends_the_same_request(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url="https://img.test/a.png", content=b"png bytes")
    httpx_mock.add_response(method="POST", url=URL, json=_result())
    async with AsyncClient(api_key=KEY, base_url=BASE, max_retries=0) as client:
        result = await client.sam3.predict(
            "https://img.test/a.png", points=[(1, 2)], negative_points=[(3, 4)], name="car"
        )
    assert isinstance(result, Sam3Prediction) and result.width == 640
    fields, files = _parts(httpx_mock.get_requests()[1])
    assert fields == {"output": ["polygon"], "point": ["1,2", "3,4,neg"], "name": ["car"]}
    assert files == {"file": ("a.png", b"png bytes")}


# ───────────────────────── the CLI ─────────────────────────


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


def _plain(output: str) -> str:
    return " ".join(output.replace("│", " ").split())


def test_cli_one_image_as_json(run: Any, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8 a")
    httpx_mock.add_response(method="POST", url=URL, json=_result())
    res = run("sam3", "predict", "a.jpg", "--text", "red car", "--confidence", "0.3", "--json")
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    assert out["prompt"] == "text" and out["annotations"][0]["name"] == "red car"
    fields, files = _parts(httpx_mock.get_requests()[0])
    assert fields["text"] == ["red car"] and fields["confidence_threshold"] == ["0.3"]
    assert files["file"] == ("a.jpg", b"\xff\xd8 a")


def test_cli_one_image_as_a_table(run: Any, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8 a")
    httpx_mock.add_response(method="POST", url=URL, json=_result())
    res = run("sam3", "predict", "a.jpg", "--text", "red car")
    assert res.exit_code == 0, res.output
    assert "1 object(s) on the 640x480 image" in _plain(res.stdout)
    assert "red car" in res.stdout and "polygon" in res.stdout


def test_cli_points_and_stdin(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=URL, json=_result(prompt="point"))
    res = run("sam3", "predict", "-", "--point", "320,240", "--point", "300,400,neg",
              "--output", "bbox", "--json", stdin=b"piped bytes")  # fmt: skip
    assert res.exit_code == 0, res.output
    fields, files = _parts(httpx_mock.get_requests()[0])
    assert fields == {"output": ["bbox"], "point": ["320,240", "300,400,neg"]}
    assert files["file"][1] == b"piped bytes"


def test_cli_several_images_are_json_lines_and_one_failure_is_its_line(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    for name in ("a.jpg", "b.jpg"):
        (tmp_path / name).write_bytes(name.encode())
    httpx_mock.add_response(method="POST", url=URL, json=_result())
    httpx_mock.add_response(
        method="POST", url=URL, status_code=400, json={"detail": "not an image"}
    )
    res = run("sam3", "predict", "*.jpg", "--box", "1,2,30,40")
    assert res.exit_code == 1, "a run with a failure exits non-zero"
    first, second = (json.loads(line) for line in res.stdout.splitlines())
    assert first["input"] == "a.jpg" and first["result"]["status"] == "success"
    assert second["input"] == "b.jpg" and second["error_type"] == "ValidationError"


def test_cli_stops_a_batch_at_a_402(run: Any, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (tmp_path / name).write_bytes(name.encode())
    httpx_mock.add_response(method="POST", url=URL, status_code=402, json={"detail": "credits"})
    res = run("sam3", "predict", "a.jpg", "b.jpg", "c.jpg", "--text", "car")
    assert res.exit_code == ExitCode.PAYMENT
    assert len(res.stdout.splitlines()) == 1, "the rest would fail the same way"
    assert "Stopped after 1 input(s)" in _plain(res.stderr)
    assert len(httpx_mock.get_requests()) == 1


def test_cli_reads_a_datasets_images_and_writes_nothing(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/shelf/storage",
        json={"data": {"storage_class": "standard", "storage_state": "idle"}},
    )
    row = {
        "id": "11111111-1111-1111-1111-111111111111", "filename": "a.jpg",
        "directory_path": "/val", "status": "new", "annotation_count": 0, "file_size": 3,
        "created_at": "2026-04-01T00:00:00Z",
    }  # fmt: skip
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/images/?dataset=shelf&limit=200",
        json={"data": [row], "pagination": {"has_more": False, "next_cursor": None}},
    )
    httpx_mock.add_response(url=f"{BASE}/api/v1/developer/images/shelf/val/a.jpg", content=b"img")
    httpx_mock.add_response(method="POST", url=URL, json=_result())
    res = run("sam3", "predict", "--dataset", "shelf", "--text", "car")
    assert res.exit_code == 0, res.output
    (line,) = (json.loads(x) for x in res.stdout.splitlines())
    assert (line["input"], line["image_id"]) == ("val/a.jpg", row["id"])
    # Only reads, plus the SAM3 call itself: nothing was written to the dataset.
    writes = [r for r in httpx_mock.get_requests() if r.method != "GET"]
    assert [str(r.url) for r in writes] == [URL]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["a.jpg"], "Give exactly one prompt"),
        (["a.jpg", "--text", "car", "--box", "1,2,3,4"], "Got --text and --box"),
        (["a.jpg", "--point", "1,2,neg"], "at least one point ON the object"),
        (["a.jpg", "--point", "1,2,maybe"], "the third part is 'neg'"),
        (["a.jpg", "--box", "1,2,3"], "--box is x,y,w,h"),
        (["a.jpg", "--text", "car", "--negative-box", "1,2,3,4"], "--negative-box goes with --box"),
        (["a.jpg", "--text", "car", "--output", "mask"], "--output is polygon or bbox"),
        (["--text", "car"], "Give at least one image (or --video / --dataset)."),
    ],
)
def test_cli_refuses_a_wrong_call_before_any_request(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path, args: list[str], message: str
) -> None:
    (tmp_path / "a.jpg").write_bytes(b"jpeg")
    res = run("sam3", "predict", *args)
    assert res.exit_code == ExitCode.INVALID
    assert message in _plain(res.output)
    assert httpx_mock.get_requests() == []
