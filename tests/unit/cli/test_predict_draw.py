"""``--draw`` / ``--look`` on ``models predict`` and ``deployments predict`` (B613).

Each prediction drawn on its image in THE annotation look - the look of every live view and the app's
image test - and saved: one image to a file (or into a folder), several images or a dataset into a
folder named after their inputs, a video's sampled frames into one drawn video.

The real client over a mocked transport, a mocked local model returning REAL results, real images and
real videos written with Pillow and OpenCV.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, ImageChops
from typer.testing import CliRunner

from pictograph.cli._app import app
from pictograph.cli._errors import ExitCode
from pictograph.inference.results import DetectionResult

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
MODEL_ID = "abcdef01-2345-6789-abcd-ef0123456789"
ENDPOINT = "https://infer.test/dep-1/predict"
DATASET = "Shelf Photos"
DATASET_PATH = "Shelf%20Photos"
GREY = (90, 96, 104)
BOX = {
    "id": "b",
    "name": "pallet",
    "type": "bbox",
    "bounding_box": {"x": 8, "y": 10, "w": 30, "h": 24},
    "confidence": 0.9,
}


def _plain(output: str) -> str:
    return " ".join(output.replace("│", " ").split())


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


def _photo(path: Path, size: tuple[int, int] = (64, 48)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, GREY).save(path)
    return path


def _photo_bytes(size: tuple[int, int] = (64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, GREY).save(buf, "PNG")
    return buf.getvalue()


def _local_model() -> MagicMock:
    local = MagicMock()
    local.predict.side_effect = lambda _image, **_: DetectionResult.model_validate(
        {"predictions": [BOX]}
    )
    return local


def _lines(output: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def _drawn(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def _is_drawn(im: Image.Image) -> bool:
    """Something was drawn: the plain grey photo carries a box (its line, its label) - far past a JPEG's noise."""
    diff = ImageChops.difference(im, Image.new("RGB", im.size, GREY))
    return max(high for _, high in diff.getextrema()) > 40


# ───────────────────────── one image ─────────────────────────


def test_one_image_to_a_file_and_its_json_stays_clean(run: Any, tmp_path: Path) -> None:
    _photo(tmp_path / "shelf.jpg")
    with patch("pictograph.resources.models.Models.load", return_value=_local_model()):
        res = run("models", "predict", "Shelf Detector", "shelf.jpg", "--draw", "out.png", "--json")
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["predictions"][0]["name"] == "pallet"  # stdout: the JSON alone
    assert "Drawn: out.png" in res.stderr
    drawn = _drawn(tmp_path / "out.png")
    assert drawn.size == (64, 48) and _is_drawn(drawn)


def test_one_image_into_a_folder_keeps_its_name(run: Any, tmp_path: Path) -> None:
    _photo(tmp_path / "shelf.jpg")
    with patch("pictograph.resources.models.Models.load", return_value=_local_model()):
        res = run("models", "predict", "Shelf Detector", "shelf.jpg", "--draw", "drawn")
    assert res.exit_code == 0, res.output
    assert _is_drawn(_drawn(tmp_path / "drawn" / "shelf.jpg"))


def test_several_images_go_into_a_folder_each_line_naming_its_drawing(
    run: Any, tmp_path: Path
) -> None:
    _photo(tmp_path / "a" / "x.jpg")
    _photo(tmp_path / "b" / "x.jpg")  # the same name twice: the second is x-2
    _photo(tmp_path / "scan.tiff")
    with patch("pictograph.resources.models.Models.load", return_value=_local_model()):
        res = run("models", "predict", "m", "a/x.jpg", "b/x.jpg", "scan.tiff", "--draw", "drawn")
    assert res.exit_code == 0, res.output
    lines = _lines(res.stdout)
    assert [line["drawn"] for line in lines] == [
        str(Path("drawn/x.jpg")),
        str(Path("drawn/x-2.jpg")),
        str(Path("drawn/scan.tiff")),
    ]
    assert all(_is_drawn(_drawn(tmp_path / line["drawn"])) for line in lines)
    assert all("result" in line for line in lines)


def test_the_look_changes_what_is_drawn(run: Any, tmp_path: Path) -> None:
    _photo(tmp_path / "shelf.png")
    (tmp_path / "look.json").write_text(json.dumps({"label": {"show": False}}))
    outs = {}
    for name, look in (("plain", None), ("smoke", "smoke"), ("json", '{"box": {"fill": 0.6}}'),
                       ("file", "@look.json")):  # fmt: skip
        args = ["models", "predict", "m", "shelf.png", "--draw", f"{name}.png"]
        if look is not None:
            args += ["--look", look]
        with patch("pictograph.resources.models.Models.load", return_value=_local_model()):
            res = run(*args)
        assert res.exit_code == 0, res.output
        outs[name] = _drawn(tmp_path / f"{name}.png").tobytes()
    assert len(set(outs.values())) == 4


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["a.png", "b.png", "--draw", "out.jpg"], "Several images are drawn into a folder"),
        (["a.png", "--draw", "out.mp4"], "An image is drawn into an image"),
        (["a.png", "--look", "smoke"], "--look needs --draw"),
        (["a.png", "--draw", "o.png", "--look", "glossy"], "is not a look"),
        (["a.png", "--draw", "o.png", "--look", '{"box": {"wdth": 3}}'], "Not a field of the look"),
        (["a.png", "--draw", "o.png", "--look", "{nope"], "Not JSON"),
        (["a.png", "--draw", "o.png", "--look", "@missing.json"], "Could not read"),
    ],
)
def test_a_draw_that_cannot_happen_is_refused_before_the_model_loads(
    run: Any, tmp_path: Path, args: list[str], message: str
) -> None:
    _photo(tmp_path / "a.png")
    _photo(tmp_path / "b.png")
    with patch("pictograph.resources.models.Models.load") as load:
        res = run("models", "predict", "m", *args)
    assert res.exit_code == ExitCode.INVALID
    assert message in _plain(res.output)
    load.assert_not_called()


# ───────────────────────── a video ─────────────────────────


def test_a_videos_frames_become_one_drawn_video(run: Any, tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    writer = cv2.VideoWriter(
        str(tmp_path / "clip.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48)
    )
    for _ in range(20):
        writer.write(np.full((48, 64, 3), 90, dtype=np.uint8))
    writer.release()
    with patch("pictograph.resources.models.Models.load", return_value=_local_model()):
        res = run("models", "predict", "m", "--video", "clip.avi", "--sample-fps", "5",
                  "--draw", "out.mp4")  # fmt: skip
    assert res.exit_code == 0, res.output
    lines = _lines(res.stdout)
    assert len(lines) == 10 and {line["drawn"] for line in lines} == {"out.mp4"}
    capture = cv2.VideoCapture(str(tmp_path / "out.mp4"))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(5, abs=0.01)  # plays in real time
        ok, frame = capture.read()
        assert ok and frame.shape == (48, 64, 3)
        assert float(np.abs(frame.astype(int) - 90).max()) > 40  # the box is in it
    finally:
        capture.release()


def test_a_video_is_never_drawn_into_an_image(run: Any, tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    (tmp_path / "clip.avi").write_bytes(b"x")
    with patch("pictograph.resources.models.Models.load") as load:
        res = run("models", "predict", "m", "--video", "clip.avi", "--draw", "frame.png")
    assert res.exit_code == ExitCode.INVALID
    load.assert_not_called()


# ───────────────────────── a dataset, a hosted model, a deployment ─────────────────────────


def _dataset(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/datasets/{DATASET_PATH}/storage",
        json={"data": {"storage_class": "standard", "storage_state": "idle"}},
    )
    rows = [
        {"id": "11111111-1111-1111-1111-111111111111", "filename": "a.jpg", "directory_path": "/",
         "status": "new", "annotation_count": 0, "file_size": 10, "created_at": "2026-04-01T00:00:00Z"},
        {"id": "22222222-2222-2222-2222-222222222222", "filename": "b 2.png",
         "directory_path": "/val", "status": "new", "annotation_count": 0, "file_size": 10,
         "created_at": "2026-04-01T00:00:00Z"},
    ]  # fmt: skip
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/images/?dataset={DATASET_PATH}&limit=200",
        json={"data": rows, "pagination": {"has_more": False, "next_cursor": None}},
    )
    for address in ("a.jpg", "val/b%202.png"):
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/developer/images/{DATASET_PATH}/{address}", content=_photo_bytes()
        )


def test_a_datasets_drawings_keep_its_folders(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    _dataset(httpx_mock)
    with patch("pictograph.resources.models.Models.load", return_value=_local_model()):
        res = run("models", "predict", "m", "--dataset", DATASET, "--draw", "drawn")
    assert res.exit_code == 0, res.output
    assert [line["drawn"] for line in _lines(res.stdout)] == [
        str(Path("drawn/a.jpg")),
        str(Path("drawn/val/b 2.png")),
    ]
    assert _is_drawn(_drawn(tmp_path / "drawn" / "val" / "b 2.png"))


def test_a_hosted_models_answer_is_drawn(run: Any, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    _photo(tmp_path / "shelf.png")
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/developer/models/Shelf%20Detector",
        json={"data": {"id": MODEL_ID, "organization_id": "org", "name": "Shelf Detector",
                       "model_type": "object_detection", "status": "ready", "visibility": "private",
                       "created_at": "2026-04-01T00:00:00Z", "updated_at": "2026-04-01T00:00:00Z"}},
    )  # fmt: skip
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/models/{MODEL_ID}/predict?confidence_threshold=0.5&top_k=3",
        json={"data": {"success": True, "annotations": [BOX], "tags": [],
                       "model_type": "object_detection", "inference_seconds": 0.1}},
    )  # fmt: skip
    res = run("models", "predict", "Shelf Detector", "shelf.png", "--remote", "--draw", "out.png")
    assert res.exit_code == 0, res.output
    assert _is_drawn(_drawn(tmp_path / "out.png"))


def test_a_deployments_answer_is_drawn_from_its_raw_json(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PICTOGRAPH_API_KEY")  # the deployment token alone authorizes it
    _photo(tmp_path / "a.png")
    _photo(tmp_path / "b.png")
    httpx_mock.add_response(
        method="POST", url=ENDPOINT, json={"model_type": "object_detection", "predictions": [BOX]}
    )
    httpx_mock.add_response(
        method="POST",
        url=ENDPOINT,
        json={
            "model_type": "classification",
            "predictions": [{"class": "indoor", "confidence": 0.9}],
        },
    )
    res = run("deployments", "predict", "dep", "a.png", "b.png", "--token", "pk_deploy_t",
              "--endpoint", ENDPOINT, "--draw", "drawn")  # fmt: skip
    assert res.exit_code == 0, res.output
    lines = _lines(res.stdout)
    # The line carries the endpoint's own JSON; the drawing reads it as its typed result.
    assert lines[0]["result"]["predictions"][0]["name"] == "pallet"
    assert all(_is_drawn(_drawn(tmp_path / line["drawn"])) for line in lines)


def test_an_answer_with_no_model_type_is_that_images_error(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PICTOGRAPH_API_KEY")
    _photo(tmp_path / "a.png")
    _photo(tmp_path / "b.png")
    for _ in range(2):
        httpx_mock.add_response(method="POST", url=ENDPOINT, json={"predictions": []})
    res = run("deployments", "predict", "dep", "a.png", "b.png", "--token", "pk_deploy_t",
              "--endpoint", ENDPOINT, "--draw", "drawn")  # fmt: skip
    assert res.exit_code == 1
    assert all("no model type it can be drawn by" in line["error"] for line in _lines(res.stdout))
