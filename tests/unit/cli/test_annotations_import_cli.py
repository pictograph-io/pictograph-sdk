"""``pictograph annotations import-coco / import-yolo / import-voc`` (B533 P1 #7).

The SDK could import COCO, YOLO and Pascal VOC; the CLI could only import COCO
(``datasets import-coco``), so an agent holding a YOLO or VOC export had no command
for it. These drive the REAL app and client over a mocked HTTP transport and read
every request: a label file must land on exactly the image it names, a class list
that does not cover the labels must be refused before anything is sent, and a name
two images share must never be guessed (an import REPLACES annotations).
"""

from __future__ import annotations

import builtins
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from pictograph.cli import _label_files
from pictograph.cli._app import app
from pictograph.cli._errors import ExitCode

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
DATASET_URL = f"{BASE}/api/v1/developer/datasets/road-signs"
LISTING = re.compile(rf"{re.escape(BASE)}/api/v1/developer/images/\?.*")
BULK = f"{BASE}/api/v1/developer/annotations/bulk"
NOW = "2026-04-01T00:00:00Z"
VOC = (
    "<annotation><filename>{f}</filename><object><name>car</name><bndbox>"
    "<xmin>10</xmin><ymin>20</ymin><xmax>40</xmax><ymax>60</ymax></bndbox></object>"
    "</annotation>"
)


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_test")
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    monkeypatch.setenv("PICTOGRAPH_MAX_RETRIES", "0")
    monkeypatch.setenv("PICTOGRAPH_OUTPUT", "json")  # failures print as JSON on stderr
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    return lambda *args: runner.invoke(app, list(args))


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def _dataset(httpx_mock: HTTPXMock, images: list[tuple[str, str]]) -> None:
    """The dataset ``road-signs`` and its images ``(id, filename)``, 100x100 px."""
    httpx_mock.add_response(
        method="GET",
        url=DATASET_URL,
        json={
            "data": {
                "id": "proj-1",
                "name": "road-signs",
                "image_count": len(images),
                "created_at": NOW,
                "classes": [],
            }
        },
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=LISTING,
        json={
            "data": [
                {
                    "id": image_id,
                    "filename": filename,
                    "width": 100,
                    "height": 100,
                    "annotation_count": 0,
                    "created_at": NOW,
                }
                for image_id, filename in images
            ]
        },
    )


def _bulk(httpx_mock: HTTPXMock) -> None:
    def echo(request: Any) -> Any:
        import httpx

        saves = json.loads(request.content)["saves"]
        return httpx.Response(
            200,
            json={
                "saved": [
                    {
                        "image_id": s["image_id"],
                        "previous_count": 0,
                        "new_count": len(s["annotations"]),
                        "status": "in_progress",
                    }
                    for s in saves
                ],
                "failed": [],
            },
        )

    httpx_mock.add_callback(echo, method="POST", url=BULK)


def _saved(httpx_mock: HTTPXMock) -> dict[str, list[dict[str, Any]]]:
    (bulk,) = [r for r in httpx_mock.get_requests() if r.method == "POST"]
    return {s["image_id"]: s["annotations"] for s in json.loads(bulk.content)["saves"]}


def _error(res: Any) -> dict[str, Any]:
    error: dict[str, Any] = json.loads(res.stderr.strip().splitlines()[-1])["error"]
    return error


# ───────────── YOLO ─────────────


def test_import_yolo_finds_the_export_data_yaml_and_matches_labels_by_stem(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    # A Roboflow / Ultralytics export: data.yaml two folders above the labels.
    _write(
        tmp_path / "export",
        {
            "data.yaml": "train: ../train/images\nnc: 2\nnames:\n  - car\n  - truck\n",
            "train/labels/a.txt": "0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1\n",
            "train/labels/b.txt": "1 0.5 0.5 0.4 0.4\n",
            "train/images/a.jpg": "not read",
        },
    )
    _dataset(httpx_mock, [("img-a", "a.jpg"), ("img-b", "b.png")])
    _bulk(httpx_mock)

    res = run(
        "annotations", "import-yolo", "road-signs", "export/train/labels", "--no-create-classes"
    )

    assert res.exit_code == ExitCode.OK, res.output
    saved = _saved(httpx_mock)
    assert [a["name"] for a in saved["img-a"]] == ["car", "truck"]
    assert saved["img-b"][0]["name"] == "truck"
    assert saved["img-b"][0]["bounding_box"] == {"x": 30.0, "y": 30.0, "w": 40.0, "h": 40.0}
    report = json.loads(res.stdout)
    assert report["classes"] == ["car", "truck"]
    assert report["class_list"].endswith("export/data.yaml")
    assert report["files"] == 2 and report["images_saved"] == 2 and report["success"] is True


def test_import_yolo_takes_an_inline_class_list(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    _write(tmp_path, {"labels/a.txt": "1 0.5 0.5 0.2 0.2\n"})
    _dataset(httpx_mock, [("img-a", "a.jpg")])
    _bulk(httpx_mock)
    res = run(
        "annotations",
        "import-yolo",
        "road-signs",
        "labels",
        "--classes",
        "car, traffic light",
        "--no-create-classes",
    )
    assert res.exit_code == ExitCode.OK, res.output
    assert _saved(httpx_mock)["img-a"][0]["name"] == "traffic light"


def test_a_label_numbering_a_class_past_the_list_is_refused_before_any_request(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    # Before: from_yolo skipped the line - the label vanished without a word.
    _write(
        tmp_path, {"labels/classes.txt": "car\n", "labels/a.txt": "0 .5 .5 .1 .1\n2 .5 .5 .1 .1\n"}
    )
    res = run("annotations", "import-yolo", "road-signs", "labels")
    assert res.exit_code == ExitCode.INVALID
    error = _error(res)
    assert error["code"] == "class_number_out_of_range"
    assert "a.txt uses class 2, but the class list" in error["message"]
    assert httpx_mock.get_requests() == []


def test_no_class_list_is_refused_with_how_to_give_one(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    _write(tmp_path, {"deep/a/labels/a.txt": "0 .5 .5 .1 .1\n"})
    res = run("annotations", "import-yolo", "road-signs", "deep/a/labels")
    assert res.exit_code == ExitCode.INVALID
    assert _error(res)["code"] == "no_class_list"
    assert "--classes data.yaml" in _error(res)["fix"]
    assert httpx_mock.get_requests() == []


def test_a_missing_class_file_is_a_usage_error(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    _write(tmp_path, {"labels/a.txt": "0 .5 .5 .1 .1\n"})
    res = run("annotations", "import-yolo", "road-signs", "labels", "--classes", "nope.yaml")
    assert res.exit_code == ExitCode.INVALID
    assert "No such file: nope.yaml" in " ".join(res.output.replace("│", " ").split())


def test_two_label_files_with_one_name_are_refused(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    _write(
        tmp_path,
        {
            "export/classes.txt": "car\n",
            "export/train/a.txt": "0 .5 .5 .1 .1\n",
            "export/valid/a.txt": "0 .5 .5 .1 .1\n",
        },
    )
    res = run("annotations", "import-yolo", "road-signs", "export")
    assert res.exit_code == ExitCode.INVALID
    assert _error(res)["code"] == "duplicate_label_files"
    assert httpx_mock.get_requests() == []


# ───────────── Pascal VOC ─────────────


def test_import_voc_lands_each_file_on_the_image_its_filename_names(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    # The XML's own name says nothing; <filename> names the image.
    _write(tmp_path, {"Annotations/x1.xml": VOC.format(f="a.jpg")})
    _dataset(httpx_mock, [("img-a", "a.jpg")])
    _bulk(httpx_mock)

    res = run("annotations", "import-voc", "road-signs", "Annotations", "--no-create-classes")

    assert res.exit_code == ExitCode.OK, res.output
    (ann,) = _saved(httpx_mock)["img-a"]
    assert ann["name"] == "car" and ann["bounding_box"] == {"x": 10, "y": 20, "w": 30, "h": 40}


def test_a_name_two_images_share_is_reported_and_never_written(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    _write(tmp_path, {"voc/a.xml": VOC.format(f="a.jpg"), "voc/b.xml": VOC.format(f="b.jpg")})
    _dataset(httpx_mock, [("img-a1", "a.jpg"), ("img-a2", "a.jpg"), ("img-b", "b.jpg")])
    _bulk(httpx_mock)

    res = run("annotations", "import-voc", "road-signs", "voc", "--no-create-classes")

    assert res.exit_code == ExitCode.FAILED
    assert set(_saved(httpx_mock)) == {"img-b"}
    report = json.loads(res.stdout)
    assert report["failures"][0]["image"] == "a.jpg"
    assert "2 images of the dataset are named 'a.jpg'" in report["failures"][0]["reason"]
    assert _error(res)["code"] == "some_files_failed"


def test_nothing_matching_is_not_found_with_the_report(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    _write(tmp_path, {"voc/z.xml": VOC.format(f="z.jpg")})
    _dataset(httpx_mock, [("img-a", "a.jpg")])

    res = run("annotations", "import-voc", "road-signs", "voc")

    assert res.exit_code == ExitCode.NOT_FOUND
    assert json.loads(res.stdout)["unmatched_files"] == ["z.jpg"]
    error = _error(res)
    assert error["code"] == "no_images_matched"
    assert "pictograph images upload" in error["fix"]


# ───────────── COCO ─────────────

COCO = {
    "images": [{"id": 1, "file_name": "a.jpg"}],
    "categories": [{"id": 1, "name": "car"}],
    "annotations": [{"id": 9, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4]}],
}


@pytest.mark.parametrize("group", ["annotations", "datasets"])
def test_import_coco_from_annotations_and_its_datasets_alias(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path, group: str
) -> None:
    (tmp_path / "coco.json").write_text(json.dumps(COCO))
    _dataset(httpx_mock, [("img-a", "a.jpg")])
    _bulk(httpx_mock)

    res = run(group, "import-coco", "road-signs", "coco.json", "--no-create-classes")

    assert res.exit_code == ExitCode.OK, res.output
    assert _saved(httpx_mock)["img-a"][0]["bounding_box"] == {"x": 1, "y": 2, "w": 3, "h": 4}
    assert json.loads(res.stdout)["annotations_saved"] == 1


def test_import_coco_names_a_missing_file(run: Any, httpx_mock: HTTPXMock) -> None:
    res = run("annotations", "import-coco", "road-signs", "missing.json")
    assert res.exit_code == ExitCode.INVALID
    assert "No such file: missing.json" in res.output


# ───────────── reading data.yaml without PyYAML ─────────────


@pytest.mark.parametrize(
    "text",
    [
        "names: ['car', \"traffic light\"]  # two\n",
        "nc: 2\nnames:\n  - car\n  - 'traffic light'\nval: x\n",
        "names:\n- car\n- traffic light\n",
        "names:\n  1: traffic light\n  0: car  # first\n",
    ],
)
def test_data_yaml_names_are_read_without_pyyaml(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    real_import = builtins.__import__

    def no_yaml(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_yaml)
    monkeypatch.delitem(__import__("sys").modules, "yaml", raising=False)
    assert _label_files.yaml_names(text) == ["car", "traffic light"]


def test_data_yaml_names_with_pyyaml() -> None:
    pytest.importorskip("yaml")
    assert _label_files.yaml_names("names:\n  0: car\n  1: bus\n") == ["car", "bus"]
    assert _label_files.yaml_names("nc: 2\n") is None
