"""``pictograph classes`` - a dataset's ontology from the CLI (B533 P1).

Before it, the CLI could rename, merge and delete a class but not list one, add one or
set its type, colour, attributes or keypoint skeleton. Every write here is a
read-modify-write that sends what it read as ``classes_base``, so the platform MERGES
the edit onto the list stored now (B529): a class someone added in the app since the
read survives. These drive the REAL app and the REAL client over a mocked HTTP
transport and read the request bodies on the wire.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from pictograph import Client
from pictograph.cli._app import app
from pictograph.cli._errors import ExitCode
from pictograph.models.dataset import DatasetClass, SkeletonTemplate

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
DATASET_URL = f"{BASE}/api/v1/developer/datasets/road-signs"
NOW = "2026-04-01T00:00:00Z"
CAR = {"name": "car", "type": "box", "color": "#e6194b"}
SIGN = {
    "name": "sign",
    "type": "polygon",
    "color": "#3cb44b",
    "attributes": [{"name": "occluded", "type": "checkbox"}],
}
PERSON = {
    "name": "person",
    "type": "keypoint",
    "color": "#4363d8",
    "skeleton": {
        "nodes": [{"name": "nose", "x": 0.5, "y": 0.1}, {"name": "neck", "x": 0.5, "y": 0.25}],
        "edges": [[0, 1]],
    },
}


def _dataset(classes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "data": {
            "id": "proj-1",
            "name": "road-signs",
            "image_count": 0,
            "created_at": NOW,
            "classes": classes,
        }
    }


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
        return runner.invoke(app, list(args), input=stdin)

    return invoke


def _read(httpx_mock: HTTPXMock, classes: list[dict[str, Any]]) -> None:
    httpx_mock.add_response(method="GET", url=DATASET_URL, json=_dataset(classes))


def _answer(httpx_mock: HTTPXMock, classes: list[dict[str, Any]]) -> None:
    httpx_mock.add_response(method="PATCH", url=DATASET_URL, json=_dataset(classes))


def _patch_body(httpx_mock: HTTPXMock) -> dict[str, Any]:
    (request,) = [r for r in httpx_mock.get_requests() if r.method == "PATCH"]
    body: dict[str, Any] = json.loads(request.content)
    return body


# ───────────── list ─────────────


def test_list_shows_every_class_with_its_type_and_joints(run: Any, httpx_mock: HTTPXMock) -> None:
    _read(httpx_mock, [CAR, SIGN, PERSON])
    res = run("classes", "list", "road-signs", "--json")
    assert res.exit_code == ExitCode.OK, res.output
    listed = json.loads(res.stdout)
    assert [c["name"] for c in listed] == ["car", "sign", "person"]
    # The skeleton survives the read - it was dropped by the SDK model before.
    assert listed[2]["skeleton"] == PERSON["skeleton"]
    assert listed[1]["attributes"] == [{"name": "occluded", "type": "checkbox"}]

    _read(httpx_mock, [CAR, SIGN, PERSON])
    table = run("classes", "list", "road-signs")
    assert "Classes of road-signs (3)" in table.stdout


# ───────────── add ─────────────


def test_add_sends_the_list_it_read_as_the_base(run: Any, httpx_mock: HTTPXMock) -> None:
    _read(httpx_mock, [CAR, PERSON])
    bus = {"name": "bus", "type": "box", "color": "#ffe119"}
    truck = {"name": "truck", "type": "box", "color": "#911eb4"}  # another writer's
    _answer(httpx_mock, [CAR, PERSON, truck, bus])

    res = run("classes", "add", "road-signs", "bus", "--type", "bbox")

    assert res.exit_code == ExitCode.OK, res.output
    body = _patch_body(httpx_mock)
    assert body["classes_base"] == [CAR, PERSON], "the base is exactly what was read"
    assert body["classes"] == [CAR, PERSON, {"name": "bus", "type": "box"}]
    assert json.loads(res.stdout) == bus


def test_add_a_keypoint_class_with_its_skeleton_and_attributes(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    (tmp_path / "skeleton.json").write_text(json.dumps(PERSON["skeleton"]))
    (tmp_path / "attrs.json").write_text(json.dumps([{"name": "pose", "type": "text"}]))
    _read(httpx_mock, [CAR])
    _answer(httpx_mock, [CAR, PERSON])

    res = run(
        "classes",
        "add",
        "road-signs",
        "person",
        "--type",
        "keypoint",
        "--color",
        "#4363d8",
        "--skeleton",
        "skeleton.json",
        "--attributes",
        "attrs.json",
    )

    assert res.exit_code == ExitCode.OK, res.output
    (added,) = [c for c in _patch_body(httpx_mock)["classes"] if c["name"] == "person"]
    assert added == {
        "name": "person",
        "type": "keypoint",
        "color": "#4363d8",
        "attributes": [{"name": "pose", "type": "text"}],
        "skeleton": PERSON["skeleton"],
    }


def test_adding_a_class_that_exists_is_a_conflict(run: Any, httpx_mock: HTTPXMock) -> None:
    _read(httpx_mock, [CAR])
    res = run("classes", "add", "road-signs", "car", "--type", "box")
    assert res.exit_code == ExitCode.CONFLICT
    assert "already has the box class 'car'" in res.output
    assert not [r for r in httpx_mock.get_requests() if r.method == "PATCH"]


def test_the_same_name_with_another_type_is_another_class(run: Any, httpx_mock: HTTPXMock) -> None:
    _read(httpx_mock, [CAR])
    car_polygon = {"name": "car", "type": "polygon", "color": "#f58231"}
    _answer(httpx_mock, [CAR, car_polygon])
    res = run("classes", "add", "road-signs", "car", "--type", "polygon")
    assert res.exit_code == ExitCode.OK, res.output
    assert json.loads(res.stdout) == car_polygon


@pytest.mark.parametrize(
    ("args", "files", "why"),
    [
        (("--type", "mask"), {}, "use one of box, polygon, polyline, keypoint, tag"),
        (
            ("--type", "box", "--skeleton", "s.json"),
            {"s.json": PERSON["skeleton"]},
            "only a keypoint class",
        ),
        (
            ("--type", "keypoint", "--skeleton", "s.json"),
            {"s.json": {"nodes": [{"name": "a", "x": 0, "y": 0}], "edges": [[0, 1]]}},
            "each edge is two different joint indexes",
        ),
        (
            ("--type", "keypoint", "--skeleton", "s.json"),
            {"s.json": {"nodes": [{"name": "a", "x": 0, "y": 0}, {"name": "a", "x": 1, "y": 1}]}},
            "joint names must be unique",
        ),
        (("--type", "box", "--attributes", "nope.json"), {}, "No such file"),
    ],
)
def test_a_wrong_class_is_refused_before_any_request(
    run: Any,
    httpx_mock: HTTPXMock,
    tmp_path: Path,
    args: tuple[str, ...],
    files: dict[str, Any],
    why: str,
) -> None:
    for name, content in files.items():
        (tmp_path / name).write_text(json.dumps(content))
    res = run("classes", "add", "road-signs", "x", *args)
    assert res.exit_code == ExitCode.INVALID
    assert why in " ".join(res.output.replace("│", " ").split())
    assert httpx_mock.get_requests() == []


# ───────────── update ─────────────


def test_update_changes_only_what_it_was_given(run: Any, httpx_mock: HTTPXMock) -> None:
    _read(httpx_mock, [CAR, SIGN])
    recoloured = {**SIGN, "color": "#000075"}
    _answer(httpx_mock, [CAR, recoloured])

    res = run("classes", "update", "road-signs", "sign", "--color", "#000075")

    assert res.exit_code == ExitCode.OK, res.output
    body = _patch_body(httpx_mock)
    assert body["classes_base"] == [CAR, SIGN]
    assert body["classes"] == [CAR, recoloured], "attributes kept, colour changed"
    assert json.loads(res.stdout)["color"] == "#000075"


def test_update_clears_attributes_and_replaces_a_skeleton(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    template = {
        "nodes": [{"name": "nose", "x": 0.5, "y": 0.1}, {"name": "tail", "x": 0.5, "y": 0.9}],
        "edges": [[0, 1]],
    }
    (tmp_path / "t.json").write_text(json.dumps(template))
    _read(httpx_mock, [SIGN, PERSON])
    _answer(httpx_mock, [{**SIGN, "attributes": []}, {**PERSON, "skeleton": template}])

    cleared = run("classes", "update", "road-signs", "sign", "--clear-attributes")
    assert cleared.exit_code == ExitCode.OK, cleared.output
    assert _patch_body(httpx_mock)["classes"][0]["attributes"] == []

    httpx_mock.reset()
    _read(httpx_mock, [SIGN, PERSON])
    _answer(httpx_mock, [SIGN, {**PERSON, "skeleton": template}])
    res = run("classes", "update", "road-signs", "person", "--skeleton", "t.json")
    assert res.exit_code == ExitCode.OK, res.output
    assert _patch_body(httpx_mock)["classes"][1]["skeleton"] == template


def test_update_needs_a_type_when_the_name_has_several(run: Any, httpx_mock: HTTPXMock) -> None:
    _read(httpx_mock, [CAR, {"name": "car", "type": "polygon", "color": "#f58231"}])
    res = run("classes", "update", "road-signs", "car", "--color", "#000000")
    assert res.exit_code == ExitCode.INVALID
    assert "say which with --type" in " ".join(res.output.replace("│", " ").split())


def test_update_of_an_unknown_class_is_not_found(
    run: Any, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PICTOGRAPH_OUTPUT", "json")
    _read(httpx_mock, [CAR])
    res = run("classes", "update", "road-signs", "bus", "--color", "#000000")
    assert res.exit_code == ExitCode.NOT_FOUND
    error = json.loads(res.stderr.strip().splitlines()[-1])["error"]
    assert error["code"] == "class_not_found"


def test_update_with_nothing_to_change_is_refused(run: Any, httpx_mock: HTTPXMock) -> None:
    res = run("classes", "update", "road-signs", "car")
    assert res.exit_code == ExitCode.INVALID
    assert httpx_mock.get_requests() == []


# ───────────── remove ─────────────


def test_remove_narrows_by_the_stored_type_token(run: Any, httpx_mock: HTTPXMock) -> None:
    """`--type bbox` reaches the box class: the ontology stores `box` (a raw `bbox`
    matched nothing - the existing `annotations delete-class --type bbox` did that)."""
    _read(httpx_mock, [CAR])
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE}/api/v1/developer/annotations/delete-class",
        json={
            "dataset_id": "proj-1",
            "name": "car",
            "config_updated": True,
            "images_updated": 0,
            "annotations_removed": 0,
        },
    )
    res = run("classes", "remove", "road-signs", "car", "--type", "bbox", "--yes")
    assert res.exit_code == ExitCode.OK, res.output
    (request,) = [r for r in httpx_mock.get_requests() if r.method == "POST"]
    sent = json.loads(request.content)
    assert sent["class_type"] == "box" and sent["name"] == "car"


# ───────────── datasets update --readme ─────────────


def test_datasets_update_takes_a_readme_file_or_stdin(
    run: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    (tmp_path / "README.md").write_text("# Road signs\n\nStop signs and more.\n")
    _answer(httpx_mock, [CAR])
    res = run("datasets", "update", "road-signs", "--readme", "README.md")
    assert res.exit_code == ExitCode.OK, res.output
    assert _patch_body(httpx_mock) == {"readme": "# Road signs\n\nStop signs and more.\n"}

    httpx_mock.reset()
    _answer(httpx_mock, [CAR])
    piped = run("datasets", "update", "road-signs", "--readme", "-", stdin="from stdin\n")
    assert piped.exit_code == ExitCode.OK, piped.output
    assert _patch_body(httpx_mock) == {"readme": "from stdin\n"}


# ───────────── the SDK ─────────────


def test_a_class_round_trip_through_the_sdk_keeps_its_skeleton(httpx_mock: HTTPXMock) -> None:
    """`datasets.get(...).classes` back into `datasets.update(classes=...)`: the
    template rides along (the SDK model dropped it, so a full replace erased it)."""
    _read(httpx_mock, [PERSON])
    _answer(httpx_mock, [PERSON])
    with Client(api_key="pk_live_test", base_url=BASE, max_retries=0) as client:
        read = client.datasets.get("road-signs").classes or []
        assert isinstance(read[0].skeleton, SkeletonTemplate)
        client.datasets.update("road-signs", classes=read, classes_base=read)
    body = _patch_body(httpx_mock)
    assert body["classes"] == [PERSON] and body["classes_base"] == [PERSON]


def test_classes_base_without_classes_is_refused() -> None:
    with Client(api_key="pk_live_test", base_url=BASE, max_retries=0) as client:
        with pytest.raises(ValueError, match="pass classes too"):
            client.datasets.update("road-signs", classes_base=[DatasetClass(name="car")])
