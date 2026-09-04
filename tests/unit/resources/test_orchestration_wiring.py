"""Every relocated operation is CALLED, through a real Client, over a fake wire.

``hasattr(Images, "upload_from_directory")`` passes even when the first line of the
body raises ``ImportError``. That is the exact failure mode this move can produce:
each relocated method reaches a sibling resource through a function-level import
(``from pictograph.resources.datasets import Datasets``) constructed off
``self._transport``, and a wrong dotted path, a circular import, or a sibling built
on the wrong transport is invisible until the method actually runs.

So nothing here is mocked at the resource level. A real :class:`Client` is built,
its real ``Datasets`` / ``Images`` / ``Annotations`` / ``Exports`` / ``Models`` are
constructed by the real code, and the ONLY substitution is the HTTP boundary -
``Transport.request`` / ``.stream_bytes`` / ``.upload_external``. The assertions are
on the requests that reached that boundary, so a method that silently did nothing
fails just as loudly as one that raised.

The companion ``test_resource_orchestration_symmetry.py`` pins the shape of the
surface; this pins that the surface works.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from pictograph import Client

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

BASE = "https://api.test.local"
KEY = "pk_live_" + "0" * 32


class FakeWire:
    """Stands in for the transport's three egress points and records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.uploads: list[str] = []
        self.routes: dict[tuple[str, str], Any] = {}

    # ── the three transport egress points ──

    def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        **_: Any,
    ) -> Any:
        self.calls.append((method, path, {"json": json, "params": params}))
        for (route_method, route_prefix), payload in self.routes.items():
            if method == route_method and path.startswith(route_prefix):
                return payload(json, params) if callable(payload) else payload
        raise AssertionError(f"unrouted {method} {path} - the test must declare it")

    def stream_bytes(self, _method: str, path: str, **_: Any) -> Iterator[bytes]:
        self.calls.append(("GET-stream", path, {}))
        yield _PNG_1X1

    def upload_external(self, url: str, _path: Any, **_: Any) -> None:
        self.uploads.append(url)

    # ── helpers ──

    def route(self, method: str, prefix: str, payload: Any) -> None:
        self.routes[(method, prefix)] = payload

    def paths(self, method: str | None = None) -> list[str]:
        return [p for m, p, _ in self.calls if method is None or m == method]

    def bodies(self, path_prefix: str) -> list[dict[str, Any]]:
        return [c["json"] for _m, p, c in self.calls if p.startswith(path_prefix) and c["json"]]


def _png(width: int = 100, height: int = 100) -> bytes:
    """A REAL decodable PNG - ``Images.upload`` reads dimensions with Pillow and
    ``augment`` / ``tile`` open the downloaded bytes, so a fake byte string would
    fail for the wrong reason (or, worse, be caught and reported as a per-image
    failure while the test still asserted a count)."""
    import io

    from PIL import Image as _PIL

    buf = io.BytesIO()
    _PIL.new("RGB", (width, height), (120, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


_PNG_1X1 = _png()


def _dataset(name: str = "ds", classes: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        # A REAL uuid shape. The resolver detects an id by shape, so a fixture
        # id that is not uuid-shaped makes an internal hand-off look like a name
        # and issue a lookup no production call would make.
        "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "organization_id": "org-1",
        "name": name,
        "classes": classes if classes is not None else [{"name": "car", "type": "bbox"}],
        "annotation_types": ["bbox"],
        "image_count": 1,
        "created_at": "2026-01-01T00:00:00Z",
    }


def _image(
    image_id: str = "img-1", filename: str = "a.jpg", annotations: int = 0
) -> dict[str, Any]:
    return {
        "id": image_id,
        "dataset_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "filename": filename,
        "status": "new",
        "annotation_count": annotations,
        "file_size": 10,
        "width": 100,
        "height": 100,
        "image_url": "https://x/img",
        "created_at": "2026-01-01T00:00:00Z",
    }


_BBOX = {"name": "car", "type": "bbox", "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4}}


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> tuple[Client, FakeWire]:
    """A real Client whose transport egress is replaced, nothing else."""
    client = Client(api_key=KEY, base_url=BASE, max_retries=0)
    wire = FakeWire()
    monkeypatch.setattr(type(client._transport), "request", wire.request, raising=True)
    monkeypatch.setattr(type(client._transport), "stream_bytes", wire.stream_bytes, raising=True)
    monkeypatch.setattr(
        type(client._transport), "upload_external", wire.upload_external, raising=True
    )
    return client, wire


# ═════════════════════ images.upload_from_directory ═════════════════════


def test_upload_from_directory_creates_the_dataset_and_uploads_each_file(
    wired: tuple[Client, FakeWire], tmp_path: Path
) -> None:
    """The whole point of the move: this reaches ``Datasets`` off the same transport.

    A wrong function-level import here raises ``ImportError`` at call time, which
    an existence check would never see.
    """
    client, wire = wired
    (tmp_path / "cars").mkdir()
    (tmp_path / "cars" / "a.png").write_bytes(_PNG_1X1)
    (tmp_path / "b.png").write_bytes(_PNG_1X1)
    (tmp_path / "notes.txt").write_text("ignored")

    wire.route("GET", "/api/v1/developer/datasets/", {"data": _dataset()})
    wire.route(
        "POST", "/api/v1/developer/images/upload-url", {"data": {"upload_url": "https://s/1"}}
    )
    wire.route("POST", "/api/v1/developer/images/register", {"data": _image()})

    report = client.images.upload_from_directory("ds", tmp_path, parallel=False)

    assert report.images_uploaded == 2, report.failures
    assert report.images_attempted == 2  # the .txt was not attempted
    # Both signed-URL requests carried the dataset UUID the DATASETS resource returned.
    directories = sorted(
        b["directory_path"] for b in wire.bodies("/api/v1/developer/images/upload-url")
    )
    assert directories == ["/", "/cars"]
    assert {b["dataset"] for b in wire.bodies("/api/v1/developer/images/upload-url")} == {
        "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    }
    assert wire.uploads == ["https://s/1", "https://s/1"]


def test_upload_from_directory_creates_a_missing_dataset_rather_than_failing(
    wired: tuple[Client, FakeWire], tmp_path: Path
) -> None:
    """``create_if_missing`` runs through the sibling resource's POST, not a stub."""
    client, wire = wired
    (tmp_path / "a.png").write_bytes(_PNG_1X1)

    from pictograph.exceptions import NotFoundError

    def _get(_body: Any, _params: Any) -> Any:
        raise NotFoundError("no such dataset")

    wire.route("GET", "/api/v1/developer/datasets/", _get)
    wire.route("POST", "/api/v1/developer/datasets/", {"data": _dataset()})
    wire.route(
        "POST", "/api/v1/developer/images/upload-url", {"data": {"upload_url": "https://s/1"}}
    )
    wire.route("POST", "/api/v1/developer/images/register", {"data": _image()})

    report = client.images.upload_from_directory("ds", tmp_path, parallel=False)

    assert report.images_uploaded == 1, report.failures
    assert any(p == "/api/v1/developer/datasets/" for p in wire.paths("POST"))


# ═════════════════════ images.augment / images.tile ═════════════════════


def _augment_tile_routes(wire: FakeWire) -> None:
    wire.route("GET", "/api/v1/developer/datasets/", {"data": _dataset()})
    wire.route("POST", "/api/v1/developer/datasets/", {"data": _dataset(name="tgt")})
    wire.route("GET", "/api/v1/developer/images/", {"data": [_image()], "total": 1})
    wire.route("GET", "/api/v1/developer/annotations/", {"annotations": [_BBOX]})
    wire.route(
        "POST", "/api/v1/developer/images/upload-url", {"data": {"upload_url": "https://s/1"}}
    )
    wire.route("POST", "/api/v1/developer/images/register", {"data": _image()})
    wire.route(
        "POST",
        "/api/v1/developer/annotations/",
        {"image_id": "img-1", "previous_count": 0, "new_count": 1, "status": "new"},
    )


def test_augment_downloads_annotates_and_uploads_each_variant(
    wired: tuple[Client, FakeWire],
) -> None:
    """Reaches Datasets AND Annotations AND its own download/upload - three seams."""
    from pictograph.augment import HorizontalFlip

    client, wire = wired
    _augment_tile_routes(wire)

    report = client.images.augment("ds", [HorizontalFlip(p=1.0)], multiplier=2, into="tgt", seed=1)

    assert report.success, report.failures
    assert report.variants_created == 2
    assert report.originals_copied == 1  # new dataset -> original copied too
    # 1 original + 2 variants uploaded, each with its annotations saved.
    assert len(wire.uploads) == 3
    assert report.annotations_written == 3
    # The image bytes were actually streamed down before being transformed.
    assert "/api/v1/developer/images/ds/img-1" in wire.paths("GET-stream")


def test_tile_cuts_the_grid_and_uploads_every_tile(wired: tuple[Client, FakeWire]) -> None:
    client, wire = wired
    _augment_tile_routes(wire)

    report = client.images.tile("ds", rows=2, cols=2, into="tgt")

    assert report.success, report.failures
    assert report.tiles_created == 4
    assert len(wire.uploads) == 4


# ═════════════════════ auto_annotate.dataset ═════════════════════


def test_auto_annotate_dataset_submits_the_batch_and_reports_the_job(
    wired: tuple[Client, FakeWire],
) -> None:
    """Reaches Datasets for the image list, then its OWN batch method."""
    client, wire = wired
    listed = _dataset()
    listed["images"] = [_image("i1", "a.jpg"), _image("i2", "b.jpg", annotations=3)]
    wire.route("GET", "/api/v1/developer/datasets/", {"data": listed})
    wire.route(
        "POST",
        "/api/v1/developer/auto-annotate/batch",
        {
            "job_id": "job-1",
            "status": "completed",
            "total_images": 1,
            "processed_images": 1,
            "failed_images": 0,
            "total_annotations_added": 5,
        },
    )
    wire.route(
        "GET",
        "/api/v1/developer/auto-annotate/batch/",
        {
            "job_id": "job-1",
            "status": "completed",
            "total_images": 1,
            "processed_images": 1,
            "failed_images": 0,
            "total_annotations_added": 5,
        },
    )

    report = client.auto_annotate.dataset("ds", [("car", "bbox")])

    assert report.success, report.failures
    assert report.job_id == "job-1"
    assert report.annotations_added == 5
    # The already-annotated image was held back, and ONLY the eligible one submitted.
    assert report.images_skipped == 1
    body = wire.bodies("/api/v1/developer/auto-annotate/batch")[0]
    assert body["image_filenames"] == ["a.jpg"]
    assert body["classes"] == [{"name": "car", "output_type": "bbox"}]


# ═════════════════════ annotations.import_* ═════════════════════


def _import_routes(wire: FakeWire) -> None:
    wire.route("GET", "/api/v1/developer/datasets/", {"data": _dataset()})
    wire.route("PATCH", "/api/v1/developer/datasets/", {"data": _dataset()})
    wire.route("GET", "/api/v1/developer/images/", {"data": [_image()], "total": 1})
    wire.route(
        "POST",
        "/api/v1/developer/annotations/bulk",
        {
            "saved": [
                {"image_id": "img-1", "previous_count": 0, "new_count": 1, "status": "complete"}
            ],
            "failed": [],
        },
    )


def test_import_coco_resolves_by_filename_and_bulk_saves(
    wired: tuple[Client, FakeWire], tmp_path: Path
) -> None:
    """Reaches Datasets AND Images AND its own bulk_save - the deepest fan-out."""
    client, wire = wired
    _import_routes(wire)
    coco = tmp_path / "instances.json"
    coco.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
                "categories": [{"id": 1, "name": "car"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4]}],
            }
        ),
        encoding="utf-8",
    )

    report = client.annotations.import_coco("ds", coco)

    assert report.success, (report.failures, report.unmatched_files)
    assert report.images_saved == 1
    assert report.annotations_saved == 1
    saved = wire.bodies("/api/v1/developer/annotations/bulk")[0]
    assert saved["saves"][0]["image_id"] == "img-1"  # resolved through the IMAGES resource
    assert saved["saves"][0]["annotations"][0]["name"] == "car"


def test_import_pascal_voc_parses_each_xml_and_saves(wired: tuple[Client, FakeWire]) -> None:
    client, wire = wired
    _import_routes(wire)
    xml = (
        "<annotation><object><name>car</name>"
        "<bndbox><xmin>1</xmin><ymin>2</ymin><xmax>4</xmax><ymax>6</ymax></bndbox>"
        "</object></annotation>"
    )

    report = client.annotations.import_pascal_voc("ds", {"a.jpg": xml})

    assert report.success, (report.failures, report.unmatched_files)
    saved = wire.bodies("/api/v1/developer/annotations/bulk")[0]
    assert saved["saves"][0]["annotations"][0]["bounding_box"] == {"x": 1, "y": 2, "w": 3, "h": 4}


def test_import_yolo_denormalizes_using_the_dimensions_it_fetched(
    wired: tuple[Client, FakeWire],
) -> None:
    """The image's stored width/height must come from the IMAGES resource - a
    broken sibling reach would either raise or silently denormalize against None."""
    client, wire = wired
    _import_routes(wire)

    report = client.annotations.import_yolo("ds", {"a.jpg": "0 0.5 0.5 0.2 0.4\n"}, ["car"])

    assert report.success, (report.failures, report.unmatched_files)
    box = wire.bodies("/api/v1/developer/annotations/bulk")[0]["saves"][0]["annotations"][0][
        "bounding_box"
    ]
    # 100x100 image: w=0.2*100=20, h=0.4*100=40, centred at (50, 50).
    assert (box["w"], box["h"]) == (20.0, 40.0)
    assert (box["x"], box["y"]) == (40.0, 30.0)


def test_import_yolo_reports_an_unmatched_filename_instead_of_inventing_an_image(
    wired: tuple[Client, FakeWire],
) -> None:
    client, wire = wired
    _import_routes(wire)

    report = client.annotations.import_yolo("ds", {"nope.jpg": "0 0.5 0.5 0.2 0.4\n"}, ["car"])

    assert report.unmatched_files == ["nope.jpg"]
    assert report.images_saved == 0
    assert not report.success
    assert "/api/v1/developer/annotations/bulk" not in wire.paths("POST")


# ═════════════════════ training.create ═════════════════════
#
# `training.from_dataset` was REMOVED (owner, 2026-07-31): training runs off an
# EXPORT, never off a dataset. The old entry point created an export behind the
# caller's back, which could be EMPTY - and an empty export is accepted, charged,
# and only fails inside the GPU container with "has no image_ids". The two tests
# that lived here asserted that behind-the-back export creation, so they are gone
# with it rather than rewritten.
#
# `Training.create` takes an export that already exists; its wiring is covered in
# `tests/unit/resources/test_training.py`.
