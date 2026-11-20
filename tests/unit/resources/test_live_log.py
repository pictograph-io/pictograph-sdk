"""The Live log in the SDK and the CLI: ``client.deployments.logs`` / ``client.workflows.logs`` and
``pictograph deployments logs`` / ``pictograph workflows logs``.

The answers are recorded in the API's exact page shape (``records``, ``next_cursor``, the window it resolved,
``frames_logged``), each record in the one record shape, with annotations as the log's engines write them (a box's
track, a keypoint's ``instance_id``, a segmentation polygon's bounding box, a classifier's tags). The CLI tests drive
a real ``Client`` over the same recorded answers - never a mocked SDK.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from pictograph import Client
from pictograph.exceptions import ApiError, ForbiddenError, NotFoundError
from pictograph.models.annotation import BBoxAnnotation, KeypointAnnotation, PolygonAnnotation
from pictograph.models.live_log import LogRecord
from pictograph.models.webhook import WebhookEndpoint, WebhookFilters
from pictograph.resources._live_log import LIVE_NOT_YET, PAGE_SIZE

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
DEP_LOGS = f"{BASE}/api/v1/developer/deployments/loading-dock/logs"
WF_LOGS = f"{BASE}/api/v1/developer/workflows/crosswalk/logs"
STREAM = "0123456789abcdef"
API_DEP = re.compile(re.escape(DEP_LOGS) + r"(\?.*)?$")
API_WF = re.compile(re.escape(WF_LOGS) + r"(\?.*)?$")
STORAGE = "https://storage.example.com/bucket/org/prt/sessions/s-1/cam/passive-1/ev-1/a.jpg?X-Goog-Signature=abc"


def _record(kind: str, index: int, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "cursor": f"cursor-{kind}-{index}",
        "kind": kind,
        "time": f"2026-11-20T18:22:{10 + index:02d}.099326Z",
        "logged_at": f"2026-11-20T18:22:{10 + index:02d}.403007Z",
        "deployment": "loading-dock",
        "workflow": None,
        "session": STREAM,
        "run": STREAM,
        "seq": 48,
        "index": index,
        "source": {"device": "edge", "kind": "stream", "name": STREAM},
        "model": {
            "name": "Street detector",
            "task": "object_detection",
            "version": 3,
            "precision": "fp16",
        },
        "frame": {
            "number": 380 + index,
            "width": 1280,
            "height": 720,
            "media_time": 51.9,
            "latency_ms": 148.4,
        },
        "annotations": [],
        "tags": [],
        "event": None,
        "capture": None,
        "stats": None,
        "status": None,
        "gap": None,
    }
    base.update(over)
    return base


BOX = {
    "attributes": {"track_id": "248"},
    "bounding_box": {"h": 132.1, "w": 186.72, "x": 1.41, "y": 585.04},
    "confidence": 0.9053,
    "id": "4870f649-826d-4913-b9db-6915276ded50",
    "name": "car",
    "type": "bbox",
}
JOINT = {
    "confidence": 0.585,
    "id": "ec56c8ba-35e7-48a6-8574-3bb4318d4b12",
    "instance_id": 1,
    "keypoint": {"x": 130.86, "y": 576.87},
    "name": "left_eye",
    "type": "keypoint",
}
REGION = {
    "bounding_box": {"h": 2, "w": 2, "x": 898, "y": 348},
    "confidence": 0.5171,
    "id": "6bb47130-360a-44bf-bc23-f8fda36e07d4",
    "name": "bottle",
    "polygon": {"paths": [[{"x": 898, "y": 348}, {"x": 898, "y": 349}, {"x": 899, "y": 349}]]},
    "type": "polygon",
}

PAGE_1 = {
    "deployment": "loading-dock",
    "records": [
        _record("frame", 0, annotations=[BOX, JOINT, REGION]),
        _record(
            "capture",
            1,
            annotations=[BOX],
            capture={
                "file": "a.jpg",
                "url": STORAGE,
                "trigger": "passive-a7f1d7bc",
                "event": "ev-1",
                "image": None,
                "dataset": None,
                "directory": None,
            },
        ),
    ],
    "next_cursor": "cursor-capture-1",
    "since": "2026-11-20T17:22:00.000000Z",
    "until": "2026-11-20T18:22:00.000000Z",
    "frames_logged": True,
}
PAGE_2 = {
    "deployment": "loading-dock",
    "records": [
        _record(
            "stats",
            2,
            model=None,
            frame=None,
            stats={
                "frames": 375,
                "inferred": 375,
                "detections": 2979,
                "events": 0,
                "fps": 10.0,
                "media_time": 50.9,
                "latency_ms": {
                    "total": {"p50": 116.0, "p90": 162.0, "p99": 234.8, "max": 331.9, "n": 375}
                },
                "measures": None,
            },
        ),
        _record(
            "frame",
            3,
            model={
                "name": "PG-COCO-ResNet50",
                "task": "classification",
                "version": None,
                "precision": "fp16",
            },
            tags=[{"name": "bus", "confidence": 0.9731}],
        ),
        _record(
            "status",
            4,
            frame={
                "number": 9,
                "width": None,
                "height": None,
                "media_time": 1.5,
                "latency_ms": None,
            },
            status={"state": "ended", "reason": None},
        ),
    ],
    "next_cursor": None,
    "since": "2026-11-20T17:22:00.000000Z",
    "until": "2026-11-20T18:22:00.000000Z",
    "frames_logged": True,
}


@pytest.fixture
def client() -> Any:
    c = Client(api_key=KEY, base_url=BASE, max_retries=0)
    yield c
    c.close()


def _params(request: Any) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for k, v in request.url.params.multi_items():
        out.setdefault(k, []).append(v)
    return out


# ------------------------------------------------------------------------------------------------------- reading


def test_logs_reads_every_page_for_you_over_the_window_the_first_page_resolved(
    httpx_mock: HTTPXMock, client: Any
) -> None:
    httpx_mock.add_response(url=API_DEP, json=PAGE_1)
    httpx_mock.add_response(url=API_DEP, json=PAGE_2)
    records = list(
        client.deployments.logs(
            "loading-dock",
            since="1h",
            kinds=["frame", "capture", "stats", "status"],
            classes=["car"],
            min_confidence=0.5,
            session=STREAM,
            every=2,
        )
    )
    assert [r.kind for r in records] == ["frame", "capture", "stats", "frame", "status"]
    first, second = httpx_mock.get_requests()
    assert str(first.url).split("?")[0] == DEP_LOGS
    assert _params(first) == {
        "limit": [str(PAGE_SIZE)],
        "since": ["1h"],
        "kinds": ["frame", "capture", "stats", "status"],
        "classes": ["car"],
        "min_confidence": ["0.5"],
        "session": [STREAM],
        "every": ["2"],
    }
    p2 = _params(second)
    assert p2["cursor"] == ["cursor-capture-1"]
    # The relative `since` the first page resolved is the one every later page reads: the window never drifts.
    assert p2["since"] == ["2026-11-20T17:22:00Z"] and p2["until"] == ["2026-11-20T18:22:00Z"]
    assert first.headers["x-api-key"] == KEY


def test_a_record_carries_the_sdks_own_annotation_types_and_every_kinds_fields(
    httpx_mock: HTTPXMock, client: Any
) -> None:
    httpx_mock.add_response(url=API_DEP, json=PAGE_1)
    httpx_mock.add_response(url=API_DEP, json=PAGE_2)
    frame, capture, stats, tagged, status = client.deployments.logs("loading-dock")
    assert isinstance(frame, LogRecord)
    box, joint, region = frame.annotations
    assert isinstance(box, BBoxAnnotation) and box.name == "car" and box.confidence == 0.9053
    assert isinstance(joint, KeypointAnnotation) and joint.instance_id == 1
    assert isinstance(region, PolygonAnnotation) and region.bounding_box is not None
    assert frame.time == datetime(2026, 11, 20, 18, 22, 10, 99326, tzinfo=timezone.utc)
    assert frame.model is not None and frame.model.version == 3 and frame.model.precision == "fp16"
    assert frame.frame is not None and (frame.frame.width, frame.frame.number) == (1280, 380)
    assert frame.source is not None and frame.source.device == "edge"
    assert capture.capture is not None and capture.capture.url == STORAGE
    assert (
        stats.stats is not None and stats.stats.fps == 10.0 and stats.stats.latency_ms is not None
    )
    assert stats.stats.latency_ms["total"]["p50"] == 116.0
    assert [t.name for t in tagged.tags] == ["bus"] and tagged.annotations == []
    assert status.status is not None and status.status.state == "ended"


def test_a_workflows_log_is_read_by_its_name(httpx_mock: HTTPXMock, client: Any) -> None:
    httpx_mock.add_response(url=API_WF, json={**PAGE_2, "workflow": "crosswalk"})
    assert len(list(client.workflows.logs("crosswalk", kinds=["event"], since="10m"))) == 3
    (request,) = httpx_mock.get_requests()
    assert str(request.url).split("?")[0] == WF_LOGS


def test_a_name_is_escaped_into_its_path(httpx_mock: HTTPXMock, client: Any) -> None:
    httpx_mock.add_response(json=PAGE_2)
    list(client.deployments.logs("dock / north"))
    (request,) = httpx_mock.get_requests()
    assert request.url.raw_path.startswith(b"/api/v1/developer/deployments/dock%20%2F%20north/logs")


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"kinds": ["frames"]}, "kinds are frame, event, capture, stats, status"),
        ({"min_confidence": 1.5}, "min_confidence is from 0 to 1"),
        ({"every": 0}, "every is a whole number"),
        ({"since": "yesterday"}, "since is `now`, a duration ago"),
        ({"until": ""}, "until is `now`"),
        ({"classes": [""]}, "classes is a list of names"),
        ({"session": "  "}, "session is a stream's id"),
    ],
)
def test_an_argument_that_cannot_be_read_is_refused_when_called_before_any_request(
    httpx_mock: HTTPXMock, client: Any, kwargs: dict[str, Any], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        client.deployments.logs("loading-dock", **kwargs)  # never iterated: the call itself refuses
    assert httpx_mock.get_requests() == []


def test_live_is_said_to_be_not_yet_and_nothing_is_asked(
    httpx_mock: HTTPXMock, client: Any
) -> None:
    with pytest.raises(NotImplementedError) as e:
        client.workflows.logs("crosswalk", live=True)
    assert str(e.value) == LIVE_NOT_YET
    assert httpx_mock.get_requests() == []


def test_times_are_sent_as_the_api_reads_them(httpx_mock: HTTPXMock, client: Any) -> None:
    httpx_mock.add_response(url=API_DEP, json=PAGE_2)
    httpx_mock.add_response(url=API_DEP, json=PAGE_2)
    list(
        client.deployments.logs(
            "loading-dock", since=timedelta(minutes=10), until=datetime(2026, 11, 20, 18)
        )
    )
    list(client.deployments.logs("loading-dock", since="2026-11-20T10:00:00-08:00", classes="car"))
    a, b = httpx_mock.get_requests()
    assert _params(a)["since"] == ["600s"] and _params(a)["until"] == ["2026-11-20T18:00:00Z"]
    assert _params(b)["since"] == ["2026-11-20T10:00:00-08:00"] and _params(b)["classes"] == ["car"]


def test_the_apis_refusals_are_typed_at_the_first_record(
    httpx_mock: HTTPXMock, client: Any
) -> None:
    httpx_mock.add_response(status_code=404, json={"detail": "Deployment 'nope' not found"})
    stream = client.deployments.logs("nope")
    with pytest.raises(NotFoundError):
        next(iter(stream))
    httpx_mock.add_response(
        status_code=403, json={"detail": {"error": "frames_not_logged", "message": "paid plans"}}
    )
    with pytest.raises(ForbiddenError):
        list(client.deployments.logs("loading-dock", kinds=["frame"]))


# ------------------------------------------------------------------------------------------------------- captures


def test_a_capture_downloads_from_its_signed_url_without_the_api_key(
    httpx_mock: HTTPXMock, client: Any, tmp_path: Path
) -> None:
    httpx_mock.add_response(url=API_DEP, json=PAGE_1)
    httpx_mock.add_response(url=STORAGE, content=b"\xff\xd8jpeg")
    capture = next(r for r in client.deployments.logs("loading-dock") if r.kind == "capture")
    assert capture.capture is not None
    saved = capture.capture.download(tmp_path)
    assert saved == tmp_path / "a.jpg" and saved.read_bytes() == b"\xff\xd8jpeg"
    storage_request = httpx_mock.get_requests(url=STORAGE)[0]
    assert "x-api-key" not in storage_request.headers, "a signed URL is its own authorization"
    assert not list(tmp_path.glob("*.part"))


def test_a_capture_saves_to_a_file_path_too_and_a_lapsed_url_says_so(
    httpx_mock: HTTPXMock, client: Any, tmp_path: Path
) -> None:
    httpx_mock.add_response(url=API_DEP, json=PAGE_1)
    capture = next(
        r for r in client.deployments.logs("loading-dock") if r.kind == "capture"
    ).capture
    assert capture is not None
    httpx_mock.add_response(url=STORAGE, content=b"img")
    assert capture.download(tmp_path / "out" / "x.jpg").read_bytes() == b"img"
    httpx_mock.add_response(url=STORAGE, status_code=403, text="<Error>ExpiredToken</Error>")
    with pytest.raises(ApiError, match="lapsed"):
        capture.download(tmp_path / "y.jpg")
    assert not (tmp_path / "y.jpg").exists() and not (tmp_path / "y.jpg.part").exists()


# ------------------------------------------------------------------------------------------------------- webhooks


def test_webhook_filters_are_sent_checked_and_read_back(httpx_mock: HTTPXMock, client: Any) -> None:
    endpoint = {
        "id": "ep-1",
        "organization_id": "org",
        "url": "https://hooks.example.com/dock",
        "event_types": ["deployment.frames"],
        "enabled": True,
        "filters": {
            "deployments": ["loading-dock"],
            "classes": ["forklift"],
            "min_confidence": 0.6,
        },
    }
    httpx_mock.add_response(json={"endpoint": endpoint, "secret": "whsec_x", "message": "m"})
    created = client.webhooks.create(
        "https://hooks.example.com/dock",
        event_types=["deployment.frames"],
        filters={"deployments": ["loading-dock"], "classes": ["forklift"], "min_confidence": 0.6},
    )
    assert isinstance(created.endpoint, WebhookEndpoint)
    assert created.endpoint.filters == WebhookFilters(
        deployments=["loading-dock"], classes=["forklift"], min_confidence=0.6
    )
    body = json.loads(httpx_mock.get_requests()[0].content)
    assert body["filters"] == {
        "deployments": ["loading-dock"],
        "classes": ["forklift"],
        "min_confidence": 0.6,
    }
    for bad, msg in [
        ({"colour": "red"}, "filters take"),
        ({"min_confidence": 2}, "from 0 to 1"),
        ({"every": 0}, "1 or more"),
    ]:
        with pytest.raises(ValueError, match=msg):
            client.webhooks.create("https://hooks.example.com/x", filters=bad)
    assert len(httpx_mock.get_requests()) == 1, "refused before any request"


# ------------------------------------------------------------------------------------------------------- the CLI


@pytest.fixture
def cli(client: Any) -> Any:
    from pictograph.cli.commands import deployments, workflows

    runner = CliRunner()
    with (
        patch.object(deployments, "get_client", return_value=client),
        patch.object(workflows, "get_client", return_value=client),
    ):
        yield runner, deployments.app, workflows.app


def test_the_cli_prints_one_record_a_line_for_jq(httpx_mock: HTTPXMock, cli: Any) -> None:
    runner, dep_app, _ = cli
    httpx_mock.add_response(url=API_DEP, json=PAGE_1)
    httpx_mock.add_response(url=API_DEP, json=PAGE_2)
    res = runner.invoke(
        dep_app,
        [
            "logs",
            "loading-dock",
            "--json",
            "--kind",
            "frame,capture",
            "--kind",
            "stats",
            "--class",
            "car",
            "--since",
            "2h",
            "--every",
            "3",
        ],
    )
    assert res.exit_code == 0, res.output
    lines = [json.loads(line) for line in res.stdout.splitlines() if line.strip()]
    assert [x["kind"] for x in lines] == ["frame", "capture", "stats", "frame", "status"], (
        "every record"
    )
    assert lines[0]["annotations"][0]["name"] == "car" and lines[0]["cursor"] == "cursor-frame-0"
    p = _params(httpx_mock.get_requests()[0])
    assert (
        p["kinds"] == ["frame", "capture", "stats"] and p["since"] == ["2h"] and p["every"] == ["3"]
    )


def test_the_cli_prints_a_table_by_default_and_stops_at_its_limit(
    httpx_mock: HTTPXMock, cli: Any
) -> None:
    runner, _, wf_app = cli
    httpx_mock.add_response(url=API_WF, json={**PAGE_1, "workflow": "crosswalk"})
    res = runner.invoke(wf_app, ["logs", "crosswalk", "--limit", "2"])
    assert res.exit_code == 0, res.output
    assert "Live log of crosswalk" in res.stdout and "car" in res.stdout and "a.jpg" in res.stdout
    assert "The first 2 records" in res.stdout, "it says where it stopped"
    assert len(httpx_mock.get_requests()) == 1, "two records needed no second page"


def test_the_cli_downloads_each_capture(httpx_mock: HTTPXMock, cli: Any, tmp_path: Path) -> None:
    runner, dep_app, _ = cli
    httpx_mock.add_response(url=API_DEP, json=PAGE_1)
    httpx_mock.add_response(url=API_DEP, json=PAGE_2)
    httpx_mock.add_response(url=STORAGE, content=b"jpeg-bytes")
    res = runner.invoke(dep_app, ["logs", "loading-dock", "--download", str(tmp_path / "caps")])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "caps" / "a.jpg").read_bytes() == b"jpeg-bytes"


@pytest.mark.parametrize("flag", ["--live", "--follow", "-f"])
def test_the_cli_says_a_live_connection_is_not_yet_whichever_flag_asks(
    httpx_mock: HTTPXMock, cli: Any, flag: str
) -> None:
    runner, dep_app, _ = cli
    res = runner.invoke(dep_app, ["logs", "loading-dock", flag])
    assert res.exit_code != 0
    assert "not available yet" in res.output
    assert "--since and --until" in res.output, "the CLI's own flags, never the SDK's since="
    assert httpx_mock.get_requests() == []


def test_the_cli_refuses_an_argument_the_sdk_cannot_read(httpx_mock: HTTPXMock, cli: Any) -> None:
    runner, dep_app, _ = cli
    res = runner.invoke(dep_app, ["logs", "loading-dock", "--kind", "frames"])
    assert res.exit_code != 0 and "kinds are frame" in res.output
    assert httpx_mock.get_requests() == []


def test_the_cli_says_what_the_api_refused_with_its_exit_code(
    httpx_mock: HTTPXMock, cli: Any
) -> None:
    from pictograph.cli._errors import ExitCode

    runner, dep_app, wf_app = cli
    httpx_mock.add_response(
        url=API_DEP, status_code=404, json={"detail": "Deployment 'loading-dock' not found"}
    )
    res = runner.invoke(dep_app, ["logs", "loading-dock"])
    assert res.exit_code == ExitCode.NOT_FOUND, res.output
    assert "not found" in res.output.lower()
    httpx_mock.add_response(
        url=API_WF,
        status_code=403,
        json={"detail": {"error": "frames_not_logged", "message": "This plan logs no frames."}},
    )
    res = runner.invoke(wf_app, ["logs", "crosswalk", "--kind", "frame", "--json"])
    assert res.exit_code == ExitCode.FORBIDDEN, res.output
    assert "logs no frames" in res.output


EP_ID = "3c5e9f12-0000-4000-8000-000000000001"


def test_the_cli_takes_a_webhooks_filters_checked_before_any_request(
    httpx_mock: HTTPXMock, client: Any, tmp_path: Path
) -> None:
    from pictograph.cli.commands import webhooks

    endpoint = {
        "id": EP_ID,
        "organization_id": "org",
        "url": "https://hooks.example.com/dock",
        "event_types": ["deployment.frames"],
        "enabled": True,
        "filters": {"deployments": ["loading-dock"], "classes": ["forklift"]},
    }
    runner = CliRunner()
    with patch.object(webhooks, "get_client", return_value=client):
        httpx_mock.add_response(json={"endpoint": endpoint, "secret": "whsec_x", "message": "m"})
        res = runner.invoke(
            webhooks.app,
            [
                "create",
                "https://hooks.example.com/dock",
                "-e",
                "deployment.frames",
                "--filters",
                '{"deployments": ["loading-dock"], "classes": ["forklift"]}',
            ],
        )
        assert res.exit_code == 0, res.output
        body = json.loads(httpx_mock.get_requests()[0].content)
        assert body["filters"] == {"deployments": ["loading-dock"], "classes": ["forklift"]}
        assert '"forklift"' in res.stdout, "create shows the endpoint's filters"
        saved = tmp_path / "filters.json"
        saved.write_text('{"min_confidence": 0.6, "every": 5}')
        httpx_mock.add_response(json={"endpoint": {**endpoint, "filters": None}})
        res = runner.invoke(webhooks.app, ["update", EP_ID, "--filters", f"@{saved}"])
        assert res.exit_code == 0, res.output
        sent = json.loads(httpx_mock.get_requests()[1].content)
        assert sent["filters"] == {"min_confidence": 0.6, "every": 5}
        for bad, said in [
            ("{not json", "not JSON"),
            ("[1]", "a JSON object"),
            ('{"colour": 1}', "filters take"),
            ('{"min_confidence": 3}', "from 0 to 1"),
            ("@/no/such/file.json", "could not read"),
        ]:
            res = runner.invoke(webhooks.app, ["update", EP_ID, "--filters", bad])
            assert res.exit_code != 0 and said in res.output, (bad, res.output)
    assert len(httpx_mock.get_requests()) == 2, "a refused --filters sends nothing"


def test_a_count_stays_a_whole_number(httpx_mock: HTTPXMock, client: Any) -> None:
    httpx_mock.add_response(url=API_DEP, json=PAGE_2)
    stats = next(r for r in client.deployments.logs(deployment="loading-dock") if r.kind == "stats")
    assert stats.stats is not None and stats.stats.latency_ms is not None
    total = stats.stats.latency_ms["total"]
    assert total["n"] == 375 and isinstance(total["n"], int) and isinstance(total["p50"], float)
    event = LogRecord.model_validate(
        {"kind": "event", "event": {"values": {"total": 2, "rate": 0.5}, "text": "2 crossed"}}
    )
    assert event.event is not None and event.event.values == {"total": 2, "rate": 0.5}
    assert isinstance(event.event.values["total"], int)
    assert json.loads(stats.model_dump_json())["stats"]["latency_ms"]["total"]["n"] == 375
