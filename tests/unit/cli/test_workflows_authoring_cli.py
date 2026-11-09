"""``pictograph workflows nodes / templates / validate`` and ``create --template`` (B533 #17).

Drives the REAL app and client over a mocked HTTP transport: what each command sends, prints and
exits with - a graph that would not run is exit 6 with every issue (on stderr, and as the JSON
error's details under ``--json``), and every refusal the CLI can make itself comes before any
request.
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
WID = "fedcba98-1111-2222-3333-444455556666"
GRAPH = {
    "version": 1,
    "nodes": [{"id": "s", "type": "source", "config": {"kind": "video"}}],
    "edges": [],
}
CATALOG: dict[str, Any] = {
    "graph": {"ports": ["frames", "detections", "events"]},
    "nodes": [
        {
            "type": "source",
            "label": "Source",
            "summary": "Where the frames come from.",
            "inputs": [],
            "output": "frames",
            "new_block": {"kind": "video"},
            "rules": [],
            "fields": [
                {
                    "key": "kind",
                    "type": "string",
                    "required": True,
                    "note": "",
                    "values": ["image", "video", "dataset", "export"],
                },
                {
                    "key": "gcs_uri",
                    "type": "string",
                    "required": "kind is image or video",
                    "note": "",
                },
            ],
        },
        {
            "type": "measure",
            "label": "Measure",
            "summary": "Counts crossings.",
            "inputs": ["detections"],
            "output": "events",
            "new_block": {"geometries": []},
            "rules": ["It needs a Track block upstream."],
            "fields": [
                {"key": "geometries", "type": "array", "required": True, "note": "", "fields": []}
            ],
        },
    ],
}
TEMPLATES = {
    "templates": [
        {
            "key": "traffic",
            "title": "Traffic monitoring",
            "description": "Count vehicles.",
            "graph": GRAPH,
        },
        {
            "key": "custom",
            "title": "Custom",
            "description": "Blank.",
            "graph": {"version": 1, "nodes": [], "edges": []},
        },
    ]
}
BROKEN = {
    "valid": False,
    "issues": [
        {
            "code": "model_missing",
            "message": "Pick a model for this block.",
            "node_id": "model",
            "severity": "error",
        },
        {
            "code": "measure_no_geometry",
            "message": "Draw at least one line or zone.",
            "node_id": "measure",
            "severity": "error",
        },
    ],
}


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_test")
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    monkeypatch.setenv("PICTOGRAPH_MAX_RETRIES", "0")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.delenv("PICTOGRAPH_OUTPUT", raising=False)
    runner = CliRunner()
    return lambda *args, stdin=None: runner.invoke(app, ["workflows", *args], input=stdin)


def _last_error(stderr: str) -> dict[str, Any]:
    return json.loads(stderr.strip().splitlines()[-1])["error"]


def _graph_file(tmp_path: Path, graph: Any = GRAPH) -> str:
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph) if not isinstance(graph, str) else graph)
    return str(path)


# ── nodes ──────────────────────────────────────────────────────────────────────


def test_nodes_lists_every_block(cli: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{API}/nodes", json=CATALOG)
    res = cli("nodes")
    assert res.exit_code == ExitCode.OK, res.output
    assert "Workflow blocks (2)" in res.stdout and "kind, gcs_uri" in res.stdout
    assert "workflows nodes <block>" in res.stderr


def test_nodes_shows_one_blocks_card_as_json(cli: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{API}/nodes", json=CATALOG)
    res = cli("nodes", "measure")
    assert res.exit_code == ExitCode.OK, res.output
    card = json.loads(res.stdout)
    assert card["type"] == "measure" and card["rules"] == ["It needs a Track block upstream."]


def test_an_unknown_block_is_not_found_and_names_the_blocks(
    cli: Any, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=f"{API}/nodes", json=CATALOG)
    res = cli("nodes", "trak", "--json")
    assert res.exit_code == ExitCode.NOT_FOUND
    error = _last_error(res.stderr)
    assert error["code"] == "not_found" and error["fix"] == "Blocks: source, measure."


def test_nodes_json_is_the_whole_catalog(cli: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{API}/nodes", json=CATALOG)
    res = cli("nodes", "--json")
    assert res.exit_code == ExitCode.OK, res.output
    assert json.loads(res.stdout)["graph"]["ports"] == ["frames", "detections", "events"]


# ── templates ──────────────────────────────────────────────────────────────────


def test_templates_list(cli: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{API}/templates", json=TEMPLATES)
    res = cli("templates")
    assert res.exit_code == ExitCode.OK, res.output
    assert "Traffic monitoring" in res.stdout and "Workflow templates (2)" in res.stdout


def test_a_templates_graph_prints_or_is_written(
    cli: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    httpx_mock.add_response(url=f"{API}/templates", json=TEMPLATES)
    httpx_mock.add_response(url=f"{API}/templates", json=TEMPLATES)
    printed = cli("templates", "traffic")
    assert printed.exit_code == ExitCode.OK, printed.output
    assert json.loads(printed.stdout) == GRAPH
    out = tmp_path / "t.json"
    written = cli("templates", "traffic", "-o", str(out))
    assert written.exit_code == ExitCode.OK, written.output
    assert json.loads(out.read_text()) == GRAPH
    assert json.loads(written.stdout) == {"template": "traffic", "graph_path": str(out)}


def test_template_refusals(cli: Any, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    no_key = cli("templates", "-o", str(tmp_path / "x.json"))
    assert no_key.exit_code == ExitCode.INVALID
    assert httpx_mock.get_requests() == []
    httpx_mock.add_response(url=f"{API}/templates", json=TEMPLATES)
    res = cli("templates", "trafic", "--json")
    assert res.exit_code == ExitCode.NOT_FOUND
    assert _last_error(res.stderr)["fix"] == "Templates: traffic, custom."


# ── validate ───────────────────────────────────────────────────────────────────


def test_a_graph_that_would_run_is_exit_0_with_its_warnings_on_stderr(
    cli: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    warn = {
        "code": "crop_no_consumer",
        "message": "Feeds nothing.",
        "node_id": "c",
        "severity": "warning",
    }
    httpx_mock.add_response(
        method="POST", url=f"{API}/validate", json={"valid": True, "issues": [warn]}
    )
    res = cli("validate", "--graph", _graph_file(tmp_path))
    assert res.exit_code == ExitCode.OK, res.output
    assert json.loads(res.stdout)["valid"] is True
    assert "warning: [crop_no_consumer] Feeds nothing. (block c)" in res.stderr
    (request,) = httpx_mock.get_requests()
    assert json.loads(request.content) == {"graph": GRAPH}


def test_a_graph_that_would_not_run_is_exit_6_with_every_issue(
    cli: Any, httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    httpx_mock.add_response(method="POST", url=f"{API}/validate", json=BROKEN)
    httpx_mock.add_response(method="POST", url=f"{API}/validate", json=BROKEN)
    text = cli("validate", "--graph", _graph_file(tmp_path))
    assert text.exit_code == ExitCode.INVALID
    flat = " ".join(text.stderr.split())
    assert "would not run - 2 error(s)" in flat
    assert "[model_missing] Pick a model for this block. (block model)" in flat
    assert "[measure_no_geometry]" in flat and text.stdout == ""
    as_json = cli("validate", "--graph", _graph_file(tmp_path), "--json")
    error = _last_error(as_json.stderr)
    assert (error["code"], error["exit_code"]) == ("invalid_workflow", ExitCode.INVALID)
    assert [i["code"] for i in error["details"]["issues"]] == [
        "model_missing",
        "measure_no_geometry",
    ]


def test_validate_a_saved_workflow_checks_its_graph(cli: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{API}/{WID}",
        json={"workflow": {"id": WID, "organization_id": "o", "name": "Lot", "graph": GRAPH}},
    )
    httpx_mock.add_response(
        method="POST", url=f"{API}/validate", json={"valid": True, "issues": []}
    )
    res = cli("validate", WID)
    assert res.exit_code == ExitCode.OK, res.output
    assert json.loads(httpx_mock.get_requests()[-1].content) == {"graph": GRAPH}


def test_validate_reads_a_graph_from_stdin(cli: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{API}/validate", json={"valid": True, "issues": []}
    )
    res = cli("validate", "--graph", "-", stdin=json.dumps(GRAPH))
    assert res.exit_code == ExitCode.OK, res.output
    assert json.loads(httpx_mock.get_requests()[0].content) == {"graph": GRAPH}


@pytest.mark.parametrize(
    ("flags", "source"),
    [
        (
            ("--dataset", "Lot cams", "--directory", "/north"),
            {"kind": "dataset", "dataset": "Lot cams", "directory": "/north"},
        ),
        (("--export", "Night export"), {"kind": "export", "export": "Night export"}),
        (
            ("--video", "https://example.com/gate.mp4"),
            {"kind": "video", "url": "https://example.com/gate.mp4"},
        ),
        (
            ("--image", "gs://bucket/org/workflows/uploads/a.jpg"),
            {"kind": "image", "image": "gs://bucket/org/workflows/uploads/a.jpg"},
        ),
    ],
)
def test_validate_checks_a_graph_against_the_input_a_run_would_name(
    cli: Any, httpx_mock: HTTPXMock, flags: tuple[str, ...], source: dict[str, Any]
) -> None:
    httpx_mock.add_response(
        method="POST", url=f"{API}/validate", json={"valid": True, "issues": []}
    )
    res = cli("validate", "--graph", "-", *flags, stdin=json.dumps(GRAPH))
    assert res.exit_code == ExitCode.OK, res.output
    assert json.loads(httpx_mock.get_requests()[0].content) == {"graph": GRAPH, "source": source}


@pytest.mark.parametrize(
    "flags",
    [
        ("--dataset", "a", "--export", "b"),
        ("--directory", "/north"),
        ("--video", "clip.mp4"),
    ],
)
def test_validate_refuses_an_input_it_cannot_name_before_any_request(
    cli: Any, httpx_mock: HTTPXMock, flags: tuple[str, ...]
) -> None:
    res = cli("validate", "--graph", "-", *flags, stdin=json.dumps(GRAPH))
    assert res.exit_code == ExitCode.INVALID, res.output
    assert httpx_mock.get_requests() == []


@pytest.mark.parametrize("args", [(), (WID, "--graph", "g.json")])
def test_validate_takes_a_workflow_or_a_graph_not_both(
    cli: Any, httpx_mock: HTTPXMock, args: tuple[str, ...]
) -> None:
    res = cli("validate", *args)
    assert res.exit_code == ExitCode.INVALID
    assert httpx_mock.get_requests() == []


@pytest.mark.parametrize(
    ("content", "why"), [("{not json", "not JSON"), ("[1, 2]", "a JSON object")]
)
def test_a_graph_file_that_is_not_a_graph_is_refused_before_any_request(
    cli: Any, httpx_mock: HTTPXMock, tmp_path: Path, content: str, why: str
) -> None:
    for command in (("validate",), ("create", "Lot"), ("update", WID)):
        res = cli(*command, "--graph", _graph_file(tmp_path, content))
        assert res.exit_code == ExitCode.INVALID, command
        assert why in " ".join(res.output.replace("│", " ").split()), command
    assert httpx_mock.get_requests() == []


# ── create ─────────────────────────────────────────────────────────────────────


def test_create_from_a_template_shows_what_it_still_needs(cli: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{API}/",
        json={
            "workflow": {
                "id": WID,
                "organization_id": "o",
                "name": "Lot",
                "graph": GRAPH,
                "template_key": "traffic",
                "validation": BROKEN["issues"],
            }
        },
    )
    res = cli("create", "Lot", "--template", "traffic")
    assert res.exit_code == ExitCode.OK, res.output
    assert json.loads(httpx_mock.get_requests()[0].content) == {
        "name": "Lot",
        "template_key": "traffic",
    }
    assert [i["code"] for i in json.loads(res.stdout)["validation"]] == [
        "model_missing",
        "measure_no_geometry",
    ]


def test_create_needs_a_graph_or_a_template(cli: Any, httpx_mock: HTTPXMock) -> None:
    res = cli("create", "Lot")
    assert res.exit_code == ExitCode.INVALID
    assert "--template" in " ".join(res.output.replace("│", " ").split())
    assert httpx_mock.get_requests() == []
