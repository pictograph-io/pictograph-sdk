"""Writing a workflow graph without guessing (B533 #17): ``nodes``, ``templates``, ``validate``
and ``create`` from a template - what each sends and how each answer reads, over the real
transport. The API's side (what the cards and templates say) is held by the backend's
tests/unit/test_workflow_catalog.py; the B522 contract replays these methods over its answers.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.models.workflow import WorkflowNodeCatalog, WorkflowTemplate, WorkflowValidation
from pictograph.resources.workflows import Workflows

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
API = f"{BASE}/api/v1/developer/workflows"
WID = "fedcba98-1111-2222-3333-444455556666"

CATALOG: dict[str, Any] = {
    "graph": {"ports": ["frames", "detections", "events"], "notes": []},
    "nodes": [
        {
            "type": "source",
            "label": "Source",
            "summary": "Where the frames come from.",
            "inputs": [],
            "output": "frames",
            "new_block": {"kind": "video", "sample_fps": 5},
            "fields": [
                {
                    "key": "kind",
                    "type": "string",
                    "required": True,
                    "default": None,
                    "note": "What the run reads.",
                    "values": ["image", "video", "dataset", "export"],
                }
            ],
            "rules": ["A workflow has exactly one Source."],
        },
        {
            "type": "measure",
            "label": "Measure",
            "summary": "Counts.",
            "inputs": ["detections"],
            "output": "events",
            "new_block": {"geometries": []},
            "fields": [
                {
                    "key": "geometries",
                    "type": "array",
                    "required": True,
                    "default": None,
                    "note": "",
                    "fields": [
                        {
                            "key": "kind",
                            "type": "string",
                            "required": True,
                            "default": None,
                            "note": "",
                            "values": ["line", "zone"],
                        },
                        {
                            "key": "direction",
                            "type": "string",
                            "required": False,
                            "default": "both",
                            "note": "",
                            "values": ["both", "in", "out"],
                        },
                    ],
                }
            ],
            "rules": ["It needs a Track block upstream."],
        },
    ],
}
TEMPLATES = {
    "templates": [
        {
            "key": "traffic",
            "title": "Traffic monitoring",
            "description": "Count vehicles.",
            "graph": {"version": 1, "nodes": [{"id": "source", "type": "source"}], "edges": []},
        },
        {
            "key": "custom",
            "title": "Custom",
            "description": "A blank canvas.",
            "graph": {"version": 1, "nodes": [], "edges": []},
        },
    ]
}
REPORT = {
    "valid": False,
    "issues": [
        {
            "code": "model_missing",
            "message": "Pick a model for this block.",
            "node_id": "model",
            "severity": "error",
        },
        {
            "code": "crop_no_consumer",
            "message": "Feeds nothing.",
            "node_id": "c",
            "severity": "warning",
        },
    ],
}


def _wf(**over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": WID,
        "organization_id": "org-1",
        "name": "Lot",
        "description": None,
        "readme": "# Lot",
        "graph": TEMPLATES["templates"][0]["graph"],
        "template_key": "traffic",
        "status": "draft",
        "last_run_id": None,
        "validation": REPORT["issues"][:1],
        "created_at": "2026-10-26T00:00:00Z",
        "updated_at": "2026-10-26T00:00:00Z",
    }
    row.update(over)
    return row


@pytest.fixture
def workflows() -> Iterator[Workflows]:
    config = ClientConfig(api_key="pk_live_test", base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    transport = Transport(config, api_key="pk_live_test")
    yield Workflows(transport)
    transport.close()


def test_nodes_reads_every_card_and_its_nested_settings(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(method="GET", url=f"{API}/nodes", json=CATALOG)
    catalog = workflows.nodes()
    assert isinstance(catalog, WorkflowNodeCatalog)
    assert [n.type for n in catalog.nodes] == ["source", "measure"]
    geometries = catalog.card("measure").fields[0]
    assert geometries.fields is not None
    assert [(f.key, f.default, f.values) for f in geometries.fields] == [
        ("kind", None, ["line", "zone"]),
        ("direction", "both", ["both", "in", "out"]),
    ]
    assert catalog.card("source").new_block == {"kind": "video", "sample_fps": 5}
    with pytest.raises(KeyError, match="Blocks: source, measure"):
        catalog.card("track")


def test_templates_read_each_with_its_graph(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(method="GET", url=f"{API}/templates", json=TEMPLATES)
    found = workflows.templates()
    assert [t.key for t in found] == ["traffic", "custom"]
    assert all(isinstance(t, WorkflowTemplate) for t in found)
    assert found[0].graph["nodes"][0]["id"] == "source"


def test_validate_sends_the_graph_and_reads_every_issue(
    httpx_mock: HTTPXMock, workflows: Workflows
) -> None:
    httpx_mock.add_response(method="POST", url=f"{API}/validate", json=REPORT)
    graph = {"version": 1, "nodes": [], "edges": []}
    report = workflows.validate(graph)
    (request,) = httpx_mock.get_requests()
    assert json.loads(request.content) == {"graph": graph}
    assert isinstance(report, WorkflowValidation) and report.valid is False
    assert [i.code for i in report.errors] == ["model_missing"]
    assert [i.severity for i in report.issues] == ["error", "warning"]


def test_create_from_a_template_sends_no_graph(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(method="POST", url=f"{API}/", json={"workflow": _wf()})
    wf = workflows.create("Lot", template_key="traffic")
    (request,) = httpx_mock.get_requests()
    assert json.loads(request.content) == {"name": "Lot", "template_key": "traffic"}
    # What the API returns about it is kept: its validation and its card.
    assert [i.code for i in wf.validation] == ["model_missing"]
    assert wf.readme == "# Lot"


def test_create_with_a_graph_still_sends_it(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(method="POST", url=f"{API}/", json={"workflow": _wf(validation=[])})
    graph = {"version": 1, "nodes": [], "edges": []}
    workflows.create("Lot", graph)
    (request,) = httpx_mock.get_requests()
    assert json.loads(request.content) == {"name": "Lot", "graph": graph}


def test_get_keeps_the_workflows_validation(httpx_mock: HTTPXMock, workflows: Workflows) -> None:
    httpx_mock.add_response(method="GET", url=f"{API}/{WID}", json={"workflow": _wf()})
    assert [i.message for i in workflows.get(WID).validation] == ["Pick a model for this block."]
