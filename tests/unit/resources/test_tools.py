"""Tests for ``client.tools`` - the vision tools an organization can call, and how.

The cards parsed here are REAL: ``tests/fixtures/tool_cards.json`` holds what the
Pictograph API returns for SAM3, a model, a deployment and a workflow, and the
platform's own test suite regenerates it whenever a card changes. So these tests
cannot pass against a shape the API does not send.

Coverage:
- ``list`` sends the kind filter, pages until the API says there is no more, and
  returns typed rows whose ``tool`` is the handle ``describe`` takes.
- ``describe`` turns a handle into the card route (names are URL-quoted, ``sam3`` has
  no name) and refuses anything that is not a handle before any request is made.
- The card model: every mode answers each input kind, costs read in dollars, the
  ``schema`` wire key lands on ``json_schema``, unknown future keys are ignored.
- ``to_markdown`` carries the calls a person (or an agent) needs.
- The async twin makes the same requests and returns the same models.
- The ``pictograph tools`` CLI: table / JSON list, Markdown / JSON card, clear refusals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pictograph import AsyncClient, Client, ToolCard, ToolSummary
from pictograph.cli._app import app
from pictograph.exceptions import ConflictError, NotFoundError
from pictograph.models.tool import ToolMode
from pictograph.resources.tools import card_path

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
API = f"{BASE}/api/v1/developer/tools/vision"

_FIXTURE = json.loads(
    (Path(__file__).resolve().parents[2] / "fixtures" / "tool_cards.json").read_text(
        encoding="utf-8"
    )
)
CARDS: dict[str, dict[str, Any]] = _FIXTURE["cards"]
ROWS: list[dict[str, Any]] = _FIXTURE["list"]
HANDLES = sorted(CARDS)


def _page(rows: list[dict[str, Any]], *, offset: int, total: int) -> dict[str, Any]:
    return {
        "data": rows,
        "pagination": {
            "limit": 500,
            "offset": offset,
            "total": total,
            "has_more": offset + len(rows) < total,
        },
    }


@pytest.fixture
def client() -> Iterator[Client]:
    c = Client(api_key=KEY, base_url=BASE, max_retries=0)
    yield c
    c.close()


# ───────────── the fixture is what these tests think it is ─────────────


def test_fixture_holds_one_card_of_every_kind() -> None:
    assert {c["kind"] for c in CARDS.values()} == {"sam3", "model", "deployment", "workflow"}
    assert [r["tool"] for r in ROWS] == list(CARDS)


# ───────────── list ─────────────


def test_list_returns_typed_rows(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{API}/?limit=500&offset=0", json=_page(ROWS, offset=0, total=len(ROWS))
    )
    tools = client.tools.list()
    assert all(isinstance(t, ToolSummary) for t in tools)
    assert [t.tool for t in tools] == list(CARDS)
    assert tools[0].kind == "sam3" and tools[0].id is None
    assert all(t.summary for t in tools)


def test_list_sends_the_kind_filter(client: Client, httpx_mock: HTTPXMock) -> None:
    row = next(r for r in ROWS if r["kind"] == "workflow")
    httpx_mock.add_response(
        url=f"{API}/?limit=500&offset=0&kind=workflow", json=_page([row], offset=0, total=1)
    )
    assert [t.tool for t in client.tools.list(kind="workflow")] == [row["tool"]]


def test_list_pages_until_the_api_says_there_is_no_more(
    client: Client, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        url=f"{API}/?limit=500&offset=0", json=_page(ROWS[:2], offset=0, total=len(ROWS))
    )
    httpx_mock.add_response(
        url=f"{API}/?limit=500&offset=2", json=_page(ROWS[2:], offset=2, total=len(ROWS))
    )
    assert [t.tool for t in client.tools.list()] == list(CARDS)


def test_list_stops_on_an_empty_page_even_if_has_more_lies(
    client: Client, httpx_mock: HTTPXMock
) -> None:
    lying = _page([], offset=0, total=9)
    assert lying["pagination"]["has_more"] is True
    httpx_mock.add_response(url=f"{API}/?limit=500&offset=0", json=lying)
    assert client.tools.list() == []


# ───────────── describe ─────────────


@pytest.mark.parametrize(
    ("handle", "path"),
    [
        ("sam3", "/sam3"),
        (" sam3 ", "/sam3"),
        ("model/Shelf Detector", "/model/Shelf%20Detector"),
        ("workflow/Door counter", "/workflow/Door%20counter"),
        ("deployment/shelf-endpoint", "/deployment/shelf-endpoint"),
        # A name is quoted whole: nothing in it can become a path separator or a query.
        ("model/50% off? & more", "/model/50%25%20off%3F%20%26%20more"),
        (
            "model/aaaaaaaa-0000-4000-8000-000000000001",
            "/model/aaaaaaaa-0000-4000-8000-000000000001",
        ),
    ],
)
def test_card_path(handle: str, path: str) -> None:
    assert card_path(handle) == f"/api/v1/developer/tools/vision{path}"


@pytest.mark.parametrize(
    "bad", ["", "model", "model/", "model/   ", "dataset/x", "Shelf Detector", "sam3/x", "/x"]
)
def test_describe_refuses_what_is_not_a_handle_before_any_request(
    client: Client, httpx_mock: HTTPXMock, bad: str
) -> None:
    with pytest.raises(ValueError, match="'sam3', 'model/<name>'"):
        client.tools.describe(bad)
    assert httpx_mock.get_requests() == []


@pytest.mark.parametrize("handle", HANDLES)
def test_describe_parses_a_real_card(client: Client, httpx_mock: HTTPXMock, handle: str) -> None:
    httpx_mock.add_response(url=f"{BASE}{card_path(handle)}", json={"data": CARDS[handle]})
    card = client.tools.describe(handle)
    assert isinstance(card, ToolCard)
    assert card.tool == handle and card.kind == CARDS[handle]["kind"]
    assert card.summary and card.modes and card.parameters
    for mode in card.modes:
        assert isinstance(mode, ToolMode)
        # yes or no for each of the five ways an input can arrive.
        assert [i.kind for i in mode.inputs] == ["file", "url", "bytes", "video", "dataset"]
        assert mode.cost.summary
        assert mode.run.cli and mode.run.python
    # Every wire key of the card is one the model reads (nothing silently dropped).
    assert set(CARDS[handle]) <= set(ToolCard.model_fields)


def test_model_card_answers_the_questions_an_agent_asks(
    client: Client, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        url=f"{BASE}{card_path('model/Shelf Detector')}",
        json={"data": CARDS["model/Shelf Detector"]},
    )
    card = client.tools.describe("model/Shelf Detector")
    assert [c.name for c in card.classes] == ["bottle", "can", "box"]
    assert {c.type for c in card.classes} == {"bbox"}

    hosted, local = card.mode("hosted"), card.mode("local")
    assert hosted is not None and local is not None and card.mode("nope") is None
    # Can I pass a URL? Yes to both. A video? No: the note says what to do instead.
    assert hosted.accepts("file") and hosted.accepts("url")
    assert local.accepts("url") and not local.accepts("video")
    video = next(i for i in hosted.inputs if i.kind == "video")
    assert video.note and "workflow" in video.note
    # What does it cost? Both free; a deployment is per minute.
    assert hosted.cost.free and hosted.cost.usd == 0.0
    deploy = card.mode("deploy")
    assert deploy is not None and deploy.cost.per == "minute"
    assert deploy.cost.usd is not None and deploy.cost.usd > 0 and not deploy.cost.free
    assert {o["compute"] for o in deploy.cost.options} >= {"cpu", "t4"}
    # Does it write anything?
    assert hosted.writes is False
    dataset = card.mode("dataset")
    assert dataset is not None and isinstance(dataset.writes, str) and dataset.writes
    # What comes back?
    assert card.output is not None and card.output.type == "bbox"
    assert card.output.json_schema is not None  # `schema` on the wire
    assert card.output.json_schema["properties"]["bounding_box"]
    assert card.output.example is not None and card.output.example["name"] == "bottle"
    # The bounds the API enforces.
    confidence = next(p for p in card.parameters if p.name == "confidence")
    assert (confidence.default, confidence.minimum, confidence.maximum) == (0.5, 0.05, 0.95)
    assert card.related[0].tool == "deployment/shelf-endpoint"


def test_card_ignores_keys_a_newer_api_adds(client: Client, httpx_mock: HTTPXMock) -> None:
    newer = json.loads(json.dumps(CARDS["sam3"]))
    newer["a_future_key"] = {"x": 1}
    newer["modes"][0]["a_future_key"] = True
    httpx_mock.add_response(url=f"{API}/sam3", json={"data": newer})
    assert client.tools.describe("sam3").tool == "sam3"


def test_card_dump_round_trips_the_wire_shape() -> None:
    card = ToolCard.model_validate(CARDS["model/Shelf Detector"])
    dumped = card.model_dump(mode="json", by_alias=True)
    assert dumped["output"]["schema"] == CARDS["model/Shelf Detector"]["output"]["schema"]
    assert "json_schema" not in dumped["output"]
    assert ToolCard.model_validate(dumped) == card


def test_describe_surfaces_the_apis_refusals(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}{card_path('model/nope')}",
        status_code=404,
        json={"detail": "Model 'nope' not found"},
    )
    with pytest.raises(NotFoundError, match="nope"):
        client.tools.describe("model/nope")
    httpx_mock.add_response(
        url=f"{BASE}{card_path('model/halfway')}",
        status_code=409,
        json={"detail": "Model 'halfway' is not ready (status: training)."},
    )
    with pytest.raises(ConflictError, match="not ready"):
        client.tools.describe("model/halfway")


# ───────────── Markdown ─────────────


@pytest.mark.parametrize("handle", HANDLES)
def test_markdown_carries_every_call(handle: str) -> None:
    card = ToolCard.model_validate(CARDS[handle])
    md = card.to_markdown()
    assert md.startswith(f"# {card.name}\n")
    assert f"`{handle}`" in md and card.summary in md
    for mode in card.modes:
        assert f"(`{mode.mode}`)" in md
        assert mode.cost.summary in md
        for code in (mode.run.cli, mode.run.python, mode.run.http):
            if code:
                assert code in md, f"{handle}:{mode.mode} lost a call in the Markdown"
    for parameter in card.parameters:
        assert f"`{parameter.name}`" in md
    for note in card.notes:
        assert note in md
    assert md.endswith("\n") and "\n\n\n" not in md
    assert "—" not in md


def test_markdown_lists_a_workflows_models_and_measures_one_per_line() -> None:
    md = ToolCard.model_validate(CARDS["workflow/Door counter"]).to_markdown()
    assert "- models:\n  - Shelf Detector (kind trained, classes bottle)\n  - SAM3 (" in md
    assert "- measures:\n  - Door (kind line, measure line_cross)\n  - Aisle (" in md
    assert "- outputs: json, annotated_video" in md


def test_markdown_says_what_a_mode_takes_and_writes() -> None:
    md = ToolCard.model_validate(CARDS["model/Shelf Detector"]).to_markdown()
    assert "**Classes (bbox):** bottle, can, box" in md
    assert md.count("Takes: file, url, bytes.") == 2  # hosted and local
    assert "Takes: dataset." in md  # the dataset mode
    assert "- video: no - Sample frames yourself, or run a workflow" in md
    assert "- headline metric: 61.0% mAP" in md
    # One requirement per line: an install command never runs into the next sentence.
    assert 'Needs:\n- pip install "pictograph[inference]"\n- An API key (any role)' in md
    assert "Writes: nothing" in md
    assert "Writes: Annotations on the named images." in md
    assert "Cost: Free on every plan." in md
    assert '"bounding_box"' in md  # the output example


# ───────────── async twin ─────────────


@pytest.mark.anyio
async def test_async_twin_makes_the_same_requests(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{API}/?limit=500&offset=0&kind=model",
        json=_page([ROWS[1]], offset=0, total=1),
    )
    httpx_mock.add_response(
        url=f"{BASE}{card_path('workflow/Door counter')}",
        json={"data": CARDS["workflow/Door counter"]},
    )
    async with AsyncClient(api_key=KEY, base_url=BASE, max_retries=0) as client:
        rows = await client.tools.list(kind="model")
        card = await client.tools.describe("workflow/Door counter")
        with pytest.raises(ValueError, match="Not a tool"):
            await client.tools.describe("nope")
    assert [r.tool for r in rows] == ["model/Shelf Detector"]
    assert card == ToolCard.model_validate(CARDS["workflow/Door counter"])
    assert card.facts["tracker"] == "botsort"


# ───────────── CLI ─────────────


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[CliRunner, MagicMock]]:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", KEY)
    fake = MagicMock()
    fake.tools.list.return_value = [ToolSummary.model_validate(r) for r in ROWS]
    fake.tools.describe.side_effect = lambda tool: ToolCard.model_validate(CARDS[tool])
    with patch("pictograph.cli.commands.tools.get_client", return_value=fake):
        yield CliRunner(), fake


def test_cli_list_table_names_every_tool_and_the_next_step(
    cli: tuple[CliRunner, MagicMock],
) -> None:
    runner, fake = cli
    res = runner.invoke(app, ["tools", "list"], env={"COLUMNS": "200"})
    assert res.exit_code == 0, res.stdout
    for handle in CARDS:
        assert handle in res.stdout
    assert "pictograph tools describe <tool>" in res.stdout
    fake.tools.list.assert_called_once_with(kind=None)


def test_cli_list_json_and_kind(cli: tuple[CliRunner, MagicMock]) -> None:
    runner, fake = cli
    res = runner.invoke(app, ["tools", "list", "--kind", "model", "--json"])
    assert res.exit_code == 0, res.stdout
    assert [r["tool"] for r in json.loads(res.stdout)] == list(CARDS)
    fake.tools.list.assert_called_once_with(kind="model")


def test_cli_list_refuses_an_unknown_kind_before_calling(
    cli: tuple[CliRunner, MagicMock],
) -> None:
    runner, fake = cli
    res = runner.invoke(app, ["tools", "list", "--kind", "dataset"])
    assert res.exit_code == 2
    # (the error is drawn in a box, so it may wrap)
    assert "sam3, model, deployment, workflow" in " ".join(res.output.replace("│", " ").split())
    fake.tools.list.assert_not_called()


def test_cli_describe_prints_the_markdown_card(cli: tuple[CliRunner, MagicMock]) -> None:
    runner, _fake = cli
    res = runner.invoke(app, ["tools", "describe", "model/Shelf Detector"])
    assert res.exit_code == 0, res.stdout
    assert res.stdout == ToolCard.model_validate(CARDS["model/Shelf Detector"]).to_markdown()


def test_cli_describe_json_is_the_wire_card(cli: tuple[CliRunner, MagicMock]) -> None:
    runner, _fake = cli
    res = runner.invoke(app, ["tools", "describe", "sam3", "--json"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["tool"] == "sam3"
    assert [m["mode"] for m in payload["modes"]] == ["text", "box", "point", "dataset"]
    assert "schema" in payload["output"] and "json_schema" not in payload["output"]
    assert payload["modes"][0]["run"]["cli"].startswith("pictograph auto-annotate text ")


def test_cli_describe_says_what_a_tool_is_when_given_something_else(
    cli: tuple[CliRunner, MagicMock],
) -> None:
    runner, fake = cli
    fake.tools.describe.side_effect = Client(api_key=KEY).tools.describe  # the real refusal
    res = runner.invoke(app, ["tools", "describe", "Shelf Detector"])
    assert res.exit_code == 2
    assert "model/<name>" in " ".join(res.output.replace("│", " ").split())
