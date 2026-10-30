"""``pictograph annotation-comments`` - notes between an agent and a person (B533 P1 #13).

Annotation comments had SDK methods and developer routes but no CLI, so an agent
could not leave a note on a label it was unsure of or read the reply. These drive the
REAL app and client over a mocked HTTP transport.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from pictograph.cli._app import app
from pictograph.cli._errors import ExitCode

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
COMMENTS = f"{BASE}/api/v1/developer/annotation-comments"
IMAGE_ID = "aaaaaaaa-0000-4000-8000-000000000001"
LOOKUP = re.compile(rf"{re.escape(BASE)}/api/v1/developer/images/\?.*filename=001\.jpg.*")
NOW = "2026-04-01T00:00:00Z"


def _comment(
    cid: str, body: str, *, parent: str | None = None, resolved: bool = False, deleted: bool = False
) -> dict[str, Any]:
    return {
        "id": cid,
        "annotation_id": "ann-1",
        "body": body,
        "parent_comment_id": parent,
        "resolved": resolved,
        "is_deleted": deleted,
        "author_username": "noah",
        "created_at": NOW,
    }


THREAD = [
    _comment("c1", "Is this a yield sign?"),
    _comment("c2", "Yes - relabelled.", parent="c1"),
    _comment("c3", "Fixed already.", resolved=True),
    _comment("c4", "", deleted=True),
]


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_test")
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", BASE)
    monkeypatch.setenv("PICTOGRAPH_MAX_RETRIES", "0")
    monkeypatch.delenv("PICTOGRAPH_OUTPUT", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    def invoke(*args: str, stdin: str | None = None) -> Any:
        return runner.invoke(app, ["annotation-comments", *args], input=stdin)

    return invoke


def _image_lookup(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=LOOKUP, json={"data": [{"id": IMAGE_ID, "filename": "001.jpg", "created_at": NOW}]}
    )


def _thread(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{COMMENTS}?image_id={IMAGE_ID}", json={"comments": THREAD}
    )


def test_list_shows_the_live_comments(run: Any, httpx_mock: HTTPXMock) -> None:
    _image_lookup(httpx_mock)
    _thread(httpx_mock)
    res = run("list", "road-signs", "001.jpg", "--json")
    assert res.exit_code == ExitCode.OK, res.output
    assert [c["id"] for c in json.loads(res.stdout)] == ["c1", "c2", "c3"], "deleted hidden"


def test_list_unresolved_keeps_open_threads_with_their_replies(
    run: Any, httpx_mock: HTTPXMock
) -> None:
    _image_lookup(httpx_mock)
    _thread(httpx_mock)
    res = run("list", "road-signs", "001.jpg", "--unresolved", "--json")
    assert [c["id"] for c in json.loads(res.stdout)] == ["c1", "c2"]


def test_add_comments_on_the_annotation(run: Any, httpx_mock: HTTPXMock) -> None:
    _image_lookup(httpx_mock)
    httpx_mock.add_response(
        method="POST", url=COMMENTS, json={"comment": _comment("c9", "Is this a car? @noah")}
    )
    res = run("add", "road-signs", "001.jpg", "ann-1", "Is this a car? @noah")
    assert res.exit_code == ExitCode.OK, res.output
    (post,) = [r for r in httpx_mock.get_requests() if r.method == "POST"]
    assert json.loads(post.content) == {
        "image_id": IMAGE_ID,
        "annotation_id": "ann-1",
        "body": "Is this a car? @noah",
    }


def test_reply_finds_the_annotation_of_the_comment_it_answers(
    run: Any, httpx_mock: HTTPXMock
) -> None:
    _image_lookup(httpx_mock)
    _thread(httpx_mock)
    _image_lookup(httpx_mock)
    httpx_mock.add_response(
        method="POST", url=COMMENTS, json={"comment": _comment("c5", "Done", parent="c1")}
    )
    res = run("reply", "road-signs", "001.jpg", "c1", "-", stdin="Done\n")
    assert res.exit_code == ExitCode.OK, res.output
    (post,) = [r for r in httpx_mock.get_requests() if r.method == "POST"]
    sent = json.loads(post.content)
    assert sent["parent_comment_id"] == "c1" and sent["annotation_id"] == "ann-1"
    assert sent["body"] == "Done\n"


def test_reply_to_a_missing_comment_is_not_found(run: Any, httpx_mock: HTTPXMock) -> None:
    _image_lookup(httpx_mock)
    _thread(httpx_mock)
    res = run("reply", "road-signs", "001.jpg", "nope", "x")
    assert res.exit_code == ExitCode.NOT_FOUND
    assert not [r for r in httpx_mock.get_requests() if r.method == "POST"]


def test_resolve_and_reopen(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="PATCH", url=f"{COMMENTS}/c1", json={"comment": _comment("c1", "x", resolved=True)}
    )
    httpx_mock.add_response(
        method="PATCH", url=f"{COMMENTS}/c1", json={"comment": _comment("c1", "x")}
    )
    assert run("resolve", "c1").exit_code == ExitCode.OK
    assert run("resolve", "c1", "--reopen").exit_code == ExitCode.OK
    bodies = [json.loads(r.content) for r in httpx_mock.get_requests()]
    assert bodies == [{"resolved": True}, {"resolved": False}]


def test_an_empty_comment_is_refused_before_any_request(run: Any, httpx_mock: HTTPXMock) -> None:
    res = run("add", "road-signs", "001.jpg", "ann-1", "   ")
    assert res.exit_code == ExitCode.INVALID
    assert httpx_mock.get_requests() == []


def test_delete_asks_first(run: Any, httpx_mock: HTTPXMock) -> None:
    assert run("delete", "c1", stdin="").exit_code == ExitCode.FAILED
    httpx_mock.add_response(method="DELETE", url=f"{COMMENTS}/c1", status_code=204)
    res = run("delete", "c1", "--yes")
    assert res.exit_code == ExitCode.OK, res.output
    assert json.loads(res.stdout) == {"comment_id": "c1", "deleted": True}


def test_edit_changes_the_text(run: Any, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="PATCH", url=f"{COMMENTS}/c1", json={"comment": _comment("c1", "Now a car.")}
    )
    res = run("edit", "c1", "Now a car.")
    assert res.exit_code == ExitCode.OK, res.output
    (patch,) = httpx_mock.get_requests()
    assert json.loads(patch.content) == {"body": "Now a car."}
