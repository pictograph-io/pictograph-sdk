"""Tests for ``pictograph.resources.annotation_comments.AnnotationComments``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.models.annotation_comment import AnnotationComment
from pictograph.resources.annotation_comments import AnnotationComments

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
IMG = "11aa22bb-33cc-44dd-55ee-66ff77008899"
CID = "c-1"
PATH = f"{BASE}/api/v1/developer/annotation-comments"


@pytest.fixture
def transport() -> Transport:
    config = ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=0)  # type: ignore[arg-type]
    t = Transport(config, api_key=KEY)
    yield t
    t.close()


@pytest.fixture
def comments(transport: Transport) -> AnnotationComments:
    return AnnotationComments(transport)


def _row(**over: Any) -> dict[str, Any]:
    base = {
        "id": CID,
        "annotation_id": "ann-1",
        "body": "tighten this",
        "resolved": False,
        "user_id": "u1",
        "author_name": "Alice",
        "is_mine": True,
    }
    base.update(over)
    return base


def test_list(httpx_mock: HTTPXMock, comments: AnnotationComments) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{PATH}?image_id={IMG}", json={"comments": [_row(), _row(id="c-2")]}
    )
    out = comments.list("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", IMG)
    assert len(out) == 2
    assert all(isinstance(c, AnnotationComment) for c in out)
    assert out[0].body == "tighten this"


def test_create(httpx_mock: HTTPXMock, comments: AnnotationComments) -> None:
    httpx_mock.add_response(method="POST", url=PATH, json={"comment": _row()})
    c = comments.create(
        "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", IMG, annotation_id="ann-1", body="tighten this"
    )
    assert c.id == CID and c.is_mine is True
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {"image_id": IMG, "annotation_id": "ann-1", "body": "tighten this"}


def test_resolve(httpx_mock: HTTPXMock, comments: AnnotationComments) -> None:
    httpx_mock.add_response(
        method="PATCH", url=f"{PATH}/{CID}", json={"comment": _row(resolved=True)}
    )
    c = comments.resolve(CID)
    assert c.resolved is True
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {"resolved": True}


def test_update_body(httpx_mock: HTTPXMock, comments: AnnotationComments) -> None:
    httpx_mock.add_response(
        method="PATCH", url=f"{PATH}/{CID}", json={"comment": _row(body="edited")}
    )
    c = comments.update(CID, body="edited")
    assert c.body == "edited"
    body = json.loads(httpx_mock.get_request().read())  # type: ignore[union-attr]
    assert body == {"body": "edited"}


def test_delete(httpx_mock: HTTPXMock, comments: AnnotationComments) -> None:
    httpx_mock.add_response(method="DELETE", url=f"{PATH}/{CID}", json={"success": True, "id": CID})
    assert comments.delete(CID) is None
