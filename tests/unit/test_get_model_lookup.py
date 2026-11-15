"""``get_model`` and ``client.models.load`` look the model up through the REAL ``client.models``.

SDK 1.69.176 renamed ``models.get_by_name`` to ``models.get``, and the loader kept calling the old
name: every ``get_model`` raised ``AttributeError: 'Models' object has no attribute 'get_by_name'``
through 1.69.178, under a green suite whose loader tests faked ``client.models``. These run the
loader against the real resource over a mocked HTTP transport, stopping only at the engine build.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import pictograph.inference as inference
from pictograph import Client, get_model
from tests.unit.resources.test_models import _item, _model_payload

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stop at the engine build: record the model the loader looked up."""
    seen: dict[str, Any] = {}

    def build(model: Any, **kwargs: Any) -> str:
        seen["model"] = model
        seen["format"] = kwargs.get("format")
        return "engine"

    monkeypatch.setattr(inference, "_build_graph_model", build)
    return seen


def _client() -> Client:
    return Client(api_key="pk_live_" + "x" * 32, base_url=BASE)


@pytest.mark.parametrize("name", ["road-signs", "abcdef01-2345-6789-abcd-ef0123456789"])
def test_get_model_reads_the_model_through_client_models(
    httpx_mock: HTTPXMock, built: dict[str, Any], name: str
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/{name}",
        json=_item(_model_payload(name="road-signs")),
    )
    assert get_model(name, task="object_detection", client=_client()) == "engine"
    assert built["model"].name == "road-signs"
    assert built["format"] == "onnx"


def test_client_models_load_reads_it_the_same_way(
    httpx_mock: HTTPXMock, built: dict[str, Any]
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/road-signs",
        json=_item(_model_payload(name="road-signs")),
    )
    client = _client()
    assert client.models.load("road-signs", task="object_detection") == "engine"
    assert built["model"].name == "road-signs"
