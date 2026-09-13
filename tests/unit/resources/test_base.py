"""Tests for ``pictograph.resources._base.Resource``.

The base contributes very little by design - its full surface is the two
parse helpers. These tests pin the contract:

- A successful payload → typed Pydantic model, untouched.
- A schema-mismatched payload → :class:`ServerError` (NOT
  :class:`ValidationError` - that exception is reserved for *user* input
  errors; a malformed backend response is the server's problem).
- The offending payload is preserved on the raised error for debugging.
- ``_parse_list`` propagates the same error semantics per element.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph.exceptions import ServerError
from pictograph.resources._base import Resource


class _Sample(BaseModel):
    """Minimal Pydantic model used to exercise the parse helpers."""

    id: str
    count: int


@pytest.fixture
def resource() -> Resource:
    config = ClientConfig(api_key="pk_live_test", base_url="https://api.test.local")  # type: ignore[arg-type]
    t = Transport(config, api_key="pk_live_test")
    yield Resource(t)
    t.close()


# ───────────── _parse ─────────────


def test_parse_returns_typed_model_for_valid_payload(resource: Resource) -> None:
    parsed = resource._parse(_Sample, {"id": "abc", "count": 7})
    assert isinstance(parsed, _Sample)
    assert parsed.id == "abc"
    assert parsed.count == 7


def test_parse_raises_server_error_for_invalid_payload(resource: Resource) -> None:
    # Missing required field - backend contract violation.
    with pytest.raises(ServerError) as exc:
        resource._parse(_Sample, {"id": "abc"})
    assert "_Sample" in exc.value.message
    # The offending payload survives on the exception for debugging.
    assert exc.value.response == {"id": "abc"}


def test_parse_raises_server_error_for_wrong_type(resource: Resource) -> None:
    # ``count`` is required to be an int; a string is the wrong type.
    with pytest.raises(ServerError):
        resource._parse(_Sample, {"id": "abc", "count": "not-an-int"})


def test_parse_preserves_payload_when_payload_is_not_a_dict(resource: Resource) -> None:
    with pytest.raises(ServerError) as exc:
        resource._parse(_Sample, ["unexpected", "list"])
    assert exc.value.response == ["unexpected", "list"]


def test_parse_chains_pydantic_validation_error_as_cause(resource: Resource) -> None:
    # Preserving __cause__ lets users access the underlying Pydantic detail
    # via ``except ServerError as e: e.__cause__`` for advanced debugging.
    with pytest.raises(ServerError) as exc:
        resource._parse(_Sample, {"id": "abc"})
    assert exc.value.__cause__ is not None


# ───────────── _parse_list ─────────────


def test_parse_list_returns_typed_models_for_valid_items(resource: Resource) -> None:
    parsed = resource._parse_list(
        _Sample,
        [{"id": "a", "count": 1}, {"id": "b", "count": 2}, {"id": "c", "count": 3}],
    )
    assert all(isinstance(p, _Sample) for p in parsed)
    assert [(p.id, p.count) for p in parsed] == [("a", 1), ("b", 2), ("c", 3)]


def test_parse_list_returns_empty_list_for_empty_input(resource: Resource) -> None:
    parsed: list[_Sample] = resource._parse_list(_Sample, [])
    assert parsed == []


def test_parse_list_raises_on_first_bad_element(resource: Resource) -> None:
    items: list[Any] = [{"id": "a", "count": 1}, {"id": "b"}, {"id": "c", "count": 3}]
    with pytest.raises(ServerError) as exc:
        resource._parse_list(_Sample, items)
    # Whichever element fails first surfaces its raw value - useful for
    # narrowing down a bad backend response.
    assert exc.value.response == {"id": "b"}


def test_parse_list_does_not_mutate_input(resource: Resource) -> None:
    inputs: list[dict[str, Any]] = [{"id": "a", "count": 1}]
    original = list(inputs)
    resource._parse_list(_Sample, inputs)
    assert inputs == original


# ───────────── transport reference ─────────────


def test_resource_holds_transport_reference(resource: Resource) -> None:
    # Subclasses use ``self._transport`` directly; pin the attribute name.
    assert resource._transport is not None
    assert isinstance(resource._transport, Transport)
