"""Base class shared by every SDK resource module.

The base is intentionally thin: it owns a reference to the :class:`Transport`
and provides two helpers - :meth:`Resource._parse` and
:meth:`Resource._parse_list` - for converting a raw JSON payload into typed
Pydantic models.

Resource subclasses (``Datasets``, ``Images``, ``Annotations``, …) are
responsible for their own URL templates, query-parameter shapes, and method
ergonomics. We deliberately avoid a generic ``BaseResource[T]`` with built-in
``list/get/create/update/delete`` methods because the backend's resource
shapes diverge enough (compound names on exports, name-vs-uuid lookups on
datasets, three-step uploads on images, single-shot key reveals on api_keys)
that the abstraction would force more workarounds than it eliminated.

Backend response-shape mismatches are translated to :class:`ServerError`
because the SDK caller cannot fix them - they indicate either a backend bug
or an SDK that has fallen behind the wire format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ValidationError as PydanticValidationError

from pictograph.exceptions import ServerError

if TYPE_CHECKING:
    from typing import Any

    from pictograph._http.async_transport import AsyncTransport
    from pictograph._http.transport import Transport

ModelT = TypeVar("ModelT", bound=BaseModel)


def parse_model(model: type[ModelT], data: Any) -> ModelT:
    """Validate ``data`` against ``model``; surface failures as ``ServerError``.

    Shared by the sync :class:`Resource` and the async
    :class:`pictograph._http.async_base.AsyncResource` - response decoding is
    pure (no I/O), so both transports use the identical validation path.

    Args:
        model: Pydantic model class describing the expected response shape.
        data: Raw payload (typically a dict from ``response.json()``).

    Raises:
        ServerError: ``data`` does not match ``model``'s schema. The offending
            payload is preserved in :attr:`ServerError.response` for debugging.
    """
    try:
        return model.model_validate(data)
    except PydanticValidationError as e:
        raise ServerError(
            f"Backend response did not match expected {model.__name__} schema: {e}",
            response=data,
        ) from e


def parse_model_list(model: type[ModelT], items: list[Any]) -> list[ModelT]:
    """Validate every element of ``items`` against ``model`` (fresh list, input unmutated)."""
    return [parse_model(model, item) for item in items]


class Resource:
    """Base class for SDK resource modules.

    Subclasses receive a :class:`Transport` and use it to make HTTP requests,
    plus the inherited :meth:`_parse` helper for response decoding.
    """

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    def _parse(self, model: type[ModelT], data: Any) -> ModelT:
        """Validate ``data`` against ``model``; surface failures as ``ServerError``."""
        return parse_model(model, data)

    def _parse_list(self, model: type[ModelT], items: list[Any]) -> list[ModelT]:
        """Validate every element of ``items`` against ``model``.

        Returns a fresh list - the input is not mutated.
        """
        return parse_model_list(model, items)


class AsyncResource:
    """Base class for the async SDK resource modules (:class:`pictograph.AsyncClient`).

    The async twin of :class:`Resource`: it holds an :class:`AsyncTransport`
    reference and reuses the identical (pure, I/O-free) Pydantic decoding
    helpers, so sync and async responses validate through exactly one code path.
    """

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    def _parse(self, model: type[ModelT], data: Any) -> ModelT:
        """Validate ``data`` against ``model``; surface failures as ``ServerError``."""
        return parse_model(model, data)

    def _parse_list(self, model: type[ModelT], items: list[Any]) -> list[ModelT]:
        """Validate every element of ``items`` against ``model`` (fresh list)."""
        return parse_model_list(model, items)
