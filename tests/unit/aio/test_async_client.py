"""Tests for ``pictograph.AsyncClient`` construction, config, and lifecycle."""

from __future__ import annotations

import pytest

from pictograph import AsyncClient
from pictograph.aio import (
    AsyncAnnotations,
    AsyncDatasets,
    AsyncImages,
    AsyncModels,
    AsyncTraining,
)
from pictograph.exceptions import ConfigurationError, ValidationError

pytestmark = pytest.mark.anyio

_RESOURCES = [
    "datasets",
    "images",
    "annotations",
    "exports",
    "training",
    "models",
    "deployments",
    "credits",
    "organizations",
    "directories",
    "batch",
    "search",
    "auto_annotate",
    "video",
    "connectors",
    "api_keys",
    "webhooks",
    "workflows",
    "tasks",
]


def test_construction_with_explicit_key() -> None:
    client = AsyncClient(api_key="pk_live_x", base_url="https://api.test.local")
    assert client._config.base_url == "https://api.test.local"


def test_missing_key_raises_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PICTOGRAPH_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        AsyncClient()


def test_env_key_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_env")
    client = AsyncClient()
    assert client._transport._api_key == "pk_live_env"


@pytest.mark.parametrize("bad", [{"timeout": 0}, {"timeout": -1}, {"max_retries": -1}])
def test_invalid_config_rejected(bad: dict[str, float]) -> None:
    with pytest.raises((ValidationError, ValueError)):
        AsyncClient(api_key="pk_live_x", **bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", _RESOURCES)
def test_all_resources_wired(name: str) -> None:
    client = AsyncClient(api_key="pk_live_x")
    assert hasattr(client, name)


def test_resource_types() -> None:
    client = AsyncClient(api_key="pk_live_x")
    assert isinstance(client.datasets, AsyncDatasets)
    assert isinstance(client.images, AsyncImages)
    assert isinstance(client.annotations, AsyncAnnotations)
    assert isinstance(client.training, AsyncTraining)
    assert isinstance(client.models, AsyncModels)


def test_repr_hides_api_key() -> None:
    client = AsyncClient(api_key="pk_live_supersecret", base_url="https://api.test.local")
    assert "supersecret" not in repr(client)
    assert "api.test.local" in repr(client)


async def test_context_manager_closes() -> None:
    async with AsyncClient(api_key="pk_live_x") as client:
        assert client is not None
    # aclose is idempotent
    await client.aclose()


def test_shared_transport_across_resources() -> None:
    client = AsyncClient(api_key="pk_live_x")
    # Every resource shares the single transport (one connection pool).
    assert client.datasets._transport is client.images._transport
    assert client.datasets._transport is client.workflows._transport
