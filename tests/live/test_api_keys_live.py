"""Live: API key CRUD."""

from __future__ import annotations

import pytest

from pictograph import Client
from pictograph.models.api_key import ApiKey, CreatedApiKey

pytestmark = pytest.mark.live


def test_list_contains_current_key(client: Client) -> None:
    keys = client.api_keys.list()
    assert isinstance(keys, list)
    assert len(keys) >= 1
    for k in keys:
        assert isinstance(k, ApiKey)
        assert k.id
        assert k.key_prefix


def test_create_get_update_delete(client: Client, unique_name: str) -> None:
    org = client.organizations.me()
    created = client.api_keys.create(unique_name, organization=org.id, role="viewer")
    assert isinstance(created, CreatedApiKey)
    assert created.api_key.startswith("pk_live_")
    key_id = created.key_id
    try:
        fetched = client.api_keys.get(key_id)
        assert fetched.id == key_id
        assert fetched.name == unique_name

        updated = client.api_keys.update(key_id, name=unique_name + "-renamed")
        assert updated.name == unique_name + "-renamed"

        updated2 = client.api_keys.update(key_id, is_active=False)
        assert updated2.is_active is False
    finally:
        client.api_keys.delete(key_id)


def test_update_requires_at_least_one_field(client: Client) -> None:
    with pytest.raises(ValueError):
        client.api_keys.update("00000000-0000-0000-0000-000000000000")
