"""Tests for ``pictograph._internal.config.ClientConfig``."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from pictograph._internal.config import ClientConfig


def test_defaults_applied_when_no_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    # Isolate from any PICTOGRAPH_* vars set in the developer's shell.
    for var in (
        "PICTOGRAPH_API_KEY",
        "PICTOGRAPH_BASE_URL",
        "PICTOGRAPH_TIMEOUT",
        "PICTOGRAPH_MAX_RETRIES",
    ):
        monkeypatch.delenv(var, raising=False)
    config = ClientConfig()
    assert config.api_key is None
    assert config.base_url == "https://api.pictograph.io"
    assert config.timeout == 30.0
    assert config.max_retries == 3


def test_env_var_populates_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_from_env")
    config = ClientConfig()
    assert isinstance(config.api_key, SecretStr)
    assert config.api_key.get_secret_value() == "pk_live_from_env"


def test_env_var_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PICTOGRAPH_API_KEY", raising=False)
    monkeypatch.setenv("pictograph_api_key", "pk_live_lower")
    config = ClientConfig()
    assert config.api_key is not None
    assert config.api_key.get_secret_value() == "pk_live_lower"


def test_explicit_kwarg_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "from_env")
    config = ClientConfig(api_key="explicit")  # type: ignore[arg-type]
    assert config.api_key is not None
    assert config.api_key.get_secret_value() == "explicit"


def test_env_var_populates_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PICTOGRAPH_BASE_URL", "https://staging.pictograph.io")
    config = ClientConfig()
    assert config.base_url == "https://staging.pictograph.io"


def test_env_var_populates_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PICTOGRAPH_TIMEOUT", "60.5")
    config = ClientConfig()
    assert config.timeout == 60.5


def test_env_var_populates_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PICTOGRAPH_MAX_RETRIES", "10")
    config = ClientConfig()
    assert config.max_retries == 10


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_timeout_must_be_positive(bad: float) -> None:
    with pytest.raises(ValidationError) as exc:
        ClientConfig(timeout=bad)
    err = exc.value.errors()[0]
    assert err["loc"] == ("timeout",)
    assert err["type"] in {"greater_than", "greater_than_equal"}


@pytest.mark.parametrize("bad", [-1, -10])
def test_max_retries_must_be_non_negative(bad: int) -> None:
    with pytest.raises(ValidationError) as exc:
        ClientConfig(max_retries=bad)
    err = exc.value.errors()[0]
    assert err["loc"] == ("max_retries",)


def test_max_retries_zero_is_allowed_meaning_no_retries() -> None:
    config = ClientConfig(max_retries=0)
    assert config.max_retries == 0


def test_unknown_env_vars_with_prefix_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    # Future PICTOGRAPH_* env vars (e.g., PICTOGRAPH_ENABLE_CACHE) shouldn't
    # break older SDK versions. extra="ignore" enforces this.
    monkeypatch.setenv("PICTOGRAPH_FUTURE_FEATURE", "anything")
    # No exception expected.
    ClientConfig()


def test_api_key_repr_does_not_leak_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PICTOGRAPH_API_KEY", "pk_live_supersecret_dont_log")
    config = ClientConfig()
    rep = repr(config)
    assert "supersecret" not in rep
    assert "**" in rep or "SecretStr" in rep
