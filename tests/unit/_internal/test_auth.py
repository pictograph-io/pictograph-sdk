"""Tests for ``pictograph._internal.auth.resolve_api_key``."""

from __future__ import annotations

import pytest

from pictograph._internal.auth import ENV_VAR_NAME, resolve_api_key
from pictograph.exceptions import ConfigurationError


def test_explicit_key_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR_NAME, "from_env")
    assert resolve_api_key(explicit="from_arg") == "from_arg"


def test_env_var_used_when_explicit_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR_NAME, "pk_live_from_env")
    assert resolve_api_key() == "pk_live_from_env"


def test_explicit_none_falls_through_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR_NAME, "from_env")
    assert resolve_api_key(explicit=None) == "from_env"


def test_explicit_empty_string_falls_through_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR_NAME, "from_env")
    assert resolve_api_key(explicit="") == "from_env"


def test_explicit_whitespace_only_falls_through_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # A user accidentally passing ``api_key=" "`` shouldn't masquerade as a key.
    monkeypatch.setenv(ENV_VAR_NAME, "from_env")
    assert resolve_api_key(explicit="   ") == "from_env"


def test_env_var_whitespace_only_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same hygiene applied to env input - a stray space in the shell init must
    # not register as "key present".
    monkeypatch.setenv(ENV_VAR_NAME, "    ")
    with pytest.raises(ConfigurationError):
        resolve_api_key()


def test_explicit_key_is_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Trailing newline from heredocs / file reads is common; strip silently.
    monkeypatch.delenv(ENV_VAR_NAME, raising=False)
    assert resolve_api_key(explicit="pk_live_x\n") == "pk_live_x"


def test_env_var_value_is_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR_NAME, "  pk_live_envtrim  ")
    assert resolve_api_key() == "pk_live_envtrim"


def test_no_key_anywhere_raises_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR_NAME, raising=False)
    with pytest.raises(ConfigurationError) as exc:
        resolve_api_key()
    err = exc.value
    assert "No Pictograph API key" in err.message
    # The fix message must name the env var (so users can copy-paste) and the
    # api_key= kwarg (so SDK callers know the alternative).
    assert err.fix is not None
    assert ENV_VAR_NAME in err.fix
    assert "api_key" in err.fix
    assert err.docs_url is not None


def test_env_var_name_constant_matches_documented_value() -> None:
    # If we ever rename the env var, this test will surface every doc page,
    # README example, error message, and CI integration that needs updating.
    assert ENV_VAR_NAME == "PICTOGRAPH_API_KEY"
