"""Client configuration loaded from explicit kwargs and environment variables.

Resolution order (first non-empty value wins):

1. Explicit kwargs passed to ``Client(...)``.
2. ``PICTOGRAPH_*`` environment variables (e.g. ``PICTOGRAPH_API_KEY``,
   ``PICTOGRAPH_BASE_URL``, ``PICTOGRAPH_TIMEOUT``, ``PICTOGRAPH_MAX_RETRIES``).
3. Built-in defaults defined on this model.

The CLI (``pictograph login``) writes ``~/.pictograph/config.toml``, but that
file is read **only by the CLI** (see ``cli/_config.py``) - the core ``Client``
deliberately does not read it, keeping the SDK free of TOML/file-system
dependencies. For programmatic use, pass ``api_key=`` or set ``PICTOGRAPH_API_KEY``.
"""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ClientConfig(BaseSettings):
    """Resolved configuration for a :class:`pictograph.Client` instance.

    Construct directly with explicit values, or rely on environment-variable
    population by leaving fields unset. Pydantic Settings reads env vars at
    instantiation time.
    """

    model_config = SettingsConfigDict(
        env_prefix="PICTOGRAPH_",
        env_file=None,
        extra="ignore",
        case_sensitive=False,
    )

    api_key: SecretStr | None = None
    """API key. ``SecretStr`` so it never leaks via ``repr`` or logs."""

    base_url: str = "https://api.pictograph.io"
    """Root URL for API calls. Trailing slashes are stripped at request time."""

    timeout: float = Field(default=30.0, gt=0)
    """Per-request timeout in seconds. Must be positive."""

    max_retries: int = Field(default=3, ge=0)
    """Number of retry attempts for retryable failures. ``0`` disables retries."""
