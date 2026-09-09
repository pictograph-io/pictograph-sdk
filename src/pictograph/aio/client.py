"""Top-level :class:`AsyncClient` - the ``asyncio`` entry point for the SDK.

The async counterpart of :class:`pictograph.Client`. It owns a single
:class:`pictograph._http.async_transport.AsyncTransport` (one shared
``httpx.AsyncClient`` connection pool, HTTP/2) and exposes the same resource
surface, coroutine-for-method::

    import asyncio
    from pictograph import AsyncClient


    async def main() -> None:
        async with AsyncClient() as client:
            datasets = await client.datasets.list(limit=5)
            async for img in client.images.iter(datasets[0].id):
                print(img.filename)


    asyncio.run(main())

Every resource method that performs I/O is a coroutine (``await`` it); the
``iter`` accessors return an :class:`~pictograph._http.pagination.AsyncOffsetPager`
you ``async for`` over. Credential/config resolution is identical to the sync
:class:`~pictograph.Client` (explicit kwargs → ``PICTOGRAPH_*`` env → defaults).

Use it as an async context manager to guarantee socket cleanup, or call
``await client.aclose()`` explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pictograph._http.async_transport import AsyncTransport
from pictograph._internal.auth import resolve_api_key
from pictograph._internal.config import ClientConfig
from pictograph.aio.resources.annotation_comments import AsyncAnnotationComments
from pictograph.aio.resources.annotations import AsyncAnnotations
from pictograph.aio.resources.api_keys import AsyncApiKeys
from pictograph.aio.resources.auto_annotate import AsyncAutoAnnotate
from pictograph.aio.resources.batch import AsyncBatch
from pictograph.aio.resources.connectors import AsyncConnectors
from pictograph.aio.resources.credits import AsyncCredits
from pictograph.aio.resources.datasets import AsyncDatasets
from pictograph.aio.resources.deployments import AsyncDeployments
from pictograph.aio.resources.directories import AsyncDirectories
from pictograph.aio.resources.exports import AsyncExports
from pictograph.aio.resources.images import AsyncImages
from pictograph.aio.resources.model_evaluations import AsyncModelEvaluations
from pictograph.aio.resources.models import AsyncModels
from pictograph.aio.resources.notifications import AsyncNotifications
from pictograph.aio.resources.organizations import AsyncOrganizations
from pictograph.aio.resources.search import AsyncSearch
from pictograph.aio.resources.tasks import AsyncTasks
from pictograph.aio.resources.training import AsyncTraining
from pictograph.aio.resources.video import AsyncVideo
from pictograph.aio.resources.webhooks import AsyncWebhooks
from pictograph.aio.resources.workflows import AsyncWorkflows

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self


class AsyncClient:
    """Async Pictograph API client (:mod:`asyncio`).

    Args:
        api_key: API key (``pk_live_...``). Falls back to ``PICTOGRAPH_API_KEY``
            when ``None``.
        base_url: API root URL. Defaults to ``https://api.pictograph.io``;
            override via ``PICTOGRAPH_BASE_URL`` or this kwarg.
        timeout: Per-request timeout in seconds. Default 30.
        max_retries: Retry attempts for transient failures. ``0`` disables
            retries. Default 3.

    Attributes mirror :class:`pictograph.Client` exactly (``datasets``,
    ``images``, ``annotations``, ``exports``, ``training``, ``models``,
    ``deployments``, ``credits``, ``organizations``, ``directories``,
    ``batch``, ``search``, ``auto_annotate``, ``video``, ``connectors``,
    ``api_keys``, ``storage``, ``webhooks``, ``workflows``) - each an async resource.

    Raises:
        ConfigurationError: No API key is available from any source.
        ValidationError: ``timeout`` is non-positive or ``max_retries`` is negative.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        resolved_key = resolve_api_key(api_key)

        config_kwargs: dict[str, object] = {}
        if base_url is not None:
            config_kwargs["base_url"] = base_url
        if timeout is not None:
            config_kwargs["timeout"] = timeout
        if max_retries is not None:
            config_kwargs["max_retries"] = max_retries
        self._config = ClientConfig(**config_kwargs)  # type: ignore[arg-type]

        self._transport = AsyncTransport(self._config, api_key=resolved_key)

        self.datasets = AsyncDatasets(self._transport)
        self.images = AsyncImages(self._transport)
        self.annotations = AsyncAnnotations(self._transport)
        self.annotation_comments = AsyncAnnotationComments(self._transport)
        self.exports = AsyncExports(self._transport)
        self.training = AsyncTraining(self._transport)
        self.models = AsyncModels(self._transport)
        self.model_evaluations = AsyncModelEvaluations(self._transport)
        self.deployments = AsyncDeployments(self._transport)
        self.credits = AsyncCredits(self._transport)
        self.notifications = AsyncNotifications(self._transport)
        self.organizations = AsyncOrganizations(self._transport)
        self.directories = AsyncDirectories(self._transport)
        self.batch = AsyncBatch(self._transport)
        self.search = AsyncSearch(self._transport)
        self.auto_annotate = AsyncAutoAnnotate(self._transport)
        self.video = AsyncVideo(self._transport)
        self.connectors = AsyncConnectors(self._transport)
        self.api_keys = AsyncApiKeys(self._transport)
        self.webhooks = AsyncWebhooks(self._transport)
        self.workflows = AsyncWorkflows(self._transport)
        self.tasks = AsyncTasks(self._transport)

    # ───────────── lifecycle ─────────────

    async def aclose(self) -> None:
        """Release all sockets held by the underlying transport (idempotent)."""
        await self._transport.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ───────────── introspection ─────────────

    def __repr__(self) -> str:
        # Never include the API key in repr - would leak through logs / tracebacks.
        return f"AsyncClient(base_url={self._config.base_url!r})"
