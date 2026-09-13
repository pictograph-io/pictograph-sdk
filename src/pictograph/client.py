"""Top-level :class:`Client` for the Pictograph SDK.

The Client owns a :class:`pictograph._http.transport.Transport` and exposes
resource accessors:

- :attr:`Client.datasets` - list, get, download datasets
- :attr:`Client.images` - get metadata, upload, download, delete images
- :attr:`Client.annotations` - read, save, delete annotations (canonical JSON)
- :attr:`Client.exports` - create, list, get, download, delete exports
- :attr:`Client.training` - create / list / get / cancel training runs
- :attr:`Client.models` - list / get / download / delete trained models
- :attr:`Client.credits` - balance / history / cost estimates for gating
- :attr:`Client.notifications` - poll the org job-event feed (training/export done)
- :attr:`Client.organizations` - current org info, members, invites
- :attr:`Client.batch` - bulk move / copy / delete / update on images
- :attr:`Client.search` - visual similarity + auto-tag search
- :attr:`Client.auto_annotate` - SAM3 point/box/text + batch auto-annotation
- :attr:`Client.video` - upload videos and extract frames into a dataset
- :attr:`Client.connectors` - V7 / Roboflow dataset import
- :attr:`Client.api_keys` - manage API keys

Resource accessors are eager-instantiated on Client construction. They share
the single underlying Transport (and therefore the single httpx connection
pool); creating multiple Clients is uncommon and only needed when talking to
two different API endpoints in the same process.

Resolution order for credentials and config - first non-empty value wins:

1. Explicit ``Client(api_key=..., base_url=..., timeout=..., max_retries=...)``.
2. ``PICTOGRAPH_*`` environment variables.
3. Built-in defaults.

Use as a context manager to guarantee socket cleanup on exit::

    from pictograph import Client

    with Client() as client:
        for ds in client.datasets.iter():
            print(ds.name)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pictograph._http.transport import Transport
from pictograph._internal.auth import resolve_api_key
from pictograph._internal.config import ClientConfig
from pictograph.resources.annotation_comments import AnnotationComments
from pictograph.resources.annotations import Annotations
from pictograph.resources.api_keys import ApiKeys
from pictograph.resources.auto_annotate import AutoAnnotate
from pictograph.resources.batch import Batch
from pictograph.resources.connectors import Connectors
from pictograph.resources.credits import Credits
from pictograph.resources.datasets import Datasets
from pictograph.resources.deployments import Deployments
from pictograph.resources.directories import Directories
from pictograph.resources.exports import Exports
from pictograph.resources.images import Images
from pictograph.resources.model_evaluations import ModelEvaluations
from pictograph.resources.models import Models
from pictograph.resources.notifications import Notifications
from pictograph.resources.organizations import Organizations
from pictograph.resources.search import Search
from pictograph.resources.tasks import Tasks
from pictograph.resources.training import Training
from pictograph.resources.video import Video
from pictograph.resources.webhooks import Webhooks
from pictograph.resources.workflows import Workflows

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self


class Client:
    """Pictograph API client.

    Args:
        api_key: API key (``pk_live_...``). Falls back to the
            ``PICTOGRAPH_API_KEY`` environment variable when ``None``.
        base_url: API root URL. Defaults to ``https://api.pictograph.io``;
            override via the ``PICTOGRAPH_BASE_URL`` env var or this kwarg.
        timeout: Per-request timeout in seconds. Default 30.
        max_retries: Number of retry attempts for transient failures.
            ``0`` disables retries entirely. Default 3.

    Attributes:
        datasets: :class:`pictograph.resources.datasets.Datasets`
        images: :class:`pictograph.resources.images.Images`
        annotations: :class:`pictograph.resources.annotations.Annotations`
        exports: :class:`pictograph.resources.exports.Exports`
        training: :class:`pictograph.resources.training.Training`
        models: :class:`pictograph.resources.models.Models`
        credits: :class:`pictograph.resources.credits.Credits`
        organizations: :class:`pictograph.resources.organizations.Organizations`
        batch: :class:`pictograph.resources.batch.Batch`
        search: :class:`pictograph.resources.search.Search`
        auto_annotate: :class:`pictograph.resources.auto_annotate.AutoAnnotate`
        video: :class:`pictograph.resources.video.Video`
        connectors: :class:`pictograph.resources.connectors.Connectors`
        api_keys: :class:`pictograph.resources.api_keys.ApiKeys`

    Raises:
        ConfigurationError: No API key is available from any source.
        ValidationError: ``timeout`` is non-positive or ``max_retries`` is
            negative.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        # Resolve credentials first so a missing key surfaces immediately,
        # before we touch the network or build any sockets.
        resolved_key = resolve_api_key(api_key)

        # ClientConfig pulls remaining values from env vars when kwargs are
        # None, then validates timeout/max_retries via Pydantic.
        config_kwargs: dict[str, object] = {}
        if base_url is not None:
            config_kwargs["base_url"] = base_url
        if timeout is not None:
            config_kwargs["timeout"] = timeout
        if max_retries is not None:
            config_kwargs["max_retries"] = max_retries
        self._config = ClientConfig(**config_kwargs)  # type: ignore[arg-type]

        self._transport = Transport(self._config, api_key=resolved_key)

        # Resource accessors. Constructed eagerly: each is a thin wrapper
        # holding only a transport reference, so allocation cost is trivial
        # and lazy property dispatch would buy nothing.
        self.datasets = Datasets(self._transport)
        self.images = Images(self._transport)
        self.annotations = Annotations(self._transport)
        self.annotation_comments = AnnotationComments(self._transport)
        self.exports = Exports(self._transport)
        self.training = Training(self._transport)
        self.models = Models(self._transport)
        self.model_evaluations = ModelEvaluations(self._transport)
        self.deployments = Deployments(self._transport)
        self.credits = Credits(self._transport)
        self.notifications = Notifications(self._transport)
        self.organizations = Organizations(self._transport)
        self.directories = Directories(self._transport)
        self.batch = Batch(self._transport)
        self.search = Search(self._transport)
        self.auto_annotate = AutoAnnotate(self._transport)
        self.video = Video(self._transport)
        self.connectors = Connectors(self._transport)
        self.api_keys = ApiKeys(self._transport)
        self.webhooks = Webhooks(self._transport)
        self.workflows = Workflows(self._transport)
        self.tasks = Tasks(self._transport)

    # ───────────── lifecycle ─────────────

    def close(self) -> None:
        """Release all sockets held by the underlying transport.

        Idempotent: safe to call multiple times. Use the context-manager form
        for automatic cleanup in normal control flow.
        """
        self._transport.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # ───────────── introspection ─────────────

    def __repr__(self) -> str:
        # Never include the API key in repr - would leak through logs / tracebacks.
        return f"Client(base_url={self._config.base_url!r})"
