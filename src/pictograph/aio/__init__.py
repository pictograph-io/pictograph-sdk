"""Async (:mod:`asyncio`) client surface for the Pictograph SDK.

The async twin of the synchronous top-level package. :class:`AsyncClient` mirrors
:class:`pictograph.Client` method-for-method, with every I/O method a coroutine
and every ``iter`` accessor returning an
:class:`~pictograph._http.pagination.AsyncOffsetPager` you ``async for`` over::

    import asyncio
    from pictograph import AsyncClient


    async def main() -> None:
        async with AsyncClient() as client:
            for ds in await client.datasets.list(limit=5):
                print(ds.name)


    asyncio.run(main())

:class:`AsyncClient` is also re-exported at the top level as
``pictograph.AsyncClient``. The async resource classes live here for callers who
want to type-annotate against them directly.
"""

from __future__ import annotations

from pictograph.aio.client import AsyncClient
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
from pictograph.aio.resources.models import AsyncModels
from pictograph.aio.resources.organizations import AsyncOrganizations
from pictograph.aio.resources.search import AsyncSearch
from pictograph.aio.resources.tasks import AsyncTasks
from pictograph.aio.resources.training import AsyncTraining
from pictograph.aio.resources.video import AsyncVideo
from pictograph.aio.resources.webhooks import AsyncWebhooks
from pictograph.aio.resources.workflows import AsyncWorkflows

__all__ = [
    "AsyncAnnotations",
    "AsyncApiKeys",
    "AsyncAutoAnnotate",
    "AsyncBatch",
    "AsyncClient",
    "AsyncConnectors",
    "AsyncCredits",
    "AsyncDatasets",
    "AsyncDeployments",
    "AsyncDirectories",
    "AsyncExports",
    "AsyncImages",
    "AsyncModels",
    "AsyncOrganizations",
    "AsyncSearch",
    "AsyncTasks",
    "AsyncTraining",
    "AsyncVideo",
    "AsyncWebhooks",
    "AsyncWorkflows",
]
