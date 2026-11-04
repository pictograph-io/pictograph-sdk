"""Turn the names a user knows into the ids the API needs - async twin.

The async mirror of :mod:`pictograph.resources._resolve`. It exists because the
sync resolver reaches for the sync resources (``Datasets(transport).get(...)``)
and a coroutine cannot call those without blocking the loop.

Keeping the rules in one place is what matters, so the SHAPE test
(``looks_like_id``) is imported from the sync module rather than restated -
there must not be two answers to "is this a uuid". Only the lookups, which are
the part that actually has to await, are duplicated here.

Every docstring rule from the sync module applies verbatim: detection is by
shape so a caller holding an id pays no lookup, ambiguity RAISES and names the
candidates, and nothing is cached because a cache is wrong the moment something
is renamed mid-session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pictograph.resources._resolve import IMAGE_PATH_HELP, image_path, looks_like_id

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from pictograph._http.async_transport import AsyncTransport

__all__ = [
    "IMAGE_PATH_HELP",
    "dataset_id",
    "deployment_id",
    "deployment_ids",
    "directory_id",
    "export_id",
    "image_path",
    "looks_like_id",
    "model_id",
    "organization_id",
    "webhook_endpoint_id",
    "workflow_id",
    "workflow_ids",
]


async def dataset_id(transport: AsyncTransport, dataset: str) -> str:
    """`dataset` is a dataset NAME (or already an id) -> the id.

    Raises:
        NotFoundError: no dataset with that name in your organization.
    """
    if looks_like_id(dataset):
        return dataset
    from pictograph.aio.resources.datasets import AsyncDatasets

    return (await AsyncDatasets(transport).get(dataset)).id


async def model_id(transport: AsyncTransport, model: str) -> str:
    """`model` is a model NAME (or already an id) -> the id."""
    if looks_like_id(model):
        return model
    from pictograph.aio.resources.models import AsyncModels

    return (await AsyncModels(transport).get(model)).id


async def deployment_id(transport: AsyncTransport, deployment: str) -> str:
    """`deployment` is a deployment NAME (or already an id) -> the id.

    See the sync twin for why a live match wins over a terminated one: the name
    constraint is partial on `status <> 'terminated'`, so a name can match one
    live deployment plus any number of dead ones.

    Raises:
        NotFoundError: no deployment with that name in your organization.
        ValueError: the name matches several terminated deployments.
    """
    if looks_like_id(deployment):
        return deployment

    from pictograph.aio.resources.deployments import AsyncDeployments
    from pictograph.exceptions import NotFoundError

    # Re-filtered client-side for the same reason as the sync twin: an older
    # backend ignores the name= param, and the whole page would then read as
    # ambiguous. The server filter stays an optimisation, not a dependency.
    fetched = await AsyncDeployments(transport).list(name=deployment, limit=100)
    matches = [d for d in fetched if d.name == deployment]
    if not matches:
        raise NotFoundError(f"No deployment named {deployment!r} in your organization.")

    live = [d for d in matches if d.status != "terminated"]
    if len(live) == 1:
        return live[0].id
    if not live and len(matches) == 1:
        return matches[0].id
    pool = live or matches
    raise ValueError(
        f"{deployment!r} matches {len(pool)} deployments "
        f"({', '.join(sorted(d.id for d in pool))}). Pass the id of the one you mean."
    )


async def deployment_ids(transport: AsyncTransport, deployments: Sequence[str]) -> list[str]:
    """Resolve a BATCH of deployment names/ids in ONE request, order preserved.

    The async twin of the sync batch resolver - see it for why the bulk verbs
    must not resolve one name per entry.
    """
    wanted = list(deployments)
    if all(looks_like_id(d) for d in wanted):
        return wanted

    from pictograph.aio.resources.deployments import AsyncDeployments
    from pictograph.exceptions import NotFoundError

    by_name: dict[str, list[Any]] = {}
    async for dep in AsyncDeployments(transport).iter():
        by_name.setdefault(dep.name, []).append(dep)

    out: list[str] = []
    for entry in wanted:
        if looks_like_id(entry):
            out.append(entry)
            continue
        rows = by_name.get(entry, [])
        if not rows:
            raise NotFoundError(f"No deployment named {entry!r} in your organization.")
        live = [d for d in rows if d.status != "terminated"]
        if len(live) == 1:
            out.append(live[0].id)
        elif not live and len(rows) == 1:
            out.append(rows[0].id)
        else:
            pool = live or rows
            raise ValueError(
                f"{entry!r} matches {len(pool)} deployments "
                f"({', '.join(sorted(d.id for d in pool))}). Pass ids instead."
            )
    return out


async def webhook_endpoint_id(transport: AsyncTransport, endpoint: str) -> str:
    """`endpoint` is a webhook URL (or already an id) -> the endpoint id.

    See the sync twin: a webhook endpoint has no name, so the registered URL is
    the handle, matched exactly.

    Raises:
        NotFoundError: no endpoint registered at that URL in your organization.
    """
    if looks_like_id(endpoint):
        return endpoint

    from pictograph.aio.resources.webhooks import AsyncWebhooks
    from pictograph.exceptions import NotFoundError

    for ep in await AsyncWebhooks(transport).list():
        if ep.url == endpoint:
            return ep.id
    raise NotFoundError(f"No webhook endpoint registered at {endpoint!r} in your organization.")


async def workflow_id(transport: AsyncTransport, workflow: str) -> str:
    """`workflow` is a workflow NAME (or already an id) -> the id.

    Raises:
        NotFoundError: no workflow with that name in your organization.
    """
    if looks_like_id(workflow):
        return workflow

    from pictograph.aio.resources.workflows import AsyncWorkflows
    from pictograph.exceptions import NotFoundError

    for wf in await AsyncWorkflows(transport).list():
        if wf.name == workflow:
            return wf.id
    raise NotFoundError(f"No workflow named {workflow!r} in your organization.")


async def workflow_ids(transport: AsyncTransport, workflows: Sequence[str]) -> list[str]:
    """Resolve a BATCH of workflow names/ids in ONE request, order preserved."""
    wanted = list(workflows)
    if all(looks_like_id(w) for w in wanted):
        return wanted

    from pictograph.aio.resources.workflows import AsyncWorkflows
    from pictograph.exceptions import NotFoundError

    by_name = {wf.name: wf.id for wf in await AsyncWorkflows(transport).list()}
    out: list[str] = []
    for entry in wanted:
        if looks_like_id(entry):
            out.append(entry)
        elif entry in by_name:
            out.append(by_name[entry])
        else:
            raise NotFoundError(f"No workflow named {entry!r} in your organization.")
    return out


async def export_id(transport: AsyncTransport, dataset: str, export: str) -> str:
    """`(dataset, export name)` -> the export id. `export` may already be an id."""
    if looks_like_id(export):
        return export

    from pictograph.aio.resources.exports import AsyncExports

    return (await AsyncExports(transport).get(dataset, export)).id


async def organization_id(transport: AsyncTransport, organization: str | None) -> str:
    """`organization` is an org NAME or SLUG (or an id, or None) -> the id.

    See the sync twin: `None` means your own organization, which is the only one
    the API will accept.

    Raises:
        NotFoundError: the name/slug is not your organization.
    """
    # An id short-circuits BEFORE the me() round-trip - the escape hatch must
    # not cost a lookup, which is the whole reason detection is by shape.
    if organization is not None and looks_like_id(organization):
        return organization

    from pictograph.aio.resources.organizations import AsyncOrganizations

    mine = await AsyncOrganizations(transport).me()
    if organization is None or organization in (mine.name, mine.slug):
        return mine.id

    from pictograph.exceptions import NotFoundError

    raise NotFoundError(
        f"{organization!r} is not your organization ({mine.name!r} / {mine.slug!r}). "
        f"An API key can only manage keys for its own organization."
    )


async def directory_id(transport: AsyncTransport, dataset: str, directory: str) -> str:
    """`(dataset, directory path)` -> the directory id. `directory` may already be an id.

    Raises:
        NotFoundError: no directory at that path in the dataset.
    """
    if looks_like_id(directory):
        return directory

    from pictograph.aio.resources.directories import AsyncDirectories
    from pictograph.exceptions import NotFoundError

    wanted = directory if directory.startswith("/") else f"/{directory}"
    for f in await AsyncDirectories(transport).list(dataset):
        if f.full_path == wanted:
            return f.id
    raise NotFoundError(f"No directory at path {wanted!r} in dataset {dataset!r}.")
