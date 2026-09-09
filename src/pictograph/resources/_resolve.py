"""Turn the names a user knows into the ids the API needs.

Owner, 2026-07-31: *"any user-facing methods only run on dataset names / image
names / export names, etc. not ids, which are internal DB references."*

A uuid is a database detail. It appears in a user's code only because they had to
fetch it from somewhere first, which means an extra call, a variable to carry,
and an argument nobody can read back later. Names are what they typed into the
UI, so names are the interface.

An id is still ACCEPTED wherever a name is - detected syntactically, so passing
one costs nothing and a caller holding an id from a previous response never has
to unwrap it. That detection is deliberately by SHAPE rather than by trying the
name first and falling back on 404 (the older `Models.get_by_name` idiom): a
wrong guess there costs a whole round-trip and logs a spurious 404 server-side.

Names are resolved on EVERY call rather than cached. A cache here would be wrong
in the one case that matters - a dataset renamed between two calls in the same
session - and the resolution is a single indexed lookup.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from pictograph._http.transport import Transport

# Canonical 8-4-4-4-12 hex. Deliberately strict: a name that merely CONTAINS a
# uuid ("run 3f9c…-backup") must resolve as a name, not be mistaken for an id.
_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def looks_like_id(value: str) -> bool:
    """True when `value` is a bare uuid, i.e. already an id."""
    return bool(_UUID.match(value.strip()))


def dataset_id(transport: Transport, dataset: str) -> str:
    """`dataset` is a dataset NAME (or already an id) -> the id.

    Raises:
        NotFoundError: no dataset with that name in your organization.
    """
    if looks_like_id(dataset):
        return dataset
    from pictograph.resources.datasets import Datasets

    return Datasets(transport).get(dataset).id


def model_id(transport: Transport, model: str) -> str:
    """`model` is a model NAME (or already an id) -> the id."""
    if looks_like_id(model):
        return model
    from pictograph.resources.models import Models

    return Models(transport).get(model).id


def image_id(
    transport: Transport,
    dataset: str,
    image: str,
    *,
    directory_path: str | None = None,
) -> str:
    """`(dataset, filename)` -> the image id. `image` may already be an id.

    A user knows a dataset by name and an image by its FILENAME - the pair they
    can read off the grid. Resolving that used to be impossible without paging
    the entire dataset, so the developer images route grew an exact `filename`
    filter (2026-07-31) and this is one indexed lookup.

    `(virtual_directory_path, filename)` is the uniqueness pair, not filename alone,
    so the same name can legitimately live in two directories. That case RAISES and
    names the directories rather than silently taking the first match - picking one
    would edit the wrong image and nothing downstream would ever say so.

    Raises:
        NotFoundError: no image with that filename in the dataset.
        ValueError: the filename is ambiguous; pass `directory_path`.
    """
    if looks_like_id(image):
        return image

    from pictograph.exceptions import NotFoundError
    from pictograph.resources.images import Images

    matches = Images(transport).list(
        dataset, filename=image, directory_path=directory_path, limit=2
    )
    if not matches:
        where = f" in directory {directory_path!r}" if directory_path else ""
        raise NotFoundError(f"No image named {image!r} in dataset {dataset!r}{where}.")
    if len(matches) > 1:
        directories = sorted({m.directory_path or "/" for m in matches})
        raise ValueError(
            f"{image!r} exists in more than one directory of {dataset!r} "
            f"({', '.join(directories)}). Pass directory_path= to say which."
        )
    return matches[0].id


def deployment_id(transport: Transport, deployment: str) -> str:
    """`deployment` is a deployment NAME (or already an id) -> the id.

    A deployment's name is org-unique only among LIVE ones - the constraint is
    `uq_deployments_org_name_active`, partial on `status <> 'terminated'`, so
    that deleting a deployment frees its name for the next deploy. A name can
    therefore match one live row plus any number of terminated ones.

    So a live match always wins: that is the deployment a user means when they
    say the name today. Only when every match is terminated is one returned, and
    only if there is exactly one - otherwise this RAISES rather than guess which
    piece of history was meant.

    Raises:
        NotFoundError: no deployment with that name in your organization.
        ValueError: the name matches several terminated deployments.
    """
    if looks_like_id(deployment):
        return deployment

    from pictograph.exceptions import NotFoundError
    from pictograph.resources.deployments import Deployments

    # The name= filter is server-side, but re-filter here so this stays correct
    # against a backend that predates it: an older API ignores the unknown query
    # param and returns the whole page, which would otherwise look like a pile of
    # ambiguous matches. That makes the server filter a pure optimisation and
    # removes the ordering hazard between publishing the SDK and deploying core.
    matches = [
        d for d in Deployments(transport).list(name=deployment, limit=100) if d.name == deployment
    ]
    if not matches:
        raise NotFoundError(f"No deployment named {deployment!r} in your organization.")

    live = [d for d in matches if d.status != "terminated"]
    if len(live) == 1:
        return live[0].id
    if not live and len(matches) == 1:
        return matches[0].id
    if not live:
        raise ValueError(
            f"{deployment!r} matches {len(matches)} terminated deployments. "
            f"Pass the id of the one you mean."
        )
    # Belt and braces: the partial unique index should make this unreachable.
    raise ValueError(
        f"{deployment!r} matches {len(live)} live deployments "
        f"({', '.join(sorted(d.id for d in live))}). Pass the id of the one you mean."
    )


def deployment_ids(transport: Transport, deployments: Sequence[str]) -> list[str]:
    """Resolve a BATCH of deployment names/ids in ONE request, order preserved.

    The bulk verbs take a list, so resolving each entry through
    :func:`deployment_id` would cost one lookup per name - the N+1 that a bulk
    endpoint exists to avoid. An org holds few deployments, so one full listing
    resolves the whole batch. Entries that are already ids cost nothing, and a
    batch of nothing but ids issues no request at all.

    Raises:
        NotFoundError: a name in the batch matches no deployment.
        ValueError: a name matches several terminated deployments.
    """
    wanted = list(deployments)
    if all(looks_like_id(d) for d in wanted):
        return wanted

    from pictograph.exceptions import NotFoundError
    from pictograph.resources.deployments import Deployments

    by_name: dict[str, list[Any]] = {}
    for dep in Deployments(transport).iter():
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


def webhook_endpoint_id(transport: Transport, endpoint: str) -> str:
    """`endpoint` is a webhook URL (or already an id) -> the endpoint id.

    A webhook endpoint has no name column - what a user knows it by is the URL
    they registered, and a URL is unique within an organization, which makes that
    unambiguous. So the URL is the handle here, the way a path is for a directory.

    Matched EXACTLY. A trailing slash is a different URL to the constraint, so
    normalising one away could resolve to a genuinely different endpoint and
    rotate the wrong secret.

    Raises:
        NotFoundError: no endpoint registered at that URL in your organization.
    """
    if looks_like_id(endpoint):
        return endpoint

    from pictograph.exceptions import NotFoundError
    from pictograph.resources.webhooks import Webhooks

    for ep in Webhooks(transport).list():
        if ep.url == endpoint:
            return ep.id
    raise NotFoundError(f"No webhook endpoint registered at {endpoint!r} in your organization.")


def workflow_id(transport: Transport, workflow: str) -> str:
    """`workflow` is a workflow NAME (or already an id) -> the id.

    a name is unique within an organization, and the backend's create path
    auto-suffixes a colliding name to `name (2)` rather than rejecting it, so a
    name always identifies exactly one workflow. Resolved by listing, which is
    org-scoped and unpaginated.

    Raises:
        NotFoundError: no workflow with that name in your organization.
    """
    if looks_like_id(workflow):
        return workflow

    from pictograph.exceptions import NotFoundError
    from pictograph.resources.workflows import Workflows

    for wf in Workflows(transport).list():
        if wf.name == workflow:
            return wf.id
    raise NotFoundError(f"No workflow named {workflow!r} in your organization.")


def workflow_ids(transport: Transport, workflows: Sequence[str]) -> list[str]:
    """Resolve a BATCH of workflow names/ids in ONE request, order preserved.

    Same reasoning as the deployment batch resolver: a bulk verb must not become
    N+1. A batch of pure ids issues no request at all.

    Raises:
        NotFoundError: a name in the batch matches no workflow.
    """
    wanted = list(workflows)
    if all(looks_like_id(w) for w in wanted):
        return wanted

    from pictograph.exceptions import NotFoundError
    from pictograph.resources.workflows import Workflows

    by_name = {wf.name: wf.id for wf in Workflows(transport).list()}
    out: list[str] = []
    for entry in wanted:
        if looks_like_id(entry):
            out.append(entry)
        elif entry in by_name:
            out.append(by_name[entry])
        else:
            raise NotFoundError(f"No workflow named {entry!r} in your organization.")
    return out


def export_id(transport: Transport, dataset: str, export: str) -> str:
    """`(dataset, export name)` -> the export id. `export` may already be an id.

    An export name is unique within its dataset, not globally, so the pair is
    the handle - the same shape `Exports.get(dataset_name, export_name)` already
    uses. A bare id short-circuits and the dataset argument is then unused.
    """
    if looks_like_id(export):
        return export

    from pictograph.resources.exports import Exports

    return Exports(transport).get(dataset, export).id


def organization_id(transport: Transport, organization: str | None) -> str:
    """`organization` is an org NAME or SLUG (or an id, or None) -> the id.

    `None` means "the calling key's own organization", which is the only answer
    the API will accept anyway: an API key that names a DIFFERENT organization
    is rejected with 403 by `verify_org_access` before the handler runs. So the
    argument that used to be a required uuid could only ever hold one value the
    SDK can fetch for itself.

    A name or slug is therefore checked against your own organization rather
    than looked up globally - there is no endpoint that resolves someone else's
    org, and if there were, passing it would still 403.

    Raises:
        NotFoundError: the name/slug is not your organization.
    """
    # An id short-circuits BEFORE the me() round-trip - the escape hatch must
    # not cost a lookup, which is the whole reason detection is by shape.
    if organization is not None and looks_like_id(organization):
        return organization

    from pictograph.resources.organizations import Organizations

    mine = Organizations(transport).me()
    if organization is None or organization in (mine.name, mine.slug):
        return mine.id

    from pictograph.exceptions import NotFoundError

    raise NotFoundError(
        f"{organization!r} is not your organization ({mine.name!r} / {mine.slug!r}). "
        f"An API key can only manage keys for its own organization."
    )


def directory_id(transport: Transport, dataset: str, directory: str) -> str:
    """`(dataset, directory path)` -> the directory id. `directory` may already be an id.

    A directory is known by its PATH (`/train/cars`), which is what the grid shows
    and what every other directory argument in this SDK already takes. The backend
    addresses stats/rename/delete by uuid, so this maps one to the other.

    Resolved by listing the dataset's directories rather than by a lookup route:
    there is no by-path endpoint, and a dataset holds a couple of dozen directories
    at most, so this is one small request instead of a schema change.

    Raises:
        NotFoundError: no directory at that path in the dataset.
    """
    if looks_like_id(directory):
        return directory

    from pictograph.exceptions import NotFoundError
    from pictograph.resources.directories import Directories

    wanted = directory if directory.startswith("/") else f"/{directory}"
    for f in Directories(transport).list(dataset):
        if f.full_path == wanted:
            return f.id
    raise NotFoundError(f"No directory at path {wanted!r} in dataset {dataset!r}.")


def image_segment(
    transport: Transport,
    image: str,
    *,
    directory_path: str | None = None,
) -> str:
    """The `{image_path}` segment for the per-image routes.

    Those routes address an image as `{dataset}/{image_path}`, where
    `image_path` folds the directory INTO the filename - `val/000000546325.jpg`.
    Verified against production: with the directory it is 200, without it 404,
    and a UUID in that slot is 404 too.

    So this is not `image_id`'s job and must not be built from it. The SDK used
    to resolve the filename to a UUID and call `/annotations/{uuid}` and
    `/images/{uuid}`, paths the API no longer serves - which is why every
    per-image annotation read, image delete, review and split call was broken.

    A filename needs NO round-trip (the point of server-side resolution). Only a
    UUID does, because the route cannot take one: it is looked up once to
    recover the directory and filename it stands for.
    """
    if looks_like_id(image):
        row = transport.request("GET", f"/api/v1/developer/images/{image}")
        data = row.get("data", row) if isinstance(row, dict) else {}
        directory = (data.get("directory_path") or "").strip("/")
        filename = data.get("filename") or ""
        return f"{directory}/{filename}" if directory else filename
    directory = (directory_path or "").strip("/")
    return f"{directory}/{image}" if directory else image
