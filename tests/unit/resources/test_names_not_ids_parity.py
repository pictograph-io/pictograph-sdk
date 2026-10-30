"""A converted method must lead with the SAME names on both clients.

The names-not-ids sweep converted `images`, `annotations` and `directories` on the
sync client and left the async twins addressing rows by uuid. That is not a
cosmetic gap: `client.directories.list("road-signs")` worked while
`aio.directories.list("road-signs")` sent the literal string "road-signs" into the
URL path as a project id. Two clients, two contracts, one SDK.

`test_resource_orchestration_symmetry.TWINNED` was supposed to catch exactly
this, but it is a hand-maintained five-entry tuple covering the orchestrations,
so nothing about `list`/`get`/`delete` was ever compared.

This is the mechanical version. `CONVERTED` is the single source of truth for
the sweep: every row states the leading parameters the method must take, and the
test asserts BOTH the sync class and its async twin honour it. A resource cannot
be half-converted without a red test, and the table doubles as the sweep's
progress ledger.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

# (module stem, method, leading parameter names) - applied to BOTH
# `pictograph.resources.<stem>` and `pictograph.aio.resources.<stem>`.
CONVERTED: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    # --- shipped 2026-07-31 (sync), async twins landed with this table ---
    ("images", "list", ("dataset",)),
    ("images", "iter", ("dataset",)),
    ("images", "upload", ("dataset", "file_path")),
    ("images", "bulk_upload", ("dataset", "file_paths")),
    ("images", "bulk_tag", ("dataset", "images", "tags")),
    ("images", "assign_splits", ("dataset",)),
    ("images", "upload_from_directory", ("dataset", "source_path")),
    ("annotations", "get", ("dataset", "image")),
    ("annotations", "save", ("dataset", "image", "annotations")),
    ("annotations", "delete", ("dataset", "image")),
    ("annotations", "rename_class", ("dataset", "class_name", "new_name")),
    ("annotations", "merge_class", ("dataset", "class_name", "into")),
    ("annotations", "delete_class", ("dataset", "class_name")),
    ("directories", "list", ("dataset",)),
    ("directories", "tree", ("dataset",)),
    ("directories", "stats", ("dataset", "directory")),
    ("directories", "delete", ("dataset", "directory")),
    ("directories", "create", ("dataset", "directory")),
    ("directories", "rename", ("dataset", "directory", "new_name")),
    # --- deployments ---
    ("deployments", "get", ("deployment",)),
    ("deployments", "pause", ("deployment",)),
    ("deployments", "resume", ("deployment",)),
    ("deployments", "update", ("deployment",)),
    ("deployments", "delete", ("deployment",)),
    ("deployments", "create", ("model",)),
    ("deployments", "wait_until_active", ("deployment",)),
    ("deployments", "bulk_pause", ("deployments",)),
    ("deployments", "bulk_resume", ("deployments",)),
    ("deployments", "bulk_delete", ("deployments",)),
    # --- webhooks (no name column; the registered URL is the handle) ---
    ("webhooks", "get", ("endpoint",)),
    ("webhooks", "update", ("endpoint",)),
    ("webhooks", "delete", ("endpoint",)),
    ("webhooks", "test", ("endpoint",)),
    ("webhooks", "rotate_secret", ("endpoint",)),
    # --- workflows ---
    ("workflows", "get", ("workflow",)),
    ("workflows", "update", ("workflow",)),
    ("workflows", "delete", ("workflow",)),
    ("workflows", "run", ("workflow",)),
    ("workflows", "bulk_delete", ("workflows",)),
    # --- model_evaluations, search, models ---
    ("model_evaluations", "create", ("model", "dataset", "export")),
    ("model_evaluations", "evaluate", ("model", "dataset", "export")),
    ("search", "by_similarity", ("dataset", "image")),
    ("models", "predict", ("model",)),
    # --- the image-level stragglers ---
    ("images", "get", ("dataset", "image")),
    ("images", "download", ("dataset", "image", "output_path")),
    ("images", "delete", ("dataset", "image")),
    ("images", "review", ("dataset", "image", "action")),
    ("images", "set_split", ("dataset", "image", "split")),
    ("api_keys", "create", ("name",)),
    ("annotation_comments", "list", ("dataset", "image")),
    ("annotation_comments", "create", ("dataset", "image")),
    # --- B603 slice 4 (SDK 1.69.176): one word per primitive, the address first ---
    ("exports", "get", ("dataset", "export")),
    ("exports", "download", ("dataset", "export", "output_path")),
    ("exports", "delete", ("dataset", "export")),
    ("exports", "wait_for_completion", ("dataset", "export")),
    ("training", "create", ("dataset", "export")),
    ("training", "quote", ("dataset", "export")),
    ("auto_annotate", "box", ("dataset", "image")),
    ("auto_annotate", "point", ("dataset", "image")),
    ("auto_annotate", "text", ("dataset", "image")),
    ("auto_annotate", "batch", ("dataset", "images")),
    ("batch", "copy_to_dataset", ("dataset", "images", "into", "directory")),
    ("video", "probe", ("video",)),
    ("video", "extract_frames", ("dataset", "video", "directory")),
    ("models", "download_file", ("model", "filename")),
)

# Keyword-only filters that must take a name, checked separately from the
# leading-parameter table because they are keyword-only by design.
CONVERTED_KEYWORDS: tuple[tuple[str, str, str, str], ...] = (
    # (module stem, method, keyword that must exist, keyword that must be gone)
    ("deployments", "list", "model", "model_id"),
    ("deployments", "iter", "model", "model_id"),
    ("webhooks", "deliveries", "endpoint", "endpoint_id"),
    ("model_evaluations", "list", "model", "model_id"),
    ("model_evaluations", "iter", "model", "model_id"),
    ("api_keys", "list", "organization", "organization_id"),
    ("api_keys", "create", "organization", "organization_id"),
)

# Converted methods that exist ONLY on the sync client, with the reason. These
# still have to take the name; they simply have no twin to compare against.
CONVERTED_SYNC_ONLY: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "models",
        "load",
        ("model",),
        "local inference is CPU-bound ONNX work, so there is no async load",
    ),
    ("datasets", "as_pytorch", ("dataset",), "hands back a torch Dataset; the loop is synchronous"),
    ("images", "augment", ("dataset",), "CPU-bound Pillow work per image"),
    ("images", "tile", ("dataset",), "CPU-bound Pillow work per image"),
)

# Resources whose async twin genuinely does not exist. Empty on purpose: every
# resource in CONVERTED has one today. Adding an entry should require saying why.
_NO_ASYNC_TWIN: dict[str, str] = {}


def _sync_class(stem: str) -> type:
    import importlib

    module = importlib.import_module(f"pictograph.resources.{stem}")
    return getattr(module, stem.title().replace("_", ""))


def _async_class(stem: str) -> type:
    import importlib

    module = importlib.import_module(f"pictograph.aio.resources.{stem}")
    return getattr(module, "Async" + stem.title().replace("_", ""))


def _leading(cls: type, method: str, count: int) -> Sequence[str]:
    params = list(inspect.signature(getattr(cls, method)).parameters)[1:]
    return params[:count]


@pytest.mark.parametrize(("stem", "method", "leading"), CONVERTED)
def test_sync_leads_with_the_names(stem: str, method: str, leading: tuple[str, ...]) -> None:
    assert tuple(_leading(_sync_class(stem), method, len(leading))) == leading


@pytest.mark.parametrize(("stem", "method", "leading"), CONVERTED)
def test_async_twin_leads_with_the_same_names(
    stem: str, method: str, leading: tuple[str, ...]
) -> None:
    """The bug this file exists for: a sync-only conversion.

    Before this test, `AsyncDirectories.list` took `project_id` while `Directories.list`
    took `dataset`, so the same call was correct on one client and a 404 on
    the other.
    """
    if stem in _NO_ASYNC_TWIN:
        pytest.skip(_NO_ASYNC_TWIN[stem])
    assert tuple(_leading(_async_class(stem), method, len(leading))) == leading


@pytest.mark.parametrize(("stem", "method", "leading"), CONVERTED)
def test_no_parallel_id_keyword_survives(stem: str, method: str, leading: tuple[str, ...]) -> None:
    """ "There must not be two ways to say the same thing."

    A conversion that ADDS `dataset` while leaving `dataset_id=` in place
    passes the two tests above and still ships the ambiguity the sweep exists to
    remove - so the old parameter has to be gone, not merely demoted.
    """
    retired = {f"{name.removesuffix('_name')}_id" for name in leading if name.endswith("_name")}
    # A bare `deployment`/`model` parameter retires `deployment_id`/`model_id`
    # the same way `dataset` retires `dataset_id`.
    retired |= {f"{name}_id" for name in leading if not name.endswith(("_name", "_path", "s"))}
    retired |= {f"{name[:-1]}_ids" for name in leading if name.endswith("s")}
    for cls in (_sync_class(stem), _async_class(stem)):
        if stem in _NO_ASYNC_TWIN and cls is not _sync_class(stem):
            continue
        params = set(inspect.signature(getattr(cls, method)).parameters)
        assert not (params & retired), (
            f"{cls.__name__}.{method} still accepts {sorted(params & retired)} "
            f"alongside {list(leading)}"
        )


@pytest.mark.parametrize(("stem", "method", "keeps", "retires"), CONVERTED_KEYWORDS)
def test_keyword_filters_take_a_name_on_both_clients(
    stem: str, method: str, keeps: str, retires: str
) -> None:
    """A keyword FILTER is user-facing surface too.

    `deployments.list(model_id=...)` made a caller fetch a model just to filter
    by it - the same defect as a positional id, one argument to the right.
    """
    for cls in (_sync_class(stem), _async_class(stem)):
        params = set(inspect.signature(getattr(cls, method)).parameters)
        assert keeps in params, f"{cls.__name__}.{method} has no {keeps!r} filter"
        assert retires not in params, f"{cls.__name__}.{method} still accepts {retires!r}"


@pytest.mark.parametrize(("stem", "method", "leading", "why"), CONVERTED_SYNC_ONLY)
def test_sync_only_methods_still_lead_with_the_names(
    stem: str, method: str, leading: tuple[str, ...], why: str
) -> None:
    """No async twin (`why`), but the name contract still applies."""
    assert tuple(_leading(_sync_class(stem), method, len(leading))) == leading
    assert not hasattr(_async_class(stem), method), (
        f"Async{stem.title()}.{method} exists now - move this row into CONVERTED ({why})"
    )


# Parameter names that end like a property of a primitive but are allowed: each is the
# primitive's own new value or a list of class names, never a second way to address it.
_ALLOWED_SUFFIXED = {
    "new_name",
    "class_name",
    "class_names",
    "display_name",
    # VOC XML documents keyed by image filename - names the mapping, not an address.
    "xml_by_filename",
}


def test_no_public_method_takes_an_id_or_a_name_suffix() -> None:
    """R1 (REFERENCE_LANGUAGE.md): a primitive is its bare name - ``dataset=``,
    ``export=``, ``image=`` - never ``*_id`` / ``*_name`` / ``*_filename``. Walks every
    public method of every resource on both clients, so a new one cannot bring the old
    vocabulary back."""
    import re

    from pictograph import AsyncClient, Client

    bad = re.compile(r"(_id|_ids|_name|_names|_filename|_filenames)$")
    found = []
    for client in (Client(api_key="pk_live_x"), AsyncClient(api_key="pk_live_x")):
        for attr in dir(client):
            res = getattr(client, attr, None)
            if attr.startswith("_") or not hasattr(res, "_transport"):
                continue
            for name in dir(res):
                fn = getattr(res, name, None)
                if name.startswith("_") or not callable(fn):
                    continue
                for p in inspect.signature(fn).parameters:
                    if bad.search(p) and p not in _ALLOWED_SUFFIXED:
                        found.append(f"{type(res).__name__}.{name}({p})")
    assert not found, found
