"""Every sync resource method and its async twin take the same arguments.

``test_resource_orchestration_symmetry.TWINNED`` asserts this too, but from a
HAND-MAINTAINED five-entry tuple covering the orchestrations. Nothing about the
other ~200 methods was ever compared, and that is where the drift lived:

* ``client.directories.list("road-signs")`` worked while
  ``aio.directories.list("road-signs")`` put the literal string into the URL path as
  a project id - the names-not-ids sweep shipped sync-only.
* ``await client.exports.create(..., organize_by_split=True)`` raised
  ``TypeError``; async callers could not produce a split-organized ZIP at all.
* ``aio.models.download`` had neither ``precision`` nor ``target`` and its
  ``format`` literal omitted ``pte`` and ``engine``, so async callers could not
  fetch those artifacts, could not ask for fp16, and could not recover from the
  ``ConflictError`` that exists to say "name a target" - because there was no
  argument to name one with.

So this one DISCOVERS the pairs instead of listing them: every
``pictograph.resources.<mod>.<Cls>`` with a ``pictograph.aio.resources.<mod>.Async<Cls>``
is compared method by method, parameter names and defaults. Asymmetries that are
real are enumerated below with the reason, so a new one cannot appear quietly.

Two clients, one contract.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

import pytest

import pictograph.resources as _sync_pkg

# Sync-only methods, with the reason no async twin exists. Four are CPU- or
# GPU-bound: the wall time is Pillow or a training/inference job, not HTTP, so
# asyncio buys nothing. The fifth (AutoAnnotate.dataset) IS an API call and so is
# a genuine gap flagged for the owner - see its entry. Adding an entry should
# require saying why.
SYNC_ONLY: dict[tuple[str, str], str] = {
    # NOT purely "asyncio buys nothing" like the four below: this one is an API
    # call, and while polling one SAM3 batch job means an async twin would not
    # make the job itself finish sooner, a twin WOULD stop the call blocking the
    # caller's event loop for the whole run. Whether to add it is an owner call;
    # recorded here so it stays a deliberate gap, not silent drift, exactly like
    # the Images.bulk_upload knob below.
    ("AutoAnnotate", "dataset"): "API call, poll-bound; async twin is an owner call",
    ("Images", "augment"): "CPU-bound Pillow work per image; asyncio would not overlap it",
    ("Images", "tile"): "same - the cost is Pillow, and re-runs must stay deterministic",
    ("Datasets", "as_pytorch"): (
        "hands back a torch Dataset for a training loop; the loop is synchronous "
        "and the DataLoader does its own worker fan-out"
    ),
    ("Models", "load"): (
        "builds a local inference engine from downloaded weights - the work is "
        "torch/ORT in-process, and the resulting model.predict() is sync"
    ),
}

# Parameters that legitimately exist on ONE side of a twin, keyed by
# (class, method, parameter). Anything else is a defect this test catches.
JUSTIFIED: dict[tuple[str, str, str], str] = {
    ("Images", "upload_from_directory", "parallel"): (
        "sync only: opting out of the thread pool. The async twin's concurrency is "
        "asyncio.gather, which max_workers=1 already serialises."
    ),
    ("Annotations", "import_coco", "max_concurrency"): (
        "async only: how many bulk_save chunks are in flight. The sync importer "
        "issues them one after another, so it has nothing to bound."
    ),
    ("Annotations", "import_pascal_voc", "max_concurrency"): "same as import_coco",
    ("Annotations", "import_yolo", "max_concurrency"): "same as import_coco",
    ("Storage", "wait_for_import", "sleep"): (
        "sync only: a test seam for the blocking sleep. The async twin awaits "
        "asyncio.sleep, which a test controls with its own event loop."
    ),
    # INCONSISTENT, deliberately recorded rather than silently tolerated: this
    # is the same knob under two names, and the ADJACENT method
    # (Images.upload_from_directory) calls it max_workers on BOTH sides. Porting sync
    # to async therefore breaks on one method and not its neighbour. Renaming a
    # public parameter is an API decision, so it is flagged for the owner rather
    # than changed here.
    ("Images", "bulk_upload", "max_workers"): "sync name for the concurrency bound (see below)",
    ("Images", "bulk_upload", "max_concurrency"): (
        "async name for the SAME knob - inconsistent with upload_from_directory, "
        "which uses max_workers on both sides. Flagged for the owner."
    ),
}


def _class_pairs() -> list[tuple[str, type, type]]:
    """Every (name, sync class, async class) discovered under resources/."""
    pairs: list[tuple[str, type, type]] = []
    for mod in pkgutil.iter_modules(_sync_pkg.__path__):
        if mod.name.startswith("_"):
            continue
        sync_mod = importlib.import_module(f"pictograph.resources.{mod.name}")
        try:
            aio_mod = importlib.import_module(f"pictograph.aio.resources.{mod.name}")
        except ModuleNotFoundError:
            continue
        for name, cls in vars(sync_mod).items():
            if not (inspect.isclass(cls) and cls.__module__ == sync_mod.__name__):
                continue
            twin = getattr(aio_mod, f"Async{name}", None)
            if isinstance(twin, type):
                pairs.append((name, cls, twin))
    return pairs


PAIRS = _class_pairs()


def _public_methods(cls: type) -> dict[str, Any]:
    return {n: f for n, f in vars(cls).items() if not n.startswith("_") and callable(f)}


def _params(fn: Any) -> dict[str, Any]:
    return {k: v.default for k, v in inspect.signature(fn).parameters.items() if k != "self"}


def test_the_pairs_were_actually_discovered() -> None:
    """Guard the guard: an empty discovery makes every check below vacuous."""
    assert len(PAIRS) >= 20, f"only {len(PAIRS)} sync/aio class pairs found"
    total = sum(len(_public_methods(s)) for _, s, _ in PAIRS)
    assert total >= 100, f"only {total} public methods discovered across the pairs"


def test_every_sync_method_has_an_async_twin_or_a_stated_reason() -> None:
    unexplained = [
        f"  {name}.{meth}"
        for name, sync_cls, aio_cls in PAIRS
        for meth in _public_methods(sync_cls)
        if getattr(aio_cls, meth, None) is None and (name, meth) not in SYNC_ONLY
    ]
    assert not unexplained, (
        "these exist on the sync client and not the async one, with no reason "
        "recorded in SYNC_ONLY:\n" + "\n".join(unexplained)
    )


def test_twinned_methods_take_the_same_arguments() -> None:
    failures: list[str] = []
    compared = 0

    for name, sync_cls, aio_cls in PAIRS:
        for meth, sync_fn in _public_methods(sync_cls).items():
            aio_fn = getattr(aio_cls, meth, None)
            if aio_fn is None:
                continue
            compared += 1
            sync_p, aio_p = _params(sync_fn), _params(aio_fn)

            for param in sorted(set(sync_p) - set(aio_p)):
                if (name, meth, param) not in JUSTIFIED:
                    failures.append(
                        f"  {name}.{meth}({param}=...) is SYNC-ONLY - an async caller "
                        f"cannot pass it, and gets TypeError if they try"
                    )
            for param in sorted(set(aio_p) - set(sync_p)):
                if (name, meth, param) not in JUSTIFIED:
                    failures.append(f"  {name}.{meth}({param}=...) is ASYNC-ONLY")
            for param in sorted(set(sync_p) & set(aio_p)):
                if sync_p[param] != aio_p[param]:
                    failures.append(
                        f"  {name}.{meth}({param}=) defaults to {sync_p[param]!r} on the "
                        f"sync client and {aio_p[param]!r} on the async one"
                    )

    assert compared >= 100, f"only {compared} twinned methods compared - discovery is broken"
    if failures:
        pytest.fail(
            f"{len(failures)} signature divergence(s) between the sync and async "
            "clients. Two clients, one contract - add the argument to both, or record "
            "the reason in JUSTIFIED:\n" + "\n".join(failures),
            pytrace=False,
        )


def test_the_download_format_vocabulary_matches() -> None:
    """A Literal is part of the contract too, and this one really did diverge.

    The async twin's `format` omitted `pte` and `engine`, so the two clients
    disagreed about which artifacts exist - a divergence the parameter-name
    comparison above cannot see, because the NAME matched.
    """
    from pictograph.aio.resources.models import AsyncModels
    from pictograph.resources.models import Models

    sync_fmt = inspect.signature(Models.download).parameters["format"].annotation
    aio_fmt = inspect.signature(AsyncModels.download).parameters["format"].annotation
    assert sync_fmt == aio_fmt, (
        f"models.download format vocabulary differs:\n  sync: {sync_fmt}\n  aio : {aio_fmt}"
    )
