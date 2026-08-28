"""Every request the SDK sends must bind against the backend's real routes.

Why this exists: the P1 rename moved `dataset_name` to `dataset` on the dataset,
image, annotation and directory route families. The SDK kept sending the old key.
`client.images.list("COCO-5k")` - the most basic read in the library - returned
400 against production, and so did everything that resolves an image through it.
The full unit suite passed, mypy --strict passed, and the published wheel was
broken. Nothing in the repo compared what the SDK SENDS to what the API TAKES.

That is what this does. It drives every resource method through a transport that
records instead of sending, then checks each recorded request against a generated
`rest-routes.json` snapshot (produced from the running API by the service's own
route generator):

  * the path resolves to a real route, honouring `{x:path}` converters;
  * the HTTP method is one that route allows;
  * every query param and body key is one that route accepts.

It is offline and free - no API key, no credits, no network - so it can run in
the ordinary suite rather than being a live-probe task nobody runs.

BOTH CLIENTS ARE COVERED. This gate originally walked only the synchronous
``Client``, and the async tree is a near-mechanical copy of it - which is exactly
where a path silently stops matching its twin. It did: ``AsyncDirectories.list``
and ``.tree`` still built ``/directories/list/{id}`` and ``/directories/tree/{id}``
long after P1 retired id-addressing and the sync twin was repaired to address by
name. Every ``AsyncClient().directories`` call was broken in the published wheel,
past a green suite, because nothing exercised the async twin. Adding
``AsyncClient`` here failed on that immediately.

The route snapshot is generated from the running API and does not ship in this
repository, so this whole module is opt-in (see ``tests/conftest.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import tempfile
import types
from pathlib import Path
from typing import Any

import pytest
from PIL import Image as _PILImage
from tests.conftest import ENV_REST_ROUTES_SNAPSHOT, companion_skip_reason, companion_source

from pictograph import AsyncClient, Client

SNAPSHOT = companion_source(ENV_REST_ROUTES_SNAPSHOT)

# Params every route accepts or that never reach the wire as user data.
_IGNORED_PARAMS = {"auth", "request", "background_tasks", "response", "x_api_key", "X-API-Key"}

# Methods that do not issue a request, or whose arguments cannot be synthesized
# without real remote state. Each is covered elsewhere; listing them here keeps
# the check honest instead of silently skipping whatever happens to raise.
_SKIP = {
    "connect",  # returns a DeploymentClient, no request
    "load",  # downloads then loads weights from disk
}


def _load_routes() -> list[dict[str, Any]]:
    if not SNAPSHOT.exists():  # pragma: no cover - snapshot not configured
        # Module level: the parametrization below is built from these routes, so
        # without the snapshot there is nothing to collect.
        pytest.skip(companion_skip_reason(ENV_REST_ROUTES_SNAPSHOT), allow_module_level=True)
    return json.loads(SNAPSHOT.read_text())["routes"]


ROUTES = _load_routes()


def _segments(path: str) -> list[str]:
    return [s for s in path.strip("/").split("/") if s]


def _match(doc: list[str], route: list[str]) -> bool:
    if not route:
        return not doc
    head, *rest = route
    if head.startswith("{") and head.endswith("}") and ":path}" in head:
        return any(_match(doc[take:], rest) for take in range(1, len(doc) - len(rest) + 1))
    if not doc:
        return False
    if not (head.startswith("{") and head.endswith("}")) and doc[0] != head:
        return False
    return _match(doc[1:], rest)


def _find(path: str, method: str) -> tuple[dict[str, Any] | None, list[str]]:
    segs = _segments(path)
    hits = [r for r in ROUTES if _match(segs, _segments(r["path"]))]
    if not hits:
        return None, []
    exact = [r for r in hits if method.upper() in r["methods"]]
    allowed = sorted({m for r in hits for m in r["methods"]})
    return (exact[0] if exact else None), allowed


class _Recorder:
    """A transport that records the request instead of performing it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any], Any]] = []
        self.base_url = "https://api.pictograph.io"
        self.api_key = "pk_live_placeholder"

    def _record(self, method: str, path: str, **kw: Any) -> Any:
        self.calls.append((method, path, kw.get("params") or {}, kw.get("json")))
        # A shape permissive enough that most response parsing survives long
        # enough for the NEXT call in a multi-request method to be recorded too.
        return {"data": [], "id": "00000000-0000-0000-0000-000000000000"}

    def request(self, method: str, path: str, **kw: Any) -> Any:
        return self._record(method, path, **kw)

    def request_raw(self, method: str, path: str, **kw: Any) -> Any:
        return self._record(method, path, **kw)

    def stream(self, method: str, path: str, **kw: Any) -> Any:
        self._record(method, path, **kw)
        return iter(())

    def __getattr__(self, name: str) -> Any:
        # Any other transport helper records as a GET rather than exploding, so
        # an unmodelled helper cannot make a method silently untested.
        def _f(*a: Any, **k: Any) -> Any:
            if a and isinstance(a[0], str) and a[0].startswith("/"):
                return self._record("GET", a[0], **k)
            return {"data": []}

        return _f


# Optional parameters that ADDRESS something and so must be supplied for the
# method to reach the wire at all.
_FILL_OPTIONAL = {
    "name",
    "dataset_name",
    "model",
    "image",
    "export_name",
    "directory_path",
    "organization",
    "deployment",
    "workflow",
    "endpoint",
    "run",
    "output_path",
    "split",
    "action",
    "tags",
    "new_name",
    "old_name",
    "source_name",
    "target_name",
    "role",
    "email",
    # These gate a method on "at least one of these was given" - `batch.update`
    # and `search.by_tag` each raise without one, which is why both used to fall
    # silent and go unchecked.
    "status",
    "objects",
    # `datasets.download` needs a destination before it will stream anything.
    "output_dir",
}

_UUID = "00000000-0000-0000-0000-000000000000"

# A REAL tiny JPEG on disk. Ten methods raised before reaching the wire purely
# for want of a file - among them `images.upload`, the most-used call in the
# library - so their paths were never bound against a route at all. The harness
# supplies real paths now rather than exempting them.
_FIXTURE_DIR = Path(tempfile.mkdtemp(prefix="wire-contract-"))
_FIXTURE_JPEG = _FIXTURE_DIR / "probe.jpg"
# Pillow is an SDK runtime dependency, so this needs no new dep and produces a
# genuinely decodable JPEG - a hand-written byte blob is easy to get subtly wrong.
_PILImage.new("RGB", (8, 8), (127, 127, 127)).save(_FIXTURE_JPEG, format="JPEG")
_FIXTURE_SUBDIR = _FIXTURE_DIR / "classA"
_FIXTURE_SUBDIR.mkdir(exist_ok=True)
(_FIXTURE_SUBDIR / "probe.jpg").write_bytes(_FIXTURE_JPEG.read_bytes())


def _dummy(param: inspect.Parameter, *, as_uuid: bool) -> Any:
    """A plausible value, chosen by annotation then by name.

    ``as_uuid`` drives the SECOND pass. A method that takes a name usually
    resolves it with an extra request first, and the recorder's canned response
    does not parse, so the method dies before issuing the request that actually
    matters - `annotations.delete_class` recorded only its dataset lookup and
    its POST body went unchecked. Passing a UUID makes `looks_like_id` short
    -circuit the resolve, so the real call is reached. Running BOTH passes and
    unioning them covers the name branch and the id branch; running either alone
    leaves a hole, and this gate exists because of holes exactly like it.
    """
    ann = param.annotation
    text = str(ann)
    name = param.name
    # Real on-disk paths, so file-taking methods reach the wire instead of
    # raising and being scored as "expected silent".
    if name in ("file_path", "path", "video_path"):
        return str(_FIXTURE_JPEG)
    if name == "file_paths":
        return [str(_FIXTURE_JPEG)]
    if name in ("directory", "output_dir", "folder"):
        return str(_FIXTURE_DIR)
    if "int" in text and "str" not in text:
        return 1
    if "float" in text and "str" not in text:
        return 1.0
    if "bool" in text and "str" not in text:
        return False
    if "dict" in text or "Mapping" in text:
        return {}
    if "Sequence" in text or "list" in text or "Iterable" in text:
        return ["a"]
    if name.endswith("_id") or name == "id":
        return _UUID
    return _UUID if as_uuid else "x"


def _resources_of(client: Any) -> list[tuple[str, Any]]:
    out = []
    for rname in sorted(dir(client)):
        if rname.startswith("_"):
            continue
        res = getattr(client, rname)
        if type(res).__module__.startswith("pictograph"):
            out.append((rname, res))
    return out


def _resources() -> list[tuple[str, Any]]:
    return _resources_of(Client(api_key="pk_live_placeholder"))


def _async_resources() -> list[tuple[str, Any]]:
    return _resources_of(AsyncClient(api_key="pk_live_placeholder"))


def _methods_of(res: Any) -> list[tuple[str, Any]]:
    """(name, function) for every public method on a resource CLASS."""
    out = []
    for mname in sorted(dir(type(res))):
        if mname.startswith("_") or mname in _SKIP:
            continue
        fn = getattr(type(res), mname, None)
        if isinstance(fn, types.FunctionType):
            out.append((mname, fn))
    return out


def _synth_args(sig: inspect.Signature, *, as_uuid: bool) -> tuple[list[Any], dict[str, Any]]:
    """Build a plausible call for ``sig``, skipping ``self``.

    Shared by the sync and async walks deliberately: these rules (which optional
    params must still be filled, which ``*_id`` twins stay unset) are the whole
    reason the harness reaches the wire at all, and two copies would drift the
    way the resources they exercise already did.
    """
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for pname, p in list(sig.parameters.items())[1:]:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is not inspect.Parameter.empty:
            # Optional NAME parameters must still be filled. Methods like
            # `datasets.get(name=None, *, dataset_id=None)` raise
            # "pass exactly one" when given neither, so leaving them unset
            # meant 38 methods - most of datasets and models - issued no
            # request at all and were scored as passing.
            #
            # `*_id` twins stay unset so the NAME branch is the one taken;
            # passing both is the error case, not a test.
            if p.default is not None or pname.endswith("_id") or pname == "id":
                continue
            if pname not in _FILL_OPTIONAL:
                continue
        if p.kind == p.KEYWORD_ONLY:
            kwargs[pname] = _dummy(p, as_uuid=as_uuid)
        else:
            args.append(_dummy(p, as_uuid=as_uuid))
    return args, kwargs


def _exercise(*, as_uuid: bool) -> list[tuple[str, str, str, dict[str, Any], Any]]:
    """(label, method, path, params, body) for every request the SYNC SDK emits."""
    emitted = []
    for rname, res in _resources():
        for mname, fn in _methods_of(res):
            try:
                sig = inspect.signature(fn)
            except (ValueError, TypeError):
                continue
            rec = _Recorder()
            bound = type(res)(rec)  # type: ignore[call-arg]
            args, kwargs = _synth_args(sig, as_uuid=as_uuid)
            try:
                result = getattr(bound, mname)(*args, **kwargs)
                # A pager is LAZY - it issues nothing until consumed, so every
                # `iter` looked silent. Pull one page to make the request happen.
                if hasattr(result, "__iter__") and not isinstance(result, (str, bytes, list, dict)):
                    with contextlib.suppress(Exception):
                        next(iter(result), None)
            except Exception:
                pass  # partial progress still recorded the requests it made
            for method, path, params, body in rec.calls:
                emitted.append((f"client.{rname}.{mname}", method, path, params, body))
    return emitted


class _MaybeAwaited:
    """A value usable whether or not the caller awaits it.

    The catch-all transport helper cannot know if an unmodelled method is a
    coroutine. Building a coroutine eagerly leaks an un-awaited one when it is
    not; this defers creation to ``__await__``, so the un-awaited case allocates
    nothing.
    """

    def __init__(self, value: Any) -> None:
        self._value = value

    def __await__(self) -> Any:
        async def _c() -> Any:
            return self._value

        return _c().__await__()

    def __getitem__(self, key: Any) -> Any:
        return self._value[key]

    def get(self, key: Any, default: Any = None) -> Any:
        return self._value.get(key, default)

    def __iter__(self) -> Any:
        return iter(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)


class _AsyncRecorder(_Recorder):
    """The async twin of :class:`_Recorder`.

    Async resources ``await`` the transport, so ``request``/``request_raw`` must be
    coroutines; ``stream`` must return an async iterator. Everything else - the
    canned response shape, the catch-all helper - is inherited so the two
    recorders cannot answer differently.
    """

    async def request(self, method: str, path: str, **kw: Any) -> Any:  # type: ignore[override]
        return self._record(method, path, **kw)

    async def request_raw(self, method: str, path: str, **kw: Any) -> Any:  # type: ignore[override]
        return self._record(method, path, **kw)

    def stream(self, method: str, path: str, **kw: Any) -> Any:
        self._record(method, path, **kw)

        async def _empty() -> Any:
            return
            yield  # pragma: no cover - makes this an async generator

        return _empty()

    def __getattr__(self, name: str) -> Any:
        # Returns a SYNC callable handing back a lazily-awaitable result. An
        # `async def` here builds a coroutine even when the caller never awaits
        # it, and an un-awaited coroutine surfaces as a PytestUnraisableException
        # that fails the run for a reason unrelated to any route.
        def _f(*a: Any, **k: Any) -> Any:
            if a and isinstance(a[0], str) and a[0].startswith("/"):
                return _MaybeAwaited(self._record("GET", a[0], **k))
            return _MaybeAwaited({"data": []})

        return _f


def _exercise_async(*, as_uuid: bool) -> list[tuple[str, str, str, dict[str, Any], Any]]:
    """(label, method, path, params, body) for every request the ASYNC SDK emits.

    The async tree is a near-mechanical copy of the sync one, which is precisely
    why it needs its own walk: a copy that stops tracking its original is
    invisible to a gate that only reads the original.
    """
    emitted = []
    for rname, res in _async_resources():
        for mname, fn in _methods_of(res):
            try:
                sig = inspect.signature(fn)
            except (ValueError, TypeError):
                continue
            rec = _AsyncRecorder()
            bound = type(res)(rec)  # type: ignore[call-arg]
            args, kwargs = _synth_args(sig, as_uuid=as_uuid)

            async def _drive(
                bound: Any = bound,
                mname: str = mname,
                args: list[Any] = args,
                kwargs: dict[str, Any] = kwargs,
            ) -> None:
                result = getattr(bound, mname)(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                # Async pagers are lazy too - consume one item so the page fetch
                # actually happens (the sync walk learned this the hard way).
                if hasattr(result, "__aiter__"):
                    with contextlib.suppress(Exception):
                        async for _ in result:
                            break

            with contextlib.suppress(Exception):
                asyncio.run(_drive())
            for method, path, params, body in rec.calls:
                emitted.append((f"aio.{rname}.{mname}", method, path, params, body))
    return emitted


def _union() -> list[tuple[str, str, str, dict[str, Any], Any]]:
    seen: dict[str, tuple[str, str, str, dict[str, Any], Any]] = {}
    for as_uuid in (False, True):
        for walk in (_exercise, _exercise_async):
            for row in walk(as_uuid=as_uuid):
                label, method, path, params, body = row
                # Key on the SHAPE, not the values, so the two passes dedupe.
                key = f"{label}|{method}|{len(path.split('/'))}|{sorted(params)}|" + (
                    str(sorted(body)) if isinstance(body, dict) else ""
                )
                seen.setdefault(key, row)
    return list(seen.values())


EMITTED = _union()


def test_the_harness_actually_exercised_the_sdk() -> None:
    """Guard the guard: a harness that records nothing would pass everything."""
    assert len(EMITTED) > 100, f"only {len(EMITTED)} requests recorded - harness broken"


def test_the_harness_actually_exercised_the_async_sdk() -> None:
    """The async half needs its own floor.

    Without this, an ``AsyncClient`` that stopped constructing - or an
    ``_AsyncRecorder`` that stopped recording - would take the async coverage to
    zero while the sync count alone kept the guard above satisfied. Silent loss
    of half the surface is the failure mode this whole file exists to prevent.
    """
    async_rows = [e for e in EMITTED if e[0].startswith("aio.")]
    assert len(async_rows) > 100, (
        f"only {len(async_rows)} ASYNC requests recorded - the async walk is broken"
    )


# Methods that legitimately issue no HTTP request in either pass. Anything that
# falls silent for another reason - a NameError inside a path builder, say - is a
# NEW entry here and fails, instead of vanishing into `except Exception: pass`
# and being scored as a pass. That exact hole hid an undefined `quote` in
# `images._image_route`: the three methods that call it raised before recording,
# so the suite went green over code that could not run.
#
# Each entry needs a REAL local file or a caller-shaped dict this harness cannot
# invent, so the method raises before reaching the wire. Named individually, not
# pattern-matched, so adding a method never quietly widens the exemption.
_EXPECTED_SILENT = {
    "client.datasets.as_pytorch",  # builds a torch Dataset, no request
    "client.models.get_local_path",  # pure cache-path computation
    "client.models.predict",  # loads weights, then runs locally
    "client.video.upload",  # needs a real VIDEO; a JPEG is rejected before the wire
    "client.webhooks.update",  # requires a caller-shaped updates mapping
}


def test_no_method_silently_emits_nothing() -> None:
    spoke = {label.rsplit("-", 0)[0] for label, *_ in ((e[0], e) for e in EMITTED)}
    all_methods = set()
    for prefix, resources in (("client", _resources()), ("aio", _async_resources())):
        for rname, res in resources:
            for mname, _fn in _methods_of(res):
                all_methods.add(f"{prefix}.{rname}.{mname}")
    exempt = _EXPECTED_SILENT | {
        # The async twin of each sync exemption, for the identical reason: a real
        # local file or a caller-shaped mapping this harness cannot invent.
        f"aio.{label.split('.', 1)[1]}"
        for label in _EXPECTED_SILENT
    }
    silent = sorted(all_methods - spoke - exempt)
    assert not silent, (
        "these methods issued NO request in either pass - they most likely raised "
        f"before reaching the wire, which this gate would otherwise score as a pass: {silent}"
    )


@pytest.mark.parametrize("label,method,path,params,body", EMITTED, ids=lambda v: str(v)[:40])
def test_request_binds_against_the_backend(
    label: str, method: str, path: str, params: dict[str, Any], body: Any
) -> None:
    route, allowed = _find(path, method)
    assert allowed, f"{label}: {method} {path} - NO SUCH ROUTE"
    assert route is not None, f"{label}: {method} {path} - route allows only {allowed}"

    accepted_q = set(route["params"]) - _IGNORED_PARAMS
    sent_q = set(params) - _IGNORED_PARAMS
    unknown_q = sent_q - accepted_q
    assert not unknown_q, (
        f"{label}: {method} {path} sends query param(s) {sorted(unknown_q)} "
        f"which the route does not accept (it takes {sorted(accepted_q)})"
    )

    if isinstance(body, dict) and route["body"]:
        unknown_b = set(body) - set(route["body"])
        assert not unknown_b, (
            f"{label}: {method} {path} sends body field(s) {sorted(unknown_b)} "
            f"which the route model does not declare (it takes {sorted(route['body'])})"
        )
