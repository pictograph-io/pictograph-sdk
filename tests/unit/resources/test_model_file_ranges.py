"""A model's files come in 8 MiB byte ranges - the way the CDN edge caches a large file.

Every path that downloads a model's file (``models.download`` and ``models.download_file``,
sync and async, and so the CLI, the agent tool and ``get_model`` on top of them) goes through
one ranged download. These tests drive it through those public methods against a scripted
file server that answers ranges the way the API's model-file URLs do (a ``206`` with its
``Content-Range``, ``Content-Length`` and ``ETag``), with the real 8 MiB range size, and hold
what it must do:

- assemble the ranges byte for byte, the first asked for alone and then up to three at a time;
- read a server that ignores ``Range`` (a ``200``) as one streamed plain download;
- ask again for a range the network drops or cuts short, under the client's retry policy;
- refuse a ``206`` whose ``Content-Range`` is missing, short, or not the range asked for, a
  file that changes under the download, and a size the model's file list contradicts;
- never leave the temporary ``.part`` file behind, on success or on failure.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import struct
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from pictograph._http.async_transport import AsyncTransport
from pictograph._http.ranged import parse_content_range
from pictograph._http.retry import RetryPolicy
from pictograph._http.transport import Transport
from pictograph._internal.config import ClientConfig
from pictograph._version import __version__
from pictograph.aio.resources.models import AsyncModels
from pictograph.exceptions import ApiError, NetworkError, RequestTimeoutError
from pictograph.resources.models import Models, _check_safetensors

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator
    from pathlib import Path

    from pytest_httpx import HTTPXMock

BASE = "https://api.test.local"
KEY = "pk_live_test"
MODEL = "abcdef01-2345-6789-abcd-ef0123456789"
VERSION = "11111111-1111-1111-1111-111111111111"
DOWNLOAD_ROUTE = f"{BASE}/api/v1/developer/models/{MODEL}/download?format=onnx"
FILE_URL = "https://files.test/api/cdn/m/org/model/b2Jq/1794973788399173/model.onnx?Expires=1&KeyName=k&Signature=s"

#: The range size the CDN edge caches a model file by, and the app's and the iPhone's.
RANGE = 8 * 1024 * 1024


def _payload(size: int, seed: int = 7) -> bytes:
    return random.Random(seed).randbytes(size)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _range(start: int, end: int) -> str:
    return f"bytes={start}-{end}"


def _windows(size: int) -> list[str]:
    """The ranges a file of ``size`` bytes is asked for in, in file order."""
    return [_range(at, min(at + RANGE, size) - 1) for at in range(0, size, RANGE)]


# ───────────── the scripted file server ─────────────


Script = dict[tuple[str | None, int], "Callable[[RangeServer, httpx.Request], httpx.Response]"]


class _CutStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """A body that sends ``head``, waits ``pause`` seconds, then loses its connection."""

    def __init__(self, head: bytes, pause: float = 0.0) -> None:
        self._head = head
        self._pause = pause

    def __iter__(self) -> Iterator[bytes]:
        yield self._head
        time.sleep(self._pause)
        raise httpx.ReadError("connection reset by peer")

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._head
        await asyncio.sleep(self._pause)
        raise httpx.ReadError("connection reset by peer")


class RangeServer:
    """One file, served as the API serves a model file.

    A single ``bytes=a-b`` range is a ``206`` with ``Content-Range: bytes a-b/size`` (its end
    held to the file), ``Content-Length`` and ``ETag``; a request without ``Range``, or any
    request when ``ranges=False``, is the whole file (``200``). ``script[(range, n)]`` answers
    the ``n``-th request (from 1) for ``range`` instead, so a test can drop, shorten or garble
    exactly that one. ``delay`` holds each answer after the first that long, so ranges overlap;
    ``slow[(range, n)]`` holds that one request longer still.
    """

    def __init__(
        self,
        content: bytes,
        *,
        script: Script | None = None,
        ranges: bool = True,
        delay: float = 0.0,
        slow: dict[tuple[str | None, int], float] | None = None,
        etag: str = '"1794973788399173"',
    ) -> None:
        self.content = content
        self.script: Script = script or {}
        self.ranges = ranges
        self.delay = delay
        self.slow = slow or {}
        self.etag = etag
        self.requests: list[httpx.Request] = []
        self.order: list[tuple[str, str | None]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()

    @property
    def seen(self) -> list[str | None]:
        """The ``Range`` header of every request, in the order they came."""
        return [r.headers.get("range") for r in self.requests]

    def _enter(self, request: httpx.Request) -> tuple[str | None, int]:
        rng = request.headers.get("range")
        with self._lock:
            self.requests.append(request)
            self.order.append(("start", rng))
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            return rng, self.seen.count(rng)

    def _hold(self, rng: str | None, n: int) -> float:
        return (self.delay if len(self.requests) > 1 else 0.0) + self.slow.get((rng, n), 0.0)

    def _leave(self, rng: str | None) -> None:
        with self._lock:
            self.in_flight -= 1
            self.order.append(("end", rng))

    def _answer(self, request: httpx.Request, rng: str | None, n: int) -> httpx.Response:
        scripted = self.script.get((rng, n))
        if scripted is not None:
            return scripted(self, request)
        if rng is None or not self.ranges or not self.content:
            # as the API answers: no range of an empty file is satisfiable, so it comes whole
            return self.whole()
        start, end = self.bounds(rng)
        return self.partial(start, end)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        rng, n = self._enter(request)
        try:
            time.sleep(self._hold(rng, n))
            return self._answer(request, rng, n)
        finally:
            self._leave(rng)

    async def serve_async(self, request: httpx.Request) -> httpx.Response:
        rng, n = self._enter(request)
        try:
            await asyncio.sleep(self._hold(rng, n))
            return self._answer(request, rng, n)
        finally:
            self._leave(rng)

    def bounds(self, rng: str) -> tuple[int, int]:
        m = re.fullmatch(r"bytes=(\d+)-(\d+)", rng)
        assert m is not None, f"not a single closed range: {rng!r}"
        return int(m[1]), min(int(m[2]), len(self.content) - 1)

    def whole(self) -> httpx.Response:
        return httpx.Response(200, headers={"ETag": self.etag}, content=self.content)

    def partial(
        self,
        start: int,
        end: int,
        *,
        body: bytes | None = None,
        content_range: str | None = None,
        etag: str | None = None,
    ) -> httpx.Response:
        headers = {
            "Accept-Ranges": "bytes",
            "ETag": etag or self.etag,
            "Content-Range": content_range or f"bytes {start}-{end}/{len(self.content)}",
        }
        return httpx.Response(
            206, headers=headers, content=self.content[start : end + 1] if body is None else body
        )

    def cut(self, rng: str, keep: int, pause: float = 0.0) -> httpx.Response:
        """The range's head, then (after ``pause`` seconds) the connection lost."""
        start, end = self.bounds(rng)
        headers = {
            "Accept-Ranges": "bytes",
            "ETag": self.etag,
            "Content-Range": f"bytes {start}-{end}/{len(self.content)}",
        }
        return httpx.Response(
            206, headers=headers, stream=_CutStream(self.content[start : start + keep], pause)
        )


# ───────────── fixtures ─────────────


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _Sleeps:
    """The retry policy's backoff waits, recorded instead of slept."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)

    async def wait(self, seconds: float) -> None:
        self.waits.append(seconds)


@pytest.fixture
def sleeps() -> _Sleeps:
    return _Sleeps()


def _policy(sleeps: _Sleeps, retries: int) -> RetryPolicy:
    return RetryPolicy(
        max_retries=retries, sleep=sleeps, async_sleep=sleeps.wait, rng=lambda _a, _b: 1.0
    )


def _config() -> ClientConfig:
    return ClientConfig(api_key=KEY, base_url=BASE, timeout=10.0, max_retries=2)  # type: ignore[arg-type]


@pytest.fixture
def models(sleeps: _Sleeps) -> Iterator[Models]:
    transport = Transport(_config(), api_key=KEY, retry_policy=_policy(sleeps, retries=2))
    yield Models(transport)
    transport.close()


@pytest.fixture
async def amodels(sleeps: _Sleeps) -> AsyncIterator[AsyncModels]:
    transport = AsyncTransport(_config(), api_key=KEY, retry_policy=_policy(sleeps, retries=2))
    yield AsyncModels(transport)
    await transport.aclose()


def _grant(httpx_mock: HTTPXMock, server: RangeServer, *, is_async: bool = False) -> None:
    """The download route's grant, then the file's URL served by ``server``."""
    httpx_mock.add_response(
        method="GET", url=DOWNLOAD_ROUTE, json={"data": {"download_url": FILE_URL}}
    )
    httpx_mock.add_callback(
        server.serve_async if is_async else server, method="GET", url=FILE_URL, is_reusable=True
    )


def _nothing_left(out: Path) -> None:
    assert not out.exists(), "a failed download left a file at the destination"
    assert not out.with_name(out.name + ".part").exists(), "the .part file was left behind"


# ───────────── assembled byte for byte ─────────────


def test_a_model_file_comes_in_8_mib_ranges_assembled_byte_for_byte(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    content = _payload(4 * RANGE + 1000)  # five ranges, the last 1000 bytes
    server = RangeServer(content, delay=0.1)  # long enough for three to overlap on a slow machine
    _grant(httpx_mock, server)
    out = tmp_path / "model.onnx"

    assert models.download(MODEL, output_path=out) == out

    assert _digest(out.read_bytes()) == _digest(content)
    windows = _windows(len(content))
    assert server.seen[0] == windows[0] == _range(0, RANGE - 1)
    assert sorted(server.seen[1:], key=str) == sorted(windows[1:], key=str)
    # the first range is asked for alone - nothing else until it is answered...
    assert server.order[:2] == [("start", windows[0]), ("end", windows[0])]
    # ...then three at a time, never more
    assert server.max_in_flight == 3
    for request in server.requests:
        assert request.headers["accept-encoding"] == "identity"
        assert request.headers["user-agent"] == f"pictograph-python/{__version__}"
        assert "x-api-key" not in request.headers, "the API key went to the file's host"
    assert not (tmp_path / "model.onnx.part").exists()


def test_ranges_land_at_their_offsets_whatever_order_they_finish(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    content = _payload(3 * RANGE + 10, seed=11)
    windows = _windows(len(content))

    def slow(server: RangeServer, request: httpx.Request) -> httpx.Response:
        time.sleep(0.3)  # the second range finishes after the third and fourth
        return server.partial(*server.bounds(windows[1]))

    server = RangeServer(content, script={(windows[1], 1): slow})
    _grant(httpx_mock, server)

    out = models.download(MODEL, output_path=tmp_path / "m.onnx")

    assert _digest(out.read_bytes()) == _digest(content)
    assert server.order.index(("end", windows[1])) > server.order.index(("end", windows[3]))


def test_a_small_file_is_one_range(httpx_mock: HTTPXMock, models: Models, tmp_path: Path) -> None:
    content = b'{"model_type": "object_detection"}'
    server = RangeServer(content)
    _grant(httpx_mock, server)

    out = models.download(MODEL, output_path=tmp_path / "config.json")

    assert out.read_bytes() == content
    assert server.seen == [_range(0, RANGE - 1)]


# ───────────── a server without ranges ─────────────


def test_a_server_that_ignores_ranges_is_read_as_one_plain_download(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    content = _payload(2 * RANGE + 5, seed=3)
    server = RangeServer(content, ranges=False)  # every answer is 200, the whole file
    _grant(httpx_mock, server)
    progress: list[tuple[int, int]] = []

    out = models.download(
        MODEL, output_path=tmp_path / "m.onnx", progress=lambda a, b: progress.append((a, b))
    )

    assert _digest(out.read_bytes()) == _digest(content)
    assert server.seen == [_range(0, RANGE - 1)], "one request, which asked for a range"
    assert progress[-1] == (len(content), len(content))
    assert not (tmp_path / "m.onnx.part").exists()


def test_an_empty_file_answered_whole_is_an_empty_file(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    server = RangeServer(b"")  # the API answers a range of an empty file 200, empty
    _grant(httpx_mock, server)

    out = models.download(MODEL, output_path=tmp_path / "empty.onnx")

    assert out.read_bytes() == b""
    assert server.seen == [_range(0, RANGE - 1)]


def test_an_empty_file_a_server_will_not_range_is_asked_for_whole(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    def unsatisfiable(server: RangeServer, request: httpx.Request) -> httpx.Response:
        return httpx.Response(416, headers={"Content-Range": "bytes */0"})

    server = RangeServer(b"", script={(_range(0, RANGE - 1), 1): unsatisfiable})
    _grant(httpx_mock, server)

    out = models.download(MODEL, output_path=tmp_path / "empty.onnx")

    assert out.read_bytes() == b""
    assert server.seen == [_range(0, RANGE - 1), None]


# ───────────── a range lost or cut short is asked for again ─────────────


def test_a_range_whose_connection_drops_is_asked_for_again(
    httpx_mock: HTTPXMock, models: Models, sleeps: _Sleeps, tmp_path: Path
) -> None:
    content = _payload(2 * RANGE + 99, seed=5)
    second = _windows(len(content))[1]
    server = RangeServer(content, script={(second, 1): lambda s, _r: s.cut(second, keep=4096)})
    _grant(httpx_mock, server)

    out = models.download(MODEL, output_path=tmp_path / "m.onnx")

    assert _digest(out.read_bytes()) == _digest(content)
    assert server.seen.count(second) == 2
    assert sleeps.waits == [1.0], "the retry policy's backoff, once"


def test_a_range_answered_short_is_asked_for_again(
    httpx_mock: HTTPXMock, models: Models, sleeps: _Sleeps, tmp_path: Path
) -> None:
    content = _payload(RANGE + 4096, seed=6)
    first = _windows(len(content))[0]

    def short(server: RangeServer, request: httpx.Request) -> httpx.Response:
        # its Content-Range says the whole range; half of it arrives
        return server.partial(0, RANGE - 1, body=content[: RANGE // 2])

    server = RangeServer(content, script={(first, 1): short})
    _grant(httpx_mock, server)

    out = models.download(MODEL, output_path=tmp_path / "m.onnx")

    assert _digest(out.read_bytes()) == _digest(content)
    assert server.seen == [first, first, _windows(len(content))[1]]
    assert len(sleeps.waits) == 1


def test_a_range_answered_with_a_passing_failure_is_asked_for_again(
    httpx_mock: HTTPXMock, models: Models, sleeps: _Sleeps, tmp_path: Path
) -> None:
    content = _payload(RANGE + 1, seed=8)
    second = _windows(len(content))[1]
    unavailable = httpx.Response(503, headers={"Retry-After": "1"}, text="Storage did not answer.")
    server = RangeServer(content, script={(second, 1): lambda _s, _r: unavailable})
    _grant(httpx_mock, server)

    out = models.download(MODEL, output_path=tmp_path / "m.onnx")

    assert _digest(out.read_bytes()) == _digest(content)
    assert server.seen.count(second) == 2
    assert len(sleeps.waits) == 1


def test_a_range_that_never_arrives_fails_the_download_and_leaves_nothing(
    httpx_mock: HTTPXMock, models: Models, sleeps: _Sleeps, tmp_path: Path
) -> None:
    content = _payload(2 * RANGE + 1, seed=9)
    second = _windows(len(content))[1]
    lost = {(second, n): (lambda s, _r: s.cut(second, keep=10)) for n in (1, 2, 3)}
    server = RangeServer(content, script=lost)
    _grant(httpx_mock, server)
    out = tmp_path / "m.onnx"

    with pytest.raises(NetworkError):
        models.download(MODEL, output_path=out)

    assert server.seen.count(second) == 3, "asked for once, then max_retries (2) more times"
    assert len(sleeps.waits) == 2
    _nothing_left(out)


def test_a_range_that_keeps_timing_out_raises_request_timeout_and_leaves_nothing(
    httpx_mock: HTTPXMock, models: Models, sleeps: _Sleeps, tmp_path: Path
) -> None:
    content = _payload(2 * RANGE + 1, seed=16)
    second = _windows(len(content))[1]

    def stalled(server: RangeServer, request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("the read timed out", request=request)

    server = RangeServer(content, script={(second, n): stalled for n in (1, 2, 3)})
    _grant(httpx_mock, server)
    out = tmp_path / "m.onnx"

    with pytest.raises(RequestTimeoutError):
        models.download(MODEL, output_path=out)

    assert server.seen.count(second) == 3, "asked for once, then max_retries (2) more times"
    assert len(sleeps.waits) == 2
    _nothing_left(out)


# ───────────── refused ─────────────


def _refused(
    httpx_mock: HTTPXMock,
    models: Models,
    tmp_path: Path,
    answer: Callable[[RangeServer, httpx.Request], httpx.Response],
    *,
    match: str,
) -> RangeServer:
    """The second of three ranges answered by ``answer``: the download fails at once (never
    asked for again, though the policy retries) and leaves nothing behind."""
    content = _payload(2 * RANGE + 500, seed=10)
    second = _windows(len(content))[1]
    server = RangeServer(content, script={(second, 1): answer})
    _grant(httpx_mock, server)
    out = tmp_path / "m.onnx"

    with pytest.raises(ApiError, match=match):
        models.download(MODEL, output_path=out)

    assert server.seen.count(second) == 1, "a refusal is not a passing failure: never retried"
    _nothing_left(out)
    return server


def test_a_range_that_is_not_the_range_asked_for_is_refused(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    # bytes 2R..2R+499 answered for the range R..2R-1
    _refused(
        httpx_mock,
        models,
        tmp_path,
        lambda s, _r: s.partial(2 * RANGE, 2 * RANGE + 499),
        match="asked for",
    )


def test_a_range_shorter_than_asked_for_is_refused(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    # bytes R..R+99 (with exactly those bytes) for the range R..2R-1
    _refused(
        httpx_mock, models, tmp_path, lambda s, _r: s.partial(RANGE, RANGE + 99), match="asked for"
    )


def test_a_range_with_no_content_range_is_refused(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    def bare(server: RangeServer, request: httpx.Request) -> httpx.Response:
        start, end = server.bounds(request.headers["range"])
        return httpx.Response(206, content=server.content[start : end + 1])

    _refused(httpx_mock, models, tmp_path, bare, match="Content-Range")


def test_a_range_with_more_bytes_than_its_content_range_is_refused(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    def long(server: RangeServer, request: httpx.Request) -> httpx.Response:
        start, end = server.bounds(request.headers["range"])
        return server.partial(start, end, body=server.content[start : end + 2])

    _refused(httpx_mock, models, tmp_path, long, match="more")


def test_a_file_whose_size_changes_under_the_download_is_refused(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    def grown(server: RangeServer, request: httpx.Request) -> httpx.Response:
        start, end = server.bounds(request.headers["range"])
        return server.partial(
            start, end, content_range=f"bytes {start}-{end}/{len(server.content) + 1}"
        )

    _refused(httpx_mock, models, tmp_path, grown, match="changed")


def test_a_file_whose_etag_changes_under_the_download_is_refused(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    def rebuilt(server: RangeServer, request: httpx.Request) -> httpx.Response:
        return server.partial(*server.bounds(request.headers["range"]), etag='"1794973788399999"')

    _refused(httpx_mock, models, tmp_path, rebuilt, match="changed")


def test_the_whole_file_in_answer_to_a_later_range_is_refused(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    _refused(httpx_mock, models, tmp_path, lambda s, _r: s.whole(), match="whole file")


def test_an_expired_link_fails_at_once(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    expired = httpx.Response(403, json={"detail": "This link is not valid."})
    server = _refused(httpx_mock, models, tmp_path, lambda _s, _r: expired, match="HTTP 403")
    assert server.seen.count(_windows(2 * RANGE + 500)[2]) <= 1


# ───────────── the size the API lists ─────────────


def _manifest(size: int | None) -> dict[str, Any]:
    return {
        "versions": [{"version_id": VERSION, "version_number": 1, "is_latest": True}],
        "files": [
            {
                "version_id": VERSION,
                "name": "model.onnx",
                "kind": "weights",
                "format": "onnx",
                "size_bytes": size,
            }
        ],
    }


def _file_grant(httpx_mock: HTTPXMock, server: RangeServer, listed: int | None) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/{MODEL}/files",
        json={"data": _manifest(listed)},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/{MODEL}/files/download?version={VERSION}&name=model.onnx",
        json={"data": {"download_url": FILE_URL, "filename": "model.onnx"}},
    )
    httpx_mock.add_callback(server, method="GET", url=FILE_URL, is_reusable=True)


def test_download_file_refuses_a_file_whose_size_its_file_list_contradicts(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    content = _payload(2 * RANGE + 3, seed=12)
    server = RangeServer(content)
    _file_grant(httpx_mock, server, listed=len(content) - 1)
    out = tmp_path / "model.onnx"

    with pytest.raises(ApiError, match=f"{len(content) - 1:,}"):
        models.download_file(MODEL, filename="model.onnx", output_path=out)

    assert server.seen == [_range(0, RANGE - 1)], "refused at the first range's size"
    _nothing_left(out)


def test_download_file_takes_a_file_of_the_size_its_file_list_says(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    content = _payload(RANGE + 3, seed=13)
    server = RangeServer(content)
    _file_grant(httpx_mock, server, listed=len(content))

    out = models.download_file(MODEL, filename="model.onnx", output_path=tmp_path / "model.onnx")

    assert _digest(out.read_bytes()) == _digest(content)
    assert server.seen == _windows(len(content))


# ───────────── the resources reach a file only through the transport ─────────────


class _StandInTransport:
    """A transport with what the API's own SDK contract harness models - ``request`` and
    ``_config`` - plus ``download_ranged``, which records its calls. Anything else a resource
    reaches for (the transport's private retry policy, an httpx client of its own) fails the
    test: the harness answers such an attribute with an inert helper, so a resource that
    depends on one breaks that harness for every published SDK."""

    def __init__(self, grant: dict[str, Any]) -> None:
        self._config = SimpleNamespace(timeout=30.0, base_url=BASE)
        self.grant = grant
        self.downloads: list[dict[str, Any]] = []

    def request(self, _method: str, _path: str, **_kw: Any) -> dict[str, Any]:
        return self.grant

    def download_ranged(self, url: str, output_path: Any, **kw: Any) -> Any:
        self.downloads.append({"url": url, "output_path": output_path, **kw})
        return output_path

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"a resource reached past the transport's methods: .{name}")


class _AsyncStandInTransport(_StandInTransport):
    async def request(self, _method: str, _path: str, **_kw: Any) -> dict[str, Any]:  # type: ignore[override]
        return self.grant

    async def download_ranged(self, url: str, output_path: Any, **kw: Any) -> Any:  # type: ignore[override]
        self.downloads.append({"url": url, "output_path": output_path, **kw})
        return output_path


def test_the_resources_reach_a_file_only_through_the_transport(
    httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    grant = {"data": {"download_url": FILE_URL}}
    transport = _StandInTransport(grant)
    models = Models(transport)  # type: ignore[arg-type]

    models.download(MODEL, output_path=tmp_path / "m.safetensors", format="safetensors")
    transport.grant = {"data": {"download_url": FILE_URL}}
    models.download_file(MODEL, filename="model.onnx", version=VERSION, output_path=tmp_path / "f")

    assert [d["url"] for d in transport.downloads] == [FILE_URL, FILE_URL]
    first, second = transport.downloads
    assert first["error_prefix"] == "Model download"
    assert first["check"] is _check_safetensors
    assert second["error_prefix"] == "Model file download"
    assert second["expected_size"] is None  # a version id: the manifest was not read
    assert not httpx_mock.get_requests(), "a resource sent a request of its own"


@pytest.mark.anyio
async def test_async_the_resources_reach_a_file_only_through_the_transport(
    httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    transport = _AsyncStandInTransport({"data": {"download_url": FILE_URL}})
    models = AsyncModels(transport)  # type: ignore[arg-type]

    await models.download(MODEL, output_path=tmp_path / "m.onnx")
    await models.download_file(
        MODEL, filename="model.onnx", version=VERSION, output_path=tmp_path / "f"
    )

    assert [d["url"] for d in transport.downloads] == [FILE_URL, FILE_URL]
    assert transport.downloads[0]["check"] is None
    assert not httpx_mock.get_requests(), "a resource sent a request of its own"


# ───────────── a safetensors file is checked before it lands ─────────────


SAFETENSORS_ROUTE = f"{BASE}/api/v1/developer/models/{MODEL}/download?format=safetensors"


def _safetensors() -> bytes:
    header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    return struct.pack("<Q", len(header)) + header + b"\x00\x00\x80\x3f"


def _safetensors_grant(httpx_mock: HTTPXMock, server: RangeServer, *, is_async: bool) -> None:
    httpx_mock.add_response(
        method="GET", url=SAFETENSORS_ROUTE, json={"data": {"download_url": FILE_URL}}
    )
    httpx_mock.add_callback(
        server.serve_async if is_async else server, method="GET", url=FILE_URL, is_reusable=True
    )


def test_a_safetensors_file_lands_once_its_header_checks_out(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    server = RangeServer(_safetensors())
    _safetensors_grant(httpx_mock, server, is_async=False)

    out = models.download(MODEL, output_path=tmp_path / "m.safetensors", format="safetensors")

    assert out.read_bytes() == _safetensors()


def test_a_safetensors_file_that_is_not_one_is_refused_and_leaves_nothing(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    server = RangeServer(b"<html>not a safetensors file</html>")
    _safetensors_grant(httpx_mock, server, is_async=False)
    out = tmp_path / "m.safetensors"

    with pytest.raises(ApiError, match="safetensors file failed validation"):
        models.download(MODEL, output_path=out, format="safetensors")

    _nothing_left(out)


@pytest.mark.anyio
async def test_async_a_safetensors_file_that_is_not_one_is_refused_and_leaves_nothing(
    httpx_mock: HTTPXMock, amodels: AsyncModels, tmp_path: Path
) -> None:
    server = RangeServer(b"<html>not a safetensors file</html>")
    _safetensors_grant(httpx_mock, server, is_async=True)
    out = tmp_path / "m.safetensors"

    with pytest.raises(ApiError, match="safetensors file failed validation"):
        await amodels.download(MODEL, output_path=out, format="safetensors")

    _nothing_left(out)


# ───────────── progress, and an interrupted download ─────────────


@pytest.mark.parametrize(
    ("header", "parsed"),
    [
        ("bytes 0-8388607/107833286", (0, 8388607, 107833286)),
        ("bytes 0-0/1", (0, 0, 1)),
        ("Bytes  5-9/10 ", (5, 9, 10)),
        (None, None),
        ("", None),
        ("bytes 0-99/*", None),  # a size the server does not know: nothing to place it in
        ("bytes */0", None),  # the 416 form
        ("bytes 9-5/10", None),  # backwards
        ("bytes 0-10/10", None),  # past the end of the file
        ("bytes 0-0--1/0", None),
        ("items 0-1/2", None),
    ],
)
def test_content_range_parsing(header: str | None, parsed: tuple[int, int, int] | None) -> None:
    assert parse_content_range(header) == parsed


def _backtracking_server(content: bytes) -> RangeServer:
    """Where progress could go backwards: the second range drops after 4 MiB (counted, then
    taken back), and the last range lands while the second is fetched again (slowly) - so
    the file is momentarily less far along than it was reported to be."""
    windows = _windows(len(content))
    second, last = windows[1], windows[-1]
    return RangeServer(
        content,
        delay=0.02,
        script={(second, 1): lambda s, _r: s.cut(second, keep=RANGE // 2, pause=0.15)},
        slow={(last, 1): 0.3, (second, 2): 0.4},
    )


def test_progress_counts_up_to_the_files_size_on_the_calling_thread(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    content = _payload(4 * RANGE + 7, seed=14)
    second, last = _windows(len(content))[1], _windows(len(content))[-1]
    server = _backtracking_server(content)
    _grant(httpx_mock, server)
    calls: list[tuple[int, int, threading.Thread]] = []

    models.download(
        MODEL,
        output_path=tmp_path / "m.onnx",
        chunk_size=1 << 20,  # 1 MiB pieces: the dropped range's first 4 MiB are counted
        progress=lambda a, b: calls.append((a, b, threading.current_thread())),
    )

    # the case where it could go wrong: ranges fetched on other threads, one asked for again
    # after another range landed
    assert server.seen.count(second) == 2
    second_ends = [i for i, event in enumerate(server.order) if event == ("end", second)]
    assert server.order.index(("end", last)) < second_ends[-1], (
        "the last range landed while the second was fetched again"
    )
    assert server.max_in_flight >= 2
    assert calls, "progress was never reported"
    assert {total for _, total, _ in calls} == {len(content)}
    loaded = [a for a, _, _ in calls]
    assert loaded == sorted(loaded), "progress went backwards (a range asked for again)"
    assert loaded[-1] == len(content)
    assert {thread for _, _, thread in calls} == {threading.current_thread()}


def test_an_interrupted_download_stops_and_leaves_nothing(
    httpx_mock: HTTPXMock, models: Models, tmp_path: Path
) -> None:
    content = _payload(9 * RANGE + 1, seed=15)  # ten ranges: the first, then three batches of three
    server = RangeServer(content, delay=0.2)
    _grant(httpx_mock, server)
    out = tmp_path / "m.onnx"

    def interrupt(loaded: int, total: int) -> None:
        if loaded > RANGE:  # once the first batch after the first range is in
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        models.download(MODEL, output_path=out, progress=interrupt)

    _nothing_left(out)
    assert server.max_in_flight >= 2, "interrupted while ranges were fetched side by side"
    assert len(server.seen) < len(_windows(len(content))), "the other ranges were stopped"
    assert server.in_flight == 0, "a range was still being fetched after the download returned"


# ───────────── the async twin ─────────────


@pytest.mark.anyio
async def test_async_a_model_file_comes_in_8_mib_ranges_assembled_byte_for_byte(
    httpx_mock: HTTPXMock, amodels: AsyncModels, tmp_path: Path
) -> None:
    content = _payload(4 * RANGE + 1000, seed=21)
    server = RangeServer(content, delay=0.1)
    _grant(httpx_mock, server, is_async=True)
    out = tmp_path / "model.onnx"

    assert await amodels.download(MODEL, output_path=out) == out

    assert _digest(out.read_bytes()) == _digest(content)
    windows = _windows(len(content))
    assert server.seen[0] == windows[0]
    assert sorted(server.seen[1:], key=str) == sorted(windows[1:], key=str)
    assert server.order[:2] == [("start", windows[0]), ("end", windows[0])]
    assert server.max_in_flight == 3
    for request in server.requests:
        assert request.headers["accept-encoding"] == "identity"
        assert request.headers["user-agent"] == f"pictograph-python-async/{__version__}"
        assert "x-api-key" not in request.headers
    assert not (tmp_path / "model.onnx.part").exists()


@pytest.mark.anyio
async def test_async_a_server_that_ignores_ranges_is_read_as_one_plain_download(
    httpx_mock: HTTPXMock, amodels: AsyncModels, tmp_path: Path
) -> None:
    content = _payload(RANGE + 5, seed=22)
    server = RangeServer(content, ranges=False)
    _grant(httpx_mock, server, is_async=True)

    out = await amodels.download(MODEL, output_path=tmp_path / "m.onnx")

    assert _digest(out.read_bytes()) == _digest(content)
    assert server.seen == [_range(0, RANGE - 1)]


@pytest.mark.anyio
async def test_async_an_empty_file_a_server_will_not_range_is_asked_for_whole(
    httpx_mock: HTTPXMock, amodels: AsyncModels, tmp_path: Path
) -> None:
    def unsatisfiable(server: RangeServer, request: httpx.Request) -> httpx.Response:
        return httpx.Response(416, headers={"Content-Range": "bytes */0"})

    server = RangeServer(b"", script={(_range(0, RANGE - 1), 1): unsatisfiable})
    _grant(httpx_mock, server, is_async=True)

    out = await amodels.download(MODEL, output_path=tmp_path / "empty.onnx")

    assert out.read_bytes() == b""
    assert server.seen == [_range(0, RANGE - 1), None]


@pytest.mark.anyio
async def test_async_a_range_whose_connection_drops_is_asked_for_again(
    httpx_mock: HTTPXMock, amodels: AsyncModels, sleeps: _Sleeps, tmp_path: Path
) -> None:
    content = _payload(2 * RANGE + 99, seed=23)
    second = _windows(len(content))[1]
    server = RangeServer(content, script={(second, 1): lambda s, _r: s.cut(second, keep=4096)})
    _grant(httpx_mock, server, is_async=True)

    out = await amodels.download(MODEL, output_path=tmp_path / "m.onnx")

    assert _digest(out.read_bytes()) == _digest(content)
    assert server.seen.count(second) == 2
    assert sleeps.waits == [1.0]


@pytest.mark.anyio
async def test_async_a_range_answered_short_is_asked_for_again(
    httpx_mock: HTTPXMock, amodels: AsyncModels, sleeps: _Sleeps, tmp_path: Path
) -> None:
    content = _payload(RANGE + 4096, seed=28)
    first = _windows(len(content))[0]

    def short(server: RangeServer, request: httpx.Request) -> httpx.Response:
        return server.partial(0, RANGE - 1, body=content[: RANGE // 2])

    server = RangeServer(content, script={(first, 1): short})
    _grant(httpx_mock, server, is_async=True)

    out = await amodels.download(MODEL, output_path=tmp_path / "m.onnx")

    assert _digest(out.read_bytes()) == _digest(content)
    assert server.seen == [first, first, _windows(len(content))[1]]
    assert len(sleeps.waits) == 1


@pytest.mark.anyio
async def test_async_a_range_that_keeps_timing_out_raises_request_timeout(
    httpx_mock: HTTPXMock, amodels: AsyncModels, sleeps: _Sleeps, tmp_path: Path
) -> None:
    content = _payload(2 * RANGE + 1, seed=29)
    second = _windows(len(content))[1]

    def stalled(server: RangeServer, request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("the read timed out", request=request)

    server = RangeServer(content, script={(second, n): stalled for n in (1, 2, 3)})
    _grant(httpx_mock, server, is_async=True)
    out = tmp_path / "m.onnx"

    with pytest.raises(RequestTimeoutError):
        await amodels.download(MODEL, output_path=out)

    assert server.seen.count(second) == 3
    assert len(sleeps.waits) == 2
    _nothing_left(out)


@pytest.mark.anyio
async def test_async_a_range_that_is_not_the_range_asked_for_is_refused(
    httpx_mock: HTTPXMock, amodels: AsyncModels, tmp_path: Path
) -> None:
    content = _payload(3 * RANGE + 500, seed=24)
    second = _windows(len(content))[1]
    server = RangeServer(
        content, delay=0.05, script={(second, 1): lambda s, _r: s.partial(2 * RANGE, 3 * RANGE - 1)}
    )
    _grant(httpx_mock, server, is_async=True)
    out = tmp_path / "m.onnx"

    with pytest.raises(ApiError, match="asked for"):
        await amodels.download(MODEL, output_path=out)

    assert server.seen.count(second) == 1
    _nothing_left(out)
    await asyncio.sleep(0.2)  # a range still being fetched would land in this time
    assert server.in_flight == 0
    _nothing_left(out)


@pytest.mark.anyio
async def test_async_download_file_refuses_a_file_whose_size_its_file_list_contradicts(
    httpx_mock: HTTPXMock, amodels: AsyncModels, tmp_path: Path
) -> None:
    content = _payload(RANGE + 3, seed=25)
    server = RangeServer(content)
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/{MODEL}/files",
        json={"data": _manifest(len(content) + 1)},
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE}/api/v1/developer/models/{MODEL}/files/download?version={VERSION}&name=model.onnx",
        json={"data": {"download_url": FILE_URL, "filename": "model.onnx"}},
    )
    httpx_mock.add_callback(server.serve_async, method="GET", url=FILE_URL, is_reusable=True)
    out = tmp_path / "model.onnx"

    with pytest.raises(ApiError, match=f"{len(content) + 1:,}"):
        await amodels.download_file(MODEL, filename="model.onnx", output_path=out)

    assert server.seen == [_range(0, RANGE - 1)]
    _nothing_left(out)


@pytest.mark.anyio
async def test_async_progress_counts_up_to_the_files_size_on_the_event_loop(
    httpx_mock: HTTPXMock, amodels: AsyncModels, tmp_path: Path
) -> None:
    content = _payload(4 * RANGE + 7, seed=27)
    second = _windows(len(content))[1]
    server = _backtracking_server(content)
    _grant(httpx_mock, server, is_async=True)
    calls: list[tuple[int, int, threading.Thread]] = []

    await amodels.download(
        MODEL,
        output_path=tmp_path / "m.onnx",
        chunk_size=1 << 20,
        progress=lambda a, b: calls.append((a, b, threading.current_thread())),
    )

    assert server.seen.count(second) == 2
    assert server.max_in_flight >= 2
    assert {total for _, total, _ in calls} == {len(content)}
    loaded = [a for a, _, _ in calls]
    assert loaded == sorted(loaded), "progress went backwards (a range asked for again)"
    assert loaded[-1] == len(content)
    assert {thread for _, _, thread in calls} == {threading.current_thread()}


@pytest.mark.anyio
async def test_async_an_interrupted_download_stops_and_leaves_nothing(
    httpx_mock: HTTPXMock, amodels: AsyncModels, tmp_path: Path
) -> None:
    content = _payload(5 * RANGE + 1, seed=26)
    server = RangeServer(content, delay=0.05)
    _grant(httpx_mock, server, is_async=True)
    out = tmp_path / "m.onnx"

    task = asyncio.ensure_future(amodels.download(MODEL, output_path=out))
    deadline = time.monotonic() + 10
    while len(server.seen) < 3:  # the ranges after the first under way
        if time.monotonic() > deadline or task.done():
            task.cancel()
            pytest.fail(f"the ranges after the first never started (asked for: {server.seen})")
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    _nothing_left(out)
    assert server.in_flight == 0
    assert len(server.seen) < len(_windows(len(content))), "the other ranges were stopped"
