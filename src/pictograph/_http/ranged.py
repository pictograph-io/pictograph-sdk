"""A model's files downloaded in byte ranges - the way the CDN edge caches a large file.

A model's files reach a client as signed URLs that the API's CDN edge validates and caches. The
edge keeps a large file only by its byte ranges, and only from responses that carry their length,
which a whole large file streamed in one response does not. So the SDK fetches a model's file the
way the app and the mobile app do: in 8 MiB ranges (``Range: bytes=from-through``), the first
asked for alone (its answer gives the file's size, and teaches the edge that the file serves
ranges), then up to three at a time. Each range is written straight into a temporary ``.part``
file beside the destination, at its own offset, and the file is renamed into place once every
byte is there. The whole file is never held in memory.

A range is taken only as it was asked for:

- a ``206`` whose ``Content-Range`` names exactly the bytes asked for, of a file whose size (and
  ``ETag``, when the server sends one) never changes from one range to the next, and whose size
  is the one the API listed for the file, when it lists one;
- carrying exactly as many bytes as that ``Content-Range`` says.

A range cut short or lost to the network, or answered with a passing failure (``408``, ``429``,
``5xx``), is asked for again under the client's retry policy (its backoff and ``max_retries``).
Anything else fails the download at once - a ``403`` (an expired link), a ``206`` for other
bytes or without a ``Content-Range``, a file that changed - and nothing is left behind.

A server that answers the first range with the whole file (``200``) does not serve ranges: that
answer is the download, streamed to disk as one plain request. One that answers it ``416`` (an
empty file it will not range) is asked once more without a range.

Resources reach it through the transport (``Transport.download_ranged`` and its async twin),
which hands it the client's retry policy and timeout. Only a model's files come this way: an
export or a dataset's images stream in one request.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import threading
from collections import deque
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Final

import httpx

from pictograph._version import __version__
from pictograph.exceptions import ApiError, NetworkError, RequestTimeoutError

if TYPE_CHECKING:
    from collections.abc import Callable

    from pictograph._http.retry import RetryPolicy

RANGE_SIZE: Final = 8 * 1024 * 1024
"""8 MiB: the range the CDN edge caches a model file by (each answer carries its length), and the
range the app and the mobile app fetch one in."""

IN_FLIGHT: Final = 3
"""How many ranges are fetched at once after the first, each on its own connection. Measured on a
108 MB model file: three connections came in 2.3-2.7 s from a warm edge where one took 8 s and
three ranges multiplexed on one HTTP/2 connection took 4-6 s."""

READ_TIMEOUT: Final = 600.0
"""Seconds a range may wait for its next bytes before it counts as lost (and is asked again)."""

_PROGRESS_EVERY: Final = 0.05
"""Seconds between progress reports while ranges are fetched on worker threads."""

_CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+)", re.IGNORECASE)


def parse_content_range(value: str | None) -> tuple[int, int, int] | None:
    """``(first, last, size)`` from a ``Content-Range: bytes first-last/size`` header.

    ``None`` when it is absent, malformed, of an unknown size (``bytes 0-99/*``), or names bytes
    outside the file - none of which a ranged download can place.
    """
    if not value:
        return None
    match = _CONTENT_RANGE.fullmatch(value.strip())
    if match is None:
        return None
    first, last, size = (int(group) for group in match.groups())
    if first > last or last >= size:
        return None
    return first, last, size


def _content_length(response: httpx.Response) -> int | None:
    raw = response.headers.get("content-length")
    if raw is None or not raw.strip().isdigit():
        return None
    return int(raw)


def _open_for_write(path: Path) -> BinaryIO:
    return path.open("wb")


def _open_for_update(path: Path) -> BinaryIO:
    return path.open("r+b")


class _AbandonedError(Exception):
    """Raised inside a range when the download is being abandoned (another range failed)."""


class _RangedDownload:
    """One file's download: what is known of the file, what has arrived, and each range's rules.

    The rules every range is held to live here once, for the sync and the async fetch alike.
    """

    def __init__(
        self,
        url: str,
        *,
        chunk_size: int,
        expected_size: int | None,
        error_prefix: str,
        user_agent: str,
    ) -> None:
        self.url = url
        self.user_agent = user_agent
        self.chunk_size = chunk_size
        self.expected_size = expected_size
        self.prefix = error_prefix
        self.size: int | None = None
        self.etag: str | None = None
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self._arrived = 0
        self._reported = 0

    # ───── requests ─────

    def headers(self, first: int, last: int | None) -> dict[str, str]:
        """``last=None`` asks for the whole file (no ``Range``)."""
        out = {
            # a range's length is checked byte for byte, so never a compressed copy of it
            "Accept-Encoding": "identity",
            "User-Agent": self.user_agent,
        }
        if last is not None:
            out["Range"] = f"bytes={first}-{last}"
        return out

    def check_running(self) -> None:
        """Before a range is asked for (again): not when the download is being abandoned."""
        if self.stop.is_set():
            raise _AbandonedError

    def windows_after(self, first_size: int) -> list[tuple[int, int]]:
        """The ranges still to fetch once the first (``first_size`` bytes long) is in."""
        size = self.size or 0
        return [(at, min(at + RANGE_SIZE, size) - 1) for at in range(first_size, size, RANGE_SIZE)]

    # ───── what a range must be ─────

    def refused(self, reason: str, response: httpx.Response | None = None) -> ApiError:
        return ApiError(
            f"{self.prefix} failed: {reason}",
            status_code=response.status_code if response is not None else None,
            request_id=response.headers.get("x-request-id") if response is not None else None,
        )

    def failed_status(self, response: httpx.Response) -> ApiError:
        return ApiError(
            f"{self.prefix} failed: HTTP {response.status_code}",
            status_code=response.status_code,
            request_id=response.headers.get("x-request-id"),
            response=response.text,
        )

    def _settle_size(self, size: int, response: httpx.Response) -> None:
        with self._lock:
            if self.size is None:
                if self.expected_size is not None and size != self.expected_size:
                    raise self.refused(
                        f"the file is {size:,} bytes, but the model's file list says "
                        f"{self.expected_size:,}",
                        response,
                    )
                self.size = size
            elif size != self.size:
                raise self.refused(
                    f"the file changed while it downloaded (it was {self.size:,} bytes, "
                    f"now {size:,})",
                    response,
                )

    def _settle_etag(self, response: httpx.Response) -> None:
        etag = response.headers.get("etag")
        if etag is None:
            return
        with self._lock:
            if self.etag is None:
                self.etag = etag
            elif etag != self.etag:
                raise self.refused(
                    f"the file changed while it downloaded (ETag {self.etag} became {etag})",
                    response,
                )

    def accept(
        self, response: httpx.Response, first: int, last: int | None
    ) -> tuple[int | None, bool]:
        """Check a ``200`` / ``206`` against the range asked for, before its body is read.

        Returns ``(length, whole)``: how many bytes the body must carry (``None`` when a whole
        file comes without a length) and whether it is the whole file. Raises the refusal.
        """
        if response.status_code == 200:
            if first != 0:
                raise self.refused(
                    f"the server answered bytes {first}-{last} with the whole file", response
                )
            length = _content_length(response)
            if length is not None:
                self._settle_size(length, response)
            self._settle_etag(response)
            return length, True
        content_range = response.headers.get("content-range")
        parsed = parse_content_range(content_range)
        if parsed is None:
            raise self.refused(
                f"the server answered bytes {first}-{last} without a usable Content-Range "
                f"({content_range!r})",
                response,
            )
        got_first, got_last, size = parsed
        want_last = size - 1 if last is None else min(last, size - 1)
        if (got_first, got_last) != (first, want_last):
            raise self.refused(
                f"the server answered bytes {got_first}-{got_last} of {size:,} when bytes "
                f"{first}-{want_last} were asked for",
                response,
            )
        self._settle_size(size, response)
        self._settle_etag(response)
        return got_last - got_first + 1, False

    def check_piece(self, got: int, piece: int, length: int | None, first: int) -> None:
        if self.stop.is_set():
            raise _AbandonedError
        if length is not None and got + piece > length:
            raise self.refused(
                f"the server sent more than the {length:,} bytes it declared for bytes {first}+"
            )

    def short(self, got: int, length: int | None, first: int) -> NetworkError | None:
        """The error for a body that ended early (it is asked for again), else ``None``."""
        if length is None or got >= length:
            return None
        return NetworkError(
            f"{self.prefix} failed: bytes {first}+ arrived cut short ({got:,} of {length:,})"
        )

    def whole_done(self, got: int, response: httpx.Response) -> None:
        """A whole file without a ``Content-Length``: its size is what arrived."""
        if self.size is None:
            self._settle_size(got, response)

    # ───── progress ─────

    def count(self, n: int) -> None:
        with self._lock:
            self._arrived += n

    def report(self, progress: Callable[[int, int], None] | None) -> None:
        """Tell ``progress`` how far the file is, if further than last told: never backwards,
        even when a range cut short is asked for again."""
        if progress is None:
            return
        with self._lock:
            arrived = self._arrived
            if arrived <= self._reported:
                return
            self._reported = arrived
            size = self.size or 0
        progress(arrived, size)

    def verify(self, tmp: Path) -> None:
        """The file on disk is the whole file, or the download fails."""
        on_disk = tmp.stat().st_size
        if self.size is not None and on_disk != self.size:
            raise self.refused(f"{on_disk:,} of the file's {self.size:,} bytes were written")


# ───────────── sync ─────────────


def _client(timeout: float) -> httpx.Client:
    # HTTP/1.1 on purpose: one connection a range (the client is shared by the worker threads,
    # and httpx's HTTP/2 is not thread-safe on a shared client). No credentials: the URL is signed.
    return httpx.Client(
        http2=False,
        timeout=httpx.Timeout(timeout, read=READ_TIMEOUT),
        limits=httpx.Limits(max_connections=IN_FLIGHT, max_keepalive_connections=IN_FLIGHT),
    )


def _fetch_range(
    state: _RangedDownload,
    http: httpx.Client,
    retry: RetryPolicy,
    fh: BinaryIO,
    first: int,
    last: int | None,
    on_bytes: Callable[[], None] | None,
) -> bool:
    """Bytes ``first..last`` into ``fh`` at ``first``, asked again per ``retry`` when lost.

    Returns True when the server sent the whole file instead (a ``200``).
    """
    whole: list[bool] = []

    def attempt() -> httpx.Response:
        state.check_running()
        got = 0
        try:
            with http.stream("GET", state.url, headers=state.headers(first, last)) as response:
                if response.status_code not in (200, 206):
                    response.read()  # an error page: its words and its Retry-After
                    return response
                length, is_whole = state.accept(response, first, last)
                fh.seek(first)
                for piece in response.iter_bytes(chunk_size=state.chunk_size):
                    state.check_piece(got, len(piece), length, first)
                    fh.write(piece)
                    got += len(piece)
                    state.count(len(piece))
                    if on_bytes is not None:
                        on_bytes()
                cut = state.short(got, length, first)
                if cut is not None:
                    raise cut
                if is_whole:
                    fh.truncate()
                    state.whole_done(got, response)
                whole.append(is_whole)
                return response
        except httpx.TimeoutException as e:
            state.count(-got)
            raise RequestTimeoutError(f"{state.prefix} failed: {e}") from e
        except httpx.RequestError as e:
            state.count(-got)
            raise NetworkError(f"{state.prefix} failed: {e}") from e
        except NetworkError:
            state.count(-got)
            raise

    response = retry.execute(attempt, method="GET", has_idempotency_key=False)
    if whole:
        return whole[0]
    if response.status_code == 416 and first == 0 and last is not None:
        # An empty file, from a server that will not range one: ask for it whole.
        return _fetch_range(state, http, retry, fh, 0, None, on_bytes)
    raise state.failed_status(response)


def _fetch_rest(
    state: _RangedDownload,
    http: httpx.Client,
    retry: RetryPolicy,
    tmp: Path,
    windows: list[tuple[int, int]],
    progress: Callable[[int, int], None] | None,
) -> None:
    """The ranges after the first, up to ``IN_FLIGHT`` at a time on worker threads; progress is
    reported from this (the calling) thread."""
    queue = deque(windows)
    take = threading.Lock()

    def next_window() -> tuple[int, int] | None:
        with take:
            if state.stop.is_set() or not queue:
                return None
            return queue.popleft()

    def worker() -> None:
        with tmp.open("r+b") as fh:
            while (window := next_window()) is not None:
                # a whole file (200) for a later range is refused before a byte of it lands
                _fetch_range(state, http, retry, fh, *window, None)

    with ThreadPoolExecutor(
        max_workers=min(IN_FLIGHT, len(windows)), thread_name_prefix="pictograph-download"
    ) as pool:
        pending = {pool.submit(worker) for _ in range(min(IN_FLIGHT, len(windows)))}
        try:
            while pending:
                done, pending = wait(pending, timeout=_PROGRESS_EVERY, return_when=FIRST_EXCEPTION)
                state.report(progress)
                for future in done:
                    future.result()
        except BaseException:
            # One range failed, or the caller was interrupted: the others stop at their next
            # piece, and the pool is joined before the .part file is removed.
            state.stop.set()
            raise


def download(
    url: str,
    output_path: str | Path,
    *,
    timeout: float,
    retry: RetryPolicy,
    chunk_size: int,
    progress: Callable[[int, int], None] | None = None,
    expected_size: int | None = None,
    error_prefix: str = "Download",
    check: Callable[[Path], None] | None = None,
) -> Path:
    """Download a model's file from its signed ``url`` to ``output_path``, in byte ranges.

    Args:
        url: The file's signed URL. Nothing of the client's (its API key) is sent to it.
        output_path: The destination. Missing folders are created; the bytes land in a sibling
            ``.part`` file, renamed into place only once the whole file is there (and checked).
        timeout: Connect timeout, seconds.
        retry: The client's retry policy - a range lost or answered with a passing failure is
            asked for again under it.
        chunk_size: How many bytes are read and written at a time.
        progress: ``(bytes_so_far, total_bytes)``, from the calling thread; ``total_bytes`` is
            the file's size, ``0`` while it is unknown (a server without ranges or a length).
        expected_size: The size the API listed for the file: any other size is refused.
        error_prefix: Starts the message of every error (``"Model download"``).
        check: Called with the finished ``.part`` file before it is renamed; whatever it
            raises fails the download (and removes the file).

    Returns:
        The output path.

    Raises:
        ApiError: The server refused (its status), or answered something that is not the file.
        NetworkError: A range never arrived, after every retry.
    """
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    state = _RangedDownload(
        url,
        chunk_size=chunk_size,
        expected_size=expected_size,
        error_prefix=error_prefix,
        user_agent=f"pictograph-python/{__version__}",
    )
    try:
        with _client(timeout) as http:
            with tmp.open("wb") as fh:
                whole = _fetch_range(
                    state, http, retry, fh, 0, RANGE_SIZE - 1, lambda: state.report(progress)
                )
            if not whole:
                windows = state.windows_after(RANGE_SIZE)
                if windows:
                    _fetch_rest(state, http, retry, tmp, windows, progress)
        state.verify(tmp)
        state.report(progress)
        if check is not None:
            check(tmp)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    tmp.replace(out)
    return out


# ───────────── async ─────────────


def _async_client(timeout: float) -> httpx.AsyncClient:
    # HTTP/1.1: a connection a range moves a large file faster than ranges multiplexed on one.
    return httpx.AsyncClient(
        http2=False,
        timeout=httpx.Timeout(timeout, read=READ_TIMEOUT),
        limits=httpx.Limits(max_connections=IN_FLIGHT, max_keepalive_connections=IN_FLIGHT),
    )


async def _fetch_range_async(
    state: _RangedDownload,
    http: httpx.AsyncClient,
    retry: RetryPolicy,
    fh: BinaryIO,
    first: int,
    last: int | None,
    progress: Callable[[int, int], None] | None,
) -> bool:
    """The async twin of :func:`_fetch_range` (disk writes off the event loop)."""
    whole: list[bool] = []

    async def attempt() -> httpx.Response:
        state.check_running()
        got = 0
        try:
            async with http.stream(
                "GET", state.url, headers=state.headers(first, last)
            ) as response:
                if response.status_code not in (200, 206):
                    await response.aread()
                    return response
                length, is_whole = state.accept(response, first, last)
                await asyncio.to_thread(fh.seek, first)
                async for piece in response.aiter_bytes(chunk_size=state.chunk_size):
                    state.check_piece(got, len(piece), length, first)
                    await asyncio.to_thread(fh.write, piece)
                    got += len(piece)
                    state.count(len(piece))
                    state.report(progress)
                cut = state.short(got, length, first)
                if cut is not None:
                    raise cut
                if is_whole:
                    await asyncio.to_thread(fh.truncate)
                    state.whole_done(got, response)
                whole.append(is_whole)
                return response
        except httpx.TimeoutException as e:
            state.count(-got)
            raise RequestTimeoutError(f"{state.prefix} failed: {e}") from e
        except httpx.RequestError as e:
            state.count(-got)
            raise NetworkError(f"{state.prefix} failed: {e}") from e
        except NetworkError:
            state.count(-got)
            raise

    response = await retry.execute_async(attempt, method="GET", has_idempotency_key=False)
    if whole:
        return whole[0]
    if response.status_code == 416 and first == 0 and last is not None:
        return await _fetch_range_async(state, http, retry, fh, 0, None, progress)
    raise state.failed_status(response)


async def _fetch_rest_async(
    state: _RangedDownload,
    http: httpx.AsyncClient,
    retry: RetryPolicy,
    tmp: Path,
    windows: list[tuple[int, int]],
    progress: Callable[[int, int], None] | None,
) -> None:
    queue = deque(windows)

    async def worker() -> None:
        # Opened and closed on the loop (quick calls): a cancelled worker never strands its file.
        fh = _open_for_update(tmp)
        try:
            while queue:
                first, last = queue.popleft()
                await _fetch_range_async(state, http, retry, fh, first, last, progress)
        finally:
            fh.close()

    tasks = [asyncio.ensure_future(worker()) for _ in range(min(IN_FLIGHT, len(windows)))]
    try:
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    finally:
        # One range failed, or the caller cancelled: stop the others and wait for them, so no
        # range is still writing when the .part file is removed.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    for task in done:
        error = None if task.cancelled() else task.exception()
        if error is not None:
            raise error


async def download_async(
    url: str,
    output_path: str | Path,
    *,
    timeout: float,
    retry: RetryPolicy,
    chunk_size: int,
    progress: Callable[[int, int], None] | None = None,
    expected_size: int | None = None,
    error_prefix: str = "Download",
    check: Callable[[Path], None] | None = None,
) -> Path:
    """The async twin of :func:`download` - the same ranges, rules and errors."""
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    state = _RangedDownload(
        url,
        chunk_size=chunk_size,
        expected_size=expected_size,
        error_prefix=error_prefix,
        user_agent=f"pictograph-python-async/{__version__}",
    )
    try:
        async with _async_client(timeout) as http:
            # The file's open, close, removal and rename stay on the loop (quick calls), so a
            # cancellation can strand neither a handle nor a half-done rename.
            fh = _open_for_write(tmp)
            try:
                whole = await _fetch_range_async(
                    state, http, retry, fh, 0, RANGE_SIZE - 1, progress
                )
            finally:
                fh.close()
            if not whole:
                windows = state.windows_after(RANGE_SIZE)
                if windows:
                    await _fetch_rest_async(state, http, retry, tmp, windows, progress)
        state.verify(tmp)
        state.report(progress)
        if check is not None:
            await asyncio.to_thread(check, tmp)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    tmp.replace(out)
    return out
