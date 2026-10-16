"""Server-Sent Events parsing and chunked file iteration.

Two utilities live here, both used by :mod:`pictograph._http.transport`:

- :func:`parse_sse` decodes a Server-Sent Events byte stream (per the WHATWG
  spec) into a generator of :class:`SSEEvent` objects. Each event's ``data``
  field is JSON-parsed when possible, falling back to the raw string. Used by
  ``client.training.stream``, ``client.connectors.stream_status``, etc.

- :func:`chunked_file_iterator` produces an iterator of byte chunks from a
  local file with an optional progress callback. Used by
  ``client.images.upload`` to PUT large files to object storage without
  loading them fully into memory.

Both functions are pure: no I/O setup, no global state, easy to test.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024
"""8 MB. Tuned for the storage backend's resumable-upload sweet spot (256 KB
minimum chunk, multi-MB maximises throughput on broadband)."""


@dataclass(frozen=True)
class SSEEvent:
    """A single parsed Server-Sent Event.

    Attributes:
        event: Event type (``"message"`` by default per spec).
        data: Parsed JSON payload, or raw ``str`` if parsing failed.
        id: ``Last-Event-ID`` token for stream resumption (``None`` if
            absent in the wire payload).
        retry: Server-suggested reconnect delay in milliseconds (``None`` if
            absent or non-integer in wire payload).
    """

    event: str
    data: Any
    id: str | None = None
    retry: int | None = None


def parse_sse(stream: Iterable[bytes]) -> Iterator[SSEEvent]:
    """Decode a byte stream of Server-Sent Events.

    Stream parsing follows the WHATWG SSE spec
    (https://html.spec.whatwg.org/multipage/server-sent-events.html):

    - Events are delimited by blank lines.
    - Lines starting with ``:`` are comments (ignored).
    - Fields use ``name: value`` syntax (one leading space after the colon
      stripped, per spec).
    - Multi-line ``data:`` fields are concatenated with ``"\\n"``.
    - Events without a ``data`` field are not dispatched.
    - Trailing partial events (no terminating blank line) are not dispatched.

    ``\\r\\n`` and lone ``\\r`` line endings are normalised to ``\\n`` before
    parsing.

    Bytes are decoded with an *incremental* UTF-8 decoder so a multibyte
    character split across two network chunks (``response.iter_bytes()`` breaks
    at arbitrary byte boundaries) is buffered rather than raising
    ``UnicodeDecodeError`` mid-stream - only a genuinely malformed sequence
    surfaces an error, on flush at end-of-stream.
    """
    buffer = ""
    decoder = codecs.getincrementaldecoder("utf-8")()
    for chunk in stream:
        if not chunk:
            continue
        # ``final=False`` (the default) holds back a trailing partial multibyte
        # character until the rest of its bytes arrive in the next chunk.
        buffer += decoder.decode(chunk)
        # Normalise newlines once per chunk; cheaper than per-line.
        buffer = buffer.replace("\r\n", "\n").replace("\r", "\n")
        while True:
            sep = buffer.find("\n\n")
            if sep == -1:
                break
            event_text = buffer[:sep]
            buffer = buffer[sep + 2 :]
            event = _parse_event_block(event_text)
            if event is not None:
                yield event
    # Flush: surfaces a truly incomplete trailing multibyte sequence (vs a
    # boundary split, which is now buffered). Any flushed text can only be a
    # trailing partial event with no terminating blank line, which the spec
    # says is not dispatched - so we discard it but still force the decode.
    decoder.decode(b"", final=True)


async def parse_sse_async(stream: AsyncIterable[bytes]) -> AsyncIterator[SSEEvent]:
    """Async twin of :func:`parse_sse` - decode an async byte stream of SSE.

    Byte-for-byte identical framing/decoding logic (shared
    :func:`_parse_event_block`); only the upstream chunk source is awaited.
    """
    buffer = ""
    decoder = codecs.getincrementaldecoder("utf-8")()
    async for chunk in stream:
        if not chunk:
            continue
        buffer += decoder.decode(chunk)
        buffer = buffer.replace("\r\n", "\n").replace("\r", "\n")
        while True:
            sep = buffer.find("\n\n")
            if sep == -1:
                break
            event_text = buffer[:sep]
            buffer = buffer[sep + 2 :]
            event = _parse_event_block(event_text)
            if event is not None:
                yield event
    decoder.decode(b"", final=True)


def _parse_event_block(text: str) -> SSEEvent | None:
    """Parse a single event block (the text between blank-line separators)."""
    event_type = "message"
    data_lines: list[str] = []
    event_id: str | None = None
    retry_ms: int | None = None

    for line in text.split("\n"):
        if not line or line.startswith(":"):
            continue
        if ":" in line:
            name, _, raw_value = line.partition(":")
            # Per spec: strip exactly one leading space if present.
            value = raw_value[1:] if raw_value.startswith(" ") else raw_value
        else:
            name = line
            value = ""

        if name == "data":
            data_lines.append(value)
        elif name == "event":
            event_type = value or "message"
        elif name == "id":
            event_id = value
        elif name == "retry":
            # Non-integer ``retry`` values are silently ignored per SSE spec.
            with contextlib.suppress(ValueError):
                retry_ms = int(value)
        # Other fields are silently ignored.

    if not data_lines:
        return None

    raw_data = "\n".join(data_lines)
    parsed: Any
    try:
        parsed = json.loads(raw_data)
    except json.JSONDecodeError:
        # Caller may legitimately send non-JSON payloads; preserve as string.
        parsed = raw_data

    return SSEEvent(event=event_type, data=parsed, id=event_id, retry=retry_ms)


def chunked_file_iterator(
    path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress: Callable[[int, int], None] | None = None,
    total_size: int | None = None,
) -> Iterator[bytes]:
    """Yield ``path`` byte-chunks with optional progress reporting.

    Args:
        path: File to read. Opened in binary mode.
        chunk_size: Bytes per yielded chunk. Must be positive. The final
            chunk may be smaller.
        progress: Optional callback invoked after each chunk read with
            ``(bytes_sent_so_far, total_bytes)`` - drive progress bars.
        total_size: Authoritative byte total for the progress callback. Pass
            the SAME ``stat().st_size`` the caller used for a ``Content-Length``
            header so the header and the progress total can never disagree
            from a second, independent ``stat`` (see ``upload_external``).
            Falls back to ``path.stat().st_size`` when omitted.

    Empty files produce zero chunks and zero callback invocations.

    Raises:
        ValueError: ``chunk_size`` is non-positive.
        FileNotFoundError: ``path`` does not exist.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    total = total_size if total_size is not None else path.stat().st_size
    sent = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            sent += len(chunk)
            if progress is not None:
                progress(sent, total)
            yield chunk


async def chunked_file_iterator_async(
    path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress: Callable[[int, int], None] | None = None,
    total_size: int | None = None,
) -> AsyncIterator[bytes]:
    """Async twin of :func:`chunked_file_iterator` for ``AsyncClient`` uploads.

    Each blocking ``read`` is off-loaded to a worker thread
    (:func:`asyncio.to_thread`) so the event loop is never blocked on disk I/O
    while streaming a large file to a signed upload URL.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    total = (
        total_size
        if total_size is not None
        else await asyncio.to_thread(lambda: path.stat().st_size)
    )
    sent = 0
    fh = await asyncio.to_thread(path.open, "rb")
    try:
        while True:
            chunk = await asyncio.to_thread(fh.read, chunk_size)
            if not chunk:
                break
            sent += len(chunk)
            if progress is not None:
                progress(sent, total)
            yield chunk
    finally:
        await asyncio.to_thread(fh.close)
