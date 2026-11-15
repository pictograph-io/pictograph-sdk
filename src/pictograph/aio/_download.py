"""Shared async helper: stream a signed URL to a local file with atomic rename.

Used by :class:`pictograph.aio.resources.exports.AsyncExports`,
:class:`~pictograph.aio.resources.models.AsyncModels`, and
:class:`~pictograph.aio.resources.datasets.AsyncDatasets`. Mirrors the sync
``_stream_to_file`` helpers: bytes land in a sibling ``.part`` file and are
renamed atomically on success, so a failed transfer never leaves a truncated
file at the destination. Local disk writes are off-loaded to a worker thread so
the event loop is never blocked on file I/O between network chunks.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from pictograph._http.streaming import DEFAULT_CHUNK_SIZE
from pictograph.exceptions import ApiError

if TYPE_CHECKING:
    from collections.abc import Callable


async def stream_url_to_file(
    download_url: str,
    output_path: str | Path,
    *,
    timeout: float,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress: Callable[[int, int], None] | None = None,
    error_prefix: str = "Download",
) -> Path:
    """Stream ``download_url`` (an unauthenticated signed URL) to ``output_path``.

    Args:
        download_url: Absolute signed URL to GET (no SDK auth is attached).
        output_path: Local destination. Parent dirs are created if missing.
        timeout: Connect timeout seconds (read timeout is a generous 600s).
        chunk_size: Streaming chunk size (default 8 MB).
        progress: Optional ``(bytes_so_far, total_bytes)`` callback.
        error_prefix: Prefix for the :class:`ApiError` message on a non-2xx.

    Returns:
        The output path.

    Raises:
        ApiError: The signed-URL GET returned a non-2xx status.
    """
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")

    try:
        async with (
            httpx.AsyncClient(
                http2=True,
                timeout=httpx.Timeout(timeout, read=600.0),
            ) as gcs,
            gcs.stream("GET", download_url) as response,
        ):
            if response.status_code >= 300:
                await response.aread()
                raise ApiError(
                    f"{error_prefix} failed: HTTP {response.status_code}",
                    status_code=response.status_code,
                    response=response.text,
                )
            total = int(response.headers.get("Content-Length", 0))
            sent = 0
            fh = await asyncio.to_thread(tmp.open, "wb")
            try:
                async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                    await asyncio.to_thread(fh.write, chunk)
                    sent += len(chunk)
                    if progress is not None:
                        progress(sent, total)
            finally:
                await asyncio.to_thread(fh.close)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise

    await asyncio.to_thread(tmp.replace, out)
    return out
