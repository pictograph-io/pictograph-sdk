"""Async HTTP transport - the ``asyncio`` twin of :class:`Transport`.

:class:`AsyncTransport` is the single integration point between the async SDK
(:class:`pictograph.AsyncClient`) and ``httpx.AsyncClient``. It mirrors the sync
:class:`pictograph._http.transport.Transport` method-for-method - ``request``,
``stream_bytes``, ``stream_sse``, ``upload_external`` - and reuses every piece
of *pure* policy so the two transports can never drift:

- retry decisions + backoff (:meth:`RetryPolicy.execute_async`),
- idempotency-key attachment (:mod:`pictograph._http.idempotency`),
- ``httpx`` → SDK exception mapping and JSON decoding
  (:func:`pictograph._http.transport.parse_json_response` /
  :func:`~pictograph._http.transport.raise_for_error_response`).

Unlike the sync transport (HTTP/1.1, because it is shared across a
``ThreadPoolExecutor`` and ``httpx``+h2 is not thread-safe), the async client
runs entirely inside one event loop with cooperative concurrency, so HTTP/2
multiplexing is both safe and a genuine throughput win for the async fan-out
workflows (pairs with the backend's async serve paths).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from pictograph._http.idempotency import (
    IDEMPOTENCY_HEADER,
    generate_key,
    needs_idempotency,
)
from pictograph._http.retry import RetryPolicy
from pictograph._http.streaming import (
    DEFAULT_CHUNK_SIZE,
    SSEEvent,
    chunked_file_iterator_async,
    parse_sse_async,
)
from pictograph._http.transport import (
    DEFAULT_MAX_CONNECTIONS,
    DEFAULT_MAX_KEEPALIVE,
    parse_json_response,
    pin_url_to_base,
    raise_for_error_response,
)
from pictograph._version import __version__
from pictograph.exceptions import (
    ApiError,
    NetworkError,
    RequestTimeoutError,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping
    from pathlib import Path
    from types import TracebackType

    from typing_extensions import Self

    from pictograph._internal.config import ClientConfig


def _build_user_agent() -> str:
    """User-Agent header - flags async SDK callers to backend telemetry."""
    return f"pictograph-python-async/{__version__}"


class AsyncTransport:
    """Async composition of httpx, retry, idempotency, and error mapping.

    Constructed by :class:`pictograph.AsyncClient`; async resources receive a
    reference and use the high-level coroutines (``request``, ``stream_bytes``,
    ``stream_sse``, ``upload_external``).

    Args:
        config: Resolved client configuration (base URL, timeout, retries).
        api_key: Resolved API key.
        retry_policy: Optional override; defaults to a :class:`RetryPolicy`
            seeded from ``config.max_retries``.
        client: Optional pre-built ``httpx.AsyncClient`` (tests inject a mock
            transport). When ``None``, a fresh client is built from ``config``.
    """

    def __init__(
        self,
        config: ClientConfig,
        api_key: str,
        *,
        retry_policy: RetryPolicy | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._api_key = api_key
        self._owns_client = client is None
        self._client = client if client is not None else self._build_client()
        self._retry = retry_policy or RetryPolicy(max_retries=config.max_retries)

    def _build_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._config.base_url.rstrip("/"),
            timeout=self._config.timeout,
            http2=True,
            limits=httpx.Limits(
                max_connections=DEFAULT_MAX_CONNECTIONS,
                max_keepalive_connections=DEFAULT_MAX_KEEPALIVE,
            ),
            headers={
                "X-API-Key": self._api_key,
                "User-Agent": _build_user_agent(),
                "Accept": "application/json",
            },
        )

    # ───────────── lifecycle ─────────────

    async def aclose(self) -> None:
        """Close the underlying httpx.AsyncClient (no-op for injected clients)."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ───────────── request (JSON in/out) ─────────────

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        data: Any = None,
        files: Any = None,
        headers: Mapping[str, str] | None = None,
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Execute a JSON HTTP request and return the parsed body.

        Returns ``None`` for 204 / empty 2xx bodies. Raises an
        :class:`ApiError` subclass for non-2xx responses,
        :class:`NetworkError` / :class:`RequestTimeoutError` for transport
        failures. Semantics are identical to :meth:`Transport.request`.
        """
        method_upper = method.upper()
        request_headers: dict[str, str] = dict(headers or {})
        # Never let a server-supplied host receive the API key.
        path = pin_url_to_base(path, self._config.base_url)

        if needs_idempotency(method_upper):
            request_headers.setdefault(IDEMPOTENCY_HEADER, idempotency_key or generate_key())
        has_idem = IDEMPOTENCY_HEADER in request_headers

        request_timeout = timeout if timeout is not None else self._config.timeout

        async def do_request() -> httpx.Response:
            try:
                return await self._client.request(
                    method_upper,
                    path,
                    params=params,
                    json=json,
                    data=data,
                    files=files,
                    headers=request_headers,
                    timeout=request_timeout,
                )
            except httpx.TimeoutException as e:
                raise RequestTimeoutError(str(e)) from e
            except httpx.RequestError as e:
                raise NetworkError(str(e)) from e

        response = await self._retry.execute_async(
            do_request,
            method=method_upper,
            has_idempotency_key=has_idem,
        )
        return parse_json_response(response)

    # ───────────── streaming downloads ─────────────

    async def stream_bytes(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream a response body as raw bytes (image / model downloads).

        Errors before the first byte raise :class:`ApiError`/`NetworkError`;
        mid-stream errors surface as :class:`NetworkError`. Streaming responses
        are not retried (resuming a partial transfer is the backend's job).
        """
        request_timeout = timeout if timeout is not None else self._config.timeout
        request_headers: dict[str, str] = dict(headers or {})
        method_upper = method.upper()
        # Never let a server-supplied host receive the API key.
        path = pin_url_to_base(path, self._config.base_url)

        try:
            async with self._client.stream(
                method_upper,
                path,
                params=params,
                headers=request_headers,
                timeout=request_timeout,
            ) as response:
                if response.status_code >= 300:
                    await response.aread()  # populate content for error parsing
                    raise_for_error_response(response)
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.TimeoutException as e:
            raise RequestTimeoutError(str(e)) from e
        except httpx.RequestError as e:
            raise NetworkError(str(e)) from e

    # ───────────── streaming SSE ─────────────

    async def stream_sse(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        last_event_id: str | None = None,
    ) -> AsyncIterator[SSEEvent]:
        """Stream Server-Sent Events from a long-lived endpoint.

        Read timeout is disabled (SSE connections are intentionally long);
        connect timeout still applies. ``last_event_id`` adds the
        ``Last-Event-ID`` header for resumable streams.
        """
        sse_headers: dict[str, str] = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
        }
        if headers:
            sse_headers.update(headers)
        if last_event_id is not None:
            sse_headers["Last-Event-ID"] = last_event_id

        sse_timeout = httpx.Timeout(self._config.timeout, read=None)

        bytes_iter = self.stream_bytes(
            "GET",
            path,
            params=params,
            headers=sse_headers,
            timeout=sse_timeout,
        )
        async for event in parse_sse_async(bytes_iter):
            yield event

    # ───────────── chunked external upload ─────────────

    async def upload_external(
        self,
        url: str,
        path: Path,
        *,
        content_type: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        progress: Callable[[int, int], None] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> None:
        """PUT a local file to an arbitrary URL (e.g. a signed upload URL).

        Bypasses the SDK base URL and authenticated headers - used for the
        two-step upload flow. The file is streamed off the event loop via
        :func:`chunked_file_iterator_async`.
        """
        upload_timeout: float | httpx.Timeout
        if timeout is None:
            upload_timeout = httpx.Timeout(self._config.timeout, read=300.0, write=300.0)
        else:
            upload_timeout = timeout

        file_size = path.stat().st_size
        body_iter = chunked_file_iterator_async(
            path, chunk_size=chunk_size, progress=progress, total_size=file_size
        )
        upload_headers = {
            "Content-Type": content_type,
            "Content-Length": str(file_size),
        }

        try:
            async with httpx.AsyncClient(timeout=upload_timeout) as put_client:
                response = await put_client.put(
                    url,
                    content=body_iter,
                    headers=upload_headers,
                )
        except httpx.TimeoutException as e:
            raise RequestTimeoutError(str(e)) from e
        except httpx.RequestError as e:
            raise NetworkError(str(e)) from e

        if response.status_code not in (200, 201):
            raise ApiError(
                f"Upload failed: HTTP {response.status_code}",
                status_code=response.status_code,
                response=response.text,
            )
