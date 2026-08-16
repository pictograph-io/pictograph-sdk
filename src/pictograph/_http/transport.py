"""HTTP transport composing httpx, retry, idempotency, and error mapping.

The :class:`Transport` is the single integration point between the SDK and
``httpx``. Resource modules call :meth:`Transport.request`, :meth:`stream_bytes`,
:meth:`stream_sse`, and :meth:`upload_external` - they never touch ``httpx``
directly. That isolation lets us:

- Translate ``httpx`` exceptions to the SDK's structured exception hierarchy
  in exactly one place.
- Apply retry + idempotency uniformly to every call site without leaking
  policy knobs to resources.
- Swap the backend transport (e.g., for ``vcrpy`` cassettes or test mocks)
  without changing resource code.

The class is private (``_http`` package). Public callers use :class:`pictograph.Client`.
"""

from __future__ import annotations

import json as _json
import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

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
    chunked_file_iterator,
    parse_sse,
)
from pictograph._version import __version__
from pictograph.exceptions import (
    ApiError,
    NetworkError,
    RequestTimeoutError,
    ServerError,
    from_response,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from pathlib import Path
    from types import TracebackType

    from typing_extensions import Self

    from pictograph._internal.config import ClientConfig

logger = logging.getLogger(__name__)


# httpx connection-pool defaults tuned for batch ML workflows: 20 concurrent
# connections covers parallel image upload/download fan-out without exhausting
# server-side rate limits at typical tier caps.
DEFAULT_MAX_CONNECTIONS = 20
DEFAULT_MAX_KEEPALIVE = 10


def _build_user_agent() -> str:
    """User-Agent header - used by backend telemetry to identify SDK callers."""
    return f"pictograph-python/{__version__}"


def pin_url_to_base(path: str, base_url: str) -> str:
    """Reduce a URL to a path resolved against ``base_url``, ignoring its host.

    SECURITY: the authenticated client attaches the API key as a
    client-level default header, so EVERY request it makes carries the caller's
    credentials. Some request targets are SERVER-SUPPLIED absolute URLs - e.g.
    the ``annotation_url`` in a dataset ``/download`` listing is fetched through
    this transport. A malicious or compromised API could point one at another
    host, and httpx would honour that host, sending the ``X-API-Key`` straight
    to the attacker.

    So the transport never lets a server-supplied scheme/host decide where an
    authenticated request goes: an absolute URL is stripped to its path + query
    and re-resolved against the configured ``base_url``. A relative path is
    returned unchanged (httpx resolves it against ``base_url`` as before).

    This is host *pinning*, deliberately chosen over "strip the key on a
    cross-host fetch": pinning never contacts the foreign host at all (so it
    can't even be used as a blind-SSRF beacon), and it does not regress the
    documented direct-Cloud-Run ``base_url`` config, where a legitimate
    ``annotation_url`` on ``api.pictograph.io`` differs from a ``base_url``
    pointing straight at the backend's own service host yet names the same
    backend - the path is what matters.
    """
    parsed = urlsplit(path)
    if not parsed.scheme and not parsed.netloc:
        # Relative path (the common case) - httpx resolves it against base_url.
        return path
    base_host = urlsplit(base_url).netloc
    if parsed.netloc != base_host:
        logger.warning(
            "Ignoring server-supplied host %r on an authenticated request and "
            "pinning to the configured API host %r; the API key is only ever "
            "sent to the configured host.",
            parsed.netloc,
            base_host,
        )
    # Keep only path + query; drop scheme/host/fragment so httpx resolves the
    # remainder against base_url.
    return urlunsplit(("", "", parsed.path, parsed.query, ""))


def parse_json_response(response: httpx.Response) -> Any:
    """Decode a fully-read 2xx JSON body or raise the mapped SDK error.

    Pure (no I/O): the response body must already be read (true for both the
    sync ``client.request`` and the async ``await client.request`` non-stream
    paths). Shared by :class:`Transport` and
    :class:`pictograph._http.async_transport.AsyncTransport` so error mapping
    lives in exactly one place.
    """
    request_id = response.headers.get("X-Request-Id")

    if 200 <= response.status_code < 300:
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except _json.JSONDecodeError as e:
            # 2xx with garbage body - server bug. Surface as ServerError so
            # callers can treat it identically to a real 5xx.
            raise ServerError(
                f"Server returned {response.status_code} with non-JSON body: {e}",
                status_code=response.status_code,
                request_id=request_id,
                response=response.text,
            ) from e

    body: Any
    try:
        body = response.json()
    except _json.JSONDecodeError:
        body = response.text or None
    raise from_response(
        response.status_code,
        body=body,
        request_id=request_id,
        headers=response.headers,
    )


def raise_for_error_response(response: httpx.Response) -> None:
    """Map a non-2xx (already-read) streaming response to an SDK error and raise."""
    body: Any
    try:
        body = response.json()
    except _json.JSONDecodeError:
        body = response.text or None
    raise from_response(
        response.status_code,
        body=body,
        request_id=response.headers.get("X-Request-Id"),
        headers=response.headers,
    )


class Transport:
    """Composes httpx, retry, idempotency, and error mapping.

    Constructed by :class:`pictograph.Client`; resources receive a reference
    and use the high-level methods (``request``, ``stream_bytes``,
    ``stream_sse``, ``upload_external``).

    Args:
        config: Resolved client configuration (base URL, timeout, retries).
        api_key: Resolved API key (already-resolved from
            :func:`pictograph._internal.auth.resolve_api_key`).
        retry_policy: Optional override; defaults to a :class:`RetryPolicy`
            seeded from ``config.max_retries``.
        client: Optional pre-built ``httpx.Client``. Used by tests to inject
            a mock transport. When ``None``, a fresh client is built using
            ``config``.
    """

    def __init__(
        self,
        config: ClientConfig,
        api_key: str,
        *,
        retry_policy: RetryPolicy | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._api_key = api_key
        self._owns_client = client is None
        self._client = client if client is not None else self._build_client()
        self._retry = retry_policy or RetryPolicy(max_retries=config.max_retries)

    def _build_client(self) -> httpx.Client:
        # HTTP/1.1 on purpose. httpx+h2 is not thread-safe for concurrent
        # request calls on a shared client - the h2 connection's stream
        # dict mutates under iteration, raising
        # ``RuntimeError: dictionary changed size during iteration``.
        # Our workflows (upload_dataset_from_directory, etc.) use a
        # ThreadPoolExecutor over the single Client instance, so HTTP/1.1
        # keep-alive is the safe default. Per-thread clients would also
        # work but break the single-transport design.
        return httpx.Client(
            base_url=self._config.base_url.rstrip("/"),
            timeout=self._config.timeout,
            http2=False,
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

    def close(self) -> None:
        """Close the underlying httpx.Client (no-op for injected clients)."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # ───────────── request (JSON in/out) ─────────────

    def request(
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

        Returns ``None`` for 204 No Content or empty 2xx bodies. Raises an
        :class:`ApiError` subclass for non-2xx responses, :class:`NetworkError`
        / :class:`RequestTimeoutError` for transport failures.

        Args:
            method: HTTP method.
            path: Path relative to ``base_url`` (or absolute URL).
            params: Query parameters.
            json: Request body to JSON-encode.
            data: Form-encoded body (mutually exclusive with ``json``).
            files: Multipart files (mutually exclusive with ``json``).
            headers: Extra request headers (auth/UA already attached).
            idempotency_key: Override the auto-generated key for write methods.
                Ignored for ``GET``/``HEAD``/``OPTIONS``/``DELETE``.
            timeout: Per-request timeout override.
        """
        method_upper = method.upper()
        request_headers: dict[str, str] = dict(headers or {})
        # Never let a server-supplied host receive the API key.
        path = pin_url_to_base(path, self._config.base_url)

        if needs_idempotency(method_upper):
            request_headers.setdefault(IDEMPOTENCY_HEADER, idempotency_key or generate_key())
        has_idem = IDEMPOTENCY_HEADER in request_headers

        request_timeout = timeout if timeout is not None else self._config.timeout

        def do_request() -> httpx.Response:
            try:
                return self._client.request(
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

        response = self._retry.execute(
            do_request,
            method=method_upper,
            has_idempotency_key=has_idem,
        )
        return parse_json_response(response)

    # ───────────── streaming downloads ─────────────

    def stream_bytes(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> Iterator[bytes]:
        """Stream a response body as raw bytes (for image / model downloads).

        Errors before the first byte raise :class:`ApiError`/`NetworkError`.
        Errors mid-stream surface as :class:`NetworkError`. Streaming
        responses are not retried (resuming partial transfers is the
        backend's job).
        """
        request_timeout = timeout if timeout is not None else self._config.timeout
        request_headers: dict[str, str] = dict(headers or {})
        method_upper = method.upper()
        # Never let a server-supplied host receive the API key.
        path = pin_url_to_base(path, self._config.base_url)

        try:
            with self._client.stream(
                method_upper,
                path,
                params=params,
                headers=request_headers,
                timeout=request_timeout,
            ) as response:
                if response.status_code >= 300:
                    response.read()  # populate response.content for error parsing
                    raise_for_error_response(response)
                # 2xx - yield bytes incrementally.
                yield from response.iter_bytes()
        except httpx.TimeoutException as e:
            raise RequestTimeoutError(str(e)) from e
        except httpx.RequestError as e:
            raise NetworkError(str(e)) from e

    # ───────────── streaming SSE ─────────────

    def stream_sse(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        last_event_id: str | None = None,
    ) -> Iterator[SSEEvent]:
        """Stream Server-Sent Events from a long-lived endpoint.

        Read timeout is disabled (SSE connections are intentionally long).
        Connect timeout still applies. ``last_event_id`` adds the
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

        # Disable read timeout for SSE; keep connect timeout sensible.
        sse_timeout = httpx.Timeout(self._config.timeout, read=None)

        bytes_iter = self.stream_bytes(
            "GET",
            path,
            params=params,
            headers=sse_headers,
            timeout=sse_timeout,
        )
        yield from parse_sse(bytes_iter)

    # ───────────── chunked external upload ─────────────

    def upload_external(
        self,
        url: str,
        path: Path,
        *,
        content_type: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        progress: Callable[[int, int], None] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> None:
        """PUT a local file to an arbitrary URL (e.g., a signed upload URL).

        Bypasses the SDK base URL and authenticated headers - used for the
        Pictograph two-step upload flow where the backend issues a signed
        storage URL and the SDK PUTs the bytes straight to that host, never
        relaying them through the API.

        Args:
            url: Absolute URL to PUT to (typically a signed upload URL).
            path: Local file to upload.
            content_type: MIME type for the ``Content-Type`` header (must
                match what was passed when the signed URL was generated).
            chunk_size: Bytes per upload chunk.
            progress: Optional ``(sent, total)`` progress callback.
            timeout: Override timeout. Defaults to a generous read/write
                timeout because upload duration grows with file size.
        """
        upload_timeout: float | httpx.Timeout
        if timeout is None:
            upload_timeout = httpx.Timeout(self._config.timeout, read=300.0, write=300.0)
        else:
            upload_timeout = timeout

        # Stat ONCE and thread the result to the iterator's progress total, so
        # the Content-Length header and the streamed body's accounting derive
        # from the same measurement (a second independent stat could disagree
        # if the file is being written concurrently → a wrong Content-Length).
        file_size = path.stat().st_size
        body_iter = chunked_file_iterator(
            path, chunk_size=chunk_size, progress=progress, total_size=file_size
        )
        upload_headers = {
            "Content-Type": content_type,
            "Content-Length": str(file_size),
        }

        try:
            response = httpx.put(
                url,
                content=body_iter,
                headers=upload_headers,
                timeout=upload_timeout,
            )
        except httpx.TimeoutException as e:
            raise RequestTimeoutError(str(e)) from e
        except httpx.RequestError as e:
            raise NetworkError(str(e)) from e

        if response.status_code not in (200, 201):
            # Object-storage error responses are XML/text, not JSON; surface verbatim.
            raise ApiError(
                f"Upload failed: HTTP {response.status_code}",
                status_code=response.status_code,
                response=response.text,
            )
