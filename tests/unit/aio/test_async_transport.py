"""Tests for ``pictograph._http.async_transport.AsyncTransport``.

Async mirror of ``tests/unit/_http/test_transport.py`` - same behavioural
contracts (auth headers, idempotency keys, error translation, retry, streaming),
verified against ``httpx.AsyncClient`` via ``pytest-httpx``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from pictograph._http.async_transport import AsyncTransport
from pictograph._http.idempotency import IDEMPOTENCY_HEADER
from pictograph._http.retry import RetryPolicy
from pictograph._http.streaming import SSEEvent
from pictograph._internal.config import ClientConfig
from pictograph._version import __version__
from pictograph.exceptions import (
    ApiError,
    AuthError,
    NetworkError,
    NotFoundError,
    PaymentRequiredError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
    ValidationError,
)

from .conftest import API_KEY, BASE_URL

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from pytest_httpx import HTTPXMock

pytestmark = pytest.mark.anyio


async def _no_sleep(_: float) -> None:
    return None


@pytest.fixture
async def transport_with_retries(config: ClientConfig) -> AsyncIterator[AsyncTransport]:
    policy = RetryPolicy(
        max_retries=2,
        backoff_base=0.01,
        async_sleep=_no_sleep,
        rng=lambda _a, _b: 1.0,
    )
    t = AsyncTransport(config, api_key=API_KEY, retry_policy=policy)
    yield t
    await t.aclose()


# ───────────── auth + headers + idempotency ─────────────


async def test_attaches_auth_and_ua_headers(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    httpx_mock.add_response(method="GET", url=f"{BASE_URL}/api/v1/x", json={"ok": 1})
    await transport.request("GET", "/api/v1/x")
    sent = httpx_mock.get_request()
    assert sent is not None
    assert sent.headers["X-API-Key"] == API_KEY
    assert sent.headers["User-Agent"] == f"pictograph-python-async/{__version__}"
    assert sent.headers["Accept"] == "application/json"


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH"])
async def test_write_methods_get_idempotency_key(
    httpx_mock: HTTPXMock, transport: AsyncTransport, method: str
) -> None:
    httpx_mock.add_response(method=method, url=f"{BASE_URL}/api/v1/x", json={})
    await transport.request(method, "/api/v1/x", json={"a": 1})
    sent = httpx_mock.get_request()
    assert sent is not None
    assert IDEMPOTENCY_HEADER in sent.headers


@pytest.mark.parametrize("method", ["GET", "DELETE"])
async def test_read_methods_no_idempotency_key(
    httpx_mock: HTTPXMock, transport: AsyncTransport, method: str
) -> None:
    httpx_mock.add_response(method=method, url=f"{BASE_URL}/api/v1/x", json={})
    await transport.request(method, "/api/v1/x")
    sent = httpx_mock.get_request()
    assert sent is not None
    assert IDEMPOTENCY_HEADER not in sent.headers


async def test_explicit_idempotency_key_honoured(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    httpx_mock.add_response(method="POST", url=f"{BASE_URL}/api/v1/x", json={})
    await transport.request("POST", "/api/v1/x", json={}, idempotency_key="my-key")
    sent = httpx_mock.get_request()
    assert sent is not None
    assert sent.headers[IDEMPOTENCY_HEADER] == "my-key"


# ───────────── response decoding ─────────────


async def test_204_returns_none(httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
    httpx_mock.add_response(method="DELETE", url=f"{BASE_URL}/api/v1/x", status_code=204)
    assert await transport.request("DELETE", "/api/v1/x") is None


async def test_empty_2xx_body_returns_none(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    httpx_mock.add_response(method="GET", url=f"{BASE_URL}/api/v1/x", status_code=200, content=b"")
    assert await transport.request("GET", "/api/v1/x") is None


async def test_non_json_2xx_raises_server_error(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{BASE_URL}/api/v1/x", status_code=200, content=b"<html>nope</html>"
    )
    with pytest.raises(ServerError):
        await transport.request("GET", "/api/v1/x")


# ───────────── error mapping ─────────────


@pytest.mark.parametrize(
    ("status", "exc"),
    [
        (401, AuthError),
        (402, PaymentRequiredError),
        (404, NotFoundError),
        (422, ValidationError),
        (429, RateLimitError),
        (500, ServerError),
    ],
)
async def test_error_status_maps_to_typed_exception(
    httpx_mock: HTTPXMock,
    transport: AsyncTransport,
    status: int,
    exc: type[Exception],
) -> None:
    # 429/5xx would retry on a retrying transport; the default has max_retries=0.
    httpx_mock.add_response(
        method="GET", url=f"{BASE_URL}/api/v1/x", status_code=status, json={"detail": "boom"}
    )
    with pytest.raises(exc):
        await transport.request("GET", "/api/v1/x")


async def test_timeout_maps_to_request_timeout(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    httpx_mock.add_exception(httpx.ReadTimeout("slow"), method="GET", url=f"{BASE_URL}/api/v1/x")
    with pytest.raises(RequestTimeoutError):
        await transport.request("GET", "/api/v1/x")


async def test_network_error_maps(httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
    httpx_mock.add_exception(
        httpx.ConnectError("no route"), method="GET", url=f"{BASE_URL}/api/v1/x"
    )
    with pytest.raises(NetworkError):
        await transport.request("GET", "/api/v1/x")


# ───────────── retry (execute_async) ─────────────


async def test_retry_succeeds_after_transient_503(
    httpx_mock: HTTPXMock, transport_with_retries: AsyncTransport
) -> None:
    httpx_mock.add_response(method="GET", url=f"{BASE_URL}/api/v1/x", status_code=503, json={})
    httpx_mock.add_response(
        method="GET", url=f"{BASE_URL}/api/v1/x", status_code=200, json={"ok": 1}
    )
    result = await transport_with_retries.request("GET", "/api/v1/x")
    assert result == {"ok": 1}
    assert len(httpx_mock.get_requests()) == 2


async def test_retry_exhausted_raises(
    httpx_mock: HTTPXMock, transport_with_retries: AsyncTransport
) -> None:
    for _ in range(3):  # initial + 2 retries
        httpx_mock.add_response(method="GET", url=f"{BASE_URL}/api/v1/x", status_code=503, json={})
    with pytest.raises(ServerError):
        await transport_with_retries.request("GET", "/api/v1/x")
    assert len(httpx_mock.get_requests()) == 3


async def test_post_auto_keyed_retries_reuse_same_key(
    httpx_mock: HTTPXMock, transport_with_retries: AsyncTransport
) -> None:
    httpx_mock.add_response(method="POST", url=f"{BASE_URL}/api/v1/x", status_code=503, json={})
    httpx_mock.add_response(
        method="POST", url=f"{BASE_URL}/api/v1/x", status_code=200, json={"ok": 1}
    )
    result = await transport_with_retries.request("POST", "/api/v1/x", json={"y": 1})
    assert result == {"ok": 1}
    reqs = httpx_mock.get_requests()
    assert len(reqs) == 2
    assert reqs[0].headers[IDEMPOTENCY_HEADER] == reqs[1].headers[IDEMPOTENCY_HEADER]


async def test_network_error_retried_then_succeeds(
    httpx_mock: HTTPXMock, transport_with_retries: AsyncTransport
) -> None:
    httpx_mock.add_exception(httpx.ConnectError("blip"), method="GET", url=f"{BASE_URL}/api/v1/x")
    httpx_mock.add_response(
        method="GET", url=f"{BASE_URL}/api/v1/x", status_code=200, json={"ok": 1}
    )
    assert await transport_with_retries.request("GET", "/api/v1/x") == {"ok": 1}


# ───────────── streaming ─────────────


async def test_stream_bytes_yields_chunks(httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
    httpx_mock.add_response(method="GET", url=f"{BASE_URL}/api/v1/img", content=b"abcdef")
    chunks = [c async for c in transport.stream_bytes("GET", "/api/v1/img")]
    assert b"".join(chunks) == b"abcdef"


async def test_stream_bytes_error_before_first_byte_raises(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{BASE_URL}/api/v1/img", status_code=404, json={"detail": "gone"}
    )
    with pytest.raises(NotFoundError):
        _ = [c async for c in transport.stream_bytes("GET", "/api/v1/img")]


async def test_stream_sse_parses_events(httpx_mock: HTTPXMock, transport: AsyncTransport) -> None:
    body = b'event: progress\ndata: {"pct": 50}\n\ndata: "done"\n\n'
    httpx_mock.add_response(method="GET", url=f"{BASE_URL}/api/v1/stream", content=body)
    events = [e async for e in transport.stream_sse("/api/v1/stream")]
    assert events == [
        SSEEvent(event="progress", data={"pct": 50}),
        SSEEvent(event="message", data="done"),
    ]


# ───────────── upload_external ─────────────


async def test_upload_external_puts_file(
    httpx_mock: HTTPXMock, transport: AsyncTransport, tmp_path: Path
) -> None:
    f = tmp_path / "x.jpg"
    f.write_bytes(b"imagedata")
    httpx_mock.add_response(method="PUT", url="https://gcs.example/signed", status_code=200)
    sent_progress: list[tuple[int, int]] = []
    await transport.upload_external(
        "https://gcs.example/signed",
        f,
        content_type="image/jpeg",
        progress=lambda s, t: sent_progress.append((s, t)),
    )
    req = httpx_mock.get_request(url="https://gcs.example/signed")
    assert req is not None
    assert req.headers["Content-Type"] == "image/jpeg"
    assert req.headers["Content-Length"] == str(len(b"imagedata"))
    assert sent_progress and sent_progress[-1] == (len(b"imagedata"), len(b"imagedata"))


async def test_upload_external_non_2xx_raises_api_error(
    httpx_mock: HTTPXMock, transport: AsyncTransport, tmp_path: Path
) -> None:
    f = tmp_path / "x.jpg"
    f.write_bytes(b"data")
    httpx_mock.add_response(method="PUT", url="https://gcs.example/signed", status_code=403)
    with pytest.raises(ApiError):
        await transport.upload_external("https://gcs.example/signed", f, content_type="image/jpeg")


# ───────────── lifecycle ─────────────


async def test_context_manager_closes(config: ClientConfig) -> None:
    async with AsyncTransport(config, api_key=API_KEY) as t:
        assert t is not None
    # aclose is idempotent
    await t.aclose()


# ───────────── auth host pinning ─────────────
#
# Async mirror: the same server-supplied-host credential leak, closed at the
# transport layer, so every current and future async call site is covered.


async def test_absolute_url_pinned_to_base_host(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    httpx_mock.add_response(json={})  # catch-all
    await transport.request("GET", "https://evil.example/steal?x=1")
    reqs = httpx_mock.get_requests()
    assert reqs, "no request captured"
    assert all(r.url.host == "api.test.local" for r in reqs), (
        f"request leaked to a foreign host: {[str(r.url) for r in reqs]}"
    )
    assert not any(r.url.host == "evil.example" for r in reqs)
    assert not any(r.url.host != "api.test.local" and r.headers.get("X-API-Key") for r in reqs)
    assert reqs[-1].url.path == "/steal"
    assert reqs[-1].url.query == b"x=1"


async def test_stream_bytes_absolute_url_pinned_to_base_host(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    httpx_mock.add_response(content=b"data")
    async for _ in transport.stream_bytes("GET", "https://evil.example/leak"):
        pass
    reqs = httpx_mock.get_requests()
    assert reqs, "no request captured"
    assert all(r.url.host == "api.test.local" for r in reqs)
    assert not any(r.url.host == "evil.example" for r in reqs)


async def test_same_host_absolute_url_unchanged(
    httpx_mock: HTTPXMock, transport: AsyncTransport
) -> None:
    httpx_mock.add_response(json={"annotations": []})
    result = await transport.request("GET", f"{BASE_URL}/api/v1/developer/annotations/abc/file")
    assert result == {"annotations": []}
    sent = httpx_mock.get_request()
    assert sent is not None
    assert sent.url.host == "api.test.local"
    assert sent.url.path == "/api/v1/developer/annotations/abc/file"
    assert sent.headers["X-API-Key"] == API_KEY
