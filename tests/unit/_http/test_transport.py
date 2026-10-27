"""Tests for ``pictograph._http.transport.Transport``.

The Transport composes httpx + retry + idempotency + error mapping. These
tests exercise each integration point at the boundary level - they verify
*behavioural contracts* (auth headers attached, idempotency keys generated,
errors translated, streaming works) rather than re-testing the lower-level
modules (those are covered by their own unit tests).

We use ``pytest-httpx``'s ``httpx_mock`` fixture which patches httpx so any
client (including ones created inside Transport) is intercepted.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import httpx
import pytest

from pictograph._http.idempotency import IDEMPOTENCY_HEADER
from pictograph._http.retry import RetryPolicy
from pictograph._http.streaming import SSEEvent
from pictograph._http.transport import Transport
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

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_httpx import HTTPXMock

BASE_URL = "https://api.test.local"
API_KEY = "pk_live_test"


# ───────────── fixtures ─────────────


@pytest.fixture
def config() -> ClientConfig:
    return ClientConfig(
        api_key=API_KEY,  # type: ignore[arg-type]
        base_url=BASE_URL,
        timeout=10.0,
        max_retries=2,
    )


@pytest.fixture
def transport(config: ClientConfig) -> Transport:
    """Default transport with retries disabled for fast, deterministic tests.

    Tests that need to exercise retry behaviour use ``transport_with_retries``
    instead and register the matching number of canned responses.
    """
    policy = RetryPolicy(max_retries=0, sleep=lambda _: None, rng=lambda _a, _b: 1.0)
    t = Transport(config, api_key=API_KEY, retry_policy=policy)
    yield t
    t.close()


@pytest.fixture
def transport_with_retries(config: ClientConfig) -> Transport:
    """Transport with 2 retries (3 attempts total), instant sleep, fixed jitter."""
    policy = RetryPolicy(max_retries=2, sleep=lambda _: None, rng=lambda _a, _b: 1.0)
    t = Transport(config, api_key=API_KEY, retry_policy=policy)
    yield t
    t.close()


# ───────────── basic request / response handling ─────────────


def test_get_returns_parsed_json(httpx_mock: HTTPXMock, transport: Transport) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/api/v1/datasets",
        json={"datasets": [{"id": "1"}]},
    )
    result = transport.request("GET", "/api/v1/datasets")
    assert result == {"datasets": [{"id": "1"}]}


def test_204_returns_none(httpx_mock: HTTPXMock, transport: Transport) -> None:
    httpx_mock.add_response(
        method="DELETE",
        url=f"{BASE_URL}/api/v1/x/1",
        status_code=204,
    )
    assert transport.request("DELETE", "/api/v1/x/1") is None


def test_2xx_with_empty_body_returns_none(httpx_mock: HTTPXMock, transport: Transport) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/api/v1/empty",
        status_code=200,
        content=b"",
    )
    assert transport.request("GET", "/api/v1/empty") is None


# ───────────── headers (auth + UA) ─────────────


def test_x_api_key_header_attached(httpx_mock: HTTPXMock, transport: Transport) -> None:
    httpx_mock.add_response(method="GET", url=f"{BASE_URL}/api/v1/x", json={})
    transport.request("GET", "/api/v1/x")
    sent = httpx_mock.get_request()
    assert sent is not None
    assert sent.headers["X-API-Key"] == API_KEY


def test_user_agent_header_includes_sdk_version(
    httpx_mock: HTTPXMock, transport: Transport
) -> None:
    httpx_mock.add_response(method="GET", url=f"{BASE_URL}/api/v1/x", json={})
    transport.request("GET", "/api/v1/x")
    sent = httpx_mock.get_request()
    assert sent is not None
    assert sent.headers["User-Agent"] == f"pictograph-python/{__version__}"


def test_accept_header_is_json_by_default(httpx_mock: HTTPXMock, transport: Transport) -> None:
    httpx_mock.add_response(method="GET", url=f"{BASE_URL}/api/v1/x", json={})
    transport.request("GET", "/api/v1/x")
    sent = httpx_mock.get_request()
    assert sent is not None
    assert "application/json" in sent.headers["Accept"]


def test_extra_headers_merged_with_defaults(httpx_mock: HTTPXMock, transport: Transport) -> None:
    httpx_mock.add_response(method="GET", url=f"{BASE_URL}/api/v1/x", json={})
    transport.request("GET", "/api/v1/x", headers={"X-Custom": "v"})
    sent = httpx_mock.get_request()
    assert sent is not None
    assert sent.headers["X-Custom"] == "v"
    # Defaults must still be present.
    assert sent.headers["X-API-Key"] == API_KEY


# ───────────── idempotency keys ─────────────


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH"])
def test_writes_get_auto_idempotency_key(
    httpx_mock: HTTPXMock, transport: Transport, method: str
) -> None:
    httpx_mock.add_response(method=method, url=f"{BASE_URL}/api/v1/x", json={})
    transport.request(method, "/api/v1/x", json={"y": 1})
    sent = httpx_mock.get_request()
    assert sent is not None
    key = sent.headers.get(IDEMPOTENCY_HEADER)
    assert key is not None
    assert re.fullmatch(r"[0-9a-f]{32}", key) is not None


@pytest.mark.parametrize("method", ["GET", "HEAD", "DELETE", "OPTIONS"])
def test_reads_do_not_get_idempotency_key(
    httpx_mock: HTTPXMock, transport: Transport, method: str
) -> None:
    httpx_mock.add_response(method=method, url=f"{BASE_URL}/api/v1/x", json={})
    transport.request(method, "/api/v1/x")
    sent = httpx_mock.get_request()
    assert sent is not None
    assert IDEMPOTENCY_HEADER not in sent.headers


def test_user_supplied_idempotency_key_used(httpx_mock: HTTPXMock, transport: Transport) -> None:
    httpx_mock.add_response(method="POST", url=f"{BASE_URL}/api/v1/x", json={})
    transport.request("POST", "/api/v1/x", json={"y": 1}, idempotency_key="user-key-123")
    sent = httpx_mock.get_request()
    assert sent is not None
    assert sent.headers[IDEMPOTENCY_HEADER] == "user-key-123"


def test_user_idempotency_key_via_headers_kwarg_preserved(
    httpx_mock: HTTPXMock, transport: Transport
) -> None:
    # If the caller already set the header, don't overwrite.
    httpx_mock.add_response(method="POST", url=f"{BASE_URL}/api/v1/x", json={})
    transport.request("POST", "/api/v1/x", json={}, headers={IDEMPOTENCY_HEADER: "preset"})
    sent = httpx_mock.get_request()
    assert sent is not None
    assert sent.headers[IDEMPOTENCY_HEADER] == "preset"


# ───────────── error mapping ─────────────


@pytest.mark.parametrize(
    ("status", "expected_cls"),
    [
        (400, ValidationError),
        (401, AuthError),
        (402, PaymentRequiredError),
        (404, NotFoundError),
        (422, ValidationError),
        (429, RateLimitError),
        (500, ServerError),
    ],
)
def test_error_responses_translated_to_typed_exceptions(
    httpx_mock: HTTPXMock,
    transport: Transport,
    status: int,
    expected_cls: type[ApiError],
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/api/v1/x",
        status_code=status,
        json={"detail": "msg"},
    )
    with pytest.raises(expected_cls) as exc:
        transport.request("GET", "/api/v1/x")
    assert exc.value.status_code == status


def test_error_includes_request_id_from_response_header(
    httpx_mock: HTTPXMock, transport: Transport
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/api/v1/x",
        status_code=500,
        headers={"X-Request-Id": "req_abc"},
        json={"detail": "boom"},
    )
    with pytest.raises(ServerError) as exc:
        transport.request("GET", "/api/v1/x")
    assert exc.value.request_id == "req_abc"


def test_error_with_non_json_body_uses_raw_text(
    httpx_mock: HTTPXMock, transport: Transport
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/api/v1/x",
        status_code=502,
        content=b"<html>Bad Gateway</html>",
        headers={"Content-Type": "text/html"},
    )
    with pytest.raises(ServerError) as exc:
        transport.request("GET", "/api/v1/x")
    assert "Bad Gateway" in str(exc.value.response)


def test_2xx_with_invalid_json_raises_server_error(
    httpx_mock: HTTPXMock, transport: Transport
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/api/v1/x",
        status_code=200,
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(ServerError) as exc:
        transport.request("GET", "/api/v1/x")
    assert exc.value.status_code == 200


# ───────────── transport-layer exceptions ─────────────


def test_timeout_translated_to_request_timeout_error(
    httpx_mock: HTTPXMock, transport: Transport
) -> None:
    httpx_mock.add_exception(httpx.ReadTimeout("read timeout"))
    with pytest.raises(RequestTimeoutError):
        transport.request("GET", "/api/v1/x")


def test_connect_error_translated_to_network_error(
    httpx_mock: HTTPXMock, transport: Transport
) -> None:
    httpx_mock.add_exception(httpx.ConnectError("dns failure"))
    with pytest.raises(NetworkError):
        transport.request("GET", "/api/v1/x")


# ───────────── retry integration ─────────────


def test_retry_succeeds_after_transient_503(
    httpx_mock: HTTPXMock, transport_with_retries: Transport
) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{BASE_URL}/api/v1/x", status_code=503, json={"detail": "x"}
    )
    httpx_mock.add_response(
        method="GET", url=f"{BASE_URL}/api/v1/x", status_code=200, json={"ok": True}
    )
    result = transport_with_retries.request("GET", "/api/v1/x")
    assert result == {"ok": True}
    # Two requests sent - one failed, one succeeded.
    assert len(httpx_mock.get_requests()) == 2


def test_retry_exhausted_raises_typed_error(
    httpx_mock: HTTPXMock, transport_with_retries: Transport
) -> None:
    # max_retries=2 → 3 attempts total.
    for _ in range(3):
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE_URL}/api/v1/x",
            status_code=503,
            json={"detail": "still down"},
        )
    with pytest.raises(ServerError):
        transport_with_retries.request("GET", "/api/v1/x")
    assert len(httpx_mock.get_requests()) == 3


def test_post_without_idempotency_kwarg_still_retried_because_auto_keyed(
    httpx_mock: HTTPXMock, transport_with_retries: Transport
) -> None:
    # The SDK auto-attaches an Idempotency-Key for POST → retries are safe.
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/api/v1/x",
        status_code=503,
        json={"detail": "x"},
    )
    httpx_mock.add_response(
        method="POST", url=f"{BASE_URL}/api/v1/x", status_code=200, json={"ok": True}
    )
    result = transport_with_retries.request("POST", "/api/v1/x", json={"y": 1})
    assert result == {"ok": True}
    # Both retries used the SAME idempotency key - that's the whole point.
    requests = httpx_mock.get_requests()
    assert len(requests) == 2
    assert requests[0].headers[IDEMPOTENCY_HEADER] == requests[1].headers[IDEMPOTENCY_HEADER]


# ───────────── streaming bytes ─────────────


def test_stream_bytes_yields_chunks(httpx_mock: HTTPXMock, transport: Transport) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/api/v1/img/1",
        content=b"image_payload_bytes",
    )
    chunks = b"".join(transport.stream_bytes("GET", "/api/v1/img/1"))
    assert chunks == b"image_payload_bytes"


def test_stream_bytes_4xx_raises_typed_error_before_yielding(
    httpx_mock: HTTPXMock, transport: Transport
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/api/v1/img/1",
        status_code=404,
        json={"detail": "missing"},
    )
    with pytest.raises(NotFoundError):
        # Drain the iterator to force the request.
        list(transport.stream_bytes("GET", "/api/v1/img/1"))


# ───────────── streaming SSE ─────────────


def test_stream_sse_yields_parsed_events(httpx_mock: HTTPXMock, transport: Transport) -> None:
    sse_payload = (
        b'event: progress\ndata: {"epoch": 1}\n\n'
        b'event: progress\ndata: {"epoch": 2}\n\n'
        b'event: done\ndata: {"final": true}\n\n'
    )
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/api/v1/training/1/stream",
        content=sse_payload,
        headers={"Content-Type": "text/event-stream"},
    )
    events = list(transport.stream_sse("/api/v1/training/1/stream"))
    assert len(events) == 3
    assert all(isinstance(e, SSEEvent) for e in events)
    assert [e.event for e in events] == ["progress", "progress", "done"]
    assert events[0].data == {"epoch": 1}
    assert events[2].data == {"final": True}


def test_stream_sse_attaches_last_event_id_header(
    httpx_mock: HTTPXMock, transport: Transport
) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/api/v1/training/1/stream",
        content=b"data: x\n\n",
    )
    list(transport.stream_sse("/api/v1/training/1/stream", last_event_id="evt-42"))
    sent = httpx_mock.get_request()
    assert sent is not None
    assert sent.headers["Last-Event-ID"] == "evt-42"
    assert sent.headers["Accept"] == "text/event-stream"


# ───────────── upload_external (chunked file PUT) ─────────────


def test_upload_external_puts_file_to_signed_url(
    httpx_mock: HTTPXMock, transport: Transport, tmp_path: Path
) -> None:
    f = tmp_path / "img.jpg"
    f.write_bytes(b"a" * 100)
    httpx_mock.add_response(
        method="PUT",
        url="https://storage.example/upload-token",
        status_code=200,
    )
    transport.upload_external(
        "https://storage.example/upload-token",
        f,
        content_type="image/jpeg",
    )
    sent = httpx_mock.get_request()
    assert sent is not None
    assert sent.headers["Content-Type"] == "image/jpeg"
    assert sent.headers["Content-Length"] == "100"


def test_upload_external_invokes_progress_callback(
    httpx_mock: HTTPXMock, transport: Transport, tmp_path: Path
) -> None:
    f = tmp_path / "img.jpg"
    f.write_bytes(b"a" * 32)
    httpx_mock.add_response(method="PUT", url="https://storage/u", status_code=200)
    seen: list[tuple[int, int]] = []

    def record(sent: int, total: int) -> None:
        seen.append((sent, total))

    transport.upload_external(
        "https://storage/u",
        f,
        content_type="image/jpeg",
        chunk_size=8,
        progress=record,
    )
    assert seen == [(8, 32), (16, 32), (24, 32), (32, 32)]


def test_upload_external_failure_raises_apierror(
    httpx_mock: HTTPXMock, transport: Transport, tmp_path: Path
) -> None:
    f = tmp_path / "img.jpg"
    f.write_bytes(b"x")
    httpx_mock.add_response(
        method="PUT",
        url="https://storage/u",
        status_code=403,
        content=b"<Error>denied</Error>",
    )
    with pytest.raises(ApiError) as exc:
        transport.upload_external("https://storage/u", f, content_type="image/jpeg")
    assert exc.value.status_code == 403
    assert "denied" in str(exc.value.response)


def test_upload_external_network_error(
    httpx_mock: HTTPXMock, transport: Transport, tmp_path: Path
) -> None:
    f = tmp_path / "img.jpg"
    f.write_bytes(b"x")
    httpx_mock.add_exception(httpx.ConnectError("no route"))
    with pytest.raises(NetworkError):
        transport.upload_external("https://storage/u", f, content_type="image/jpeg")


# ───────────── lifecycle ─────────────


def test_close_owned_client_marks_underlying_httpx_client_closed(
    config: ClientConfig,
) -> None:
    """The default client is owned; close() must release its sockets."""
    t = Transport(config, api_key=API_KEY)
    assert t._client.is_closed is False
    t.close()
    assert t._client.is_closed is True


def test_owned_client_close_is_idempotent(config: ClientConfig) -> None:
    """A second close() on an owned client must be a no-op, not an error."""
    t = Transport(config, api_key=API_KEY)
    t.close()
    t.close()  # must not raise
    assert t._client.is_closed is True


def test_context_manager_closes_owned_client_on_exit(config: ClientConfig) -> None:
    """Exiting the with-statement releases sockets via httpx.Client.close()."""
    t = Transport(config, api_key=API_KEY)
    with t as ctx:
        # Inside the context the client is open - calls work.
        assert ctx is t
        assert t._client.is_closed is False
    assert t._client.is_closed is True


def test_injected_client_is_not_closed_by_transport_close(config: ClientConfig) -> None:
    """The caller-owned client must survive Transport.close() - it may be shared."""
    client = httpx.Client(base_url=BASE_URL)
    t = Transport(config, api_key=API_KEY, client=client)
    t.close()
    # The injected client must remain usable after Transport.close().
    assert client.is_closed is False
    client.close()
    assert client.is_closed is True


def test_injected_client_survives_context_manager_exit(config: ClientConfig) -> None:
    """Same as above but via context manager - ownership rule is symmetric."""
    client = httpx.Client(base_url=BASE_URL)
    with Transport(config, api_key=API_KEY, client=client):
        pass
    assert client.is_closed is False
    client.close()


# ───────────── per-request timeout override ─────────────


def test_request_timeout_kwarg_overrides_client_default(
    httpx_mock: HTTPXMock, transport: Transport
) -> None:
    httpx_mock.add_response(method="GET", url=f"{BASE_URL}/api/v1/slow", json={"ok": True})
    transport.request("GET", "/api/v1/slow", timeout=5.0)
    # We can't directly assert on timeout via pytest-httpx, but we can ensure
    # the request succeeded - i.e., the override didn't break the path.
    assert httpx_mock.get_request() is not None


# ───────────── params propagation ─────────────


def test_query_params_attached_to_request(httpx_mock: HTTPXMock, transport: Transport) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/api/v1/x?limit=50&offset=100",
        json={},
    )
    transport.request("GET", "/api/v1/x", params={"limit": 50, "offset": 100})
    sent = httpx_mock.get_request()
    assert sent is not None
    assert "limit=50" in str(sent.url)
    assert "offset=100" in str(sent.url)


# ───────────── json body propagation ─────────────


def test_json_body_serialized_correctly(httpx_mock: HTTPXMock, transport: Transport) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/api/v1/x",
        match_json={"name": "person", "value": 42},
        json={"ok": True},
    )
    result = transport.request("POST", "/api/v1/x", json={"name": "person", "value": 42})
    assert result == {"ok": True}


# ───────────── auth host pinning ─────────────
#
# The X-API-Key is a client-level default header, so every request the
# authenticated client makes carries it. Some request paths are SERVER-SUPPLIED
# absolute URLs (e.g. the ``annotation_url`` in a dataset download listing). A
# malicious or compromised API could point one at another host and the key would
# leave with it. The transport pins every URL to the configured base_url host.


def test_absolute_url_pinned_to_base_host(httpx_mock: HTTPXMock, transport: Transport) -> None:
    """A server-supplied foreign host is ignored; the request is pinned to base."""
    httpx_mock.add_response(json={})  # catch-all: matches any URL
    transport.request("GET", "https://evil.example/steal?x=1")
    reqs = httpx_mock.get_requests()
    assert reqs, "no request captured"
    # The foreign host was never contacted...
    assert all(r.url.host == "api.test.local" for r in reqs), (
        f"request leaked to a foreign host: {[str(r.url) for r in reqs]}"
    )
    assert not any(r.url.host == "evil.example" for r in reqs)
    # ...so the API key was never sent off-host.
    assert not any(r.url.host != "api.test.local" and r.headers.get("X-API-Key") for r in reqs)
    # The path (+ query) is preserved, resolved against base_url.
    assert reqs[-1].url.path == "/steal"
    assert reqs[-1].url.query == b"x=1"


def test_stream_bytes_absolute_url_pinned_to_base_host(
    httpx_mock: HTTPXMock, transport: Transport
) -> None:
    """The streaming path pins a foreign host too (covers stream_sse via it)."""
    httpx_mock.add_response(content=b"data")
    list(transport.stream_bytes("GET", "https://evil.example/leak"))
    reqs = httpx_mock.get_requests()
    assert reqs, "no request captured"
    assert all(r.url.host == "api.test.local" for r in reqs)
    assert not any(r.url.host == "evil.example" for r in reqs)


def test_same_host_absolute_url_unchanged(httpx_mock: HTTPXMock, transport: Transport) -> None:
    """The normal case - an ``annotation_url`` on the API host - still works,
    key attached, path preserved. This is what makes pinning transparent."""
    httpx_mock.add_response(json={"annotations": []})
    result = transport.request("GET", f"{BASE_URL}/api/v1/developer/annotations/abc/file")
    assert result == {"annotations": []}
    sent = httpx_mock.get_request()
    assert sent is not None
    assert sent.url.host == "api.test.local"
    assert sent.url.path == "/api/v1/developer/annotations/abc/file"
    assert sent.headers["X-API-Key"] == API_KEY


def test_relative_path_unaffected_by_pinning(httpx_mock: HTTPXMock, transport: Transport) -> None:
    httpx_mock.add_response(json={})
    transport.request("GET", "/api/v1/datasets")
    sent = httpx_mock.get_request()
    assert sent is not None
    assert sent.url.host == "api.test.local"
    assert sent.url.path == "/api/v1/datasets"


# ───────────── pin_url_to_base (pure helper) ─────────────


def test_pin_url_to_base_relative_path_unchanged() -> None:
    from pictograph._http.transport import pin_url_to_base

    assert pin_url_to_base("/api/v1/datasets", BASE_URL) == "/api/v1/datasets"
    assert pin_url_to_base("/a/b?c=1", BASE_URL) == "/a/b?c=1"


def test_pin_url_to_base_foreign_host_stripped_to_path() -> None:
    from pictograph._http.transport import pin_url_to_base

    assert pin_url_to_base("https://evil.example/steal?x=1", BASE_URL) == "/steal?x=1"


def test_pin_url_to_base_same_host_reduced_to_path() -> None:
    from pictograph._http.transport import pin_url_to_base

    # Same host: scheme/host dropped, path+query kept - resolves identically.
    assert (
        pin_url_to_base(f"{BASE_URL}/api/v1/annotations/abc/file", BASE_URL)
        == "/api/v1/annotations/abc/file"
    )


def test_pin_url_to_base_logs_warning_on_foreign_host(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from pictograph._http.transport import pin_url_to_base

    with caplog.at_level(logging.WARNING, logger="pictograph._http.transport"):
        pin_url_to_base("https://evil.example/x", BASE_URL)
    assert any(
        "evil.example" in rec.message or "evil.example" in str(rec.args) for rec in caplog.records
    )
