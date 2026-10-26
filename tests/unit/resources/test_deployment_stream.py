"""A deployment's live streams (``DeploymentClient.open_stream``): what the SDK sends, and what it
makes of the deployment's answers.

The stream's calls (open, overlay, state, stop) are HTTP, mocked here; its results come on a
WebSocket, which these tests serve for real (a local ``websockets`` server scripted per test:
the messages it sends, then how it closes or drops), so the client's own socket code - its
handshake, its reads, its reconnects - runs exactly as it does against a deployment.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from pictograph import (
    ClassificationResult,
    DeploymentClient,
    DeploymentStream,
    DetectionResult,
    StreamFrame,
)
from pictograph._http.retry import RetryPolicy
from pictograph.exceptions import (
    AuthError,
    ConflictError,
    NetworkError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from pictograph.resources import _deployment_stream
from pictograph.resources._deployment_stream import sessions_url, stream_source
from tests.unit.stream_server import (
    Connection,
    ResultsServer,
    box,
    frame,
    opened,
    serve_results,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_httpx import HTTPXMock

TOKEN = "pk_deploy_tok"  # noqa: S105 - test fixture, not a real secret


def _fast_retry(max_retries: int = 2) -> RetryPolicy:
    return RetryPolicy(max_retries=max_retries, sleep=lambda _s: None)


@pytest.fixture
def results() -> Iterator[ResultsServer]:
    with serve_results() as server:
        yield server


@pytest.fixture(autouse=True)
def _no_reconnect_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_deployment_stream, "_backoff", lambda _drops: 0.0)


def _open(
    httpx_mock: HTTPXMock,
    results: ResultsServer,
    *,
    task: Any = "object_detection",
    source: str = "rtsp://camera.example.com:554/live",
) -> DeploymentStream[Any]:
    httpx_mock.add_response(
        method="POST",
        url=f"{results.origin}/us-west/abc123/sessions",
        status_code=201,
        json=opened("whip" if source == "whip" else "rtsp"),
    )
    client: Any = DeploymentClient(results.endpoint, TOKEN, task=task, retry_policy=_fast_retry())
    return client.open_stream(source)  # type: ignore[no-any-return]


# ── the source and the URL ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("whip", {"kind": "whip"}),
        (" WHIP ", {"kind": "whip"}),
        ("rtsp://cam.example.com/live", {"kind": "rtsp", "uri": "rtsp://cam.example.com/live"}),
        ("rtsps://cam.example.com/live", {"kind": "rtsp", "uri": "rtsps://cam.example.com/live"}),
        ("https://cam.example.com/whep", {"kind": "whep", "uri": "https://cam.example.com/whep"}),
    ],
)
def test_a_source_is_whip_or_a_camera_the_deployment_connects_to(
    source: str, expected: dict[str, str]
) -> None:
    assert stream_source(source) == expected


@pytest.mark.parametrize(
    "source", ["http://user:secret@cam.example.com/x", "/dev/video0", "file.mp4"]
)
def test_any_other_source_is_refused_without_repeating_it(source: str) -> None:
    with pytest.raises(ValueError, match="camera URL") as exc:
        stream_source(source)
    assert "secret" not in str(exc.value)
    assert "cam.example.com" not in str(exc.value)


def test_streams_open_where_the_endpoint_predicts() -> None:
    assert sessions_url("https://workflows.pictograph.io/us-west/abc/predict") == (
        "https://workflows.pictograph.io/us-west/abc/sessions"
    )
    with pytest.raises(ValueError, match="/predict"):
        sessions_url("https://workflows.pictograph.io/us-west/abc")


# ── opening ──────────────────────────────────────────────────────────────────────────────


def test_open_sends_the_source_and_the_token_and_reads_the_stream(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    stream = _open(httpx_mock, results, source="whip")
    (request,) = httpx_mock.get_requests()
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"
    assert json.loads(request.content) == {"source": {"kind": "whip"}, "overlay": True}
    assert stream.id == "7955b01ae9212f34"
    assert stream.kind == "whip"
    assert stream.stream_limit == 4
    assert stream.overlay is True
    assert stream.expires_at is not None and stream.expires_at.hour == 15
    assert stream.ice_servers == [{"urls": "stun:stun.l.google.com:19302"}]
    base = f"{results.origin}/us-west/abc123/sessions/7955b01ae9212f34"
    assert stream.url == base
    assert stream.whep_url == f"{base}/whep"
    assert stream.whip_url == f"{base}/whip"
    assert stream.results_url == base.replace("http://", "ws://") + "/results"
    assert TOKEN not in repr(stream)


def test_a_stream_can_open_without_its_overlay(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{results.origin}/us-west/abc123/sessions",
        status_code=201,
        json={**opened(), "overlay": False},
    )
    client = DeploymentClient(results.endpoint, TOKEN, retry_policy=_fast_retry())
    stream = client.open_stream("rtsps://cam.example.com/live", overlay=False)
    assert json.loads(httpx_mock.get_requests()[0].content) == {
        "source": {"kind": "rtsp", "uri": "rtsps://cam.example.com/live"},
        "overlay": False,
    }
    assert stream.overlay is False


def test_a_camera_stream_has_no_whip_url(httpx_mock: HTTPXMock, results: ResultsServer) -> None:
    assert _open(httpx_mock, results).whip_url is None


def test_open_waits_out_a_container_that_did_not_start_then_opens(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    url = f"{results.origin}/us-west/abc123/sessions"
    httpx_mock.add_response(
        method="POST",
        url=url,
        status_code=503,
        headers={"Retry-After": "1"},
        json={"error": "deployment_starting"},
    )
    httpx_mock.add_response(method="POST", url=url, status_code=201, json=opened())
    client = DeploymentClient(results.endpoint, TOKEN, retry_policy=_fast_retry(2))
    assert client.open_stream("rtsp://cam.example.com/live").id == "7955b01ae9212f34"
    assert len(httpx_mock.get_requests()) == 2


def test_open_never_retries_a_full_deployment(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{results.origin}/us-west/abc123/sessions",
        status_code=429,
        json={
            "error": "stream_limit",
            "detail": "This deployment runs 4 stream(s) at once.",
            "stream_limit": 4,
        },
    )
    client = DeploymentClient(results.endpoint, TOKEN, retry_policy=_fast_retry(2))
    with pytest.raises(RateLimitError) as exc:
        client.open_stream("whip")
    assert exc.value.code == "stream_limit"
    assert exc.value.fix is not None and "max_containers" in exc.value.fix
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.parametrize(
    ("status", "error", "exc_cls", "fix_says"),
    [
        (409, "streams_unavailable", ConflictError, "Pause and resume"),
        (409, "deployment_not_active", ConflictError, "resume"),
        (400, "unsafe_source", ValidationError, "public"),
        (401, "unauthorized", AuthError, "pk_deploy_"),
        (502, "session_failed", ServerError, "Retry"),
    ],
)
def test_open_refusals_are_typed_with_what_to_do(
    httpx_mock: HTTPXMock,
    results: ResultsServer,
    status: int,
    error: str,
    exc_cls: type[Exception],
    fix_says: str,
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{results.origin}/us-west/abc123/sessions",
        status_code=status,
        json={"error": error},
    )
    client = DeploymentClient(results.endpoint, TOKEN, retry_policy=_fast_retry(0))
    with pytest.raises(exc_cls) as exc:
        client.open_stream("rtsp://cam.example.com/live")
    assert getattr(exc.value, "code", None) == error
    assert fix_says in (getattr(exc.value, "fix", None) or "")


# ── reading its frames ─────────────────────────────────────────────────────────────────────


def test_frames_are_typed_results_until_the_camera_stops(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    results.script = [
        Connection(
            messages=[
                {"type": "stats", "sources": [{"frames": 0}]},
                frame(0, box("car", 0.9), box("person", 0.6)),
                frame(1, box("car", 0.8)),
            ],
            code=1000,
            reason="its source ended",
        )
    ]
    stream = _open(httpx_mock, results)
    frames = list(stream.frames())
    assert [f.index for f in frames] == [0, 1]
    first = frames[0]
    assert isinstance(first, StreamFrame)
    assert isinstance(first.result, DetectionResult)
    assert [p.name for p in first.result.predictions] == ["car", "person"]
    assert first.result.backend == "tensorrt"
    assert first.result.inference_ms == 9.6
    assert (first.width, first.height) == (1280, 720)
    assert first.latency_ms.engine == 11.7
    assert first.pts == 0.0
    assert first.message["pictograph"]["annotation_count"] == 2
    assert stream.ended == "source_ended"
    assert stream.ended_detail == "its source ended"
    # The socket carried the token and the results protocol.
    (seen,) = results.seen
    assert seen["authorization"] == f"Bearer {TOKEN}"
    assert seen["subprotocol"] == "pictograph.results.v1"
    assert seen["path"] == "/us-west/abc123/sessions/7955b01ae9212f34/results"


def test_messages_give_every_message_as_sent(httpx_mock: HTTPXMock, results: ResultsServer) -> None:
    stats = {"type": "stats", "sources": [{"frames": 30}]}
    results.script = [Connection(messages=[stats, frame(0)], reason="no request for 30 s")]
    stream = _open(httpx_mock, results)
    assert [m["type"] for m in stream.messages()] == ["stats", "frame"]
    assert stream.ended == "idle"


def test_a_classifier_stream_gives_ranked_labels(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    message = frame(0, task="classification")
    message["pictograph"]["image_tags"] = ["ripe", "green"]
    message["tag_confidences"] = {"ripe": 0.91, "green": 0.07}
    results.script = [Connection(messages=[message], reason="its source ended")]
    stream = _open(httpx_mock, results, task="classification")
    (got,) = list(stream.frames())
    assert isinstance(got.result, ClassificationResult)
    assert [(c.name, c.confidence) for c in got.result.classes] == [
        ("ripe", 0.91),
        ("green", 0.07),
    ]


def test_a_dropped_connection_is_read_again(httpx_mock: HTTPXMock, results: ResultsServer) -> None:
    results.script = [
        Connection(messages=[frame(0)], drop=True),
        Connection(messages=[frame(1)], code=4502, reason="deployment_unreachable"),
        Connection(messages=[frame(2)], reason="its viewer stopped it"),
    ]
    stream = _open(httpx_mock, results)
    assert [f.index for f in stream.frames()] == [0, 1, 2]
    assert stream.ended == "stopped"
    assert len(results.seen) == 3


@pytest.mark.parametrize(
    ("reason", "ended"),
    [
        ("its viewer stopped it", "stopped"),
        ("its source ended", "source_ended"),
        ("no request for 30 s", "idle"),
        ("its 4 h limit", "time_limit"),
        ("the container is stopping", "container_stopped"),
        ("a reason this SDK does not know", "ended"),
    ],
)
def test_every_way_a_stream_ends_is_named(
    httpx_mock: HTTPXMock, results: ResultsServer, reason: str, ended: str
) -> None:
    results.script = [Connection(messages=[frame(0)], reason=reason)]
    stream = _open(httpx_mock, results)
    assert [f.index for f in stream.frames()] == [0]
    assert (stream.ended, stream.ended_detail) == (ended, reason)


def test_a_connection_that_drops_but_delivers_is_never_given_up_on(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    drops = _deployment_stream.RECONNECTS + 2
    results.script = [Connection(messages=[frame(n)], drop=True) for n in range(drops)]
    results.script.append(Connection(messages=[frame(drops)], reason="its source ended"))
    stream = _open(httpx_mock, results)
    assert [f.index for f in stream.frames()] == list(range(drops + 1))


def test_a_connection_that_keeps_dropping_fails_the_read(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    results.script = [Connection(drop=True) for _ in range(_deployment_stream.RECONNECTS + 1)]
    stream = _open(httpx_mock, results)
    with pytest.raises(NetworkError, match="did not come back"):
        list(stream.frames())
    assert len(results.seen) == _deployment_stream.RECONNECTS + 1


def test_an_engine_that_stops_is_an_error_with_its_words(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    results.script = [
        Connection(messages=[frame(0)], reason="its engine stopped: the decoder failed")
    ]
    stream = _open(httpx_mock, results)
    read = stream.frames()
    assert next(read).index == 0
    with pytest.raises(ServerError) as exc:
        next(read)
    assert exc.value.code == "session_failed"
    assert "the decoder failed" in str(exc.value)
    assert stream.ended == "failed"


def test_a_stream_that_is_gone_just_ends(httpx_mock: HTTPXMock, results: ResultsServer) -> None:
    results.script = [Connection(code=4410, reason="session_ended")]
    stream = _open(httpx_mock, results)
    assert list(stream.frames()) == []
    assert stream.ended == "ended"


def test_a_refused_read_is_typed(httpx_mock: HTTPXMock, results: ResultsServer) -> None:
    results.script = [Connection(code=4401, reason="unauthorized")]
    stream = _open(httpx_mock, results)
    with pytest.raises(AuthError) as exc:
        list(stream.frames())
    assert exc.value.code == "unauthorized"


def test_a_frame_of_another_task_than_declared_is_refused(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    results.script = [Connection(messages=[frame(0)], reason="its source ended")]
    stream = _open(httpx_mock, results, task="classification")
    with pytest.raises(ValidationError, match="object_detection"):
        list(stream.frames())


# ── changing it, and stopping it ──────────────────────────────────────────────────────────


def test_the_overlay_switches(httpx_mock: HTTPXMock, results: ResultsServer) -> None:
    stream = _open(httpx_mock, results)
    httpx_mock.add_response(method="POST", url=f"{stream.url}/overlay", json={"overlay": False})
    assert stream.set_overlay(False) is False
    assert stream.overlay is False
    assert json.loads(httpx_mock.get_requests()[-1].content) == {"on": False}


def test_state_is_the_deployments_account(httpx_mock: HTTPXMock, results: ResultsServer) -> None:
    stream = _open(httpx_mock, results)
    httpx_mock.add_response(
        method="GET",
        url=stream.url,
        json={
            "session": stream.id,
            "kind": "rtsp",
            "opened_at": "2026-10-26T11:44:44Z",
            "expires_at": "2026-10-26T15:44:44Z",
            "engine": {"status": "running", "overlay": True, "whep": [{"viewers": 1}]},
        },
    )
    state = stream.state()
    assert state.session == stream.id
    assert state.engine["whep"] == [{"viewers": 1}]


def test_stop_stops_once(httpx_mock: HTTPXMock, results: ResultsServer) -> None:
    stream = _open(httpx_mock, results)
    httpx_mock.add_response(
        method="DELETE", url=stream.url, json={"session": stream.id, "stopped": True}
    )
    stream.stop()
    stream.stop()
    assert [r.method for r in httpx_mock.get_requests()] == ["POST", "DELETE"]
    assert stream.ended == "stopped"
    # A stopped stream reads nothing more.
    assert list(stream.frames()) == []


@pytest.mark.parametrize("status", [404, 410])
def test_stopping_a_stream_that_already_ended_is_quiet(
    httpx_mock: HTTPXMock, results: ResultsServer, status: int
) -> None:
    stream = _open(httpx_mock, results)
    httpx_mock.add_response(
        method="DELETE", url=stream.url, status_code=status, json={"error": "session_ended"}
    )
    stream.stop()
    assert stream.ended == "stopped"


def test_leaving_the_with_block_stops_it(httpx_mock: HTTPXMock, results: ResultsServer) -> None:
    stream = _open(httpx_mock, results)
    httpx_mock.add_response(method="DELETE", url=stream.url, json={"stopped": True})
    with stream:
        pass
    assert httpx_mock.get_requests()[-1].method == "DELETE"


def test_a_failed_stop_never_hides_the_error_that_left_the_block(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    stream = _open(httpx_mock, results)
    httpx_mock.add_response(
        method="DELETE", url=stream.url, status_code=502, json={"error": "deployment_unreachable"}
    )
    with pytest.raises(KeyError), stream:
        raise KeyError("the caller's own error")


def test_a_failed_stop_raises_when_nothing_else_did(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    stream = _open(httpx_mock, results)
    httpx_mock.add_response(
        method="DELETE", url=stream.url, status_code=502, json={"error": "deployment_unreachable"}
    )
    with pytest.raises(ServerError), stream:
        pass
