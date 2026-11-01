"""A deployment's live streams (``DeploymentClient.open_stream``): what the SDK sends, and what it
makes of the deployment's answers.

The stream's calls (open, overlay, state, stop) are HTTP, mocked here; its results come on a
WebSocket, which these tests serve for real (a local ``websockets`` server scripted per test:
the messages it sends, then how it closes or drops), so the client's own socket code - its
handshake, its reads, its reconnects - runs exactly as it does against a deployment.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from pictograph import (
    ClassificationResult,
    DeploymentClient,
    DeploymentStream,
    DetectionResult,
    PictographError,
    StreamFrame,
    StreamStalledError,
)
from pictograph._http.retry import RetryPolicy
from pictograph.exceptions import (
    ApiError,
    AuthError,
    ConflictError,
    NetworkError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from pictograph.resources import _deployment_stream
from pictograph.resources._deployment_stream import predict_url, sessions_url, stream_source
from tests.unit.stream_server import (
    SESSION,
    SESSION_PATH,
    Connection,
    ResultsServer,
    box,
    frame,
    joined_state,
    opened,
    serve_results,
    stats_message,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_httpx import HTTPXMock

TOKEN = "pk_deploy_tok"  # noqa: S105 - test fixture, not a real secret


def _drain(items: Iterator[Any], into: list[Any]) -> None:
    """Read ``items`` to their end (or an error), keeping each in ``into``."""
    into.extend(items)


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


def test_an_rtsp_camera_takes_its_transport_and_buffer_mode() -> None:
    # B579 slice 2b: a recording streamed over a TCP-only tunnel, timed by its sender's clock.
    assert stream_source("rtsp://cam.example.com/rec", transport="tcp", buffer_mode="none") == {
        "kind": "rtsp",
        "uri": "rtsp://cam.example.com/rec",
        "transport": "tcp",
        "buffer_mode": "none",
    }
    assert stream_source("rtsps://cam.example.com/live", transport="udp") == {
        "kind": "rtsp",
        "uri": "rtsps://cam.example.com/live",
        "transport": "udp",
    }
    # B579 (b): a recording never reconnected - its end ends the stream.
    assert stream_source("rtsp://cam.example.com/rec", reconnect="never") == {
        "kind": "rtsp",
        "uri": "rtsp://cam.example.com/rec",
        "reconnect": "never",
    }


@pytest.mark.parametrize(
    ("source", "options", "match"),
    [
        ("whip", {"transport": "tcp"}, "RTSP camera's options"),
        ("https://cam.example.com/whep", {"buffer_mode": "none"}, "RTSP camera's options"),
        (
            "rtsp://cam.example.com/live",
            {"transport": "quic"},
            "transport is one of auto, tcp, udp",
        ),
        (
            "rtsp://cam.example.com/live",
            {"buffer_mode": "slaved"},
            "buffer_mode is one of auto, none",
        ),
        ("whip", {"reconnect": "never"}, "RTSP camera's options"),
        (
            "rtsp://cam.example.com/live",
            {"reconnect": "sometimes"},
            "reconnect is one of always, never",
        ),
    ],
)
def test_rtsp_options_are_an_rtsp_cameras_and_its_values_only(
    source: str, options: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        stream_source(source, **options)


def test_streams_open_on_the_endpoints_sessions() -> None:
    """A deployment's endpoint_url ends in /predict, a workflow's in /sessions (B579: it takes streams, never a
    predict): both open their streams on the endpoint's sessions, and a predict goes to its /predict."""
    for given in (
        "https://workflows.pictograph.io/us-west/abc/predict",
        "https://workflows.pictograph.io/us-west/abc/sessions",
        "https://workflows.pictograph.io/us-west/abc/sessions/",
    ):
        assert sessions_url(given) == "https://workflows.pictograph.io/us-west/abc/sessions"
        assert predict_url(given) == "https://workflows.pictograph.io/us-west/abc/predict"
    for wrong in (
        "https://workflows.pictograph.io/us-west/abc",
        "https://workflows.pictograph.io/us-west/abc/whep",
    ):
        with pytest.raises(ValueError, match=r"/predict .* or /sessions"):
            sessions_url(wrong)


# ── opening ──────────────────────────────────────────────────────────────────────────────


def test_open_sends_the_source_and_the_token_and_reads_the_stream(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    stream = _open(httpx_mock, results, source="whip")
    (request,) = httpx_mock.get_requests()
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"
    assert json.loads(request.content) == {
        "source": {"kind": "whip"},
        "overlay": True,
        "stream": True,
    }
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


def test_a_stream_says_whether_it_runs_outside_its_region(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    """B597: its deployment's region had no GPU free for 30 s, so it started elsewhere in the US."""
    httpx_mock.add_response(
        method="POST",
        url=f"{results.origin}/us-west/abc123/sessions",
        status_code=201,
        json=opened("rtsp", fallback=True),
    )
    client: Any = DeploymentClient(results.endpoint, TOKEN, retry_policy=_fast_retry())
    assert client.open_stream("rtsp://camera.example.com:554/live").fallback is True
    assert _open(httpx_mock, results).fallback is False, (
        "an answer from before B597 runs in its region"
    )


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
        "stream": True,
    }
    assert stream.overlay is False


def test_a_stream_can_open_with_its_live_view_off(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    """B607: a loop that reads only results opens its stream with no picture sent - and the answer says so."""
    httpx_mock.add_response(
        method="POST",
        url=f"{results.origin}/us-west/abc123/sessions",
        status_code=201,
        json={**opened(), "stream": False},
    )
    client = DeploymentClient(results.endpoint, TOKEN, retry_policy=_fast_retry())
    stream = client.open_stream("rtsps://cam.example.com/live", stream=False)
    assert json.loads(httpx_mock.get_requests()[0].content) == {
        "source": {"kind": "rtsp", "uri": "rtsps://cam.example.com/live"},
        "overlay": True,
        "stream": False,
    }
    assert stream.stream is False
    assert stream.overlay is True


def test_an_answer_from_before_the_stream_switch_streams(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    before = opened()
    del before["stream"]
    del before["paths"]["stream"]
    httpx_mock.add_response(
        method="POST", url=f"{results.origin}/us-west/abc123/sessions", status_code=201, json=before
    )
    client = DeploymentClient(results.endpoint, TOKEN, retry_policy=_fast_retry())
    stream = client.open_stream("rtsps://cam.example.com/live")
    assert stream.stream is True
    with pytest.raises(ConflictError, match="no stream switch") as e:
        stream.set_stream(False)
    assert e.value.code == "stream_switch_unavailable"
    assert len(httpx_mock.get_requests()) == 1, "refused before anything is sent"


def test_a_workflow_streams_outputs_go_with_its_open_and_come_back_resolved(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    """B607: a workflow endpoint's stream is given its outputs when it opens (a run's are its own) - checked before
    the request - and its endpoint answers them resolved (each folder by its dataset's id, with the folder it writes)."""
    resolved = {
        "passive_capture": {
            "dataset": "p-1",
            "dataset_name": "Street scenes",
            "directory": "/bg",
            "new_folder_each_run": False,
            "folder": "/bg",
            "every_s": 10.0,
        }
    }
    httpx_mock.add_response(
        method="POST",
        url=f"{results.origin}/us-west/abc123/sessions",
        status_code=201,
        json={**opened(), "outputs": resolved},
    )
    client = DeploymentClient(results.endpoint, TOKEN, retry_policy=_fast_retry())
    stream = client.open_stream(
        "rtsp://camera.example.com:554/live",
        outputs={"passive_capture": {"dataset": "Street scenes", "directory": "/bg"}},
    )
    sent = json.loads(httpx_mock.get_requests()[0].content)
    assert sent["outputs"] == {
        "annotated_video": False,
        "passive_capture": {
            "dataset": "Street scenes",
            "directory": "/bg",
            "new_folder_each_run": False,
            "every_s": 10.0,
        },
    }
    assert stream.outputs is not None and stream.outputs.passive_capture is not None
    assert stream.outputs.passive_capture.dataset_name == "Street scenes"
    # Mistyped outputs never reach the endpoint.
    with pytest.raises(ValueError):
        client.open_stream("rtsp://camera.example.com:554/live", outputs={"passive": {}})
    assert len(httpx_mock.get_requests()) == 1
    # A stream opened with none answers none.
    assert _open(httpx_mock, results).outputs is None


def test_a_stream_sends_its_rtsp_cameras_options(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=f"{results.origin}/us-west/abc123/sessions",
        status_code=201,
        json=opened(),
    )
    client = DeploymentClient(results.endpoint, TOKEN, retry_policy=_fast_retry())
    client.open_stream(
        "rtsp://cam.example.com/rec", transport="tcp", buffer_mode="none", reconnect="never"
    )
    assert json.loads(httpx_mock.get_requests()[0].content) == {
        "source": {
            "kind": "rtsp",
            "uri": "rtsp://cam.example.com/rec",
            "transport": "tcp",
            "buffer_mode": "none",
            "reconnect": "never",
        },
        "overlay": True,
        "stream": True,
    }


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


def test_open_waits_out_a_workflow_endpoint_being_updated(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    """A workflow's endpoint deployed again after a save (B579) answers 409 ``endpoint_updating`` with a Retry-After:
    the open is tried again after it, as for a stream container starting."""
    url = f"{results.origin}/us-west/abc123/sessions"
    httpx_mock.add_response(
        method="POST",
        url=url,
        status_code=409,
        headers={"Retry-After": "1"},
        json={"error": "endpoint_updating"},
    )
    httpx_mock.add_response(method="POST", url=url, status_code=201, json=opened())
    client = DeploymentClient(results.endpoint, TOKEN, retry_policy=_fast_retry(2))
    assert client.open_stream("whip").id == "7955b01ae9212f34"
    assert len(httpx_mock.get_requests()) == 2


def test_open_never_retries_a_conflict_it_cannot_wait_out(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    from pictograph.exceptions import ConflictError

    httpx_mock.add_response(
        method="POST",
        url=f"{results.origin}/us-west/abc123/sessions",
        status_code=409,
        json={"error": "deployment_not_active", "status": "paused"},
    )
    client = DeploymentClient(results.endpoint, TOKEN, retry_policy=_fast_retry(2))
    with pytest.raises(ConflictError):
        client.open_stream("whip")
    assert len(httpx_mock.get_requests()) == 1


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


def test_a_frame_carries_when_it_was_captured_and_how_long_it_took_to_reach_you(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    captured = time.time_ns() - 40_000_000  # 40 ms ago, by this machine's clock (it sent the video)
    timed = {**frame(0), "capture_ns": captured}
    results.script = [Connection(messages=[timed, frame(1)], reason="its source ended")]
    stream = _open(httpx_mock, results, source="whip")
    first, second = list(stream.frames())
    assert first.capture_ns == captured
    expected = datetime.fromtimestamp(captured // 1_000_000_000, tz=timezone.utc).replace(
        microsecond=(captured % 1_000_000_000) // 1000
    )
    assert first.capture_time == expected
    assert first.capture_time is not None and first.capture_time.tzinfo is timezone.utc
    assert first.capture_latency_ms is not None
    assert 40.0 <= first.capture_latency_ms < 5000.0
    assert first.capture_latency_ms == (first.received_ns - captured) / 1e6
    # Before its sender's first report a frame has no capture time - never a guess.
    assert (second.capture_ns, second.capture_time, second.capture_latency_ms) == (None, None, None)


def test_a_capture_time_of_zero_is_none() -> None:
    message = {**frame(0), "capture_ns": 0}
    result: Any = _deployment_stream.frame_result(message, "object_detection")
    zero = StreamFrame(message, result)
    assert (zero.capture_ns, zero.capture_time, zero.capture_latency_ms) == (None, None, None)


# ── a stall ───────────────────────────────────────────────────────────────────────────────


def test_a_stall_says_how_long_its_source_has_been_quiet_by_the_deployments_own_measure(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    # Its frames, then the engine's heartbeat: alive, its source quiet for 4 s and more.
    results.script = [
        Connection(
            messages=[frame(0), frame(1), stats_message(quiet=4.0, heartbeat=True), 3.0],
            reason="its source ended",
        )
    ]
    stream = _open(httpx_mock, results)
    seen: list[StreamFrame[Any]] = []
    started = time.monotonic()
    with pytest.raises(StreamStalledError) as raised:
        _drain(stream.frames(idle_timeout=0.5), seen)
    assert 0.5 <= time.monotonic() - started < 2.0, "raised by the reader's own window"
    assert [f.index for f in seen] == [0, 1]
    assert 4.4 <= raised.value.quiet_s < 6.0, "the source's quiet, as the deployment measured it"
    assert raised.value.alive is True
    assert "the stream is up and its source sends nothing" in raised.value.message
    assert stream.ended is None, "the stream goes on"


def test_a_quiet_below_the_timeout_is_read_through(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    results.script = [
        Connection(
            messages=[
                frame(0),
                stats_message(quiet=1.2, heartbeat=True),
                stats_message(quiet=0.03),
                frame(1),
            ],
            reason="its source ended",
        )
    ]
    stream = _open(httpx_mock, results)
    assert [f.index for f in stream.frames(idle_timeout=2.0)] == [0, 1]
    assert stream.ended == "source_ended"


def test_a_stream_has_a_frame_while_any_of_its_sources_sends(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    # Two sources: one quiet for 5 s, the other sent a frame a moment ago - the stream's quiet is the latter's.
    two = stats_message(quiet=5.0)
    two["sources"].append({"id": "other", "last_frame_s": 0.04})
    results.script = [Connection(messages=[frame(0), two, 3.0], reason="its source ended")]
    stream = _open(httpx_mock, results)
    with pytest.raises(StreamStalledError) as raised:
        for _ in stream.frames(idle_timeout=0.5):
            pass
    assert raised.value.quiet_s < 2.0


def test_a_frame_ends_the_quiet_the_deployment_reported(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    # A heartbeat mid-stall (30 s), then its frames come back, then a new silence: the new stall is the new one's.
    results.script = [
        Connection(
            messages=[stats_message(quiet=30.0, heartbeat=True), frame(0), 3.0],
            reason="its source ended",
        )
    ]
    stream = _open(httpx_mock, results)
    with pytest.raises(StreamStalledError) as raised:
        for _ in stream.frames(idle_timeout=0.5):
            pass
    assert raised.value.quiet_s < 2.0


def test_a_silent_connection_raises_by_the_readers_own_clock(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    # A frame, then nothing at all - no frame, no stats - with the connection still open.
    results.script = [Connection(messages=[frame(0), 3.0], reason="its source ended")]
    stream = _open(httpx_mock, results)
    started = time.monotonic()
    with pytest.raises(StreamStalledError) as raised:
        for _ in stream.frames(idle_timeout=0.6):
            pass
    waited = time.monotonic() - started
    assert 0.6 <= waited < 2.0
    assert raised.value.alive is False
    assert 0.6 <= raised.value.quiet_s < 2.0
    assert "its connection may be down" in raised.value.message


def test_stats_with_no_frames_raise_as_a_quiet_source(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    # The deployment talks (its numbers each second, here without a last_frame_s) but sends no frame.
    talk: list[dict[str, Any] | float] = []
    for _ in range(8):
        talk += [0.2, stats_message()]
    results.script = [Connection(messages=[frame(0), *talk], reason="its source ended")]
    stream = _open(httpx_mock, results)
    with pytest.raises(StreamStalledError) as raised:
        for _ in stream.frames(idle_timeout=0.7):
            pass
    assert raised.value.alive is True
    assert 0.7 <= raised.value.quiet_s < 1.6


def test_a_stream_that_never_sends_a_frame_stalls_from_when_its_reading_began(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    results.script = [
        Connection(messages=[stats_message(heartbeat=True), 3.0], reason="its source ended")
    ]
    stream = _open(httpx_mock, results)
    started = time.monotonic()
    with pytest.raises(StreamStalledError) as raised:
        for _ in stream.frames(idle_timeout=0.5):
            pass
    assert 0.5 <= time.monotonic() - started < 2.0
    assert raised.value.quiet_s >= 0.5


def test_the_time_spent_on_a_frame_is_never_the_streams_silence(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    results.script = [
        Connection(
            messages=[frame(0), stats_message(quiet=0.03), frame(1)], reason="its source ended"
        )
    ]
    stream = _open(httpx_mock, results)
    seen = []
    for f in stream.frames(idle_timeout=0.4):
        seen.append(f.index)
        time.sleep(0.6)  # a robot acting on the frame, longer than the timeout
    assert seen == [0, 1]


def test_reading_again_after_a_stall_waits_another_timeout_for_the_frames(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    # The first read stalls; the second begins mid-stall (the heartbeat says 30 s) - it waits its own window,
    # never raising at once, and reads the frames when they come back.
    results.script = [
        Connection(messages=[frame(0), stats_message(quiet=3.0, heartbeat=True), 2.0]),
        Connection(
            messages=[stats_message(quiet=30.0, heartbeat=True), 0.3, frame(7), frame(8)],
            reason="its source ended",
        ),
    ]
    stream = _open(httpx_mock, results)
    with pytest.raises(StreamStalledError):
        for _ in stream.frames(idle_timeout=0.5):
            pass
    assert [f.index for f in stream.frames(idle_timeout=0.5)] == [7, 8]
    assert len(results.seen) == 2


def test_messages_take_an_idle_timeout_too(httpx_mock: HTTPXMock, results: ResultsServer) -> None:
    results.script = [
        Connection(
            messages=[frame(0), stats_message(quiet=5.0, heartbeat=True), 2.0],
            reason="its source ended",
        )
    ]
    stream = _open(httpx_mock, results)
    got: list[dict[str, Any]] = []
    with pytest.raises(StreamStalledError) as raised:
        _drain(stream.messages(idle_timeout=0.5), got)
    assert [m["type"] for m in got] == ["frame", "stats"]
    assert raised.value.quiet_s >= 5.0


def test_without_an_idle_timeout_a_stall_is_read_like_any_message(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    heartbeat = stats_message(quiet=9.0, heartbeat=True)
    results.script = [
        Connection(messages=[frame(0), heartbeat, frame(1)], reason="its source ended")
    ]
    stream = _open(httpx_mock, results)
    assert [m.get("heartbeat", False) for m in stream.messages()] == [False, True, False]


@pytest.mark.parametrize("bad", [0, -1.5])
def test_an_idle_timeout_is_seconds_above_zero(
    bad: float, httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    results.script = [Connection(messages=[frame(0)], reason="its source ended")]
    stream = _open(httpx_mock, results)
    with pytest.raises(ValueError, match="idle_timeout"):
        next(stream.frames(idle_timeout=bad))
    with pytest.raises(ValueError, match="idle_timeout"):
        next(stream.messages(idle_timeout=bad))


def test_a_stall_is_a_pictograph_error_that_says_what_to_do() -> None:
    stall = StreamStalledError("No frame for 3.0 s.", quiet_s=3.0, alive=True)
    assert isinstance(stall, PictographError)
    assert stall.fix is not None and "frames() again" in stall.fix
    assert stall.docs_url == "https://pictograph.io/docs/streams"
    assert "Fix:" in str(stall)


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


def test_the_stream_switches(httpx_mock: HTTPXMock, results: ResultsServer) -> None:
    """B607: the live view's pictures off and on again while its results go on."""
    stream = _open(httpx_mock, results)
    httpx_mock.add_response(
        method="POST", url=f"{stream.url}/stream", json={"overlay": True, "stream": False}
    )
    assert stream.set_stream(False) is False
    assert stream.stream is False
    assert json.loads(httpx_mock.get_requests()[-1].content) == {"on": False}
    httpx_mock.add_response(
        method="POST", url=f"{stream.url}/stream", json={"overlay": True, "stream": True}
    )
    assert stream.set_stream(True) is True
    assert stream.stream is True
    assert json.loads(httpx_mock.get_requests()[-1].content) == {"on": True}


def test_a_stream_switch_the_endpoint_refuses_says_why(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    stream = _open(httpx_mock, results)
    httpx_mock.add_response(
        method="POST",
        url=f"{stream.url}/stream",
        status_code=409,
        json={
            "error": "this output records or feeds more than its WebRTC viewers: its pictures are always sent"
        },
    )
    with pytest.raises(ConflictError):
        stream.set_stream(False)
    assert stream.stream is True, "refused: nothing changed"


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


# ── joining a stream: a workflow's file run's live view (B607 slice C) ─────────────────────


def _join(
    httpx_mock: HTTPXMock, results: ResultsServer, *, state: dict[str, Any] | None = None
) -> DeploymentStream[Any]:
    httpx_mock.add_response(
        method="GET", url=f"{results.sessions}/{SESSION}?ice=1", json=state or joined_state()
    )
    client: Any = DeploymentClient(
        results.endpoint, TOKEN, task="object_detection", retry_policy=_fast_retry()
    )
    return client.join_stream(SESSION)  # type: ignore[no-any-return]


def test_a_runs_live_view_is_joined_by_its_id(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    """B607 slice C (Noah: "if ON then the stream is visible to the user in the run modal ... it can be turned on mid
    run"): a file run's live view is a stream joined, never opened - its state read with the ICE servers a viewer joins
    with, its two switches as its engine has them now, its frames read as a stream's."""
    results.script = [Connection(messages=[frame(0), frame(1)], reason="the run completed")]
    live = _join(httpx_mock, results)
    read = httpx_mock.get_requests()[0]
    assert (read.method, read.url.params.get("ice")) == ("GET", "1")
    assert read.headers["authorization"] == f"Bearer {TOKEN}"
    assert (live.id, live.kind, live.joined) == (SESSION, "run", True)
    assert (live.stream, live.overlay) == (False, True)
    assert live.whip_url is None
    assert live.whep_url == f"{results.origin}{SESSION_PATH}/whep"
    assert live.ice_servers == [{"urls": "stun:stun.l.google.com:19302"}]
    assert [f.index for f in live.frames()] == [0, 1]
    assert live.ended == "run_completed"


@pytest.mark.parametrize(
    ("reason", "ended"),
    [
        ("the run completed", "run_completed"),
        ("the run failed", "run_failed"),
        ("the run cancelled", "run_cancelled"),
    ],
)
def test_a_runs_live_view_ends_with_its_run(
    httpx_mock: HTTPXMock, results: ResultsServer, reason: str, ended: str
) -> None:
    """Its run's end closes its results with the run's words: the view ends, never an error (a failed run's failure is
    its run's, read on its row)."""
    results.script = [Connection(messages=[frame(0)], reason=reason)]
    live = _join(httpx_mock, results)
    assert [f.index for f in live.frames()] == [0]
    assert (live.ended, live.ended_detail) == (ended, reason)


def test_a_join_waits_for_its_runs_engine(httpx_mock: HTTPXMock, results: ResultsServer) -> None:
    """A run's engine starts with its first frames: while it does, the state carries nothing of it, and the join asks
    again - its switches and paths are the engine's own once it answers."""
    url = f"{results.sessions}/{SESSION}?ice=1"
    httpx_mock.add_response(method="GET", url=url, json=joined_state(engine={}))
    httpx_mock.add_response(method="GET", url=url, json=joined_state())
    slept: list[float] = []
    with httpx.Client() as http:
        live = _deployment_stream.join_stream(
            http,
            endpoint=results.endpoint,
            headers={},
            task=None,
            session=SESSION,
            sleep=slept.append,
        )
    assert len(httpx_mock.get_requests()) == 2
    assert slept == [_deployment_stream._JOIN_POLL_S]
    assert live.stream is False


def test_a_join_whose_engine_never_answers_says_so(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    httpx_mock.add_response(
        method="GET", url=f"{results.sessions}/{SESSION}?ice=1", json=joined_state(engine={})
    )
    with httpx.Client() as http, pytest.raises(ServerError) as exc:
        _deployment_stream.join_stream(
            http, endpoint=results.endpoint, headers={}, task=None, session=SESSION, wait_s=0.0
        )
    assert exc.value.code == "engine_starting"


@pytest.mark.parametrize(
    "session", ["", "0123", "0123456789ABCDEF", "../../etc/passwd", f"{SESSION}0"]
)
def test_a_join_refuses_what_is_no_streams_id_before_any_request(
    httpx_mock: HTTPXMock, results: ResultsServer, session: str
) -> None:
    client: Any = DeploymentClient(results.endpoint, TOKEN)
    with pytest.raises(ValueError, match="not a stream's id"):
        client.join_stream(session)
    assert httpx_mock.get_requests() == []


def test_joining_a_run_that_ended_is_typed(httpx_mock: HTTPXMock, results: ResultsServer) -> None:
    httpx_mock.add_response(
        method="GET",
        url=f"{results.sessions}/{SESSION}?ice=1",
        status_code=410,
        json={"error": "session_ended", "message": "the run ended: completed"},
    )
    client: Any = DeploymentClient(results.endpoint, TOKEN)
    with pytest.raises(ApiError) as exc:
        client.join_stream(SESSION)
    assert (exc.value.status_code, exc.value.code) == (410, "session_ended")


def test_stopping_a_runs_live_view_only_stops_watching(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    """A run's live view ends with its run (the gateway refuses its DELETE: run_live_view): stop() - and leaving a with
    block - sends nothing, and only stops watching."""
    live = _join(httpx_mock, results)
    live.stop()
    live.stop()
    with live:
        pass
    assert [r.method for r in httpx_mock.get_requests()] == ["GET"]
    assert live.ended == "stopped"
    assert list(live.frames()) == []


def test_a_runs_live_view_switches_on_its_engine(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    live = _join(httpx_mock, results)
    httpx_mock.add_response(
        method="POST", url=f"{live.url}/stream", json={"overlay": True, "stream": True}
    )
    assert live.set_stream(True) is True
    httpx_mock.add_response(method="POST", url=f"{live.url}/overlay", json={"overlay": False})
    assert live.set_overlay(False) is False
    assert [json.loads(r.content) for r in httpx_mock.get_requests()[1:]] == [
        {"on": True},
        {"on": False},
    ]


def test_a_stream_closes_the_connection_it_owns_once_it_stops(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    """watch_run's stream owns the connection it opened: its stop closes it, once."""
    live = _join(httpx_mock, results)
    closed: list[bool] = []
    live._owns(lambda: closed.append(True))
    live.stop()
    live.stop()
    assert closed == [True]


def test_a_failed_stop_keeps_the_connection_it_owns_open(
    httpx_mock: HTTPXMock, results: ResultsServer
) -> None:
    """An opened stream whose stop failed is not stopped: what it owns stays open, for the stop tried again."""
    stream = _open(httpx_mock, results)
    closed: list[bool] = []
    stream._owns(lambda: closed.append(True))
    httpx_mock.add_response(
        method="DELETE", url=stream.url, status_code=502, json={"error": "deployment_unreachable"}
    )
    with pytest.raises(ServerError):
        stream.stop()
    assert closed == []
    httpx_mock.add_response(method="DELETE", url=stream.url, json={"stopped": True})
    stream.stop()
    assert closed == [True]
