"""A deployment's live streams: video goes in, each frame's result comes out as it is inferred.

Open one on a :class:`~pictograph.DeploymentClient` and read its frames::

    from pictograph import DeploymentClient

    deployment = DeploymentClient(
        endpoint="https://workflows.pictograph.io/us-west/<slug>/predict",  # its endpoint_url
        api_key="pk_deploy_...",
        task="object_detection",
    )
    with deployment.open_stream("rtsp://camera.example.com:554/live") as stream:
        for frame in stream.frames():
            print(frame.index, [p.name for p in frame.result.predictions])

``open_stream`` takes where the video comes from:

* a camera's URL - ``rtsp://``, ``rtsps://``, or a WebRTC camera's ``https://`` WHEP URL -
  which the deployment connects to (its address must be public); or
* ``"whip"``: you send the video yourself, H.264 from any WHIP client (OBS, GStreamer's
  ``whipclientsink``), to :attr:`DeploymentStream.whip_url` with the same token.

Each frame's ``result`` is the same typed result :meth:`DeploymentClient.infer` returns, and its
``capture_time`` when it was captured, by its sender's clock (``capture_latency_ms``: capture to
result on yours). To watch the video with the results drawn on it, play
:attr:`DeploymentStream.whep_url` in any WHEP player, with the same token.

A camera that stalls never leaves a reader waiting unawares: ``frames(idle_timeout=2)`` raises
:class:`~pictograph.StreamStalledError` after two seconds without a frame (the deployment sends its
numbers each second, saying how long its source has been quiet, so a stall is told from a dead
connection). The stream goes on: read again to wait for its frames, or stop it.

A stream stops when you call :meth:`DeploymentStream.stop` or leave the ``with`` block, when
its camera stops sending, after 30 seconds with nothing reading it or sending to it, and four
hours after it opened. It is billed as the deployment's uptime while it runs.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Final, Generic, Literal, TypeVar, cast
from urllib.parse import urlsplit

import httpx

from pictograph.exceptions import (
    ConflictError,
    NetworkError,
    PictographError,
    RequestTimeoutError,
    ServerError,
    StreamStalledError,
    ValidationError,
    from_response,
)
from pictograph.inference.results import (
    TASK_RESULT_TYPES,
    AnyResult,
    InferenceResult,
    TaskName,
    build_result,
)
from pictograph.models.deployment import OpenedStream, StreamLatency, StreamState
from pictograph.resources._gateway import UNREPORTED_DEVICE, parse_response, with_gateway_hint
from pictograph.resources._outputs import RunOutputs, outputs_body

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from types import TracebackType

    from websockets.sync.client import ClientConnection

    from pictograph._http.retry import RetryPolicy
    from pictograph.models.workflow import WorkflowRunWrites

R_co = TypeVar("R_co", bound=InferenceResult, covariant=True)

#: The results socket's subprotocol (LOCKSTEP: the inference gateway's ``RESULTS_PROTOCOL``).
RESULTS_PROTOCOL: Final = "pictograph.results.v1"
#: How long opening a stream may take: the gateway holds the open while a stream container
#: starts (up to two minutes) and while its engine loads the model (up to three).
OPEN_TIMEOUT_S: Final = 330.0
#: Any other call on a stream: its state, its overlay, its stop.
CALL_TIMEOUT_S: Final = 35.0
#: The largest message the results socket takes (as large as the gateway relays).
MAX_MESSAGE_BYTES: Final = 16 << 20
#: How many times in a row the results socket reconnects after its connection drops.
RECONNECTS: Final = 3
#: The longest a reader given ``idle_timeout`` waits on a silent socket before it looks at its clock.
_STALL_POLL_S: Final = 0.25

#: The refusals of an open that made no stream, so it is tried again (after their Retry-After): a
#: workflow's endpoint being deployed again after a save (``endpoint_updating``, a 409) among them.
_RETRY_OPEN: Final = frozenset(
    {"deployment_starting", "sessions_unavailable", "gateway_unavailable", "endpoint_updating"}
)
#: How the deployment says a stream's engine failed (a normal close carries the reason).
_ENGINE_STOPPED: Final = "its engine stopped"
#: Why a stream ended, from the deployment's words for it, as :attr:`DeploymentStream.ended`.
_ENDED: Final[dict[str, str]] = {
    "its viewer stopped it": "stopped",
    "its source ended": "source_ended",
    "the container is stopping": "container_stopped",
}


#: An RTSP camera's own options, as the deployment takes them (its engine's words).
RtspTransport = Literal["auto", "tcp", "udp"]
RtspBufferMode = Literal["auto", "none"]
RtspReconnect = Literal["always", "never"]
_RTSP_OPTIONS: Final[dict[str, tuple[str, ...]]] = {
    "transport": ("auto", "tcp", "udp"),
    "buffer_mode": ("auto", "none"),
    "reconnect": ("always", "never"),
}


def stream_source(
    source: str,
    *,
    transport: RtspTransport | None = None,
    buffer_mode: RtspBufferMode | None = None,
    reconnect: RtspReconnect | None = None,
) -> dict[str, str]:
    """The stream's source as the deployment takes it: ``"whip"``, or a camera it connects to -
    an RTSP camera with its ``transport``, ``buffer_mode`` and ``reconnect`` when given."""
    options = {
        key: value
        for key, value in (
            ("transport", transport),
            ("buffer_mode", buffer_mode),
            ("reconnect", reconnect),
        )
        if value is not None
    }
    for key, value in options.items():
        if value not in _RTSP_OPTIONS[key]:
            raise ValueError(f"{key} is one of {', '.join(_RTSP_OPTIONS[key])}; got {value!r}.")
    text = source.strip()
    scheme = urlsplit(text).scheme.lower()
    if options and scheme not in ("rtsp", "rtsps"):
        raise ValueError(
            f"{' and '.join(options)} are an RTSP camera's options (rtsp:// or rtsps://)."
        )
    if text.lower() == "whip":
        return {"kind": "whip"}
    if scheme in ("rtsp", "rtsps"):
        return {"kind": "rtsp", "uri": text, **options}
    if scheme == "https":
        return {"kind": "whep", "uri": text}
    # Never repeat the source: a camera's URL can carry its password.
    got = f"a {scheme}:// URL" if scheme else "neither"
    raise ValueError(
        "A stream's source is a camera URL (rtsp://, rtsps://, or a WebRTC camera's https:// "
        f'WHEP URL), or "whip" to send the video yourself; got {got}.'
    )


def _endpoint_root(endpoint: str) -> str:
    """An endpoint URL without its last path segment: a model deployment's ``endpoint_url`` ends in ``/predict``, a
    workflow's in ``/sessions`` (it takes streams, never a predict). Anything else is refused, naming both."""
    base = endpoint.rstrip("/")
    for tail in ("/predict", "/sessions"):
        if base.endswith(tail):
            return base[: -len(tail)]
    raise ValueError(
        "An endpoint URL ends in /predict (a deployment's endpoint_url) or /sessions (a workflow's)."
    )


def sessions_url(endpoint: str) -> str:
    """Where an endpoint's streams open: a deployment's ``endpoint_url`` with ``/sessions`` for ``/predict``, or a
    workflow's ``endpoint_url`` as it is."""
    return _endpoint_root(endpoint) + "/sessions"


def predict_url(endpoint: str) -> str:
    """Where an endpoint answers one image: a deployment's ``endpoint_url`` as it is (a workflow's endpoint refuses
    it, ``predict_unavailable``: it takes streams)."""
    return _endpoint_root(endpoint) + "/predict"


def frame_result(message: Mapping[str, Any], task: TaskName) -> AnyResult:
    """A frame's result as :meth:`DeploymentClient.infer` returns one: its annotations (a
    classifier's ranked labels) through the SDK's one result builder."""
    record = message.get("pictograph") or {}
    predictions: list[dict[str, Any]]
    if task == "classification":
        scores = message.get("tag_confidences") or {}
        predictions = [
            {"class": tag, "confidence": float(scores.get(tag, 0.0))}
            for tag in record.get("image_tags") or []
        ]
    else:
        predictions = list(record.get("annotations") or [])
    latency = message.get("latency_ms") or {}
    inference = latency.get("inference")
    return build_result(
        {"model_type": task, "predictions": predictions},
        task=task,
        # Every stream runs on PRT, Pictograph's realtime engine: a TensorRT engine of the model.
        backend="tensorrt",
        device=UNREPORTED_DEVICE,
        inference_ms=float(inference) if isinstance(inference, (int, float)) else None,
        source="This deployment's stream",
    )


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _backoff(drops: int) -> float:
    """Seconds before reading the stream again after its connection dropped ``drops`` times in a row."""
    return float(min(0.5 * 2 ** (drops - 1), 4.0))


def _capture_time(capture_ns: int | None) -> datetime | None:
    """A frame's capture time (nanoseconds since the Unix epoch) as an aware UTC datetime, to the microsecond."""
    if capture_ns is None or capture_ns <= 0:
        return None
    seconds, nanos = divmod(capture_ns, 1_000_000_000)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(microsecond=nanos // 1000)


def _source_quiet(message: Mapping[str, Any]) -> float | None:
    """How long a stats message says the stream has been without a frame: its sources' least ``last_frame_s`` (the
    deployment's own measure, in every stats message and each heartbeat - the numbers it sends when its source sends
    nothing). None when it does not say (no stats, or no frame yet)."""
    if message.get("type") != "stats":
        return None
    quiet = [
        q
        for s in message.get("sources") or []
        if isinstance(s, dict) and (q := _float(s.get("last_frame_s"))) is not None
    ]
    return min(quiet) if quiet else None


class _StallWatch:
    """A reader's ``idle_timeout``: raises :class:`StreamStalledError` once the reader has gone that long without a
    frame - counted from its last frame, or from when it began reading (so reading again after a stall waits another
    ``idle_timeout`` for the frames to come back, never raising again at once); the time a caller spends on a frame is
    not the stream's silence. How long the stream's source has really been quiet is the deployment's own measure when it
    sent one (its stats' ``last_frame_s``, its heartbeats' while the source sends nothing), else this reader's."""

    def __init__(self, idle_timeout: float | None) -> None:
        if idle_timeout is not None and not idle_timeout > 0:
            raise ValueError(
                "idle_timeout is a number of seconds above 0 (None: wait as long as the stream runs)."
            )
        self.idle_timeout = idle_timeout
        now = time.monotonic()
        self._frame_at = now  # its last frame, else when the reading began
        self._heard_at = now  # its last message of any kind
        self._source_quiet: tuple[float, float] | None = (
            None  # the deployment's last_frame_s, and when it came
        )

    @property
    def poll(self) -> float | None:
        """How long a read may wait on a silent socket before the clock is looked at (None: forever)."""
        return None if self.idle_timeout is None else min(self.idle_timeout, _STALL_POLL_S)

    def resumed(self) -> None:
        """The caller is done with a frame: the silence counts from now."""
        self._frame_at = time.monotonic()

    def saw(self, message: Mapping[str, Any] | None) -> None:
        """A message read (None: a read that waited ``poll`` with nothing): raises once the stream stalled."""
        if self.idle_timeout is None:
            return
        now = time.monotonic()
        if message is not None:
            self._heard_at = now
            if message.get("type") == "frame":
                self._frame_at = now
                self._source_quiet = None
                return
            quiet = _source_quiet(message)
            if quiet is not None:
                self._source_quiet = (quiet, now)
        waited = now - self._frame_at
        if waited < self.idle_timeout:
            return
        # Anything since its last frame (or since the reading began): the deployment kept talking - it sends its
        # numbers at least every two seconds or so, a heartbeat while its source sends nothing.
        alive = self._heard_at > self._frame_at
        said = self._source_quiet
        quiet = max(waited, said[0] + (now - said[1])) if said is not None else waited
        raise StreamStalledError(
            f"No frame for {quiet:.1f} s: "
            + (
                "the stream is up and its source sends nothing."
                if alive
                else "nothing came from the stream in that time - its connection may be down."
            ),
            quiet_s=quiet,
            alive=alive,
        )


class StreamFrame(Generic[R_co]):
    """One frame of a stream: its result, where it sits in the stream, when it was captured and how long it took."""

    __slots__ = (
        "capture_ns",
        "height",
        "index",
        "latency_ms",
        "measures",
        "message",
        "pts",
        "received_ns",
        "result",
        "width",
    )

    def __init__(self, message: dict[str, Any], result: R_co) -> None:
        record = message.get("pictograph") or {}
        #: When the SDK read it: nanoseconds since the Unix epoch, on this machine's clock.
        self.received_ns: int = time.time_ns()
        #: The frame's number in the stream (its first frame is 0).
        self.index: int | None = _int(message.get("frame"))
        #: Its time in the stream, in seconds.
        self.pts: float | None = _float(message.get("pts"))
        #: When it was captured: nanoseconds since the Unix epoch, by its sender's clock - a camera's,
        #: or the machine that sent the video - from its RTCP sender reports. None until the sender's
        #: first report (a second or so into a stream), and for a source with no capture clock.
        _ns = _int(message.get("capture_ns"))
        self.capture_ns: int | None = _ns if _ns is not None and _ns > 0 else None
        #: How long it took, in milliseconds (``total``: from its capture to its result, on the
        #: deployment's clock; ``engine``: inside the deployment's engine).
        self.latency_ms = StreamLatency.model_validate(message.get("latency_ms") or {})
        #: Its size in pixels (None when the deployment did not report it).
        self.width: int | None = _int(record.get("image_width"))
        self.height: int | None = _int(record.get("image_height"))
        #: Its result: the same typed result :meth:`DeploymentClient.infer` returns.
        self.result: R_co = result
        #: Each line's and zone's counts on this frame, by its id - a workflow's (its endpoint's
        #: ``live.measures`` names each): ``{"m0": {"in": 4, "out": 1, "total": 5}}``. Empty for a
        #: deployment's stream, or a workflow without lines or zones.
        self.measures: dict[str, Any] = dict(message.get("measures") or {})
        #: The message as the deployment sent it; its Pictograph JSON is under ``"pictograph"``.
        self.message: dict[str, Any] = message

    @property
    def capture_time(self) -> datetime | None:
        """When it was captured, by its sender's clock: an aware UTC datetime (None until the sender's
        first report)."""
        return _capture_time(self.capture_ns)

    @property
    def capture_latency_ms(self) -> float | None:
        """From its capture to its result reaching you, in milliseconds, on your clock: exact when this
        machine sent the video (its frames are timed by this clock), as close as this machine's clock
        and the camera's agree otherwise (NTP). None while it has no capture time."""
        if self.capture_ns is None:
            return None
        return (self.received_ns - self.capture_ns) / 1e6

    def __repr__(self) -> str:
        return (
            f"StreamFrame(index={self.index}, result={type(self.result).__name__}, "
            f"latency_ms={self.latency_ms.engine})"
        )


class DeploymentStream(Generic[R_co]):
    """A live stream through a deployment: its video goes in, each frame's result comes out.

    Opened by :meth:`DeploymentClient.open_stream`. Use it as a context manager, so that it
    stops when you are done (it is billed while it runs)::

        with deployment.open_stream("rtsp://camera.example.com:554/live") as stream:
            for frame in stream.frames():
                print(frame.index, frame.result)

    It calls the deployment through its :class:`DeploymentClient`'s connection: keep the
    client open while the stream runs.
    """

    def __init__(
        self,
        opened: OpenedStream,
        *,
        origin: str,
        headers: Mapping[str, str],
        http: httpx.Client,
        task: TaskName | None,
    ) -> None:
        self._opened = opened
        self._origin = origin.rstrip("/")
        self._headers = dict(headers)
        self._http = http
        self._task = task
        self._overlay = opened.overlay
        self._stream = opened.stream
        self._stopped = False
        #: Why the stream ended, once it has; None while it runs. ``"stopped"`` (by :meth:`stop`,
        #: here or elsewhere), ``"source_ended"`` (its camera stopped sending), ``"idle"`` (30
        #: seconds with nothing reading it or sending to it), ``"time_limit"`` (four hours),
        #: ``"container_stopped"``, or ``"ended"`` (it was gone when read).
        self.ended: str | None = None
        #: The deployment's own words for why it ended, when it gave some.
        self.ended_detail: str | None = None

    # ── what the deployment answered when it opened ──

    @property
    def id(self) -> str:
        """The stream's id."""
        return self._opened.session

    @property
    def kind(self) -> str:
        """Where its video comes from: ``"whip"`` (sent to it), ``"rtsp"`` or ``"whep"`` (a camera
        the deployment connects to)."""
        return self._opened.kind

    @property
    def overlay(self) -> bool:
        """Whether the live view draws the results on the video (see :meth:`set_overlay`)."""
        return self._overlay

    @property
    def stream(self) -> bool:
        """Whether the live view's pictures are sent (see :meth:`set_stream`). Off, ``frames()`` goes on."""
        return self._stream

    @property
    def stream_limit(self) -> int:
        """How many streams the deployment runs at once."""
        return self._opened.stream_limit

    @property
    def opened_at(self) -> datetime | None:
        """When it opened."""
        return self._opened.opened_at

    @property
    def expires_at(self) -> datetime | None:
        """When it ends on its own: four hours after it opened."""
        return self._opened.expires_at

    @property
    def ice_servers(self) -> list[dict[str, Any]]:
        """The ICE servers a WebRTC client (a WHIP sender, a WHEP player) connects with."""
        return list(self._opened.ice_servers)

    @property
    def fallback(self) -> bool:
        """Whether it runs elsewhere in the US: its deployment's region had no GPU free for 30 seconds."""
        return self._opened.fallback

    @property
    def outputs(self) -> WorkflowRunWrites | None:
        """What it writes beside its log - a workflow endpoint's stream, opened with ``outputs``: its outputs as
        resolved (each folder its dataset by id with its name, and the folder it writes). None: its log alone."""
        from pictograph.models.workflow import WorkflowRunWrites as Writes

        given = self._opened.outputs
        return Writes.model_validate(given) if given is not None else None

    @property
    def url(self) -> str:
        """The stream's own URL: ``GET`` reads its state, ``DELETE`` stops it."""
        return self._origin + self._opened.paths.session

    @property
    def whep_url(self) -> str:
        """Where to watch it: the video with the results drawn on it, played by any WHEP player
        (a receive-only H.264 offer, with the deployment's token as its bearer)."""
        return self._origin + self._opened.paths.whep

    @property
    def whip_url(self) -> str | None:
        """Where a ``"whip"`` stream's video is sent - H.264, from any WHIP client, with the
        deployment's token as its bearer. None for a camera the deployment connects to."""
        whip = self._opened.paths.whip
        return self._origin + whip if whip else None

    @property
    def results_url(self) -> str:
        """The WebSocket each frame's result comes on (what :meth:`frames` reads)."""
        scheme, rest = self._origin.split("://", 1)
        return ("wss" if scheme == "https" else "ws") + "://" + rest + self._opened.paths.results

    # ── reading it ──

    def frames(self, *, idle_timeout: float | None = None) -> Iterator[StreamFrame[R_co]]:
        """Each frame's result, as the deployment infers it, until the stream ends.

        Yields :class:`StreamFrame`, whose ``result`` is the typed result
        :meth:`DeploymentClient.infer` returns. Stops when the stream ends (:attr:`ended` says
        why). A dropped connection is reconnected (the frames in between are missed: they are
        live).

        Args:
            idle_timeout: Seconds this read may go without a frame before it raises
                :class:`~pictograph.StreamStalledError` - a camera that stalled, a sender gone, a
                connection down - instead of waiting. The stream goes on: read again to wait
                another ``idle_timeout`` for its frames to come back. ``None`` (the default) waits
                as long as the stream runs. The time you spend on a frame does not count.

        Raises:
            StreamStalledError: no frame for ``idle_timeout`` seconds.
            AuthError / NotFoundError / ConflictError: the deployment refused the read.
            ServerError: the stream's engine stopped (``code`` ``session_failed``).
            NetworkError: the connection dropped and did not come back.
            ValidationError: the deployment serves another task than this client's ``task``.
        """
        watch = _StallWatch(idle_timeout)
        for message in self._read(watch.poll):
            watch.saw(message)
            if message is not None and message.get("type") == "frame":
                result = frame_result(message, self._task_of(message))
                yield StreamFrame(message, cast("R_co", result))
                watch.resumed()

    def messages(self, *, idle_timeout: float | None = None) -> Iterator[dict[str, Any]]:
        """Every message the stream sends, as the deployment sends it, until the stream ends.

        Each frame's result (``"type": "frame"``, its Pictograph JSON under ``"pictograph"``),
        the stream's numbers each second (``"stats"``; when its source sends nothing, a
        ``"heartbeat": true`` one saying how long, each source's ``last_frame_s``), and its events.
        :meth:`frames` is the typed view of the frames among them.

        Args:
            idle_timeout: as :meth:`frames`'s - raise :class:`~pictograph.StreamStalledError` after
                this many seconds without a frame (other messages still coming or not).
        """
        watch = _StallWatch(idle_timeout)
        for message in self._read(watch.poll):
            watch.saw(message)
            if message is not None:
                yield message
                if message.get("type") == "frame":
                    watch.resumed()

    def _read(self, poll: float | None) -> Iterator[dict[str, Any] | None]:
        """The stream's messages until it ends, reconnecting a dropped connection; ``None`` each time a read waited
        ``poll`` seconds with nothing (never, when ``poll`` is None)."""
        drops = 0
        while self.ended is None:
            try:
                connection = self._connect()
            except Exception as e:  # the socket's own errors: its address, its handshake, a timeout
                refused = _refusal(e)
                if refused is not None:
                    raise refused from e
                drops += 1
                if drops > RECONNECTS:
                    raise NetworkError(f"The stream's results could not be reached: {e}") from e
                time.sleep(_backoff(drops))
                continue
            code, reason = 1006, ""  # until a close is read: none (no close frame)
            with connection:
                while True:
                    try:
                        raw = connection.recv(timeout=poll)
                    except TimeoutError:
                        # Nothing for `poll` seconds: the reader looks at its clock.
                        yield None
                        continue
                    except Exception as e:  # its close (normal or not) ends the read
                        code, reason = _close_of(e)
                        break
                    drops = 0
                    yield _parsed(raw)
            if not self._dropped(code, reason):
                return
            drops += 1
            if drops > RECONNECTS:
                if 4500 <= code < 4600:
                    body = {"error": reason} if reason else None
                    raise with_gateway_hint(
                        from_response(code - 4000, body=body), body, stream=True
                    )
                raise NetworkError(
                    f"The stream's results connection dropped ({code}) and did not come back."
                )
            time.sleep(_backoff(drops))

    # ── changing it ──

    def set_overlay(self, on: bool) -> bool:
        """Draw the results on the live view (True), or show the video as the camera sends it
        (False), from the next frame. Returns whether the overlay is on now."""
        body = parse_response(
            self._call("POST", self._opened.paths.overlay, {"on": bool(on)}), stream=True
        )
        overlay = body.get("overlay")
        self._overlay = overlay if isinstance(overlay, bool) else bool(on)
        return self._overlay

    def set_stream(self, on: bool) -> bool:
        """Send the live view's pictures (True), or not (False), from the next frame. Off, the deployment
        draws and encodes no picture and the live view (``whep_url``) gets none - each frame's result, its
        events and its log go on, and the deployment spends nothing on the picture. On again, its next
        picture is a keyframe. Returns whether the live view is sent now.

        Raises:
            ConflictError: the endpoint cannot turn this stream's pictures off.
        """
        if self._opened.paths.stream is None:
            raise ConflictError(
                "This stream's endpoint has no stream switch: its live view is always sent.",
                code="stream_switch_unavailable",
                fix="Open the stream again: endpoints answer a stream switch since SDK 1.69.181.",
            )
        body = parse_response(
            self._call("POST", self._opened.paths.stream, {"on": bool(on)}), stream=True
        )
        stream = body.get("stream")
        self._stream = stream if isinstance(stream, bool) else bool(on)
        return self._stream

    def state(self) -> StreamState:
        """The stream's state, as the deployment reports it now."""
        return StreamState.model_validate(
            parse_response(self._call("GET", self._opened.paths.session), stream=True)
        )

    def stop(self) -> None:
        """Stop the stream; its billing stops with it. Calling it again does nothing."""
        if self._stopped:
            return
        resp = self._call("DELETE", self._opened.paths.session)
        if resp.status_code not in (404, 410):  # either: it had already ended
            parse_response(resp, stream=True)
        self._stopped = True
        if self.ended is None:
            self.ended = "stopped"

    def __enter__(self) -> DeploymentStream[R_co]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            self.stop()
        except PictographError:
            if exc_type is None:
                raise  # never hide the error that left the block behind this one

    def __repr__(self) -> str:
        # Never the token: it would leak through logs and tracebacks.
        return f"DeploymentStream(id={self.id!r}, kind={self.kind!r}, ended={self.ended!r})"

    # ── internals ──

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> httpx.Response:
        try:
            return self._http.request(
                method,
                self._origin + path,
                headers=self._headers,
                json=body,
                timeout=CALL_TIMEOUT_S,
            )
        except httpx.TimeoutException as e:
            raise RequestTimeoutError(str(e)) from e
        except httpx.RequestError as e:
            raise NetworkError(str(e)) from e

    def _connect(self) -> ClientConnection:
        # Imported here, so `import pictograph` never loads the WebSocket client.
        from websockets.sync.client import connect
        from websockets.typing import Subprotocol

        return connect(
            self.results_url,
            additional_headers=self._headers,
            subprotocols=[Subprotocol(RESULTS_PROTOCOL)],
            open_timeout=CALL_TIMEOUT_S,
            max_size=MAX_MESSAGE_BYTES,
        )

    def _dropped(self, code: int, reason: str) -> bool:
        """What the results socket's close means: True when its connection dropped (read it
        again); False when the stream ended (:attr:`ended` says why). Raises a refusal."""
        if code == 1000 and reason:
            self.ended_detail = reason
            if reason.startswith(_ENGINE_STOPPED):
                self.ended = "failed"
                raise ServerError(
                    f"The stream's engine stopped{reason[len(_ENGINE_STOPPED) :]}.",
                    code="session_failed",
                    fix="Open a new stream; if it stops again, contact support@pictograph.io.",
                )
            self.ended = _ENDED.get(reason) or (
                "idle"
                if reason.startswith("no request")
                else "time_limit"
                if reason.endswith("limit")
                else "ended"
            )
            return False
        if code == 4410:
            self.ended = "ended"
            return False
        if 4400 <= code < 4500:
            body = {"error": reason} if reason else None
            raise with_gateway_hint(from_response(code - 4000, body=body), body, stream=True)
        return True  # 1000 with no reason, 1001, 1006, 1011, 4502, 4503: the connection, not the stream

    def _task_of(self, message: Mapping[str, Any]) -> TaskName:
        model = message.get("model")
        reported = model.get("task") if isinstance(model, dict) else None
        if reported is not None and reported not in TASK_RESULT_TYPES:
            raise ServerError(
                f"The stream reported a task this SDK does not know: {reported!r}.",
                response=dict(message),
                fix="Upgrade the SDK (pip install -U pictograph); messages() gives every message "
                "as it was sent meanwhile.",
            )
        if self._task is not None and reported is not None and reported != self._task:
            raise ValidationError(
                f"This deployment serves a {reported!r} model, but the client was constructed "
                f"with task={self._task!r}.",
                response=dict(message),
                fix=f"Construct the client with task={reported!r}, or omit task: the stream names its own.",
            )
        task = cast("TaskName | None", reported) or self._task
        if task is None:
            raise ServerError(
                "The stream's frames name no task, so their results cannot be typed.",
                response=dict(message),
                fix="Construct the client with task=..., or read messages() for every message as it was sent.",
            )
        return task


def _close_of(error: Exception) -> tuple[int, str]:
    """The close code and reason a WebSocket's closing exception carries (1006: no close frame)."""
    from websockets.exceptions import ConnectionClosed

    if isinstance(error, ConnectionClosed):
        if error.rcvd is not None:
            return error.rcvd.code, error.rcvd.reason
        return 1006, ""
    raise error


def _refusal(error: Exception) -> PictographError | None:
    """The typed error for a results socket the gateway refused at its handshake (a 4xx), else None."""
    from websockets.exceptions import InvalidStatus

    if not isinstance(error, InvalidStatus):
        return None
    status = error.response.status_code
    if not 400 <= status < 500:
        return None
    body: Any
    try:
        body = json.loads(error.response.body or b"null")
    except ValueError:
        body = None
    return with_gateway_hint(from_response(status, body=body), body, stream=True)


def _parsed(raw: str | bytes) -> dict[str, Any]:
    try:
        message = json.loads(raw)
    except ValueError as e:
        raise ServerError(f"The stream sent a message that is not JSON: {e}") from e
    if not isinstance(message, dict):
        raise ServerError("The stream sent a message that is not a JSON object.")
    return message


def open_stream(
    http: httpx.Client,
    *,
    endpoint: str,
    headers: Mapping[str, str],
    task: TaskName | None,
    retry: RetryPolicy,
    source: str,
    overlay: bool,
    stream: bool = True,
    transport: RtspTransport | None = None,
    buffer_mode: RtspBufferMode | None = None,
    reconnect: RtspReconnect | None = None,
    outputs: RunOutputs | None = None,
) -> DeploymentStream[Any]:
    """Open a stream on the deployment at ``endpoint`` (what :meth:`DeploymentClient.open_stream` calls)."""
    url = sessions_url(endpoint)
    out_body = outputs_body(outputs)
    body: dict[str, Any] = {
        "source": stream_source(
            source, transport=transport, buffer_mode=buffer_mode, reconnect=reconnect
        ),
        "overlay": bool(overlay),
        "stream": bool(stream),
    }
    if out_body is not None:
        body["outputs"] = out_body
    attempt = 0
    while True:
        try:
            resp = http.post(url, json=body, headers=dict(headers), timeout=OPEN_TIMEOUT_S)
        except httpx.TimeoutException as e:
            raise RequestTimeoutError(
                f"The stream did not open within {OPEN_TIMEOUT_S:.0f} s: {e}"
            ) from e
        except httpx.RequestError as e:
            raise NetworkError(str(e)) from e
        if (
            resp.status_code in (409, 503)
            and attempt < retry.max_retries
            and _error_of(resp) in _RETRY_OPEN
            and retry.wait_to_retry(resp, attempt)
        ):
            attempt += 1
            continue
        opened = OpenedStream.model_validate(parse_response(resp, stream=True))
        parts = urlsplit(url)
        return DeploymentStream(
            opened, origin=f"{parts.scheme}://{parts.netloc}", headers=headers, http=http, task=task
        )


def _error_of(resp: httpx.Response) -> str | None:
    try:
        body = resp.json()
    except ValueError:
        return None
    error = body.get("error") if isinstance(body, dict) else None
    return error if isinstance(error, str) else None
