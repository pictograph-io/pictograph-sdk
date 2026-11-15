"""A deployment stream's results socket, served for real in a test.

A local ``websockets`` server scripted per test - the messages each connection sends, then how
it ends (a close with a code and reason, or a cut with no close frame) - so the client's own
socket code (its handshake, its reads, its reconnects) runs exactly as it does against a
deployment's gateway. Shared by the SDK's stream tests and the CLI's ``deployments stream``.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from websockets.sync.server import ServerConnection, serve

if TYPE_CHECKING:
    from collections.abc import Iterator

SESSION = "7955b01ae9212f34"
SESSION_PATH = f"/us-west/abc123/sessions/{SESSION}"
_BOX = {"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0}


@dataclass
class Connection:
    """What one connection to the results socket sends, then how it ends: a close
    (``code``, ``reason``), or ``drop`` - the socket cut with no close frame (the client reads
    1006). A number among ``messages`` is a pause, in seconds: nothing sent for that long (a
    camera that stalled, a connection gone quiet)."""

    messages: list[dict[str, Any] | float] = field(default_factory=list)
    code: int = 1000
    reason: str = ""
    drop: bool = False


@dataclass
class ResultsServer:
    """The server: ``script`` (one :class:`Connection` per connection, in order) and what each
    connection's handshake carried (``seen``)."""

    port: int = 0
    script: list[Connection] = field(default_factory=list)
    seen: list[dict[str, Any]] = field(default_factory=list)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def endpoint(self) -> str:
        """A deployment's endpoint_url on this server's host."""
        return f"{self.origin}/us-west/abc123/predict"

    @property
    def sessions(self) -> str:
        return f"{self.origin}/us-west/abc123/sessions"


@contextmanager
def serve_results() -> Iterator[ResultsServer]:
    state = ResultsServer()

    def handler(ws: ServerConnection) -> None:
        request = ws.request
        state.seen.append(
            {
                "path": request.path if request else None,
                "authorization": request.headers.get("Authorization") if request else None,
                "subprotocol": ws.subprotocol,
            }
        )
        if not state.script:
            ws.close(4404, "session_not_found")
            return
        conn = state.script.pop(0)
        for message in conn.messages:
            if isinstance(message, (int, float)):
                time.sleep(message)
                continue
            try:
                ws.send(json.dumps(message))
            except Exception:  # the client left (a reader that raised closes its socket)
                return
        if conn.drop:
            ws.socket.close()
            return
        ws.close(conn.code, conn.reason)

    with serve(
        handler,
        "127.0.0.1",
        0,
        subprotocols=["pictograph.results.v1"],  # type: ignore[list-item]
    ) as server:
        state.port = server.socket.getsockname()[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield state
        finally:
            server.shutdown()
            thread.join(timeout=5)


def opened(kind: str = "rtsp", **over: Any) -> dict[str, Any]:
    """What a gateway answers to an open (``POST {sessions}``)."""
    paths = {
        "session": SESSION_PATH,
        "whep": f"{SESSION_PATH}/whep",
        "overlay": f"{SESSION_PATH}/overlay",
        "stream": f"{SESSION_PATH}/stream",
        "results": f"{SESSION_PATH}/results",
    }
    if kind == "whip":
        paths["whip"] = f"{SESSION_PATH}/whip"
    return {
        "session": SESSION,
        "kind": kind,
        "overlay": True,
        "stream": True,
        "stream_limit": 4,
        "idle_timeout_s": 30,
        "opened_at": "2026-10-26T11:44:44Z",
        "expires_at": "2026-10-26T15:44:44Z",
        "ice_servers": [{"urls": "stun:stun.l.google.com:19302"}],
        "paths": paths,
        **over,
    }


def joined_state(**over: Any) -> dict[str, Any]:
    """What a gateway answers a join's read of a stream's state (``GET {session}?ice=1``): a workflow file run's live
    view (B607 slice C) - its paths (no whip), its engine's account of its two switches, the ICE servers a viewer joins
    with."""
    return {
        "session": SESSION,
        "kind": "run",
        "opened_at": "2026-10-30T11:44:44Z",
        "expires_at": "2026-10-30T12:54:44Z",
        "fallback": False,
        "paths": {
            "session": SESSION_PATH,
            "whep": f"{SESSION_PATH}/whep",
            "overlay": f"{SESSION_PATH}/overlay",
            "stream": f"{SESSION_PATH}/stream",
            "results": f"{SESSION_PATH}/results",
        },
        "engine": {
            "status": "running",
            "uptime_s": 4.2,
            "frame_messages": 120,
            "overlay": True,
            "stream": False,
        },
        "ice_servers": [{"urls": "stun:stun.l.google.com:19302"}],
        **over,
    }


def frame(n: int, *annotations: dict[str, Any], task: str = "object_detection") -> dict[str, Any]:
    """A frame message, as the deployment's engine sends it."""
    return {
        "type": "frame",
        "schema": 2,
        "engine": "pictograph-prt",
        "source": SESSION,
        "frame": n,
        "pts": n / 30,
        "latency_ms": {"total": 12.5, "engine": 11.7, "inference": 9.6},
        "model": {"name": "fixture", "task": task},
        "pictograph": {
            "image_width": 1280,
            "image_height": 720,
            "image_tags": [],
            "annotation_count": len(annotations),
            "annotations": list(annotations),
        },
    }


def stats_message(*, quiet: float | None = None, heartbeat: bool = False) -> dict[str, Any]:
    """A stats message, as the deployment's engine sends it each second: its source's ``last_frame_s``
    (``quiet``) when given, marked ``heartbeat`` when the engine sent it because its source sends nothing."""
    source: dict[str, Any] = {"id": SESSION, "frames": 10, "decoded": 10, "dropped": 0}
    if quiet is not None:
        source["last_frame_s"] = quiet
    message: dict[str, Any] = {
        "type": "stats",
        "engine": "pictograph-prt",
        "pipeline": "session",
        "ts_ns": 1_792_125_225_000_000_000,
        "sources": [source],
    }
    if heartbeat:
        message["heartbeat"] = True
    return message


def box(name: str, confidence: float) -> dict[str, Any]:
    """A bounding-box annotation, as a frame carries it."""
    return {
        "id": f"id-{name}",
        "name": name,
        "type": "bbox",
        "bounding_box": dict(_BOX),
        "confidence": confidence,
    }
