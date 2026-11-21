"""The Live log's reads, shared by ``client.deployments.logs`` and ``client.workflows.logs`` (sync and async).

Every argument is checked when ``logs()`` is called - before any request, never at the first iteration. The history
comes page by page from a generator (the caller loops over records, never over pages), and the window a first page
resolves (``since`` / ``until`` as times) is the one every later page reads, so a relative ``since="10m"`` does not
drift. ``live=True`` is a live connection (Server-Sent Events): its records as they arrive, for as long as the caller
loops - the connection reopened from its last event's id whenever it drops or the server ends it (every few
minutes), so the caller never sees one end.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pictograph.exceptions import ApiError, NetworkError, RateLimitError, ServerError
from pictograph.models.live_log import LOG_KINDS, LogPage, LogRecord
from pictograph.resources._base import parse_model

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence

    from pictograph._http.streaming import SSEEvent

logger = logging.getLogger("pictograph")

#: Records each request reads (the API's largest page is 1000).
PAGE_SIZE = 500

#: A live connection's quiet longest: the server says something at least every 15 s (a keepalive), so this long of
#: nothing is a dead connection - opened again from where it stood.
LIVE_READ_TIMEOUT_S = 45.0

#: How long a live connection waits before it is opened again after a drop or a refusal to retry (each a step
#: longer, back to the first after a record arrives); a server's ``Retry-After`` is honoured when it is longer.
RECONNECT_DELAYS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 15.0)

#: The header naming one ``logs(live=True)``'s connection across its reconnections: the server counts it once under
#: the plan's live connections, never once a reconnection.
CONNECTION_HEADER = "X-Pictograph-Live-Connection"

#: What ``until`` with ``live=True`` says (the API's own words).
LIVE_HAS_NO_END = (
    "A live connection has no end: leave until= out (since= starts it with the history), or read the "
    "history alone with since= and until=."
)

_DURATION = re.compile(
    r"^\s*(\d+(\.\d+)?\s*(ms|secs?|s|mins?|m|hrs?|h|days?|d|wks?|w)\s*)+$", re.IGNORECASE
)

TimeBound = str | datetime | timedelta


def _bound(value: TimeBound | None, name: str) -> str | None:
    """A window bound as the API takes it: a datetime as ISO 8601 (naive is UTC), a timedelta as that long ago, a
    string as given (``now``, a duration ago - ``10m``, ``1h30m``, ``2d`` - or an ISO 8601 time)."""
    if value is None:
        return None
    if isinstance(value, timedelta):
        if value.total_seconds() < 0:
            raise ValueError(
                f"{name} is a time in the past: a timedelta of how long ago, not a negative one."
            )
        return f"{value.total_seconds():g}s"
    if isinstance(value, datetime):
        aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return aware.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(
                f"{name} is `now`, a duration ago (`10m`) or an ISO 8601 time; got an empty string."
            )
        if text.lower() == "now" or _DURATION.match(text):
            return text
        try:
            datetime.fromisoformat(text.replace("Z", "+00:00").replace("z", "+00:00"))
        except ValueError:
            raise ValueError(
                f"{name} is `now`, a duration ago (`30s`, `10m`, `1h30m`, `2d`) or an ISO 8601 time "
                f"(`2026-11-20T18:00:00Z`); got {value!r}."
            ) from None
        return text
    raise TypeError(f"{name} is a str, a datetime or a timedelta; got {type(value).__name__}.")


def _names(values: Sequence[str] | str | None, what: str) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    out = [str(v) for v in values]
    if not out or any(not v.strip() for v in out):
        raise ValueError(f"{what} is a list of names, none of them empty.")
    return out


def log_params(
    *,
    live: bool,
    since: TimeBound | None,
    until: TimeBound | None,
    kinds: Sequence[str] | str | None,
    classes: Sequence[str] | str | None,
    min_confidence: float | None,
    session: str | None,
    every: int | None,
) -> dict[str, Any]:
    """The first page's query - or the live connection's - every argument checked (``ValueError`` / ``TypeError``
    naming it)."""
    if live and until is not None:
        raise ValueError(LIVE_HAS_NO_END)
    params: dict[str, Any] = {"live": "true"} if live else {"limit": PAGE_SIZE}
    s, u = _bound(since, "since"), _bound(until, "until")
    if s is not None:
        params["since"] = s
    if u is not None:
        params["until"] = u
    named_kinds = _names(kinds, "kinds")
    if named_kinds is not None:
        unknown = [k for k in named_kinds if k not in LOG_KINDS]
        if unknown:
            raise ValueError(
                f"kinds are {', '.join(LOG_KINDS)}; got {', '.join(map(repr, unknown))}."
            )
        params["kinds"] = named_kinds
    named_classes = _names(classes, "classes")
    if named_classes is not None:
        params["classes"] = named_classes
    if min_confidence is not None:
        if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
            raise TypeError("min_confidence is a number from 0 to 1.")
        if not 0 <= float(min_confidence) <= 1:
            raise ValueError(f"min_confidence is from 0 to 1; got {min_confidence}.")
        params["min_confidence"] = float(min_confidence)
    if session is not None:
        if not isinstance(session, str) or not session.strip():
            raise ValueError("session is a stream's id (or a workflow run's).")
        params["session"] = session.strip()
    if every is not None:
        if isinstance(every, bool) or not isinstance(every, int) or every < 1:
            raise ValueError(f"every is a whole number of frames, 1 or more; got {every!r}.")
        params["every"] = every
    return params


def log_path(base: str, name: str) -> str:
    """A resource's log route: its name in the path, escaped (a name may hold a space or a slash)."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Name the deployment or workflow whose log to read.")
    return f"{base}{quote(name.strip(), safe='')}/logs"


def _next_params(params: Mapping[str, Any], page: LogPage) -> dict[str, Any] | None:
    """The next page's query: the first page's, at its cursor, over the window the first page resolved."""
    if not page.next_cursor:
        return None
    out = {**params, "cursor": page.next_cursor}
    if page.since is not None:
        out["since"] = page.since.isoformat().replace("+00:00", "Z")
    if page.until is not None:
        out["until"] = page.until.isoformat().replace("+00:00", "Z")
    return out


def iter_records(
    fetch: Callable[[dict[str, Any]], Any], params: dict[str, Any]
) -> Iterator[LogRecord]:
    """Every record, page by page (``fetch(query)`` answers one page)."""
    query: dict[str, Any] | None = params
    while query is not None:
        page = parse_model(LogPage, fetch(query))
        yield from page.records
        query = _next_params(params, page)


async def aiter_records(
    fetch: Callable[[dict[str, Any]], Awaitable[Any]], params: dict[str, Any]
) -> AsyncIterator[LogRecord]:
    """:func:`iter_records`, awaited."""
    query: dict[str, Any] | None = params
    while query is not None:
        page = parse_model(LogPage, await fetch(query))
        for record in page.records:
            yield record
        query = _next_params(params, page)


# ------------------------------------------------------------------------------------------------------- live


def _live_error(data: Any) -> ApiError:
    """The connection's last event when it cannot go on (``event: error``), as the error the API would answer."""
    body = data if isinstance(data, dict) else {}
    return ApiError(
        str(body.get("message") or "The live connection ended with an error."),
        code=str(body["error"]) if body.get("error") else None,
        response=body or data,
    )


def _delay(failures: int, error: BaseException) -> float:
    """How long to wait before opening the connection again: the next step, or the server's ``Retry-After`` when it
    asks for longer."""
    step = RECONNECT_DELAYS[min(failures, len(RECONNECT_DELAYS) - 1)]
    retry_after = getattr(error, "retry_after", None)
    return max(step, float(retry_after)) if isinstance(retry_after, (int, float)) else step


def _reconnectable(error: BaseException) -> bool:
    """A drop (the network, a quiet past LIVE_READ_TIMEOUT_S) or a refusal to retry (429, 5xx) - never a refusal of
    the request itself (4xx: it is raised to the caller)."""
    return isinstance(error, (NetworkError, ServerError, RateLimitError))


def iter_live(
    open_stream: Callable[[dict[str, Any], str | None, str], Iterator[SSEEvent]],
    params: dict[str, Any],
    *,
    sleep: Callable[[float], None] | None = None,
) -> Iterator[LogRecord]:
    """Every record of a live connection as it arrives, for as long as the caller loops (``break`` ends it, and
    closes the connection). The connection is opened again from its last event's id - a record's cursor, or where it
    stood (an id alone) - when it drops or the server ends it, so nothing is lost or seen twice."""
    connection = uuid.uuid4().hex
    last: str | None = None
    failures = 0
    while True:
        try:
            for event in open_stream(params, last, connection):
                if event.id:
                    last = event.id
                if event.event == "error":
                    raise _live_error(event.data)
                if isinstance(event.data, dict) and "kind" in event.data:
                    failures = 0
                    yield parse_model(LogRecord, event.data)
            failures = (
                0  # the server ended it (as it does every few minutes): open it again, at once
            )
        except Exception as e:
            if not _reconnectable(e):
                raise
            wait = _delay(failures, e)
            failures += 1
            logger.warning("live connection dropped (%s); opening it again in %.1f s", e, wait)
            (sleep or time.sleep)(wait)


async def aiter_live(
    open_stream: Callable[[dict[str, Any], str | None, str], AsyncIterator[SSEEvent]],
    params: dict[str, Any],
    *,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> AsyncIterator[LogRecord]:
    """:func:`iter_live`, awaited."""
    connection = uuid.uuid4().hex
    last: str | None = None
    failures = 0
    while True:
        try:
            async for event in open_stream(params, last, connection):
                if event.id:
                    last = event.id
                if event.event == "error":
                    raise _live_error(event.data)
                if isinstance(event.data, dict) and "kind" in event.data:
                    failures = 0
                    yield parse_model(LogRecord, event.data)
            failures = 0
        except Exception as e:
            if not _reconnectable(e):
                raise
            wait = _delay(failures, e)
            failures += 1
            logger.warning("live connection dropped (%s); opening it again in %.1f s", e, wait)
            await (sleep or asyncio.sleep)(wait)
