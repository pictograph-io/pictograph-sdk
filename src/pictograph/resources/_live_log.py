"""The Live log's reads, shared by ``client.deployments.logs`` and ``client.workflows.logs`` (sync and async).

Every argument is checked when ``logs()`` is called - before any request, never at the first iteration - and the
records come page by page from a generator: the caller loops over records, never over pages. The window a first
page resolves (``since`` / ``until`` as times) is the one every later page reads, so a relative ``since="10m"`` does
not drift while the pages are read.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pictograph.models.live_log import LOG_KINDS, LogPage, LogRecord
from pictograph.resources._base import parse_model

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence

#: Records each request reads (the API's largest page is 1000).
PAGE_SIZE = 500

#: What ``live=True`` says until a live connection to a log exists.
LIVE_NOT_YET = "A live connection to a log is not available yet: read its history with since= and until= for now."

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
    """The first page's query, every argument checked (``ValueError`` / ``TypeError`` naming it).

    Raises:
        NotImplementedError: ``live=True`` - a live connection to a log is not available yet.
    """
    if live:
        raise NotImplementedError(LIVE_NOT_YET)
    params: dict[str, Any] = {"limit": PAGE_SIZE}
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
