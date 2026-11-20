"""The Live log in the CLI: ``pictograph deployments logs <name>`` and ``pictograph workflows logs <name>``.

A table by default, as the CLI's other lists print (its first ``--limit`` records, 100 unless given); ``--json`` prints
one record a line, as it comes (JSON Lines, for ``jq``; every record unless ``--limit``); ``--download DIR`` saves
each capture's image there as it is read. ``--live`` (``-f`` / ``--follow`` its aliases) asks for a live connection
to the log, which is not available yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import typer

from pictograph.cli._errors import CliError, ExitCode
from pictograph.cli._format import console, err_console, print_json_line, print_table

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from pictograph.models.live_log import LogRecord

#: The table's records when ``--limit`` is not given (``--json`` prints every record).
TABLE_LIMIT = 100

LIVE_HELP = "Open a live connection to the log (not available yet). -f and --follow are aliases."
SINCE_HELP = "Where the history starts: now, a duration ago (30s, 10m, 1h30m, 2d) or an ISO 8601 time. Default: 1h."
UNTIL_HELP = "Where it ends (the same forms). Default: now."
KIND_HELP = "Only this kind: frame, event, capture, stats or status (repeat it, or comma-separate)."
CLASS_HELP = "Only records with an annotation of this class, carrying those alone (repeat it, or comma-separate)."
CONFIDENCE_HELP = "Only annotations at least this confident (0 to 1)."
EVERY_HELP = "Every Nth frame of each stream."
LIMIT_HELP = (
    "Stop after N records (0: every one). Default: 100 in the table; every record with --json."
)
JSON_HELP = "One record a line, as JSON (JSON Lines: pipe it to jq)."
DOWNLOAD_HELP = "Save each capture's image into this directory."


def _list(values: list[str] | None) -> list[str] | None:
    out = [p.strip() for v in values or [] for p in v.split(",") if p.strip()]
    return out or None


def records(
    read: Callable[..., Iterator[LogRecord]], name: str, **kwargs: Any
) -> Iterator[LogRecord]:
    """The SDK's ``logs()`` with the CLI's lists split; an argument it refuses is the CLI's own error."""
    kwargs["kinds"] = _list(kwargs.get("kinds"))
    kwargs["classes"] = _list(kwargs.get("classes"))
    try:
        return read(name, **kwargs)
    except NotImplementedError as e:
        raise CliError(str(e), exit_code=ExitCode.INVALID, code="live_unavailable") from None
    except (ValueError, TypeError) as e:
        raise CliError(str(e), exit_code=ExitCode.INVALID, code="invalid_argument") from None


def summary(record: LogRecord) -> str:
    """One line of what a record holds."""
    if record.kind == "event" and record.event is not None:
        words = record.event.name or record.event.kind or "event"
        return f"{words}: {record.event.text}" if record.event.text else words
    if record.kind == "stats" and record.stats is not None:
        s = record.stats
        fps = f"{s.fps:g} fps" if s.fps is not None else "- fps"
        return f"{fps}, {s.frames or 0} frames, {s.detections or 0} detections"
    if record.kind == "status" and record.status is not None:
        return record.status.state + (f" ({record.status.reason})" if record.status.reason else "")
    counts: dict[str, int] = {}
    for a in record.annotations:
        counts[a.name] = counts.get(a.name, 0) + 1
    for t in record.tags:
        counts[t.name] = counts.get(t.name, 0) + 1
    found = ", ".join(f"{n} x{c}" if c > 1 else n for n, c in counts.items()) or "nothing found"
    if record.kind == "capture" and record.capture is not None:
        return f"{record.capture.file or 'image'} ({found})"
    return found


def _time(record: LogRecord) -> str:
    return record.time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if record.time is not None else ""


def show(
    stream: Iterator[LogRecord],
    *,
    title: str,
    json_output: bool,
    limit: int | None,
    download: Path | None,
) -> None:
    """Print the records - a table, or a JSON line each as it comes - and save captures when asked."""
    cap = limit if limit is not None else (0 if json_output else TABLE_LIMIT)
    if download is not None:
        download.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    stopped = False
    for shown, record in enumerate(stream, start=1):
        # At the limit, stop with this record: reading the next would fetch a page only to say there is one.
        stopped = bool(cap) and shown >= cap
        saved: Path | None = None
        if download is not None and record.kind == "capture" and record.capture is not None:
            try:
                saved = record.capture.download(download)
            except Exception as e:
                err_console.print(f"[yellow]capture {record.capture.file} not saved:[/yellow] {e}")
        if json_output:
            print_json_line(record.model_dump(mode="json"))
            if saved is not None:
                err_console.print(f"saved {saved}")
        else:
            rows.append(
                {
                    "time": _time(record),
                    "kind": record.kind,
                    "session": record.session or "",
                    "record": summary(record) + (f" -> {saved}" if saved is not None else ""),
                }
            )
        if stopped:
            break
    if json_output:
        return
    print_table(rows, columns=["time", "kind", "session", "record"], title=title, no_wrap=("time",))
    if stopped:
        console.print(
            f"[dim]The first {cap} records: --limit 0 for every one, or narrow --since.[/dim]"
        )


def live_option() -> Any:
    return typer.Option("--live", "--follow", "-f", help=LIVE_HELP)
