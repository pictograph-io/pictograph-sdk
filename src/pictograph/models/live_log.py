"""The Live log's record - one shape for every record of a deployment's or a workflow's log, read as it arrives
(``live=True``) or page by page by :meth:`pictograph.resources.deployments.Deployments.logs` and
:meth:`pictograph.resources.workflows.Workflows.logs`, and carried by the ``deployment.frames`` and
``capture.saved`` webhooks.

A record is one thing the log holds: a ``frame`` (a model's answer on one frame - its annotations), an ``event`` (a
workflow's trigger firing: a line crossed, a zone entered), a ``capture`` (an image a stream captured, with a URL to
download it), ``stats`` (a stream's numbers each second), ``status`` (a stream's state), or - on a live connection
alone - a ``gap`` (records the connection could not carry, and the cursor to read them from).

:attr:`LogRecord.cursor` is the record's identity and the place a read resumes after it: the same on a history page
and in a webhook - deduplicate on it there. A live connection's cursor carries that identity and then where the
record arrived (what resumes the connection exactly); a live connection never gives a record twice.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from pictograph.models.annotation import Annotation

#: What a record is.
LogKind = Literal["frame", "event", "capture", "stats", "status", "gap"]

#: Every kind a history page holds (a ``gap`` belongs to a live connection).
LOG_KINDS: tuple[str, ...] = ("frame", "event", "capture", "stats", "status")

#: How long a capture's ``url`` works after the page or delivery that carried it.
CAPTURE_URL_MINUTES = 15


class LogSource(BaseModel):
    """Where a record came from."""

    model_config = ConfigDict(extra="ignore")

    device: str | None = Field(
        default=None,
        description="cloud (a GPU in the cloud), or edge (the viewer's own device: a browser or an iPhone).",
    )
    kind: str | None = Field(
        default=None,
        description="What was run: stream (rtsp, whip or whep when known), image, video, dataset or export.",
    )
    name: str | None = Field(default=None, description="The source's name, as the engine named it.")


class LogModel(BaseModel):
    """The model that answered a frame."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    name: str | None = None
    task: str | None = Field(default=None, description="object_detection, keypoint_detection, ...")
    version: int | None = Field(default=None, description="The model version a deployment serves.")
    precision: str | None = Field(default=None, description="fp32 or fp16.")


class LogFrame(BaseModel):
    """The frame a record is about."""

    model_config = ConfigDict(extra="ignore")

    number: int | None = Field(
        default=None, description="Its number in its stream or video (the first is 0)."
    )
    width: int | None = None
    height: int | None = None
    media_time: float | None = Field(
        default=None, description="Its time in its stream or video, in seconds."
    )
    latency_ms: float | None = Field(
        default=None, description="From its capture to its answer, in milliseconds."
    )


class LogTag(BaseModel):
    """A classifier's label on a frame."""

    model_config = ConfigDict(extra="ignore")

    name: str
    confidence: float | None = None


class LogEvent(BaseModel):
    """A workflow's trigger firing (a record of kind ``event``)."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = Field(default=None, description="The event's id (its captures name it).")
    trigger: str | None = Field(default=None, description="The trigger that fired.")
    kind: str | None = Field(
        default=None, description="crossing, entry, exit, dwell, appearance or condition."
    )
    name: str | None = Field(
        default=None, description="What a person reads, as the app's Logs tab names it."
    )
    values: dict[str, float | int] = Field(
        default_factory=dict, description="The trigger's rule values (a count a whole number)."
    )
    text: str | None = Field(default=None, description='Its values in words ("1 crossed").')
    captured: bool = Field(
        default=False, description="Whether it took an image (a capture record)."
    )


class LogCapture(BaseModel):
    """An image a stream captured (a record of kind ``capture``): download it with :meth:`download`."""

    model_config = ConfigDict(extra="ignore")

    file: str | None = Field(default=None, description="The image's file name.")
    url: str | None = Field(
        default=None,
        description="The image, signed for 15 minutes (read the log again for a fresh one).",
    )
    trigger: str | None = Field(
        default=None, description="What took it (an event's trigger, or Passive capture)."
    )
    event: str | None = Field(default=None, description="The event it was taken on.")
    image: str | None = Field(
        default=None, description="Its name in its dataset, once saved there."
    )
    dataset: str | None = Field(default=None, description="The dataset it was saved into.")
    directory: str | None = Field(default=None, description="The folder it was saved into.")

    def _target(self, path: str | Path) -> Path:
        out = Path(path).expanduser()
        if out.is_dir():
            out = out / (self.file or "capture.jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def _url(self) -> str:
        if not self.url:
            raise ValueError("This capture has no URL to download (its image could not be signed).")
        return self.url

    def download(self, path: str | Path = ".") -> Path:
        """Save the image to ``path`` - a directory (it keeps its file name) or a file path - and return where.

        The URL is signed: no API key is sent. It works for 15 minutes after the page that carried it; after that,
        read the log again for a fresh one.
        """
        url = self._url()
        out = self._target(path)
        tmp = out.with_name(out.name + ".part")
        try:
            with (
                httpx.Client(timeout=httpx.Timeout(30.0, read=120.0)) as storage,
                storage.stream("GET", url) as r,
            ):
                if r.status_code >= 300:
                    r.read()
                    raise _download_error(r.status_code)
                with tmp.open("wb") as fh:
                    for chunk in r.iter_bytes():
                        fh.write(chunk)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        tmp.replace(out)
        return out

    async def download_async(self, path: str | Path = ".") -> Path:
        """:meth:`download`, awaited."""
        url = self._url()
        out = self._target(path)
        tmp = out.with_name(out.name + ".part")
        try:
            async with (
                httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=120.0)) as storage,
                storage.stream("GET", url) as r,
            ):
                if r.status_code >= 300:
                    await r.aread()
                    raise _download_error(r.status_code)
                with tmp.open("wb") as fh:
                    async for chunk in r.aiter_bytes():
                        fh.write(chunk)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        tmp.replace(out)
        return out


def _download_error(status: int) -> Exception:
    from pictograph.exceptions import ApiError

    if status in (400, 403):
        return ApiError(
            f"The capture's URL was refused (HTTP {status}): it lapsed - read the log again for a fresh one.",
            status_code=status,
        )
    if status == 404:
        return ApiError(
            "The capture's image is not stored (its upload did not land).", status_code=status
        )
    return ApiError(f"The capture could not be downloaded: HTTP {status}", status_code=status)


class LogStats(BaseModel):
    """A stream's numbers (a record of kind ``stats``), sent each second while it runs."""

    model_config = ConfigDict(extra="ignore")

    frames: int | None = Field(default=None, description="Frames it has taken.")
    inferred: int | None = Field(default=None, description="Frames its model answered.")
    detections: int | None = Field(default=None, description="Annotations its model found.")
    events: int | None = Field(default=None, description="Events it fired.")
    fps: float | None = Field(default=None, description="Frames a second.")
    media_time: float | None = Field(
        default=None, description="How far into its stream, in seconds."
    )
    latency_ms: dict[str, dict[str, float | int]] | None = Field(
        default=None,
        description="Latency percentiles (p50, p90, p99, max, n) of total, engine and inference.",
    )
    measures: dict[str, Any] | None = Field(
        default=None, description="Each line's and zone's counts."
    )


class LogStatus(BaseModel):
    """A stream's state (a record of kind ``status``): ``ended`` when its source ended."""

    model_config = ConfigDict(extra="ignore")

    state: str
    reason: str | None = None


class LogGap(BaseModel):
    """Records a live connection could not carry (a record of kind ``gap``): it fell further behind than the live
    window holds, a batch never reached it, or the log was not streamed for a while. They are in the history,
    between the last record the connection carried and the gap's ``time``::

        client.deployments.logs("loading-dock", since=last.time, until=gap_record.time)

    (``backfill`` is the same place as a cursor, for the REST API's ``cursor``.)"""

    model_config = ConfigDict(extra="ignore")

    missed: int | None = Field(
        default=None, description="How many of the log's rows it stands in for, when known."
    )
    backfill: str | None = Field(
        default=None, description="The cursor to read them from in the history."
    )


class LogRecord(BaseModel):
    """One record of a Live log: every kind in one shape; the fields of another kind are ``None``.

    ``annotations`` are the frame's Pictograph annotations (:mod:`pictograph.models.annotation`, the class under
    ``name``); a classifier's labels are ``tags``. ``session`` is what was started: a stream's id - or, for a
    workflow's file run, the run's id. ``time`` is the frame's capture time when its source has a clock, else when
    the engine answered it.
    """

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    cursor: str | None = Field(
        default=None, description="The record's identity, and where a read resumes after it."
    )
    kind: LogKind
    time: datetime | None = None
    logged_at: datetime | None = Field(default=None, description="When the log stored it.")
    deployment: str | None = None
    workflow: str | None = None
    session: str | None = Field(default=None, description="A stream's id, or a workflow run's.")
    run: str | None = Field(default=None, description="The engine run that wrote it.")
    seq: int | None = Field(default=None, description="Its batch in its run.")
    index: int | None = Field(default=None, description="Its place in its batch.")
    source: LogSource | None = None
    model: LogModel | None = None
    frame: LogFrame | None = None
    annotations: list[Annotation] = Field(default_factory=list)
    tags: list[LogTag] = Field(default_factory=list)
    event: LogEvent | None = None
    capture: LogCapture | None = None
    stats: LogStats | None = None
    status: LogStatus | None = None
    gap: LogGap | None = None


class LogPage(BaseModel):
    """One page of a Live log, as the API answers it (the SDK's ``logs()`` reads every page for you)."""

    model_config = ConfigDict(extra="ignore")

    records: list[LogRecord] = Field(default_factory=list)
    next_cursor: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    frames_logged: bool = True
    deployment: str | None = None
    workflow: str | None = None


__all__ = [
    "CAPTURE_URL_MINUTES",
    "LOG_KINDS",
    "LogCapture",
    "LogEvent",
    "LogFrame",
    "LogGap",
    "LogKind",
    "LogModel",
    "LogPage",
    "LogRecord",
    "LogSource",
    "LogStats",
    "LogStatus",
    "LogTag",
]
