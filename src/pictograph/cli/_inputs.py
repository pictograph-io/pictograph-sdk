"""What the ``predict`` commands run on: images, a video's frames, or a dataset's images.

One reading for ``models predict`` and ``deployments predict``, so the same line
works against a model on this machine, a model on Pictograph and a deployment.

**Images** (the positional arguments):

- a **path** to an image file;
- a **glob** (``frames/*.jpg``, ``shots/**/*.png``) - expanded HERE, sorted, so a
  quoted pattern behaves the same in every shell and on Windows. A file that exists
  is always that file, even when its name contains ``[``, ``*`` or ``?``;
- an **http(s) URL**, fetched on this machine;
- **``-``**, the image bytes on standard input (``curl ... | pictograph models
  predict "Shelf Detector" -``).

**A video** (``--video``, repeatable; a path, an http(s) URL, or ``-``): decoded
on this machine and sampled at ``--sample-fps`` frames per second with the SAME
rule a workflow uses (:mod:`pictograph._video`, held to the workflow runner by the
platform's tests), so ``frame`` and ``t`` on a line name the frame a workflow over the
same file would have run.

**A dataset** (``--dataset``, optionally ``--directory``): the images of one of
your datasets, read and never written - predictions are printed, nothing is saved.
A dataset in cold storage is refused before anything runs (its images are not read
while it is cold; restore it first), and so is a folder with no images.

Every source becomes the same lazy stream of :class:`Item`, and :func:`emit` prints
one JSON line per item as soon as it is done. Everything that can be checked before
the first prediction is checked first: a pattern that matches nothing, a path that
does not exist, a video OpenCV cannot open, a cold or empty dataset - so a mistake
stops the command up front instead of half way through a batch.
"""

from __future__ import annotations

import contextlib
import glob
import itertools
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import httpx
import typer

from pictograph._image_input import FETCH_TIMEOUT_SECONDS, is_url
from pictograph._video import DEFAULT_SAMPLE_FPS, MIN_SAMPLE_FPS, sample_frames
from pictograph.cli._errors import ExitCode, classify
from pictograph.exceptions import (
    AuthError,
    ForbiddenError,
    NetworkError,
    PaymentRequiredError,
    RateLimitError,
)
from pictograph.resources._resolve import looks_like_id

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from pictograph import Client
    from pictograph.models.image import Image

STDIN = "-"
_GLOB_CHARS = frozenset("*?[")

HELP = (
    "One or more images: a file path, a glob ('frames/*.jpg'), an http(s) URL, "
    "or - for image bytes on stdin. Or pass --video / --dataset instead."
)
VIDEO_HELP = (
    "Run on a video's frames instead of images: a path, an http(s) URL, or - (stdin). "
    "Repeat for several videos. Decoded on this machine, so it needs OpenCV: "
    # `\\[` keeps Rich (which renders --help) from reading `[inference]` as markup.
    "pip install 'pictograph\\[inference]'. Each line carries the frame's index and "
    "time (t, seconds)."
)
SAMPLE_FPS_HELP = (
    "With --video: frames per second to run (default 1). The same frames a "
    "workflow samples from the same file; the video's own frame rate is the ceiling."
)
DATASET_HELP = (
    "Run on the images of one of your datasets, read-only: predictions are printed, "
    "nothing is saved (to save them as draft labels: pictograph auto-annotate dataset). "
    "A dataset in cold storage must be restored first."
)
DIRECTORY_HELP = (
    "With --dataset: only the images in this folder ('/val'; '/' is the root). "
    "Exactly this folder, not its subfolders."
)
LIMIT_HELP = "Stop after this many images or frames."

# Frames handed to a HOSTED model or a deployment travel as JPEG at this quality
# (lossless PNG of a 4K frame overshoots the hosted 25 MB upload cap). A LOCAL
# model gets the decoded frame itself, so nothing is re-encoded on that path.
FRAME_JPEG_QUALITY = 95

# An error that would fail EVERY remaining input the same way: the key, its role,
# the credit balance, or capacity that is spent for the day. Printing it once per
# image only buries it, so a batch stops at the first one.
STOPS_THE_BATCH: tuple[type[Exception], ...] = (
    AuthError,
    ForbiddenError,
    PaymentRequiredError,
    RateLimitError,
)

FrameFormat = Literal["array", "jpeg"]

# Image rows listed per request while reading a dataset.
_DATASET_PAGE = 200


@dataclass(frozen=True)
class Item:
    """One prediction to make.

    ``fields`` open its JSON line - ``input`` always, plus ``frame`` and ``t`` for
    a video frame or ``image_id`` for a dataset image. ``load`` produces what the
    command hands the model; it runs inside the per-item error handling, so an
    image that cannot be read becomes that item's error line, not a crash.
    """

    fields: dict[str, Any]
    load: Callable[[], Any]


@dataclass(frozen=True)
class Source:
    """What a ``predict`` command was asked to run on, checked and ready to open.

    Built by :func:`parse_source` with no network calls; :func:`open_items` does
    the checks that need the API (a dataset's storage state and first page).
    """

    kind: Literal["images", "video", "dataset"]
    values: list[str] = field(default_factory=list)
    stdin: bytes | None = None
    sample_fps: float = DEFAULT_SAMPLE_FPS
    dataset: str | None = None
    directory: str | None = None
    limit: int | None = None

    @property
    def single_image(self) -> bool:
        """One plain image: the commands print its result as JSON or a table."""
        return self.kind == "images" and len(self.values) == 1


# ───────────────────────── images ─────────────────────────


def expand_inputs(values: list[str]) -> list[str]:
    """The images to run, in order: globs expanded and sorted, the rest as typed.

    Raises:
        typer.BadParameter: Nothing was given, ``-`` appears twice, a glob matches
            no file, or a path does not exist.
    """
    if not values:
        raise typer.BadParameter(
            "Give at least one image (or --video / --dataset).", param_hint="IMAGES"
        )
    if values.count(STDIN) > 1:
        raise typer.BadParameter(
            "- (stdin) can only be given once: there is one stdin.", param_hint="IMAGES"
        )
    out: list[str] = []
    for value in values:
        if value == STDIN or is_url(value):
            out.append(value)
        elif Path(value).expanduser().is_file():
            # A file that exists is that file, whatever is in its name:
            # `frame[01].jpg` is a name before it is a pattern (read as a pattern
            # it would match `frame0.jpg` instead, and run the wrong image).
            out.append(value)
        elif _GLOB_CHARS & set(value):
            matches = sorted(
                p
                # glob.glob, not Path.glob: the pattern may be absolute or start with ~.
                for p in glob.glob(str(Path(value).expanduser()), recursive=True)  # noqa: PTH207
                if Path(p).is_file()
            )
            if not matches:
                raise typer.BadParameter(f"No file matches {value!r}.", param_hint="IMAGES")
            out.extend(matches)
        else:
            raise typer.BadParameter(f"No such file: {value}", param_hint="IMAGES")
    return out


def read_stdin(param_hint: str = "IMAGES", *, example: str | None = None) -> bytes:
    """The bytes piped to this command (an image, or a video with ``--video -``)."""
    data = sys.stdin.buffer.read()
    if not data:
        example = example or (
            "`cat clip.mp4 | pictograph models predict <model> --video -`"
            if param_hint == "--video"
            else "`cat photo.jpg | pictograph models predict <model> -`"
        )
        raise typer.BadParameter(
            f"- was given but nothing arrived on stdin. Pipe it in: {example}.",
            param_hint=param_hint,
        )
    return data


def load(value: str, stdin: bytes | None) -> str | Path | bytes:
    """What to hand the SDK for one expanded image input."""
    if value == STDIN:
        if stdin is None:  # pragma: no cover - expand_inputs + the caller guarantee it
            raise typer.BadParameter("- was given but stdin was not read.", param_hint="IMAGES")
        return stdin
    return value if is_url(value) else Path(value).expanduser()


# ───────────────────────── the source, checked ─────────────────────────


def canonical_directory(value: str) -> str:
    """A folder as the API stores it: ``/`` for the root, else ``/a/b``.

    ``val``, ``/val`` and ``/val/`` are one folder (the backend's own
    ``canonical_virtual_directory``); a ``..`` segment or a NUL byte is refused.
    """
    cleaned = value.strip().strip("/")
    if "\x00" in cleaned or ".." in cleaned.split("/"):
        raise typer.BadParameter(f"Not a folder path: {value!r}", param_hint="--directory")
    return f"/{cleaned}" if cleaned else "/"


def parse_source(
    images: list[str] | None,
    *,
    video: list[str] | None = None,
    sample_fps: float | None = None,
    dataset: str | None = None,
    directory: str | None = None,
    limit: int | None = None,
) -> Source:
    """Check what was asked for, with no network calls, and read stdin if named.

    Exactly one kind of input per run: images, ``--video``, or ``--dataset``.

    Raises:
        typer.BadParameter: Two kinds at once, an option that belongs to another
            kind, or any of :func:`expand_inputs`' refusals; for a video, OpenCV
            not installed, a path that does not exist, or a file it cannot open.
    """
    given = [name for name, value in (("IMAGES", images), ("--video", video),
                                      ("--dataset", dataset)) if value]  # fmt: skip
    if len(given) > 1:
        raise typer.BadParameter(
            f"Give images, --video or --dataset - one kind per run (got {' and '.join(given)}).",
            param_hint=given[1],
        )
    if directory is not None and not dataset:
        raise typer.BadParameter("--directory needs --dataset.", param_hint="--directory")
    if sample_fps is not None and not video:
        raise typer.BadParameter("--sample-fps needs --video.", param_hint="--sample-fps")

    if dataset:
        return Source(
            kind="dataset",
            dataset=dataset,
            directory=canonical_directory(directory) if directory is not None else None,
            limit=limit,
        )
    if video:
        return _video_source(video, sample_fps, limit)
    values = expand_inputs(images or [])
    stdin = read_stdin() if STDIN in values else None
    return Source(kind="images", values=values, stdin=stdin, limit=limit)


def _video_source(videos: list[str], sample_fps: float | None, limit: int | None) -> Source:
    if videos.count(STDIN) > 1:
        raise typer.BadParameter(
            "- (stdin) can only be given once: there is one stdin.", param_hint="--video"
        )
    cv2 = _cv2()
    for value in videos:
        if value == STDIN or is_url(value):
            continue  # fetched when its turn comes; a failure is that video's line
        path = Path(value).expanduser()
        if not path.is_file():
            raise typer.BadParameter(f"No such file: {value}", param_hint="--video")
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise typer.BadParameter(
                    f"Could not open {value} as a video - the file may be corrupt, "
                    "truncated, or an unsupported codec or container.",
                    param_hint="--video",
                )
        finally:
            capture.release()
    stdin = read_stdin("--video") if STDIN in videos else None
    return Source(
        kind="video",
        values=list(videos),
        stdin=stdin,
        sample_fps=DEFAULT_SAMPLE_FPS if sample_fps is None else sample_fps,
        limit=limit,
    )


def _cv2() -> Any:
    try:
        import cv2
    except ImportError:
        raise typer.BadParameter(
            "Reading a video needs OpenCV, which comes with the local inference "
            "extra: pip install 'pictograph[inference]'. (A workflow reads video "
            "on Pictograph instead: pictograph workflows invoke <workflow> --file clip.mp4.)",
            param_hint="--video",
        ) from None
    return cv2


# ───────────────────────── the items ─────────────────────────


def open_items(
    source: Source, client: Client | None = None, *, frames: FrameFormat = "array"
) -> Iterator[Item]:
    """The lazy stream of items to run, after the checks that need the API.

    ``frames`` is how a video frame is handed over: ``"array"`` (the decoded BGR
    numpy array - a local model) or ``"jpeg"`` (bytes - a hosted model or a
    deployment, which take an uploaded image). A dataset needs ``client``.

    Raises:
        typer.BadParameter: A dataset in cold storage, or a dataset / folder with
            no images.
    """
    if source.kind == "dataset":
        if client is None:  # pragma: no cover - every caller passes one for a dataset
            raise typer.BadParameter("Reading a dataset needs an account API key.")
        items: Iterator[Item] = _dataset_items(source, client)
    elif source.kind == "video":
        items = _video_items(source, frames)
    else:
        items = (_image_item(value, source.stdin) for value in source.values)
    if source.limit is not None:
        items = _limited(items, source.limit)
    return items


def _limited(items: Iterator[Item], limit: int) -> Iterator[Item]:
    """The first ``limit`` items, closing the source when they are done.

    A bare ``islice`` would leave the source suspended (a video still open, a
    dataset's temp folder still on disk) until garbage collection.
    """
    try:
        yield from itertools.islice(items, limit)
    finally:
        close = getattr(items, "close", None)
        if callable(close):
            close()


def _image_item(value: str, stdin: bytes | None) -> Item:
    return Item({"input": value}, lambda: load(value, stdin))


def _video_items(source: Source, frames: FrameFormat) -> Iterator[Item]:
    cv2 = _cv2()
    for video in source.values:
        with tempfile.TemporaryDirectory(prefix="pictograph-video-") as tmp:
            try:
                path = _video_path(video, source.stdin, Path(tmp))
            except Exception as exc:  # this video's error line; the next one still runs
                yield Item({"input": video}, _raiser(exc))
                continue
            capture = cv2.VideoCapture(path)
            try:
                if not capture.isOpened():
                    yield Item(
                        {"input": video},
                        _raiser(
                            ValueError(
                                f"Could not open {video} as a video - the file may be "
                                "corrupt, truncated, or an unsupported codec or container."
                            )
                        ),
                    )
                    continue
                for index, seconds, frame in sample_frames(capture, source.sample_fps):
                    yield Item(
                        {"input": video, "frame": index, "t": round(seconds, 4)},
                        _frame_loader(frame, frames),
                    )
            finally:
                capture.release()


def _frame_loader(frame: Any, frames: FrameFormat) -> Callable[[], Any]:
    if frames == "array":
        return lambda: frame

    def encode() -> bytes:
        import cv2

        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY])
        if not ok:
            raise ValueError("Could not encode the frame as JPEG.")
        return bytes(buffer.tobytes())

    return encode


def _video_path(video: str, stdin: bytes | None, tmp: Path) -> str:
    """A local file OpenCV can open: the path itself, or the URL / stdin spooled."""
    if video == STDIN:
        if stdin is None:  # pragma: no cover - parse_source read it
            raise ValueError("- was given but stdin was not read.")
        target = tmp / "stdin.video"
        target.write_bytes(stdin)
        return str(target)
    if is_url(video):
        # Spooled whole rather than streamed into OpenCV: a downloaded file is
        # seekable and reports its frame rate, so the sampling matches a local file.
        target = tmp / "url.video"
        try:
            with (
                httpx.stream(
                    "GET", video, timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=True
                ) as response,
                target.open("wb") as out,
            ):
                if response.status_code != 200:
                    raise ValueError(
                        f"Could not fetch the video at {video}: the server answered "
                        f"HTTP {response.status_code}."
                    )
                for chunk in response.iter_bytes():
                    out.write(chunk)
        except httpx.HTTPError as exc:
            raise NetworkError(f"Could not fetch the video at {video}: {exc}") from exc
        return str(target)
    return str(Path(video).expanduser())


def _dataset_items(source: Source, client: Client) -> Iterator[Item]:
    dataset = str(source.dataset)
    status = client.datasets.storage_status(dataset)
    if status.storage_class != "standard" or status.storage_state != "idle":
        # The platform's own rule (`utils/dataset_storage.is_dataset_warm`): a
        # dataset's images are read only while it is standard AND idle.
        name = _quoted(dataset)
        if status.storage_class == "standard":
            why = (
                f"is moving to cold storage, and its images are not read while it "
                f"moves (pictograph datasets storage {name} shows its state)"
            )
        elif status.storage_state == "idle":
            why = (
                "is in cold storage, and its images are not read while it is: "
                f"restore it first (pictograph datasets restore {name})"
            )
        else:
            why = (
                "is being restored from cold storage: run this again when it "
                f"finishes (pictograph datasets storage {name} shows its state)"
            )
        raise typer.BadParameter(f"Dataset {dataset!r} {why}.", param_hint="--dataset")
    # A --limit smaller than a page asks the API for just that many rows.
    page_size = min(_DATASET_PAGE, source.limit) if source.limit else _DATASET_PAGE
    pager = iter(
        client.images.iter(
            dataset, directory_path=source.directory, page_size=page_size, max_total=source.limit
        )
    )
    first = next(pager, None)
    if first is None:
        where = f" in {source.directory}" if source.directory else ""
        hint = " (only that folder, not its subfolders)" if source.directory else ""
        raise typer.BadParameter(
            f"Dataset {dataset!r} has no images{where}{hint}.", param_hint="--dataset"
        )
    return _dataset_stream(client, dataset, itertools.chain([first], pager))


def _dataset_stream(client: Client, dataset: str, images: Iterator[Image]) -> Iterator[Item]:
    with tempfile.TemporaryDirectory(prefix="pictograph-dataset-") as tmp:
        for number, image in enumerate(images):
            address = image_address(image)
            yield Item(
                {"input": address, "image_id": image.id},
                _image_loader(client, dataset, image, address, Path(tmp) / f"{number}.image"),
            )


def image_address(image: Image) -> str:
    """``val/x.jpg`` - an image's address inside its dataset (root: ``x.jpg``)."""
    directory = (image.directory_path or "").strip("/")
    return f"{directory}/{image.filename}" if directory else image.filename


def _image_loader(
    client: Client, dataset: str, image: Image, address: str, target: Path
) -> Callable[[], bytes]:
    def fetch() -> bytes:
        # By its address: one request, no lookup. A root file whose NAME is a bare
        # uuid would read as an id, so that one goes by its real id instead. The
        # local name is ours (`<n>.image`), never the stored filename.
        ref = image.id if looks_like_id(address) else address
        path = client.images.download(dataset, ref, target)
        try:
            return path.read_bytes()
        finally:
            with contextlib.suppress(OSError):
                path.unlink()

    return fetch


def _raiser(exc: Exception) -> Callable[[], Any]:
    def fail() -> Any:
        raise exc

    return fail


def _quoted(value: str) -> str:
    return f"'{value}'" if any(c.isspace() for c in value) else value


# ───────────────────────── running them ─────────────────────────


def emit(items: Iterable[Item], run: Callable[[Any], Any], *, stop_hint: str = "") -> None:
    """Run ``run`` over every item and print one JSON object per line.

    ``{...fields, "result": ...}`` for an item that worked and
    ``{...fields, "error": ..., "error_type": ..., "error_code": ...}`` for one that
    did not, in order, each line written as soon as its item is done. ``error_code``
    is the code the failure has in the CLI's error table (``not_found``,
    ``invalid_input``, ...). A batch never stops at a bad image; it stops at an
    error that would fail every remaining one the same way (:data:`STOPS_THE_BATCH`),
    with a note on stderr, and exits with THAT error's code (2 auth, 3 forbidden,
    7 credits, 8 rate limited). Otherwise the exit code is 1 when any item failed.
    """
    failed = 0
    done = 0
    stream = iter(items)
    try:
        for item in stream:
            line = dict(item.fields)
            try:
                line["result"] = run(item.load())
            except Exception as exc:
                failed += 1
                failure = classify(exc)
                line["error"] = str(exc)
                line["error_type"] = type(exc).__name__
                line["error_code"] = failure.code if failure is not None else "error"
                typer.echo(json.dumps(line, default=str))
                if isinstance(exc, STOPS_THE_BATCH):
                    typer.echo(
                        f"Stopped after {done + failed} input(s): {type(exc).__name__} would "
                        f"fail every one after it the same way, so the rest were not run.{stop_hint}",
                        err=True,
                    )
                    code = failure.exit_code if failure is not None else ExitCode.FAILED
                    raise typer.Exit(code=int(code)) from None
                continue
            done += 1
            typer.echo(json.dumps(line, default=str))
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()  # releases a video capture / a dataset's temp folder now
    if failed:
        raise typer.Exit(code=int(ExitCode.FAILED))


__all__ = [
    "DATASET_HELP",
    "DIRECTORY_HELP",
    "HELP",
    "LIMIT_HELP",
    "MIN_SAMPLE_FPS",
    "SAMPLE_FPS_HELP",
    "STDIN",
    "STOPS_THE_BATCH",
    "VIDEO_HELP",
    "Item",
    "Source",
    "canonical_directory",
    "emit",
    "expand_inputs",
    "image_address",
    "load",
    "open_items",
    "parse_source",
    "read_stdin",
]
