"""How the CLI fails: one exit code per kind of failure, and a JSON error on stderr.

An agent driving Pictograph through the CLI cannot branch on red text, so every
failure maps to one exit code. The table is a contract, documented in ``docs/cli.md``
("Exit codes") and in the bundled skill:

- 0 ``OK`` - success.
- 1 ``FAILED`` - anything else: a job ended failed or cancelled, some inputs of a
  batch failed, a confirmation was declined.
- 2 ``AUTH`` - no API key, or the key was refused (401).
- 3 ``FORBIDDEN`` - the key's role or the plan does not allow it (403).
- 4 ``NOT_FOUND`` - no such dataset, image, model, ... (404).
- 5 ``CONFLICT`` - conflicts with the current state: it exists already, a dataset
  in cold storage, a job not finished (409).
- 6 ``INVALID`` - the input is wrong: a bad flag or value, a missing or malformed
  file, or the API refused the request (400 / 422).
- 7 ``PAYMENT`` - not enough compute credit (402).
- 8 ``RATE_LIMITED`` - too many requests; ``retry_after`` says when to retry (429).
- 9 ``UNAVAILABLE`` - a server error, the network, a request that timed out, or a
  stream with no frame for its ``--idle-timeout`` (``stream_stalled``).
- 10 ``STILL_RUNNING`` - a ``--wait`` ran out of time; the job keeps running.
- 130 ``INTERRUPTED`` - Ctrl-C.

With ``--json`` anywhere on the command line, or ``PICTOGRAPH_OUTPUT=json`` in the
environment, a failure is also ONE line of JSON, the last line on stderr: an object
with one key, ``error``, holding ``code``, ``message``, ``exit_code``, ``type`` (the
exception class), ``status`` (HTTP), ``retry_after`` (seconds), ``details``, ``fix``,
``docs_url`` and ``request_id``. Every key is always present (``null`` when it does
not apply). ``code`` is the API's own error code when it sent one
(``dataset_cold``, ``insufficient_credits``, ...) and otherwise a stable code for the
kind of failure (``not_found``, ``usage_error``, ``job_failed``, ``wait_timeout``, ...).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import typer
from typer.core import TyperGroup, TyperOption

from pictograph.cli._format import print_error
from pictograph.exceptions import (
    ApiError,
    AuthError,
    ConfigurationError,
    ConflictError,
    ForbiddenError,
    NetworkError,
    NotFoundError,
    PaymentRequiredError,
    PictographError,
    PollTimeoutError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
    StreamStalledError,
    UnsafeCheckpointError,
    ValidationError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

OUTPUT_ENV = "PICTOGRAPH_OUTPUT"
"""``PICTOGRAPH_OUTPUT=json``: every command with ``--json`` prints JSON, and every
failure prints a JSON error on stderr."""


class ExitCode(IntEnum):
    """The CLI's exit codes, one per kind of failure (see the module docstring)."""

    OK = 0
    FAILED = 1
    AUTH = 2
    FORBIDDEN = 3
    NOT_FOUND = 4
    CONFLICT = 5
    INVALID = 6
    PAYMENT = 7
    RATE_LIMITED = 8
    UNAVAILABLE = 9
    STILL_RUNNING = 10
    INTERRUPTED = 130


class CliError(PictographError):
    """A failure the CLI itself detects, carrying the exit code and code it maps to.

    Raise it from a command instead of printing an error and ``typer.Exit``-ing, so
    the failure renders the same way as every other (text, or JSON under ``--json``)
    and exits with the code its kind has in the table.
    """

    def __init__(
        self,
        message: str,
        *,
        exit_code: ExitCode = ExitCode.FAILED,
        code: str = "error",
        fix: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, fix=fix)
        self.exit_code = exit_code
        self.code = code
        self.details = dict(details) if details is not None else None


# Typer >= 0.27 vendors click, so the click classes are reached through Typer's own
# exports rather than ``import click`` (which may not even be installed).
_CLICK_EXCEPTION: type[Exception] = next(
    c for c in typer.BadParameter.__mro__ if c.__name__ == "ClickException"
)
_USAGE_ERROR: type[Exception] = next(
    c for c in typer.BadParameter.__mro__ if c.__name__ == "UsageError"
)

# The kind of failure for each SDK error class. Order matters: a subclass is listed
# before its base (RequestTimeoutError before NetworkError).
_BY_CLASS: tuple[tuple[type[PictographError], ExitCode, str], ...] = (
    (AuthError, ExitCode.AUTH, "unauthorized"),
    (ForbiddenError, ExitCode.FORBIDDEN, "forbidden"),
    (NotFoundError, ExitCode.NOT_FOUND, "not_found"),
    (ConflictError, ExitCode.CONFLICT, "conflict"),
    (ValidationError, ExitCode.INVALID, "validation_error"),
    (PaymentRequiredError, ExitCode.PAYMENT, "payment_required"),
    (RateLimitError, ExitCode.RATE_LIMITED, "rate_limited"),
    (ServerError, ExitCode.UNAVAILABLE, "server_error"),
    (PollTimeoutError, ExitCode.STILL_RUNNING, "wait_timeout"),
    (StreamStalledError, ExitCode.UNAVAILABLE, "stream_stalled"),
    (RequestTimeoutError, ExitCode.UNAVAILABLE, "request_timeout"),
    (NetworkError, ExitCode.UNAVAILABLE, "network_error"),
    (ConfigurationError, ExitCode.INVALID, "configuration_error"),
    (UnsafeCheckpointError, ExitCode.FAILED, "unsafe_checkpoint"),
)

# A plain ``ApiError`` (a status the SDK has no subclass for) maps by its status.
_BY_STATUS: dict[int, tuple[ExitCode, str]] = {
    401: (ExitCode.AUTH, "unauthorized"),
    402: (ExitCode.PAYMENT, "payment_required"),
    403: (ExitCode.FORBIDDEN, "forbidden"),
    404: (ExitCode.NOT_FOUND, "not_found"),
    408: (ExitCode.UNAVAILABLE, "request_timeout"),
    409: (ExitCode.CONFLICT, "conflict"),
    410: (ExitCode.NOT_FOUND, "gone"),
    413: (ExitCode.INVALID, "payload_too_large"),
    423: (ExitCode.CONFLICT, "locked"),
    429: (ExitCode.RATE_LIMITED, "rate_limited"),
}


_CLI_NOT_FOUND_FIX = (
    "Names are case-sensitive. List what exists with the group's list command, e.g. "
    "`pictograph datasets list`, `pictograph models list`, `pictograph images list <dataset>`."
)


@dataclass(frozen=True)
class Failure:
    """One failure, classified: what the process exits with and what it reports."""

    exit_code: ExitCode
    code: str
    message: str
    type: str
    status: int | None = None
    retry_after: float | None = None
    details: Any = None
    fix: str | None = None
    docs_url: str | None = None
    request_id: str | None = None
    # How to print it for a person (the JSON form never uses this).
    human: str = field(default="", repr=False)

    def as_json(self) -> dict[str, Any]:
        """The ``{"error": {...}}`` object, every key present."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "exit_code": int(self.exit_code),
                "type": self.type,
                "status": self.status,
                "retry_after": self.retry_after,
                "details": self.details,
                "fix": self.fix,
                "docs_url": self.docs_url,
                "request_id": self.request_id,
            }
        }


def _api_details(exc: ApiError) -> Any:
    """``error.details`` from the API's envelope, plus what the SDK parsed out of it."""
    body = exc.response if isinstance(exc.response, dict) else {}
    if exc.code in ("job_failed", "job_cancelled"):
        # A wait_for_* helper raised with the job itself as the response.
        return {key: body.get(key) for key in ("id", "status") if body.get(key) is not None}
    envelope = body.get("error") if isinstance(body.get("error"), dict) else {}
    details: Any = envelope.get("details") if envelope else None
    if isinstance(exc, ValidationError) and exc.field_errors and not details:
        details = {"errors": exc.field_errors}
    if isinstance(exc, PaymentRequiredError):
        extra = {
            "required_micro_usd": exc.credit_cost,
            "remaining_micro_usd": exc.credits_remaining,
            "unit": exc.unit,
            "upgrade_url": exc.upgrade_url,
        }
        extra = {k: v for k, v in extra.items() if v is not None}
        if extra:
            details = {**(details if isinstance(details, dict) else {}), **extra}
    return details


def _classify_api(exc: ApiError) -> tuple[ExitCode, str]:
    for cls, exit_code, code in _BY_CLASS:
        if isinstance(exc, cls):
            return exit_code, code
    status = exc.status_code
    if status is not None and status in _BY_STATUS:
        return _BY_STATUS[status]
    if status is not None and 400 <= status < 500:
        return ExitCode.INVALID, "bad_request"
    if status is not None and status >= 500:
        return ExitCode.UNAVAILABLE, "server_error"
    return ExitCode.FAILED, "error"


def _pydantic_errors(exc: ValueError) -> list[dict[str, Any]] | None:
    """Per-field errors of a ``pydantic.ValidationError`` (a ValueError), else None."""
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return None
    try:
        found = errors(include_url=False, include_context=False, include_input=False)
    except TypeError:
        return None
    if not isinstance(found, list):
        return None
    return [
        {"loc": list(e.get("loc", ())), "msg": e.get("msg"), "type": e.get("type")}
        for e in found
        if isinstance(e, dict)
    ]


def _usage_param(exc: BaseException, ctx: Any) -> str | None:
    """The argument or option a usage error is about, as click names it in the message
    (``'--limit' / '-n'``, ``'name'``), or ``None`` when it is about the whole command."""
    hint = getattr(exc, "param_hint", None)
    if isinstance(hint, str) and hint:
        return hint
    option = getattr(exc, "option_name", None)  # NoSuchOption
    if isinstance(option, str) and option:
        return option
    param = getattr(exc, "param", None)
    get_hint = getattr(param, "get_error_hint", None)
    if callable(get_hint) and ctx is not None:
        try:
            found = get_hint(ctx)
        except Exception:  # pragma: no cover - a hint is a nicety, never a crash
            return None
        return found if isinstance(found, str) and found else None
    return None


def classify(exc: BaseException) -> Failure | None:
    """The :class:`Failure` an exception is, or ``None`` for one the CLI does not own.

    ``None`` means a bug (a ``KeyError``, an ``AttributeError``, ...): it keeps its
    traceback, because hiding it would hide the bug.
    """
    name = type(exc).__name__
    if isinstance(exc, CliError):
        return Failure(
            exc.exit_code,
            exc.code,
            exc.message,
            name,
            details=exc.details,
            fix=exc.fix,
            docs_url=exc.docs_url,
            human=str(exc),
        )
    if isinstance(exc, ApiError):
        exit_code, default_code = _classify_api(exc)
        if isinstance(exc, NotFoundError) and exc.fix == NotFoundError.fix:
            # The SDK's default advice names a Python method; say what to RUN instead.
            exc.fix = _CLI_NOT_FOUND_FIX
        return Failure(
            exit_code,
            exc.code or default_code,
            exc.message,
            name,
            status=exc.status_code,
            retry_after=exc.retry_after if isinstance(exc, RateLimitError) else None,
            details=_api_details(exc),
            fix=exc.fix,
            docs_url=exc.docs_url,
            request_id=exc.request_id,
            human=str(exc),
        )
    if isinstance(exc, PictographError):
        exit_code, code = ExitCode.FAILED, "error"
        for cls, mapped_exit, mapped_code in _BY_CLASS:
            if isinstance(exc, cls):
                exit_code, code = mapped_exit, mapped_code
                break
        details = (
            {"quiet_s": round(exc.quiet_s, 3), "alive": exc.alive}
            if isinstance(exc, StreamStalledError)
            else None
        )
        return Failure(
            exit_code,
            code,
            exc.message,
            name,
            details=details,
            fix=exc.fix,
            docs_url=exc.docs_url,
            human=str(exc),
        )
    if name == "NoArgsIsHelpError":
        # `pictograph datasets` with no subcommand: the help was already printed.
        ctx = getattr(exc, "ctx", None)
        commands = getattr(getattr(ctx, "command", None), "commands", None)
        return Failure(
            ExitCode.INVALID,
            "missing_command",
            "Missing command." if commands else "Missing arguments.",
            name,
            details={"commands": sorted(commands)} if commands else None,
            fix="Add --help to the command to see what it takes.",
        )
    if isinstance(exc, _USAGE_ERROR):
        message = exc.format_message() if hasattr(exc, "format_message") else str(exc)
        ctx = getattr(exc, "ctx", None)
        path = getattr(ctx, "command_path", None)
        param = _usage_param(exc, ctx)
        return Failure(
            ExitCode.INVALID,
            "usage_error",
            message,
            name,
            details={"param": param} if param else None,
            fix=f"Run `{path} --help` for its arguments and options." if path else None,
        )
    if isinstance(exc, _CLICK_EXCEPTION):
        message = exc.format_message() if hasattr(exc, "format_message") else str(exc)
        is_file = name == "FileError"
        return Failure(
            ExitCode.INVALID if is_file else ExitCode.FAILED,
            "file_error" if is_file else "error",
            message,
            name,
            human=message,
        )
    if isinstance(exc, typer.Abort):
        message = (
            "Aborted: the confirmation was declined, or there was no terminal to ask "
            "on. Commands that ask for one take --yes to confirm in advance."
        )
        return Failure(ExitCode.FAILED, "aborted", message, name, human=message)
    if isinstance(exc, (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError)):
        where = f": {exc.filename}" if exc.filename else ""
        message = f"{exc.strerror or name}{where}"
        code = "file_not_found" if isinstance(exc, FileNotFoundError) else "file_error"
        return Failure(ExitCode.INVALID, code, message, name, human=message)
    if isinstance(exc, ValueError):
        # A local input the SDK or the CLI refused before any request: a malformed
        # JSON / annotation file, a bad argument value, an image it cannot read.
        # (A response the SDK cannot parse is a ServerError, never a ValueError.)
        errors = _pydantic_errors(exc)
        message = str(exc) or name
        return Failure(
            ExitCode.INVALID,
            "invalid_input",
            message,
            name,
            details={"errors": errors} if errors else None,
            human=message,
        )
    return None


def exit_code_for(error_type: str | None) -> ExitCode:
    """The exit code of an SDK error known only by its class NAME.

    For an error a report recorded rather than raised (``AnnotationFailure.error_type``):
    the same table :func:`classify` applies to a live exception. An unknown name, or
    none, is :attr:`ExitCode.FAILED`.
    """
    import pictograph.exceptions as sdk_errors

    cls = getattr(sdk_errors, error_type, None) if error_type else None
    if not (isinstance(cls, type) and issubclass(cls, PictographError)):
        return ExitCode.FAILED
    for known, exit_code, _code in _BY_CLASS:
        if issubclass(cls, known):
            return exit_code
    return ExitCode.FAILED


def output_is_json(environ: Mapping[str, str] | None = None) -> bool:
    """``PICTOGRAPH_OUTPUT=json`` (any case) is set."""
    env = os.environ if environ is None else environ
    return env.get(OUTPUT_ENV, "").strip().lower() == "json"


def json_requested(argv: Sequence[str], environ: Mapping[str, str] | None = None) -> bool:
    """A failure should print as JSON: ``--json`` was passed, or the env asks for it."""
    if output_is_json(environ):
        return True
    for arg in argv:
        if arg == "--":
            return False
        if arg == "--json":
            return True
    return False


def json_defaults(command: Any) -> dict[str, Any]:
    """A click ``default_map`` that turns every ``--json`` flag on.

    Walks the command tree: a command's own ``--json`` flag becomes ``{param: True}``
    and a group nests its subcommands' maps under their names - so a ``--json``
    added to any command later is covered without touching this.
    """
    out: dict[str, Any] = {}
    for param in getattr(command, "params", None) or ():
        if "--json" in (getattr(param, "opts", None) or ()):
            out[param.name] = True
    for name, sub in (getattr(command, "commands", None) or {}).items():
        nested = json_defaults(sub)
        if nested:
            out[name] = nested
    return out


JSON_ALWAYS_HELP = "Accepted on every command; this one always prints JSON."
"""Help of the ``--json`` flag a command that only ever prints JSON is given."""


def has_json_flag(command: Any) -> bool:
    """The command declares a ``--json`` option of its own."""
    return any("--json" in (getattr(p, "opts", None) or ()) for p in command.params or ())


def accept_json_everywhere(command: Any) -> None:
    """Give every command that always prints JSON a ``--json`` flag that changes nothing.

    An agent passes ``--json`` to every command it runs. A command that prints a table
    or a sentence takes a real ``--json`` of its own; one that only ever prints JSON has
    nothing to switch, and click used to refuse the flag there ("No such option:
    --json", exit 6) - so it gets a visible flag that is accepted and ignored
    (``expose_value=False``: the command never receives it). Walks a group's
    subcommands; idempotent. ``tests/unit/cli/test_json_everywhere.py`` holds every
    command that gets this flag to printing JSON only.
    """
    subcommands = getattr(command, "commands", None)
    if subcommands:
        for sub in subcommands.values():
            accept_json_everywhere(sub)
        return
    if getattr(command, "params", None) is None or has_json_flag(command):
        return
    command.params.append(
        TyperOption(
            param_decls=["--json"],
            is_flag=True,
            default=False,
            expose_value=False,
            help=JSON_ALWAYS_HELP,
        )
    )


def report(failure: Failure, *, as_json: bool) -> None:
    """Print a failure: one JSON line on stderr, or red text for a person."""
    if as_json:
        sys.stderr.write(json.dumps(failure.as_json(), default=str) + "\n")
        sys.stderr.flush()
        return
    if failure.human:
        print_error(failure.human)


def render_usage_error(exc: BaseException, *, rich_markup_mode: Any) -> None:
    """Print a click usage error for a person exactly as Typer does on its own."""
    if type(exc).__name__ == "NoArgsIsHelpError":
        return  # its help was printed when it was raised
    try:
        from typer import rich_utils
    except ImportError:  # pragma: no cover - rich is part of the [cli] extra
        rich_utils = None  # type: ignore[assignment]
    if rich_utils is not None and rich_markup_mode is not None:
        rich_utils.rich_format_error(exc)  # type: ignore[arg-type]
        return
    show = getattr(exc, "show", None)
    if callable(show):
        show()


class PictographCli(TyperGroup):
    """The root command: every failure exits with its code in the table and, when
    asked (``--json`` / ``PICTOGRAPH_OUTPUT=json``), prints a JSON error on stderr.

    It lives on the click group, not in ``pictograph.cli._app.main``, so that every
    way of running a command - the ``pictograph`` script, one command group on its
    own, ``typer.testing.CliRunner`` - fails the same way. Every Typer app in the CLI
    is built by :func:`command_group`, which sets this class.
    """

    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: Any,
    ) -> Any:
        argv = list(sys.argv[1:] if args is None else args)
        as_json = json_requested(argv)
        accept_json_everywhere(self)
        if output_is_json() and "default_map" not in extra:
            # PICTOGRAPH_OUTPUT=json: every --json flag defaults on, in every command.
            extra["default_map"] = json_defaults(self)
        try:
            rv = super().main(
                args=args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=False,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        except Exception as exc:
            failure = classify(exc)
            if failure is None:
                raise  # a bug: keep its traceback
            if not as_json and isinstance(exc, _USAGE_ERROR):
                render_usage_error(exc, rich_markup_mode=self.rich_markup_mode)
            else:
                report(failure, as_json=as_json)
            code = int(failure.exit_code)
        else:
            # Without standalone mode, an explicit typer.Exit(n) comes back as n;
            # a command that simply returns gives None.
            code = rv if isinstance(rv, int) and not isinstance(rv, bool) else int(ExitCode.OK)
        if standalone_mode:
            sys.exit(code)
        return code


def command_group(**kwargs: Any) -> typer.Typer:
    """A Typer app whose failures follow the CLI's error contract (:class:`PictographCli`).

    Every command group in the CLI is built with this, never ``typer.Typer`` directly,
    so a group run on its own fails exactly like the full ``pictograph`` command.
    """
    return typer.Typer(cls=PictographCli, **kwargs)


__all__ = [
    "JSON_ALWAYS_HELP",
    "OUTPUT_ENV",
    "CliError",
    "ExitCode",
    "Failure",
    "PictographCli",
    "accept_json_everywhere",
    "classify",
    "command_group",
    "exit_code_for",
    "has_json_flag",
    "json_defaults",
    "json_requested",
    "output_is_json",
    "render_usage_error",
    "report",
]
