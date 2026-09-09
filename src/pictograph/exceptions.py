"""Exception hierarchy for the Pictograph SDK.

All errors raised by the SDK derive from :class:`PictographError`. HTTP-level
errors derive from :class:`ApiError` and carry ``status_code``, ``request_id``,
``fix`` (one-line actionable suggestion) and ``docs_url``.

Catch patterns::

    try:
        client.training.create(...)
    except PaymentRequiredError as e:
        # Out of compute credit. e.credits_remaining is the µUSD balance
        # (e.unit == "micro_usd"); e.credit_cost is the µUSD shortfall.
        ...
    except RateLimitError as e:
        # Back off. e.retry_after gives seconds (None if not provided).
        ...
    except ApiError as e:
        # Any other HTTP-level error. ``e.code`` is a stable machine-readable
        # slug from the API's error envelope ("not_found", "conflict", ...),
        # so you can branch on it instead of parsing ``str(e)``.
        logger.error("Pictograph %s [%s]: %s -- %s", e.status_code, e.code, e, e.fix)
    except PictographError:
        # Catch-all for transport/config/network issues too.
        raise

Every HTTP error body is the standardized envelope
``{"error": {"code", "message", "details"}, "detail": <legacy>}`` - the SDK
reads ``error.code`` into :attr:`ApiError.code` and stays back-compatible with
older ``{"detail": ...}`` responses. The ``.fix`` and ``.docs_url`` fields exist
so SDK callers (especially LLM agents) can self-heal without parsing tracebacks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

DOCS_BASE = "https://pictograph.io/docs"


class PictographError(Exception):
    """Base class for every error raised by the Pictograph SDK."""

    # Class-level defaults; instances may override via __init__.
    fix: str | None = None
    docs_url: str | None = None

    def __init__(
        self,
        message: str,
        *,
        fix: str | None = None,
        docs_url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if fix is not None:
            self.fix = fix
        if docs_url is not None:
            self.docs_url = docs_url

    def __str__(self) -> str:
        parts = [self.message]
        if self.fix:
            parts.append(f"  Fix: {self.fix}")
        if self.docs_url:
            parts.append(f"  Docs: {self.docs_url}")
        return "\n".join(parts)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r})"


# ───────────── configuration & setup ─────────────


class ConfigurationError(PictographError):
    """The SDK is misconfigured - missing key, bad base URL, unreadable config file."""

    docs_url: str | None = f"{DOCS_BASE}/installation"


# ───────────── transport (pre-response) ─────────────


class NetworkError(PictographError):
    """An HTTP request failed before producing a response (DNS, TCP, TLS, etc)."""

    docs_url: str | None = f"{DOCS_BASE}/error-handling"
    fix: str | None = "Check your internet connection and that api.pictograph.io is reachable."


class RequestTimeoutError(NetworkError):
    """An HTTP request exceeded its configured timeout."""

    fix: str | None = (
        "Increase Client(timeout=...) for slow endpoints, or break large "
        "requests into smaller batches."
    )


class PollTimeoutError(PictographError):
    """An asynchronous server-side operation did not finish in the polling window.

    Distinct from :class:`RequestTimeoutError`: each individual HTTP poll
    succeeded, but the underlying job (export, training run, batch SAM3) is
    still in progress when the caller's timeout elapsed. The job continues
    running on the server - fetch its status later via
    ``client.<resource>.get(...)`` to check.
    """

    docs_url: str | None = f"{DOCS_BASE}/error-handling"
    fix: str | None = (
        "Increase the timeout= kwarg, or fetch status later via the "
        "resource's .get() method - the server-side job is still running."
    )


# ───────────── HTTP response errors ─────────────


class ApiError(PictographError):
    """Base for any error reported by the API via an HTTP response.

    Attributes:
        status_code: HTTP status code (3-digit integer) or ``None`` if synthesised.
        code: Stable, machine-readable error slug from the API's ``error.code``
            envelope (e.g. ``"not_found"``, ``"insufficient_credits"``,
            ``"rate_limited"``) - ``None`` for a legacy response without the
            envelope. Prefer branching on ``code`` over parsing the message.
        request_id: Server-issued ``X-Request-Id`` header value, useful for support.
        response: Parsed JSON body (a dict, typically) or raw text fallback.
    """

    docs_url: str | None = f"{DOCS_BASE}/error-handling"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        request_id: str | None = None,
        response: Any = None,
        fix: str | None = None,
        docs_url: str | None = None,
    ) -> None:
        super().__init__(message, fix=fix, docs_url=docs_url)
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.response = response

    def __str__(self) -> str:
        header = self.message
        meta: list[str] = []
        if self.status_code is not None:
            meta.append(f"status={self.status_code}")
        if self.request_id:
            meta.append(f"request_id={self.request_id}")
        if meta:
            header = f"{header} ({', '.join(meta)})"
        parts = [header]
        if self.fix:
            parts.append(f"  Fix: {self.fix}")
        if self.docs_url:
            parts.append(f"  Docs: {self.docs_url}")
        return "\n".join(parts)


class AuthError(ApiError):
    """401 - the API key is missing, malformed, expired, or revoked."""

    docs_url: str | None = f"{DOCS_BASE}/authentication"
    fix: str | None = (
        "Verify PICTOGRAPH_API_KEY is set and starts with 'pk_live_'. "
        "Generate a new key at Settings → API Keys."
    )


class ForbiddenError(ApiError):
    """403 - the key authenticated, but lacks role for this action or org scope."""

    docs_url: str | None = f"{DOCS_BASE}/authentication"
    fix: str | None = (
        "The API key's role does not permit this action. "
        "Write/delete operations require admin or owner role."
    )


class NotFoundError(ApiError):
    """404 - the requested resource does not exist (or is not visible to this key)."""

    fix: str | None = (
        "Verify the ID or name. Names are case-sensitive. "
        "Use the resource's .iter() method to enumerate what's available."
    )


class ValidationError(ApiError):
    """400 / 422 - request payload or query parameters failed validation.

    ``field_errors`` carries per-field details when the backend returns them in
    FastAPI / Pydantic format::

        [{"loc": ["body", "annotations", 0, "name"], "msg": "field required"}]
    """

    def __init__(
        self,
        message: str,
        *,
        field_errors: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.field_errors: list[dict[str, Any]] = field_errors or []


class ConflictError(ApiError):
    """409 - request conflicts with current state (duplicate name, version mismatch)."""

    fix: str | None = (
        "Resource already exists or operation conflicts with current state. "
        "Pick a different name, or fetch the existing resource first."
    )


class PaymentRequiredError(ApiError):
    """402 - insufficient compute credit or feature requires a paid plan.

    Compute credit is USD-denominated; the backend's 402 body reports the
    shortfall in **micro-USD (µUSD)** (``1 USD = 1_000_000 µUSD``) via the
    canonical ``error.details.required_micro_usd`` / ``remaining_micro_usd``
    (with ``unit="micro_usd"``), falling back to the legacy ``required`` /
    ``remaining`` keys on older responses.

    Attributes:
        credit_cost: Compute credit the operation would have consumed, in µUSD
            (from ``error.details.required_micro_usd``, else legacy ``required``).
        credits_remaining: Compute credit the organization has available now,
            in µUSD (from ``error.details.remaining_micro_usd``, else legacy
            ``remaining``).
        unit: Unit of ``credit_cost`` / ``credits_remaining`` - ``"micro_usd"``
            for the current contract.
        upgrade_url: Direct link to the billing page for the org (when provided).
    """

    docs_url: str | None = f"{DOCS_BASE}/concepts/credits-and-billing"
    fix: str | None = (
        "Top up compute credit in Settings → Billing, or upgrade your plan. "
        "Use client.credits.estimate(...) to gate operations before invoking."
    )

    def __init__(
        self,
        message: str,
        *,
        credit_cost: int | None = None,
        credits_remaining: int | None = None,
        unit: str | None = None,
        upgrade_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.credit_cost = credit_cost
        self.credits_remaining = credits_remaining
        self.unit = unit
        self.upgrade_url = upgrade_url


class RateLimitError(ApiError):
    """429 - request rate exceeded.

    ``retry_after`` is the parsed ``Retry-After`` header in seconds, or ``None``
    if absent. The SDK auto-waits when ``retry_after <= 120`` (the default
    auto-wait threshold), matching ``RetryPolicy._delay_for_response``.
    """

    docs_url: str | None = f"{DOCS_BASE}/rate-limits"

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        if "fix" not in kwargs or kwargs.get("fix") is None:
            # Tailor the fix message to whether we know how long to wait.
            if retry_after is not None:
                self.fix = (
                    f"Wait {retry_after:.0f}s before retrying. "
                    "The SDK auto-retries when retry_after <= 120s."
                )
            else:
                self.fix = "Reduce request rate or upgrade for a higher rate-limit tier."


class ServerError(ApiError):
    """5xx - server-side failure. Usually transient; the SDK auto-retries 5xx."""

    fix: str | None = (
        "The server encountered an error. If this persists, contact "
        "support@pictograph.io with the request_id."
    )


# ───────────── factory ─────────────


# Status codes mapped to direct subclass constructors. Special-case codes
# (402, 422, 429) are handled in `from_response` so they can extract extra
# fields from headers/body before instantiation.
_STATUS_TO_EXCEPTION: dict[int, type[ApiError]] = {
    401: AuthError,
    403: ForbiddenError,
    404: NotFoundError,
    409: ConflictError,
}


def from_response(
    status_code: int,
    *,
    message: str | None = None,
    body: Any = None,
    request_id: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> ApiError:
    """Construct the right :class:`ApiError` subclass for an HTTP response.

    Args:
        status_code: 3-digit HTTP status from the response.
        message: Override message; if absent, derived from ``body['detail']``.
        body: Parsed JSON body (dict). Used to extract context (credits, fields).
        request_id: ``X-Request-Id`` header value, if present.
        headers: Response headers, scanned for ``Retry-After`` on 429.

    Returns:
        An ``ApiError`` instance whose concrete type matches ``status_code``.
    """
    msg = _resolve_message(message, body, status_code)
    common: dict[str, Any] = {
        "status_code": status_code,
        "code": _resolve_code(body),
        "request_id": request_id,
        "response": body,
    }

    if status_code == 429:
        return RateLimitError(
            msg,
            retry_after=_parse_retry_after(headers),
            **common,
        )

    if status_code == 402:
        cost, remaining, unit, upgrade_url = _extract_credit_fields(body)
        return PaymentRequiredError(
            msg,
            credit_cost=cost,
            credits_remaining=remaining,
            unit=unit,
            upgrade_url=upgrade_url,
            **common,
        )

    if status_code in (400, 422):
        return ValidationError(
            msg,
            field_errors=_extract_field_errors(body),
            **common,
        )

    cls = _STATUS_TO_EXCEPTION.get(status_code)
    if cls is not None:
        return cls(msg, **common)

    if 500 <= status_code < 600:
        return ServerError(msg, **common)

    return ApiError(msg, **common)


def _envelope(body: Any) -> dict[str, Any]:
    """The standardized ``{"error": {"code", "message", "details"}}`` object
    if present, else ``{}``.

    The backend wraps every error in this envelope but keeps the legacy
    ``detail`` field alongside it, so the extractors below prefer the envelope
    and fall back to ``detail`` for older responses.
    """
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return err
    return {}


def _resolve_code(body: Any) -> str | None:
    """The stable machine-readable ``error.code`` slug, or ``None`` if the
    response predates the envelope."""
    code = _envelope(body).get("code")
    return code if isinstance(code, str) and code else None


def _resolve_message(message: str | None, body: Any, status_code: int) -> str:
    """Pick the best human message: explicit > envelope > body['detail'] > fallback."""
    if message:
        return message
    env_msg = _envelope(body).get("message")
    if isinstance(env_msg, str) and env_msg:
        return env_msg
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str) and detail:
            return detail
        # Legacy flat-body shapes (a bare ``error``/``message`` string).
        for key in ("error", "message"):
            val = body.get(key)
            if isinstance(val, str) and val:
                return val
        # The backend's own legacy ``HTTPException(detail={...})`` shape (e.g.
        # the 402 credit path) nests the human text inside the dict - surface it
        # so ``.message`` / ``.fix`` stay useful for agent self-healing.
        if isinstance(detail, dict):
            nested = detail.get("message") or detail.get("detail")
            if isinstance(nested, str) and nested:
                return nested
    return f"HTTP {status_code}"


def _extract_credit_fields(body: Any) -> tuple[Any, Any, Any, Any]:
    """Pull the canonical credit-gate fields from a 402 body.

    Reads the envelope's ``error.details`` first (canonical
    ``required_micro_usd`` / ``remaining_micro_usd`` / ``unit`` / ``upgrade_url``),
    then falls back to the legacy top-level / nested-``detail`` shapes
    (``required`` / ``remaining`` / ``credit_cost`` / ``credits_remaining``).
    Returns ``(credit_cost, credits_remaining, unit, upgrade_url)``.
    """
    details = _envelope(body).get("details")
    details = details if isinstance(details, dict) else {}
    top: dict[str, Any] = body if isinstance(body, dict) else {}
    detail_raw = top.get("detail")
    nested: dict[str, Any] = detail_raw if isinstance(detail_raw, dict) else {}

    cost = (
        _first_present(details, "required_micro_usd", "credit_cost", "required")
        or _first_present(top, "credit_cost", "required")
        or _first_present(nested, "credit_cost", "required")
    )
    remaining = (
        _first_present(details, "remaining_micro_usd", "credits_remaining", "remaining")
        or _first_present(top, "credits_remaining", "remaining")
        or _first_present(nested, "credits_remaining", "remaining")
    )
    unit = details.get("unit") or top.get("unit") or nested.get("unit")
    upgrade_url = details.get("upgrade_url") or top.get("upgrade_url") or nested.get("upgrade_url")
    return cost, remaining, unit, upgrade_url


def _extract_field_errors(body: Any) -> list[dict[str, Any]] | None:
    """Per-field validation errors, from ``error.details.errors`` (envelope) or
    the legacy ``{"detail": [{"loc": [...], "msg": "..."}, ...]}`` list."""
    details = _envelope(body).get("details")
    if isinstance(details, dict) and isinstance(details.get("errors"), list):
        return [d for d in details["errors"] if isinstance(d, dict)]
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, list):
            return [d for d in detail if isinstance(d, dict)]
    return None


def _first_present(body: Mapping[str, Any], *keys: str) -> Any:
    """Return the first value present (and not None) for any of ``keys``."""
    for key in keys:
        if key in body and body[key] is not None:
            return body[key]
    return None


def _parse_retry_after(headers: Mapping[str, str] | None) -> float | None:
    """Parse ``Retry-After`` - supports plain seconds and HTTP-date formats.

    Returns ``None`` if missing or unparseable. Negative durations clamp to 0.
    """
    if headers is None:
        return None
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if not raw:
        return None
    raw = str(raw).strip()
    # Most APIs use plain seconds.
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    # RFC 7231 also allows HTTP-date format.
    try:
        target = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    delta = (target - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, delta)


class UnsafeCheckpointError(PictographError):
    """A PyTorch checkpoint could not be read without executing embedded code.

    ``torch.load(weights_only=False)`` runs the file's own ``__reduce__``, so
    loading an untrusted checkpoint is equivalent to running it. Models can be
    forked between organizations, which means a ``.pth`` is not always yours.

    The SDK allowlists the inert non-tensor types its own checkpoints carry, so
    Pictograph-trained artifacts load on the SAFE path. This is raised when a file
    needs something beyond that allowlist. If you produced the file yourself and
    trust it, re-load with ``allow_unsafe_pickle=True``.
    """

    def __init__(self, path: object, cause: Exception | None = None) -> None:
        super().__init__(
            f"Refusing to load {path} because it cannot be read without executing "
            f"code embedded in the file. If you created this checkpoint and trust "
            f"it, pass allow_unsafe_pickle=True."
            + (f" (underlying error: {cause})" if cause else "")
        )
        self.path = path
        self.cause = cause
