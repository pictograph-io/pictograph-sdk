"""Tests for ``pictograph.exceptions``.

These tests cover:
- The constructor / string-formatting contract that callers see.
- The ``from_response`` factory's status-code → subclass dispatch.
- ``Retry-After`` parsing for both seconds and HTTP-date formats.
- Per-class defaults (``fix``, ``docs_url``) and per-instance overrides.
- Pickle round-trips (errors cross process boundaries via multiprocessing).

Each test asserts a specific invariant. None are smoke tests.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timedelta, timezone

import pytest

from pictograph.exceptions import (
    DOCS_BASE,
    ApiError,
    AuthError,
    ConfigurationError,
    ConflictError,
    ForbiddenError,
    NetworkError,
    NotFoundError,
    PaymentRequiredError,
    PictographError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
    ValidationError,
    _parse_retry_after,
    from_response,
)

# ───────────── PictographError base ─────────────


def test_pictograph_error_str_message_only() -> None:
    err = PictographError("bare error")
    assert str(err) == "bare error"


def test_pictograph_error_str_with_fix_and_docs() -> None:
    err = PictographError("boom", fix="restart", docs_url="https://x/y")
    assert str(err) == "boom\n  Fix: restart\n  Docs: https://x/y"


def test_pictograph_error_repr_does_not_leak_extra_state() -> None:
    err = PictographError("boom", fix="secret hint")
    assert repr(err) == "PictographError('boom')"


def test_pictograph_error_per_instance_fix_overrides_class_default() -> None:
    # AuthError carries a class-level fix; instance kwarg must win.
    err = AuthError("nope", fix="custom guidance")
    assert err.fix == "custom guidance"


def test_pictograph_error_class_default_used_when_instance_kwarg_absent() -> None:
    err = AuthError("nope")
    assert err.fix is not None
    assert "pk_live_" in err.fix


# ───────────── Class hierarchy ─────────────


@pytest.mark.parametrize(
    ("cls", "parents"),
    [
        (AuthError, (ApiError, PictographError, Exception)),
        (ForbiddenError, (ApiError, PictographError, Exception)),
        (NotFoundError, (ApiError, PictographError, Exception)),
        (ValidationError, (ApiError, PictographError, Exception)),
        (ConflictError, (ApiError, PictographError, Exception)),
        (PaymentRequiredError, (ApiError, PictographError, Exception)),
        (RateLimitError, (ApiError, PictographError, Exception)),
        (ServerError, (ApiError, PictographError, Exception)),
        (NetworkError, (PictographError, Exception)),
        (RequestTimeoutError, (NetworkError, PictographError, Exception)),
        (ConfigurationError, (PictographError, Exception)),
    ],
)
def test_class_hierarchy(cls: type[BaseException], parents: tuple[type, ...]) -> None:
    for parent in parents:
        assert issubclass(cls, parent), f"{cls.__name__} should subclass {parent.__name__}"


# ───────────── ApiError str formatting ─────────────


def test_api_error_str_with_status_and_request_id() -> None:
    err = ApiError("server died", status_code=503, request_id="req_123")
    expected_first_line = "server died (status=503, request_id=req_123)"
    assert str(err).startswith(expected_first_line)


def test_api_error_str_with_status_only() -> None:
    err = ApiError("nope", status_code=404)
    assert "(status=404)" in str(err)
    assert "request_id" not in str(err)


def test_api_error_str_without_status_or_request_id() -> None:
    # Synthesised ApiError (rare) shouldn't render an empty parenthetical.
    err = ApiError("synthetic")
    assert str(err) == "synthetic\n  Docs: " + (ApiError.docs_url or "")


# ───────────── docs_url defaults ─────────────


@pytest.mark.parametrize(
    ("cls", "expected_path"),
    [
        (AuthError, "/authentication"),
        (ForbiddenError, "/authentication"),
        (RateLimitError, "/rate-limits"),
        (PaymentRequiredError, "/concepts/credits-and-billing"),
        (ConfigurationError, "/installation"),
    ],
)
def test_docs_url_defaults(cls: type[ApiError | PictographError], expected_path: str) -> None:
    assert cls.docs_url is not None
    assert cls.docs_url.startswith(DOCS_BASE)
    assert cls.docs_url.endswith(expected_path)


# ───────────── ValidationError ─────────────


def test_validation_error_default_field_errors_is_empty_list() -> None:
    err = ValidationError("bad")
    assert err.field_errors == []


def test_validation_error_preserves_field_errors() -> None:
    fe = [{"loc": ["body", "annotations", 0, "name"], "msg": "field required"}]
    err = ValidationError("bad", field_errors=fe)
    assert err.field_errors == fe


def test_validation_error_passes_kwargs_to_apierror_base() -> None:
    err = ValidationError("bad", status_code=422, request_id="r1", response={"x": 1})
    assert err.status_code == 422
    assert err.request_id == "r1"
    assert err.response == {"x": 1}


# ───────────── PaymentRequiredError ─────────────


def test_payment_required_captures_credit_fields() -> None:
    err = PaymentRequiredError(
        "out of credits",
        credit_cost=50,
        credits_remaining=10,
        upgrade_url="https://app.pictograph.io/billing",
    )
    assert err.credit_cost == 50
    assert err.credits_remaining == 10
    assert err.upgrade_url == "https://app.pictograph.io/billing"


def test_payment_required_credit_fields_default_to_none() -> None:
    err = PaymentRequiredError("out")
    assert err.credit_cost is None
    assert err.credits_remaining is None
    assert err.unit is None
    assert err.upgrade_url is None


def test_payment_required_captures_unit() -> None:
    err = PaymentRequiredError(
        "out of credits",
        credit_cost=2_000_000,
        credits_remaining=10_000,
        unit="micro_usd",
    )
    assert err.unit == "micro_usd"


# ───────────── RateLimitError ─────────────


def test_rate_limit_error_default_fix_includes_retry_after_when_known() -> None:
    err = RateLimitError("slow down", retry_after=30.0)
    assert err.retry_after == 30.0
    assert err.fix is not None
    assert "30s" in err.fix


def test_rate_limit_error_default_fix_when_retry_after_missing() -> None:
    err = RateLimitError("slow down")
    assert err.retry_after is None
    assert err.fix is not None
    assert "Reduce request rate" in err.fix


def test_rate_limit_error_explicit_fix_overrides_default() -> None:
    err = RateLimitError("slow down", retry_after=10.0, fix="upgrade your plan")
    assert err.fix == "upgrade your plan"


# ───────────── from_response - dispatch ─────────────


@pytest.mark.parametrize(
    ("status_code", "expected_cls"),
    [
        (400, ValidationError),
        (401, AuthError),
        (402, PaymentRequiredError),
        (403, ForbiddenError),
        (404, NotFoundError),
        (409, ConflictError),
        (422, ValidationError),
        (429, RateLimitError),
        (500, ServerError),
        (502, ServerError),
        (503, ServerError),
        (504, ServerError),
    ],
)
def test_from_response_dispatch(status_code: int, expected_cls: type[ApiError]) -> None:
    err = from_response(status_code, body={"detail": "msg"})
    assert isinstance(err, expected_cls)
    assert err.status_code == status_code


def test_from_response_unmapped_4xx_returns_base_apierror() -> None:
    # 418 is unmapped; should land at ApiError, not a 4xx-bucketed subclass.
    err = from_response(418, body={"detail": "I'm a teapot"})
    assert type(err) is ApiError
    assert err.status_code == 418


# ───────────── from_response - message resolution ─────────────


def test_from_response_message_explicit_wins() -> None:
    err = from_response(500, message="explicit", body={"detail": "from-body"})
    assert err.message == "explicit"


def test_from_response_message_falls_back_to_body_detail_string() -> None:
    err = from_response(500, body={"detail": "from-body"})
    assert err.message == "from-body"


def test_from_response_message_falls_back_to_body_error_field() -> None:
    err = from_response(500, body={"error": "from-error-key"})
    assert err.message == "from-error-key"


def test_from_response_message_falls_back_to_status_string_for_dict_without_detail() -> None:
    err = from_response(500, body={"random": "thing"})
    assert err.message == "HTTP 500"


def test_from_response_message_falls_back_for_non_dict_body() -> None:
    err = from_response(500, body="<html>500</html>")
    assert err.message == "HTTP 500"


def test_from_response_message_with_no_body() -> None:
    err = from_response(500)
    assert err.message == "HTTP 500"


def test_from_response_422_with_list_detail_does_not_use_it_as_message() -> None:
    body = {"detail": [{"loc": ["a"], "msg": "x"}]}
    err = from_response(422, body=body)
    # List detail isn't a string - message should fall back rather than render the list.
    assert err.message == "HTTP 422"


def test_from_response_message_extracted_from_dict_detail() -> None:
    # The backend's HTTPException(detail={"message": ..., "code": ...}) shape:
    # surface the nested human message instead of the bare "HTTP <status>".
    err = from_response(409, body={"detail": {"message": "name already taken", "code": "dup"}})
    assert err.message == "name already taken"


def test_from_response_message_from_dict_detail_alt_nested_key() -> None:
    err = from_response(409, body={"detail": {"detail": "nested detail text"}})
    assert err.message == "nested detail text"


def test_from_response_dict_detail_without_text_falls_back_to_status() -> None:
    # A dict detail carrying no string message/detail must not crash or invent
    # text - it falls back to the status string.
    err = from_response(409, body={"detail": {"code": "dup", "count": 3}})
    assert err.message == "HTTP 409"


# ───────────── from_response - 402 ─────────────


def test_from_response_402_extracts_credit_fields() -> None:
    body = {
        "detail": "out of credits",
        "credit_cost": 50,
        "credits_remaining": 10,
        "upgrade_url": "https://up",
    }
    err = from_response(402, body=body)
    assert isinstance(err, PaymentRequiredError)
    assert err.credit_cost == 50
    assert err.credits_remaining == 10
    assert err.upgrade_url == "https://up"


def test_from_response_402_falls_back_to_alt_key_names() -> None:
    # The current contract uses 'required' / 'remaining' (µUSD) + 'unit'.
    body = {
        "error": "insufficient_credits",
        "required": 2_000_000,  # µUSD ($2.00)
        "remaining": 50_000,  # µUSD ($0.05)
        "unit": "micro_usd",
    }
    err = from_response(402, body=body)
    assert isinstance(err, PaymentRequiredError)
    assert err.credit_cost == 2_000_000
    assert err.credits_remaining == 50_000
    assert err.unit == "micro_usd"


def test_from_response_402_extracts_unit_from_nested_detail() -> None:
    # FastAPI HTTPException(detail={...}) nesting (e.g. training).
    body = {
        "detail": {
            "message": "Insufficient credits for training run.",
            "required": 2_000_000,
            "remaining": 0,
            "unit": "micro_usd",
            "block_reason": "insufficient_credits",
        }
    }
    err = from_response(402, body=body)
    assert isinstance(err, PaymentRequiredError)
    assert err.credit_cost == 2_000_000
    assert err.credits_remaining == 0
    assert err.unit == "micro_usd"


def test_from_response_402_handles_missing_credit_fields() -> None:
    err = from_response(402, body={"detail": "out"})
    assert isinstance(err, PaymentRequiredError)
    assert err.credit_cost is None
    assert err.credits_remaining is None
    assert err.unit is None


def test_from_response_402_with_non_dict_body_is_payment_required() -> None:
    # A 402 unambiguously means payment-required regardless of body shape, so it
    # always maps to PaymentRequiredError (credit fields simply stay None when
    # the body carries none).
    err = from_response(402, body="error html")
    assert type(err) is PaymentRequiredError
    assert err.credit_cost is None


# ───────────── from_response - 422 ─────────────


def test_from_response_422_extracts_field_errors() -> None:
    body = {
        "detail": [
            {"loc": ["body", "annotations", 0, "name"], "msg": "field required"},
            {"loc": ["body", "image_id"], "msg": "uuid expected"},
        ],
    }
    err = from_response(422, body=body)
    assert isinstance(err, ValidationError)
    assert len(err.field_errors) == 2
    assert err.field_errors[0]["msg"] == "field required"


def test_from_response_422_with_string_detail_has_empty_field_errors() -> None:
    err = from_response(422, body={"detail": "validation failed"})
    assert isinstance(err, ValidationError)
    assert err.field_errors == []
    assert err.message == "validation failed"


def test_from_response_400_uses_validation_error() -> None:
    err = from_response(400, body={"detail": "bad"})
    assert isinstance(err, ValidationError)


# ───────────── from_response - 429 ─────────────


def test_from_response_429_parses_retry_after_seconds() -> None:
    err = from_response(429, body={"detail": "slow"}, headers={"Retry-After": "42"})
    assert isinstance(err, RateLimitError)
    assert err.retry_after == 42.0


def test_from_response_429_parses_retry_after_http_date() -> None:
    future = datetime.now(timezone.utc) + timedelta(seconds=60)
    # RFC 7231 IMF-fixdate format
    http_date = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    err = from_response(429, body={"detail": "slow"}, headers={"Retry-After": http_date})
    assert isinstance(err, RateLimitError)
    # Allow a few seconds of clock drift in test execution.
    assert err.retry_after is not None
    assert 50 <= err.retry_after <= 65


def test_from_response_429_without_retry_after_header() -> None:
    err = from_response(429, body={"detail": "slow"}, headers={})
    assert isinstance(err, RateLimitError)
    assert err.retry_after is None


def test_from_response_429_without_headers_at_all() -> None:
    err = from_response(429, body={"detail": "slow"})
    assert isinstance(err, RateLimitError)
    assert err.retry_after is None


def test_from_response_429_parses_lowercase_header() -> None:
    err = from_response(429, body={"detail": "slow"}, headers={"retry-after": "10"})
    assert isinstance(err, RateLimitError)
    assert err.retry_after == 10.0


# ───────────── from_response - meta ─────────────


def test_from_response_propagates_request_id_and_response_body() -> None:
    body = {"detail": "x"}
    err = from_response(500, body=body, request_id="req_xyz")
    assert err.request_id == "req_xyz"
    assert err.response is body


# ───────────── from_response - standardized error envelope ─────────────


def test_error_code_read_from_envelope() -> None:
    body = {
        "error": {"code": "not_found", "message": "Dataset not found."},
        "detail": "Dataset not found.",
    }
    err = from_response(404, body=body)
    assert isinstance(err, NotFoundError)
    assert err.code == "not_found"


def test_error_code_none_for_legacy_body() -> None:
    # A response predating the envelope (bare `detail`) has no machine code.
    err = from_response(404, body={"detail": "gone"})
    assert err.code is None


def test_envelope_message_preferred_over_detail() -> None:
    body = {
        "error": {"code": "conflict", "message": "Envelope message."},
        "detail": "legacy detail",
    }
    err = from_response(409, body=body)
    # The envelope's message wins, and code is captured.
    assert err.message == "Envelope message."
    assert err.code == "conflict"


def test_envelope_402_reads_canonical_micro_usd_from_details() -> None:
    body = {
        "error": {
            "code": "insufficient_credits",
            "message": "Not enough compute credits.",
            "details": {
                "required_micro_usd": 2_000_000,
                "remaining_micro_usd": 50_000,
                "required": 2_000_000,  # legacy aliases coexist
                "remaining": 50_000,
                "unit": "micro_usd",
                "upgrade_url": "/settings?tab=billing",
            },
        },
        "detail": {"error": "insufficient_credits", "required": 2_000_000, "remaining": 50_000},
    }
    err = from_response(402, body=body)
    assert isinstance(err, PaymentRequiredError)
    assert err.code == "insufficient_credits"
    assert err.credit_cost == 2_000_000
    assert err.credits_remaining == 50_000
    assert err.unit == "micro_usd"
    assert err.upgrade_url == "/settings?tab=billing"


def test_envelope_422_reads_field_errors_from_details() -> None:
    body = {
        "error": {
            "code": "validation_error",
            "message": "The request failed validation.",
            "details": {
                "errors": [
                    {"loc": ["body", "count"], "msg": "int expected", "type": "int_parsing"},
                ],
            },
        },
        "detail": [{"loc": ["body", "count"], "msg": "int expected", "type": "int_parsing"}],
    }
    err = from_response(422, body=body)
    assert isinstance(err, ValidationError)
    assert err.code == "validation_error"
    assert len(err.field_errors) == 1
    assert err.field_errors[0]["msg"] == "int expected"


def test_envelope_and_legacy_detail_coexist_real_backend_shape() -> None:
    # The exact shape the backend now emits: envelope + verbatim legacy detail.
    body = {
        "error": {"code": "forbidden", "message": "You do not have permission."},
        "detail": "You do not have permission.",
    }
    err = from_response(403, body=body)
    assert isinstance(err, ForbiddenError)
    assert err.code == "forbidden"
    assert err.message == "You do not have permission."


def test_rate_limited_code_from_envelope() -> None:
    body = {
        "error": {
            "code": "rate_limited",
            "message": "Rate limit exceeded.",
            "details": {"retry_after": 30},
        },
        "detail": "Rate limit exceeded.",
    }
    err = from_response(429, body=body, headers={"Retry-After": "30"})
    assert isinstance(err, RateLimitError)
    assert err.code == "rate_limited"
    assert err.retry_after == 30.0


# ───────────── _parse_retry_after edge cases ─────────────


def test_parse_retry_after_negative_clamps_to_zero() -> None:
    past = datetime.now(timezone.utc) - timedelta(seconds=300)
    http_date = past.strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert _parse_retry_after({"Retry-After": http_date}) == 0.0


def test_parse_retry_after_invalid_string_returns_none() -> None:
    assert _parse_retry_after({"Retry-After": "not-a-date-or-number"}) is None


def test_parse_retry_after_empty_string_returns_none() -> None:
    assert _parse_retry_after({"Retry-After": ""}) is None


def test_parse_retry_after_none_headers_returns_none() -> None:
    assert _parse_retry_after(None) is None


def test_parse_retry_after_negative_seconds_clamps_to_zero() -> None:
    assert _parse_retry_after({"Retry-After": "-15"}) == 0.0


# ───────────── Pickle round-trip ─────────────


@pytest.mark.parametrize(
    "err",
    [
        PictographError("p"),
        ConfigurationError("c"),
        NetworkError("n"),
        RequestTimeoutError("t"),
        ApiError("a", status_code=500, code="internal_error", request_id="r"),
        AuthError("auth", status_code=401, code="unauthorized", request_id="r"),
        ValidationError(
            "v",
            status_code=422,
            field_errors=[{"loc": ["x"], "msg": "y"}],
        ),
        PaymentRequiredError(
            "p",
            status_code=402,
            credit_cost=2_000_000,
            credits_remaining=50_000,
            unit="micro_usd",
        ),
        RateLimitError("rl", status_code=429, retry_after=5.0),
    ],
)
def test_exceptions_round_trip_through_pickle(err: PictographError) -> None:
    """Errors must survive pickle so they can cross multiprocessing boundaries."""
    restored = pickle.loads(pickle.dumps(err))
    assert type(restored) is type(err)
    assert restored.message == err.message
    if isinstance(err, ApiError):
        assert isinstance(restored, ApiError)
        assert restored.status_code == err.status_code
        assert restored.code == err.code
        assert restored.request_id == err.request_id
    if isinstance(err, ValidationError):
        assert isinstance(restored, ValidationError)
        assert restored.field_errors == err.field_errors
    if isinstance(err, PaymentRequiredError):
        assert isinstance(restored, PaymentRequiredError)
        assert restored.credit_cost == err.credit_cost
        assert restored.credits_remaining == err.credits_remaining
        assert restored.unit == err.unit
    if isinstance(err, RateLimitError):
        assert isinstance(restored, RateLimitError)
        assert restored.retry_after == err.retry_after
