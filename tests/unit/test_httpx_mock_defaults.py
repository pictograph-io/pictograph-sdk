"""Pin the ``pytest-httpx`` defaults that ~15 tests rest their whole case on.

A recurring shape in `tests/unit/resources/`::

    def test_delete_defaults_no_cascade(httpx_mock, directories):
        httpx_mock.add_response(method="DELETE", url=f"{_PATH}/{ID}?cascade=false", json={...})
        directories.delete(DATASET, ID)  # must not raise

There is no `assert` in it. What makes it a real test is two plugin defaults
working together: the registered URL carries `?cascade=false`, so a client that
sent `cascade=true` would not MATCH it, and
`assert_all_responses_were_requested=True` then fails the test because the
registered response went unused. Flip either default and roughly fifteen tests
across `test_directories`, `test_models`, `test_search`, `test_organizations`,
`test_webhooks`, `test_workflows`, `test_exports` and `test_api_keys` keep
passing while asserting nothing at all.

That is not hypothetical: `pytest-httpx` is FLOORED in the dev extra
(`pytest-httpx>=0.30`), not pinned, so a routine upgrade is all it would take -
and nothing in any of those tests says out loud that it depends on this.

This is the cheap version of the fix. The thorough one is to give each of those
tests an explicit assertion on the recorded request; until then
this makes the dependency visible and gives the upgrade something to trip over.
"""

from __future__ import annotations

import inspect

import pytest_httpx


def test_an_unrequested_mock_still_fails_the_test() -> None:
    """`assert_all_responses_were_requested` is the assertion in those tests."""
    default = (
        inspect.signature(pytest_httpx._HTTPXMockOptions.__init__)
        .parameters["assert_all_responses_were_requested"]
        .default
    )
    assert default is True, (
        "pytest-httpx no longer fails a test whose registered response was never "
        "requested. ~15 SDK tests carry no explicit assert and rely on exactly "
        "that - they are now passing while checking nothing. Either pin the "
        "previous pytest-httpx, or give those tests real assertions."
    )


def test_an_unexpected_request_still_fails_the_test() -> None:
    """The other half: a call the test never registered must not pass silently."""
    default = (
        inspect.signature(pytest_httpx._HTTPXMockOptions.__init__)
        .parameters["assert_all_requests_were_expected"]
        .default
    )
    assert default is True, (
        "pytest-httpx no longer fails on a request no response was registered for, "
        "so an SDK method calling an unexpected endpoint would go unnoticed."
    )


def test_a_response_is_not_reusable_by_default() -> None:
    """`can_send_already_matched_responses=True` would let ONE registered response
    satisfy N calls - which is how a test asserting "exactly one request" stops
    being able to tell one from many."""
    default = (
        inspect.signature(pytest_httpx._HTTPXMockOptions.__init__)
        .parameters["can_send_already_matched_responses"]
        .default
    )
    assert default is False


def test_the_options_class_is_still_where_we_look() -> None:
    """Guard the guard: these three read a PRIVATE symbol, so a rename would make
    them error rather than silently pass - but say so out loud, because an
    ImportError here means the checks above are not running, not that the
    defaults are fine."""
    assert hasattr(pytest_httpx, "_HTTPXMockOptions"), (
        "pytest-httpx moved _HTTPXMockOptions; the default-pinning checks in this "
        "module need re-pointing at whatever replaced it."
    )
    params = inspect.signature(pytest_httpx._HTTPXMockOptions.__init__).parameters
    for name in (
        "assert_all_responses_were_requested",
        "assert_all_requests_were_expected",
        "can_send_already_matched_responses",
    ):
        assert name in params, f"{name} is gone from pytest-httpx's options"
