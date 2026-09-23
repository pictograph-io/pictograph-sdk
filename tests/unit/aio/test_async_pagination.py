"""Tests for ``pictograph._http.pagination.AsyncOffsetPager``.

Mirrors the sync ``OffsetPager`` termination + cap semantics against an async
``fetch_page``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pictograph._http.pagination import AsyncOffsetPager

pytestmark = pytest.mark.anyio


def _make_fetcher(pages: list[list[dict[str, Any]]]) -> tuple[Any, list[tuple[int, int]]]:
    """Return an async fetch_page returning canned pages, plus a call log."""
    calls: list[tuple[int, int]] = []

    async def fetch(offset: int, limit: int) -> dict[str, Any]:
        calls.append((offset, limit))
        idx = offset // limit if limit else 0
        items = pages[idx] if idx < len(pages) else []
        return {"items": items}

    return fetch, calls


async def test_iterates_across_multiple_pages() -> None:
    fetch, _calls = _make_fetcher([[{"n": 1}, {"n": 2}], [{"n": 3}]])
    pager: AsyncOffsetPager[dict[str, Any]] = AsyncOffsetPager(
        fetch, items_key="items", page_size=2
    )
    got = [x async for x in pager]
    assert got == [{"n": 1}, {"n": 2}, {"n": 3}]


async def test_all_materialises() -> None:
    fetch, _calls = _make_fetcher([[{"n": 1}], []])
    pager: AsyncOffsetPager[dict[str, Any]] = AsyncOffsetPager(
        fetch, items_key="items", page_size=1
    )
    assert await pager.all() == [{"n": 1}]


async def test_short_page_stops_without_extra_fetch() -> None:
    fetch, calls = _make_fetcher([[{"n": 1}]])  # only one item, page_size 5
    pager: AsyncOffsetPager[dict[str, Any]] = AsyncOffsetPager(
        fetch, items_key="items", page_size=5
    )
    assert await pager.all() == [{"n": 1}]
    assert len(calls) == 1  # short page → no second round-trip


async def test_max_total_caps_mid_page() -> None:
    fetch, _calls = _make_fetcher([[{"n": 1}, {"n": 2}, {"n": 3}]])
    pager: AsyncOffsetPager[dict[str, Any]] = AsyncOffsetPager(
        fetch, items_key="items", page_size=3, max_total=2
    )
    assert await pager.all() == [{"n": 1}, {"n": 2}]


async def test_max_total_zero_fetches_nothing() -> None:
    fetch, calls = _make_fetcher([[{"n": 1}]])
    pager: AsyncOffsetPager[dict[str, Any]] = AsyncOffsetPager(
        fetch, items_key="items", page_size=2, max_total=0
    )
    assert await pager.all() == []
    assert calls == []


async def test_first_returns_one_and_stops() -> None:
    fetch, calls = _make_fetcher([[{"n": 1}, {"n": 2}]])
    pager: AsyncOffsetPager[dict[str, Any]] = AsyncOffsetPager(
        fetch, items_key="items", page_size=2
    )
    assert await pager.first() == {"n": 1}
    assert len(calls) == 1


async def test_first_empty_returns_none() -> None:
    fetch, _calls = _make_fetcher([[]])
    pager: AsyncOffsetPager[dict[str, Any]] = AsyncOffsetPager(
        fetch, items_key="items", page_size=2
    )
    assert await pager.first() is None


async def test_parse_item_applied() -> None:
    fetch, _calls = _make_fetcher([[{"n": 1}, {"n": 2}]])
    pager: AsyncOffsetPager[int] = AsyncOffsetPager(
        fetch, items_key="items", page_size=2, parse_item=lambda raw: int(raw["n"])
    )
    assert await pager.all() == [1, 2]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"items_key": "x", "page_size": 0}, "page_size"),
        ({"items_key": "x", "max_total": -1}, "max_total"),
        ({"items_key": ""}, "items_key"),
    ],
)
def test_construction_validation(kwargs: dict[str, Any], match: str) -> None:
    async def fetch(_o: int, _l: int) -> dict[str, Any]:
        return {"x": []}

    with pytest.raises(ValueError, match=match):
        AsyncOffsetPager(fetch, **kwargs)
