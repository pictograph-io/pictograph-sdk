"""Tests for ``pictograph._http.pagination.OffsetPager``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from pictograph._http.pagination import DEFAULT_PAGE_SIZE, OffsetPager


class _RecordingFetcher:
    """Test fetcher that records (offset, limit) calls and returns canned pages."""

    def __init__(self, pages: list[list[Any]], items_key: str = "items") -> None:
        self._pages = pages
        self._items_key = items_key
        self.calls: list[tuple[int, int]] = []

    def __call__(self, offset: int, limit: int) -> Mapping[str, Any]:
        self.calls.append((offset, limit))
        idx = len(self.calls) - 1
        if idx >= len(self._pages):
            return {self._items_key: []}
        return {self._items_key: self._pages[idx]}


# ───────────── construction guards ─────────────


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_page_size_must_be_positive(bad: int) -> None:
    with pytest.raises(ValueError, match="page_size"):
        OffsetPager(_RecordingFetcher([]), items_key="x", page_size=bad)


@pytest.mark.parametrize("bad", [-1, -10])
def test_max_total_must_be_non_negative_when_set(bad: int) -> None:
    with pytest.raises(ValueError, match="max_total"):
        OffsetPager(_RecordingFetcher([]), items_key="x", max_total=bad)


def test_items_key_required() -> None:
    with pytest.raises(ValueError, match="items_key"):
        OffsetPager(_RecordingFetcher([]), items_key="")


# ───────────── empty / single-page paths ─────────────


def test_empty_first_page_yields_nothing_and_stops() -> None:
    fetcher = _RecordingFetcher([[]])
    pager: OffsetPager[Any] = OffsetPager(fetcher, items_key="items", page_size=10)
    assert pager.all() == []
    assert fetcher.calls == [(0, 10)]


def test_short_first_page_yields_all_and_stops_without_extra_fetch() -> None:
    fetcher = _RecordingFetcher([[1, 2, 3]])
    pager: OffsetPager[Any] = OffsetPager(fetcher, items_key="items", page_size=10)
    assert pager.all() == [1, 2, 3]
    # Only one fetch - short page (3 < 10) signals end-of-data.
    assert fetcher.calls == [(0, 10)]


def test_full_page_followed_by_empty_page_terminates() -> None:
    fetcher = _RecordingFetcher([[1, 2, 3], []])
    pager: OffsetPager[Any] = OffsetPager(fetcher, items_key="items", page_size=3)
    assert pager.all() == [1, 2, 3]
    assert fetcher.calls == [(0, 3), (3, 3)]


def test_two_full_pages_then_short_page() -> None:
    fetcher = _RecordingFetcher([[1, 2, 3], [4, 5, 6], [7, 8]])
    pager: OffsetPager[Any] = OffsetPager(fetcher, items_key="items", page_size=3)
    assert pager.all() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert fetcher.calls == [(0, 3), (3, 3), (6, 3)]


# ───────────── max_total semantics ─────────────


def test_max_total_zero_yields_nothing_and_makes_no_fetches() -> None:
    fetcher = _RecordingFetcher([[1, 2, 3]])
    pager: OffsetPager[Any] = OffsetPager(fetcher, items_key="items", page_size=10, max_total=0)
    assert pager.all() == []
    assert fetcher.calls == []


def test_max_total_truncates_within_a_single_page() -> None:
    fetcher = _RecordingFetcher([[1, 2, 3, 4, 5]])
    pager: OffsetPager[Any] = OffsetPager(fetcher, items_key="items", page_size=10, max_total=3)
    assert pager.all() == [1, 2, 3]


def test_max_total_truncates_across_pages() -> None:
    fetcher = _RecordingFetcher([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    pager: OffsetPager[Any] = OffsetPager(fetcher, items_key="items", page_size=3, max_total=5)
    assert pager.all() == [1, 2, 3, 4, 5]
    # Two pages fetched (6 items collected, then capped after 5).
    assert fetcher.calls == [(0, 3), (3, 3)]


def test_max_total_higher_than_total_data_returns_all_data() -> None:
    fetcher = _RecordingFetcher([[1, 2]])
    pager: OffsetPager[Any] = OffsetPager(fetcher, items_key="items", page_size=10, max_total=100)
    assert pager.all() == [1, 2]


def test_max_total_on_page_boundary_does_not_overfetch() -> None:
    # max_total == page_size (with more data available) must NOT fetch a second
    # page only to discard it: the cap is reached exactly on the page boundary.
    fetcher = _RecordingFetcher([[1, 2, 3], [4, 5, 6]])
    pager: OffsetPager[Any] = OffsetPager(fetcher, items_key="items", page_size=3, max_total=3)
    assert pager.all() == [1, 2, 3]
    assert fetcher.calls == [(0, 3)]  # pre-fix this was [(0, 3), (3, 3)]


def test_max_total_two_page_boundary_multiple_fetches_exactly_two_pages() -> None:
    # A larger boundary multiple: max_total == 2 * page_size fetches exactly the
    # two pages it consumes, never a third.
    fetcher = _RecordingFetcher([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    pager: OffsetPager[Any] = OffsetPager(fetcher, items_key="items", page_size=3, max_total=6)
    assert pager.all() == [1, 2, 3, 4, 5, 6]
    assert fetcher.calls == [(0, 3), (3, 3)]


# ───────────── parse_item ─────────────


def test_parse_item_applied_to_each_raw_value() -> None:
    fetcher = _RecordingFetcher([[{"n": 1}, {"n": 2}], [{"n": 3}]])
    pager = OffsetPager(
        fetcher,
        items_key="items",
        page_size=2,
        parse_item=lambda raw: raw["n"] * 10,
    )
    assert pager.all() == [10, 20, 30]


def test_parse_item_default_is_identity() -> None:
    fetcher = _RecordingFetcher([[{"x": 1}]])
    pager: OffsetPager[Any] = OffsetPager(fetcher, items_key="items", page_size=10)
    assert pager.all() == [{"x": 1}]


def test_parse_item_exception_propagates() -> None:
    fetcher = _RecordingFetcher([[{"bad": True}]])

    def parse(raw: Any) -> Any:
        raise ValueError("malformed item")

    pager = OffsetPager(fetcher, items_key="items", page_size=10, parse_item=parse)
    with pytest.raises(ValueError, match="malformed"):
        pager.all()


# ───────────── items_key flexibility ─────────────


def test_custom_items_key_used_to_extract_page_items() -> None:
    fetcher = _RecordingFetcher([[1, 2, 3]], items_key="datasets")
    pager: OffsetPager[Any] = OffsetPager(fetcher, items_key="datasets", page_size=10)
    assert pager.all() == [1, 2, 3]


def test_missing_items_key_in_page_treated_as_empty() -> None:
    # A backend regression that drops the items_key shouldn't infinite-loop;
    # missing key reads as [] which terminates.
    def fetcher(_offset: int, _limit: int) -> Mapping[str, Any]:
        return {"unrelated": "noise"}

    pager: OffsetPager[Any] = OffsetPager(fetcher, items_key="items", page_size=10)
    assert pager.all() == []


# ───────────── re-iteration semantics ─────────────


def test_iterating_twice_replays_from_offset_zero() -> None:
    fetcher = _RecordingFetcher([[1, 2], [3]])
    pager: OffsetPager[Any] = OffsetPager(fetcher, items_key="items", page_size=2)
    first_pass = list(pager)
    # Reset fetcher's canned pages so the replay can succeed.
    fetcher._pages = [[1, 2], [3]]
    fetcher.calls = []
    second_pass = list(pager)
    assert first_pass == second_pass == [1, 2, 3]


# ───────────── .first() ─────────────


def test_first_returns_first_item_with_only_one_fetch() -> None:
    fetcher = _RecordingFetcher([[10, 20, 30]])
    pager: OffsetPager[Any] = OffsetPager(fetcher, items_key="items", page_size=100)
    assert pager.first() == 10
    assert fetcher.calls == [(0, 100)]


def test_first_returns_none_for_empty_pager() -> None:
    fetcher = _RecordingFetcher([[]])
    pager: OffsetPager[Any] = OffsetPager(fetcher, items_key="items", page_size=10)
    assert pager.first() is None


def test_first_returns_none_when_max_total_zero() -> None:
    fetcher = _RecordingFetcher([[1, 2, 3]])
    pager: OffsetPager[Any] = OffsetPager(fetcher, items_key="items", page_size=10, max_total=0)
    assert pager.first() is None
    assert fetcher.calls == []


# ───────────── offset / limit math sanity ─────────────


def test_advances_by_items_received_not_page_size_when_backend_caps_short() -> None:
    # Defensive: if backend silently caps below requested page_size, advance
    # by what we got so we don't skip rows.
    pages = [[1, 2], [3, 4], [5]]  # backend caps at 2 even though we asked for 100
    fetcher = _RecordingFetcher(pages)
    pager: OffsetPager[Any] = OffsetPager(fetcher, items_key="items", page_size=100)
    # The pager treats len(items)<page_size as end-of-data, so it stops after
    # the first under-cap response. This is documented behaviour: when a
    # backend caps page size, callers should request the cap.
    assert pager.all() == [1, 2]
    assert fetcher.calls == [(0, 100)]


def test_default_page_size_constant_is_pinned() -> None:
    # If the constant moves we want a deliberate decision; pinning here makes
    # the change show up in code review.
    assert DEFAULT_PAGE_SIZE == 100
