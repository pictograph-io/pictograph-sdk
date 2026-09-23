"""Auto-paging iterators over offset/limit list endpoints.

The Pictograph developer API uses ``?limit=N&offset=M`` query parameters and
returns ``{<resource_key>: [...]}``-shaped responses - the house-rules
envelope is ``{"data": [...], "pagination": {..., "has_more": bool}}``; some
legacy endpoints still use a bespoke key (``{"models": [...]}``).
:class:`OffsetPager` wraps both so resource methods can expose
``client.datasets.iter()`` without callers writing offset math:

    >>> for ds in client.datasets.iter():  # doctest: +SKIP
    ...     print(ds.name)

Termination rules:

- A server-computed ``pagination.has_more == False`` ends iteration after the
  page (authoritative when present - the standard envelope).
- An empty page (``items == []``) ends iteration.
- A short page (``len(items) < page_size``) ends iteration after yielding
  those items - saves one round-trip on the common "last page" case.
- The optional ``max_total`` argument truncates iteration once N items have
  been yielded, even mid-page.

A user-supplied ``parse_item`` callable transforms each raw dict into a
typed object (typically a Pydantic model). Without it, raw dicts are yielded.

The iterator is single-use per ``__iter__`` call: each call to ``iter(pager)``
restarts at offset 0 with a fresh sequence of fetches. Callers needing
random-access semantics should materialize via :meth:`OffsetPager.all`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping

T = TypeVar("T")

DEFAULT_PAGE_SIZE = 100
"""Reasonable default; backend hard-caps at 1000."""


def _page_has_more(page: Mapping[str, Any]) -> bool | None:
    """Read the server-computed ``pagination.has_more`` flag, if present.

    Returns ``None`` for legacy envelopes without a pagination object (the
    caller falls back to the empty/short-page heuristics).
    """
    pagination = page.get("pagination")
    if isinstance(pagination, dict):
        has_more = pagination.get("has_more")
        if isinstance(has_more, bool):
            return has_more
    return None


class OffsetPager(Generic[T]):
    """Iterate over items spanning multiple offset/limit-paginated responses."""

    def __init__(
        self,
        fetch_page: Callable[[int, int], Mapping[str, Any]],
        *,
        items_key: str,
        page_size: int = DEFAULT_PAGE_SIZE,
        parse_item: Callable[[Any], T] | None = None,
        max_total: int | None = None,
    ) -> None:
        if page_size <= 0:
            raise ValueError(f"page_size must be > 0, got {page_size}")
        if max_total is not None and max_total < 0:
            raise ValueError(f"max_total must be >= 0 when set, got {max_total}")
        if not items_key:
            raise ValueError("items_key must be a non-empty string")
        self._fetch_page = fetch_page
        self._items_key = items_key
        self._page_size = page_size
        self._max_total = max_total
        self._parse_item: Callable[[Any], T] = parse_item or (lambda raw: raw)

    def __iter__(self) -> Iterator[T]:
        if self._max_total == 0:
            return
        offset = 0
        yielded = 0
        while True:
            page = self._fetch_page(offset, self._page_size)
            items = page.get(self._items_key, [])
            if not items:
                return
            for raw in items:
                if self._max_total is not None and yielded >= self._max_total:
                    return
                yield self._parse_item(raw)
                yielded += 1
            # Cap reached on a page boundary → don't fetch a page we'd discard.
            # Without this, a max_total that is an exact multiple of page_size
            # fetches one wasted extra page (a round-trip + a rate-limit token).
            if self._max_total is not None and yielded >= self._max_total:
                return
            # Server-computed has_more (standard envelope) is authoritative when
            # present - stops exactly at the org-wide total, saving the final
            # empty-page round-trip even on an exact page-size boundary.
            if _page_has_more(page) is False:
                return
            # Short page → no more pages exist; save a round-trip.
            if len(items) < self._page_size:
                return
            # Advance by the number we actually received (defends against a
            # backend that silently caps page size).
            offset += len(items)

    def all(self) -> list[T]:
        """Materialize the entire (potentially multi-page) result set."""
        return list(self)

    def first(self) -> T | None:
        """Return the first item, or ``None`` if the result set is empty.

        Stops iteration after the first item - only one page is fetched if
        non-empty, zero pages if ``max_total`` is 0.
        """
        for item in self:
            return item
        return None


class AsyncOffsetPager(Generic[T]):
    """Async twin of :class:`OffsetPager` - ``async for`` over paginated pages.

    Termination rules and cap semantics are identical to the sync pager; only
    the page fetch is awaited. Use it as::

        async for ds in client.datasets.iter():  # doctest: +SKIP
            print(ds.name)

    or materialise via ``await pager.all()`` / peek with ``await pager.first()``.
    """

    def __init__(
        self,
        fetch_page: Callable[[int, int], Awaitable[Mapping[str, Any]]],
        *,
        items_key: str,
        page_size: int = DEFAULT_PAGE_SIZE,
        parse_item: Callable[[Any], T] | None = None,
        max_total: int | None = None,
    ) -> None:
        if page_size <= 0:
            raise ValueError(f"page_size must be > 0, got {page_size}")
        if max_total is not None and max_total < 0:
            raise ValueError(f"max_total must be >= 0 when set, got {max_total}")
        if not items_key:
            raise ValueError("items_key must be a non-empty string")
        self._fetch_page = fetch_page
        self._items_key = items_key
        self._page_size = page_size
        self._max_total = max_total
        self._parse_item: Callable[[Any], T] = parse_item or (lambda raw: raw)

    async def __aiter__(self) -> AsyncIterator[T]:
        if self._max_total == 0:
            return
        offset = 0
        yielded = 0
        while True:
            page = await self._fetch_page(offset, self._page_size)
            items = page.get(self._items_key, [])
            if not items:
                return
            for raw in items:
                if self._max_total is not None and yielded >= self._max_total:
                    return
                yield self._parse_item(raw)
                yielded += 1
            if self._max_total is not None and yielded >= self._max_total:
                return
            if _page_has_more(page) is False:
                return
            if len(items) < self._page_size:
                return
            offset += len(items)

    async def all(self) -> list[T]:
        """Materialize the entire (potentially multi-page) result set."""
        return [item async for item in self]

    async def first(self) -> T | None:
        """Return the first item, or ``None`` if the result set is empty."""
        async for item in self:
            return item
        return None
