"""Tests for ``pictograph._http.streaming``: SSE parser + chunked file iter."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from pictograph._http.streaming import (
    DEFAULT_CHUNK_SIZE,
    SSEEvent,
    chunked_file_iterator,
    parse_sse,
)


def _stream(*chunks: str) -> Iterable[bytes]:
    """Convert string chunks to a byte iterable for ``parse_sse``."""
    for c in chunks:
        yield c.encode("utf-8")


# ───────────── single events ─────────────


def test_parse_sse_single_event_with_json_data() -> None:
    events = list(parse_sse(_stream('data: {"x": 1}\n\n')))
    assert len(events) == 1
    assert events[0].event == "message"
    assert events[0].data == {"x": 1}
    assert events[0].id is None
    assert events[0].retry is None


def test_parse_sse_default_event_type_is_message() -> None:
    events = list(parse_sse(_stream("data: hello\n\n")))
    assert events[0].event == "message"


def test_parse_sse_explicit_event_type() -> None:
    events = list(parse_sse(_stream("event: progress\ndata: 50\n\n")))
    assert events[0].event == "progress"
    # "50" is valid JSON, so it parses to int.
    assert events[0].data == 50


def test_parse_sse_id_field_propagates() -> None:
    events = list(parse_sse(_stream("id: 42\ndata: hi\n\n")))
    assert events[0].id == "42"


def test_parse_sse_retry_field_propagates() -> None:
    events = list(parse_sse(_stream("retry: 3000\ndata: hi\n\n")))
    assert events[0].retry == 3000


def test_parse_sse_retry_with_non_integer_value_is_silently_ignored() -> None:
    # Per spec: invalid retry values are ignored without erroring.
    events = list(parse_sse(_stream("retry: soon\ndata: hi\n\n")))
    assert events[0].retry is None


# ───────────── multi-line data ─────────────


def test_parse_sse_multiline_data_joined_with_newline() -> None:
    events = list(parse_sse(_stream("data: line1\ndata: line2\ndata: line3\n\n")))
    assert events[0].data == "line1\nline2\nline3"


def test_parse_sse_data_with_no_space_after_colon() -> None:
    events = list(parse_sse(_stream("data:tight\n\n")))
    assert events[0].data == "tight"


def test_parse_sse_data_strips_only_one_leading_space() -> None:
    # Per spec: strip ONE leading space; preserve the rest.
    events = list(parse_sse(_stream("data:  two-spaces\n\n")))
    assert events[0].data == " two-spaces"


def test_parse_sse_data_with_empty_value_yields_empty_string() -> None:
    events = list(parse_sse(_stream("data:\n\n")))
    assert events[0].data == ""


# ───────────── multiple events ─────────────


def test_parse_sse_two_events_back_to_back() -> None:
    events = list(parse_sse(_stream("data: a\n\ndata: b\n\n")))
    assert [e.data for e in events] == ["a", "b"]


def test_parse_sse_event_split_across_chunks() -> None:
    # The parser must buffer across the chunk boundary mid-event.
    events = list(parse_sse(_stream("event: pro", "gress\ndata: ", '{"v":', " 1}\n", "\n")))
    assert len(events) == 1
    assert events[0].event == "progress"
    assert events[0].data == {"v": 1}


def test_parse_sse_blank_line_split_across_chunks() -> None:
    # The blank-line separator itself can straddle a chunk boundary.
    events = list(parse_sse(_stream("data: x\n", "\ndata: y\n\n")))
    assert [e.data for e in events] == ["x", "y"]


# ───────────── line-ending handling ─────────────


def test_parse_sse_handles_crlf_line_endings() -> None:
    events = list(parse_sse(_stream("data: hello\r\n\r\n")))
    assert events[0].data == "hello"


def test_parse_sse_handles_lone_cr_line_endings() -> None:
    events = list(parse_sse(_stream("data: hello\r\r")))
    assert events[0].data == "hello"


def test_parse_sse_handles_mixed_endings() -> None:
    events = list(parse_sse(_stream("data: a\r\ndata: b\n\r\n")))
    assert events[0].data == "a\nb"


# ───────────── comments / unknown fields ─────────────


def test_parse_sse_comment_lines_ignored() -> None:
    events = list(parse_sse(_stream(": keepalive ping\ndata: real\n\n")))
    assert events[0].data == "real"


def test_parse_sse_unknown_field_ignored() -> None:
    events = list(parse_sse(_stream("custom: ignored\ndata: real\n\n")))
    assert events[0].data == "real"


def test_parse_sse_field_without_colon_treated_as_name_only() -> None:
    # Per spec: a line without a colon is a field name with empty value.
    # ``data`` with empty value is still data - empty event has no payload
    # data line, so it would not dispatch.
    events = list(parse_sse(_stream("data\n\n")))
    assert events[0].data == ""


# ───────────── partial / malformed events ─────────────


def test_parse_sse_event_without_data_not_dispatched() -> None:
    # An event with only metadata (no `data:` line) is not yielded per spec.
    events = list(parse_sse(_stream("event: ping\n\n")))
    assert events == []


def test_parse_sse_partial_trailing_event_not_dispatched() -> None:
    # Stream ends without the terminating blank line - the trailing event
    # text remains in the buffer and never yields. Per spec.
    events = list(parse_sse(_stream("data: incomplete")))
    assert events == []


def test_parse_sse_invalid_json_data_returned_as_raw_string() -> None:
    # Non-JSON data is preserved verbatim - useful for plain-text SSE streams.
    events = list(parse_sse(_stream("data: not{json\n\n")))
    assert events[0].data == "not{json"


def test_parse_sse_empty_chunk_in_stream_does_not_break() -> None:
    # Mid-stream empty chunk (e.g. keepalive, network hiccup) is benign.
    events = list(parse_sse(_stream("data: a\n", "", "\n")))
    assert events[0].data == "a"


def test_parse_sse_unicode_multibyte_decoded_correctly() -> None:
    # Russian + emoji + CJK to exercise multi-byte UTF-8 across chunks.
    events = list(parse_sse(_stream('data: {"text": "Привет 🌍 你好"}\n\n')))
    assert events[0].data == {"text": "Привет 🌍 你好"}


def test_parse_sse_multibyte_char_split_across_byte_chunks() -> None:
    # ``response.iter_bytes()`` breaks at arbitrary byte offsets, so a 4-byte
    # emoji (🌍 == f0 9f 8c 8d) can straddle a chunk boundary. The parser must
    # buffer the partial bytes via an incremental decoder rather than raising
    # UnicodeDecodeError mid-stream (it did before the incremental-decoder fix).
    payload = 'data: {"emoji": "🌍"}\n\n'.encode()
    emoji_start = payload.index(b"\xf0")
    # Split right in the middle of the 4-byte emoji sequence.
    chunks = [payload[: emoji_start + 2], payload[emoji_start + 2 :]]
    events = list(parse_sse(iter(chunks)))
    assert len(events) == 1
    assert events[0].data == {"emoji": "🌍"}


def test_parse_sse_multibyte_char_split_one_byte_per_chunk() -> None:
    # The pathological case: every single byte arrives in its own chunk.
    payload = 'data: "你好🌍"\n\n'.encode()
    chunks = [payload[i : i + 1] for i in range(len(payload))]
    events = list(parse_sse(iter(chunks)))
    assert events[0].data == "你好🌍"


def test_parse_sse_event_field_empty_falls_back_to_message_default() -> None:
    events = list(parse_sse(_stream("event:\ndata: x\n\n")))
    assert events[0].event == "message"


# ───────────── SSEEvent dataclass ─────────────


def test_sse_event_is_frozen() -> None:
    evt = SSEEvent(event="message", data={"x": 1})
    with pytest.raises(Exception):
        evt.event = "other"  # type: ignore[misc]


def test_sse_event_equality_and_hashability() -> None:
    a = SSEEvent(event="m", data="x", id="1")
    b = SSEEvent(event="m", data="x", id="1")
    assert a == b
    # Frozen dataclasses are hashable by default (when all fields are too).
    assert hash(a) == hash(b)


# ───────────── chunked_file_iterator ─────────────


def test_chunked_file_iterator_reads_full_file_in_correct_chunks(
    tmp_path: Path,
) -> None:
    p = tmp_path / "data.bin"
    p.write_bytes(b"abcdefghij")  # 10 bytes
    chunks = list(chunked_file_iterator(p, chunk_size=4))
    assert chunks == [b"abcd", b"efgh", b"ij"]


def test_chunked_file_iterator_progress_callback_receives_running_totals(
    tmp_path: Path,
) -> None:
    p = tmp_path / "data.bin"
    p.write_bytes(b"a" * 10)
    seen: list[tuple[int, int]] = []

    def record(sent: int, total: int) -> None:
        seen.append((sent, total))

    list(chunked_file_iterator(p, chunk_size=3, progress=record))
    assert seen == [(3, 10), (6, 10), (9, 10), (10, 10)]


def test_chunked_file_iterator_total_size_overrides_stat(tmp_path: Path) -> None:
    """When total_size is passed, the progress total uses it verbatim (no second
    stat). upload_external relies on this so the Content-Length header and the
    streamed-body total always agree even if the file is mutated concurrently."""
    p = tmp_path / "data.bin"
    p.write_bytes(b"a" * 10)
    seen: list[tuple[int, int]] = []

    def record(sent: int, total: int) -> None:
        seen.append((sent, total))

    # Pass an authoritative total that differs from the on-disk size.
    list(chunked_file_iterator(p, chunk_size=4, progress=record, total_size=10))
    assert [t for _, t in seen] == [10, 10, 10]  # total comes from total_size, not re-stat


def test_chunked_file_iterator_empty_file_yields_zero_chunks(
    tmp_path: Path,
) -> None:
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    seen: list[tuple[int, int]] = []

    def record(sent: int, total: int) -> None:
        seen.append((sent, total))

    chunks = list(chunked_file_iterator(p, chunk_size=4, progress=record))
    assert chunks == []
    assert seen == []


def test_chunked_file_iterator_file_smaller_than_chunk_yields_one_chunk(
    tmp_path: Path,
) -> None:
    p = tmp_path / "tiny.bin"
    p.write_bytes(b"hi")
    chunks = list(chunked_file_iterator(p, chunk_size=1024))
    assert chunks == [b"hi"]


def test_chunked_file_iterator_file_exactly_chunk_size_yields_one_chunk(
    tmp_path: Path,
) -> None:
    p = tmp_path / "exact.bin"
    p.write_bytes(b"x" * 8)
    chunks = list(chunked_file_iterator(p, chunk_size=8))
    assert chunks == [b"x" * 8]


@pytest.mark.parametrize("bad", [0, -1, -1024])
def test_chunked_file_iterator_rejects_non_positive_chunk_size(tmp_path: Path, bad: int) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"data")
    with pytest.raises(ValueError, match="chunk_size"):
        # Drain the iterator to surface the eager validation.
        list(chunked_file_iterator(p, chunk_size=bad))


def test_chunked_file_iterator_progress_optional(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    chunks = list(chunked_file_iterator(p, chunk_size=2))
    assert chunks == [b"ab", b"c"]


def test_chunked_file_iterator_default_chunk_size_constant_is_pinned() -> None:
    assert DEFAULT_CHUNK_SIZE == 8 * 1024 * 1024
