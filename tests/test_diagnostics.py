"""Tests for chunking diagnostics."""
from __future__ import annotations

import pytest

from ragcheck.chunkers import Chunk
from ragcheck.diagnostics import (
    ChunkingDiagnostics,
    chunking_diagnostics,
    format_histogram_ascii,
)


def _chunk(doc: str, cid: str, text: str, start: int = 0, end: int = 0) -> Chunk:
    return Chunk(doc_id=doc, chunk_id=cid, text=text, start=start, end=end)


class TestChunkingDiagnostics:
    def test_empty_input(self):
        d = chunking_diagnostics([])
        assert d.total_chunks == 0
        assert d.total_docs == 0
        assert d.coverage == 0.0

    def test_basic_counts(self):
        chunks = [
            _chunk("d1", "c1", "hello world"),
            _chunk("d1", "c2", "another chunk"),
            _chunk("d2", "c3", "second doc"),
        ]
        d = chunking_diagnostics(chunks)
        assert d.total_chunks == 3
        assert d.total_docs == 2
        assert d.unique_chunks == 3
        assert d.duplicate_chunks == 0

    def test_duplicate_detection(self):
        chunks = [
            _chunk("d1", "c1", "repeated text"),
            _chunk("d2", "c2", "repeated text"),
            _chunk("d3", "c3", "different text"),
        ]
        d = chunking_diagnostics(chunks)
        assert d.unique_chunks == 2
        assert d.duplicate_chunks == 1
        assert d.duplicate_ratio == pytest.approx(1 / 3, rel=1e-6)

    def test_duplicate_whitespace_normalisation(self):
        chunks = [
            _chunk("d1", "c1", "hello  world"),
            _chunk("d2", "c2", "hello world  "),
        ]
        d = chunking_diagnostics(chunks)
        assert d.unique_chunks == 1
        assert d.duplicate_chunks == 1

    def test_size_histogram_buckets(self):
        chunks = [
            _chunk("d1", "c1", "x" * 30),
            _chunk("d1", "c2", "x" * 60),
            _chunk("d1", "c3", "x" * 250),
            _chunk("d1", "c4", "x" * 3000),
        ]
        d = chunking_diagnostics(chunks)
        assert d.size_histogram["0-49"] == 1
        assert d.size_histogram["50-99"] == 1
        assert d.size_histogram["200-499"] == 1
        assert d.size_histogram["2000+"] == 1

    def test_avg_and_median(self):
        chunks = [
            _chunk("d1", "c1", "x" * 10),
            _chunk("d1", "c2", "x" * 20),
            _chunk("d1", "c3", "x" * 30),
        ]
        d = chunking_diagnostics(chunks)
        assert d.avg_chunk_chars == pytest.approx(20.0)
        assert d.median_chunk_chars == 20.0
        assert d.min_chunk_chars == 10
        assert d.max_chunk_chars == 30

    def test_median_even_count(self):
        chunks = [
            _chunk("d1", f"c{i}", "x" * n)
            for i, n in enumerate([10, 20, 30, 40])
        ]
        d = chunking_diagnostics(chunks)
        assert d.median_chunk_chars == 25.0

    def test_coverage_full(self):
        chunks = [
            _chunk("d1", "c1", "a"),
            _chunk("d1", "c2", "b"),
        ]
        d = chunking_diagnostics(chunks, gold_relevant_chunk_ids={"c1", "c2"})
        assert d.coverage == 1.0

    def test_coverage_partial(self):
        chunks = [_chunk("d1", "c1", "a")]
        d = chunking_diagnostics(chunks, gold_relevant_chunk_ids={"c1", "missing"})
        assert d.coverage == 0.5

    def test_coverage_empty_gold_defaults_to_one(self):
        chunks = [_chunk("d1", "c1", "a")]
        d = chunking_diagnostics(chunks)
        assert d.coverage == 1.0

    def test_orphan_chunks(self):
        chunks = [
            _chunk("d1", "c1", "a"),
            _chunk("d1", "c2", "b"),
            _chunk("d1", "c3", "c"),
        ]
        d = chunking_diagnostics(chunks, reachable_chunk_ids={"c1"})
        assert d.orphan_chunk_count == 2
        assert d.orphan_chunk_ratio == pytest.approx(2 / 3)

    def test_orphan_no_reachable_means_unmeasured(self):
        chunks = [_chunk("d1", "c1", "a")]
        d = chunking_diagnostics(chunks)
        assert d.orphan_chunk_count == 0
        assert d.orphan_chunk_ratio == 0.0

    def test_empty_chunks_counted(self):
        chunks = [
            _chunk("d1", "c1", ""),
            _chunk("d1", "c2", "real"),
        ]
        d = chunking_diagnostics(chunks)
        assert d.empty_chunks == 1

    def test_to_dict_keys_sorted(self):
        chunks = [_chunk("d1", "c1", "a" * 100)]
        d = chunking_diagnostics(chunks)
        out = d.to_dict()
        keys = list(out["size_histogram"].keys())
        assert keys == sorted(keys)

    def test_to_dict_rounds_floats(self):
        chunks = [_chunk("d1", "c1", "a" * 7), _chunk("d1", "c2", "b" * 13)]
        d = chunking_diagnostics(chunks)
        out = d.to_dict()
        assert isinstance(out["avg_chunk_chars"], float)

    def test_only_empty_chunks(self):
        chunks = [_chunk("d1", "c1", ""), _chunk("d1", "c2", "")]
        d = chunking_diagnostics(chunks)
        assert d.avg_chunk_chars == 0.0
        assert d.min_chunk_chars == 0
        assert d.max_chunk_chars == 0


class TestHistogramFormatter:
    def test_empty(self):
        assert format_histogram_ascii({}) == "(empty)"

    def test_shows_bar_width_proportional(self):
        out = format_histogram_ascii({"a": 2, "b": 10}, width=20)
        assert "a" in out
        assert "b" in out
        # The row with higher count should have more # characters
        lines = out.split("\n")
        a_line = next(l for l in lines if l.startswith("a"))
        b_line = next(l for l in lines if l.startswith("b"))
        assert a_line.count("#") < b_line.count("#")

    def test_canonical_order_when_buckets_match(self):
        hist = {"0-49": 1, "50-99": 2, "100-199": 3}
        out = format_histogram_ascii(hist)
        # First line should be 0-49
        assert out.split("\n")[0].startswith("0-49")

    def test_non_canonical_fallback_alphabetical(self):
        out = format_histogram_ascii({"zeta": 1, "alpha": 2})
        assert out.split("\n")[0].startswith("alpha")
