"""Tests for chunkers."""
from __future__ import annotations

import pytest

from ragcheck.chunkers import (
    Chunk,
    FixedTokenChunker,
    SemanticBoundaryChunker,
    SentenceChunker,
    SlidingWindowChunker,
    StructuralMarkdownChunker,
    get_chunker,
    list_chunkers,
    register_chunker,
    split_sentences,
)


class TestFixedTokenChunker:
    def test_single_short_doc(self):
        c = FixedTokenChunker(tokens_per_chunk=100)
        chunks = c.chunk("d1", "one two three four five")
        assert len(chunks) == 1
        assert chunks[0].doc_id == "d1"
        assert chunks[0].text == "one two three four five"

    def test_multi_chunk_no_overlap(self):
        c = FixedTokenChunker(tokens_per_chunk=2, overlap=0)
        chunks = c.chunk("d1", "a b c d e")
        assert [ch.text for ch in chunks] == ["a b", "c d", "e"]

    def test_multi_chunk_with_overlap(self):
        c = FixedTokenChunker(tokens_per_chunk=3, overlap=1)
        chunks = c.chunk("d1", "a b c d e")
        texts = [ch.text for ch in chunks]
        assert texts[0] == "a b c"
        assert texts[1].startswith("c")

    def test_empty_text(self):
        c = FixedTokenChunker(tokens_per_chunk=5)
        assert c.chunk("d1", "") == []

    def test_whitespace_only(self):
        c = FixedTokenChunker(tokens_per_chunk=5)
        assert c.chunk("d1", "   \n\t  ") == []

    def test_deterministic_ids(self):
        c = FixedTokenChunker(tokens_per_chunk=3)
        a = c.chunk("d1", "one two three four")
        b = c.chunk("d1", "one two three four")
        assert [ch.chunk_id for ch in a] == [ch.chunk_id for ch in b]

    def test_invalid_tokens_per_chunk(self):
        with pytest.raises(ValueError):
            FixedTokenChunker(tokens_per_chunk=0)

    def test_invalid_overlap_negative(self):
        with pytest.raises(ValueError):
            FixedTokenChunker(tokens_per_chunk=5, overlap=-1)

    def test_invalid_overlap_too_large(self):
        with pytest.raises(ValueError):
            FixedTokenChunker(tokens_per_chunk=3, overlap=3)

    def test_name(self):
        assert FixedTokenChunker().name == "fixed-token"


class TestSlidingWindowChunker:
    def test_covers_whole_text(self):
        c = SlidingWindowChunker(size=5, stride=3)
        text = "abcdefghij"  # 10 chars
        chunks = c.chunk("d1", text)
        # chunks at positions 0, 3, 6, 9 → windows "abcde","defgh","ghij","j"
        assert chunks[0].text == "abcde"
        assert chunks[-1].end == 10

    def test_stride_equals_size(self):
        c = SlidingWindowChunker(size=3, stride=3)
        chunks = c.chunk("d1", "abcdef")
        assert len(chunks) == 2
        assert chunks[0].text == "abc"
        assert chunks[1].text == "def"

    def test_empty(self):
        assert SlidingWindowChunker(size=5, stride=2).chunk("d1", "") == []

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            SlidingWindowChunker(size=0, stride=1)

    def test_invalid_stride_zero(self):
        with pytest.raises(ValueError):
            SlidingWindowChunker(size=5, stride=0)

    def test_stride_larger_than_size_rejected(self):
        with pytest.raises(ValueError):
            SlidingWindowChunker(size=3, stride=5)

    def test_name(self):
        assert SlidingWindowChunker().name == "sliding-window"


class TestSplitSentences:
    def test_basic(self):
        result = split_sentences("One. Two. Three.")
        assert result == ["One.", "Two.", "Three."]

    def test_handles_exclamation(self):
        result = split_sentences("Wow! Really? Yes.")
        assert result == ["Wow!", "Really?", "Yes."]

    def test_does_not_split_on_abbreviation(self):
        result = split_sentences("Dr. Smith examined the patient. It went well.")
        assert len(result) == 2

    def test_does_not_split_on_eg(self):
        result = split_sentences("Many languages, e.g. Python, Go. This is fine.")
        assert len(result) == 2

    def test_empty(self):
        assert split_sentences("") == []

    def test_whitespace_only(self):
        assert split_sentences("   \n\n   ") == []


class TestSentenceChunker:
    def test_groups_sentences_below_max(self):
        c = SentenceChunker(max_chars=100)
        chunks = c.chunk("d1", "One. Two. Three. Four.")
        assert len(chunks) == 1

    def test_splits_at_max_chars(self):
        c = SentenceChunker(max_chars=15, min_chars=5)
        text = "A short one. " * 5
        chunks = c.chunk("d1", text)
        assert len(chunks) > 1

    def test_empty(self):
        assert SentenceChunker(max_chars=100).chunk("d1", "") == []

    def test_invalid_max_chars(self):
        with pytest.raises(ValueError):
            SentenceChunker(max_chars=0)

    def test_invalid_min_chars_negative(self):
        with pytest.raises(ValueError):
            SentenceChunker(max_chars=10, min_chars=-1)

    def test_invalid_min_ge_max(self):
        with pytest.raises(ValueError):
            SentenceChunker(max_chars=10, min_chars=10)

    def test_name(self):
        assert SentenceChunker().name == "sentence"


class TestSemanticBoundaryChunker:
    def test_paragraph_breaks_hard_cut(self):
        text = "Paragraph one has content.\n\nParagraph two has different content."
        c = SemanticBoundaryChunker(max_chars=1000)
        chunks = c.chunk("d1", text)
        assert len(chunks) == 2

    def test_long_paragraph_is_capped(self):
        text = ("This is sentence one. " * 30).strip()
        c = SemanticBoundaryChunker(max_chars=100)
        chunks = c.chunk("d1", text)
        assert len(chunks) > 1

    def test_empty(self):
        assert SemanticBoundaryChunker(max_chars=100).chunk("d1", "") == []

    def test_invalid_max_chars(self):
        with pytest.raises(ValueError):
            SemanticBoundaryChunker(max_chars=0)

    def test_name(self):
        assert SemanticBoundaryChunker().name == "semantic-boundary"


class TestStructuralMarkdownChunker:
    def test_each_heading_is_own_chunk(self):
        md = "# Heading A\ncontent a\n\n# Heading B\ncontent b"
        c = StructuralMarkdownChunker(max_chars=1000)
        chunks = c.chunk("d1", md)
        assert len(chunks) == 2
        assert any("Heading A" in ch.meta.get("heading", "") for ch in chunks)

    def test_no_headings_fallback(self):
        c = StructuralMarkdownChunker(max_chars=1000)
        chunks = c.chunk("d1", "Just some plain text without headings.")
        assert len(chunks) >= 1

    def test_empty(self):
        assert StructuralMarkdownChunker().chunk("d1", "") == []

    def test_long_section_gets_capped(self):
        body = ("Filler sentence. " * 80)
        md = f"# Big\n{body}"
        c = StructuralMarkdownChunker(max_chars=200)
        chunks = c.chunk("d1", md)
        assert len(chunks) > 1

    def test_invalid_max_chars(self):
        with pytest.raises(ValueError):
            StructuralMarkdownChunker(max_chars=0)

    def test_name(self):
        assert StructuralMarkdownChunker().name == "structural-markdown"


class TestRegistry:
    def test_builtins_registered(self):
        assert "fixed-token" in list_chunkers()
        assert "sliding-window" in list_chunkers()
        assert "sentence" in list_chunkers()
        assert "semantic-boundary" in list_chunkers()
        assert "structural-markdown" in list_chunkers()

    def test_list_is_sorted(self):
        assert list_chunkers() == sorted(list_chunkers())

    def test_get_chunker_by_name(self):
        c = get_chunker("fixed-token", tokens_per_chunk=10)
        assert isinstance(c, FixedTokenChunker)

    def test_get_chunker_unknown(self):
        with pytest.raises(KeyError):
            get_chunker("no-such-chunker")

    def test_get_chunker_case_insensitive(self):
        c = get_chunker("FIXED-TOKEN")
        assert isinstance(c, FixedTokenChunker)

    def test_register_custom(self):
        class Null:
            name = "null"
            def chunk(self, doc_id, text):
                return []
        register_chunker("null-test", Null)
        assert "null-test" in list_chunkers()
        assert get_chunker("null-test").name == "null"

    def test_register_empty_name_rejected(self):
        with pytest.raises(ValueError):
            register_chunker("", lambda: None)


class TestChunkDataclass:
    def test_chunk_is_frozen(self):
        ch = Chunk(doc_id="d", chunk_id="c", text="t", start=0, end=1)
        with pytest.raises(Exception):  # dataclass FrozenInstanceError
            ch.doc_id = "x"  # type: ignore

    def test_chunk_defaults(self):
        ch = Chunk(doc_id="d", chunk_id="c", text="t", start=0, end=1)
        assert ch.meta == {}
