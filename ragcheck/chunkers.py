"""Chunking strategies for splitting documents into retrievable units.

Ships five built-in chunkers and a pluggable registry so users can add their
own without patching the package.

All chunkers:
- Take a single string `text` plus chunker-specific config
- Return a list of `Chunk` dataclasses with deterministic `chunk_id`
- Produce stable output across runs given the same input
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class Chunk:
    """A single chunk of a document.

    chunk_id is a content-addressed sha1 prefix of doc_id + start + end, so
    two runs of the same chunker on the same input produce identical ids.
    """

    doc_id: str
    chunk_id: str
    text: str
    start: int
    end: int
    meta: Dict[str, str] = field(default_factory=dict)


def _make_chunk_id(doc_id: str, start: int, end: int, suffix: str = "") -> str:
    payload = f"{doc_id}:{start}:{end}:{suffix}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


class Chunker(Protocol):
    """Protocol implemented by all chunkers.

    A chunker takes a doc_id and text and returns a list of Chunk. Chunker
    instances are expected to be pure — they may not read from disk or the
    network.
    """

    name: str

    def chunk(self, doc_id: str, text: str) -> List[Chunk]:  # pragma: no cover
        ...


class FixedTokenChunker:
    """Split text into chunks of approximately `tokens_per_chunk` whitespace-tokens.

    This is deliberately a simple whitespace tokenizer — it avoids the
    tiktoken dependency while producing results close enough to byte-pair
    encoding for retrieval evaluation purposes. Users who need BPE exact
    counts can plug in their own chunker.
    """

    name = "fixed-token"

    def __init__(self, tokens_per_chunk: int = 64, overlap: int = 0) -> None:
        if tokens_per_chunk <= 0:
            raise ValueError(f"tokens_per_chunk must be > 0, got {tokens_per_chunk}")
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        if overlap >= tokens_per_chunk:
            raise ValueError(
                f"overlap ({overlap}) must be < tokens_per_chunk ({tokens_per_chunk})"
            )
        self.tokens_per_chunk = tokens_per_chunk
        self.overlap = overlap

    def chunk(self, doc_id: str, text: str) -> List[Chunk]:
        if not text:
            return []
        tokens: List[str] = []
        spans: List[tuple] = []  # (start_char, end_char)
        for match in re.finditer(r"\S+", text):
            tokens.append(match.group(0))
            spans.append((match.start(), match.end()))
        if not tokens:
            return []
        out: List[Chunk] = []
        step = self.tokens_per_chunk - self.overlap
        idx = 0
        while idx < len(tokens):
            end_idx = min(idx + self.tokens_per_chunk, len(tokens))
            start_char = spans[idx][0]
            end_char = spans[end_idx - 1][1]
            chunk_text = text[start_char:end_char]
            cid = _make_chunk_id(doc_id, start_char, end_char, "fixed")
            out.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=cid,
                    text=chunk_text,
                    start=start_char,
                    end=end_char,
                    meta={"chunker": self.name},
                )
            )
            if end_idx == len(tokens):
                break
            idx += step
        return out


class SlidingWindowChunker:
    """Character-based sliding window chunker.

    Splits text into overlapping windows of `size` characters with `stride`
    characters between window starts. Useful when you need dense coverage
    with a known amount of redundancy.
    """

    name = "sliding-window"

    def __init__(self, size: int = 400, stride: int = 200) -> None:
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")
        if stride > size:
            raise ValueError(f"stride ({stride}) must be <= size ({size})")
        self.size = size
        self.stride = stride

    def chunk(self, doc_id: str, text: str) -> List[Chunk]:
        if not text:
            return []
        out: List[Chunk] = []
        pos = 0
        n = len(text)
        while pos < n:
            end = min(pos + self.size, n)
            chunk_text = text[pos:end]
            cid = _make_chunk_id(doc_id, pos, end, "sw")
            out.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=cid,
                    text=chunk_text,
                    start=pos,
                    end=end,
                    meta={"chunker": self.name},
                )
            )
            if end == n:
                break
            pos += self.stride
        return out


# Sentence splitter: handles common terminators. We avoid variable-width
# lookbehinds (which Python's `re` does not support) by post-filtering
# candidate splits for abbreviation safety.
_SENTENCE_CANDIDATE = re.compile(r"(?<=[\.!?])\s+(?=[A-Z\"'(\[])")

_ABBREV_SUFFIXES = {
    "mr", "mrs", "ms", "dr", "jr", "sr", "st", "etc", "vs", "eg", "ie",
    "cf", "al", "fig", "no", "inc", "ltd", "co", "corp",
}


def split_sentences(text: str) -> List[str]:
    """Simple sentence splitter with abbreviation-aware filtering.

    Returns a list of non-empty sentence strings. We walk candidate split
    points and drop any that land immediately after a known abbreviation so
    "e.g.", "Dr.", "vs." etc. don't incorrectly end a sentence.
    """
    if not text.strip():
        return []
    segments: List[str] = []
    last = 0
    for match in _SENTENCE_CANDIDATE.finditer(text):
        pre = text[:match.start()]
        # Only '.' terminators (not '!' or '?') can hide an abbreviation.
        if pre.endswith("."):
            tail = pre[:-1]  # strip the single trailing dot
            word_match = re.search(r"(\w+)$", tail)
            if word_match:
                word = word_match.group(1).lower()
                if word in _ABBREV_SUFFIXES:
                    continue
                # Short single-segment token separated by another dot (e.g. "e.g",
                # "i.e", "U.S") — treat as abbreviation.
                if len(word) <= 2 and tail[:-len(word)].endswith("."):
                    continue
                if len(word) == 1 and word.isalpha():
                    # Single-letter initial like "J. Smith" — skip
                    continue
        seg = text[last:match.start()].strip()
        if seg:
            segments.append(seg)
        last = match.end()
    tail_seg = text[last:].strip()
    if tail_seg:
        segments.append(tail_seg)
    return segments


class SentenceChunker:
    """Group consecutive sentences into chunks of approximately `max_chars` each."""

    name = "sentence"

    def __init__(self, max_chars: int = 400, min_chars: int = 50) -> None:
        if max_chars <= 0:
            raise ValueError(f"max_chars must be > 0, got {max_chars}")
        if min_chars < 0:
            raise ValueError(f"min_chars must be >= 0, got {min_chars}")
        if min_chars >= max_chars:
            raise ValueError(f"min_chars ({min_chars}) must be < max_chars ({max_chars})")
        self.max_chars = max_chars
        self.min_chars = min_chars

    def chunk(self, doc_id: str, text: str) -> List[Chunk]:
        if not text.strip():
            return []
        sentences = split_sentences(text)
        if not sentences:
            return []
        out: List[Chunk] = []
        cursor = 0  # char position in text
        buffer: List[str] = []
        buf_start: Optional[int] = None
        buf_end: Optional[int] = None
        for sent in sentences:
            idx = text.find(sent, cursor)
            if idx < 0:
                idx = cursor
            sent_end = idx + len(sent)
            cursor = sent_end
            if buf_start is None:
                buf_start = idx
            buf_end = sent_end
            buffer.append(sent)
            current_len = (buf_end or 0) - (buf_start or 0)
            if current_len >= self.max_chars:
                chunk_text = text[buf_start:buf_end]
                cid = _make_chunk_id(doc_id, buf_start, buf_end, "sent")
                out.append(
                    Chunk(
                        doc_id=doc_id,
                        chunk_id=cid,
                        text=chunk_text,
                        start=buf_start,
                        end=buf_end,
                        meta={"chunker": self.name},
                    )
                )
                buffer = []
                buf_start = None
                buf_end = None
        if buffer and buf_start is not None and buf_end is not None:
            if (buf_end - buf_start) >= self.min_chars or not out:
                chunk_text = text[buf_start:buf_end]
                cid = _make_chunk_id(doc_id, buf_start, buf_end, "sent")
                out.append(
                    Chunk(
                        doc_id=doc_id,
                        chunk_id=cid,
                        text=chunk_text,
                        start=buf_start,
                        end=buf_end,
                        meta={"chunker": self.name},
                    )
                )
            elif out:
                # Attach small trailing buffer to the previous chunk
                prev = out[-1]
                new_end = buf_end
                new_text = text[prev.start:new_end]
                new_cid = _make_chunk_id(doc_id, prev.start, new_end, "sent")
                out[-1] = Chunk(
                    doc_id=doc_id,
                    chunk_id=new_cid,
                    text=new_text,
                    start=prev.start,
                    end=new_end,
                    meta=prev.meta,
                )
        return out


class SemanticBoundaryChunker:
    """Paragraph-aware chunker that splits on blank lines then caps length.

    This is a heuristic, not an embedding-similarity clusterer — the goal is
    to respect natural prose boundaries while keeping chunks below a size
    ceiling. Two blank lines (a paragraph break) is a hard cut. Within a
    paragraph, we only split if the running buffer exceeds `max_chars`.
    """

    name = "semantic-boundary"

    def __init__(self, max_chars: int = 500) -> None:
        if max_chars <= 0:
            raise ValueError(f"max_chars must be > 0, got {max_chars}")
        self.max_chars = max_chars

    def chunk(self, doc_id: str, text: str) -> List[Chunk]:
        if not text.strip():
            return []
        out: List[Chunk] = []
        paragraphs: List[tuple] = []  # (start, end, text)
        for match in re.finditer(r"[^\n]+(?:\n[^\n]+)*", text):
            paragraphs.append((match.start(), match.end(), match.group(0)))
        for p_start, p_end, p_text in paragraphs:
            if len(p_text) <= self.max_chars:
                cid = _make_chunk_id(doc_id, p_start, p_end, "sb")
                out.append(
                    Chunk(
                        doc_id=doc_id,
                        chunk_id=cid,
                        text=p_text,
                        start=p_start,
                        end=p_end,
                        meta={"chunker": self.name},
                    )
                )
                continue
            # Cap long paragraphs by sentence splitting
            sentences = split_sentences(p_text)
            buffer_text: List[str] = []
            buffer_start = p_start
            offset = 0
            for sent in sentences:
                local_idx = p_text.find(sent, offset)
                if local_idx < 0:
                    local_idx = offset
                offset = local_idx + len(sent)
                buffer_text.append(sent)
                joined = " ".join(buffer_text)
                if len(joined) >= self.max_chars:
                    chunk_end = p_start + offset
                    chunk_text = text[buffer_start:chunk_end]
                    cid = _make_chunk_id(doc_id, buffer_start, chunk_end, "sb")
                    out.append(
                        Chunk(
                            doc_id=doc_id,
                            chunk_id=cid,
                            text=chunk_text,
                            start=buffer_start,
                            end=chunk_end,
                            meta={"chunker": self.name},
                        )
                    )
                    buffer_start = chunk_end
                    buffer_text = []
            if buffer_text:
                chunk_end = p_end
                chunk_text = text[buffer_start:chunk_end]
                cid = _make_chunk_id(doc_id, buffer_start, chunk_end, "sb")
                out.append(
                    Chunk(
                        doc_id=doc_id,
                        chunk_id=cid,
                        text=chunk_text,
                        start=buffer_start,
                        end=chunk_end,
                        meta={"chunker": self.name},
                    )
                )
        return out


class StructuralMarkdownChunker:
    """Markdown-aware chunker: each heading section becomes its own chunk.

    Splits on ATX headings (#, ##, ###, ####). Subsections are kept with their
    parent heading as a prefix (first 60 chars) in `meta['heading']`. Long
    sections are capped at `max_chars` using a SemanticBoundaryChunker-style
    secondary split.
    """

    name = "structural-markdown"

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

    def __init__(self, max_chars: int = 800) -> None:
        if max_chars <= 0:
            raise ValueError(f"max_chars must be > 0, got {max_chars}")
        self.max_chars = max_chars

    def chunk(self, doc_id: str, text: str) -> List[Chunk]:
        if not text.strip():
            return []
        headings = list(self._HEADING_RE.finditer(text))
        if not headings:
            # No headings → fall back to semantic-boundary behavior on whole doc
            fallback = SemanticBoundaryChunker(max_chars=self.max_chars)
            chunks = fallback.chunk(doc_id, text)
            return [
                Chunk(
                    doc_id=c.doc_id,
                    chunk_id=c.chunk_id,
                    text=c.text,
                    start=c.start,
                    end=c.end,
                    meta={"chunker": self.name, "heading": ""},
                )
                for c in chunks
            ]
        out: List[Chunk] = []
        for i, h in enumerate(headings):
            section_start = h.start()
            section_end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            heading_text = h.group(2).strip()[:60]
            section_text = text[section_start:section_end].rstrip()
            if len(section_text) <= self.max_chars:
                cid = _make_chunk_id(doc_id, section_start, section_end, "md")
                out.append(
                    Chunk(
                        doc_id=doc_id,
                        chunk_id=cid,
                        text=section_text,
                        start=section_start,
                        end=section_end,
                        meta={"chunker": self.name, "heading": heading_text},
                    )
                )
                continue
            # Section too long: cap by semantic-boundary
            inner = SemanticBoundaryChunker(max_chars=self.max_chars)
            for c in inner.chunk(doc_id, section_text):
                abs_start = section_start + c.start
                abs_end = section_start + c.end
                cid = _make_chunk_id(doc_id, abs_start, abs_end, "md")
                out.append(
                    Chunk(
                        doc_id=doc_id,
                        chunk_id=cid,
                        text=text[abs_start:abs_end],
                        start=abs_start,
                        end=abs_end,
                        meta={"chunker": self.name, "heading": heading_text},
                    )
                )
        return out


# --- Registry -----------------------------------------------------------------

_REGISTRY: Dict[str, Callable[..., Chunker]] = {}


def register_chunker(name: str, factory: Callable[..., Chunker]) -> None:
    """Register a chunker factory under a string name.

    Factories are callables that accept keyword arguments and return a Chunker
    instance. Names are case-insensitive and trimmed.
    """
    key = name.strip().lower()
    if not key:
        raise ValueError("chunker name must be non-empty")
    _REGISTRY[key] = factory


def get_chunker(name: str, **kwargs) -> Chunker:
    """Construct a chunker by registered name."""
    key = name.strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"unknown chunker: {name!r}. available: {available}")
    return _REGISTRY[key](**kwargs)


def list_chunkers() -> List[str]:
    """Return sorted list of registered chunker names."""
    return sorted(_REGISTRY.keys())


# Register built-ins
register_chunker("fixed-token", FixedTokenChunker)
register_chunker("sliding-window", SlidingWindowChunker)
register_chunker("sentence", SentenceChunker)
register_chunker("semantic-boundary", SemanticBoundaryChunker)
register_chunker("structural-markdown", StructuralMarkdownChunker)
