"""Chunking diagnostics: coverage, duplicate ratio, orphan chunks, size histogram.

These are the debug tools a RAG engineer builds three times in their career.
They surface obvious failures (all chunks are duplicates; half the corpus is
unreachable; chunks are all 20 chars long) before you even look at retrieval
metrics.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence

from ragcheck.chunkers import Chunk


@dataclass
class ChunkingDiagnostics:
    """Structural properties of a chunked corpus.

    Every field is deterministic for a given chunk list — sorting is explicit
    wherever needed and floats are not compared to non-finite values.
    """

    total_chunks: int = 0
    total_docs: int = 0
    unique_chunks: int = 0
    duplicate_chunks: int = 0
    duplicate_ratio: float = 0.0
    avg_chunk_chars: float = 0.0
    median_chunk_chars: float = 0.0
    min_chunk_chars: int = 0
    max_chunk_chars: int = 0
    size_histogram: Dict[str, int] = field(default_factory=dict)
    orphan_chunk_count: int = 0
    orphan_chunk_ratio: float = 0.0
    coverage: float = 0.0
    empty_chunks: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_chunks": self.total_chunks,
            "total_docs": self.total_docs,
            "unique_chunks": self.unique_chunks,
            "duplicate_chunks": self.duplicate_chunks,
            "duplicate_ratio": round(self.duplicate_ratio, 6),
            "avg_chunk_chars": round(self.avg_chunk_chars, 6),
            "median_chunk_chars": round(self.median_chunk_chars, 6),
            "min_chunk_chars": self.min_chunk_chars,
            "max_chunk_chars": self.max_chunk_chars,
            "size_histogram": dict(sorted(self.size_histogram.items())),
            "orphan_chunk_count": self.orphan_chunk_count,
            "orphan_chunk_ratio": round(self.orphan_chunk_ratio, 6),
            "coverage": round(self.coverage, 6),
            "empty_chunks": self.empty_chunks,
        }


# Histogram bucket edges (char length). Last bucket is open-ended.
_HISTOGRAM_BUCKETS: List[tuple] = [
    ("0-49", 0, 50),
    ("50-99", 50, 100),
    ("100-199", 100, 200),
    ("200-499", 200, 500),
    ("500-999", 500, 1000),
    ("1000-1999", 1000, 2000),
    ("2000+", 2000, 10**12),
]


def _bucket_for(length: int) -> str:
    for name, lo, hi in _HISTOGRAM_BUCKETS:
        if lo <= length < hi:
            return name
    return _HISTOGRAM_BUCKETS[-1][0]


def _content_hash(text: str) -> str:
    """sha1 of whitespace-normalised chunk text. Duplicate detector."""
    normalised = " ".join(text.split()).lower()
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()


def _median(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def chunking_diagnostics(
    chunks: Sequence[Chunk],
    *,
    gold_relevant_chunk_ids: Iterable[str] = (),
    reachable_chunk_ids: Iterable[str] = (),
) -> ChunkingDiagnostics:
    """Compute structural diagnostics over a list of chunks.

    - `gold_relevant_chunk_ids`: ids that must exist in the chunk list for
      coverage to be 1.0. If empty, coverage is 1.0 (nothing to cover).
    - `reachable_chunk_ids`: ids that are reachable from any query in the
      retrieval system (e.g. union of top-k across all queries). Chunks in
      the chunk list but absent from this set are counted as orphans. If
      empty, orphan_count is 0 (meaning: not measured).
    """
    diag = ChunkingDiagnostics()
    diag.total_chunks = len(chunks)
    if diag.total_chunks == 0:
        return diag

    docs = {c.doc_id for c in chunks}
    diag.total_docs = len(docs)

    # Lengths
    lengths = [len(c.text) for c in chunks]
    diag.empty_chunks = sum(1 for n in lengths if n == 0)
    non_empty_lengths = [n for n in lengths if n > 0]
    if non_empty_lengths:
        diag.avg_chunk_chars = sum(non_empty_lengths) / len(non_empty_lengths)
        diag.median_chunk_chars = _median(non_empty_lengths)
        diag.min_chunk_chars = min(non_empty_lengths)
        diag.max_chunk_chars = max(non_empty_lengths)

    # Histogram
    histogram: Dict[str, int] = {name: 0 for name, _, _ in _HISTOGRAM_BUCKETS}
    for n in lengths:
        histogram[_bucket_for(n)] += 1
    diag.size_histogram = histogram

    # Duplicates
    seen_hashes: Dict[str, int] = {}
    for c in chunks:
        h = _content_hash(c.text)
        seen_hashes[h] = seen_hashes.get(h, 0) + 1
    diag.unique_chunks = len(seen_hashes)
    duplicates = sum(count - 1 for count in seen_hashes.values() if count > 1)
    diag.duplicate_chunks = duplicates
    diag.duplicate_ratio = duplicates / diag.total_chunks if diag.total_chunks else 0.0

    # Coverage
    chunk_id_set = {c.chunk_id for c in chunks}
    gold = {g for g in gold_relevant_chunk_ids}
    if gold:
        covered = sum(1 for g in gold if g in chunk_id_set)
        diag.coverage = covered / len(gold)
    else:
        diag.coverage = 1.0

    # Orphans
    reachable = {r for r in reachable_chunk_ids}
    if reachable:
        orphans = sum(1 for cid in chunk_id_set if cid not in reachable)
        diag.orphan_chunk_count = orphans
        diag.orphan_chunk_ratio = orphans / len(chunk_id_set) if chunk_id_set else 0.0
    else:
        diag.orphan_chunk_count = 0
        diag.orphan_chunk_ratio = 0.0

    return diag


def format_histogram_ascii(histogram: Mapping[str, int], width: int = 40) -> str:
    """Render a histogram dict as an ASCII bar chart. Deterministic order."""
    if not histogram:
        return "(empty)"
    ordered = list(histogram.items())
    # Preserve canonical bucket order if it matches our constants
    canonical_order = [name for name, _, _ in _HISTOGRAM_BUCKETS]
    if set(histogram.keys()) <= set(canonical_order):
        ordered = [(name, histogram.get(name, 0)) for name in canonical_order]
    else:
        ordered.sort()
    max_count = max((v for _, v in ordered), default=0)
    lines: List[str] = []
    for name, count in ordered:
        bar_len = 0
        if max_count > 0:
            bar_len = round((count / max_count) * width)
        lines.append(f"{name:<12} | {'#' * bar_len} ({count})")
    return "\n".join(lines)
