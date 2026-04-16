"""Information retrieval metrics for RAG evaluation.

All metrics take a ranked list of retrieved items plus a set/list of relevant
items (ground truth). No external services. Every function is pure and
deterministic.

Naming convention:
- retrieved: ordered list of retrieved document/chunk IDs (highest rank first)
- relevant: iterable of relevant document/chunk IDs (ground truth)
- k: cutoff for @k metrics; if k > len(retrieved), the whole list is used
- relevance: dict mapping id -> graded relevance (>= 0); absent = 0
"""
from __future__ import annotations

import math
from typing import Iterable, List, Mapping, Optional, Sequence


def _as_set(xs: Iterable[str]) -> set:
    """Coerce an iterable of ids to a set, rejecting non-string ids early."""
    out: set = set()
    for x in xs:
        if not isinstance(x, str):
            raise TypeError(f"ids must be str, got {type(x).__name__}: {x!r}")
        out.add(x)
    return out


def _clip_k(retrieved: Sequence[str], k: int) -> int:
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    return min(k, len(retrieved))


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of distinct relevant items that appear in the top-k retrieved list.

    Counts each relevant item at most once, even if it appears multiple times
    in the retrieved list.

    - If there are no relevant items, returns 0.0 (convention: a query with
      no ground-truth answer has undefined recall; we surface 0.0 so aggregate
      averages are well defined and the per-query drop is visible).
    - If k == 0, returns 0.0.
    """
    rel_set = _as_set(relevant)
    if not rel_set:
        return 0.0
    k_eff = _clip_k(retrieved, k)
    if k_eff == 0:
        return 0.0
    seen: set = set()
    for item in retrieved[:k_eff]:
        if item in rel_set:
            seen.add(item)
    return len(seen) / len(rel_set)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of top-k retrieved items that are relevant.

    - If k == 0, returns 0.0.
    - If retrieved is shorter than k, the denominator is still k (standard IR
      convention) — this penalises systems that return fewer than k items.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k == 0:
        return 0.0
    rel_set = _as_set(relevant)
    top = retrieved[:k]
    hits = sum(1 for x in top if x in rel_set)
    return hits / k


def hit_rate_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if any top-k retrieved item is relevant, else 0.0.

    Also known as 'success@k' in IR literature.
    """
    rel_set = _as_set(relevant)
    if not rel_set:
        return 0.0
    k_eff = _clip_k(retrieved, k)
    if k_eff == 0:
        return 0.0
    for item in retrieved[:k_eff]:
        if item in rel_set:
            return 1.0
    return 0.0


def mean_reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Reciprocal rank (1/position) of the first relevant retrieved item.

    Position is 1-indexed. Returns 0.0 if no relevant item is retrieved or
    there are no relevant items.
    """
    rel_set = _as_set(relevant)
    if not rel_set:
        return 0.0
    for i, item in enumerate(retrieved, start=1):
        if item in rel_set:
            return 1.0 / i
    return 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain for a sequence of gains at cutoff k.

    Uses the standard IR formulation: sum_i (2^gain_i - 1) / log2(i + 1)
    with 1-indexed positions.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    total = 0.0
    limit = min(k, len(gains))
    for i in range(limit):
        gain = gains[i]
        # Reject non-finite gains explicitly: ``gain < 0`` is False for NaN,
        # so a bare ``gain < 0`` check let NaN and +Inf slip through and
        # produced a NaN DCG downstream — which aggregates into a NaN
        # summary and either crashes later or silently propagates to the
        # run JSON. Cycle E H3.
        if isinstance(gain, float) and (math.isnan(gain) or math.isinf(gain)):
            raise ValueError(
                f"DCG gain must be finite, got {gain} at position {i}"
            )
        if gain < 0:
            raise ValueError(f"DCG gain must be >= 0, got {gain} at position {i}")
        numerator = (2.0 ** gain) - 1.0
        denom = math.log2(i + 2)  # log2(i+1) with 1-indexed position → log2(i+2)
        total += numerator / denom
    return total


def ndcg_at_k(
    retrieved: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
) -> float:
    """Normalised discounted cumulative gain at cutoff k.

    - `relevance` maps id -> graded relevance (>= 0). Missing ids are treated
      as 0. Binary relevance is supported by mapping relevant ids to 1.0.
    - If the ideal DCG is 0 (no positively-graded items exist for this query
      at all), returns 0.0.
    - Duplicates in ``retrieved`` are credited only on their first occurrence;
      later duplicates contribute zero gain. Without this guard a ranked list
      like ``[A, A, B]`` against relevance ``{A: 1, B: 1}`` would produce a
      DCG larger than the ideal DCG (which is computed over the distinct
      positively-graded items), and ``nDCG`` could exceed ``1.0`` — a math
      impossibility that would silently pass CI thresholds like
      ``nDCG@5 >= 0.9``. Cycle D H2.
    """
    for rid, rv in relevance.items():
        if not isinstance(rid, str):
            raise TypeError(f"relevance keys must be str, got {type(rid).__name__}: {rid!r}")
        # ``rv < 0`` is False for NaN/+Inf, so relax it into an explicit
        # finite check first. Without this, a gold file containing NaN
        # (which Python's json parser happily decodes as an extension)
        # produced a NaN nDCG value that silently made every run look
        # broken. Cycle E H3.
        if isinstance(rv, float) and (math.isnan(rv) or math.isinf(rv)):
            raise ValueError(
                f"relevance values must be finite, got {rv} for id {rid!r}"
            )
        if rv < 0:
            raise ValueError(f"relevance values must be >= 0, got {rv} for id {rid!r}")
    k_eff = _clip_k(retrieved, k) if k > 0 else 0
    if k_eff == 0:
        return 0.0
    seen: set = set()
    gains: List[float] = []
    for x in retrieved[:k_eff]:
        if x in seen:
            gains.append(0.0)
        else:
            seen.add(x)
            gains.append(float(relevance.get(x, 0.0)))
    dcg = dcg_at_k(gains, k_eff)
    ideal_gains = sorted((float(v) for v in relevance.values() if v > 0), reverse=True)
    if not ideal_gains:
        return 0.0
    idcg = dcg_at_k(ideal_gains, k_eff)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def context_precision(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: Optional[int] = None,
) -> float:
    """Weighted precision with rank discount, in the style of LangChain/Ragas.

    Interprets the retrieved list as an ordered context window and returns the
    average precision@i computed at each position i where the i-th item is
    relevant. This rewards putting relevant items near the top.

    If k is None, uses the full retrieved list. If no relevant item is
    retrieved, returns 0.0.
    """
    rel_set = _as_set(relevant)
    if not rel_set:
        return 0.0
    limit = len(retrieved) if k is None else min(k, len(retrieved))
    if limit == 0:
        return 0.0
    precisions: List[float] = []
    seen_relevant = 0
    for i in range(limit):
        if retrieved[i] in rel_set:
            seen_relevant += 1
            precisions.append(seen_relevant / (i + 1))
    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)


def context_recall(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: Optional[int] = None,
) -> float:
    """Fraction of relevant items present in the retrieved window.

    Equivalent to recall@k when k is given; when k is None, uses the entire
    retrieved list. Returns 0.0 if there are no relevant items.
    """
    rel_set = _as_set(relevant)
    if not rel_set:
        return 0.0
    limit = len(retrieved) if k is None else min(k, len(retrieved))
    if limit == 0:
        return 0.0
    hits = sum(1 for x in retrieved[:limit] if x in rel_set)
    return hits / len(rel_set)


def f1_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Harmonic mean of precision@k and recall@k. 0.0 if both are 0."""
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p == 0.0 and r == 0.0:
        return 0.0
    return 2 * p * r / (p + r)


def average_precision(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Average precision over the full retrieved list.

    AP = sum_i (precision_at_i * is_relevant_i) / |relevant|
    """
    rel_set = _as_set(relevant)
    if not rel_set:
        return 0.0
    seen_relevant = 0
    weighted_sum = 0.0
    for i, item in enumerate(retrieved, start=1):
        if item in rel_set:
            seen_relevant += 1
            weighted_sum += seen_relevant / i
    return weighted_sum / len(rel_set)


def aggregate_metric(per_query: Sequence[float]) -> float:
    """Mean of per-query scores, with explicit 0.0 on empty input.

    Used for reducing per-query metrics (e.g. list of recall@5 per query)
    into an aggregate score for a run.
    """
    if not per_query:
        return 0.0
    total = 0.0
    for v in per_query:
        if not isinstance(v, (int, float)):
            raise TypeError(f"per-query score must be numeric, got {type(v).__name__}")
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"per-query score must be finite, got {v}")
        total += float(v)
    return total / len(per_query)


def format_float(value: float, precision: int = 6) -> str:
    """Fixed-precision float formatting used throughout JSON output.

    Guards against NaN/Inf by substituting "null" (parsed back as None).
    """
    if math.isnan(value) or math.isinf(value):
        return "null"
    return f"{value:.{precision}f}"


METRIC_NAMES: List[str] = [
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    "precision@1",
    "precision@3",
    "precision@5",
    "precision@10",
    "hit_rate@1",
    "hit_rate@3",
    "hit_rate@5",
    "hit_rate@10",
    "mrr",
    "ndcg@1",
    "ndcg@3",
    "ndcg@5",
    "ndcg@10",
    "context_precision",
    "context_recall",
]

DEFAULT_K_VALUES: List[int] = [1, 3, 5, 10]
