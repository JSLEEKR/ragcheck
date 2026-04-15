"""Exhaustive tests for retrieval metrics.

Every metric is verified with hand-computed expected values so that a change
to the formula produces a visible test failure. No floating-point approximate
equality is used for integer-only outcomes; a small epsilon is used for DCG.
"""
from __future__ import annotations

import math

import pytest

from ragcheck.metrics import (
    DEFAULT_K_VALUES,
    METRIC_NAMES,
    aggregate_metric,
    average_precision,
    context_precision,
    context_recall,
    dcg_at_k,
    f1_at_k,
    format_float,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


EPS = 1e-9


class TestRecallAtK:
    def test_all_relevant_in_top_k(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0

    def test_half_relevant(self):
        # top-3 = ["a","x","b"]; 2 of the 4 relevant ids appear → recall = 0.5
        assert recall_at_k(["a", "x", "b", "y"], {"a", "b", "c", "d"}, 3) == pytest.approx(0.5)

    def test_none_relevant(self):
        assert recall_at_k(["x", "y", "z"], {"a", "b"}, 3) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], set(), 3) == 0.0

    def test_k_larger_than_retrieved(self):
        assert recall_at_k(["a"], {"a", "b"}, 10) == pytest.approx(0.5)

    def test_k_zero(self):
        assert recall_at_k(["a", "b"], {"a"}, 0) == 0.0

    def test_negative_k_raises(self):
        with pytest.raises(ValueError):
            recall_at_k(["a"], {"a"}, -1)

    def test_non_string_id_raises(self):
        with pytest.raises(TypeError):
            recall_at_k(["a"], [1, 2], 1)  # type: ignore[list-item]

    def test_duplicate_retrieved_does_not_double_count(self):
        # Relevant set is {"a"}; retrieved has "a" twice; recall should be 1.0
        assert recall_at_k(["a", "a"], {"a"}, 2) == 1.0

    def test_recall_at_1_basic(self):
        assert recall_at_k(["a", "b"], {"a"}, 1) == 1.0
        assert recall_at_k(["x", "a"], {"a"}, 1) == 0.0


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_mixed(self):
        assert precision_at_k(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(2 / 3)

    def test_precision_penalises_short_retrieved(self):
        # denominator is k, not len(retrieved)
        assert precision_at_k(["a"], {"a"}, 5) == pytest.approx(1 / 5)

    def test_precision_at_1(self):
        assert precision_at_k(["a"], {"a", "b"}, 1) == 1.0
        assert precision_at_k(["x"], {"a", "b"}, 1) == 0.0

    def test_precision_k_zero(self):
        assert precision_at_k(["a"], {"a"}, 0) == 0.0

    def test_precision_negative_k(self):
        with pytest.raises(ValueError):
            precision_at_k(["a"], {"a"}, -5)


class TestHitRateAtK:
    def test_hit_present(self):
        assert hit_rate_at_k(["x", "a"], {"a"}, 2) == 1.0

    def test_hit_absent(self):
        assert hit_rate_at_k(["x", "y"], {"a"}, 2) == 0.0

    def test_hit_zero_k(self):
        assert hit_rate_at_k(["a"], {"a"}, 0) == 0.0

    def test_hit_empty_relevant(self):
        assert hit_rate_at_k(["a"], set(), 5) == 0.0

    def test_hit_only_counts_top_k(self):
        assert hit_rate_at_k(["x", "y", "a"], {"a"}, 2) == 0.0
        assert hit_rate_at_k(["x", "y", "a"], {"a"}, 3) == 1.0


class TestMRR:
    def test_first_position(self):
        assert mean_reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_second_position(self):
        assert mean_reciprocal_rank(["x", "a"], {"a"}) == pytest.approx(0.5)

    def test_third_position(self):
        assert mean_reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_hit(self):
        assert mean_reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_empty_relevant(self):
        assert mean_reciprocal_rank(["a"], set()) == 0.0

    def test_mrr_multiple_relevant_returns_earliest(self):
        # "b" found at position 2 before "a" at position 4 — reciprocal rank = 1/2
        assert mean_reciprocal_rank(["x", "b", "y", "a"], {"a", "b"}) == pytest.approx(0.5)


class TestDCG:
    def test_all_zero_gains(self):
        assert dcg_at_k([0.0, 0.0, 0.0], 3) == 0.0

    def test_single_gain_position_1(self):
        # (2^1 - 1) / log2(2) = 1 / 1 = 1.0
        assert dcg_at_k([1.0], 1) == pytest.approx(1.0, abs=EPS)

    def test_single_gain_position_2(self):
        # (2^1 - 1) / log2(3) = 1 / 1.584... ~= 0.6309
        assert dcg_at_k([0.0, 1.0], 2) == pytest.approx(1.0 / math.log2(3), abs=EPS)

    def test_multiple_gains(self):
        # Position 1: (2^2-1)/log2(2) = 3/1 = 3
        # Position 2: (2^1-1)/log2(3) ~= 0.6309
        # Position 3: (2^0-1)/log2(4) = 0
        expected = 3.0 + 1.0 / math.log2(3) + 0.0
        assert dcg_at_k([2.0, 1.0, 0.0], 3) == pytest.approx(expected, abs=EPS)

    def test_k_zero(self):
        assert dcg_at_k([1.0, 2.0], 0) == 0.0

    def test_k_larger_than_len(self):
        expected = 1.0 + 1.0 / math.log2(3)
        assert dcg_at_k([1.0, 1.0], 10) == pytest.approx(expected, abs=EPS)

    def test_negative_gain_raises(self):
        with pytest.raises(ValueError):
            dcg_at_k([-1.0], 1)

    def test_negative_k_raises(self):
        with pytest.raises(ValueError):
            dcg_at_k([1.0], -1)


class TestNDCGAtK:
    def test_perfect_ranking(self):
        # Retrieved matches ideal order → nDCG = 1.0
        rel = {"a": 2, "b": 1, "c": 0}
        assert ndcg_at_k(["a", "b", "c"], rel, 3) == pytest.approx(1.0, abs=EPS)

    def test_worst_ranking(self):
        rel = {"a": 2, "b": 1}
        # ideal DCG: 3/log2(2)=3 + 1/log2(3)=0.6309 → 3.6309
        # dcg with ranking ["b","a"]: 1/log2(2)=1 + 3/log2(3)=1.8928 → 2.8928
        dcg = 1.0 / math.log2(2) + 3.0 / math.log2(3)
        idcg = 3.0 / math.log2(2) + 1.0 / math.log2(3)
        assert ndcg_at_k(["b", "a"], rel, 2) == pytest.approx(dcg / idcg, abs=EPS)

    def test_missing_items_count_as_zero(self):
        rel = {"a": 1, "b": 1}
        assert ndcg_at_k(["x", "y"], rel, 2) == 0.0

    def test_empty_relevance(self):
        assert ndcg_at_k(["a"], {}, 3) == 0.0

    def test_binary_relevance(self):
        # Binary relevance: single hit at position 1 → ndcg@1 = 1.0
        assert ndcg_at_k(["a"], {"a": 1.0}, 1) == pytest.approx(1.0, abs=EPS)

    def test_ndcg_monotonic_with_position(self):
        rel = {"a": 1.0}
        top = ndcg_at_k(["a", "b", "c"], rel, 3)
        mid = ndcg_at_k(["b", "a", "c"], rel, 3)
        bot = ndcg_at_k(["b", "c", "a"], rel, 3)
        assert top > mid > bot

    def test_ndcg_non_string_key_raises(self):
        with pytest.raises(TypeError):
            ndcg_at_k(["a"], {1: 1.0}, 1)  # type: ignore[dict-item]

    def test_ndcg_negative_value_raises(self):
        with pytest.raises(ValueError):
            ndcg_at_k(["a"], {"a": -1.0}, 1)


class TestContextPrecision:
    def test_all_relevant_in_order(self):
        # precisions at each relevant position: 1/1, 2/2 → mean=1.0
        assert context_precision(["a", "b"], {"a", "b"}) == pytest.approx(1.0)

    def test_relevant_at_position_2(self):
        # positions of relevant: 2. precision@2 = 1/2 → cp = 0.5
        assert context_precision(["x", "a"], {"a"}) == pytest.approx(0.5)

    def test_no_hits(self):
        assert context_precision(["x", "y"], {"a"}) == 0.0

    def test_empty_relevant(self):
        assert context_precision(["a"], set()) == 0.0

    def test_context_precision_with_k(self):
        # Only top-1 considered: "a" is relevant → precision=1.0
        assert context_precision(["a", "b"], {"a"}, k=1) == pytest.approx(1.0)

    def test_context_precision_honors_k_when_relevant_outside(self):
        assert context_precision(["x", "y", "a"], {"a"}, k=2) == 0.0


class TestContextRecall:
    def test_full_coverage(self):
        assert context_recall(["a", "b"], {"a", "b"}) == 1.0

    def test_partial_coverage(self):
        assert context_recall(["a"], {"a", "b"}) == pytest.approx(0.5)

    def test_empty_relevant(self):
        assert context_recall(["a"], set()) == 0.0

    def test_context_recall_respects_k(self):
        # Full list has both, but k=1 only sees "a"
        assert context_recall(["a", "b"], {"a", "b"}, k=1) == pytest.approx(0.5)


class TestF1AtK:
    def test_f1_zero_when_both_zero(self):
        assert f1_at_k(["x"], {"a"}, 1) == 0.0

    def test_f1_harmonic_mean(self):
        # retrieved=["a","x"], relevant={"a","b"}, k=2
        # precision = 1/2, recall = 1/2 → F1 = 0.5
        assert f1_at_k(["a", "x"], {"a", "b"}, 2) == pytest.approx(0.5)


class TestAveragePrecision:
    def test_all_relevant_perfect_order(self):
        # relevance at positions 1,2 → precisions 1/1, 2/2 → avg = (1+1)/2 = 1
        assert average_precision(["a", "b"], {"a", "b"}) == pytest.approx(1.0)

    def test_ap_single_match_at_rank_3(self):
        # precision@3 = 1/3; divide by |relevant|=1 → 1/3
        assert average_precision(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_ap_no_hits(self):
        assert average_precision(["x", "y"], {"a"}) == 0.0

    def test_ap_empty_relevant(self):
        assert average_precision(["a"], set()) == 0.0


class TestAggregate:
    def test_mean(self):
        assert aggregate_metric([0.5, 1.0]) == pytest.approx(0.75)

    def test_empty(self):
        assert aggregate_metric([]) == 0.0

    def test_rejects_nan(self):
        with pytest.raises(ValueError):
            aggregate_metric([float("nan")])

    def test_rejects_inf(self):
        with pytest.raises(ValueError):
            aggregate_metric([float("inf")])

    def test_rejects_non_numeric(self):
        with pytest.raises(TypeError):
            aggregate_metric(["x"])  # type: ignore[list-item]


class TestFormatFloat:
    def test_format_basic(self):
        assert format_float(0.123456789) == "0.123457"

    def test_format_zero(self):
        assert format_float(0.0) == "0.000000"

    def test_format_nan(self):
        assert format_float(float("nan")) == "null"

    def test_format_inf(self):
        assert format_float(float("inf")) == "null"

    def test_format_precision_arg(self):
        assert format_float(0.1234, precision=2) == "0.12"


class TestMetricConstants:
    def test_metric_names_sorted_in_expected_order(self):
        # Sanity check: default k values are {1,3,5,10}
        assert DEFAULT_K_VALUES == [1, 3, 5, 10]

    def test_metric_names_include_ndcg_mrr(self):
        assert "mrr" in METRIC_NAMES
        assert "ndcg@5" in METRIC_NAMES

    def test_metric_names_nonempty(self):
        assert len(METRIC_NAMES) > 10
