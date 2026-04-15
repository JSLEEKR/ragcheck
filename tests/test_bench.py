"""Tests for the bundled benchmark runner."""
from __future__ import annotations

import pytest

from ragcheck.bench import BenchResult, format_bench_table, run_bench
from ragcheck.fixtures import (
    BEIR_FIQA_DIR,
    MS_MARCO_DIR,
    NEEDLE_DIR,
    fixture_dir,
    list_fixture_names,
)
from ragcheck.fixtures.needle_haystack import (
    generate_needle_corpus,
    materialise,
)


class TestFixtureMetadata:
    def test_list_fixtures(self):
        names = list_fixture_names()
        assert "beir_fiqa" in names
        assert "ms_marco" in names
        assert "needle_haystack" in names

    def test_fixture_dir_known(self):
        assert fixture_dir("beir_fiqa") == BEIR_FIQA_DIR
        assert fixture_dir("ms_marco") == MS_MARCO_DIR
        assert fixture_dir("needle_haystack") == NEEDLE_DIR

    def test_fixture_dir_unknown(self):
        with pytest.raises(KeyError):
            fixture_dir("nope")


class TestNeedleFixture:
    def test_generate_returns_mix(self):
        items = generate_needle_corpus(n_needles=3, n_haystack_per_needle=4)
        kinds = {kind for _, _, kind in items}
        assert "needle" in kinds
        assert "haystack" in kinds

    def test_generate_count(self):
        items = generate_needle_corpus(n_needles=3, n_haystack_per_needle=4)
        assert len(items) == 3 + 3 * 4

    def test_needle_contains_unique_phrase(self):
        items = generate_needle_corpus(n_needles=3, n_haystack_per_needle=2)
        needles = [text for doc, text, k in items if k == "needle"]
        # Each needle has a unique alpha-omega code
        codes = set()
        for t in needles:
            assert "NEEDLE-" in t
            code = t.split("ALPHA-")[1].split("-OMEGA")[0]
            codes.add(code)
        assert len(codes) == 3

    def test_materialise_writes_files(self):
        dir_path = materialise()
        assert dir_path.exists()
        assert any(dir_path.glob("needle_*.txt"))
        assert any(dir_path.glob("haystack_*.txt"))


class TestBenchRun:
    def test_returns_three_results(self):
        results = run_bench()
        assert len(results) == 3

    def test_results_have_metrics(self):
        results = run_bench()
        for r in results:
            assert r.n_queries > 0
            assert 0.0 <= r.recall_at_5 <= 1.0
            assert 0.0 <= r.mrr <= 1.0

    def test_total_elapsed_under_thirty_seconds(self):
        results = run_bench()
        total = sum(r.elapsed_seconds for r in results)
        assert total < 30.0

    def test_bench_deterministic(self):
        r1 = run_bench()
        r2 = run_bench()
        for a, b in zip(r1, r2):
            assert a.recall_at_5 == b.recall_at_5
            assert a.mrr == b.mrr


class TestBenchFormatting:
    def test_to_dict_shape(self):
        r = BenchResult(
            fixture="x",
            n_docs=2,
            n_queries=3,
            recall_at_5=0.5,
            mrr=0.6,
            ndcg_at_5=0.7,
            elapsed_seconds=0.1,
        )
        d = r.to_dict()
        assert d["fixture"] == "x"
        assert d["recall@5"] == 0.5

    def test_format_table_contains_header(self):
        results = run_bench()
        text = format_bench_table(results)
        assert "fixture" in text
        assert "recall@5" in text

    def test_format_table_total_elapsed(self):
        results = run_bench()
        text = format_bench_table(results)
        assert "total elapsed" in text


class TestBeirFixture:
    def test_corpus_exists(self):
        corpus = BEIR_FIQA_DIR / "corpus"
        assert corpus.exists()
        files = list(corpus.glob("*.txt"))
        assert len(files) >= 5

    def test_gold_parses(self):
        from ragcheck.corpus import load_gold_set
        g = load_gold_set(BEIR_FIQA_DIR / "gold.json")
        assert len(g.questions) >= 5
        assert g.synthetic is False


class TestMsMarcoFixture:
    def test_corpus_exists(self):
        corpus = MS_MARCO_DIR / "corpus"
        assert corpus.exists()
        files = list(corpus.glob("*.txt"))
        assert len(files) >= 5

    def test_gold_parses(self):
        from ragcheck.corpus import load_gold_set
        g = load_gold_set(MS_MARCO_DIR / "gold.json")
        assert len(g.questions) >= 5
