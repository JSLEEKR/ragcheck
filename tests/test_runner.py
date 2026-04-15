"""Tests for the full evaluation runner."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragcheck.chunkers import FixedTokenChunker
from ragcheck.corpus import GoldSet, Query
from ragcheck.embedders import HashEmbedder
from ragcheck.runner import (
    RunConfig,
    dump_result_json,
    load_result_json,
    run_evaluation,
)


class TestRunEvaluationBasic:
    def test_end_to_end(self, tiny_corpus: Path, tiny_gold: Path):
        config = RunConfig(
            corpus_path=str(tiny_corpus),
            gold_path=str(tiny_gold),
            chunker="fixed-token",
            chunker_args={"tokens_per_chunk": 30},
            embedder="hash",
            embedder_args={"dim": 128},
        )
        result = run_evaluation(config=config)
        assert result.corpus_stats["total_documents"] == 3
        assert result.corpus_stats["total_queries"] == 3
        assert len(result.per_query) == 3
        # Core metrics present
        assert "recall@5" in result.summary
        assert "mrr" in result.summary
        assert "ndcg@5" in result.summary

    def test_without_per_query(self, tiny_corpus: Path, tiny_gold: Path):
        config = RunConfig(
            corpus_path=str(tiny_corpus),
            gold_path=str(tiny_gold),
            include_per_query=False,
        )
        result = run_evaluation(config=config)
        assert result.per_query == []

    def test_without_diagnostics(self, tiny_corpus: Path, tiny_gold: Path):
        config = RunConfig(
            corpus_path=str(tiny_corpus),
            gold_path=str(tiny_gold),
            include_diagnostics=False,
        )
        result = run_evaluation(config=config)
        assert result.diagnostics == {}

    def test_with_timestamp(self, tiny_corpus: Path, tiny_gold: Path):
        config = RunConfig(
            corpus_path=str(tiny_corpus),
            gold_path=str(tiny_gold),
            include_timestamp=True,
        )
        result = run_evaluation(config=config)
        assert result.timestamp is not None
        assert result.timestamp.endswith("Z")

    def test_requires_paths_or_config(self):
        with pytest.raises(ValueError):
            run_evaluation()


class TestDeterminism:
    def test_two_runs_produce_identical_json(self, tiny_corpus: Path, tiny_gold: Path, tmp_path: Path):
        config = RunConfig(
            corpus_path=str(tiny_corpus),
            gold_path=str(tiny_gold),
        )
        r1 = run_evaluation(config=config)
        r2 = run_evaluation(config=config)
        p1 = tmp_path / "a.json"
        p2 = tmp_path / "b.json"
        dump_result_json(r1, p1)
        dump_result_json(r2, p2)
        assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")

    def test_same_result_with_changed_config_label_no_effect_on_metrics(
        self, tiny_corpus: Path, tiny_gold: Path
    ):
        c1 = RunConfig(corpus_path=str(tiny_corpus), gold_path=str(tiny_gold), label="a")
        c2 = RunConfig(corpus_path=str(tiny_corpus), gold_path=str(tiny_gold), label="b")
        r1 = run_evaluation(config=c1)
        r2 = run_evaluation(config=c2)
        assert r1.summary == r2.summary


class TestRelevanceResolution:
    def test_doc_level_relevance_propagates_to_chunks(self, small_docs, small_gold):
        config = RunConfig(corpus_path="(in-memory)", gold_path="(in-memory)")
        result = run_evaluation(
            corpus=small_docs,
            gold=small_gold,
            config=config,
            chunker=FixedTokenChunker(tokens_per_chunk=10),
            embedder=HashEmbedder(dim=64),
        )
        # Each query looks for an id that matches exactly one doc; top-1 should hit
        assert result.summary["hit_rate@5"] > 0.0

    def test_empty_corpus_produces_zero_summary(self):
        config = RunConfig(corpus_path="(x)", gold_path="(y)")
        gold = GoldSet(questions=[Query(query_id="q", text="anything", relevance={"a": 1.0})])
        result = run_evaluation(
            corpus=[],
            gold=gold,
            config=config,
            chunker=FixedTokenChunker(),
            embedder=HashEmbedder(),
        )
        # metrics should still be present and 0.0
        assert result.summary["recall@5"] == 0.0

    def test_empty_gold_produces_empty_per_query(self, small_docs):
        config = RunConfig(corpus_path="(x)", gold_path="(y)")
        result = run_evaluation(
            corpus=small_docs,
            gold=GoldSet(),
            config=config,
            chunker=FixedTokenChunker(),
            embedder=HashEmbedder(),
        )
        assert result.per_query == []
        # Summary is an empty dict (no metrics to compute)
        assert result.summary == {}


class TestJSONOutput:
    def test_json_sorted_keys(self, tiny_corpus: Path, tiny_gold: Path, tmp_path: Path):
        config = RunConfig(corpus_path=str(tiny_corpus), gold_path=str(tiny_gold))
        result = run_evaluation(config=config)
        p = tmp_path / "out.json"
        dump_result_json(result, p)
        # Load once to verify it parses, then inspect raw text for key order
        json.loads(p.read_text(encoding="utf-8"))
        text = p.read_text(encoding="utf-8")
        first_keys_order = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith('"') and '":' in line:
                key = line.split('"')[1]
                first_keys_order.append(key)
                if len(first_keys_order) > 4:
                    break
        # The very first field should be "config" (alphabetical)
        assert first_keys_order[0] == "config"

    def test_json_compact_mode(self, tiny_corpus: Path, tiny_gold: Path, tmp_path: Path):
        config = RunConfig(corpus_path=str(tiny_corpus), gold_path=str(tiny_gold))
        result = run_evaluation(config=config)
        p = tmp_path / "compact.json"
        dump_result_json(result, p, pretty=False)
        text = p.read_text(encoding="utf-8")
        # No 2-space indentation in compact mode
        assert "\n  " not in text

    def test_load_result_json_roundtrip(self, tiny_corpus: Path, tiny_gold: Path, tmp_path: Path):
        config = RunConfig(corpus_path=str(tiny_corpus), gold_path=str(tiny_gold))
        result = run_evaluation(config=config)
        p = tmp_path / "r.json"
        dump_result_json(result, p)
        loaded = load_result_json(p)
        assert loaded["tool_version"]
        assert "summary" in loaded

    def test_load_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_result_json(tmp_path / "missing.json")

    def test_json_floats_rounded_to_6(self, tiny_corpus: Path, tiny_gold: Path, tmp_path: Path):
        config = RunConfig(corpus_path=str(tiny_corpus), gold_path=str(tiny_gold))
        result = run_evaluation(config=config)
        p = tmp_path / "r.json"
        dump_result_json(result, p)
        text = p.read_text(encoding="utf-8")
        # Make sure we don't have long decimal expansions
        for line in text.split("\n"):
            if ":" in line and "." in line.split(":")[-1]:
                tail = line.split(":")[-1]
                if "e" not in tail.lower():
                    # decimal string
                    decimals_part = tail.split(".")[-1].split(",")[0].strip()
                    digits = "".join(ch for ch in decimals_part if ch.isdigit())
                    assert len(digits) <= 6


class TestNumpyEmbedderPath:
    def test_numpy_embedder_via_factory_requires_mapping(self):
        from ragcheck.runner import _build_embedder
        with pytest.raises(ValueError):
            _build_embedder("numpy", {})


class TestPublicAPI:
    """Regression (H1): README documents ``ragcheck.load_result_json`` and
    ``ragcheck.dump_result_json`` as part of the programmatic API. They must
    be importable from the top-level package."""

    def test_load_result_json_exported(self):
        import ragcheck
        assert hasattr(ragcheck, "load_result_json")
        assert ragcheck.load_result_json is load_result_json

    def test_dump_result_json_exported(self):
        import ragcheck
        assert hasattr(ragcheck, "dump_result_json")
        assert ragcheck.dump_result_json is dump_result_json

    def test_load_roundtrip_via_public_api(self, tmp_path, tiny_corpus, tiny_gold):
        import ragcheck
        config = RunConfig(corpus_path=str(tiny_corpus), gold_path=str(tiny_gold))
        result = run_evaluation(config=config)
        p = tmp_path / "r.json"
        ragcheck.dump_result_json(result, p)
        loaded = ragcheck.load_result_json(p)
        assert loaded["tool_version"]
