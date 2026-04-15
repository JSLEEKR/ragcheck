"""Bundled benchmark runner.

`ragcheck bench` runs a tiny evaluation on each of the three built-in
fixtures and prints a summary table. Designed to complete in well under 30
seconds on any machine.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from ragcheck.fixtures import BEIR_FIQA_DIR, MS_MARCO_DIR, NEEDLE_DIR
from ragcheck.fixtures.needle_haystack import materialise as materialise_needle
from ragcheck.runner import RunConfig, run_evaluation


@dataclass
class BenchResult:
    fixture: str
    n_docs: int
    n_queries: int
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    elapsed_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fixture": self.fixture,
            "n_docs": self.n_docs,
            "n_queries": self.n_queries,
            "recall@5": round(self.recall_at_5, 6),
            "mrr": round(self.mrr, 6),
            "ndcg@5": round(self.ndcg_at_5, 6),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def _run_fixture(name: str, corpus_dir: Path, gold_path: Path) -> BenchResult:
    config = RunConfig(
        corpus_path=str(corpus_dir),
        gold_path=str(gold_path),
        chunker="fixed-token",
        chunker_args={"tokens_per_chunk": 40, "overlap": 10},
        embedder="hash",
        embedder_args={"dim": 256, "ngram": 2},
        label=f"bench-{name}",
    )
    t0 = time.perf_counter()
    result = run_evaluation(config=config)
    elapsed = time.perf_counter() - t0
    summary = result.summary
    return BenchResult(
        fixture=name,
        n_docs=int(result.corpus_stats.get("total_documents", 0)),
        n_queries=int(result.corpus_stats.get("total_queries", 0)),
        recall_at_5=float(summary.get("recall@5", 0.0)),
        mrr=float(summary.get("mrr", 0.0)),
        ndcg_at_5=float(summary.get("ndcg@5", 0.0)),
        elapsed_seconds=elapsed,
    )


def run_bench() -> List[BenchResult]:
    """Run every bundled fixture and return the results list."""
    materialise_needle()
    return [
        _run_fixture("beir_fiqa", BEIR_FIQA_DIR / "corpus", BEIR_FIQA_DIR / "gold.json"),
        _run_fixture("ms_marco", MS_MARCO_DIR / "corpus", MS_MARCO_DIR / "gold.json"),
        _run_fixture(
            "needle_haystack", NEEDLE_DIR / "corpus", NEEDLE_DIR / "gold.json"
        ),
    ]


def format_bench_table(results: List[BenchResult]) -> str:
    """Render a compact ASCII table of bench results."""
    lines = [
        "fixture          | docs | queries | recall@5 | mrr      | ndcg@5   | seconds",
        "-----------------|------|---------|----------|----------|----------|--------",
    ]
    for r in results:
        lines.append(
            f"{r.fixture:<16} | {r.n_docs:>4} | {r.n_queries:>7} | "
            f"{r.recall_at_5:>8.4f} | {r.mrr:>8.4f} | {r.ndcg_at_5:>8.4f} | "
            f"{r.elapsed_seconds:>6.3f}"
        )
    total = sum(r.elapsed_seconds for r in results)
    lines.append("-----------------|------|---------|----------|----------|----------|--------")
    lines.append(f"total elapsed: {total:.3f}s")
    return "\n".join(lines)
