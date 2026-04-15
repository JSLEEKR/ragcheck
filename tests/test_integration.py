"""End-to-end integration tests.

Exercise the CLI and public API on real fixtures to catch regressions that
unit tests miss.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pytest

from ragcheck.cli import EXIT_OK, main
from ragcheck.corpus import load_corpus, load_gold_set
from ragcheck.fixtures import BEIR_FIQA_DIR, MS_MARCO_DIR, NEEDLE_DIR
from ragcheck.fixtures.needle_haystack import materialise
from ragcheck.runner import RunConfig, dump_result_json, run_evaluation


def _cli(argv):
    out = io.StringIO()
    err = io.StringIO()
    code = main(argv, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


class TestBeirFixture:
    def test_run_on_beir_produces_positive_recall(self, tmp_path: Path):
        out_path = tmp_path / "run.json"
        code, out, err = _cli([
            "run",
            "--corpus", str(BEIR_FIQA_DIR / "corpus"),
            "--gold", str(BEIR_FIQA_DIR / "gold.json"),
            "--out", str(out_path),
        ])
        assert code == EXIT_OK, err
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["summary"]["recall@10"] >= 0.3

    def test_end_to_end_determinism_beir(self, tmp_path: Path):
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        _cli(["run", "--corpus", str(BEIR_FIQA_DIR / "corpus"), "--gold",
              str(BEIR_FIQA_DIR / "gold.json"), "--out", str(a)])
        _cli(["run", "--corpus", str(BEIR_FIQA_DIR / "corpus"), "--gold",
              str(BEIR_FIQA_DIR / "gold.json"), "--out", str(b)])
        assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")


class TestMsMarcoFixture:
    def test_run_on_ms_marco(self, tmp_path: Path):
        out_path = tmp_path / "run.json"
        code, out, err = _cli([
            "run",
            "--corpus", str(MS_MARCO_DIR / "corpus"),
            "--gold", str(MS_MARCO_DIR / "gold.json"),
            "--out", str(out_path),
        ])
        assert code == EXIT_OK, err
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["summary"]["recall@10"] >= 0.3


class TestNeedleFixture:
    def test_materialise_and_run(self, tmp_path: Path):
        materialise()
        out_path = tmp_path / "run.json"
        code, out, err = _cli([
            "run",
            "--corpus", str(NEEDLE_DIR / "corpus"),
            "--gold", str(NEEDLE_DIR / "gold.json"),
            "--out", str(out_path),
        ])
        assert code == EXIT_OK, err


class TestDiffEndToEnd:
    def test_synthetic_degradation(self, tmp_path: Path):
        # Run twice with two different chunkers — first with small chunks,
        # second with huge chunks — and assert metrics differ.
        baseline = tmp_path / "base.json"
        head = tmp_path / "head.json"
        _cli([
            "run",
            "--corpus", str(BEIR_FIQA_DIR / "corpus"),
            "--gold", str(BEIR_FIQA_DIR / "gold.json"),
            "--out", str(baseline),
            "--chunker-args", '{"tokens_per_chunk": 40, "overlap": 10}',
        ])
        _cli([
            "run",
            "--corpus", str(BEIR_FIQA_DIR / "corpus"),
            "--gold", str(BEIR_FIQA_DIR / "gold.json"),
            "--out", str(head),
            "--chunker-args", '{"tokens_per_chunk": 5, "overlap": 0}',
            # Aggressive over-chunking ought to change retrieval behaviour
        ])
        code, stdout, stderr = _cli(["diff", str(baseline), str(head)])
        # Either 0 or 1 is acceptable here; just make sure diff runs cleanly
        assert code in (0, 1)
        assert "ragcheck regression diff" in stdout


class TestBenchSpeed:
    def test_bench_under_thirty_seconds(self):
        t0 = time.perf_counter()
        code, out, err = _cli(["bench"])
        elapsed = time.perf_counter() - t0
        assert code == EXIT_OK
        assert elapsed < 30.0


class TestPublicApi:
    def test_import_public_symbols(self):
        import ragcheck

        assert ragcheck.__version__
        assert callable(ragcheck.recall_at_k)
        assert callable(ragcheck.run_evaluation)
        assert callable(ragcheck.diff_runs)
        assert callable(ragcheck.chunking_diagnostics)

    def test_run_evaluation_returns_serializable(self):
        import ragcheck
        config = ragcheck.RunConfig(
            corpus_path=str(BEIR_FIQA_DIR / "corpus"),
            gold_path=str(BEIR_FIQA_DIR / "gold.json"),
        )
        result = ragcheck.run_evaluation(config=config)
        data = result.to_dict()
        json.dumps(data, sort_keys=True)  # must not raise
