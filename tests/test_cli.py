"""Tests for the CLI."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ragcheck.cli import EXIT_FAIL, EXIT_OK, EXIT_USAGE, main


def run_cli(argv, tmp_path: Path = None):
    out = io.StringIO()
    err = io.StringIO()
    code = main(argv, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


class TestNoArgs:
    def test_prints_help(self):
        code, out, err = run_cli([])
        assert code == EXIT_USAGE
        assert "usage" in out.lower() or "usage" in err.lower() or "COMMAND" in out


class TestVersion:
    def test_version_flag(self):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0


class TestRunCommand:
    def test_basic_run(self, tiny_corpus, tiny_gold, tmp_path: Path):
        out_path = tmp_path / "out.json"
        code, out, err = run_cli([
            "run",
            "--corpus", str(tiny_corpus),
            "--gold", str(tiny_gold),
            "--out", str(out_path),
        ])
        assert code == EXIT_OK, err
        assert out_path.exists()
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["tool_version"]

    def test_missing_corpus(self, tmp_path: Path):
        gold = tmp_path / "g.json"
        gold.write_text(json.dumps({"questions": []}), encoding="utf-8")
        out = tmp_path / "out.json"
        code, stdout, stderr = run_cli([
            "run",
            "--corpus", str(tmp_path / "nope"),
            "--gold", str(gold),
            "--out", str(out),
        ])
        assert code == EXIT_USAGE
        assert "corpus not found" in stderr

    def test_missing_gold(self, tiny_corpus, tmp_path: Path):
        code, stdout, stderr = run_cli([
            "run",
            "--corpus", str(tiny_corpus),
            "--gold", str(tmp_path / "nope.json"),
            "--out", str(tmp_path / "out.json"),
        ])
        assert code == EXIT_USAGE
        assert "gold set not found" in stderr

    def test_bad_chunker_args_json(self, tiny_corpus, tiny_gold, tmp_path: Path):
        code, stdout, stderr = run_cli([
            "run",
            "--corpus", str(tiny_corpus),
            "--gold", str(tiny_gold),
            "--out", str(tmp_path / "o.json"),
            "--chunker-args", "{not json",
        ])
        assert code == EXIT_USAGE

    def test_bad_k_value(self, tiny_corpus, tiny_gold, tmp_path: Path):
        code, stdout, stderr = run_cli([
            "run",
            "--corpus", str(tiny_corpus),
            "--gold", str(tiny_gold),
            "--out", str(tmp_path / "o.json"),
            "--k", "-1",
        ])
        assert code == EXIT_USAGE

    def test_unknown_chunker(self, tiny_corpus, tiny_gold, tmp_path: Path):
        code, stdout, stderr = run_cli([
            "run",
            "--corpus", str(tiny_corpus),
            "--gold", str(tiny_gold),
            "--out", str(tmp_path / "o.json"),
            "--chunker", "fake-chunker",
        ])
        assert code == EXIT_USAGE

    def test_compact_output(self, tiny_corpus, tiny_gold, tmp_path: Path):
        out_path = tmp_path / "c.json"
        code, stdout, stderr = run_cli([
            "run",
            "--corpus", str(tiny_corpus),
            "--gold", str(tiny_gold),
            "--out", str(out_path),
            "--compact",
        ])
        assert code == EXIT_OK
        assert "\n  " not in out_path.read_text(encoding="utf-8")


class TestDiffCommand:
    def _run_and_save(self, tiny_corpus, tiny_gold, tmp_path: Path, name: str, label: str = ""):
        from ragcheck.runner import RunConfig, dump_result_json, run_evaluation
        config = RunConfig(
            corpus_path=str(tiny_corpus),
            gold_path=str(tiny_gold),
            label=label,
        )
        r = run_evaluation(config=config)
        out = tmp_path / name
        dump_result_json(r, out)
        return out

    def test_identical_runs_exit_zero(self, tiny_corpus, tiny_gold, tmp_path: Path):
        base = self._run_and_save(tiny_corpus, tiny_gold, tmp_path, "base.json", "b")
        head = self._run_and_save(tiny_corpus, tiny_gold, tmp_path, "head.json", "h")
        code, out, err = run_cli(["diff", str(base), str(head)])
        assert code == EXIT_OK

    def test_missing_baseline_usage_error(self, tmp_path: Path):
        (tmp_path / "h.json").write_text(
            '{"summary": {}, "config": {}, "corpus_stats": {}}', encoding="utf-8"
        )
        code, out, err = run_cli(["diff", str(tmp_path / "missing.json"), str(tmp_path / "h.json")])
        assert code == EXIT_USAGE

    def test_degradation_detected(self, tmp_path: Path):
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text(json.dumps({"summary": {"recall@5": 0.9}, "config": {}, "corpus_stats": {}}), encoding="utf-8")
        b.write_text(json.dumps({"summary": {"recall@5": 0.7}, "config": {}, "corpus_stats": {}}), encoding="utf-8")
        code, out, err = run_cli(["diff", str(a), str(b)])
        assert code == 1

    def test_no_fail_override(self, tmp_path: Path):
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text(json.dumps({"summary": {"recall@5": 0.9}, "config": {}, "corpus_stats": {}}), encoding="utf-8")
        b.write_text(json.dumps({"summary": {"recall@5": 0.1}, "config": {}, "corpus_stats": {}}), encoding="utf-8")
        code, out, err = run_cli(["diff", str(a), str(b), "--no-fail"])
        assert code == EXIT_OK

    def test_writes_json_out(self, tmp_path: Path):
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text(json.dumps({"summary": {}, "config": {}, "corpus_stats": {}}), encoding="utf-8")
        b.write_text(json.dumps({"summary": {}, "config": {}, "corpus_stats": {}}), encoding="utf-8")
        out_path = tmp_path / "diff.json"
        code, _, _ = run_cli(["diff", str(a), str(b), "--out", str(out_path)])
        assert code == EXIT_OK
        assert out_path.exists()

    def test_writes_markdown(self, tmp_path: Path):
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text(json.dumps({"summary": {}, "config": {}, "corpus_stats": {}}), encoding="utf-8")
        b.write_text(json.dumps({"summary": {}, "config": {}, "corpus_stats": {}}), encoding="utf-8")
        md_path = tmp_path / "d.md"
        code, _, _ = run_cli(["diff", str(a), str(b), "--markdown", str(md_path)])
        assert code == EXIT_OK
        assert md_path.exists()

    def test_bad_threshold_format(self, tmp_path: Path):
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text(json.dumps({"summary": {}, "config": {}, "corpus_stats": {}}), encoding="utf-8")
        b.write_text(json.dumps({"summary": {}, "config": {}, "corpus_stats": {}}), encoding="utf-8")
        code, _, err = run_cli(["diff", str(a), str(b), "--threshold", "bad_no_equals"])
        assert code == EXIT_USAGE


class TestBenchCommand:
    def test_runs_ok(self):
        code, out, err = run_cli(["bench"])
        assert code == EXIT_OK
        assert "fixture" in out

    def test_json_mode(self):
        code, out, err = run_cli(["bench", "--json"])
        assert code == EXIT_OK
        data = json.loads(out)
        assert isinstance(data, list)
        assert len(data) == 3


class TestSynthCommand:
    def test_basic(self, tiny_corpus, tmp_path: Path):
        out_path = tmp_path / "gold.json"
        code, out, err = run_cli([
            "synth",
            "--corpus", str(tiny_corpus),
            "--questions", "3",
            "--out", str(out_path),
        ])
        assert code == EXIT_OK
        assert out_path.exists()
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["synthetic"] is True
        assert len(data["questions"]) == 3

    def test_warns_synthetic(self, tiny_corpus, tmp_path: Path):
        code, out, err = run_cli([
            "synth",
            "--corpus", str(tiny_corpus),
            "--questions", "2",
            "--out", str(tmp_path / "g.json"),
        ])
        assert "synthetic" in out.lower()

    def test_invalid_questions_zero(self, tiny_corpus, tmp_path: Path):
        code, out, err = run_cli([
            "synth",
            "--corpus", str(tiny_corpus),
            "--questions", "0",
            "--out", str(tmp_path / "g.json"),
        ])
        assert code == EXIT_USAGE

    def test_missing_corpus(self, tmp_path: Path):
        code, out, err = run_cli([
            "synth",
            "--corpus", str(tmp_path / "missing"),
            "--questions", "2",
            "--out", str(tmp_path / "g.json"),
        ])
        assert code == EXIT_USAGE


class TestReportCommand:
    def test_renders_markdown(self, tiny_corpus, tiny_gold, tmp_path: Path):
        from ragcheck.runner import RunConfig, dump_result_json, run_evaluation
        r = run_evaluation(config=RunConfig(corpus_path=str(tiny_corpus), gold_path=str(tiny_gold)))
        run_path = tmp_path / "r.json"
        dump_result_json(r, run_path)
        code, out, err = run_cli(["report", "--in", str(run_path), "--format", "md"])
        assert code == EXIT_OK
        assert "ragcheck report" in out

    def test_renders_html_to_file(self, tiny_corpus, tiny_gold, tmp_path: Path):
        from ragcheck.runner import RunConfig, dump_result_json, run_evaluation
        r = run_evaluation(config=RunConfig(corpus_path=str(tiny_corpus), gold_path=str(tiny_gold)))
        run_path = tmp_path / "r.json"
        out_path = tmp_path / "r.html"
        dump_result_json(r, run_path)
        code, _, _ = run_cli(["report", "--in", str(run_path), "--format", "html", "--out", str(out_path)])
        assert code == EXIT_OK
        assert out_path.exists()

    def test_renders_json(self, tiny_corpus, tiny_gold, tmp_path: Path):
        from ragcheck.runner import RunConfig, dump_result_json, run_evaluation
        r = run_evaluation(config=RunConfig(corpus_path=str(tiny_corpus), gold_path=str(tiny_gold)))
        run_path = tmp_path / "r.json"
        dump_result_json(r, run_path)
        code, out, err = run_cli(["report", "--in", str(run_path), "--format", "json"])
        assert code == EXIT_OK
        data = json.loads(out)
        assert "summary" in data

    def test_missing_input(self, tmp_path: Path):
        code, out, err = run_cli(["report", "--in", str(tmp_path / "missing.json")])
        assert code == EXIT_USAGE


class TestChunkersCommand:
    def test_list(self):
        code, out, err = run_cli(["chunkers"])
        assert code == EXIT_OK
        assert "fixed-token" in out

    def test_list_explicit(self):
        code, out, err = run_cli(["chunkers", "list"])
        assert code == EXIT_OK


class TestEmbeddersCommand:
    def test_list(self):
        code, out, err = run_cli(["embedders"])
        assert code == EXIT_OK
        assert "hash" in out


class TestReadmeQuickstart:
    """Regression (H2): the two chunker-args examples shown in README must
    parse and run. Previously:
      - sliding-window used ``window_tokens``/``stride_tokens`` (actual
        params are ``size``/``stride``)
      - semantic-boundary used ``overlap_chars`` (no such parameter exists)
    Both failed with exit code 2 on copy-paste. This test uses the exact
    argument shapes now documented in the README.
    """

    def test_readme_sliding_window_example(self, tmp_path, tiny_corpus, tiny_gold):
        out_path = tmp_path / "sliding.json"
        code, out, err = run_cli([
            "run",
            "--corpus", str(tiny_corpus),
            "--gold", str(tiny_gold),
            "--out", str(out_path),
            "--chunker", "sliding-window",
            "--chunker-args", '{"size": 120, "stride": 60}',
        ])
        assert code == EXIT_OK, f"expected 0, got {code}: {err}"
        assert out_path.exists()

    def test_readme_semantic_boundary_example(self, tmp_path, tiny_corpus, tiny_gold):
        out_path = tmp_path / "sb.json"
        code, out, err = run_cli([
            "run",
            "--corpus", str(tiny_corpus),
            "--gold", str(tiny_gold),
            "--out", str(out_path),
            "--chunker", "semantic-boundary",
            "--chunker-args", '{"max_chars": 1200}',
        ])
        assert code == EXIT_OK, f"expected 0, got {code}: {err}"
        assert out_path.exists()

    def test_legacy_window_tokens_rejected(self, tmp_path, tiny_corpus, tiny_gold):
        """Explicitly lock in that the old (wrong) parameter name errors."""
        out_path = tmp_path / "x.json"
        code, out, err = run_cli([
            "run",
            "--corpus", str(tiny_corpus),
            "--gold", str(tiny_gold),
            "--out", str(out_path),
            "--chunker", "sliding-window",
            "--chunker-args", '{"window_tokens": 120, "stride_tokens": 60}',
        ])
        assert code != EXIT_OK
