"""Tests for regression diff."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragcheck.diff import (
    DEFAULT_THRESHOLDS,
    diff_runs,
    dump_diff_json,
    render_diff_markdown,
)


def _run_payload(summary: dict, label: str = "", corpus_sha: str = "") -> dict:
    return {
        "config": {"label": label},
        "summary": dict(summary),
        "corpus_stats": {"corpus_sha1": corpus_sha},
    }


class TestDiffBasics:
    def test_identical_runs_are_flat(self):
        base = _run_payload({"recall@5": 0.5, "mrr": 0.5})
        head = _run_payload({"recall@5": 0.5, "mrr": 0.5})
        diff = diff_runs(base, head)
        assert diff.exit_code == 0
        assert diff.degraded == []
        assert set(diff.flat) == {"recall@5", "mrr"}

    def test_detects_degradation(self):
        base = _run_payload({"recall@5": 0.5})
        head = _run_payload({"recall@5": 0.40})
        diff = diff_runs(base, head)
        assert diff.exit_code == 1
        assert "recall@5" in diff.degraded

    def test_detects_improvement(self):
        base = _run_payload({"recall@5": 0.5})
        head = _run_payload({"recall@5": 0.60})
        diff = diff_runs(base, head)
        assert diff.exit_code == 0
        assert "recall@5" in diff.improved

    def test_small_change_is_flat(self):
        base = _run_payload({"recall@5": 0.5})
        head = _run_payload({"recall@5": 0.505})
        diff = diff_runs(base, head)
        assert "recall@5" in diff.flat

    def test_custom_threshold_metric_specific(self):
        base = _run_payload({"recall@5": 0.5})
        head = _run_payload({"recall@5": 0.495})
        diff = diff_runs(base, head, thresholds={"recall": 0.001})
        assert diff.exit_code == 1

    def test_no_fail_option(self):
        base = _run_payload({"recall@5": 0.5})
        head = _run_payload({"recall@5": 0.1})
        diff = diff_runs(base, head, fail_on_degraded=False)
        assert diff.exit_code == 0
        assert "recall@5" in diff.degraded


class TestDiffEdgeCases:
    def test_metric_new_in_head(self):
        base = _run_payload({"recall@5": 0.5})
        head = _run_payload({"recall@5": 0.5, "ndcg@5": 0.7})
        diff = diff_runs(base, head)
        statuses = {d.metric: d.status for d in diff.deltas}
        assert statuses["ndcg@5"] == "new"

    def test_metric_removed_from_head_is_degraded(self):
        base = _run_payload({"recall@5": 0.5, "ndcg@5": 0.8})
        head = _run_payload({"recall@5": 0.5})
        diff = diff_runs(base, head)
        statuses = {d.metric: d.status for d in diff.deltas}
        assert statuses["ndcg@5"] == "removed"
        assert diff.exit_code == 1

    def test_corpus_change_detection(self):
        base = _run_payload({"recall@5": 0.5}, corpus_sha="abc")
        head = _run_payload({"recall@5": 0.5}, corpus_sha="def")
        diff = diff_runs(base, head)
        assert diff.corpus_changed is True

    def test_no_corpus_sha_means_no_change_flag(self):
        base = _run_payload({"recall@5": 0.5})
        head = _run_payload({"recall@5": 0.5})
        diff = diff_runs(base, head)
        assert diff.corpus_changed is False

    def test_empty_summaries(self):
        base = _run_payload({})
        head = _run_payload({})
        diff = diff_runs(base, head)
        assert diff.exit_code == 0
        assert diff.deltas == []

    def test_zero_baseline_rel_delta_positive(self):
        base = _run_payload({"recall@5": 0.0})
        head = _run_payload({"recall@5": 0.1})
        diff = diff_runs(base, head)
        by_metric = {d.metric: d for d in diff.deltas}
        assert by_metric["recall@5"].rel_delta == 1.0

    def test_zero_both_rel_delta_zero(self):
        base = _run_payload({"recall@5": 0.0})
        head = _run_payload({"recall@5": 0.0})
        diff = diff_runs(base, head)
        by_metric = {d.metric: d for d in diff.deltas}
        assert by_metric["recall@5"].rel_delta == 0.0

    def test_zero_baseline_negative_head_rel_delta(self):
        base = _run_payload({"recall@5": 0.0})
        head = _run_payload({"recall@5": -0.001})
        diff = diff_runs(base, head)
        by_metric = {d.metric: d for d in diff.deltas}
        assert by_metric["recall@5"].rel_delta == -1.0


class TestDiffSerialization:
    def test_to_dict_sorted_metrics(self):
        base = _run_payload({"recall@5": 0.5, "mrr": 0.5})
        head = _run_payload({"recall@5": 0.5, "mrr": 0.5})
        diff = diff_runs(base, head)
        out = diff.to_dict()
        metric_order = [d["metric"] for d in out["deltas"]]
        assert metric_order == sorted(metric_order)

    def test_dump_json_writes_file(self, tmp_path: Path):
        base = _run_payload({"recall@5": 0.5})
        head = _run_payload({"recall@5": 0.5})
        diff = diff_runs(base, head)
        p = tmp_path / "diff.json"
        dump_diff_json(diff, p)
        data = json.loads(p.read_text(encoding="utf-8"))
        assert "deltas" in data
        assert "summary" in data

    def test_dump_json_creates_parent_dir(self, tmp_path: Path):
        base = _run_payload({})
        head = _run_payload({})
        diff = diff_runs(base, head)
        p = tmp_path / "nested" / "diff.json"
        dump_diff_json(diff, p)
        assert p.exists()

    def test_dump_json_deterministic(self, tmp_path: Path):
        base = _run_payload({"recall@5": 0.5, "mrr": 0.6})
        head = _run_payload({"recall@5": 0.51, "mrr": 0.59})
        diff1 = diff_runs(base, head)
        diff2 = diff_runs(base, head)
        p1 = tmp_path / "a.json"
        p2 = tmp_path / "b.json"
        dump_diff_json(diff1, p1)
        dump_diff_json(diff2, p2)
        assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")


class TestDiffMarkdown:
    def test_renders_header(self):
        base = _run_payload({"recall@5": 0.5}, label="baseline")
        head = _run_payload({"recall@5": 0.5}, label="candidate")
        diff = diff_runs(base, head)
        md = render_diff_markdown(diff)
        assert "# ragcheck regression diff" in md
        assert "baseline" in md
        assert "candidate" in md

    def test_renders_all_metrics_table(self):
        base = _run_payload({"recall@5": 0.5, "mrr": 0.5})
        head = _run_payload({"recall@5": 0.5, "mrr": 0.55})
        diff = diff_runs(base, head)
        md = render_diff_markdown(diff)
        assert "recall@5" in md
        assert "mrr" in md

    def test_renders_degraded_section_when_present(self):
        base = _run_payload({"recall@5": 0.5})
        head = _run_payload({"recall@5": 0.3})
        diff = diff_runs(base, head)
        md = render_diff_markdown(diff)
        assert "Degraded metrics" in md

    def test_omits_degraded_section_when_none(self):
        base = _run_payload({"recall@5": 0.5})
        head = _run_payload({"recall@5": 0.5})
        diff = diff_runs(base, head)
        md = render_diff_markdown(diff)
        assert "Degraded metrics" not in md

    def test_exit_code_text_shown(self):
        base = _run_payload({"recall@5": 0.5})
        head = _run_payload({"recall@5": 0.1})
        diff = diff_runs(base, head)
        md = render_diff_markdown(diff)
        assert "Exit code" in md
        assert "1" in md

    def test_corpus_changed_mentioned(self):
        base = _run_payload({"recall@5": 0.5}, corpus_sha="abc")
        head = _run_payload({"recall@5": 0.5}, corpus_sha="def")
        diff = diff_runs(base, head)
        md = render_diff_markdown(diff)
        assert "Corpus changed" in md


class TestThresholdMatching:
    def test_exact_metric_threshold_wins(self):
        base = _run_payload({"recall@5": 0.5})
        head = _run_payload({"recall@5": 0.498})
        diff = diff_runs(base, head, thresholds={"recall@5": 0.001, "recall": 0.5})
        assert diff.exit_code == 1

    def test_prefix_threshold_applied(self):
        base = _run_payload({"recall@5": 0.5})
        head = _run_payload({"recall@5": 0.495})
        diff = diff_runs(base, head, thresholds={"recall": 0.001})
        assert diff.exit_code == 1

    def test_unknown_metric_uses_default(self):
        base = _run_payload({"unknown_metric": 0.5})
        head = _run_payload({"unknown_metric": 0.6})
        diff = diff_runs(base, head)
        # Default threshold 0.02; delta 0.1 > threshold → improved
        assert "unknown_metric" in diff.improved


class TestDefaultThresholds:
    def test_defaults_cover_all_core_metrics(self):
        assert "recall" in DEFAULT_THRESHOLDS
        assert "precision" in DEFAULT_THRESHOLDS
        assert "mrr" in DEFAULT_THRESHOLDS
        assert "ndcg" in DEFAULT_THRESHOLDS
