"""Tests for JSON/HTML/Markdown report rendering."""
from __future__ import annotations

import json
import math

import pytest

from ragcheck.report import render_html, render_json, render_markdown
from ragcheck.runner import RunConfig, run_evaluation


@pytest.fixture
def sample_run(tiny_corpus, tiny_gold):
    config = RunConfig(
        corpus_path=str(tiny_corpus),
        gold_path=str(tiny_gold),
    )
    r = run_evaluation(config=config)
    return r.to_dict()


class TestRenderJSON:
    def test_pretty_default(self):
        text = render_json({"b": 1, "a": 2})
        assert text.endswith("\n")
        # sorted keys: "a" before "b"
        assert text.find('"a"') < text.find('"b"')

    def test_compact(self):
        text = render_json({"b": 1, "a": 2}, pretty=False)
        assert "\n  " not in text

    def test_roundtrip(self, sample_run):
        text = render_json(sample_run)
        data = json.loads(text)
        assert data["tool_version"]

    def test_unicode_preserved(self):
        text = render_json({"k": "한글"})
        assert "한글" in text


class TestRenderMarkdown:
    def test_contains_sections(self, sample_run):
        md = render_markdown(sample_run)
        assert "# ragcheck report" in md
        assert "Aggregate metrics" in md
        assert "Chunking diagnostics" in md
        assert "Per-query metrics" in md

    def test_config_table(self, sample_run):
        md = render_markdown(sample_run)
        assert "| Corpus |" in md

    def test_histogram_ascii_present(self, sample_run):
        md = render_markdown(sample_run)
        assert "Size histogram" in md

    def test_handles_empty_run(self):
        md = render_markdown({})
        assert "ragcheck report" in md

    def test_handles_missing_diagnostics(self, sample_run):
        del sample_run["diagnostics"]
        md = render_markdown(sample_run)
        assert "Chunking diagnostics" not in md


class TestRenderHTML:
    def test_contains_html_skeleton(self, sample_run):
        html = render_html(sample_run)
        assert html.startswith("<!doctype html>")
        assert "<title>ragcheck report" in html

    def test_escapes_html_entities(self):
        run = {
            "config": {"label": "<script>alert(1)</script>"},
            "summary": {},
            "corpus_stats": {},
            "diagnostics": {},
            "per_query": [],
        }
        html = render_html(run)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_summary_rows_rendered(self, sample_run):
        html = render_html(sample_run)
        assert "<tr><td>recall@5</td>" in html or "recall@5" in html

    def test_empty_run_renders(self):
        html = render_html({})
        assert "<body>" in html

    def test_per_query_section_when_present(self, sample_run):
        html = render_html(sample_run)
        assert "Per-query metrics" in html

    def test_no_per_query_section_when_empty(self):
        run = {
            "config": {"label": "x"},
            "summary": {},
            "corpus_stats": {},
            "diagnostics": {},
            "per_query": [],
        }
        html = render_html(run)
        assert "Per-query metrics" not in html


class TestRegressionCycleA:
    """Regression tests for bugs found in Phase 3 Eval Cycle A."""

    def test_render_html_handles_none_per_query_metric(self):
        """H3: per-query metric of None must not crash render_html.

        The runner emits ``None`` for metrics that are undefined on a given
        query (e.g. MRR when the query has zero relevants). The HTML
        renderer must render that as ``null`` rather than raising
        ``TypeError: float() argument must be ... not 'NoneType'``.
        """
        run = {
            "config": {"label": "x"},
            "summary": {},
            "corpus_stats": {},
            "diagnostics": {},
            "per_query": [
                {"query_id": "q1", "metrics": {"recall@5": None, "mrr": 0.5}},
                {"query_id": "q2", "metrics": {"recall@5": 0.25, "mrr": None}},
            ],
        }
        html = render_html(run)
        assert "<td>null</td>" in html
        assert "q1" in html and "q2" in html

    def test_render_markdown_handles_none_per_query_metric(self):
        """Mirror of H3 — Markdown path already handled None; lock it in."""
        run = {
            "config": {"label": "x"},
            "summary": {},
            "corpus_stats": {},
            "diagnostics": {},
            "per_query": [
                {"query_id": "q1", "metrics": {"recall@5": None, "mrr": 0.5}},
            ],
        }
        md = render_markdown(run)
        # Columns are sorted alphabetically (mrr, recall@5).
        assert "| q1 | 0.500000 | null |" in md

    def test_markdown_normalises_windows_paths(self):
        """M1: backslash paths must be rendered with forward slashes so
        reports generated on Windows compare byte-for-byte with Linux."""
        run = {
            "config": {
                "label": "x",
                "corpus_path": "C:\\data\\corpus",
                "gold_path": "C:\\data\\gold.json",
            },
            "summary": {},
            "corpus_stats": {},
            "diagnostics": {},
            "per_query": [],
        }
        md = render_markdown(run)
        assert "C:/data/corpus" in md
        assert "C:\\data\\corpus" not in md

    def test_html_normalises_windows_paths(self):
        run = {
            "config": {
                "label": "x",
                "corpus_path": "C:\\data\\corpus",
            },
            "summary": {},
            "corpus_stats": {},
            "diagnostics": {},
            "per_query": [],
        }
        html = render_html(run)
        assert "C:/data/corpus" in html
        assert "C:\\data\\corpus" not in html


class TestRegressionCycleB:
    """Regression tests for bugs found in Phase 3 Eval Cycle B."""

    def test_summary_nan_inf_render_as_null_markdown(self):
        """Cycle B M2: NaN / +Inf / -Inf in the *summary* path must render as
        ``null`` in Markdown rather than leaking ``nan`` / ``inf`` literals.
        Cycle A only fixed the per-query path and missed this codepath.
        """
        run = {
            "config": {"label": "x"},
            "summary": {
                "recall@5": float("nan"),
                "mrr": float("inf"),
                "ndcg@5": float("-inf"),
                "precision@5": 0.5,
            },
            "corpus_stats": {},
            "diagnostics": {},
            "per_query": [],
        }
        md = render_markdown(run)
        assert "nan" not in md.lower()
        assert "inf" not in md.lower()
        # Each non-finite metric should render as null
        assert "| recall@5 | null |" in md
        assert "| mrr | null |" in md
        assert "| ndcg@5 | null |" in md
        # Finite values still render with 6-decimal precision
        assert "| precision@5 | 0.500000 |" in md

    def test_summary_nan_inf_render_as_null_html(self):
        run = {
            "config": {"label": "x"},
            "summary": {
                "recall@5": float("nan"),
                "mrr": float("inf"),
                "ndcg@5": float("-inf"),
                "precision@5": 0.5,
            },
            "corpus_stats": {},
            "diagnostics": {},
            "per_query": [],
        }
        html = render_html(run)
        # Non-finite floats must be filtered out before string formatting
        assert "nan" not in html.lower()
        assert "inf" not in html.lower()
        assert "<td>null</td>" in html
        assert "0.500000" in html

    def test_per_query_nan_inf_render_as_null(self):
        run = {
            "config": {"label": "x"},
            "summary": {},
            "corpus_stats": {},
            "diagnostics": {},
            "per_query": [
                {
                    "query_id": "q1",
                    "metrics": {
                        "recall@5": float("nan"),
                        "mrr": float("inf"),
                    },
                },
            ],
        }
        md = render_markdown(run)
        html = render_html(run)
        assert "nan" not in md.lower()
        assert "inf" not in md.lower()
        assert "nan" not in html.lower()
        assert "inf" not in html.lower()
        # Sanity: math.nan/inf would otherwise be the literal Python repr
        assert math.isnan(float("nan"))


class TestRenderDeterminism:
    def test_markdown_deterministic(self, sample_run):
        a = render_markdown(sample_run)
        b = render_markdown(sample_run)
        assert a == b

    def test_html_deterministic(self, sample_run):
        a = render_html(sample_run)
        b = render_html(sample_run)
        assert a == b

    def test_json_deterministic(self, sample_run):
        a = render_json(sample_run)
        b = render_json(sample_run)
        assert a == b
