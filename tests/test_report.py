"""Tests for JSON/HTML/Markdown report rendering."""
from __future__ import annotations

import json
from pathlib import Path

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
