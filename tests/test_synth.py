"""Tests for the synthetic gold-set bootstrap."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragcheck.corpus import load_gold_set, save_gold_set
from ragcheck.synth import _sentence_to_question, synth_gold_set


class TestSynthGoldSet:
    def test_generates_requested_count(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text(
            "The interest rate is the cost of money. Central banks adjust it.",
            encoding="utf-8",
        )
        (tmp_path / "b.txt").write_text(
            "Kubernetes is a container orchestrator. It schedules pods.",
            encoding="utf-8",
        )
        gold = synth_gold_set(tmp_path, n_questions=3, seed=42)
        assert len(gold.questions) == 3
        assert gold.synthetic is True
        for q in gold.questions:
            assert q.synthetic is True
            assert len(q.relevant_doc_ids) == 1

    def test_deterministic(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("One. Two. Three.", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Alpha. Beta. Gamma.", encoding="utf-8")
        g1 = synth_gold_set(tmp_path, n_questions=4, seed=7)
        g2 = synth_gold_set(tmp_path, n_questions=4, seed=7)
        assert [q.query_id for q in g1.questions] == [q.query_id for q in g2.questions]
        assert [q.text for q in g1.questions] == [q.text for q in g2.questions]

    def test_empty_corpus(self, tmp_path: Path):
        gold = synth_gold_set(tmp_path, n_questions=5)
        assert gold.questions == []
        assert gold.synthetic is True

    def test_invalid_n_questions(self, tmp_path: Path):
        with pytest.raises(ValueError):
            synth_gold_set(tmp_path, n_questions=0)

    def test_roundtrip_through_file(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("Hello. World.", encoding="utf-8")
        gold = synth_gold_set(tmp_path, n_questions=2, seed=1)
        out = tmp_path / "gold.json"
        save_gold_set(gold, out)
        loaded = load_gold_set(out)
        assert len(loaded.questions) == 2
        assert loaded.synthetic is True

    def test_cycles_when_more_questions_than_docs(self, tmp_path: Path):
        (tmp_path / "only.txt").write_text("The one and only doc.", encoding="utf-8")
        gold = synth_gold_set(tmp_path, n_questions=3, seed=42)
        assert len(gold.questions) == 3
        # All should target the same doc
        assert all(q.relevant_doc_ids == ["only"] for q in gold.questions)


class TestSentenceToQuestion:
    def test_strips_trailing_punctuation(self):
        q = _sentence_to_question("The rate is 5%.")
        assert not q.endswith("..")
        assert q.endswith("?")

    def test_the_leading_article_replaced(self):
        q = _sentence_to_question("The Docker daemon runs in the background.")
        assert q.lower().startswith("what")

    def test_long_sentence_truncated(self):
        long = "A " * 200
        q = _sentence_to_question(long)
        assert len(q) < 200

    def test_not_leading_article(self):
        q = _sentence_to_question("Python uses indentation.")
        assert q.startswith("What about")

    def test_empty_input(self):
        q = _sentence_to_question("")
        assert q.endswith("?")
