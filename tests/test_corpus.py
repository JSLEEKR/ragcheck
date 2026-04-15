"""Tests for corpus and gold-set loading."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragcheck.corpus import (
    Document,
    GoldSet,
    Query,
    iter_corpus_paths,
    load_corpus,
    load_gold_set,
    save_gold_set,
)


class TestLoadCorpus:
    def test_loads_txt_and_md(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
        (tmp_path / "b.md").write_text("beta", encoding="utf-8")
        docs = load_corpus(tmp_path)
        ids = [d.doc_id for d in docs]
        assert "a" in ids
        assert "b" in ids

    def test_missing_directory(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_corpus(tmp_path / "missing")

    def test_not_a_directory(self, tmp_path: Path):
        f = tmp_path / "file.txt"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            load_corpus(f)

    def test_ignores_hidden_files(self, tmp_path: Path):
        (tmp_path / ".hidden.txt").write_text("secret", encoding="utf-8")
        (tmp_path / "visible.txt").write_text("ok", encoding="utf-8")
        docs = load_corpus(tmp_path)
        assert len(docs) == 1
        assert docs[0].doc_id == "visible"

    def test_ignores_unknown_extensions(self, tmp_path: Path):
        (tmp_path / "code.py").write_text("print()", encoding="utf-8")
        (tmp_path / "note.txt").write_text("note", encoding="utf-8")
        docs = load_corpus(tmp_path)
        assert len(docs) == 1

    def test_nested_directories(self, tmp_path: Path):
        nested = tmp_path / "sub"
        nested.mkdir()
        (nested / "inner.txt").write_text("inner", encoding="utf-8")
        docs = load_corpus(tmp_path)
        assert docs[0].doc_id == "sub/inner"

    def test_sorted_by_doc_id(self, tmp_path: Path):
        (tmp_path / "z.txt").write_text("z", encoding="utf-8")
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "m.txt").write_text("m", encoding="utf-8")
        docs = load_corpus(tmp_path)
        assert [d.doc_id for d in docs] == ["a", "m", "z"]

    def test_utf8_content(self, tmp_path: Path):
        (tmp_path / "f.txt").write_text("한글 テキスト", encoding="utf-8")
        docs = load_corpus(tmp_path)
        assert "한글" in docs[0].text

    def test_empty_file_included(self, tmp_path: Path):
        (tmp_path / "empty.txt").write_text("", encoding="utf-8")
        docs = load_corpus(tmp_path)
        assert len(docs) == 1
        assert docs[0].text == ""

    def test_len_returns_text_length(self, tmp_path: Path):
        (tmp_path / "x.txt").write_text("hello", encoding="utf-8")
        docs = load_corpus(tmp_path)
        assert len(docs[0]) == 5

    def test_iter_corpus_paths(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "b.md").write_text("b", encoding="utf-8")
        paths = list(iter_corpus_paths(tmp_path))
        assert len(paths) == 2


class TestLoadGoldSet:
    def test_basic(self, tmp_path: Path):
        data = {
            "version": 1,
            "questions": [
                {"query_id": "q1", "text": "hi", "relevant_doc_ids": ["a"]},
            ],
        }
        p = tmp_path / "g.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        g = load_gold_set(p)
        assert g.version == 1
        assert len(g.questions) == 1
        assert g.questions[0].query_id == "q1"

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_gold_set(tmp_path / "x.json")

    def test_not_an_object(self, tmp_path: Path):
        p = tmp_path / "g.json"
        p.write_text(json.dumps([1, 2]), encoding="utf-8")
        with pytest.raises(ValueError):
            load_gold_set(p)

    def test_questions_not_a_list(self, tmp_path: Path):
        p = tmp_path / "g.json"
        p.write_text(json.dumps({"questions": "oops"}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_gold_set(p)

    def test_duplicate_query_id(self, tmp_path: Path):
        p = tmp_path / "g.json"
        p.write_text(
            json.dumps({
                "questions": [
                    {"query_id": "q1", "text": "a"},
                    {"query_id": "q1", "text": "b"},
                ],
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_gold_set(p)

    def test_auto_assigns_query_id_when_missing(self, tmp_path: Path):
        p = tmp_path / "g.json"
        p.write_text(
            json.dumps({"questions": [{"text": "a"}, {"text": "b"}]}),
            encoding="utf-8",
        )
        g = load_gold_set(p)
        ids = [q.query_id for q in g.questions]
        assert len(set(ids)) == 2

    def test_relevance_numeric_coerced(self, tmp_path: Path):
        p = tmp_path / "g.json"
        p.write_text(
            json.dumps({
                "questions": [
                    {
                        "query_id": "q1",
                        "text": "a",
                        "relevance": {"d": 2, "e": "3"},
                    },
                ],
            }),
            encoding="utf-8",
        )
        g = load_gold_set(p)
        assert g.questions[0].relevance == {"d": 2.0, "e": 3.0}

    def test_relevance_negative_rejected(self, tmp_path: Path):
        p = tmp_path / "g.json"
        p.write_text(
            json.dumps({"questions": [{"query_id": "q", "relevance": {"d": -1}}]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_gold_set(p)

    def test_relevance_bad_type_rejected(self, tmp_path: Path):
        p = tmp_path / "g.json"
        p.write_text(
            json.dumps({"questions": [{"query_id": "q", "relevance": {"d": "bad"}}]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_gold_set(p)

    def test_questions_sorted_by_id(self, tmp_path: Path):
        p = tmp_path / "g.json"
        p.write_text(
            json.dumps({
                "questions": [
                    {"query_id": "z"},
                    {"query_id": "a"},
                    {"query_id": "m"},
                ],
            }),
            encoding="utf-8",
        )
        g = load_gold_set(p)
        assert [q.query_id for q in g.questions] == ["a", "m", "z"]


class TestSaveGoldSet:
    def test_roundtrip(self, tmp_path: Path):
        gold = GoldSet(
            version=1,
            source="x",
            questions=[
                Query(
                    query_id="q1",
                    text="hi",
                    relevant_doc_ids=["a"],
                    relevance={"a": 1.0},
                )
            ],
        )
        p = tmp_path / "out.json"
        save_gold_set(gold, p)
        assert p.exists()
        loaded = load_gold_set(p)
        assert loaded.questions[0].query_id == "q1"

    def test_writes_deterministic_json(self, tmp_path: Path):
        gold = GoldSet(
            version=1,
            questions=[
                Query(query_id="q1", text="a", relevant_doc_ids=["z", "a"]),
            ],
        )
        p = tmp_path / "a.json"
        save_gold_set(gold, p)
        content = p.read_text(encoding="utf-8")
        # relevant_doc_ids should be sorted
        assert content.find('"a"') < content.find('"z"')

    def test_creates_parent_dir(self, tmp_path: Path):
        p = tmp_path / "nested" / "deep" / "out.json"
        gold = GoldSet()
        save_gold_set(gold, p)
        assert p.exists()


class TestDocumentAndQuery:
    def test_document_frozen(self):
        from dataclasses import FrozenInstanceError
        d = Document(doc_id="a", path="a.txt", text="hi")
        with pytest.raises(FrozenInstanceError):
            d.doc_id = "x"  # type: ignore[misc]

    def test_query_to_dict_sorted(self):
        q = Query(
            query_id="q1",
            text="hi",
            relevant_doc_ids=["z", "a"],
            relevance={"z": 1.0, "a": 2.0},
        )
        out = q.to_dict()
        assert out["relevant_doc_ids"] == ["a", "z"]
        assert list(out["relevance"].keys()) == ["a", "z"]

    def test_gold_set_to_dict(self):
        g = GoldSet(version=1, questions=[Query(query_id="q1", text="a")])
        out = g.to_dict()
        assert out["version"] == 1
        assert len(out["questions"]) == 1
