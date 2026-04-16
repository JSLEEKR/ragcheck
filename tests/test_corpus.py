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

    def test_ignores_hidden_directories(self, tmp_path: Path):
        """Cycle C H1 regression: do not walk into hidden dirs like .git.

        Without this guard, ``ragcheck run --corpus .`` from a repo root
        would silently ingest .git/HEAD and any other dot-directory
        contents as documents.
        """
        git = tmp_path / ".git"
        git.mkdir()
        (git / "config.txt").write_text("SECRET TOKEN", encoding="utf-8")
        (git / "HEAD.md").write_text("ref: refs/heads/main", encoding="utf-8")
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "stub.txt").write_text("stub", encoding="utf-8")
        (tmp_path / "real.txt").write_text("real content", encoding="utf-8")
        docs = load_corpus(tmp_path)
        ids = [d.doc_id for d in docs]
        assert ids == ["real"]
        # Ensure the secret never even appears in the loaded corpus
        for d in docs:
            assert "SECRET" not in d.text

    def test_ignores_nested_hidden_directories(self, tmp_path: Path):
        """A hidden directory at any depth, not just the root."""
        outer = tmp_path / "project"
        outer.mkdir()
        (outer / "doc.md").write_text("real", encoding="utf-8")
        cache = outer / ".cache"
        cache.mkdir()
        (cache / "junk.md").write_text("internal", encoding="utf-8")
        # Even doubly-nested
        deep = outer / "a" / ".bin"
        deep.mkdir(parents=True)
        (deep / "blob.txt").write_text("never", encoding="utf-8")
        docs = load_corpus(tmp_path)
        ids = sorted(d.doc_id for d in docs)
        assert ids == ["project/doc"]


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


class TestLoadCorpusSymlinkSafetyCycleD:
    """Cycle D H1: load_corpus must not follow symlinks/junctions out of root."""

    def test_windows_junction_is_skipped(self, tmp_path: Path):
        """A directory junction (mklink /J) inside the corpus must NOT
        be ingested if it points outside the corpus root.

        On non-Windows or when junction creation fails, the test is
        skipped — the probe cannot be constructed without cmd.exe.
        """
        import os
        import subprocess

        if os.name != "nt":
            pytest.skip("directory junctions are a Windows feature")

        secret = tmp_path / "secret"
        secret.mkdir()
        (secret / "password.txt").write_text("TOPSECRET", encoding="utf-8")
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "real.txt").write_text("real content", encoding="utf-8")

        link = corpus / "escape"
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(secret)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"mklink /J failed (not elevated?): {result.stderr}")

        docs = load_corpus(corpus)
        ids = {d.doc_id for d in docs}
        assert "real" in ids, "legitimate corpus file must still be loaded"
        assert "escape/password" not in ids, (
            "directory junction escaping corpus root MUST be skipped"
        )
        # Ensure the secret text never ended up in any document
        for d in docs:
            assert "TOPSECRET" not in d.text

    def test_posix_symlink_outside_root_is_skipped(self, tmp_path: Path):
        """On POSIX, a file symlink pointing outside the corpus root
        must be skipped. On Windows without admin, file symlink
        creation fails and the test is skipped.
        """
        import os

        secret = tmp_path / "secret"
        secret.mkdir()
        target = secret / "password.txt"
        target.write_text("TOPSECRET", encoding="utf-8")
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "real.txt").write_text("real content", encoding="utf-8")
        try:
            os.symlink(target, corpus / "sym.txt")
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this platform")

        docs = load_corpus(corpus)
        ids = {d.doc_id for d in docs}
        assert "real" in ids
        assert "sym" not in ids
        for d in docs:
            assert "TOPSECRET" not in d.text

    def test_symlink_inside_root_is_allowed(self, tmp_path: Path):
        """A symlink that still resolves inside the corpus root should
        be loaded normally — only escapes are rejected.
        """
        import os

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "real.txt").write_text("original", encoding="utf-8")
        try:
            os.symlink(corpus / "real.txt", corpus / "link.txt")
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this platform")

        docs = load_corpus(corpus)
        ids = {d.doc_id for d in docs}
        # both 'real' and 'link' resolve inside corpus so both are loaded
        assert "real" in ids
        assert "link" in ids


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


class TestLoadGoldSetFiniteRelevanceCycleE:
    """Reject NaN/Inf in gold-set ``relevance`` at load time.

    Cycle E M1: Python's json parser accepts literal ``NaN`` and
    ``Infinity`` tokens as an extension of the JSON spec, so a gold
    file containing ``"relevance": {"a": NaN}`` loaded without error.
    The downstream failure was far from the source — the user saw
    ``per-query score must be finite`` from ``aggregate_metric`` with
    no indication that their gold file was the fault. The fix rejects
    non-finite relevance right at the load boundary with an error that
    mentions the id.
    """

    def test_load_rejects_nan_relevance(self, tmp_path):
        p = tmp_path / "gold.json"
        p.write_text(
            '{"version":1,"questions":[{"query_id":"q1","text":"t",'
            '"relevant_doc_ids":["a"],"relevance":{"a":NaN}}]}'
        )
        with pytest.raises(ValueError, match="finite"):
            load_gold_set(p)

    def test_load_rejects_positive_inf_relevance(self, tmp_path):
        p = tmp_path / "gold.json"
        p.write_text(
            '{"version":1,"questions":[{"query_id":"q1","text":"t",'
            '"relevant_doc_ids":["a"],"relevance":{"a":Infinity}}]}'
        )
        with pytest.raises(ValueError, match="finite"):
            load_gold_set(p)

    def test_load_rejects_negative_inf_relevance(self, tmp_path):
        p = tmp_path / "gold.json"
        p.write_text(
            '{"version":1,"questions":[{"query_id":"q1","text":"t",'
            '"relevant_doc_ids":["a"],"relevance":{"a":-Infinity}}]}'
        )
        with pytest.raises(ValueError):
            load_gold_set(p)

    def test_load_finite_relevance_still_works(self, tmp_path):
        # Regression guard: the happy path still parses.
        p = tmp_path / "gold.json"
        p.write_text(
            '{"version":1,"questions":[{"query_id":"q1","text":"t",'
            '"relevant_doc_ids":["a"],"relevance":{"a":2.5}}]}'
        )
        gold = load_gold_set(p)
        assert gold.questions[0].relevance == {"a": 2.5}


class TestSaveGoldSetSourceNormalizationCycleE:
    """``save_gold_set`` must emit forward-slash ``source`` on every OS.

    Cycle E M3: Cycle D M3 normalised ``RunConfig.corpus_path`` /
    ``gold_path`` in the run JSON so comparisons between Windows and
    POSIX runs are byte-identical. But ``save_gold_set`` (used by
    ``ragcheck synth``) wrote ``source = str(corpus_path)`` verbatim,
    so a gold file produced by ``synth`` on Windows still contained
    literal backslashes and was not byte-identical to one produced on
    POSIX. This completes the cross-platform determinism contract for
    every JSON artefact ragcheck emits.
    """

    def test_save_gold_set_normalises_source_path(self, tmp_path):
        gold = GoldSet(
            version=1,
            source=r"C:\Users\alice\corpus",
            questions=[Query(query_id="q1", text="a")],
            synthetic=True,
        )
        out = tmp_path / "gold.json"
        save_gold_set(gold, out)
        with out.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["source"] == "C:/Users/alice/corpus"

    def test_save_gold_set_posix_source_unchanged(self, tmp_path):
        gold = GoldSet(
            version=1,
            source="/home/alice/corpus",
            questions=[Query(query_id="q1", text="a")],
        )
        out = tmp_path / "gold.json"
        save_gold_set(gold, out)
        with out.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["source"] == "/home/alice/corpus"

    def test_synth_round_trip_source_forward_slash(self, tmp_path):
        # End-to-end: synth on Windows-style path yields a forward-slash
        # source field in the gold JSON, mirroring RunConfig path
        # normalisation so cross-platform determinism holds for the
        # synth subcommand as well.
        from ragcheck.synth import synth_gold_set
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.txt").write_text("Hello world. A test document.")
        (corpus / "b.txt").write_text("Another doc. Different words.")
        gold = synth_gold_set(corpus, n_questions=2, seed=42)
        out = tmp_path / "gold.json"
        save_gold_set(gold, out)
        with out.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # source must never contain a backslash in the serialised form
        assert "\\" not in data["source"]


class TestSaveGoldSetLineEndingsCycleF:
    """Cycle F H3: save_gold_set must write LF line endings on every
    platform. Python's text-mode line translation rewrites ``\\n`` to
    ``\\r\\n`` on Windows unless ``newline="\\n"`` is passed, which
    breaks byte-identical cross-platform determinism.

    Every other JSON writer in the package already passes the kwarg;
    ``save_gold_set`` was the lone holdout. These tests guard against
    regressions that would reintroduce CRLF leakage.
    """

    def test_file_contains_no_crlf_bytes(self, tmp_path: Path):
        # Multi-line JSON (indent=2) is the stress test: many newlines,
        # any of which would be translated to CRLF on Windows without
        # the newline kwarg.
        gold = GoldSet(
            version=1,
            source="corpus",
            questions=[
                Query(
                    query_id=f"q{i:03d}",
                    text=f"question {i}",
                    relevant_doc_ids=[f"doc_{i}"],
                )
                for i in range(5)
            ],
        )
        out = tmp_path / "gold.json"
        save_gold_set(gold, out)
        raw = out.read_bytes()
        assert b"\r\n" not in raw, (
            "save_gold_set must never emit CRLF; cross-platform "
            "determinism requires LF-only line endings"
        )
        # Sanity: must still contain LF newlines from indent=2.
        assert raw.count(b"\n") > 3

    def test_round_trip_after_write_is_byte_stable(self, tmp_path: Path):
        gold = GoldSet(
            version=1,
            source="corpus",
            questions=[
                Query(query_id="q01", text="hello", relevant_doc_ids=["a"]),
                Query(query_id="q02", text="world", relevant_doc_ids=["b"]),
            ],
        )
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        save_gold_set(gold, a)
        reloaded = load_gold_set(a)
        save_gold_set(reloaded, b)
        # Byte-identical across save -> load -> save cycles.
        assert a.read_bytes() == b.read_bytes()

    def test_empty_gold_set_still_lf_only(self, tmp_path: Path):
        gold = GoldSet(version=1, source="c")
        out = tmp_path / "empty.json"
        save_gold_set(gold, out)
        raw = out.read_bytes()
        assert b"\r\n" not in raw
        assert raw.endswith(b"\n")

    def test_matches_dump_result_json_line_ending_contract(self, tmp_path: Path):
        # Cross-check: dump_result_json and dump_diff_json both emit
        # LF-only. save_gold_set must follow the same contract so all
        # JSON artifacts in a run directory share byte-identical line
        # endings across OSes.
        from ragcheck.diff import DiffResult, dump_diff_json

        gold = GoldSet(
            version=1,
            source="c",
            questions=[Query(query_id="q01", text="x")],
        )
        g_out = tmp_path / "gold.json"
        d_out = tmp_path / "diff.json"
        save_gold_set(gold, g_out)
        dump_diff_json(DiffResult(), d_out)
        assert b"\r\n" not in g_out.read_bytes()
        assert b"\r\n" not in d_out.read_bytes()
