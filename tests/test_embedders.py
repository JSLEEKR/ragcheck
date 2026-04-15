"""Tests for embedders."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ragcheck.embedders import (
    HashEmbedder,
    NumpyEmbedder,
    OpenAIEmbedder,
    SentenceTransformersEmbedder,
    cosine_similarity,
    get_embedder,
    list_embedders,
    top_k_indices,
)


class TestHashEmbedder:
    def test_output_shape(self):
        e = HashEmbedder(dim=64)
        out = e.embed(["hello world", "another text"])
        assert out.shape == (2, 64)

    def test_l2_normalised(self):
        e = HashEmbedder(dim=64)
        out = e.embed(["some text here"])
        norm = np.linalg.norm(out[0])
        assert np.isclose(norm, 1.0, atol=1e-9) or np.isclose(norm, 0.0)

    def test_deterministic(self):
        e1 = HashEmbedder(dim=64)
        e2 = HashEmbedder(dim=64)
        out1 = e1.embed(["the quick brown fox"])
        out2 = e2.embed(["the quick brown fox"])
        assert np.allclose(out1, out2)

    def test_different_texts_different_vectors(self):
        e = HashEmbedder(dim=128)
        out = e.embed(["alpha beta gamma", "delta epsilon zeta"])
        assert not np.allclose(out[0], out[1])

    def test_empty_string_is_zero(self):
        e = HashEmbedder(dim=64)
        out = e.embed([""])
        assert np.allclose(out[0], np.zeros(64))

    def test_similar_texts_high_cosine(self):
        e = HashEmbedder(dim=256)
        out = e.embed([
            "kubernetes manages containers",
            "kubernetes is a container orchestrator",
        ])
        sim = float(np.dot(out[0], out[1]))
        assert sim > 0.1

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            HashEmbedder(dim=0)

    def test_invalid_ngram(self):
        with pytest.raises(ValueError):
            HashEmbedder(ngram=0)

    def test_name(self):
        assert HashEmbedder().name == "hash"


class TestNumpyEmbedder:
    def test_roundtrip(self):
        mapping = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
        e = NumpyEmbedder(mapping)
        out = e.embed(["a", "b"])
        assert out.shape == (2, 2)
        # After normalisation, vectors remain the same
        assert np.isclose(float(out[0][0]), 1.0)

    def test_dim_mismatch_raises(self):
        mapping = {"a": np.array([1.0]), "b": np.array([1.0, 0.0])}
        with pytest.raises(ValueError):
            NumpyEmbedder(mapping)

    def test_empty_mapping_raises(self):
        with pytest.raises(ValueError):
            NumpyEmbedder({})

    def test_missing_id_raises(self):
        e = NumpyEmbedder({"a": np.array([1.0, 0.0])})
        with pytest.raises(KeyError):
            e.embed(["z"])

    def test_from_directory(self, tmp_path: Path):
        vec = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.save(tmp_path / "vectors.npy", vec)
        (tmp_path / "ids.json").write_text(json.dumps(["x", "y"]), encoding="utf-8")
        e = NumpyEmbedder.from_directory(tmp_path)
        assert e.dim == 2
        out = e.embed(["x"])
        assert out.shape == (1, 2)

    def test_from_directory_missing_vectors(self, tmp_path: Path):
        (tmp_path / "ids.json").write_text(json.dumps(["x"]), encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            NumpyEmbedder.from_directory(tmp_path)

    def test_from_directory_missing_ids(self, tmp_path: Path):
        vec = np.array([[1.0, 0.0]])
        np.save(tmp_path / "vectors.npy", vec)
        with pytest.raises(FileNotFoundError):
            NumpyEmbedder.from_directory(tmp_path)

    def test_from_directory_length_mismatch(self, tmp_path: Path):
        vec = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.save(tmp_path / "vectors.npy", vec)
        (tmp_path / "ids.json").write_text(json.dumps(["only_one"]), encoding="utf-8")
        with pytest.raises(ValueError):
            NumpyEmbedder.from_directory(tmp_path)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[1.0, 0.0]])
        assert np.isclose(cosine_similarity(a, b)[0][0], 1.0)

    def test_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        assert np.isclose(cosine_similarity(a, b)[0][0], 0.0)

    def test_dim_mismatch(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            cosine_similarity(a, b)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            cosine_similarity(np.array([1.0]), np.array([[1.0]]))


class TestTopKIndices:
    def test_basic(self):
        idx = top_k_indices(np.array([0.1, 0.5, 0.3]), 2)
        assert list(idx) == [1, 2]

    def test_k_zero(self):
        idx = top_k_indices(np.array([0.5, 0.1]), 0)
        assert idx.size == 0

    def test_k_larger_than_size(self):
        idx = top_k_indices(np.array([0.5, 0.1]), 10)
        assert list(idx) == [0, 1]

    def test_empty_array(self):
        idx = top_k_indices(np.array([]), 5)
        assert idx.size == 0

    def test_negative_k_raises(self):
        with pytest.raises(ValueError):
            top_k_indices(np.array([0.1]), -1)

    def test_stable_tie_break(self):
        idx = top_k_indices(np.array([0.5, 0.5, 0.5]), 3)
        assert list(idx) == [0, 1, 2]


class TestEmbedderRegistry:
    def test_list_embedders(self):
        names = list_embedders()
        assert "hash" in names
        assert names == sorted(names)

    def test_get_by_name(self):
        e = get_embedder("hash")
        assert isinstance(e, HashEmbedder)

    def test_unknown_embedder(self):
        with pytest.raises(KeyError):
            get_embedder("nonexistent")

    def test_case_insensitive(self):
        e = get_embedder("HASH")
        assert isinstance(e, HashEmbedder)


class TestLazyEmbedders:
    def test_sentence_transformers_import_error_when_missing(self, monkeypatch):
        e = SentenceTransformersEmbedder()
        # force-load path: mock sentence_transformers absent
        monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", None)
        with pytest.raises(ImportError):
            e._load()

    def test_openai_cache_path_uses_hash(self):
        e = OpenAIEmbedder()
        p1 = e._cache_path("hello")
        p2 = e._cache_path("hello")
        p3 = e._cache_path("different")
        assert p1 == p2
        assert p1 != p3

    def test_openai_embedder_name(self):
        assert OpenAIEmbedder().name == "openai"

    def test_sentence_transformers_embedder_name(self):
        assert SentenceTransformersEmbedder().name == "sentence-transformers"
