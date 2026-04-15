"""Embedders for turning text into vectors.

Three built-in adapters:
1. SentenceTransformersEmbedder — lazy import of sentence-transformers
2. OpenAIEmbedder — lazy import of openai, disk-cached by sha1(text)
3. NumpyEmbedder — reads pre-computed .npy vectors keyed by id

Plus a deterministic HashEmbedder used in tests and the `bench` command so we
never require a network or a GPU for CI.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    """Protocol implemented by all embedders.

    An embedder takes a list of strings and returns a 2D numpy array of shape
    (n, dim), L2-normalised. Row i corresponds to input i.
    """

    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Zero vectors stay zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


class HashEmbedder:
    """Deterministic hash-based embedder.

    Produces a dense vector by hashing whitespace-token shingles into a
    fixed-size dimension. No external deps, no randomness. Used as the CI
    default and in the `bench` command so tests stay fast and reproducible.
    """

    name = "hash"

    def __init__(self, dim: int = 128, ngram: int = 2) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if ngram <= 0:
            raise ValueError(f"ngram must be > 0, got {ngram}")
        self.dim = dim
        self.ngram = ngram

    def _hash_token(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "big") % self.dim

    def _token_ngrams(self, text: str) -> List[str]:
        tokens = text.lower().split()
        if len(tokens) < self.ngram:
            return tokens[:]
        out = []
        for i in range(len(tokens) - self.ngram + 1):
            out.append(" ".join(tokens[i : i + self.ngram]))
        out.extend(tokens)
        return out

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            for tok in self._token_ngrams(text):
                idx = self._hash_token(tok)
                out[i, idx] += 1.0
        return _l2_normalise(out)


class SentenceTransformersEmbedder:
    """Adapter around sentence-transformers. Lazy-imported."""

    name = "sentence-transformers"

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model
        self._model: Any = None
        self.dim = 0

    def _load(self):  # pragma: no cover - requires network/model download
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with: pip install 'ragcheck[sentence-transformers]'"
            ) from e
        self._model = SentenceTransformer(self.model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension() or 0)

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        self._load()
        assert self._model is not None
        vecs = self._model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        arr = np.asarray(vecs, dtype=np.float64)
        return _l2_normalise(arr)


class OpenAIEmbedder:
    """Adapter around OpenAI embeddings with on-disk cache.

    The cache key is sha1(model + text) so two runs over the same corpus
    never re-call the API. Lazy-imported.
    """

    name = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        cache_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".ragcheck-cache") / "openai"
        self.api_key = api_key
        self._client: Any = None
        self.dim = 1536 if "small" in model else 3072

    def _cache_path(self, text: str) -> Path:
        key = hashlib.sha1(f"{self.model}|{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _load_cached(self, text: str) -> Optional[np.ndarray]:
        p = self._cache_path(text)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return np.asarray(data["vector"], dtype=np.float64)
        return None

    def _save_cached(self, text: str, vec: np.ndarray) -> None:
        p = self._cache_path(text)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump({"model": self.model, "vector": vec.tolist()}, f)

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        uncached: List[int] = []
        uncached_texts: List[str] = []
        for i, t in enumerate(texts):
            cached = self._load_cached(t)
            if cached is not None:
                out[i] = cached
            else:
                uncached.append(i)
                uncached_texts.append(t)
        if uncached:
            if self._client is None:
                try:
                    from openai import OpenAI
                except ImportError as e:
                    raise ImportError(
                        "openai package not installed. "
                        "Install with: pip install 'ragcheck[openai]'"
                    ) from e
                self._client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
            resp = self._client.embeddings.create(model=self.model, input=uncached_texts)
            for j, idx in enumerate(uncached):
                vec = np.asarray(resp.data[j].embedding, dtype=np.float64)
                out[idx] = vec
                self._save_cached(uncached_texts[j], vec)
        return _l2_normalise(out)


class NumpyEmbedder:
    """Read pre-computed embeddings from .npy files or in-memory dict.

    Supports two modes:
    - Directory mode: `NumpyEmbedder.from_directory(path)` reads `vectors.npy`
      and `ids.json` from the directory.
    - Dict mode: `NumpyEmbedder(mapping={"id1": np.array([...]), ...})`.

    Text input to `embed()` is interpreted as an id lookup key — any id not
    in the mapping raises KeyError.
    """

    name = "numpy"

    def __init__(self, mapping: Dict[str, np.ndarray]) -> None:
        if not mapping:
            raise ValueError("NumpyEmbedder requires a non-empty mapping")
        first_key = next(iter(mapping))
        first = mapping[first_key]
        # Require exactly-1D vectors so callers can't accidentally pass a 2D
        # batch and have us silently treat its row count as the embedding
        # dimension (Cycle C M1). A 2D input leaves `shape[0]` as the row
        # count, which would then fail later during `embed()` with a
        # confusing numpy broadcast error.
        if first.ndim != 1:
            raise ValueError(
                f"NumpyEmbedder expects 1D vectors, but {first_key!r} has "
                f"shape {first.shape}. If you have a 2D matrix, use "
                f"NumpyEmbedder.from_directory or build a dict keyed by id."
            )
        dim = int(first.shape[0])
        if dim == 0:
            raise ValueError(f"NumpyEmbedder vectors must be non-empty; {first_key!r} has dim 0")
        for k, v in mapping.items():
            if v.ndim != 1:
                raise ValueError(
                    f"NumpyEmbedder expects 1D vectors, but {k!r} has shape {v.shape}"
                )
            if v.shape[0] != dim:
                raise ValueError(
                    f"all vectors must have dim {dim}, but {k!r} has shape {v.shape}"
                )
        self.dim = dim
        self.mapping = {k: _l2_normalise(v.reshape(1, -1))[0] for k, v in mapping.items()}

    @classmethod
    def from_directory(cls, path: Path) -> "NumpyEmbedder":
        path = Path(path)
        vec_path = path / "vectors.npy"
        ids_path = path / "ids.json"
        if not vec_path.exists():
            raise FileNotFoundError(f"vectors.npy not found in {path}")
        if not ids_path.exists():
            raise FileNotFoundError(f"ids.json not found in {path}")
        with ids_path.open("r", encoding="utf-8") as f:
            ids = json.load(f)
        vectors = np.load(vec_path)
        # Guard against 1D/0D matrices with a clear message (Cycle C M2).
        # Without this, ``vectors.shape[0]`` on a 1D array is the scalar
        # count and the per-id slice ``vectors[i]`` yields a 0D scalar,
        # which explodes much later with ``IndexError: tuple index out of
        # range`` during the dim check in ``__init__``.
        if vectors.ndim != 2:
            raise ValueError(
                f"vectors.npy must be a 2D array of shape (n, dim), "
                f"got ndim={vectors.ndim} shape={vectors.shape}"
            )
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"ids.json length {len(ids)} does not match vectors.npy rows {vectors.shape[0]}"
            )
        return cls({ids[i]: vectors[i] for i in range(len(ids))})

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, key in enumerate(texts):
            if key not in self.mapping:
                raise KeyError(f"id not in embedding store: {key!r}")
            out[i] = self.mapping[key]
        return out


# --- Registry -----------------------------------------------------------------

_REGISTRY: Dict[str, Callable[..., Embedder]] = {}


def _register(name: str, factory: Callable[..., Embedder]) -> None:
    _REGISTRY[name.strip().lower()] = factory


def get_embedder(name: str, **kwargs) -> Embedder:
    """Construct an embedder by registered name."""
    key = name.strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"unknown embedder: {name!r}. available: {available}")
    return _REGISTRY[key](**kwargs)


def list_embedders() -> List[str]:
    return sorted(_REGISTRY.keys())


_register("hash", HashEmbedder)
_register("sentence-transformers", SentenceTransformersEmbedder)
_register("openai", OpenAIEmbedder)
# NumpyEmbedder is NOT auto-registered because it needs a mandatory mapping.


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batch cosine similarity of row-vectors in a against row-vectors in b.

    Both matrices are assumed L2-normalised. Returns shape (a.rows, b.rows).
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"a and b must be 2D, got shapes {a.shape} and {b.shape}")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"embedding dim mismatch: {a.shape[1]} vs {b.shape[1]}")
    return a @ b.T


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k scores, descending. Deterministic tie-break by index."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k == 0 or scores.size == 0:
        return np.array([], dtype=np.int64)
    k_eff = min(k, scores.shape[0])
    # argsort with stable kind, negate for descending
    idx = np.argsort(-scores, kind="stable")[:k_eff]
    return idx
