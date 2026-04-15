"""Main evaluation pipeline.

Given a corpus, a gold set, a chunker, and an embedder, compute every metric
and produce a deterministic RunResult.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ragcheck import __version__
from ragcheck.chunkers import Chunk, Chunker, get_chunker
from ragcheck.corpus import Document, GoldSet, Query, load_corpus, load_gold_set
from ragcheck.diagnostics import chunking_diagnostics
from ragcheck.embedders import (
    Embedder,
    cosine_similarity,
    get_embedder,
    top_k_indices,
)
from ragcheck.metrics import (
    DEFAULT_K_VALUES,
    aggregate_metric,
    context_precision,
    context_recall,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


@dataclass
class RunConfig:
    """Serializable run configuration."""

    corpus_path: str
    gold_path: str
    chunker: str = "fixed-token"
    chunker_args: Dict[str, Any] = field(default_factory=dict)
    embedder: str = "hash"
    embedder_args: Dict[str, Any] = field(default_factory=dict)
    k_values: List[int] = field(default_factory=lambda: list(DEFAULT_K_VALUES))
    top_k_cap: int = 20
    include_timestamp: bool = False
    include_per_query: bool = True
    include_diagnostics: bool = True
    label: str = ""
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["k_values"] = sorted(set(int(k) for k in self.k_values))
        # Cycle D M3: normalise Windows path separators to forward slashes
        # in the serialised config. The Markdown and HTML renderers already
        # do this for display (Cycle A M1), but the raw run JSON kept
        # backslashes on Windows, which broke byte-identical cross-platform
        # determinism and round-tripped ``\\U`` / ``\\0`` style escape
        # sequences into any downstream JSON consumer.
        if isinstance(d.get("corpus_path"), str):
            d["corpus_path"] = d["corpus_path"].replace("\\", "/")
        if isinstance(d.get("gold_path"), str):
            d["gold_path"] = d["gold_path"].replace("\\", "/")
        return d


@dataclass
class PerQueryResult:
    query_id: str
    retrieved: List[str]
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "retrieved": list(self.retrieved),
            "metrics": {k: self.metrics[k] for k in sorted(self.metrics)},
        }


@dataclass
class RunResult:
    tool_version: str = __version__
    schema_version: int = 1
    config: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, float] = field(default_factory=dict)
    per_query: List[PerQueryResult] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    corpus_stats: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None

    def to_dict(self, *, float_precision: int = 6) -> Dict[str, Any]:
        def _round(v: Any) -> Any:
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    return None
                return round(v, float_precision)
            if isinstance(v, dict):
                return {k: _round(v[k]) for k in sorted(v)}
            if isinstance(v, list):
                return [_round(x) for x in v]
            return v

        payload: Dict[str, Any] = {
            "tool_version": self.tool_version,
            "schema_version": self.schema_version,
            "config": _round(self.config),
            "summary": _round(self.summary),
            "corpus_stats": _round(self.corpus_stats),
            "diagnostics": _round(self.diagnostics),
            "per_query": [
                {
                    "query_id": pq.query_id,
                    "retrieved": list(pq.retrieved),
                    "metrics": {k: _round(pq.metrics[k]) for k in sorted(pq.metrics)},
                }
                for pq in self.per_query
            ],
        }
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp
        return payload


def _build_chunker(name: str, args: Dict[str, Any]) -> Chunker:
    return get_chunker(name, **args)


def _build_embedder(name: str, args: Dict[str, Any]) -> Embedder:
    if name == "numpy":
        raise ValueError(
            "numpy embedder requires a programmatic mapping; "
            "use run_evaluation(embedder=NumpyEmbedder(...)) directly"
        )
    return get_embedder(name, **args)


def _chunk_documents(docs: Sequence[Document], chunker: Chunker) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunker.chunk(doc.doc_id, doc.text))
    return all_chunks


def _resolve_relevant_chunk_ids(
    query: Query, doc_to_chunk_ids: Dict[str, List[str]]
) -> List[str]:
    """Expand relevant_doc_ids into chunk ids when explicit chunk ids are absent."""
    if query.relevant_chunk_ids:
        return list(query.relevant_chunk_ids)
    out: List[str] = []
    for d in query.relevant_doc_ids:
        out.extend(doc_to_chunk_ids.get(d, []))
    return out


def _normalise_relevance(
    query: Query, doc_to_chunk_ids: Dict[str, List[str]]
) -> Dict[str, float]:
    """Build a chunk-level relevance map.

    If `relevance` is given, propagate each doc_id's grade to every chunk of
    that doc. If an entry in relevance is a chunk id directly, keep it.
    """
    out: Dict[str, float] = {}
    if query.relevance:
        for key, grade in query.relevance.items():
            if key in doc_to_chunk_ids:
                for cid in doc_to_chunk_ids[key]:
                    out[cid] = max(out.get(cid, 0.0), float(grade))
            else:
                out[key] = float(grade)
        return out
    # Fall back to binary relevance over the resolved chunk ids
    for cid in _resolve_relevant_chunk_ids(query, doc_to_chunk_ids):
        out[cid] = 1.0
    return out


def _query_metrics(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    relevance: Dict[str, float],
    k_values: Sequence[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevance, k)
    metrics["mrr"] = mean_reciprocal_rank(retrieved, relevant)
    metrics["context_precision"] = context_precision(retrieved, relevant)
    metrics["context_recall"] = context_recall(retrieved, relevant)
    return metrics


def run_evaluation(
    corpus: Optional[Sequence[Document]] = None,
    gold: Optional[GoldSet] = None,
    *,
    config: Optional[RunConfig] = None,
    chunker: Optional[Chunker] = None,
    embedder: Optional[Embedder] = None,
    corpus_path: Optional[Path] = None,
    gold_path: Optional[Path] = None,
) -> RunResult:
    """Run a full evaluation end-to-end.

    Inputs can be passed either as in-memory objects (for tests) or as paths
    (for the CLI). A RunConfig is required; if not provided, a default
    config is constructed from the paths.
    """
    corpus_provided_by_caller = corpus is not None
    if config is None:
        if corpus_path is None or gold_path is None:
            raise ValueError("either config or (corpus_path and gold_path) is required")
        config = RunConfig(
            corpus_path=str(corpus_path),
            gold_path=str(gold_path),
        )
    if corpus is None:
        corpus = load_corpus(Path(config.corpus_path))
    if gold is None:
        gold = load_gold_set(Path(config.gold_path))
    # Cycle D M1: surface an empty corpus (loaded from disk) as a clear
    # error instead of silently returning zeroed metrics. Previously
    # ``ragcheck run`` on a directory containing no readable files would
    # print ``recall@5: 0.000000`` etc. with exit 0, giving the user no
    # signal that the corpus was empty and every metric was meaningless.
    # Callers that build an in-memory corpus (``corpus=[]``) are unchanged
    # — that branch is used by internal tests.
    if not corpus and not corpus_provided_by_caller:
        raise ValueError(
            f"corpus is empty: no .txt/.md files found under {config.corpus_path!r}. "
            "Check the path and allowed extensions (.txt, .md, .rst, .markdown). "
            "Hidden files and directories are skipped by design."
        )
    if chunker is None:
        chunker = _build_chunker(config.chunker, config.chunker_args)
    if embedder is None:
        embedder = _build_embedder(config.embedder, config.embedder_args)

    all_chunks = _chunk_documents(corpus, chunker)
    doc_to_chunk_ids: Dict[str, List[str]] = {}
    for c in all_chunks:
        doc_to_chunk_ids.setdefault(c.doc_id, []).append(c.chunk_id)

    # Encode chunks once; encode queries in batch
    chunk_texts = [c.text for c in all_chunks]
    chunk_ids = [c.chunk_id for c in all_chunks]
    if not chunk_texts:
        chunk_matrix = np.zeros((0, embedder.dim if embedder.dim else 1), dtype=np.float64)
    else:
        chunk_matrix = embedder.embed(chunk_texts)
    query_texts = [q.text for q in gold.questions]
    if query_texts:
        query_matrix = embedder.embed(query_texts)
    else:
        query_matrix = np.zeros((0, chunk_matrix.shape[1] if chunk_matrix.size else 1))

    top_cap = max(config.top_k_cap, max(config.k_values, default=0))
    per_query: List[PerQueryResult] = []
    reachable_ids: set = set()

    for i, query in enumerate(gold.questions):
        if chunk_matrix.shape[0] == 0 or query_matrix.shape[0] == 0:
            retrieved_ids: List[str] = []
        else:
            sims = cosine_similarity(query_matrix[i : i + 1], chunk_matrix)[0]
            indices = top_k_indices(sims, top_cap)
            retrieved_ids = [chunk_ids[int(idx)] for idx in indices]
        relevant_chunk_ids = _resolve_relevant_chunk_ids(query, doc_to_chunk_ids)
        relevance = _normalise_relevance(query, doc_to_chunk_ids)
        metrics = _query_metrics(
            retrieved_ids,
            relevant_chunk_ids,
            relevance,
            config.k_values,
        )
        per_query.append(
            PerQueryResult(
                query_id=query.query_id,
                retrieved=retrieved_ids,
                metrics=metrics,
            )
        )
        reachable_ids.update(retrieved_ids)

    # Aggregate summary: mean across queries
    summary: Dict[str, float] = {}
    metric_names: List[str] = []
    if per_query:
        metric_names = sorted(per_query[0].metrics.keys())
    for mname in metric_names:
        values = [pq.metrics[mname] for pq in per_query]
        summary[mname] = aggregate_metric(values)

    # Diagnostics
    gold_chunks: set = set()
    for q in gold.questions:
        gold_chunks.update(_resolve_relevant_chunk_ids(q, doc_to_chunk_ids))
    diag_obj = chunking_diagnostics(
        all_chunks,
        gold_relevant_chunk_ids=gold_chunks,
        reachable_chunk_ids=reachable_ids,
    )

    result = RunResult(
        config=config.to_dict(),
        summary=summary,
        per_query=per_query if config.include_per_query else [],
        diagnostics=diag_obj.to_dict() if config.include_diagnostics else {},
        corpus_stats={
            "total_documents": len(corpus),
            "total_queries": len(gold.questions),
            "total_chunks": len(all_chunks),
            "corpus_sha1": _corpus_sha1(corpus),
            "synthetic_gold": gold.synthetic,
        },
    )
    if config.include_timestamp:
        result.timestamp = _iso_utc()
    return result


def _corpus_sha1(docs: Sequence[Document]) -> str:
    h = hashlib.sha1()
    for d in sorted(docs, key=lambda x: x.doc_id):
        h.update(d.doc_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(d.text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _iso_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dump_result_json(
    result: RunResult,
    path: Path,
    *,
    pretty: bool = True,
) -> None:
    """Serialise a RunResult to JSON deterministically."""
    data = result.to_dict()
    if pretty:
        text = json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2)
    else:
        text = json.dumps(
            data, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.write("\n")


def load_result_json(path: Path) -> Dict[str, Any]:
    """Load a previously-saved run.json file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"run result not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
