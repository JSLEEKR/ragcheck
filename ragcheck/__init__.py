"""ragcheck: offline RAG retrieval quality harness.

Compute recall@k, precision@k, MRR, nDCG, hit-rate, context precision/recall,
and chunking diagnostics on labeled gold sets. Regression-diff two runs.
No LLM-as-judge. Deterministic JSON/HTML/Markdown reports.
"""
from __future__ import annotations

__version__ = "1.0.0"

from ragcheck.chunkers import (
    Chunker,
    FixedTokenChunker,
    SemanticBoundaryChunker,
    SentenceChunker,
    SlidingWindowChunker,
    StructuralMarkdownChunker,
    get_chunker,
    list_chunkers,
    register_chunker,
)
from ragcheck.diagnostics import chunking_diagnostics
from ragcheck.diff import DiffResult, diff_runs
from ragcheck.embedders import (
    Embedder,
    NumpyEmbedder,
    OpenAIEmbedder,
    SentenceTransformersEmbedder,
    get_embedder,
    list_embedders,
)
from ragcheck.metrics import (
    context_precision,
    context_recall,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ragcheck.report import render_html, render_json, render_markdown
from ragcheck.runner import RunConfig, RunResult, run_evaluation

__all__ = [
    "__version__",
    # metrics
    "recall_at_k",
    "precision_at_k",
    "hit_rate_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "context_precision",
    "context_recall",
    # chunkers
    "Chunker",
    "FixedTokenChunker",
    "SlidingWindowChunker",
    "SentenceChunker",
    "SemanticBoundaryChunker",
    "StructuralMarkdownChunker",
    "get_chunker",
    "list_chunkers",
    "register_chunker",
    # embedders
    "Embedder",
    "NumpyEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformersEmbedder",
    "get_embedder",
    "list_embedders",
    # diagnostics
    "chunking_diagnostics",
    # runner
    "RunConfig",
    "RunResult",
    "run_evaluation",
    # diff
    "DiffResult",
    "diff_runs",
    # report
    "render_html",
    "render_json",
    "render_markdown",
]
