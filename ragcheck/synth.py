"""Bootstrap a synthetic gold set from a corpus.

Given a corpus directory, sample up to N documents; for each, pick the
most-distinctive sentence (highest average pairwise distance to other
document sentences under the HashEmbedder) and use it as a query whose
relevant_doc_id is the source document. Clearly marked as synthetic in the
resulting JSON.

This is NOT a human-graded gold set; it is a smoke-test fixture.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import List, Sequence

import numpy as np

from ragcheck.chunkers import split_sentences
from ragcheck.corpus import Document, GoldSet, Query, load_corpus
from ragcheck.embedders import HashEmbedder, cosine_similarity


def _pick_sentence(doc: Document, other_docs: Sequence[Document]) -> str:
    """Pick the sentence from `doc` that is most distinctive vs. other docs."""
    sentences = split_sentences(doc.text)
    if not sentences:
        return doc.text[:200]
    emb = HashEmbedder(dim=128)
    sent_mat = emb.embed(sentences)
    other_texts: List[str] = []
    for od in other_docs:
        other_texts.extend(split_sentences(od.text) or [od.text[:200]])
    if not other_texts:
        # Just return the longest
        return max(sentences, key=len)
    other_mat = emb.embed(other_texts)
    sims = cosine_similarity(sent_mat, other_mat)
    # Average similarity to OTHER docs' sentences: lower is more distinctive
    avg_sim = sims.mean(axis=1)
    idx = int(np.argmin(avg_sim))
    return sentences[idx]


def _sentence_to_question(sentence: str, seed: int = 0) -> str:
    """Turn a sentence into a plausible question.

    This is a very simple heuristic — it doesn't have to produce natural
    English; it just has to yield reasonable retrieval targets for a
    smoke-test run.
    """
    s = sentence.strip().rstrip(".!?")
    s = re.sub(r"\s+", " ", s)
    if len(s) > 120:
        s = s[:120].rsplit(" ", 1)[0] + "..."
    lower = s.lower()
    # Prefer explicit question phrasing
    for leading in ("the ", "a ", "an ", "we ", "you ", "it "):
        if lower.startswith(leading):
            return f"What {s[len(leading):]}?"
    return f"What about {s}?"


def synth_gold_set(
    corpus_path: Path,
    *,
    n_questions: int = 50,
    seed: int = 42,
) -> GoldSet:
    """Generate a synthetic gold set for a corpus.

    Each question targets exactly one document. If the corpus has fewer docs
    than `n_questions`, every doc is used and the remainder is filled by
    cycling through the doc list deterministically.
    """
    if n_questions <= 0:
        raise ValueError(f"n_questions must be > 0, got {n_questions}")
    docs = load_corpus(corpus_path)
    if not docs:
        return GoldSet(version=1, questions=[], source=str(corpus_path), synthetic=True)
    rng = random.Random(seed)
    indices = list(range(len(docs)))
    rng.shuffle(indices)
    chosen: List[int] = []
    i = 0
    while len(chosen) < n_questions:
        chosen.append(indices[i % len(indices)])
        i += 1
    questions: List[Query] = []
    for j, doc_idx in enumerate(chosen):
        doc = docs[doc_idx]
        others = [d for k, d in enumerate(docs) if k != doc_idx]
        sentence = _pick_sentence(doc, others)
        question_text = _sentence_to_question(sentence, seed=seed + j)
        questions.append(
            Query(
                query_id=f"synth-{j:04d}",
                text=question_text,
                relevant_doc_ids=[doc.doc_id],
                relevant_chunk_ids=[],
                relevance={doc.doc_id: 1.0},
                synthetic=True,
            )
        )
    return GoldSet(
        version=1,
        questions=sorted(questions, key=lambda q: q.query_id),
        source=str(corpus_path),
        synthetic=True,
    )
