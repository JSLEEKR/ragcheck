"""Shared pytest fixtures and helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from ragcheck.corpus import Document, GoldSet, Query


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "alpha.txt").write_text(
        "Alpha discusses interest rates and inflation. "
        "When central banks raise rates, inflation falls.",
        encoding="utf-8",
    )
    (root / "beta.txt").write_text(
        "Beta describes dividends and stock buybacks. "
        "Dividends are taxed differently than capital gains.",
        encoding="utf-8",
    )
    (root / "gamma.txt").write_text(
        "Gamma covers kubernetes and containers. "
        "Pods are the smallest deployable unit.",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def tiny_gold(tmp_path: Path) -> Path:
    gold_path = tmp_path / "gold.json"
    data = {
        "version": 1,
        "source": "tiny",
        "synthetic": False,
        "questions": [
            {
                "query_id": "q1",
                "text": "how do interest rates affect inflation?",
                "relevant_doc_ids": ["alpha"],
                "relevance": {"alpha": 2},
            },
            {
                "query_id": "q2",
                "text": "how are dividends taxed?",
                "relevant_doc_ids": ["beta"],
                "relevance": {"beta": 2},
            },
            {
                "query_id": "q3",
                "text": "what is a kubernetes pod?",
                "relevant_doc_ids": ["gamma"],
                "relevance": {"gamma": 2},
            },
        ],
    }
    gold_path.write_text(
        json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return gold_path


@pytest.fixture
def small_docs() -> List[Document]:
    return [
        Document(doc_id="a", path="a.txt", text="alpha alpha alpha"),
        Document(doc_id="b", path="b.txt", text="beta beta beta"),
        Document(doc_id="c", path="c.txt", text="gamma gamma gamma"),
    ]


@pytest.fixture
def small_gold() -> GoldSet:
    return GoldSet(
        version=1,
        source="small",
        questions=[
            Query(
                query_id="q1",
                text="alpha",
                relevant_doc_ids=["a"],
                relevance={"a": 1.0},
            ),
            Query(
                query_id="q2",
                text="beta",
                relevant_doc_ids=["b"],
                relevance={"b": 1.0},
            ),
        ],
    )


def ranked(*ids: str) -> List[str]:
    return list(ids)
