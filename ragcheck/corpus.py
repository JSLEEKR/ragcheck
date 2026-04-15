"""Corpus and gold-set loading.

A corpus is a directory of `.txt` or `.md` files; each file is a document.
A gold set is a JSON file with the schema:

    {
      "version": 1,
      "questions": [
        {
          "query_id": "q001",
          "text": "how do I ...",
          "relevant_doc_ids": ["doc_a", "doc_b"],
          "relevant_chunk_ids": ["doc_a::chunk1"],  # optional
          "relevance": {"doc_a": 2, "doc_b": 1}     # optional, graded
        }
      ]
    }

Everything is loaded with explicit UTF-8 encoding; paths use pathlib; order
is sorted so two runs over the same corpus are byte-identical.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List


@dataclass(frozen=True)
class Document:
    doc_id: str
    path: str
    text: str

    def __len__(self) -> int:
        return len(self.text)


@dataclass
class Query:
    query_id: str
    text: str
    relevant_doc_ids: List[str] = field(default_factory=list)
    relevant_chunk_ids: List[str] = field(default_factory=list)
    relevance: Dict[str, float] = field(default_factory=dict)
    synthetic: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "query_id": self.query_id,
            "text": self.text,
            "relevant_doc_ids": sorted(self.relevant_doc_ids),
            "relevant_chunk_ids": sorted(self.relevant_chunk_ids),
            "relevance": {k: self.relevance[k] for k in sorted(self.relevance)},
            "synthetic": self.synthetic,
        }


@dataclass
class GoldSet:
    version: int = 1
    questions: List[Query] = field(default_factory=list)
    source: str = ""
    synthetic: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "source": self.source,
            "synthetic": self.synthetic,
            "questions": [q.to_dict() for q in self.questions],
        }


ALLOWED_EXTENSIONS = frozenset({".txt", ".md", ".rst", ".markdown"})


def load_corpus(path: Path) -> List[Document]:
    """Load every .txt / .md file from a directory into a sorted Document list.

    Empty files are included. The doc_id is the path relative to the corpus
    root, with forward-slash separators and no extension.

    Hidden files **and hidden directories** (any path component starting with
    `.`) are skipped. Without this, calling ``ragcheck run --corpus .`` from a
    repo root would silently ingest ``.git/HEAD``, ``.venv/`` markdown files,
    or any other dot-directory contents into the corpus — a privacy and
    correctness footgun. See Cycle C H1 in `CHANGELOG.md`.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"corpus directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"corpus path is not a directory: {root}")
    docs: List[Document] = []
    for file in sorted(root.rglob("*")):
        if not file.is_file():
            continue
        if file.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        # Skip hidden files AND any file under a hidden directory. Walking
        # the relative path catches ``.git/config.txt``, ``.venv/X.md``,
        # ``a/.cache/y.md`` etc.
        rel_parts = file.relative_to(root).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        rel = file.relative_to(root).as_posix()
        doc_id = rel.rsplit(".", 1)[0] if "." in rel else rel
        with file.open("r", encoding="utf-8") as f:
            text = f.read()
        docs.append(Document(doc_id=doc_id, path=rel, text=text))
    return sorted(docs, key=lambda d: d.doc_id)


def load_gold_set(path: Path) -> GoldSet:
    """Parse a gold-set JSON file. Returns a GoldSet with sorted questions."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"gold set file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"gold set must be a JSON object, got {type(data).__name__}")
    version = int(data.get("version", 1))
    source = str(data.get("source", ""))
    synthetic = bool(data.get("synthetic", False))
    raw_questions = data.get("questions", [])
    if not isinstance(raw_questions, list):
        raise ValueError(f"questions must be a list, got {type(raw_questions).__name__}")
    questions: List[Query] = []
    seen_ids: set = set()
    for i, raw in enumerate(raw_questions):
        if not isinstance(raw, dict):
            raise ValueError(f"question {i} must be an object, got {type(raw).__name__}")
        query_id = str(raw.get("query_id") or f"q{i:04d}")
        if query_id in seen_ids:
            raise ValueError(f"duplicate query_id: {query_id!r}")
        seen_ids.add(query_id)
        text = str(raw.get("text", ""))
        relevant_doc_ids = [str(x) for x in raw.get("relevant_doc_ids", [])]
        relevant_chunk_ids = [str(x) for x in raw.get("relevant_chunk_ids", [])]
        rel_raw = raw.get("relevance", {})
        relevance: Dict[str, float] = {}
        if isinstance(rel_raw, dict):
            for k, v in rel_raw.items():
                if not isinstance(k, str):
                    raise ValueError(f"relevance key must be str: {k!r}")
                try:
                    fv = float(v)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"relevance value must be numeric: {v!r}") from e
                if fv < 0:
                    raise ValueError(f"relevance value must be >= 0: {v!r}")
                relevance[k] = fv
        questions.append(
            Query(
                query_id=query_id,
                text=text,
                relevant_doc_ids=relevant_doc_ids,
                relevant_chunk_ids=relevant_chunk_ids,
                relevance=relevance,
                synthetic=bool(raw.get("synthetic", synthetic)),
            )
        )
    questions.sort(key=lambda q: q.query_id)
    return GoldSet(version=version, questions=questions, source=source, synthetic=synthetic)


def save_gold_set(gold: GoldSet, path: Path) -> None:
    """Write a gold set to JSON deterministically (sorted keys, utf-8)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(
            gold.to_dict(),
            f,
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")


def iter_corpus_paths(root: Path) -> Iterator[Path]:
    root = Path(root)
    for file in sorted(root.rglob("*")):
        if file.is_file() and file.suffix.lower() in ALLOWED_EXTENSIONS:
            yield file
