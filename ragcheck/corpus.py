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
import math
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

    Symbolic links, Windows directory junctions, and any other filesystem
    construct whose *resolved* path escapes the corpus root are skipped.
    Without this guard a user could create ``corpus/escape`` as an ``mklink
    /J`` junction pointing at ``%USERPROFILE%\\.ssh`` and ``ragcheck`` would
    happily ingest private keys as if they were corpus documents (Cycle D H1).
    ``resolve()`` follows all links down to the real path, so this catches
    both POSIX symlinks and Windows directory junctions in one check.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"corpus directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"corpus path is not a directory: {root}")
    root_resolved = root.resolve()
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
        # Reject files that escape the corpus root via symlinks or
        # Windows directory junctions. ``resolve()`` yields the final
        # real path; if any link along the way pointed outside ``root``,
        # ``relative_to`` raises ValueError and we skip the entry.
        try:
            file.resolve().relative_to(root_resolved)
        except ValueError:
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
                # Python's json parser accepts ``NaN`` / ``Infinity`` /
                # ``-Infinity`` as an extension of the spec. Reject them
                # explicitly so a gold file containing ``"relevance": {"a":
                # NaN}`` is flagged at load time — otherwise the NaN
                # silently propagates into the relevance dict, breaks
                # nDCG, and makes every run look catastrophically wrong.
                # Cycle E M1.
                if math.isnan(fv) or math.isinf(fv):
                    raise ValueError(
                        f"relevance value must be finite, got {v!r} for id {k!r}"
                    )
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
    """Write a gold set to JSON deterministically (sorted keys, utf-8, LF).

    The ``source`` field is normalised to forward-slash separators so a
    gold set produced on Windows is byte-identical to one produced on
    POSIX — otherwise ``ragcheck synth`` on Windows writes literal
    ``"C:\\\\Users\\\\..."`` into the gold JSON, breaking cross-platform
    determinism (Cycle E M2 — matches Cycle D M3's fix for RunConfig).

    The file is opened with ``newline="\\n"`` so Python's text-mode line
    translation does not silently rewrite ``\\n`` to ``\\r\\n`` on
    Windows. Without this guard, a synth'd gold file on Windows is byte
    -different from the same gold file on Linux, breaking the
    cross-platform determinism contract (every other JSON writer in the
    codebase — ``dump_result_json``, ``dump_diff_json``, the CLI
    ``_write_out`` and ``--markdown`` paths — already pass this kwarg;
    ``save_gold_set`` was the lone holdout). Cycle F H3.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = gold.to_dict()
    src = data.get("source")
    if isinstance(src, str):
        data["source"] = src.replace("\\", "/")
    with p.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(
            data,
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
