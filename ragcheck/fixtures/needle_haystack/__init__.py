"""Generator module for the needle-in-haystack synthetic fixture."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple

FIXTURE_DIR = Path(__file__).parent
CORPUS_DIR = FIXTURE_DIR / "corpus"


def _haystack_sentence(seed: int) -> str:
    h = hashlib.sha1(f"hay-{seed}".encode("utf-8")).hexdigest()
    # Deterministic, generic filler; no real meaning — a typical "this is
    # generic filler text about common topics" passage that embedding models
    # collapse together.
    filler = [
        "A generic note about logistics schedules. Many teams sync weekly.",
        "Project planning requires alignment across multiple stakeholders.",
        "Quarterly business reviews often recap lagging indicators.",
        "Performance reviews emphasise clarity, ownership, and impact.",
        "The fiscal year closes in December for most US companies.",
    ]
    return filler[seed % len(filler)] + f" Ref: {h[:8]}."


def generate_needle_corpus(
    *,
    n_needles: int = 6,
    n_haystack_per_needle: int = 12,
) -> List[Tuple[str, str, str]]:
    """Return a list of (doc_id, text, kind) tuples.

    kind is either "needle" or "haystack". Needle docs contain a unique
    answer phrase keyed by index; haystack docs contain generic filler.
    """
    out: List[Tuple[str, str, str]] = []
    for i in range(n_needles):
        answer_phrase = f"NEEDLE-{i:03d}: the activation code is ALPHA-{i:05d}-OMEGA."
        text = (
            f"This document contains important information for token {i}.\n"
            f"Buried among the noise you will find the key fact.\n"
            f"{answer_phrase}\n"
            f"It does not appear anywhere else in the corpus.\n"
        )
        out.append((f"needle_{i:03d}", text, "needle"))
    for j in range(n_needles * n_haystack_per_needle):
        sentences = [_haystack_sentence(j + k) for k in range(5)]
        out.append((f"haystack_{j:03d}", "\n".join(sentences) + "\n", "haystack"))
    return out


def materialise() -> Path:
    """Write the needle corpus to disk. Returns the corpus directory.

    Idempotent: overwrites the corpus if already present so the on-disk
    content always matches `generate_needle_corpus()`.
    """
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for doc_id, text, _kind in generate_needle_corpus():
        path = CORPUS_DIR / f"{doc_id}.txt"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    return CORPUS_DIR
